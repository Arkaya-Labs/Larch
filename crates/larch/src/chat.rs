//! CPU-testable streaming chat and its durable, versioned JSONL conversation journal.

use std::{
    collections::HashMap,
    env,
    fs::{File, OpenOptions, TryLockError},
    io::{BufRead, ErrorKind, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use serde::{Deserialize, Serialize};

use larch_model::vision::Image;

use crate::{
    ChatMessage, GenParams, Larch, LoadOptions, Overflow, Role, Session, SessionOptions, TextEvent,
};

/// CLI chat options; sampling defaults are resolved by the facade at each turn.
#[derive(Args)]
pub struct Options {
    /// GGUF to load.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// DFlash drafter GGUF for the model, or `mtp` for its own MTP block (speculative decoding);
    /// LARCH_DRAFTER when unset.
    #[arg(long)]
    pub drafter: Option<PathBuf>,
    /// A drafter kept only where it costs no chat (`--drafter` wins); LARCH_DRAFTER_AUTO if unset.
    #[arg(long)]
    pub drafter_auto: Option<PathBuf>,
    /// Vision projector (mmproj GGUF) beside the model, so a resumed chat's images are seen again.
    #[arg(long)]
    pub mmproj: Option<PathBuf>,
    /// Explicit reference/debug mode.
    #[arg(long, conflicts_with = "device")]
    pub cpu: bool,
    /// GPU ordinal (default zero).
    #[arg(long)]
    pub device: Option<i32>,
    /// Maximum tokens per answer; unset runs until the model ends it or the context fills.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_tokens: Option<u32>,
    /// Explicit temperature; unset follows the current knob.
    #[arg(long)]
    pub temperature: Option<f32>,
    /// String stop, repeatable.
    #[arg(long)]
    pub stop: Vec<String>,
    /// Startup knob assignment, repeatable.
    #[arg(long = "set")]
    pub set: Vec<String>,
    /// System message for a new conversation.
    #[arg(long)]
    pub system: Option<String>,
    /// Disable parking between natural completions.
    #[arg(long)]
    pub no_keep_kv: bool,
    /// Reject overflow instead of dropping oldest exchanges.
    #[arg(long)]
    pub fail_overflow: bool,
    /// Transcript ID to resume with fresh engine KV.
    #[arg(long)]
    pub resume: Option<String>,
    /// Base data directory; transcripts go in its chat/ subdirectory.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
}

/// Words a batch waits for before it shows.
const BATCH_WORDS: usize = 6;
/// Longest held text waits; a run with no space in it (code, CJK) then shows whole.
const BATCH_WAIT: Duration = Duration::from_millis(120);

/// Groups streamed text into phrase-sized batches, so an answer appears a few words at a time
/// instead of token by token. A batch ends after a whole word, never inside one; the first piece
/// shows at once, so batching never delays the first token.
#[derive(Default)]
pub struct Batcher {
    held: String,
    since: Option<Instant>,
    started: bool,
}

impl Batcher {
    /// Holds `delta`; returns the text ready to show: whole words once a line or sentence ends,
    /// [`BATCH_WORDS`] words wait, or the oldest held text waited [`BATCH_WAIT`].
    pub fn push(&mut self, delta: &str, now: Instant) -> Option<String> {
        self.held.push_str(delta);
        if !self.started && !self.held.is_empty() {
            self.started = true;
            return Some(std::mem::take(&mut self.held));
        }
        let since = *self.since.get_or_insert(now);
        let waited = now.duration_since(since) >= BATCH_WAIT;
        let cut = self
            .held
            .rfind(char::is_whitespace)
            .map(|i| i + self.held[i..].chars().next().map_or(0, char::len_utf8));
        let words = &self.held[..cut.unwrap_or(0)];
        let ready = waited
            || words.contains('\n')
            || words.trim_end().ends_with(['.', '!', '?', ':', ';'])
            || words.split_whitespace().count() >= BATCH_WORDS;
        let take = match cut {
            Some(cut) if ready => cut,
            None if waited => self.held.len(),
            _ => return None,
        };
        let rest = self.held.split_off(take);
        self.since = (!rest.is_empty()).then_some(now);
        Some(std::mem::replace(&mut self.held, rest))
    }

    /// Returns everything still held, at the end of the answer.
    pub fn finish(&mut self) -> String {
        self.since = None;
        self.started = false;
        std::mem::take(&mut self.held)
    }
}

/// Writes `text` whole; `accepted` counts the bytes the writer took, even on an error.
fn emit(out: &mut impl Write, text: &str, accepted: &mut usize) -> std::io::Result<()> {
    let mut bytes = text.as_bytes();
    while !bytes.is_empty() {
        match out.write(bytes) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => {
                *accepted += n;
                bytes = &bytes[n..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    out.flush()
}

/// Engine metrics recorded only when a natural terminal event supplies them.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    /// Prompt token count.
    pub prompt_tokens: u32,
    /// Sampled token count, including the final token.
    pub completion_tokens: u32,
    /// Prompt tokens reused by the engine.
    pub cached_tokens: u32,
    /// Time to first token in milliseconds.
    pub ttft_ms: f64,
    /// Time per subsequent token in milliseconds.
    pub tpot_ms: f64,
    /// Engine-reported decoding throughput.
    pub tok_s: f64,
}

/// Stable transcript terminal names, independent of records' Debug representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminal {
    /// An engine stop token.
    Eos,
    /// The generated-token limit.
    Length,
    /// A caller-side string stop acknowledged by the engine.
    StopSeq,
    /// An interrupted response.
    Cancelled,
    /// A failed turn, removed from history.
    Error,
    /// The conversation was reset.
    Reset,
}

impl From<crate::Finish> for Terminal {
    fn from(finish: crate::Finish) -> Self {
        match finish {
            crate::Finish::Eos => Self::Eos,
            crate::Finish::Length => Self::Length,
            crate::Finish::StopSeq => Self::StopSeq,
            crate::Finish::None | crate::Finish::Unknown => {
                unreachable!("the engine ends Done with Eos, Length or StopSeq")
            }
        }
    }
}

/// A replayed transcript: the policy, the delivered history and the last record's outcome.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Persisted policy and original system message.
    pub options: SessionOptions,
    /// Complete delivered history, including truncated answers.
    pub history: Vec<ChatMessage>,
    /// The last turn or reset; None before the first.
    pub terminal: Option<Terminal>,
    /// None for cancellations, failures, and resets.
    pub metrics: Option<Metrics>,
    /// Exchanges removed for the last turn's context limit.
    pub dropped_exchanges: u32,
}

fn validate_metrics(metrics: Option<&Metrics>) -> Result<()> {
    if let Some(m) = metrics {
        ensure!(
            [m.ttft_ms, m.tpot_ms, m.tok_s]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "invalid transcript timing"
        );
        ensure!(
            m.cached_tokens <= m.prompt_tokens,
            "invalid transcript cached count"
        );
    }
    Ok(())
}

// A journal is one version-1 header (policy and history), then one record per turn or reset, so
// earlier history is never rewritten. A record is version 2, or 3 when a message of its exchange
// names images: a reader from before images refuses that one by its version, and a text-only chat
// stays readable by it.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    version: u32,
    options: SessionOptions,
    history: Vec<ChatMessage>,
}

impl Header {
    fn check(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported transcript version {}",
            self.version
        );
        ensure!(
            crate::session::valid_history(&self.options, &self.history),
            "invalid transcript history or policy"
        );
        Ok(())
    }

    fn snapshot(self) -> Snapshot {
        Snapshot {
            options: self.options,
            history: self.history,
            terminal: None,
            metrics: None,
            dropped_exchanges: 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TurnRecord {
    version: u32,
    terminal: Terminal,
    metrics: Option<Metrics>,
    dropped_exchanges: u32,
    /// The turn's prompt or tool results, then its answer.
    exchange: Option<Vec<ChatMessage>>,
}

impl TurnRecord {
    // What a record must hold on its own, checked before it is written and when it is replayed.
    fn check(&self) -> Result<()> {
        ensure!(
            self.version == record_version(self.exchange.as_deref().unwrap_or_default()),
            "unsupported transcript version {}",
            self.version
        );
        validate_metrics(self.metrics.as_ref())?;
        ensure!(
            self.metrics.is_some()
                == matches!(
                    self.terminal,
                    Terminal::Eos | Terminal::Length | Terminal::StopSeq
                ),
            "metrics do not match the terminal"
        );
        // A reset or a failed turn adds and drops nothing; any other turn adds one exchange.
        let empty = matches!(self.terminal, Terminal::Reset | Terminal::Error);
        ensure!(
            self.exchange.is_none() == empty && (!empty || self.dropped_exchanges == 0),
            "exchange does not match the terminal"
        );
        if let Some(messages) = &self.exchange {
            // One exchange: prompts, then the answer. Each message's fields, and what it
            // follows, are checked on replay.
            let Some((answer, prompts)) = messages.split_last() else {
                bail!("invalid transcript roles");
            };
            ensure!(
                answer.role == Role::Assistant
                    && !prompts.is_empty()
                    && prompts.iter().all(|m| m.role != Role::Assistant),
                "invalid transcript roles"
            );
            ensure!(
                answer.truncated == (self.terminal == Terminal::Cancelled),
                "invalid transcript truncation"
            );
        }
        Ok(())
    }

    // Every record keeps the history valid_history accepts, dropping whole exchanges as the session
    // did: from the history with the turn's message, never the exchange it belongs to.
    fn apply(&self, next: &mut Snapshot) -> Result<()> {
        use crate::session::{follows, oldest_exchange};
        self.check()?;
        let first = usize::from(next.options.system.is_some());
        if self.terminal == Terminal::Reset {
            next.history.truncate(first);
        }
        for message in self.exchange.iter().flatten() {
            ensure!(
                follows(&next.history[first..], message),
                "the turn does not continue the history"
            );
            next.history.push(message.clone());
        }
        for _ in 0..self.dropped_exchanges {
            let len = oldest_exchange(&next.history[first..])
                .context("invalid dropped exchange count")?;
            next.history.drain(first..first + len);
        }
        next.terminal = Some(self.terminal);
        next.metrics = self.metrics.clone();
        next.dropped_exchanges = self.dropped_exchanges;
        Ok(())
    }
}

/// The version of a turn record whose exchange is `messages`: 3 when one names images, else 2.
fn record_version(messages: &[ChatMessage]) -> u32 {
    if messages.iter().any(|m| !m.images.is_empty()) {
        3
    } else {
        2
    }
}

// The shapes share only `version`: 1 is the header, anything else a turn record, which checks its
// own. Not `#[serde(untagged)]`: it buffers numbers, which serde_json's `arbitrary_precision` (on
// whenever strided is linked, feature `tune`) hands over as maps, so no record would parse.
enum Line {
    Header(Header),
    Turn(TurnRecord),
}

impl Line {
    fn parse(line: &[u8]) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(line)?;
        Ok(
            if value.get("version").and_then(|v| v.as_u64()) == Some(1) {
                Line::Header(Header::deserialize(&value)?)
            } else {
                Line::Turn(TurnRecord::deserialize(&value)?)
            },
        )
    }
}

/// A locked journal; failed writes roll back before another append can proceed.
pub struct Transcript {
    file: File,
    poisoned: bool,
    // The replayed journal, so a record that would not replay is refused before it is written.
    latest: Option<Snapshot>,
}

impl Drop for Transcript {
    fn drop(&mut self) {
        // Close alone leaves the lock on descriptions briefly inherited by concurrent fork/exec.
        let _ = self.file.unlock();
    }
}

impl Transcript {
    /// Creates a new transcript that starts from `history`, without overwriting an existing one.
    pub fn create(path: &Path, options: SessionOptions, history: Vec<ChatMessage>) -> Result<Self> {
        let header = Header {
            version: 1,
            options,
            history,
        };
        // Refused before the file exists, so a history that would not replay leaves no file.
        header.check()?;
        let dir = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        #[cfg(unix)]
        let created = dir
            .ancestors()
            .take_while(|dir| !dir.as_os_str().is_empty() && !dir.exists())
            .count();
        std::fs::create_dir_all(dir)?;
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        // The appends are synced; the file's entry, and those of directories created for it, must
        // survive a crash too.
        #[cfg(unix)]
        for dir in dir.ancestors().take(created + 1) {
            let dir = Some(dir).filter(|dir| !dir.as_os_str().is_empty());
            File::open(dir.unwrap_or(Path::new(".")))?.sync_all()?;
        }
        let mut transcript = Self::locked(file, path)?;
        transcript.start(header.options, header.history)?;
        Ok(transcript)
    }

    fn locked(file: File, path: &Path) -> Result<Self> {
        match file.try_lock() {
            Ok(()) => Ok(Self {
                file,
                poisoned: false,
                latest: None,
            }),
            Err(TryLockError::WouldBlock) => bail!(
                "transcript {} is in use by another larch chat",
                path.display()
            ),
            Err(TryLockError::Error(error)) => {
                Err(error).with_context(|| format!("locking transcript {}", path.display()))
            }
        }
    }

    /// Validates and replays every complete record, repairing only an incomplete trailing write.
    pub fn resume(path: &Path) -> Result<(Self, Snapshot)> {
        let (transcript, latest) = Self::recover(path, false)?;
        Ok((transcript, latest.context("empty transcript")?))
    }

    // `initialize`: an existing file without a complete line is a create that did not finish.
    fn recover(path: &Path, initialize: bool) -> Result<(Self, Option<Snapshot>)> {
        // Full write access supports Windows locking and truncating an incomplete tail.
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut transcript = Self::locked(file, path)?;
        let mut bytes = Vec::new();
        transcript.file.read_to_end(&mut bytes)?;
        let end = match bytes.iter().rposition(|&b| b == b'\n') {
            Some(last) => last + 1,
            None if initialize => 0,
            None => bail!("transcript has no complete header"),
        };
        let mut latest = None;
        for (i, line) in bytes[..end].split_inclusive(|&b| b == b'\n').enumerate() {
            let mut read = || -> Result<()> {
                match Line::parse(&line[..line.len() - 1])? {
                    Line::Header(header) => {
                        ensure!(latest.is_none(), "header after the first line");
                        header.check()?;
                        latest = Some(header.snapshot());
                    }
                    Line::Turn(record) => {
                        record.apply(latest.as_mut().context("journal missing its header")?)?
                    }
                }
                Ok(())
            };
            read().with_context(|| format!("invalid transcript line {}", i + 1))?;
        }
        if end < bytes.len() {
            transcript.file.set_len(end as u64)?;
            transcript.file.sync_data()?;
        }
        transcript.latest.clone_from(&latest);
        Ok((transcript, latest))
    }

    /// Writes and syncs one turn: `exchange` is the prompt or tool result and its delivered answer
    /// (None for a failed turn), after removing the `dropped_exchanges` oldest exchanges. A turn
    /// that would not replay onto the journal is refused unwritten.
    pub fn append_turn(
        &mut self,
        terminal: Terminal,
        metrics: Option<Metrics>,
        dropped_exchanges: u32,
        exchange: Option<[ChatMessage; 2]>,
    ) -> Result<()> {
        let messages = exchange.map_or_else(Vec::new, Vec::from);
        self.append_messages(terminal, metrics, dropped_exchanges, messages)
    }

    /// [`append_turn`](Self::append_turn) for a turn of several tool results: `messages` is the
    /// prompt or the results, then the answer; empty for a failed turn or a reset.
    pub fn append_messages(
        &mut self,
        terminal: Terminal,
        metrics: Option<Metrics>,
        dropped_exchanges: u32,
        messages: Vec<ChatMessage>,
    ) -> Result<()> {
        let record = TurnRecord {
            version: record_version(&messages),
            terminal,
            metrics,
            dropped_exchanges,
            exchange: (!messages.is_empty()).then_some(messages),
        };
        let mut next = self.latest.clone().context("journal missing its header")?;
        record.apply(&mut next)?;
        self.write(&record)?;
        self.latest = Some(next);
        Ok(())
    }

    /// Writes and syncs a reset to the system message.
    pub fn append_reset(&mut self) -> Result<()> {
        self.append_turn(Terminal::Reset, None, 0, None)
    }

    // The header: written by create, or by chat into a file whose create did not finish.
    fn start(&mut self, options: SessionOptions, history: Vec<ChatMessage>) -> Result<()> {
        let header = Header {
            version: 1,
            options,
            history,
        };
        header.check()?;
        self.write(&header)?;
        self.latest = Some(header.snapshot());
        Ok(())
    }

    fn write(&mut self, line: &impl Serialize) -> Result<()> {
        self.write_with(line, |file, line| file.write_all(line))
    }

    fn write_with(
        &mut self,
        line: &impl Serialize,
        write: impl FnOnce(&mut File, &[u8]) -> std::io::Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "transcript writer poisoned; reopen through resume"
        );
        let mut line = serde_json::to_vec(line)?;
        line.push(b'\n');
        let length = self.file.seek(SeekFrom::End(0))?;
        let result = write(&mut self.file, &line)
            .and_then(|()| self.file.flush())
            .and_then(|()| self.file.sync_data());
        if let Err(error) = result {
            let rollback = self
                .file
                .set_len(length)
                .and_then(|()| self.file.seek(SeekFrom::Start(length)).map(|_| ()))
                .and_then(|()| self.file.sync_data());
            if let Err(rollback) = rollback {
                self.poisoned = true;
                return Err(error).context(format!("transcript rollback failed ({rollback}); writer poisoned; reopen through resume"));
            }
            return Err(error).context("transcript append rolled back");
        }
        Ok(())
    }
}

/// The folder of the images the transcript at `path` names, beside it as `<id>.attached.json` is:
/// `chat/<id>.images/`, one file per image, named by its reference (the sha256 of its bytes).
pub fn images_dir(path: &Path) -> PathBuf {
    path.with_extension("images")
}

/// Stores `image` beside the transcript at `path`, once: a temporary file, fsync, rename, then the
/// folder's fsync, so a crash or a full disk leaves either the whole file or none. Called before the
/// turn whose journal line names it is written, so no line names a file that is not there. A file
/// already there with other bytes (damaged on disk, or not a plain file) is replaced.
pub fn store_image(path: &Path, image: &Image) -> Result<()> {
    let dir = images_dir(path);
    let file = dir.join(image.hex());
    let intact = std::fs::symlink_metadata(&file)
        .is_ok_and(|m| m.is_file() && m.len() == image.bytes().len() as u64)
        && std::fs::read(&file).is_ok_and(|b| *b == **image.bytes());
    if intact {
        return Ok(());
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    write_new(&dir, &file, &format!(".{}.tmp", image.hex()), image.bytes())
        .with_context(|| format!("storing an image in {}", dir.display()))
}

/// Writes `bytes` to `file` in `dir` through the temporary `tmp`: created new (never through a
/// link or into a file left there), fsynced, renamed over `file`, then the folder fsynced. A
/// failure removes the temporary file.
fn write_new(dir: &Path, file: &Path, tmp: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = dir.join(tmp);
    // A stale one (a crash between create and rename) goes; unlinking never follows a link.
    match std::fs::remove_file(&tmp) {
        Err(e) if e.kind() != ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let written = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true) // O_EXCL: refuses an existing name, a link included
        .open(&tmp)
        .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
        .and_then(|()| std::fs::rename(&tmp, file))
        .and_then(|()| sync_dir(dir));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Makes a rename in `dir` durable (a no-op where the platform cannot open a folder).
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// The images `history` names, read from beside the transcript at `path`, and one notice for each it
/// cannot use: a reference that is not a sha256 (a hand-edited transcript; it never becomes a path),
/// a missing file, or one whose bytes no longer hash to its name. Those messages render as their text.
pub fn load_images(path: &Path, history: &[ChatMessage]) -> (Vec<Arc<Image>>, Vec<String>) {
    let dir = images_dir(path);
    let (mut images, mut notices) = (Vec::new(), Vec::new());
    let mut seen = std::collections::HashSet::new();
    for r in history.iter().flat_map(|m| &m.images) {
        if !seen.insert(r.clone()) {
            continue;
        }
        if !crate::session::image_ref(r) {
            notices.push(
                "an image reference in this chat is not valid; the model reads its note or [image] in its place"
                    .into(),
            );
            continue;
        }
        let short = &r[..12];
        match std::fs::read(dir.join(r)) {
            Err(_) => notices.push(format!(
                "image {short} is missing from this chat; the model reads its note or [image] in its place"
            )),
            Ok(bytes) => match Image::new(bytes.into()) {
                Ok(img) if img.hex() == *r => images.push(Arc::new(img)),
                Ok(_) => notices.push(format!(
                    "image {short} was changed on disk; the model reads its note or [image] in its place"
                )),
                Err(e) => notices.push(format!(
                    "image {short} cannot be read ({e}); the model reads its note or [image] in its place"
                )),
            },
        }
    }
    (images, notices)
}

/// The longest note [`store_note`] keeps, in bytes: a description is about 120 words.
pub const MAX_NOTE_BYTES: usize = 4096;

/// Stores `note`, the description of the image `reference` names, beside the transcript at `path`
/// as `<reference>.txt` in its images folder: a temporary file, fsync, rename. A later one replaces
/// it. Over [`MAX_NOTE_BYTES`] it is cut at a character boundary.
pub fn store_note(path: &Path, reference: &str, note: &str) -> Result<()> {
    if !crate::session::image_ref(reference) {
        anyhow::bail!("not an image reference");
    }
    let mut end = note.len().min(MAX_NOTE_BYTES);
    while !note.is_char_boundary(end) {
        end -= 1;
    }
    let dir = images_dir(path);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let file = dir.join(format!("{reference}.txt"));
    write_new(
        &dir,
        &file,
        &format!(".{reference}.txt.tmp"),
        &note.as_bytes()[..end],
    )
    .with_context(|| format!("storing an image note in {}", dir.display()))
}

/// The notes stored beside the transcript at `path` for the images `history` names, by reference;
/// a missing, unreadable or oversized one is skipped (the image then stays pixels until a new note).
pub fn load_notes(path: &Path, history: &[ChatMessage]) -> HashMap<String, String> {
    let dir = images_dir(path);
    let mut notes = HashMap::new();
    for r in history.iter().flat_map(|m| &m.images) {
        if notes.contains_key(r) || !crate::session::image_ref(r) {
            continue;
        }
        let file = dir.join(format!("{r}.txt"));
        if let Ok(bytes) = std::fs::read(&file)
            && bytes.len() <= MAX_NOTE_BYTES
            && let Ok(text) = String::from_utf8(bytes)
        {
            notes.insert(r.clone(), text);
        }
    }
    notes
}

/// Reads back into `session` (after its `restore`) the images `history` names, with their notes.
/// A note is stored only once a prompt needed its image as words, so a stored one stays a note
/// (as it did before the chat closed). Returns a notice per image that could not be read.
pub fn resume_images(session: &mut Session, path: &Path, history: &[ChatMessage]) -> Vec<String> {
    let (images, notices) = load_images(path, history);
    session.add_images(images);
    for (r, note) in load_notes(path, history) {
        session.set_image_note(&r, &note);
        session.keep_as_note(&r);
    }
    notices
}

/// Deletes the images (and their notes) beside the transcript at `path` that `history` no longer
/// names: those of exchanges dropped to fit the context or summarized away, so a picture and its
/// EXIF do not outlive the messages that carried them. Call it after the journal line that dropped
/// them is written (a crash before then leaves files a later call removes). Returns how many went.
pub fn prune_images(path: &Path, history: &[ChatMessage]) -> Result<usize> {
    let dir = images_dir(path);
    let named: std::collections::HashSet<&str> = history
        .iter()
        .flat_map(|m| m.images.iter().map(String::as_str))
        .collect();
    let entries = match std::fs::read_dir(&dir) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(0),
        r => r.with_context(|| format!("reading {}", dir.display()))?,
    };
    let mut gone = 0;
    for e in entries {
        let e = e.context("reading the chat's images")?;
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        let reference = name.strip_suffix(".txt").unwrap_or(name);
        if crate::session::image_ref(reference) && !named.contains(reference) {
            std::fs::remove_file(e.path()).context("deleting an image the chat no longer has")?;
            gone += 1;
        }
    }
    if gone > 0 {
        sync_dir(&dir).context("syncing the chat's images")?;
    }
    Ok(gone)
}

/// Deletes the images of the transcript at `path` (its chat is being deleted).
pub fn delete_images(path: &Path) -> Result<()> {
    match std::fs::remove_dir_all(images_dir(path)) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e).context("deleting the chat's images"),
        _ => Ok(()),
    }
}

/// The per-user larch directory (architecture 14): under %LOCALAPPDATA%, ~/Library/Application
/// Support, $XDG_DATA_HOME or ~/.local/share. Chat keeps transcripts in its chat/, gait its journal in gait/.
pub fn data_dir() -> Result<PathBuf> {
    #[cfg(windows)]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute()) // the XDG spec ignores empty and relative values
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    Ok(base
        .context("set --data-dir: no per-user data directory")?
        .join("larch"))
}

/// Resolves a transcript ID without allowing directory traversal.
pub fn transcript_path(base: &Path, id: &str) -> Result<PathBuf> {
    ensure!(
        !id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "invalid transcript ID"
    );
    Ok(base.join("chat").join(format!("{id}.jsonl")))
}

/// The active turn, numbered by its prompt, and the prompts a `/reset` followed.
#[derive(Default)]
struct Turns {
    active: Option<(u64, larch_engine::RequestCanceller)>,
    resets: Vec<u64>,
}

impl Turns {
    // A `/reset` cancels the turn of the prompt it follows, read during that turn or before it.
    fn cancel(&self) {
        if let Some((prompt, turn)) = &self.active
            && self.resets.contains(prompt)
        {
            turn.cancel();
        }
    }
}

/// Reads input on its own thread, so `/reset` can cancel the answer streaming on the main one.
struct Input {
    /// Each line with the number of prompts read up to it.
    lines: Receiver<(u64, std::io::Result<String>)>,
    /// Joined at the end of input, to tell a reader that panicked from EOF.
    worker: Option<thread::JoinHandle<()>>,
    turns: Arc<Mutex<Turns>>,
    /// The number sent with the line received last: a prompt's own number.
    prompt: u64,
}

impl Input {
    /// The worker stops after `/quit`, at the end of input, or at the first send after `run`
    /// returned; a read that blocks is not interrupted.
    fn start(mut reader: impl BufRead + Send + 'static) -> Self {
        let (tx, lines) = mpsc::channel();
        let turns = Arc::new(Mutex::new(Turns::default()));
        let shared = turns.clone();
        let worker = thread::spawn(move || {
            let mut prompts = 0;
            loop {
                let mut line = String::new();
                let line = match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => Ok(line.trim().to_owned()),
                    Err(error) => Err(error),
                };
                if line.as_ref().is_ok_and(String::is_empty) {
                    continue;
                }
                // `run` sends the same trimmed, non-command lines as prompts.
                prompts += u64::from(line.as_ref().is_ok_and(|line| !line.starts_with('/')));
                if line.as_ref().is_ok_and(|line| line == "/reset") {
                    let mut turns = shared.lock().expect("input");
                    turns.resets.push(prompts);
                    turns.cancel();
                }
                // A non-UTF-8 line is skipped with a message; other read errors end the input.
                let last = line.as_ref().map_or_else(
                    |error| error.kind() != ErrorKind::InvalidData,
                    |line| line == "/quit",
                );
                if tx.send((prompts, line)).is_err() || last {
                    break;
                }
            }
        });
        Self {
            lines,
            worker: Some(worker),
            turns,
            prompt: 0,
        }
    }

    /// None at the end of input; a reader that panicked is an error.
    fn next_line(&mut self) -> Option<std::io::Result<String>> {
        let Ok((prompt, line)) = self.lines.recv() else {
            let panicked = self.worker.take()?.join().is_err();
            return panicked.then(|| Err(std::io::Error::other("input reader panicked")));
        };
        self.prompt = prompt;
        Some(line)
    }

    /// Whether a `/reset` was already read after the prompt just received.
    fn reset_follows(&self) -> bool {
        let turns = self.turns.lock().expect("input");
        turns.resets.contains(&self.prompt)
    }

    /// Sets the turn of the prompt just received, or clears it.
    fn active(&self, turn: Option<larch_engine::RequestCanceller>) {
        let mut turns = self.turns.lock().expect("input");
        turns.resets.retain(|&prompt| prompt >= self.prompt); // earlier turns are over
        turns.active = turn.map(|turn| (self.prompt, turn));
        turns.cancel();
    }
}

fn validate_gpu(
    device: Option<i32>,
    count: i32,
    open: impl FnOnce() -> std::result::Result<(), larch_kernels::KError>,
) -> Result<i32> {
    let ordinal = device.unwrap_or(0);
    ensure!(
        ordinal >= 0,
        "invalid GPU ordinal {ordinal}: use --cpu for reference/debug mode"
    );
    if count == 0 {
        // Enumeration also returns zero for an insufficient driver; opening retains its versions.
        // No driver, or one that cannot report its version (have 0), is no GPU at all.
        match open() {
            Err(larch_kernels::KError::Driver { have, need }) if have > 0 => {
                return Err(crate::LoadError::DriverTooOld { have, need }.into());
            }
            Ok(())
            | Err(
                larch_kernels::KError::InvalidArg | larch_kernels::KError::Driver { have: 0, .. },
            ) => {}
            Err(error) => return Err(error).context("opening GPU device"),
        }
        // Only a backend with no GPU at all falls back; an asked-for --device stays an error (E2b).
        if device.is_none() {
            return Ok(larch_kernels::CPU);
        }
    }
    ensure!(
        count > 0,
        "no GPU device available in this backend; use --cpu for reference/debug mode"
    );
    ensure!(
        (0..count).contains(&ordinal),
        "invalid GPU ordinal {ordinal}: {count} available"
    );
    Ok(ordinal)
}

/// Runs chat after the caller has installed the supervisor guard; `input` is read only after a
/// successful startup.
pub fn run(
    opts: Options,
    input: impl BufRead + Send + 'static,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<()> {
    let mut load = LoadOptions::new(
        opts.model,
        if opts.cpu {
            larch_kernels::CPU
        } else {
            opts.device.unwrap_or(0)
        },
    );
    load.on_fault = larch_engine::OnFault::Exit;
    if opts.drafter.is_some() {
        load.drafter = opts.drafter;
    }
    if opts.drafter_auto.is_some() {
        load.drafter_auto = opts.drafter_auto;
    }
    load.mmproj = opts.mmproj;
    load.knob_overrides = opts
        .set
        .iter()
        .map(|s| crate::knob::assignment(s))
        .collect::<Result<_>>()?;
    let base = opts.data_dir.map_or_else(data_dir, Ok)?;
    let session_opts = SessionOptions {
        system: opts.system,
        keep_kv: !opts.no_keep_kv,
        on_overflow: if opts.fail_overflow {
            Overflow::Fail
        } else {
            Overflow::DropOldestExchanges
        },
    };
    let id = match opts.resume {
        Some(id) => id,
        None => format!(
            "{}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
            std::process::id()
        ),
    };
    let path = transcript_path(&base, &id)?;
    // An existing transcript is locked and checked before the weights load, so a bad one fails
    // fast under the supervisor; a new one is created only after a successful load.
    let existing = path
        .try_exists()?
        .then(|| Transcript::recover(&path, true))
        .transpose()?;
    if let Some((_, Some(snapshot))) = &existing {
        ensure!(
            session_opts
                .system
                .as_ref()
                .is_none_or(|system| snapshot.options.system.as_ref() == Some(system))
                && (!opts.no_keep_kv || !snapshot.options.keep_kv)
                && (!opts.fail_overflow || snapshot.options.on_overflow == Overflow::Fail),
            "resume options differ from persisted policy"
        );
    }
    if !opts.cpu {
        let count = larch_kernels::device_count().with_context(|| {
            format!(
                "checking GPU devices and driver: {}",
                larch_kernels::last_error()
            )
        })?;
        load.device = validate_gpu(opts.device, count, || {
            larch_kernels::Device::open(load.device).map(drop)
        })?;
        if load.device == larch_kernels::CPU {
            writeln!(
                err,
                "larch: no usable GPU found; running on the CPU reference (slow)"
            )?;
        }
    }
    let larch = Larch::load(load)?;
    // Strict CLI: overrides the engine could not apply fail startup, by name (10.2).
    let rejected: Vec<_> = larch
        .rejected_overrides()
        .iter()
        .map(|(knob, value, reason)| {
            let name = larch_records::knob_spec(*knob).map_or("?", |k| k.name.as_str());
            format!("--set {name}={value} rejected: {reason:?}")
        })
        .collect();
    ensure!(rejected.is_empty(), "{}", rejected.join("; "));
    // A fault exits the process (9.11); it waits, up to the engine's deadline, for the failed turn
    // to be reported, so the caller's error is not lost to the exit.
    // ponytail: chat holds the engine's one flush slot; compose the hooks here when 9.11's ring flush
    // lands.
    let (reported, report) = mpsc::sync_channel::<()>(1);
    larch.runtime.engine.set_fault_flush(move |_, deadline| {
        let _ = report.recv_timeout(deadline.saturating_duration_since(Instant::now()));
    })?;
    let mut session = larch.session(session_opts);
    let mut transcript = match existing {
        Some((transcript, Some(snapshot))) => {
            let history = snapshot.history.clone();
            session.restore(snapshot.options, snapshot.history)?;
            for n in resume_images(&mut session, &path, &history) {
                writeln!(err, "{n}")?;
            }
            writeln!(
                err,
                "resumed delivered history; KV is re-prefilled on the first turn"
            )?;
            transcript
        }
        Some((mut transcript, None)) => {
            transcript.start(session.opts.clone(), session.history())?;
            transcript
        }
        None => Transcript::create(&path, session.opts.clone(), session.history())?,
    };
    writeln!(err, "{}", crate::supervise::READY_LINE)?;
    writeln!(err, "transcript {id}; /status, /knob, /ab, /reset, /quit")?;
    if opts.cpu {
        writeln!(err, "CPU reference/debug mode")?;
    }
    let params = GenParams {
        max_tokens: opts.max_tokens.unwrap_or(0),
        temperature: opts.temperature,
        stop: opts.stop,
        ..GenParams::default()
    };
    let mut input = Input::start(input);
    loop {
        write!(out, "> ")?;
        out.flush()?;
        let line = match input.next_line() {
            Some(Err(error)) if error.kind() == ErrorKind::InvalidData => {
                writeln!(err, "{error}")?;
                continue;
            }
            Some(line) => line?,
            None => break,
        };
        let user = line.as_str();
        if user.starts_with('/') {
            let words: Vec<_> = user.split_whitespace().collect();
            match words.as_slice() {
                ["/quit"] => break,
                ["/status"] => writeln!(
                    out,
                    "health {:?}; rev {}; session {}; parked {} tokens",
                    larch.health(),
                    larch.knobs().effective().rev,
                    session.id(),
                    session.tokens_in_kv()
                )?,
                ["/reset"] => {
                    session.reset();
                    transcript.append_reset()?;
                    writeln!(out, "conversation reset; parked KV released")?;
                }
                ["/knob", args @ ..] => {
                    if let Err(error) = crate::knob::command(larch.knobs().as_ref(), args, out) {
                        writeln!(err, "{error}")?;
                    }
                }
                ["/ab", args @ ..] => {
                    if let Err(error) = crate::knob::ab(larch.knobs().as_ref(), args, out) {
                        writeln!(err, "{error}")?;
                    }
                }
                _ => writeln!(err, "unsupported command: {user}")?,
            }
            continue;
        }
        // Its turn would be cancelled at once and then reset.
        if input.reset_follows() {
            continue;
        }
        let guessed = larch.guess_stats();
        let mut stream = match session.send(user, params.clone()) {
            Ok(stream) => stream,
            Err(error) => {
                writeln!(err, "{error}")?;
                continue;
            }
        };
        let mut dropped = stream.dropped_exchanges();
        let mut terminal = Terminal::Error;
        let mut metrics = None;
        let mut failure = None;
        let mut accepted = 0;
        let mut batch = Batcher::default();
        input.active(Some(stream.handle.canceller()));
        for event in stream.by_ref() {
            match event {
                TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
                TextEvent::Text { delta, .. } => {
                    let Some(text) = batch.push(&delta, Instant::now()) else {
                        continue;
                    };
                    if let Err(error) = emit(out, &text, &mut accepted) {
                        stream.cancel_after_output(accepted);
                        transcript
                            .append_turn(
                                Terminal::Cancelled,
                                None,
                                dropped,
                                session.last_exchange(),
                            )
                            .context("persisting interrupted turn")?;
                        return Err(error.into());
                    }
                }
                TextEvent::Done {
                    finish,
                    usage,
                    timings,
                    ..
                } => {
                    terminal = finish.into();
                    metrics = Some(Metrics {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        cached_tokens: usage.cached_tokens,
                        ttft_ms: timings.ttft_ms,
                        tpot_ms: timings.tpot_ms,
                        tok_s: timings.tok_s,
                    });
                }
                TextEvent::Error(error) => {
                    // A failed turn restores the history it dropped from.
                    if error == crate::EngineError::Cancelled {
                        terminal = Terminal::Cancelled;
                    } else {
                        dropped = 0;
                    }
                    failure = Some(error);
                }
            }
        }
        input.active(None);
        // The stream committed the held tail with its terminal event; a writer that drops it keeps
        // only what it took, as a write error mid-answer does.
        let tail = emit(out, &batch.finish(), &mut accepted);
        if let Err(error) = tail
            && terminal != Terminal::Error
        {
            let pair = session.last_exchange().map(|[user, mut answer]| {
                answer
                    .content
                    .truncate(answer.content.floor_char_boundary(accepted));
                answer.truncated = true;
                [user, answer]
            });
            transcript
                .append_turn(Terminal::Cancelled, None, dropped, pair)
                .context("persisting interrupted turn")?;
            return Err(error.into());
        }
        let pair = (terminal != Terminal::Error)
            .then(|| session.last_exchange())
            .flatten();
        // An engine fault exits within its 20 ms deadline: report before the record's sync, which
        // can outlast it (F_FULLFSYNC, FlushFileBuffers). Its Error record carries no exchange.
        let fault = terminal == Terminal::Error
            && failure.is_some()
            && larch.health().state == larch_records::HealthState::Failed;
        if !fault {
            transcript.append_turn(terminal, metrics.clone(), dropped, pair)?;
        }
        if let Some(error) = failure {
            writeln!(err, "{error}")?;
        }
        // A cut inside a withheld block prints nothing of it; a cancel or an error says why itself.
        if terminal == Terminal::Length && stream.in_thinking() {
            let fix = if params.max_tokens == 0 {
                "the context is full; /reset"
            } else {
                "raise --max-tokens"
            };
            writeln!(err, "the answer ended inside a hidden think block; {fix}")?;
        }
        writeln!(out)?;
        if let Some(m) = &metrics {
            writeln!(
                out,
                "{:.2} tok/s · TTFT {:.2} ms · {} prompt + {} generated · {} cached · {} parked",
                m.tok_s,
                m.ttft_ms,
                m.prompt_tokens,
                m.completion_tokens,
                m.cached_tokens,
                session.tokens_in_kv()
            )?;
        } else {
            writeln!(
                out,
                "{:?} · engine usage/timings unavailable · {} parked",
                terminal,
                session.tokens_in_kv()
            )?;
        }
        let guessed = larch.guess_stats().since(guessed);
        if guessed.k.iter().any(|&n| n > 0) {
            writeln!(err, "{guessed}")?;
        }
        if dropped > 0 {
            writeln!(out, "(dropped {dropped} oldest exchanges)")?;
        }
        if fault {
            out.flush()?;
            err.flush()?;
            transcript.append_turn(terminal, metrics, dropped, None)?;
            let _ = reported.try_send(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;

    /// One of larch-model's small decoding fixtures.
    fn fixture(name: &str) -> Arc<Image> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../larch-model/tests/data/vision")
            .join(name);
        Arc::new(Image::new(std::fs::read(p).unwrap().into()).unwrap())
    }

    fn user_with(images: &[&str]) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: "look".into(),
            images: images.iter().map(|s| s.to_string()).collect(),
            ..ChatMessage::default()
        }
    }

    #[test]
    fn a_text_message_serializes_as_before_and_an_old_line_loads() {
        let m = ChatMessage {
            role: Role::User,
            content: "hi".into(),
            ..ChatMessage::default()
        };
        let line = serde_json::to_string(&m).unwrap();
        assert_eq!(line, r#"{"role":"user","content":"hi","truncated":false}"#);
        let back: ChatMessage = serde_json::from_str(&line).unwrap();
        assert_eq!(back, m);
        let with = user_with(&[&"a".repeat(64)]);
        let line = serde_json::to_string(&with).unwrap();
        assert!(line.contains(r#""images":[""#), "{line}");
        // No bytes or base64 in the line: only the 64-digit reference.
        assert!(line.len() < 150, "{line}");
    }

    #[test]
    fn image_notes_are_stored_beside_their_images_and_read_back() {
        let dir = std::env::temp_dir().join(format!("larch-img-{}-note", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c1.jsonl");
        let a = fixture("rgba.png");
        store_image(&path, &a).unwrap();
        store_note(&path, &a.hex(), "a green square").unwrap();
        store_note(&path, &a.hex(), "a green square with a hole").unwrap(); // replaces
        assert!(store_note(&path, "../../etc/passwd", "no").is_err());
        let history = [user_with(&[&a.hex(), &"f".repeat(64)])];
        let notes = load_notes(&path, &history);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[&a.hex()], "a green square with a hole");
        // An over-long note is cut at a character boundary, not refused.
        store_note(&path, &a.hex(), &"é".repeat(MAX_NOTE_BYTES)).unwrap();
        assert_eq!(load_notes(&path, &history)[&a.hex()].len(), MAX_NOTE_BYTES);
        let left: Vec<_> = std::fs::read_dir(images_dir(&path)).unwrap().collect();
        assert_eq!(left.len(), 2, "no temporary file stays");
        delete_images(&path).unwrap();
        assert!(load_notes(&path, &history).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn images_are_stored_once_beside_the_transcript_and_read_back() {
        let dir = std::env::temp_dir().join(format!("larch-img-{}-store", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c1.jsonl");
        let (a, b) = (fixture("rgba.png"), fixture("palette.png"));
        store_image(&path, &a).unwrap();
        store_image(&path, &a).unwrap(); // once
        store_image(&path, &b).unwrap();
        let names: Vec<_> = std::fs::read_dir(images_dir(&path))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 2, "no temporary file stays: {names:?}");
        let history = [user_with(&[&a.hex(), &b.hex(), &a.hex()])];
        let (images, notices) = load_images(&path, &history);
        assert_eq!(images.len(), 2);
        assert!(notices.is_empty(), "{notices:?}");
        // Missing, changed on disk, not a reference (a hand-edited `../` path), unknown: notices only.
        std::fs::write(
            images_dir(&path).join(b.hex()),
            fixture("grey16.png").bytes().as_ref(),
        )
        .unwrap();
        let unknown = "f".repeat(64);
        let history = [user_with(&[
            &a.hex(),
            &b.hex(),
            "../../etc/passwd",
            &unknown,
        ])];
        let (images, notices) = load_images(&path, &history);
        assert_eq!(
            images.iter().map(|i| i.hex()).collect::<Vec<_>>(),
            [a.hex()]
        );
        assert_eq!(notices.len(), 3, "{notices:?}");
        assert!(
            notices.iter().any(|n| n.contains("changed"))
                && notices.iter().any(|n| n.contains("not valid"))
        );
        assert!(notices.iter().any(|n| n.contains("missing")));
        delete_images(&path).unwrap();
        assert!(!images_dir(&path).exists());
        delete_images(&path).unwrap(); // nothing left is fine
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_image_write_leaves_no_file() {
        let dir = std::env::temp_dir().join(format!("larch-img-{}-fail", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c2.jsonl");
        // A file where the image folder goes: the write fails as a full disk would, and the caller
        // journals nothing that names it.
        std::fs::write(images_dir(&path), b"in the way").unwrap();
        let a = fixture("grey16.png");
        assert!(store_image(&path, &a).is_err());
        assert!(std::fs::metadata(images_dir(&path)).unwrap().is_file());
        let (images, notices) = load_images(&path, &[user_with(&[&a.hex()])]);
        assert!(images.is_empty() && notices.len() == 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn images_the_history_no_longer_names_are_deleted_with_their_notes() {
        let dir = std::env::temp_dir().join(format!("larch-img-{}-prune", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c4.jsonl");
        assert_eq!(prune_images(&path, &[]).unwrap(), 0, "no folder yet");
        let (a, b) = (fixture("rgba.png"), fixture("palette.png"));
        for img in [&a, &b] {
            store_image(&path, img).unwrap();
            store_note(&path, &img.hex(), "a square").unwrap();
        }
        std::fs::write(images_dir(&path).join("README"), b"not an image").unwrap();
        // `b`'s exchange was dropped: it and its note go; `a`, its note, and other files stay.
        assert_eq!(prune_images(&path, &[user_with(&[&a.hex()])]).unwrap(), 2);
        let mut left: Vec<_> = std::fs::read_dir(images_dir(&path))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            ["README".to_owned(), a.hex(), format!("{}.txt", a.hex())]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn image_files_are_written_new_and_a_damaged_one_is_replaced() {
        let dir = std::env::temp_dir().join(format!("larch-img-{}-link", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(images_dir(&dir.join("c3.jsonl"))).unwrap();
        let path = dir.join("c3.jsonl");
        let a = fixture("rgba.png");
        // A link where the temporary file goes is removed, not written through.
        let outside = dir.join("outside");
        std::fs::write(&outside, b"keep").unwrap();
        let tmp = images_dir(&path).join(format!(".{}.tmp", a.hex()));
        std::os::unix::fs::symlink(&outside, &tmp).unwrap();
        store_image(&path, &a).unwrap();
        assert_eq!(std::fs::read(&outside).unwrap(), b"keep");
        assert!(!tmp.exists());
        // A stored file with other bytes (damaged on disk) is written again.
        let stored = images_dir(&path).join(a.hex());
        std::fs::write(&stored, b"damaged").unwrap();
        store_image(&path, &a).unwrap();
        assert_eq!(std::fs::read(&stored).unwrap(), **a.bytes());
        let note_tmp = images_dir(&path).join(format!(".{}.txt.tmp", a.hex()));
        std::os::unix::fs::symlink(&outside, &note_tmp).unwrap();
        store_note(&path, &a.hex(), "a square").unwrap();
        assert_eq!(std::fs::read(&outside).unwrap(), b"keep");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_turn_with_images_is_a_version_3_record() {
        let path = env::temp_dir().join(format!("larch-journal-v3-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let answer = |text: &str| ChatMessage {
            role: Role::Assistant,
            content: text.into(),
            ..ChatMessage::default()
        };
        let metrics = Metrics {
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: 0,
            ttft_ms: 1.0,
            tpot_ms: 1.0,
            tok_s: 1.0,
        };
        let mut text = user_with(&[]);
        text.content = "hi".into();
        let mut transcript = Transcript::create(&path, SessionOptions::default(), vec![]).unwrap();
        for first in [text, user_with(&[&"a".repeat(64)])] {
            transcript
                .append_turn(
                    Terminal::Eos,
                    Some(metrics.clone()),
                    0,
                    Some([first, answer("ok")]),
                )
                .unwrap();
        }
        drop(transcript);
        let lines = std::fs::read_to_string(&path).unwrap();
        let versions: Vec<_> = lines
            .lines()
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l).unwrap()["version"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(versions, [1, 2, 3]);
        let (t, resumed) = Transcript::resume(&path).unwrap();
        assert_eq!(resumed.history.len(), 4);
        drop(t);
        // An image under version 2, or version 3 without one, does not replay.
        let images = format!(r#","images":["{}"]"#, "a".repeat(64));
        assert!(lines.contains(&images), "{lines}");
        for doctored in [
            lines.replacen(r#""version":3"#, r#""version":2"#, 1),
            lines.replacen(&images, "", 1),
        ] {
            std::fs::write(&path, doctored).unwrap();
            assert!(Transcript::resume(&path).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn batches_whole_words_and_shows_a_spaceless_run_once_it_waited() {
        let t0 = Instant::now();
        let mut b = Batcher::default();
        let pieces = [
            "The", " quick", " bro", "wn fox", " jumps", " over", " the", " lazy", " old",
        ];
        let shown: Vec<_> = pieces.iter().filter_map(|p| b.push(p, t0)).collect();
        assert_eq!(shown, ["The", " quick brown fox jumps over the "]);
        assert_eq!(b.push(" dog. It", t0).as_deref(), Some("lazy old dog. "));
        assert_eq!(b.push("\nfn", t0).as_deref(), Some("It\n"));
        assert_eq!(b.push("()", t0), None);
        assert_eq!(b.push("{}", t0 + BATCH_WAIT).as_deref(), Some("fn(){}"));
        assert_eq!(b.push("x", t0 + BATCH_WAIT), None);
        assert_eq!(b.finish(), "x");
    }

    #[test]
    fn gpu_preflight_rejects_absence_and_invalid_ordinals() {
        let missing = validate_gpu(Some(0), 0, || Err(larch_kernels::KError::InvalidArg))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("no GPU device") && missing.contains("--cpu"));
        let negative = validate_gpu(Some(-1), 0, || Ok(()))
            .unwrap_err()
            .to_string();
        assert!(negative.contains("invalid GPU ordinal -1"));
        assert!(negative.contains("--cpu"));
        let (ordinal, count) = (2, 2);
        let invalid = validate_gpu(Some(ordinal), count, || {
            panic!("valid counts need no probe")
        })
        .unwrap_err()
        .to_string();
        assert!(invalid.contains(&format!("invalid GPU ordinal {ordinal}")));
        assert!(invalid.contains(&format!("{count} available")));
        assert_eq!(
            validate_gpu(Some(0), 1, || panic!("valid counts need no probe")).unwrap(),
            0
        );
        assert_eq!(
            validate_gpu(Some(1), 2, || panic!("valid counts need no probe")).unwrap(),
            1
        );
        assert_eq!(
            validate_gpu(None, 2, || panic!("valid counts need no probe")).unwrap(),
            0
        );
    }

    #[test]
    fn zero_device_count_preserves_driver_failure_instead_of_claiming_no_gpu() {
        // With or without --device: an old driver is never reported as "no GPU" or a CPU fallback.
        for device in [Some(0), None] {
            let error = validate_gpu(device, 0, || {
                Err(larch_kernels::KError::Driver {
                    have: 12000,
                    need: 13000,
                })
            })
            .unwrap_err();
            assert_eq!(
                error.downcast_ref::<crate::LoadError>(),
                Some(&crate::LoadError::DriverTooOld {
                    have: 12000,
                    need: 13000
                })
            );
            assert!(error.to_string().contains("12000") && error.to_string().contains("13000"));
            assert!(!error.to_string().contains("no GPU device"));
        }
    }

    #[test]
    fn no_gpu_without_device_falls_back_to_the_cpu_but_other_open_errors_do_not() {
        // The driver cases are what the CUDA build meets on GPU-less runners: no libcuda in the
        // Linux container (need set), a failing driver query on Windows (both 0).
        for probe in [
            Ok(()),
            Err(larch_kernels::KError::InvalidArg),
            Err(larch_kernels::KError::Driver {
                have: 0,
                need: 13000,
            }),
            Err(larch_kernels::KError::Driver { have: 0, need: 0 }),
        ] {
            assert_eq!(validate_gpu(None, 0, || probe).unwrap(), larch_kernels::CPU);
        }
        let error = validate_gpu(None, 0, || Err(larch_kernels::KError::Stale))
            .unwrap_err()
            .to_string();
        assert!(error.contains("opening GPU device"), "{error}");
    }

    #[test]
    fn load_errors_retain_numeric_diagnostics_and_type() {
        for (error, numbers) in [
            (
                crate::LoadError::DoesNotFit {
                    need: 4096,
                    have: 1024,
                },
                ["4096", "1024"],
            ),
            (
                crate::LoadError::DriverTooOld {
                    have: 12000,
                    need: 13000,
                },
                ["12000", "13000"],
            ),
        ] {
            let diagnostic = anyhow::Error::from(error);
            assert_eq!(diagnostic.downcast_ref::<crate::LoadError>(), Some(&error));
            for number in numbers {
                assert!(diagnostic.to_string().contains(number));
            }
        }
    }

    #[test]
    fn a_turn_of_several_results_replays_whole() {
        let path = env::temp_dir().join(format!(
            "larch-journal-several-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let message = |role, content: &str| ChatMessage {
            role,
            content: content.into(),
            ..ChatMessage::default()
        };
        let call = |id: &str| larch_model::ToolCall {
            name: "f".into(),
            arguments: serde_json::json!({}),
            id: id.into(),
        };
        let result = |id: &str| ChatMessage {
            tool_call_id: id.into(),
            ..message(Role::Tool, "r")
        };
        let calling = ChatMessage {
            tool_calls: vec![call("a"), call("b")],
            ..message(Role::Assistant, "")
        };
        let metrics = Metrics {
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: 0,
            ttft_ms: 1.0,
            tpot_ms: 1.0,
            tok_s: 1.0,
        };
        let opening = [message(Role::User, "q"), calling];
        let mut transcript = Transcript::create(&path, SessionOptions::default(), vec![]).unwrap();
        transcript
            .append_turn(
                Terminal::Eos,
                Some(metrics.clone()),
                0,
                Some(opening.clone()),
            )
            .unwrap();
        let turn = vec![result("a"), result("b"), message(Role::Assistant, "x")];
        // Two exchanges in one record, a result out of the calls' order or twice, and a missing
        // result are refused.
        let mut two = opening.to_vec();
        two.extend([message(Role::User, "q"), message(Role::Assistant, "x")]);
        for wrong in [
            two,
            vec![result("b"), result("a"), message(Role::Assistant, "x")],
            vec![result("a"), result("a"), message(Role::Assistant, "x")],
            // The last result alone, as a two-message window of the turn would hold it.
            vec![result("b"), message(Role::Assistant, "x")],
        ] {
            assert!(
                transcript
                    .append_messages(Terminal::Eos, Some(metrics.clone()), 0, wrong)
                    .is_err()
            );
        }
        transcript
            .append_messages(Terminal::Eos, Some(metrics), 0, turn.clone())
            .unwrap();
        drop(transcript);
        let (transcript, resumed) = Transcript::resume(&path).unwrap();
        let mut history = opening.to_vec();
        history.extend(turn);
        assert_eq!(resumed.history, history);
        drop(transcript);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn partial_append_is_rolled_back_before_retry() {
        let path = env::temp_dir().join(format!(
            "larch-journal-retry-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut transcript = Transcript::create(&path, SessionOptions::default(), vec![]).unwrap();
        let length = transcript.file.metadata().unwrap().len();
        let pair = [
            ChatMessage {
                role: Role::User,
                content: "hello".into(),
                ..ChatMessage::default()
            },
            ChatMessage {
                role: Role::Assistant,
                content: "partial".into(),
                truncated: true,
                ..ChatMessage::default()
            },
        ];
        let turn = TurnRecord {
            version: 2,
            terminal: Terminal::Cancelled,
            metrics: None,
            dropped_exchanges: 0,
            exchange: Some(pair.to_vec()),
        };
        let error = transcript
            .write_with(&turn, |file, line| {
                file.write_all(&line[..17])?;
                Err(std::io::ErrorKind::StorageFull.into())
            })
            .unwrap_err();
        assert!(error.to_string().contains("rolled back"));
        assert_eq!(transcript.file.metadata().unwrap().len(), length);
        transcript
            .append_turn(Terminal::Cancelled, None, 0, Some(pair.clone()))
            .unwrap();
        let reset = TurnRecord {
            version: 2,
            terminal: Terminal::Reset,
            metrics: None,
            dropped_exchanges: 0,
            exchange: None,
        };
        let error = transcript
            .write_with(&reset, |file, line| {
                file.write_all(&line[..17])?;
                // A read-only descriptor makes rollback fail with a real I/O error.
                *file = File::open(&path)?;
                Err(std::io::ErrorKind::StorageFull.into())
            })
            .unwrap_err();
        assert!(error.to_string().contains("writer poisoned"));
        assert!(
            transcript
                .append_reset()
                .unwrap_err()
                .to_string()
                .contains("writer poisoned")
        );
        drop(transcript);
        let (transcript, resumed) = Transcript::resume(&path).unwrap();
        assert_eq!(resumed.history, pair);
        drop(transcript);
        std::fs::remove_file(path).unwrap();
    }
}
