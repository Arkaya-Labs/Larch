//! The audit log: one JSON line per event in `<data_dir>/agent/log/YYYY-MM-DD.jsonl`, with a
//! write's intent and outcome lines synced to disk.

use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::tool::Permission;

/// What an audit line records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    /// A write tool is about to run.
    Intent,
    /// A write tool finished, or failed.
    Outcome,
    /// A read tool ran.
    Call,
    /// One model turn.
    #[default]
    Turn,
    /// One step a skill runs in Rust.
    Step,
}

/// One audit event; `time` is added when it is written. Fields that do not apply stay `None`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    /// The engine session ID.
    pub session: u64,
    /// The agent chat's ID; the session ID changes with every restore and reset of its history.
    pub chat: Option<String>,
    /// The skill that ran the step, if any.
    pub skill: Option<String>,
    /// What this line records.
    pub kind: EventKind,
    /// The tool's name.
    pub tool: Option<String>,
    /// SHA-256 of the arguments' JSON, in hex.
    pub args_hash: Option<String>,
    /// The tool's permission.
    pub permission: Option<Permission>,
    /// Whether the user confirmed the write.
    pub confirmed: Option<bool>,
    /// Bytes of the tool's result before it was cut.
    pub result_bytes: Option<u64>,
    /// The error the step ended with.
    pub error: Option<String>,
    /// Prompt tokens of a model turn.
    pub prompt_tokens: Option<u32>,
    /// Prompt tokens reused from parked KV.
    pub cached_tokens: Option<u32>,
    /// Generated tokens.
    pub generated_tokens: Option<u32>,
    /// Time to first token in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Decode tokens per second.
    pub tok_s: Option<f64>,
    /// Tokens the step's call grammar forced (`Timings::grammar_forced`).
    pub grammar_forced: Option<u32>,
    /// The turn's `<think>` text, which never reaches the user.
    pub thinking: Option<String>,
    /// The skill's step: a Rust step's name, or the step a model turn belongs to.
    pub step: Option<String>,
    /// Wall time of the step in milliseconds.
    pub wall_ms: Option<f64>,
    /// What the step found, such as the facts that passed a skill's check.
    pub detail: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct Line<'a> {
    time: String,
    #[serde(flatten)]
    event: &'a AuditEvent,
}

/// The append-only audit log of one data folder.
#[derive(Clone)]
pub struct Audit {
    dir: PathBuf,
}

impl Audit {
    /// The log under `<data_dir>/agent/log/`; nothing is created until the first event.
    pub fn new(data_dir: &Path) -> Audit {
        Audit {
            dir: data_dir.join("agent").join("log"),
        }
    }

    /// The folder the log's files go in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The newest confirmed line `chat` logged for the write `tool` with `args_hash` after its last
    /// model step ended: `Outcome`, it ran; `Intent` with no outcome after it, the process died
    /// while it ran, so it may have. A process that died after a yes, before the next step was
    /// saved, leaves such a line, with the call still unanswered in the transcript.
    pub fn ran_after_last_step(
        &self,
        chat: &str,
        tool: &str,
        args_hash: &str,
    ) -> io::Result<Option<EventKind>> {
        self.after_last_step(chat, |e| match e.kind {
            kind @ (EventKind::Outcome | EventKind::Intent)
                if e.confirmed == Some(true)
                    && e.tool.as_deref() == Some(tool)
                    && e.args_hash.as_deref() == Some(args_hash) =>
            {
                Some(kind)
            }
            _ => None,
        })
    }

    /// The detail of the newest `step` line `chat` logged for the call with `args_hash` after its
    /// last model step ended: what Rust found for a call the chat was closed at, such as the
    /// research plan the user was asked about.
    pub fn step_after_last_step(
        &self,
        chat: &str,
        step: &str,
        args_hash: &str,
    ) -> io::Result<Option<serde_json::Value>> {
        self.after_last_step(chat, |e| {
            let hit = e.kind == EventKind::Step
                && e.step.as_deref() == Some(step)
                && e.args_hash.as_deref() == Some(args_hash);
            hit.then_some(e.detail).flatten()
        })
    }

    /// The newest of `chat`'s lines after its last model step that `pick` takes. The newest file
    /// is read first, from its end; a torn or foreign line is skipped.
    fn after_last_step<T>(
        &self,
        chat: &str,
        mut pick: impl FnMut(AuditEvent) -> Option<T>,
    ) -> io::Result<Option<T>> {
        let mut files: Vec<PathBuf> = match fs::read_dir(&self.dir) {
            Ok(entries) => entries
                .map(|e| e.map(|e| e.path()))
                .collect::<io::Result<_>>()?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        files.retain(|f| f.extension().is_some_and(|x| x == "jsonl"));
        // YYYY-MM-DD names sort by day.
        files.sort();
        for file in files.iter().rev() {
            let text = fs::read(file)?;
            for line in text.split(|&b| b == b'\n').rev() {
                let Ok(event) = serde_json::from_slice::<AuditEvent>(line) else {
                    continue;
                };
                if event.chat.as_deref() != Some(chat) {
                    continue;
                }
                if event.kind == EventKind::Turn {
                    return Ok(None);
                }
                if let Some(t) = pick(event) {
                    return Ok(Some(t));
                }
            }
        }
        Ok(None)
    }

    /// Writes the event as one line of today's file. An intent or outcome line is synced, so an
    /// intent is on disk before its write runs and the log tells whether a write happened when the
    /// process died in between; other lines ride on the next sync.
    pub fn append(&self, event: AuditEvent) -> io::Result<()> {
        let time = larch::bench::utc_now();
        let mut line = serde_json::to_vec(&Line {
            time: time.clone(),
            event: &event,
        })?;
        line.push(b'\n');
        // Absolute, so the folders to sync end at a real one: a relative path's last ancestor is
        // "", which cannot be opened.
        let dir = std::path::absolute(&self.dir)?;
        let path = dir.join(format!("{}.jsonl", &time[..10]));
        // A new file: its entry, and those of folders created for it, are synced after the line.
        #[cfg(unix)]
        let created =
            (!path.exists()).then(|| dir.ancestors().take_while(|d| !d.exists()).count() + 1);
        fs::create_dir_all(&dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        // A write cut short (a full disk) leaves part of a line; start a fresh one so this line
        // still parses. Readers skip the torn line.
        if file.metadata()?.len() > 0 {
            let mut last = [0];
            file.seek(SeekFrom::End(-1))?;
            file.read_exact(&mut last)?;
            if last != *b"\n" {
                line.insert(0, b'\n');
            }
        }
        // The whole line in one append, so lines from several threads do not interleave.
        file.write_all(&line)?;
        if matches!(event.kind, EventKind::Intent | EventKind::Outcome) {
            file.sync_data()?;
        }
        // As chat.rs does for transcripts: the entries must survive a crash too, once a day.
        #[cfg(unix)]
        for dir in dir.ancestors().take(created.unwrap_or(0)) {
            fs::File::open(dir)?.sync_all()?;
        }
        Ok(())
    }
}
