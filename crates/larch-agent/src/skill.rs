//! Compiled skills: Rust fixes the steps, the model fills them in.

use std::{fmt, io};

use larch::{EngineError, Larch};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    audit::AuditEvent,
    tool::{ToolCtx, ToolError},
    workers::Workers,
};

/// A skill in the fields a `SKILL.md` front matter has, so compiled and text skills can be
/// listed side by side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillSpec {
    /// The name the CLI and the audit log use.
    pub name: String,
    /// One sentence on what it does.
    pub description: String,
    /// The registry tools its model turns may call.
    pub tools: Vec<String>,
}

/// What the user gives a skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillInput {
    /// The folder to work on; it must resolve inside an allowed folder.
    pub folder: String,
    /// The user's question.
    pub question: String,
    /// Words the documents may use for the answer, which the model writes beside the question:
    /// the ranker reads them with the question's own, so a question in the user's words ("income
    /// limits for opening a Roth IRA") still finds a guide's ("modified AGI ... phase-out").
    /// Empty when there are none.
    pub search: String,
}

/// What a skill gives back.
#[derive(Clone, Debug, PartialEq)]
pub struct SkillRun {
    /// The answer, in Markdown.
    pub markdown: String,
    /// The answer alone, without the notes under it in `markdown`: what the bench scores.
    pub answer: String,
    /// A JSON copy of the facts the answer rests on, as the audit log has it.
    pub facts: Value,
    /// The audit events the run wrote, in order.
    pub steps: Vec<AuditEvent>,
}

/// A skill's result as data, for a caller that renders it itself (the MCP server's tools).
// ponytail: no Answer yet; the plan's Answer variant comes with the first caller that needs
// web-answer's result as data
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedResult {
    /// Research's sectioned report.
    Report {
        /// The sections, in the outline's order.
        sections: Vec<crate::skills::research::Section>,
        /// The facts the sections cite whose quotes were found again, by ledger number.
        sources: Vec<crate::skills::research::Evidence>,
        /// The facts cited whose quotes were not found again, dropped from the text.
        unverified: Vec<crate::skills::research::Unverified>,
        /// Why the research stopped: the planner, a budget (named), or a round with no new fact.
        stopped: String,
        /// The pages read.
        #[serde(default)]
        pages: usize,
        /// Sentences with a number or a date and no source that their writer's retry fixed (R6).
        #[serde(default)]
        fixed: usize,
        /// Such sentences Rust removed.
        #[serde(default)]
        removed: usize,
        /// Pages fetched that wrote no spelling of a subject, never read (the subject lock).
        #[serde(default)]
        off_subject: usize,
        /// Pages read that gave no fact.
        #[serde(default)]
        thin: usize,
        /// What the report's last line says besides: a subject no page named, a lock that fell
        /// back.
        #[serde(default)]
        notes: Vec<String>,
    },
    /// A coding task's change, for a human to approve before it is merged.
    Patch(crate::code::task::Packet),
}

/// Why a skill stopped.
#[derive(Debug)]
pub enum SkillError {
    /// The folder was denied, or a file could not be read.
    Tool(ToolError),
    /// The in-memory search index failed.
    Index(rusqlite::Error),
    /// A model turn failed.
    Engine(EngineError),
    /// The audit log or the daily notes could not be written.
    Log(io::Error),
}

impl fmt::Display for SkillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tool(e) => write!(f, "{e}"),
            Self::Index(e) => write!(f, "search index: {e}"),
            Self::Engine(e) => write!(f, "{e}"),
            Self::Log(e) => write!(f, "audit log or notes: {e}"),
        }
    }
}

impl std::error::Error for SkillError {}

impl From<ToolError> for SkillError {
    fn from(e: ToolError) -> Self {
        Self::Tool(e)
    }
}

impl From<rusqlite::Error> for SkillError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Index(e)
    }
}

/// A skill whose steps are Rust.
pub trait Skill {
    /// Its name, description and tools.
    fn spec() -> SkillSpec;

    /// Runs it on `larch`'s model, inside `ctx`'s folders, with at most `workers` sessions at once.
    // ponytail: takes the engine, the folders and the worker cap, not the `Agent`, which keeps no
    // engine handle (`Larch` is not `Clone`); take the agent once it carries one
    fn run(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        input: &SkillInput,
    ) -> Result<SkillRun, SkillError>;
}
