//! `recall`, `remember` and `forget` over the data folder's memory.

use serde_json::{Value, json};

use crate::config;
use crate::memory::{self, Recall, Source};
use crate::tool::{Builtin, Permission, Tool, ToolCtx, ToolError, str_arg};

pub(crate) fn tools() -> Vec<Box<dyn Tool>> {
    vec![
        Builtin::boxed(
            "recall",
            "Search memory and daily notes; source transcript searches past chats.",
            Permission::Read,
            1000,
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"enum": ["fact", "daily", "transcript"]}
                },
                "required": ["query"]
            }),
            recall,
        ),
        Builtin::write(
            "remember",
            "Save one fact about the user or their work, with why it matters, to memory.",
            200,
            // A fact with its why.
            1024,
            json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "type": {"enum": memory::TYPES}
                },
                "required": ["text", "type"]
            }),
            |args, ctx| {
                let (text, kind) = (str_arg(args, "text")?, str_arg(args, "type")?);
                memory::check_on(&ctx.data_dir)?;
                memory::check_fact(text, kind)?;
                // A fact the system prompt shows already: the saving turn re-saves those.
                if let Some(slug) = memory::known(&ctx.data_dir, text, kind)? {
                    return Err(memory::MemoryError::Known(slug).into());
                }
                Ok(args.clone())
            },
            remember,
        ),
        Builtin::write(
            "forget",
            "Delete a fact from memory by the slug MEMORY.md shows in brackets.",
            100,
            crate::tool::ARGS_TOKENS,
            json!({"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}),
            |args, _| {
                memory::check_slug(str_arg(args, "slug")?)?;
                Ok(args.clone())
            },
            forget,
        ),
    ]
}

/// Facts and daily notes, or the one source the model names: a daily note's keys find its lines
/// too. Past chats only when named, so a forgotten fact does not come back from a transcript.
// ponytail: a daily note still logs each `remember` call with its text, so a forgotten fact can
// still come back from one; log the slug alone there if that matters
fn recall(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    // A chat that opened with memory on may still offer it until its next turn.
    memory::check_on(&ctx.data_dir)?;
    let sources: &[Source] = match args.get("source").and_then(Value::as_str) {
        Some("fact") => &[Source::Fact],
        Some("daily") => &[Source::Daily, Source::Key],
        Some("transcript") => &[Source::Transcript],
        _ => &memory::AUTO,
    };
    // Read per call, a small file, so ToolCtx keeps no config: an edit counts from the next one.
    let hits = config::load(&ctx.data_dir)
        .map_err(|e| ToolError::Invalid(e.to_string()))?
        .recall_hits;
    let how = Recall {
        sources,
        hits,
        today: memory::today(),
    };
    let hits = memory::recall_with(&ctx.data_dir, str_arg(args, "query")?, how)?;
    Ok(json!(hits).to_string())
}

/// What comes before the old text in `remember`'s result when it updated a fact. The model gets
/// the old text; the call's summary, which the daily note keeps, stops here (`summary_of`).
pub(crate) const REPLACED: &str = "; it replaced: ";
/// What comes before the save's receipt, as JSON, at the end of `remember`'s result: the turn
/// takes it off (`split_receipt`) before the model, the summary or the app see the result.
const RECEIPT: char = '\u{1e}';

/// `remember`'s result without the text it replaced: what its summary and the daily note keep,
/// so a corrected or forgotten value is not written there.
pub(crate) fn summary_of(result: &str) -> &str {
    let result = split_receipt(result).0;
    result.split_once(REPLACED).map_or(result, |(slug, _)| slug)
}

/// `remember`'s result as the model reads it, and the save's receipt.
pub(crate) fn split_receipt(result: &str) -> (&str, Option<memory::Receipt>) {
    match result.split_once(RECEIPT) {
        Some((text, receipt)) => (text, serde_json::from_str(receipt).ok()),
        None => (result, None),
    }
}

fn remember(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let (text, kind) = (str_arg(args, "text")?, str_arg(args, "type")?);
    memory::check_on(&ctx.data_dir)?;
    let saved = memory::remember(&ctx.data_dir, text, kind, &ctx.source)?;
    let said = match &saved.old {
        Some(old) => format!("{}{REPLACED}{old}", saved.slug),
        None => saved.slug.clone(),
    };
    let receipt = serde_json::to_string(&saved.receipt(text)).unwrap_or_default();
    Ok(format!("{said}{RECEIPT}{receipt}"))
}

fn forget(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let slug = str_arg(args, "slug")?;
    memory::check_on(&ctx.data_dir)?;
    memory::forget(&ctx.data_dir, slug)?;
    Ok(format!("forgot '{slug}'"))
}
