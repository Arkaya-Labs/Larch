//! The one way this feature sends to the model: a seat, the turn's Stop flag, and a trace line.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use larch::{EngineError, GenParams, Larch, Prompt, Session, TextEvent};
use serde_json::Value;

use crate::{
    audit::{Audit, AuditEvent},
    runtime::inference::SeatAdmission,
    skill::SkillError,
    skills::ask_docs::{self, ms},
    workers::{self, Answer},
};

/// The skill name the trace lines carry, beside code mode's "code".
pub const SKILL: &str = "doc";

/// Where a request goes.
pub enum To<'a> {
    /// A user message, as the session's next turn.
    User(&'a mut Session, &'a str),
    /// The results of the session's last answer's calls, as `(call id, result)`.
    Results(&'a mut Session, Vec<(String, String)>),
    /// A one-shot prompt on the engine, with no session.
    Once(Prompt),
}

/// A sent request's answer, with what its trace line counted.
#[derive(Clone, Debug)]
pub struct Sent {
    /// The answer, with the engine's prompt tokens and timings.
    pub answer: Answer,
    /// Tokens of its hidden think block.
    pub thinking_tokens: usize,
    /// Milliseconds from the send to the first text the user saw; `None` when none showed.
    pub first_text_ms: Option<f64>,
}

/// Sends `to` with `p` on `larch`'s engine, hands `live` each piece of visible text, and appends
/// one trace line for the step `step` to `audit`, `detail` (ranges, the first 200 characters,
/// file names) with the thinking tokens and the time to the first text the user saw added:
/// `live` says whether a piece showed, as one that may be a call's start waits. `stop` is the
/// turn's own: once it is set the request is cancelled and this is `EngineError::Cancelled`.
#[allow(clippy::too_many_arguments)]
pub fn send(
    larch: &Larch,
    to: To<'_>,
    p: GenParams,
    step: &str,
    mut detail: Value,
    audit: &Audit,
    stop: &AtomicBool,
    live: &mut dyn FnMut(&str) -> bool,
) -> Result<Sent, SkillError> {
    let start = Instant::now();
    let (seats, class) = (SeatAdmission::of(larch), p.class);
    let (session, stream) = match to {
        // Nothing is sent after Stop: the request would keep its session busy past the turn.
        _ if stop.load(Ordering::Relaxed) => (0, Err(EngineError::Cancelled)),
        To::User(s, text) => (s.id(), seats.seat(class, stop, || s.send(text, p))),
        To::Results(s, results) => (
            s.id(),
            seats.seat(class, stop, || s.send_tool_results(results, p)),
        ),
        To::Once(prompt) => (0, seats.seat(class, stop, || larch.generate(prompt, p))),
    };
    let mut first = None;
    let read = stream.and_then(|stream| {
        workers::read(workers::Reading::new(session, stream), stop, &mut |e| {
            if let TextEvent::Text { delta, .. } = e
                && !delta.is_empty()
                && live(delta.as_str())
            {
                first.get_or_insert_with(|| ms(start));
            }
        })
    });
    let thinking_tokens = read.as_ref().map_or(0, |a| ask_docs::thought(larch, a));
    let event = match &read {
        Ok(a) => {
            detail["thinking_tokens"] = thinking_tokens.into();
            detail["first_text_ms"] = first.into();
            AuditEvent {
                wall_ms: Some(ms(start)),
                detail: Some(detail),
                ..ask_docs::turn(step, a, None)
            }
        }
        Err(e) => AuditEvent {
            detail: Some(detail),
            ..ask_docs::failed(step, session, start, e)
        },
    };
    audit
        .append(AuditEvent {
            skill: Some(SKILL.into()),
            ..event
        })
        .map_err(SkillError::Log)?;
    let answer = read.map_err(SkillError::Engine)?;
    Ok(Sent {
        answer,
        thinking_tokens,
        first_text_ms: first,
    })
}
