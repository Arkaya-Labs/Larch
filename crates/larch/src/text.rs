//! Stream adapter using PR12's UTF-8 and stop-string holdback without a second detokenizer.

use std::sync::Arc;

use larch_engine::{EngineEvent, Request, RequestHandle};
use larch_model::Detokenizer;

use crate::{EngineError, GenParams, api::Runtime, session::Turn};

pub use larch_engine::{Finish, Usage};

/// What the facade feeds after a think block its budget cut (`GenParams::thinking_budget`): the
/// sentence Qwen documents for this, then the block's end.
pub const THINKING_CLOSE: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";

/// The engine's timings plus the session's reuse and overflow information (10.1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Time to first token in milliseconds.
    pub ttft_ms: f64,
    /// Time per token after the first, in milliseconds.
    pub tpot_ms: f64,
    /// Decoding throughput after the first token.
    pub tok_s: f64,
    /// Prompt tokens reused from the session's parked sequence.
    pub cached_tokens: u32,
    /// Whole exchanges the overflow policy removed before submission.
    pub dropped_exchanges: u32,
    /// The thinking budget cut a think block, and the facade closed it with [`THINKING_CLOSE`].
    pub thinking_cut: bool,
    /// The prompt tokens of the request a budget stop continued with (`GenParams::thinking_budget`):
    /// the turn's prompt, the ids generated before the stop and, after a cut, the close; all but
    /// the last id and the close come from the parked KV. 0 when nothing continued.
    pub continuation_prompt_tokens: u32,
    /// Grammar picks forced because no allowed token was among the 64 candidates
    /// (`GenParams::grammar`), for benches to see how often it happens.
    pub grammar_forced: u32,
}

/// A generated token and its engine-supplied probability information.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    /// Sampled token ID.
    pub id: u32,
    /// Sampled token's log probability.
    pub logprob: f32,
    /// Top candidates returned by the engine, or the `score_ids` scores in their order.
    pub top: Vec<(u32, f32)>,
}

/// Text and terminal events with authoritative engine measurements.
#[derive(Clone, Debug, PartialEq)]
pub enum TextEvent {
    /// Safe decoded text; token metadata may span several held-back pieces.
    Text {
        /// UTF-8 text safe to deliver.
        delta: String,
        /// Tokens consumed since the previous text event, including held pieces.
        tokens: u32,
        /// Engine probabilities for those tokens, if requested.
        logprobs: Option<Vec<TokenLogprob>>,
    },
    /// Successful completion, including an acknowledged caller-side string stop.
    Done {
        /// Engine finish reason, or a locally matched StopSeq.
        finish: Finish,
        /// Actual engine token counts. On StopSeq they include tokens sampled after the matching
        /// one, before the engine acknowledged the stop. After a continuation
        /// (`GenParams::thinking_budget`), `prompt_tokens` and `cached_tokens` are the first
        /// request's, the turn's own prompt and reuse, and `completion_tokens` counts both
        /// requests; the continuation's prompt is `Timings::continuation_prompt_tokens`.
        usage: Usage,
        /// Actual engine timings and session reuse/overflow information. After a continuation,
        /// both requests as one: the first's time to first token, and the time per token over
        /// every token after the first.
        timings: Timings,
        /// The text inside the think blocks `hide_thinking` withheld, tags and the blank lines
        /// after a block left out, a budget cut's close sentence included; None when none was.
        thinking: Option<String>,
    },
    /// A piece of a withheld think block as it comes, for a stream whose request asked
    /// ([`GenParams::stream_thinking`]), cut at whole characters. The pieces are `Done::thinking`
    /// up to a budget cut's close sentence; the tokens count on the next text event.
    Thinking {
        /// UTF-8 text of the block, never part of the answer or history.
        delta: String,
    },
    /// Prefill progress, for a stream whose request asked ([`GenParams::progress`]): prompt
    /// tokens done of `total`, counting those a cache served. One per prefill chunk.
    Prefilled {
        /// Prompt tokens with KV so far.
        done: u32,
        /// The prompt's length.
        total: u32,
    },
    /// Cancellation or an engine failure.
    Error(EngineError),
}

/// A synchronous iterator; cancelling or dropping commits only text already delivered.
pub struct TextStream {
    pub(crate) handle: RequestHandle,
    runtime: Arc<Runtime>,
    turn: Option<Turn>,
    decoder: Detokenizer,
    delivered: String,
    tokens: u32,
    logprobs: Option<Vec<TokenLogprob>>,
    pending: Option<TextEvent>,
    terminal: bool,
    cancelled: bool,
    dropped: u32,
    hide_thinking: bool,
    /// Inside a `<think>` block, shown or withheld.
    thinking: bool,
    /// After a block, before its first visible piece.
    closed: bool,
    /// The withheld blocks' text, for `Done::thinking`.
    hidden: Option<Vec<u8>>,
    /// [`GenParams::stream_thinking`] with `hide_thinking`, and how many of `hidden`'s bytes went
    /// out as [`TextEvent::Thinking`].
    stream_thinking: bool,
    thought: usize,
    budget: Option<u32>,
    /// The request a budget cut continues: the first one with the caller's max_tokens.
    next: Option<Request>,
    /// The ids the first request generated, while `next` waits.
    ids: Vec<u32>,
    /// The first request's usage and timings, once continued.
    first: Option<(Usage, larch_engine::Timings)>,
    /// The first request's handle once continued, kept so its cancellers still cancel the turn.
    cut_from: Option<RequestHandle>,
    cut: bool,
    /// [`TextStream::stop_loops`]: the output's last characters, and whether a loop ended it.
    loop_tail: Option<String>,
    looped: bool,
    /// The text trimmed off `loop_tail` left a code block open.
    loop_fenced: bool,
}

/// The longest repeating unit, and the shortest repeated span, [`looping`] calls a loop.
const LOOP_UNIT: usize = 64;
const LOOP_SPAN: usize = 200;
const LOOP_COPIES: usize = 8;

/// A long loop ([`looping`]): the output's last `ECHO_SPAN` bytes, up to a line's end, written
/// `ECHO_COPIES` times before, within the last `ECHO_WINDOW` bytes.
const ECHO_SPAN: usize = 200;
const ECHO_COPIES: usize = 2;
const ECHO_WINDOW: usize = 16 << 10;

/// Whether `text` ends in a loop:
/// - one unit of 1 to 64 characters repeated over at least 200 characters and 8 copies: a small
///   model stuck emitting `^^^^`, a run of fences or two words in turn;
/// - or, outside a code block and at the end of a line, its last 200 bytes (a letter among them)
///   written word for word twice before: a model going round whole sentences or paragraphs
///   (Qwen3.5-2B, greedy, on a 1000-word essay, wrote the same three paragraphs over and over
///   until stopped at 14k characters; this ends it at 2.7k).
///
/// Rows of a table, a CSV or a log differ from each other somewhere before their line's end
/// (an ID, a count), so they never match, however long the part they share.
pub fn looping(text: &str) -> bool {
    looping_in(text, false)
}

/// [`looping`] on the last part of an output, which starts inside a code block when `fenced`.
fn looping_in(text: &str, fenced: bool) -> bool {
    let t = text.as_bytes();
    (1..=LOOP_UNIT).any(|p| {
        let n = p * LOOP_COPIES.max(LOOP_SPAN.div_ceil(p));
        n <= t.len() && {
            let tail = &t[t.len() - n..];
            tail.chunks(p).all(|c| c == &tail[..p])
        }
    }) || echoing(text, fenced)
}

/// Whether `text` holds an odd number of code fences: it ends inside a block, or leaves the one
/// it started in.
fn fences_odd(text: &str) -> bool {
    text.matches("```").count() % 2 == 1
}

/// The long loop of [`looping`]. Judged only at a line's end: a fragment shared by lines that end
/// differently (rows with one long field and their own IDs) is no loop. Never inside a code
/// block, where repeated blocks are often meant.
fn echoing(text: &str, fenced: bool) -> bool {
    if !text.ends_with('\n') || text.len() < ECHO_SPAN * (ECHO_COPIES + 1) {
        return false;
    }
    if fenced != fences_odd(text) {
        return false;
    }
    let from = text.floor_char_boundary(text.len() - ECHO_SPAN);
    let tail = &text[from..];
    tail.chars().any(char::is_alphabetic)
        && text[..from].matches(tail).nth(ECHO_COPIES - 1).is_some()
}

impl TextStream {
    /// Ends the answer at a loop ([`looping`]) as a stop sequence would (`Finish::StopSeq`), with
    /// the text up to there kept; [`TextStream::looped`] then says so. Off by default.
    pub fn stop_loops(&mut self, on: bool) {
        self.loop_tail = on.then(String::new);
    }

    /// The answer was ended at a loop ([`TextStream::stop_loops`]).
    pub fn looped(&self) -> bool {
        self.looped
    }

    pub(crate) fn new(
        runtime: Arc<Runtime>,
        handle: RequestHandle,
        p: &GenParams,
        turn: Option<Turn>,
        dropped: u32,
        next: Option<Request>,
        opens: bool,
    ) -> Self {
        Self {
            handle,
            runtime,
            turn,
            decoder: Detokenizer::new(p.stop.clone()),
            delivered: String::new(),
            tokens: 0,
            logprobs: (p.logprobs > 0 || !p.score_ids.is_empty()).then(Vec::new),
            pending: None,
            terminal: false,
            cancelled: false,
            dropped,
            hide_thinking: p.hide_thinking,
            thinking: opens,
            closed: false,
            hidden: (opens && p.hide_thinking).then(Vec::new),
            stream_thinking: p.hide_thinking && p.stream_thinking,
            thought: 0,
            budget: p.thinking_budget,
            next,
            ids: Vec::new(),
            first: None,
            cut_from: None,
            cut: false,
            loop_tail: None,
            looped: false,
            loop_fenced: false,
        }
    }

    /// Cancels this stream's request from another thread, while this one waits on it.
    pub fn canceller(&self) -> larch_engine::RequestCanceller {
        self.handle.canceller()
    }

    /// Returns the engine request ID.
    pub fn request_id(&self) -> u64 {
        self.handle.id
    }

    /// Inside a withheld `<think>` block; after the terminal event, the answer ended mid-thought,
    /// for example at `max_tokens`.
    pub fn in_thinking(&self) -> bool {
        self.thinking && self.hide_thinking
    }

    pub(crate) fn dropped_exchanges(&self) -> u32 {
        self.dropped
    }

    /// Cancels this turn and preserves the delivered partial answer as truncated history. Once all
    /// text is delivered and Done is pending, Done's row applies unless externally cancelled.
    pub fn cancel(&mut self) {
        if self.terminal || self.cancelled {
            return;
        }
        if self.pending.is_some() && !self.interrupted() {
            self.end(Ok(false), false);
            return;
        }
        self.handle.cancel();
        self.end(Ok(true), true);
        self.cancelled = true;
    }

    // The CLI can reject part of a yielded delta. Preserve only the UTF-8 prefix
    // actually accepted by its writer; public iterator delivery remains unchanged.
    pub(crate) fn cancel_after_output(&mut self, accepted: usize) {
        self.delivered
            .truncate(self.delivered.floor_char_boundary(accepted));
        self.pending = None; // not every piece reached the writer
        self.cancel();
    }

    fn end(&self, end: Result<bool, EngineError>, release: bool) {
        if let Some(turn) = &self.turn {
            turn.end(&self.runtime, &self.delivered, end, release);
        }
    }

    /// Tracks think blocks; whether `id` is withheld: `hide_thinking` and part of a block, its tags,
    /// a token between them, or a blank line right after it, which templates drop with the block
    /// when they render the answer again.
    // ponytail: tags are whole tokens, as in m2's and Qwen 3.5's vocabularies. Match text when a
    // model joins bench/models.toml whose tags split into several tokens.
    fn withheld(&mut self, id: u32) -> bool {
        let piece = self.runtime.tokenizer.decode_piece(id);
        let part = match piece {
            b"<think>" => {
                self.thinking = true;
                self.closed = false;
                if self.hide_thinking {
                    self.hidden.get_or_insert_default();
                }
                true
            }
            b"</think>" if self.thinking => {
                self.thinking = false;
                self.closed = true;
                true
            }
            _ if self.closed && piece.iter().all(|&b| b == b'\n') => true,
            _ => {
                self.closed = false;
                if self.thinking
                    && let Some(hidden) = &mut self.hidden
                {
                    hidden.extend_from_slice(piece);
                }
                self.thinking
            }
        };
        part && self.hide_thinking
    }

    /// The withheld text not yet streamed, up to its last whole character; None when there is none.
    fn thought(&mut self) -> Option<String> {
        let rest = &self.hidden.as_ref()?[self.thought..];
        let end = match std::str::from_utf8(rest) {
            Ok(_) => rest.len(),
            // An invalid sequence goes out as U+FFFD, as in `Done::thinking`; a split one waits.
            Err(e) => e.valid_up_to() + e.error_len().unwrap_or(0),
        };
        let delta = String::from_utf8_lossy(&rest[..end]).into_owned();
        self.thought += end;
        (end > 0).then_some(delta)
    }

    /// `id`'s safe text and whether it completed a string stop; None when withheld.
    fn feed(&mut self, id: u32) -> Option<(String, bool)> {
        // Its tokens are counted on the next text event; the stop matcher never sees them.
        (!self.withheld(id)).then(|| self.decoder.push(&self.runtime.tokenizer, id))
    }

    /// Continues a request its thinking budget ended, in the same session (`TextEvent::Done` parked
    /// its KV; a cancel would free it): the prompt, the generated ids (not their text, so the parked
    /// prefix matches to its last fed token) and, inside a block, [`THINKING_CLOSE`], with the rest
    /// of the caller's max_tokens. None when there is nothing to deliver yet.
    fn resume(
        &mut self,
        mut next: Request,
        close: Vec<u32>,
        usage: Usage,
        timings: larch_engine::Timings,
    ) -> Option<TextEvent> {
        self.cut = !close.is_empty();
        // A call the block began and the cut left open would go on after the close as the
        // answer's first words, its opener in the block: the parser saw text and the user saw the
        // call's tail. Its ids go, so the model writes the call after the close. With
        // `hide_thinking` they were withheld; with thinking shown the user already read the
        // opener as thinking text, which stays shown: the prompt and the grammar replay take the
        // cut ids either way.
        if self.cut
            && let Some(at) = self.open_call()
        {
            self.ids.truncate(at);
        }
        let mut delta = String::new();
        for &id in &close {
            let Some((piece, stopped)) = self.feed(id) else {
                continue;
            };
            delta.push_str(&piece);
            if stopped {
                // A shown block's stop can match in the close; the turn ends there, as at any stop.
                self.pending = Some(self.done(Finish::StopSeq, usage, timings));
                return Some(self.text(delta));
            }
        }
        self.first = Some((usage, timings));
        if let Some(g) = &mut next.grammar {
            // The output so far, close included: the pattern resumes where the cut left it.
            g.replay = self.ids.iter().chain(&close).copied().collect();
        }
        next.prompt.append(&mut self.ids);
        next.prompt.extend(close);
        let room = self.room(next.prompt.len());
        // `close` left room, and the budget was below max_tokens: both are at least 1.
        next.max_tokens = (next.max_tokens.saturating_sub(usage.completion_tokens)).min(room);
        let handle = match &mut self.turn {
            Some(turn) => turn.resubmit(&self.runtime, next),
            None => self.runtime.engine.submit(next),
        };
        match handle {
            Ok(handle) => self.cut_from = Some(std::mem::replace(&mut self.handle, handle)),
            Err(error) => return Some(self.terminal(TextEvent::Error(error))),
        }
        (!delta.is_empty()).then(|| self.text(delta))
    }

    /// Where the generated ids' last call opener starts when no closer follows it (`<tool_call>`
    /// and the other wrappers of the format that thinks).
    fn open_call(&self) -> Option<usize> {
        let tok = &self.runtime.tokenizer;
        let last = |s: &str| {
            let pat = tok.encode(s, false);
            (!pat.is_empty())
                .then(|| self.ids.windows(pat.len()).rposition(|w| w == pat))
                .flatten()
        };
        (crate::toolcall::Family::Qwen3.wrappers().iter())
            .filter_map(|(open, close)| {
                let at = last(open)?;
                (!last(close).is_some_and(|end| end > at)).then_some(at)
            })
            .max()
    }

    /// The close a budget cut continues with, [`THINKING_CLOSE`] inside a block and nothing after
    /// one; None when the continuation's prompt would leave no room in the context, so the turn
    /// ends at the cut as at any cap.
    fn close(&self, next: &Request) -> Option<Vec<u32>> {
        let close = if self.thinking {
            self.runtime.tokenizer.encode(THINKING_CLOSE, false)
        } else {
            Vec::new()
        };
        let len = next.prompt.len() + self.ids.len() + close.len();
        (self.room(len) > 0).then_some(close)
    }

    /// Context left after a prompt of `len` tokens.
    fn room(&self, len: usize) -> u32 {
        let len = u32::try_from(len).unwrap_or(u32::MAX);
        self.runtime.engine.deployment().ctx_max.saturating_sub(len)
    }

    fn text(&mut self, delta: String) -> TextEvent {
        self.delivered.push_str(&delta);
        TextEvent::Text {
            delta,
            tokens: std::mem::take(&mut self.tokens),
            logprobs: self.logprobs.as_mut().map(std::mem::take),
        }
    }

    fn done(&self, finish: Finish, usage: Usage, t: larch_engine::Timings) -> TextEvent {
        let (merged, t) = match self.first {
            Some(first) => merge(first, (usage, t)),
            None => (usage, t),
        };
        let continuation = self.first.map_or(0, |_| usage.prompt_tokens);
        let usage = merged;
        TextEvent::Done {
            finish,
            usage,
            timings: Timings {
                ttft_ms: t.ttft_ms,
                tpot_ms: t.tpot_ms,
                tok_s: t.tok_s,
                cached_tokens: usage.cached_tokens,
                dropped_exchanges: self.dropped,
                thinking_cut: self.cut,
                continuation_prompt_tokens: continuation,
                grammar_forced: t.grammar_forced,
            },
            thinking: self
                .hidden
                .as_ref()
                .map(|h| String::from_utf8_lossy(h).into_owned()),
        }
    }

    fn terminal(&mut self, event: TextEvent) -> TextEvent {
        match &event {
            TextEvent::Done { .. } => self.end(Ok(false), false),
            TextEvent::Error(EngineError::Cancelled) => self.end(Ok(true), true),
            TextEvent::Error(error) => self.end(Err(*error), true),
            TextEvent::Text { .. } | TextEvent::Thinking { .. } | TextEvent::Prefilled { .. } => {
                unreachable!("terminal event")
            }
        }
        self.terminal = true;
        event
    }

    // reset() and restore() cancel the active request, so a stale turn is always cancelled.
    fn interrupted(&self) -> bool {
        self.cancelled
            || self.handle.is_cancelled()
            || self
                .cut_from
                .as_ref()
                .is_some_and(RequestHandle::is_cancelled)
    }

    fn next_event(&mut self) -> Option<TextEvent> {
        if self.terminal {
            return None;
        }
        if self.interrupted() {
            self.handle.cancel();
            return Some(self.terminal(TextEvent::Error(EngineError::Cancelled)));
        }
        // Explicit cancel/drop preserves a completed turn; reset invalidates even buffered Done.
        if let Some(event) = self.pending.take() {
            return Some(self.terminal(event));
        }
        loop {
            let event =
                self.handle
                    .events
                    .recv()
                    .unwrap_or(EngineEvent::Error(EngineError::NotReady(
                        larch_records::HealthState::Failed,
                    )));
            // A reset may have happened during recv. No stale text or completion reaches history.
            if self.interrupted() {
                self.handle.cancel();
                return Some(self.terminal(TextEvent::Error(EngineError::Cancelled)));
            }
            match event {
                EngineEvent::Prefilled { done, total } => {
                    return Some(TextEvent::Prefilled { done, total });
                }
                EngineEvent::Token { id, logprob, top } => {
                    self.tokens += 1;
                    if let Some(probs) = &mut self.logprobs {
                        probs.push(TokenLogprob { id, logprob, top });
                    }
                    if self.next.is_some() {
                        self.ids.push(id);
                    }
                    let Some((delta, mut stopped)) = self.feed(id) else {
                        if self.stream_thinking
                            && let Some(delta) = self.thought()
                        {
                            return Some(TextEvent::Thinking { delta });
                        }
                        continue;
                    };
                    if !stopped && let Some(tail) = &mut self.loop_tail {
                        tail.push_str(&delta);
                        // Trimmed now and then, to the window the long loop looks back over, at a
                        // line's start so no fence is split; the fences trimmed off are counted.
                        if tail.len() > 2 * ECHO_WINDOW {
                            let cut = tail.ceil_char_boundary(tail.len() - ECHO_WINDOW);
                            let cut = tail[cut..].find('\n').map_or(cut, |i| cut + i + 1);
                            self.loop_fenced ^= fences_odd(&tail[..cut]);
                            tail.drain(..cut);
                        }
                        self.looped = looping_in(tail, self.loop_fenced);
                        stopped = self.looped;
                    }
                    if stopped {
                        self.pending = match self.runtime.engine.finish_stop(&self.handle) {
                            Ok((usage, timings, _)) => {
                                Some(self.done(Finish::StopSeq, usage, timings))
                            }
                            Err(error) => return Some(self.terminal(TextEvent::Error(error))),
                        };
                        return Some(self.text(delta));
                    }
                    if !delta.is_empty() {
                        return Some(self.text(delta));
                    }
                }
                EngineEvent::Done {
                    finish,
                    usage,
                    timings,
                    ..
                } => {
                    // Only the budget's own cap continues; a full context or the caller's cap ends.
                    if finish == Finish::Length
                        && self.budget == Some(usage.completion_tokens)
                        && let Some(next) = self.next.take()
                        && let Some(close) = self.close(&next)
                    {
                        match self.resume(next, close, usage, timings) {
                            Some(event) => return Some(event),
                            None => continue,
                        }
                    }
                    let done = self.done(finish, usage, timings);
                    let delta = self.decoder.finish();
                    if !delta.is_empty() {
                        self.pending = Some(done);
                        return Some(self.text(delta));
                    }
                    // EOS/control tokens still carry usage and logprobs even with no display text.
                    if self.tokens > 0 {
                        self.pending = Some(done);
                        return Some(self.text(String::new()));
                    }
                    return Some(self.terminal(done));
                }
                EngineEvent::Error(error) => return Some(self.terminal(TextEvent::Error(error))),
            }
        }
    }
}

/// Two requests as the turn they make: the first's prompt and reuse, both requests' generated
/// tokens, and their timings as one request spanning both would report them.
fn merge(
    (a, ta): (Usage, larch_engine::Timings),
    (b, tb): (Usage, larch_engine::Timings),
) -> (Usage, larch_engine::Timings) {
    let usage = Usage {
        completion_tokens: a.completion_tokens.saturating_add(b.completion_tokens),
        ..a
    };
    let decode = |t: larch_engine::Timings, n: u32| t.tpot_ms * f64::from(n.saturating_sub(1));
    // Everything after the first token: the first decode, the continuation's wait and prefill, and
    // its decode.
    let ms = decode(ta, a.completion_tokens) + tb.ttft_ms + decode(tb, b.completion_tokens);
    let after = f64::from(usage.completion_tokens.saturating_sub(1));
    let t = larch_engine::Timings {
        ttft_ms: ta.ttft_ms,
        tpot_ms: if after > 0.0 { ms / after } else { 0.0 },
        tok_s: if ms > 0.0 { after / (ms / 1e3) } else { 0.0 },
        grammar_forced: ta.grammar_forced.saturating_add(tb.grammar_forced),
    };
    (usage, t)
}

impl Iterator for TextStream {
    type Item = TextEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_event()
    }
}

impl Drop for TextStream {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Larch, SessionOptions};
    use larch_engine::{EngineOptions, start_with};
    use larch_kernels::CPU;
    use larch_model::{ChatTemplate, FakeModel, Tokenizer, gguf::Gguf, tiny};

    #[test]
    fn externally_cancelled_pending_done_is_truncated_on_cancel_and_drop() {
        let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        let tokenizer = Tokenizer::from_gguf(&g).unwrap();
        let template = ChatTemplate::new("{{ messages[-1].content }}", &tokenizer).unwrap();
        let engine = start_with(
            EngineOptions::new("fake".into(), CPU),
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(512)))),
        )
        .unwrap();
        let v = Larch::from_engine(engine.clone(), tokenizer, template);
        for explicit in [false, true] {
            let mut session = v.session(SessionOptions::default());
            let id = session.id();
            let mut stream = session
                .send(
                    "a",
                    GenParams {
                        max_tokens: 1,
                        temperature: Some(0.0),
                        stop: vec!["bc".into()],
                        ..GenParams::default()
                    },
                )
                .unwrap();
            // The unmatched stop prefix flushes the text while leaving Length Done buffered.
            assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "b"));
            assert!(session.tokens_in_kv() > 0);
            stream.handle.canceller().cancel();
            if explicit {
                stream.cancel();
            }
            drop(stream);
            let history = session.history();
            assert_eq!(history.len(), 2);
            assert_eq!(history[1].content, "b");
            assert!(history[1].truncated, "explicit cancel: {explicit}");
            assert_eq!(engine.parked_tokens(id), 0);
            assert_eq!(session.tokens_in_kv(), 0);
        }
    }

    /// FakeModel answers BOS with `<think>`, `<x>`; cut there by a budget of 2, the close's last
    /// `\n\n` is answered with `<a0>`, `<a1>`, EOS.
    fn thinker() -> Larch {
        thinker_with("{{ messages[-1].content }}{{ bos_token }}")
    }

    /// `thinker` on `template`.
    fn thinker_with(template: &str) -> Larch {
        use larch_model::gguf::{Value, write};
        let tails = [("<think>", 4), ("<x>", 1), ("</think>", 4), ("\n\n", 4)];
        let tails = tails.into_iter().chain([("<a0>", 1), ("<a1>", 1)]);
        let mut tokens: Vec<_> = (0..256)
            .map(|b| Value::Str(format!("<0x{b:02X}>")))
            .collect();
        let mut types = vec![Value::I32(6); 256];
        for (text, kind) in [("<cue>", 3)]
            .into_iter()
            .chain(tails)
            .chain([("<end>", 3)])
        {
            tokens.push(Value::Str(text.into()));
            types.push(Value::I32(kind));
        }
        let vocab = tokens.len() as u32;
        let kv = [
            ("tokenizer.ggml.model", Value::Str("llama".into())),
            ("tokenizer.ggml.tokens", Value::Arr(tokens)),
            ("tokenizer.ggml.token_type", Value::Arr(types)),
            ("tokenizer.ggml.bos_token_id", Value::U32(256)),
            ("tokenizer.ggml.eos_token_id", Value::U32(vocab - 1)),
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
            ("tokenizer.ggml.add_space_prefix", Value::Bool(false)),
        ];
        let none: [(&str, Vec<u64>, u32, Vec<u8>); 0] = [];
        let g = Gguf::parse(&write(&kv, &none)).unwrap();
        let tokenizer = Tokenizer::from_gguf(&g).unwrap();
        let template = ChatTemplate::new(template, &tokenizer).unwrap();
        let engine = start_with(
            EngineOptions::new("fake".into(), CPU),
            Box::new(move |_, _, _| Ok(Box::new(FakeModel::new(vocab)))),
        )
        .unwrap();
        Larch::from_engine(engine, tokenizer, template)
    }

    fn budget() -> GenParams {
        GenParams {
            temperature: Some(0.0),
            thinking_budget: Some(2),
            ..GenParams::default()
        }
    }

    #[test]
    fn a_prompt_that_opens_the_think_block_starts_the_reply_inside_it() {
        // Qwen 3.5's generation prompt ends with `<think>\n` (#369): the reply's thinking showed as
        // its answer, `</think>` and all.
        let v = thinker_with("{{ messages[-1].content }}<think>");
        let mut session = v.session(SessionOptions::default());
        let p = GenParams {
            temperature: Some(0.0),
            ..GenParams::default()
        };
        let (mut text, mut thinking) = (String::new(), None);
        for event in session.send("q", p).unwrap() {
            match event {
                TextEvent::Text { delta, .. } => text.push_str(&delta),
                TextEvent::Done { thinking: t, .. } => thinking = t,
                e => panic!("{e:?}"),
            }
        }
        assert_eq!(
            (text.as_str(), thinking.as_deref()),
            ("<a0><a1>", Some("<x>"))
        );
        // The newline after the tag, as Qwen 3.5 writes it; a closed block opens nothing.
        let opens = |text: &str| {
            v.runtime
                .opens_thinking(&v.runtime.tokenizer.encode(text, false))
        };
        assert!(opens("q<think>\n"));
        assert!(!opens("q<think>\n\n</think>\n\n"));
        assert!(!opens("q"));
    }

    #[test]
    fn a_canceller_taken_before_a_budget_cut_cancels_the_continuation() {
        let v = thinker();
        let mut session = v.session(SessionOptions::default());
        let mut stream = session.send("q", budget()).unwrap();
        // As `larch chat` takes it, before the first event.
        let canceller = stream.handle.canceller();
        assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "<a0>"));
        canceller.cancel();
        assert_eq!(
            stream.next(),
            Some(TextEvent::Error(EngineError::Cancelled))
        );
    }

    #[test]
    fn an_earlier_canceller_after_a_cut_truncates_a_pending_done() {
        let v = thinker();
        let mut session = v.session(SessionOptions::default());
        let mut stream = session.send("q", budget()).unwrap();
        let canceller = stream.handle.canceller();
        // `<a0>`, `<a1>`, then the empty text that carries EOS and leaves Done pending.
        for delta in ["<a0>", "<a1>", ""] {
            assert!(matches!(stream.next(), Some(TextEvent::Text { delta: d, .. }) if d == delta));
        }
        canceller.cancel();
        drop(stream);
        // As without a cut (`externally_cancelled_pending_done_is_truncated_on_cancel_and_drop`).
        let history = session.history();
        assert_eq!(history[1].content, "<a0><a1>");
        assert!(history[1].truncated);
        assert_eq!(session.tokens_in_kv(), 0);
    }
}

#[cfg(test)]
mod loop_tests {
    use super::{ECHO_SPAN, echoing, looping, looping_in};

    /// The three loops the image eval found end the answer; a table, a log and a rule of dashes
    /// shorter than the span do not.
    #[test]
    fn loops_are_one_short_unit_repeated() {
        assert!(looping(&format!("error here\n{}", "^".repeat(200))));
        assert!(looping(&format!("text\n{}", "```\n".repeat(60))));
        assert!(looping(
            &"\u{201c}command,\u{201d} \u{201c}option,\u{201d} ".repeat(20)
        ));
        assert!(!looping(&"^".repeat(199)));
        let table: String = (0..80)
            .map(|i| {
                format!(
                    "| SO-{:05} | 2026-10-{:02} | ${}.13 |\n",
                    10400 + i,
                    i % 28 + 1,
                    100 + i
                )
            })
            .collect();
        assert!(!looping(&table));
        assert!(!looping(&format!("{}\nok", "-".repeat(150))));
    }

    /// Paragraphs gone round a third time end the answer at the end of a line of the third copy;
    /// two copies (a quoted block written again) do not, nor a third copy cut mid-line, nor an
    /// essay whose paragraphs share phrases. A paragraph the user asked for five times ends after
    /// its third too: the stream cannot tell that loop from a model's.
    #[test]
    fn a_paragraph_written_a_third_time_is_a_loop() {
        let a = "As we look to the future, Mount Fuji continues to inspire generations of \
                 Japanese and international visitors alike. Its legacy extends beyond its \
                 physical presence, serving as a testament to the enduring power of nature.\n\n";
        let b = "Mount Fuji stands as a testament to the enduring power of nature and the human \
                 connection to it. Its symmetrical, snow-capped silhouette has captivated the \
                 hearts of people for over 1,000 years.\n\n";
        let twice = format!("# Fuji\n\n{a}{b}{a}{b}");
        assert!(!looping(&twice));
        assert!(!looping(&format!("{twice}{}", &a[..150])));
        assert!(a.len() > ECHO_SPAN + 1);
        assert!(looping(&format!("{twice}{}", &a[..a.len() - 1])));
        let essay: String = (0..40)
            .map(|i| {
                format!(
                    "Section {i}. The mountain's cultural significance has led to the \
                     development of numerous events, and in year {} more visitors came.\n\n",
                    1900 + i
                )
            })
            .collect();
        assert!(!looping(&essay));
        assert!(looping(&a.repeat(3)) && !looping(&a.repeat(2)));
        // Long runs of symbols repeated whole are the short loop's, not this one's.
        assert!(!echoing(
            &format!("{}\n", "=+".repeat(ECHO_SPAN * 2)),
            false
        ));
    }

    /// Rows that share a long field but end in their own IDs are no loop, at any point in a row
    /// (review of #514: a CSV was cut inside its third row, before the ID).
    #[test]
    fn rows_that_share_a_long_field_are_no_loop() {
        let desc = "This order was placed through the regional partner portal and requires \
                    manual verification of the shipping address, the billing contact and the \
                    customs paperwork before it can be released to the warehouse for picking \
                    and packing. It waits for the night shift.";
        assert!(desc.len() > ECHO_SPAN + 50);
        let csv: String = (0..20)
            .map(|i| format!("\"{desc}\",SO-{}\n", 10400 + i))
            .collect();
        for end in (0..=csv.len()).filter(|&e| csv.is_char_boundary(e)) {
            assert!(!looping(&csv[..end]), "cut at {end}");
        }
        let log: String = (0..30)
            .map(|i| {
                format!(
                    "2026-11-13 10:{i:02} WARN worker pool saturated; request queued behind the \
                     long-running export job, which holds the shared lock on the reports table; \
                     retrying with backoff in 30 seconds attempt={i}\n"
                )
            })
            .collect();
        assert!(!looping(&log));
    }

    /// Blocks repeated inside a code fence are left alone, also when the fence opened before the
    /// part the stream still holds; the same blocks outside one are a loop.
    #[test]
    fn repeated_blocks_in_code_are_no_loop() {
        let card = "<div class=\"card\">\n  <h2>Title</h2>\n  <p>Lorem ipsum dolor sit amet, \
                    consectetur adipiscing elit, sed do eiusmod tempor incididunt.</p>\n  \
                    <button class=\"primary\">Read more</button>\n</div>\n";
        let cards = card.repeat(8);
        assert!(!looping(&format!("```html\n{cards}")));
        assert!(!looping_in(&cards, true));
        assert!(looping(&cards));
        assert!(looping(&format!("```html\n<p>x</p>\n```\n{cards}")));
    }
}
