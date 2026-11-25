//! Whole-history conversations with explicit IDs and guarded terminal updates.

use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicU64, Ordering},
};

use larch_engine::{Request, RequestHandle};
use larch_model::ToolMessage;
use larch_model::projector::{ImagePixels, ProjectorConfig};
use larch_model::vision::Image;
use serde::{Deserialize, Serialize};

pub use larch_model::ToolCall;

use crate::{
    EngineError, GenParams, Prompt, TextStream,
    api::{Runtime, invalid, tool_messages},
    text::THINKING_CLOSE,
};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

fn fresh_id() -> u64 {
    NEXT_SESSION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("session IDs exhausted")
}

/// The supported conversation roles, persisted with stable lowercase names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions retained across reset and overflow.
    System,
    /// A submitted prompt.
    #[default]
    User,
    /// A delivered response.
    Assistant,
    /// A tool's result, answering one call of the assistant message before it.
    Tool,
}

impl Role {
    /// Returns the role name accepted by the model's chat template.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One delivered conversation message, including whether an answer was interrupted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// System, user, assistant, or tool.
    pub role: Role,
    /// Text delivered to the caller.
    pub content: String,
    /// True only for a cancelled or dropped assistant response.
    pub truncated: bool,
    /// An assistant message's tool calls, as the caller parsed them; empty is none.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "calls"
    )]
    pub tool_calls: Vec<ToolCall>,
    /// The call a tool message answers; empty is none (templates without call IDs).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
    /// A user message's images, by the sha256 of their bytes (64 lowercase hex digits), at most
    /// [`MAX_MESSAGE_IMAGES`]; the bytes live beside the transcript, never in it. Empty is none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// The most images one message carries.
pub const MAX_MESSAGE_IMAGES: usize = 8;

/// Whether `name` is an image reference: 64 lowercase hex digits, so it names a file in the chat's
/// image folder and nothing outside it.
pub fn image_ref(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// ponytail: larch-model's ToolCall derives only Deserialize; this writes the same three fields.
// Delete it when a3 derives Serialize there.
fn calls<S: serde::Serializer>(calls: &[ToolCall], s: S) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Call<'a> {
        name: &'a str,
        arguments: &'a serde_json::Value,
        id: &'a str,
    }
    s.collect_seq(calls.iter().map(|c| Call {
        name: &c.name,
        arguments: &c.arguments,
        id: &c.id,
    }))
}

/// The tool message that answers the call `id`.
fn tool_message(id: String, content: String) -> ChatMessage {
    ChatMessage {
        tool_call_id: id,
        ..ChatMessage::new(Role::Tool, content, false)
    }
}

impl ChatMessage {
    fn new(role: Role, content: String, truncated: bool) -> Self {
        Self {
            role,
            content,
            truncated,
            ..Self::default()
        }
    }

    /// Only an assistant message is truncated or calls tools, never both; only a tool names a call;
    /// no two calls share an ID, so each result names one call.
    pub(crate) fn well_formed(&self) -> bool {
        // A reference is checked where it names a file (`image_ref`): a bad one in a hand-edited
        // transcript is skipped with a notice, not a refused chat.
        (self.images.is_empty()
            || self.role == Role::User && self.images.len() <= MAX_MESSAGE_IMAGES)
            && (!self.truncated || self.role == Role::Assistant && self.tool_calls.is_empty())
            && (self.tool_calls.is_empty() || self.role == Role::Assistant)
            && (self.tool_call_id.is_empty() || self.role == Role::Tool)
            && distinct_ids(&self.tool_calls)
    }
}

/// No two calls share an ID; one call may have none (templates without call IDs).
fn distinct_ids(calls: &[ToolCall]) -> bool {
    calls
        .iter()
        .enumerate()
        .all(|(i, c)| calls[..i].iter().all(|d| d.id != c.id))
}

/// How a session handles a prompt that no longer fits its context.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Overflow {
    /// Remove whole oldest exchanges (a tool exchange with its calls, results and answers as one),
    /// preserving the system message and the current exchange.
    #[default]
    DropOldestExchanges,
    /// Reject the turn and preserve the previous history.
    Fail,
}

/// Conversation policy; KV retention is enabled by default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionOptions {
    /// Optional system message, retained across reset and overflow.
    pub system: Option<String>,
    /// Ask the engine to park KV at natural completion. False also forbids automatic global
    /// prefix retention/adoption, even when LoadOptions enables it. Reset/drop withdraws
    /// automatic entries contributed by this session; active readers retain their pins.
    pub keep_kv: bool,
    /// Policy when history plus the requested answer exceeds context.
    pub on_overflow: Overflow,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            system: None,
            keep_kv: true,
            on_overflow: Overflow::default(),
        }
    }
}

pub(crate) fn valid_history(opts: &SessionOptions, history: &[ChatMessage]) -> bool {
    let first = usize::from(opts.system.is_some());
    if let Some(system) = &opts.system
        && !history
            .first()
            .is_some_and(|m| *m == ChatMessage::new(Role::System, system.clone(), false))
    {
        return false;
    }
    history.get(first..).is_some_and(|messages| {
        (0..messages.len()).all(|i| follows(&messages[..i], &messages[i]))
            && messages.last().is_none_or(|m| m.role == Role::Assistant)
    })
}

/// Whether `next` may come after `before`, the history after the system message. An exchange is
/// a prompt and its answer, then any number of tool turns: one result per call of the answer
/// before them, in the calls' order, then their answer.
pub(crate) fn follows(before: &[ChatMessage], next: &ChatMessage) -> bool {
    let prev = before.last();
    next.well_formed()
        && match next.role {
            Role::System => false,
            Role::User => prev.is_none_or(|p| p.role == Role::Assistant),
            Role::Assistant => match prev.map(|p| p.role) {
                Some(Role::User) => true,
                Some(Role::Tool) => results(before).is_some_and(|(calls, n)| n == calls.len()),
                _ => false,
            },
            Role::Tool => results(before)
                .is_some_and(|(calls, n)| calls.get(n).is_some_and(|c| c.id == next.tool_call_id)),
        }
}

/// The calls of the assistant message before `before`'s trailing tool results, and how many
/// results there are.
fn results(before: &[ChatMessage]) -> Option<(&[ToolCall], usize)> {
    let n = before
        .iter()
        .rev()
        .take_while(|m| m.role == Role::Tool)
        .count();
    let calling = before.get(before.len().checked_sub(n + 1)?)?;
    Some((&calling.tool_calls, n))
}

/// The length of the oldest exchange in `messages` (a valid history after its system message), a
/// tool exchange whole; None when it is the last one, which overflow never drops.
pub(crate) fn oldest_exchange(messages: &[ChatMessage]) -> Option<usize> {
    let next = messages.iter().skip(1).position(|m| m.role == Role::User)?;
    Some(next + 1)
}

/// Tokens a KV pool block holds (larch-engine's `sched::BLOCK_TOKENS`, not exported).
pub(crate) const KV_BLOCK_TOKENS: u32 = 16;

/// The share of the room (the context, or the KV pool when it holds less) a prompt's images may fill
/// as pixels, newest first: an older image past it shows as its note ("pixels while recent").
pub const PIXEL_SHARE: (u32, u32) = (1, 2);

/// Whether `text` talks about a picture: one of a few words, whole.
// ponytail: an English allowlist, whole words. A miss ("photograph", another language) leaves an old
// image a note; a false hit ("the screen went black") shows the newest noted image's rows for one
// turn. "figure" is out ("figure out"). Let the agent pin the image a question names
// (`Session::pin_images`) when chats show either often.
pub fn mentions_image(text: &str) -> bool {
    const WORDS: [&str; 17] = [
        "image",
        "images",
        "picture",
        "pictures",
        "photo",
        "photos",
        "screenshot",
        "screenshots",
        "screen",
        "diagram",
        "diagrams",
        "chart",
        "charts",
        "pic",
        "pics",
        "img",
        "drawing",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .any(|w| WORDS.iter().any(|x| w.eq_ignore_ascii_case(x)))
}

/// For each message of `history`, which of its images show as text this turn (true) rather than
/// pixels. From the newest image back, images stay pixels while their rows fit the share of `room`
/// ([`PIXEL_SHARE`]); the newest message's images and `pinned` ones always do. An older image shows
/// as its note when it has one (and stays pixels until then, so nothing is lost before a note
/// exists). Once a note, an image stays one (`sticky`: a prompt that changes form re-reads every
/// row after it) unless pinned. `shrunk` are images an overflow turned into text this turn, by
/// reference: their note, or `[image]`. An image whose bytes are gone is text too, as it is without a
/// projector: its note, or `[image]` (D10).
#[allow(clippy::too_many_arguments)]
fn image_forms(
    history: &[ChatMessage],
    images: &HashMap<String, Arc<Image>>,
    notes: &HashMap<String, String>,
    pinned: &HashSet<String>,
    sticky: &HashSet<String>,
    shrunk: &HashSet<String>,
    vision: Option<&ProjectorConfig>,
    room: u32,
) -> Vec<Vec<bool>> {
    let mut out: Vec<Vec<bool>> = history
        .iter()
        .map(|m| vec![false; m.images.len()])
        .collect();
    let Some(v) = vision else {
        out.iter_mut().for_each(|row| row.fill(true));
        return out;
    };
    let budget = u64::from(room) * u64::from(PIXEL_SHARE.0) / u64::from(PIXEL_SHARE.1);
    let newest = history.iter().rposition(|m| !m.images.is_empty());
    let mut used = 0u64;
    for (mi, m) in history.iter().enumerate().rev() {
        for (ii, r) in m.images.iter().enumerate().rev() {
            let Some(img) = images.get(r).filter(|_| !shrunk.contains(r)) else {
                // Shrunk by an overflow, or its bytes are gone (a file missing from a resumed chat).
                out[mi][ii] = true;
                continue;
            };
            let rows = u64::from(v.tokens_for(img.width(), img.height()));
            let noted = notes.contains_key(r);
            let keep = Some(mi) == newest
                || pinned.contains(r)
                || (!(noted && sticky.contains(r)) && used + rows <= budget);
            if keep || !noted {
                used += rows;
            } else {
                out[mi][ii] = true;
            }
        }
    }
    out
}

/// `s` with each image mark replaced: in a prompt with images, only an image places one.
fn unmarked_str(s: &str) -> String {
    s.replace(crate::api::IMAGE_MARK, "\u{FFFD}")
}

/// [`unmarked_str`] over every string of a JSON value, keys included.
fn unmarked(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::String(s) => Value::String(unmarked_str(s)),
        Value::Array(a) => Value::Array(a.iter().map(unmarked).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (unmarked_str(k), unmarked(v)))
                .collect(),
        ),
        v => v.clone(),
    }
}

/// How an image shows as text: its note, as data the model read off a picture (not the user's
/// words), or `[image]` without one.
fn image_text(note: Option<&String>) -> String {
    match note {
        Some(note) => format!("[An earlier image, as the model described it: {note}]\n"),
        None => "[image]\n".into(),
    }
}

/// The image a question about a picture brings back as pixels for its turn: the newest one that is
/// a note, when the newest user message has no image of its own and talks about one.
fn asked_image(history: &[ChatMessage], sticky: &HashSet<String>) -> Option<String> {
    let last = history.iter().rev().find(|m| m.role == Role::User)?;
    if !last.images.is_empty() || !mentions_image(&last.content) {
        return None;
    }
    (history.iter().rev().flat_map(|m| m.images.iter().rev()))
        .find(|r| sticky.contains(*r))
        .cloned()
}

/// `messages` may follow `inner`'s history as a turn: the session is idle, each message follows the
/// one before it, and every call has its result, so the turn's answer can follow.
fn admits(inner: &SessionInner, messages: &[ChatMessage]) -> Result<(), EngineError> {
    if inner.active.is_some() {
        return Err(EngineError::InvalidRequest(invalid::SESSION_BUSY));
    }
    let first = usize::from(
        inner
            .history
            .first()
            .is_some_and(|m| m.role == Role::System),
    );
    let mut history = inner.history[first..].to_vec();
    let answer = ChatMessage::new(Role::Assistant, String::new(), false);
    for message in messages.iter().chain([&answer]) {
        if !follows(&history, message) {
            return Err(EngineError::InvalidRequest(invalid::HISTORY));
        }
        history.push(message.clone());
    }
    Ok(())
}

/// `inner`'s history and images with the turn that sends `text` and `incoming` appended.
fn with_turn(
    inner: &SessionInner,
    text: &str,
    incoming: &[Arc<Image>],
) -> (Vec<ChatMessage>, HashMap<String, Arc<Image>>) {
    let (mut history, mut images) = (inner.history.clone(), inner.images.clone());
    let mut m = ChatMessage::new(Role::User, text.into(), false);
    for img in incoming {
        m.images.push(img.hex());
        images.insert(img.hex(), img.clone());
    }
    history.push(m);
    (history, images)
}

/// Drops the images (and their notes and pixels) `inner`'s history no longer names.
fn forget_unnamed(inner: &mut SessionInner) {
    let named: HashSet<String> = (inner.history.iter())
        .flat_map(|m| m.images.iter().cloned())
        .collect();
    inner.images.retain(|r, _| named.contains(r));
    inner.notes.retain(|r, _| named.contains(r));
    inner.sticky.retain(|r| named.contains(r));
    inner.pixels.retain(|r, _| named.contains(r));
}

pub(crate) struct Active {
    pub id: u64,
    pub handle: larch_engine::RequestCanceller,
}

pub(crate) struct SessionInner {
    pub id: u64,
    pub history: Vec<ChatMessage>,
    pub active: Option<Active>,
    /// The images the history's references name, by reference; one named but missing here (a file
    /// gone or changed under a resumed transcript) renders as its message's text alone.
    pub images: HashMap<String, Arc<Image>>,
    /// Short descriptions of images, by reference ([`Session::set_image_note`]): an older image
    /// past the pixel budget shows as its description ([`image_forms`]).
    pub notes: HashMap<String, String>,
    /// Images the next turn shows as pixels whatever their age ([`Session::pin_images`]).
    pub pinned: HashSet<String>,
    /// Images a sent prompt showed as their notes: they stay notes (see [`image_forms`]).
    pub sticky: HashSet<String>,
    /// Each shown image cut for the encoder, on the caller's thread: the engine never decodes on
    /// its own (that stalled every chat, and a bad file faulted the engine).
    pub pixels: HashMap<String, Arc<ImagePixels>>,
}

pub(crate) struct Turn {
    pub inner: Arc<Mutex<SessionInner>>,
    pub session: u64,
    pub request: u64,
    pub before: Vec<ChatMessage>,
    pub record_history: bool,
}

impl Turn {
    fn owns(&self, inner: &SessionInner) -> bool {
        inner.id == self.session && inner.active.as_ref().is_some_and(|a| a.id == self.request)
    }

    /// Submits the turn's continuation (`GenParams::thinking_budget`) as its request from now on,
    /// so a reset cancels it; Cancelled when a reset or restore took the turn first.
    pub fn resubmit(
        &mut self,
        runtime: &Runtime,
        request: Request,
    ) -> Result<RequestHandle, EngineError> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if !self.owns(&inner) {
            return Err(EngineError::Cancelled);
        }
        let handle = runtime.engine.submit(request)?;
        self.request = handle.id;
        inner.active = Some(Active {
            id: handle.id,
            handle: handle.canceller(),
        });
        Ok(handle)
    }

    pub fn end(
        &self,
        runtime: &Runtime,
        text: &str,
        end: Result<bool, EngineError>,
        release: bool,
    ) {
        // Reached from Drop: a second panic on a poisoned lock would abort during unwinding.
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if !self.owns(&inner) {
            return;
        }
        match end {
            Ok(truncated) if self.record_history => {
                inner
                    .history
                    .push(ChatMessage::new(Role::Assistant, text.into(), truncated))
            }
            Ok(_) => {}
            Err(_) => inner.history = self.before.clone(),
        }
        // The turn kept every image it started with, for a rollback; once its history is final the
        // images, notes and pixels of exchanges an overflow dropped go (each image holds its file's
        // bytes, up to 32 MiB).
        forget_unnamed(&mut inner);
        inner.active = None;
        if release {
            // A queued Done may already have parked. Release is sticky; rotate the ID so future
            // natural turns can park again while the old ID can never resurrect.
            runtime.engine.release_session(inner.id);
            inner.id = fresh_id();
        }
    }
}

/// A single-active-turn conversation; its stream may outlive the Session.
pub struct Session {
    pub(crate) inner: Arc<Mutex<SessionInner>>,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) opts: SessionOptions,
}

impl Session {
    pub(crate) fn new(runtime: Arc<Runtime>, opts: SessionOptions) -> Self {
        let history = opts
            .system
            .as_ref()
            .map(|s| ChatMessage::new(Role::System, s.clone(), false))
            .into_iter()
            .collect();
        Self {
            inner: Arc::new(Mutex::new(SessionInner {
                id: fresh_id(),
                history,
                active: None,
                images: HashMap::new(),
                notes: HashMap::new(),
                pinned: HashSet::new(),
                sticky: HashSet::new(),
                pixels: HashMap::new(),
            })),
            runtime,
            opts,
        }
    }

    /// Generates from a complete caller-owned prompt using this session's existing KV prefix.
    /// The supplied prompt is tokenized exactly as by `Larch::generate`; generated answers are
    /// never appended to it or to `history()`. This supports scripted histories and raw prompts
    /// while retaining single-active-turn, cancellation and session-isolation guarantees.
    pub fn generate(&mut self, prompt: Prompt, p: GenParams) -> Result<TextStream, EngineError> {
        // A kept chat prompt parks a hybrid model's state at its history end, as `submit` does.
        let (prompt, history_end) = match prompt {
            Prompt::Messages(messages) if self.opts.keep_kv => {
                let messages = tool_messages(messages);
                let tokens = (self.runtime).encode_messages(&messages, &p.tools, p.thinking)?;
                let end = (self.runtime).history_len(&messages, &p.tools, p.thinking, &tokens);
                (tokens, end)
            }
            prompt => (self.runtime.encode_prompt(prompt, &p.tools, p.thinking)?, 0),
        };
        self.generate_tokens(prompt, history_end, p)
    }

    /// [`Self::generate`] on a prompt the caller rendered: a kept session parks a hybrid model's
    /// state at `history_end`, the length of the chat history the next prompt repeats
    /// (`Request::history_end`; 0 parks it at the turn's end).
    pub fn generate_tokens(
        &self,
        prompt: Vec<u32>,
        history_end: u32,
        p: GenParams,
    ) -> Result<TextStream, EngineError> {
        let mut inner = self.inner.lock().expect("session");
        if inner.active.is_some() {
            return Err(EngineError::InvalidRequest(invalid::SESSION_BUSY));
        }
        let mut request = self.runtime.request(prompt, &p);
        request.session = inner.id;
        request.keep_kv = self.opts.keep_kv;
        request.history_end = history_end;
        let opens = self.runtime.opens_thinking(&request.prompt);
        let (handle, next) = self.runtime.submit(request, &p)?;
        let turn = Turn {
            inner: self.inner.clone(),
            session: inner.id,
            request: handle.id,
            before: inner.history.clone(),
            record_history: false,
        };
        let stream = TextStream::new(self.runtime.clone(), handle, &p, Some(turn), 0, next, opens);
        inner.active = Some(Active {
            id: stream.request_id(),
            handle: stream.handle.canceller(),
        });
        Ok(stream)
    }

    /// Renders and tokenizes the entire history, then submits one turn under this session ID.
    pub fn send(&mut self, user: &str, p: GenParams) -> Result<TextStream, EngineError> {
        self.submit([ChatMessage::new(Role::User, user.into(), false)], p)
    }

    /// [`Session::send`] with images (the text may be empty): the user message names them and the
    /// session keeps them for later turns, which see them again. Their bytes go beside the transcript
    /// first (`Transcript::store_image`). `IMAGE` for more than [`MAX_MESSAGE_IMAGES`] or one that
    /// does not decode, `NO_VISION` on a model without a projector.
    pub fn send_images(
        &mut self,
        user: &str,
        images: &[Arc<Image>],
        p: GenParams,
    ) -> Result<TextStream, EngineError> {
        if images.is_empty() {
            return self.send(user, p);
        }
        let Some(v) = self.runtime.engine.vision() else {
            return Err(EngineError::InvalidRequest(invalid::NO_VISION));
        };
        if images.len() > MAX_MESSAGE_IMAGES {
            return Err(EngineError::InvalidRequest(invalid::IMAGE));
        }
        let mut m = ChatMessage::new(Role::User, user.into(), false);
        m.images = images.iter().map(|i| i.hex()).collect();
        // A turn the session refuses decodes nothing.
        admits(
            &self.inner.lock().expect("session"),
            std::slice::from_ref(&m),
        )?;
        // This turn's images are cut on the caller's thread (or were, as they were attached:
        // `Larch::prepare_image`), never on the engine's.
        let new = (images.iter())
            .map(|img| Ok((img.clone(), self.runtime.cut(img, &v)?)))
            .collect::<Result<Vec<_>, EngineError>>()?;
        self.submit_cut([m], p, &new)
    }

    /// Makes `images` available to the history's references (a resumed transcript's, as
    /// `Transcript::images` reads them). [`Session::restore`] keeps the ones its history still names,
    /// so this is for images the session never had (a new session over a stored chat).
    pub fn add_images(&mut self, images: impl IntoIterator<Item = Arc<Image>>) {
        let mut inner = self.inner.lock().expect("session");
        for img in images {
            inner.images.insert(img.hex(), img);
        }
    }

    /// Records a short description of the image `reference` names (what
    /// [`Session::describe_image`] wrote, or a stored one): once the image is older than the pixel
    /// budget holds, prompts show this instead of its rows, and a model without vision sees it too.
    pub fn set_image_note(&mut self, reference: &str, note: &str) {
        // A note is the model's reading of a picture, which an attacker can draw: it must not
        // carry an image mark, a special token (`<|im_start|>`, which the tokenizer would parse
        // inside content) or the projector's `<img>` wrapper. Angle brackets become their
        // single-guillemet look-alikes, and the mark becomes U+FFFD.
        let note: String = note
            .trim()
            .chars()
            .map(|c| match c {
                crate::api::IMAGE_MARK => char::REPLACEMENT_CHARACTER,
                '<' => '\u{2039}',
                '>' => '\u{203A}',
                c => c,
            })
            .collect();
        let mut inner = self.inner.lock().expect("session");
        inner.notes.insert(reference.to_string(), note);
    }

    /// Shows the image `reference` names as its note from now on, as a prompt that showed the note
    /// does (see [`image_forms`]): a reopened chat keeps what it showed before it closed.
    pub fn keep_as_note(&mut self, reference: &str) {
        let mut inner = self.inner.lock().expect("session");
        inner.sticky.insert(reference.to_string());
    }

    /// The description recorded for the image `reference` names, if any.
    pub fn image_note(&self, reference: &str) -> Option<String> {
        self.inner
            .lock()
            .expect("session")
            .notes
            .get(reference)
            .cloned()
    }

    /// The images the next prompt (with `incoming`, the turn's own images) would show as notes if
    /// they had one and have none yet, newest first: the ones to describe before that turn, so an
    /// old image shrinks to words instead of staying rows. Empty without a projector.
    pub fn images_wanting_notes(&self, incoming: &[Arc<Image>]) -> Vec<String> {
        let Some(v) = self.runtime.engine.vision() else {
            return Vec::new();
        };
        let inner = self.inner.lock().expect("session");
        let (history, images) = with_turn(&inner, "", incoming);
        // Every image as if noted: which ones the budget would turn into words.
        let mut all = inner.notes.clone();
        for r in history.iter().flat_map(|m| &m.images) {
            all.entry(r.clone()).or_default();
        }
        let forms = image_forms(
            &history,
            &images,
            &all,
            &inner.pinned,
            &inner.sticky,
            &HashSet::new(),
            Some(&v),
            self.pixel_room(),
        );
        let mut want: Vec<String> = Vec::new();
        for (m, text) in history.iter().zip(&forms).rev() {
            for (r, &as_text) in m.images.iter().zip(text).rev() {
                if as_text
                    && !inner.notes.contains_key(r)
                    && inner.images.contains_key(r)
                    && !want.contains(r)
                {
                    want.push(r.clone());
                }
            }
        }
        want
    }

    /// The prompt tokens the history's images take on the turn that sends `text` with `incoming`
    /// (the turn's own images, which it does not count), in the form that turn shows them: an image
    /// as pixels its rows, one as text its note's tokens (or `[image]`'s). The new images push older
    /// ones out of the pixel budget, and a question about a picture brings one back, as in the
    /// prompt. What a count of the history's text misses (the agent's context meter).
    pub fn image_tokens(&self, text: &str, incoming: &[Arc<Image>]) -> u64 {
        let vision = self.runtime.engine.vision();
        let inner = self.inner.lock().expect("session");
        let (history, images, forms) = self.turn_forms(&inner, text, incoming);
        let mut total = 0;
        let before = inner.history.len();
        for (m, text) in history[..before].iter().zip(&forms) {
            for (r, &as_text) in m.images.iter().zip(text) {
                total += match (as_text, vision.as_ref(), images.get(r)) {
                    (false, Some(v), Some(img)) => {
                        u64::from(v.tokens_for(img.width(), img.height()))
                    }
                    (false, ..) => 0,
                    (true, ..) => {
                        let shown = image_text(inner.notes.get(r));
                        self.runtime.tokenizer.encode(&shown, false).len() as u64
                    }
                };
            }
        }
        total
    }

    /// The images the turn that sends `text` with `incoming` would show as rows though they could
    /// be words: not the newest message's, not pinned or asked about. Turned to notes
    /// ([`Session::keep_as_note`]) they make room before an exchange is dropped or summarized.
    pub fn shrinkable_images(&self, text: &str, incoming: &[Arc<Image>]) -> Vec<String> {
        let inner = self.inner.lock().expect("session");
        let (history, _, forms) = self.turn_forms(&inner, text, incoming);
        let newest = history.iter().rposition(|m| !m.images.is_empty());
        let mut pinned = inner.pinned.clone();
        pinned.extend(asked_image(&history, &inner.sticky));
        let mut out: Vec<String> = Vec::new();
        for (mi, (m, text)) in history.iter().zip(&forms).enumerate() {
            for (r, &as_text) in m.images.iter().zip(text) {
                if !as_text && Some(mi) != newest && !pinned.contains(r) && !out.contains(r) {
                    out.push(r.clone());
                }
            }
        }
        out
    }

    /// The room the pixel budget is a share of: the context, or the KV pool when it holds less
    /// (InternVL3.5-1B's 112 KiB a token leave a 4 GB card about 9k tokens of pool beside the
    /// projector).
    fn pixel_room(&self) -> u32 {
        let d = self.runtime.engine.deployment();
        (d.ctx_max).min(d.kv_pool_blocks.saturating_mul(KV_BLOCK_TOKENS).max(1))
    }

    /// The history, images and image forms of the turn that sends `text` with `incoming`.
    #[allow(clippy::type_complexity)]
    fn turn_forms(
        &self,
        inner: &SessionInner,
        text: &str,
        incoming: &[Arc<Image>],
    ) -> (
        Vec<ChatMessage>,
        HashMap<String, Arc<Image>>,
        Vec<Vec<bool>>,
    ) {
        let (history, images) = with_turn(inner, text, incoming);
        let mut pinned = inner.pinned.clone();
        pinned.extend(asked_image(&history, &inner.sticky));
        let forms = image_forms(
            &history,
            &images,
            &inner.notes,
            &pinned,
            &inner.sticky,
            &HashSet::new(),
            self.runtime.engine.vision().as_ref(),
            self.pixel_room(),
        );
        (history, images, forms)
    }

    /// The image `reference` names, when the session has it.
    pub fn image(&self, reference: &str) -> Option<Arc<Image>> {
        self.inner
            .lock()
            .expect("session")
            .images
            .get(reference)
            .cloned()
    }

    /// Whether the engine reads images (a vision projector is loaded beside the model).
    pub fn sees_images(&self) -> bool {
        self.runtime.engine.vision().is_some()
    }

    /// The prompt rows `image` takes as pixels on this engine; None without a projector.
    pub fn image_rows(&self, image: &Image) -> Option<u32> {
        (self.runtime.engine.vision()).map(|v| v.tokens_for(image.width(), image.height()))
    }

    /// Shows `references` as pixels on the next turn, however old: the user asked about them again.
    pub fn pin_images<'a>(&mut self, references: impl IntoIterator<Item = &'a str>) {
        let mut inner = self.inner.lock().expect("session");
        inner
            .pinned
            .extend(references.into_iter().map(str::to_string));
    }

    /// Asks the model for a short description of the image `reference` names, as a background
    /// request outside the history (a later prompt shows it in place of the image's rows once the
    /// image is old; see [`Session::set_image_note`]). The session can run a turn meanwhile.
    /// `max_tokens` 0 is [`crate::api::DESCRIBE_TOKENS`]. `IMAGE` when the session lacks the image,
    /// `NO_VISION` on a model without a projector.
    pub fn describe_image(&self, reference: &str, p: GenParams) -> Result<TextStream, EngineError> {
        let image = (self.inner.lock().expect("session").images.get(reference))
            .cloned()
            .ok_or(EngineError::InvalidRequest(invalid::IMAGE))?;
        crate::api::describe(&self.runtime, &image, p)
    }

    /// Appends the result of the last answer's call `call_id` and submits the next turn, with
    /// `send`'s busy and terminal rules: on Error the tool message is removed again.
    pub fn send_tool_result(
        &mut self,
        call_id: &str,
        content: &str,
        p: GenParams,
    ) -> Result<TextStream, EngineError> {
        self.submit([tool_message(call_id.into(), content.into())], p)
    }

    /// Appends the result of the last answer's call `call_id`, then `answer` as the reply to it,
    /// without generating: for an answer the caller wrote itself. The session keeps its ID and
    /// its parked KV, so the next turn prefills only these two messages and its own prompt. A
    /// result for no call of the last answer, or for one of several, is `HISTORY`.
    pub fn answer_tool_call(
        &mut self,
        call_id: &str,
        content: &str,
        answer: &str,
        truncated: bool,
    ) -> Result<(), EngineError> {
        self.answer_tool_calls(&[(call_id, content)], answer, truncated)
    }

    /// [`answer_tool_call`](Self::answer_tool_call) for the last answer's several calls: one
    /// result per `(call_id, content)`, in the calls' order, then `answer` as the one reply to
    /// them all. A missing, unknown or out-of-order call ID is `HISTORY` before anything is
    /// appended.
    pub fn answer_tool_calls(
        &mut self,
        results: &[(&str, &str)],
        answer: &str,
        truncated: bool,
    ) -> Result<(), EngineError> {
        let mut inner = self.inner.lock().expect("session");
        if inner.active.is_some() {
            return Err(EngineError::InvalidRequest(invalid::SESSION_BUSY));
        }
        let first = usize::from(
            inner
                .history
                .first()
                .is_some_and(|m| m.role == Role::System),
        );
        let results =
            (results.iter()).map(|(id, content)| tool_message((*id).into(), (*content).into()));
        let reply = ChatMessage::new(Role::Assistant, answer.into(), truncated);
        let mut history = inner.history.clone();
        for message in results.chain([reply]) {
            if !follows(&history[first..], &message) {
                return Err(EngineError::InvalidRequest(invalid::HISTORY));
            }
            history.push(message);
        }
        inner.history = history;
        Ok(())
    }

    /// [`send_tool_result`](Self::send_tool_result) for the last answer's several calls: one
    /// tool message per `(call_id, content)`, in the calls' order, then one turn. A missing,
    /// unknown or repeated call ID is `HISTORY` before anything is appended.
    pub fn send_tool_results(
        &mut self,
        mut results: Vec<(String, String)>,
        p: GenParams,
    ) -> Result<TextStream, EngineError> {
        {
            let inner = self.inner.lock().expect("session");
            let calls = inner.history.last().map_or(&[][..], |m| &m.tool_calls);
            // An unknown ID sorts first, where `submit` refuses it.
            results.sort_by_key(|(id, _)| calls.iter().position(|c| c.id == *id));
        }
        let messages = results
            .into_iter()
            .map(|(id, content)| tool_message(id, content));
        self.submit(messages, p)
    }

    /// Gives the last answer, once, the call the caller parsed from it and the text to keep beside
    /// it (the answer without its call markup). Call it before the turn's transcript record is
    /// written.
    pub fn set_tool_calls(&mut self, content: &str, call: ToolCall) -> Result<(), EngineError> {
        self.store_calls(content, vec![call])
    }

    /// [`set_tool_calls`](Self::set_tool_calls) with several calls, which only a Qwen-architecture
    /// model takes in one turn; the error for another names its architecture. Calls need distinct
    /// IDs, so each result can name its call. An engine-side refusal downcasts to `EngineError`.
    // ponytail: the family is the GGUF architecture, so a Mistral GGUF (architecture llama) is
    // named llama. Upgrade: name the tool format when the facade learns it from the template.
    pub fn set_several_tool_calls(
        &mut self,
        content: &str,
        calls: Vec<ToolCall>,
    ) -> anyhow::Result<()> {
        if calls.len() > 1 && !self.takes_several_calls() {
            let arch = self.runtime.engine.deployment().arch;
            anyhow::bail!(
                "a {arch} model takes one tool call per turn; {} calls given",
                calls.len()
            );
        }
        Ok(self.store_calls(content, calls)?)
    }

    /// Whether a turn may hold several calls: the Qwen tool format writes one `<tool_call>` block
    /// per call.
    pub fn takes_several_calls(&self) -> bool {
        let arch = self.runtime.engine.deployment().arch;
        crate::toolcall::family_for(arch.as_str(), None) == Some(crate::toolcall::Family::Qwen3)
    }

    fn store_calls(&mut self, content: &str, calls: Vec<ToolCall>) -> Result<(), EngineError> {
        let mut inner = self.inner.lock().expect("session");
        if inner.active.is_some() {
            return Err(EngineError::InvalidRequest(invalid::SESSION_BUSY));
        }
        // render_tools parses text arguments; one it cannot parse would fail every later render.
        if calls.iter().any(|call| {
            call.arguments
                .as_str()
                .is_some_and(|a| serde_json::from_str::<serde_json::Value>(a).is_err())
        }) {
            return Err(EngineError::InvalidRequest(invalid::CHAT_TEMPLATE));
        }
        match inner.history.last_mut() {
            // Once, so the record written after it stays what the history holds.
            Some(m)
                if m.role == Role::Assistant
                    && !m.truncated
                    && m.tool_calls.is_empty()
                    && !calls.is_empty()
                    && distinct_ids(&calls) =>
            {
                m.content = content.into();
                m.tool_calls = calls;
                Ok(())
            }
            _ => Err(EngineError::InvalidRequest(invalid::HISTORY)),
        }
    }

    /// Appends `messages` (a prompt, or a turn's tool results) and submits one turn.
    fn submit(
        &mut self,
        messages: impl IntoIterator<Item = ChatMessage>,
        p: GenParams,
    ) -> Result<TextStream, EngineError> {
        self.submit_cut(messages, p, &[])
    }

    /// [`Session::submit`], with `cut` the pixels of images the caller already cut, by reference.
    fn submit_cut(
        &mut self,
        messages: impl IntoIterator<Item = ChatMessage>,
        p: GenParams,
        new: &[(Arc<Image>, Arc<ImagePixels>)],
    ) -> Result<TextStream, EngineError> {
        let mut inner = self.inner.lock().expect("session");
        let messages: Vec<ChatMessage> = messages.into_iter().collect();
        admits(&inner, &messages)?;
        // An overflow never drops the system message.
        let first = usize::from(
            inner
                .history
                .first()
                .is_some_and(|m| m.role == Role::System),
        );
        let before = inner.history.clone();
        let mut history = before.clone();
        history.extend(messages);
        // The turn's images join the session's once the turn is sent.
        let mut images = inner.images.clone();
        images.extend(new.iter().map(|(img, _)| (img.hex(), img.clone())));
        let cut: HashMap<String, Arc<ImagePixels>> = (new.iter())
            .map(|(img, px)| (img.hex(), px.clone()))
            .collect();
        let limit = self.runtime.engine.deployment().ctx_max;
        let vision = self.runtime.engine.vision();
        let room = self.pixel_room();
        // A question about a picture brings the newest noted image back as pixels for this turn.
        let mut pinned = inner.pinned.clone();
        pinned.extend(asked_image(&history, &inner.sticky));
        // Images an overflow turned into text before any exchange drops, by reference: a drop moves
        // the messages, not the images.
        let mut shrunk: HashSet<String> = HashSet::new();
        // A budget stop inside a block feeds the close too, so the answer keeps all of max_tokens.
        let continues = p
            .thinking_budget
            .filter(|&b| p.max_tokens == 0 || b < p.max_tokens);
        let close = continues.map_or(0, |_| {
            let len = self.runtime.tokenizer.encode(THINKING_CLOSE, false).len();
            u32::try_from(len).unwrap_or(u32::MAX)
        });
        let mut dropped = 0;
        // ponytail: one whole render per dropped exchange; binary-search the drop count if resuming
        // a long transcript into a small context shows up in send latency.
        let (prompt, messages, mut spans, forms) = loop {
            // A history with images marks each one's place in its message; one without renders as
            // it always has. Recent images are pixels, older ones their notes (`image_forms`).
            let with_images = history.iter().any(|m| !m.images.is_empty());
            let forms = image_forms(
                &history,
                &images,
                &inner.notes,
                &pinned,
                &inner.sticky,
                &shrunk,
                vision.as_ref(),
                room,
            );
            let mut shown = Vec::new();
            let messages: Vec<_> = history
                .iter()
                .zip(&forms)
                .map(|(m, text)| {
                    let mut content = String::new();
                    if with_images {
                        for (r, &as_text) in m.images.iter().zip(text) {
                            if as_text {
                                content.push_str(&image_text(inner.notes.get(r)));
                            } else if let Some(img) = images.get(r) {
                                shown.push((img.clone(), cut.get(r).cloned()));
                                content.push(crate::api::IMAGE_MARK);
                                content.push('\n');
                            }
                        }
                        // The mark is the image's alone: a typed one is not an image.
                        content.extend(m.content.chars().map(|c| match c {
                            crate::api::IMAGE_MARK => char::REPLACEMENT_CHARACTER,
                            c => c,
                        }));
                    } else {
                        content.clone_from(&m.content);
                    }
                    // A tool call's text (model output, often copied from a page) loses its image
                    // marks too, or one would shift every image of the prompt after it.
                    let tool_calls = match with_images {
                        false => m.tool_calls.clone(),
                        true => (m.tool_calls.iter())
                            .map(|c| ToolCall {
                                name: unmarked_str(&c.name),
                                arguments: unmarked(&c.arguments),
                                id: unmarked_str(&c.id),
                            })
                            .collect(),
                    };
                    ToolMessage {
                        role: m.role.as_str().into(),
                        content,
                        tool_calls,
                        tool_call_id: match with_images {
                            false => m.tool_call_id.clone(),
                            true => unmarked_str(&m.tool_call_id),
                        },
                    }
                })
                .collect();
            let (tokens, spans) = match !shown.is_empty() {
                false => (
                    self.runtime
                        .encode_messages(&messages, &p.tools, p.thinking)?,
                    Vec::new(),
                ),
                true => self
                    .runtime
                    .encode_images(&messages, &p.tools, p.thinking, &shown)?,
            };
            // An uncapped answer keeps an eighth of the context free when old exchanges drop.
            let room = match p.max_tokens {
                0 => limit / 8,
                n => n,
            }
            .saturating_add(close);
            let total = tokens.len() as u64 + u64::from(room);
            if total <= u64::from(limit) {
                break (tokens, messages, spans, forms);
            }
            // Before an exchange drops, the oldest image still shown as pixels (not the newest
            // message's) becomes text: its note, or `[image]`; the words around it stay.
            let newest = history.iter().rposition(|m| !m.images.is_empty());
            let older_pixels = (history.iter().zip(&forms).enumerate())
                .filter(|&(mi, _)| Some(mi) != newest)
                .flat_map(|(_, (m, text))| m.images.iter().zip(text))
                .find(|&(_, &as_text)| !as_text);
            if let Some((r, _)) = older_pixels {
                shrunk.insert(r.clone());
                continue;
            }
            let oldest = (self.opts.on_overflow == Overflow::DropOldestExchanges)
                .then(|| oldest_exchange(&history[first..]))
                .flatten();
            let Some(len) = oldest else {
                return Err(EngineError::PromptTooLong {
                    tokens: total.min(u64::from(u32::MAX)) as u32,
                    limit,
                });
            };
            history.drain(first..first + len);
            dropped += 1;
        };
        // Every image the prompt shows as rows goes with its pixels, cut here (once a session) and
        // not on the engine's thread.
        for (r, px) in cut {
            inner.pixels.entry(r.clone()).or_insert_with(|| px.clone());
        }
        if let Some(v) = vision.as_ref() {
            for span in spans.iter_mut().filter(|s| s.pixels.is_none()) {
                let r = span.image.hex();
                let px = match inner.pixels.get(&r) {
                    Some(px) => px.clone(),
                    None => {
                        let px = self.runtime.cut(&span.image, v)?;
                        inner.pixels.insert(r, px.clone());
                        px
                    }
                };
                span.pixels = Some(px);
            }
            // About 8 MB an image: keep only the ones this prompt shows once there are many.
            if inner.pixels.len() > 16 {
                let shown: HashSet<String> = spans.iter().map(|s| s.image.hex()).collect();
                inner.pixels.retain(|r, _| shown.contains(r));
            }
        }
        let history_end = if self.opts.keep_kv {
            (self.runtime).history_len(&messages, &p.tools, p.thinking, &prompt)
        } else {
            0
        };
        let mut request = self.runtime.request(prompt, &p);
        request.images = spans;
        request.session = inner.id;
        request.keep_kv = self.opts.keep_kv;
        request.history_end = history_end;
        let opens = self.runtime.opens_thinking(&request.prompt);
        let (handle, next) = self.runtime.submit(request, &p)?;
        let turn = Turn {
            inner: self.inner.clone(),
            session: inner.id,
            request: handle.id,
            before,
            record_history: true,
        };
        let stream = TextStream::new(
            self.runtime.clone(),
            handle,
            &p,
            Some(turn),
            dropped,
            next,
            opens,
        );
        // An image this prompt showed as its note stays a note (`image_forms`).
        let sticky: Vec<String> = (history.iter().zip(&forms))
            .flat_map(|(m, text)| m.images.iter().zip(text))
            .filter(|&(r, &as_text)| as_text && inner.notes.contains_key(r))
            .map(|(r, _)| r.clone())
            .collect();
        // Only a model that reads images makes notes sticky: a text-only model's notes go back to
        // pixels when a vision model is loaded again.
        if vision.is_some() {
            inner.sticky.extend(sticky);
        }
        inner.history = history;
        inner.images = images;
        inner.pinned.clear();
        inner.active = Some(Active {
            id: stream.request_id(),
            handle: stream.handle.canceller(),
        });
        Ok(stream)
    }

    /// Cancels the active turn, releases old KV, and retains only the system message under a new ID.
    pub fn reset(&mut self) {
        let mut inner = self.inner.lock().expect("session");
        if let Some(active) = inner.active.take() {
            active.handle.cancel();
        }
        self.runtime.engine.release_session(inner.id);
        inner.id = fresh_id();
        inner.history.retain(|m| m.role == Role::System);
        inner.images.clear();
        inner.notes.clear();
        inner.pinned.clear();
        inner.sticky.clear();
        inner.pixels.clear();
    }

    /// Restores validated delivered history under a fresh ID with no process-local KV. Several
    /// calls on one answer need a model that takes them (`set_several_tool_calls`). The images the
    /// restored history still names stay with the session: restore is how a caller prunes or
    /// rewrites a chat, and its pictures must not vanish with the old ID.
    pub fn restore(
        &mut self,
        opts: SessionOptions,
        history: Vec<ChatMessage>,
    ) -> Result<(), EngineError> {
        let several = history.iter().any(|m| m.tool_calls.len() > 1);
        if !valid_history(&opts, &history) || several && !self.takes_several_calls() {
            return Err(EngineError::InvalidRequest(invalid::HISTORY));
        }
        let named = |r: &String| history.iter().any(|m| m.images.contains(r));
        let (mut kept, mut notes, mut sticky) = {
            let mut inner = self.inner.lock().expect("session");
            (
                std::mem::take(&mut inner.images),
                std::mem::take(&mut inner.notes),
                std::mem::take(&mut inner.sticky),
            )
        };
        kept.retain(|r, _| named(r));
        notes.retain(|r, _| named(r));
        sticky.retain(|r| named(r));
        self.reset();
        self.opts = opts;
        let mut inner = self.inner.lock().expect("session");
        inner.history = history;
        inner.images = kept;
        inner.notes = notes;
        inner.sticky = sticky;
        Ok(())
    }

    /// Whether a turn of this session is still running.
    pub fn busy(&self) -> bool {
        self.inner.lock().expect("session").active.is_some()
    }

    /// Returns a copy of the delivered history.
    pub fn history(&self) -> Vec<ChatMessage> {
        self.inner.lock().expect("session").history.clone()
    }

    /// The last prompt and its answer, without copying the rest of the history.
    pub(crate) fn last_exchange(&self) -> Option<[ChatMessage; 2]> {
        let inner = self.inner.lock().expect("session");
        inner.history.last_chunk::<2>().cloned()
    }

    /// The last turn's messages for [`Transcript::append_messages`](crate::chat::Transcript):
    /// its prompt or tool results, then its answer; empty before the first answer.
    pub fn last_turn(&self) -> Vec<ChatMessage> {
        let inner = self.inner.lock().expect("session");
        let history = &inner.history;
        let results = history
            .iter()
            .rev()
            .skip(1)
            .take_while(|m| m.role == Role::Tool)
            .count();
        let start = history.len().checked_sub(results.max(1) + 1);
        match (history.last(), start) {
            (Some(m), Some(start)) if m.role == Role::Assistant => history[start..].to_vec(),
            _ => Vec::new(),
        }
    }

    /// Returns this conversation's current explicit ID.
    pub fn id(&self) -> u64 {
        self.inner.lock().expect("session").id
    }

    /// Returns the engine's authoritative parked KV length.
    pub fn tokens_in_kv(&self) -> u32 {
        self.runtime.engine.parked_tokens(self.id())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        self.runtime.engine.release_session(inner.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larch_model::{gguf::Gguf, tiny};

    fn picture() -> Arc<Image> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../larch-model/tests/data/vision/palette.png");
        Arc::new(Image::new(std::fs::read(p).unwrap().into()).unwrap())
    }

    fn user(text: &str, images: &[&str]) -> ChatMessage {
        let mut m = ChatMessage::new(Role::User, text.into(), false);
        m.images = images.iter().map(|s| s.to_string()).collect();
        m
    }

    /// Three exchanges with an image each (4 rows apiece through the generated projector), then a
    /// question without one.
    fn chat(question: &str) -> Vec<ChatMessage> {
        let a = |t: &str| ChatMessage::new(Role::Assistant, t.into(), false);
        vec![
            user("first", &["a"]),
            a("one"),
            user("second", &["b"]),
            a("two"),
            user("third", &["c"]),
            a("three"),
            user(question, &[]),
        ]
    }

    #[test]
    fn image_marks_leave_a_tool_calls_text() {
        let mark = crate::api::IMAGE_MARK;
        let mut obj = serde_json::Map::new();
        obj.insert(
            format!("k{mark}"),
            serde_json::json!([format!("a{mark}b"), 1, { "x": mark.to_string() }]),
        );
        let out = super::unmarked(&serde_json::Value::Object(obj)).to_string();
        assert!(
            !out.contains(mark) && out.matches('\u{FFFD}').count() == 3,
            "{out}"
        );
        assert_eq!(super::unmarked_str(&format!("{mark}id")), "\u{FFFD}id");
    }

    #[test]
    fn older_images_show_as_their_notes_past_the_pixel_budget() {
        let g = Gguf::parse(&tiny::Projector::new(32, 5).gguf()).unwrap();
        let v = ProjectorConfig::from_gguf(&g).unwrap();
        assert_eq!(v.tokens_for(picture().width(), picture().height()), 4);
        let images: HashMap<String, Arc<Image>> =
            ["a", "b", "c"].map(|r| (r.to_string(), picture())).into();
        let all_notes: HashMap<String, String> = ["a", "b", "c"]
            .map(|r| (r.to_string(), format!("note {r}")))
            .into();
        let none = HashSet::new();
        let no_shrink = HashSet::new();
        // A 16-token room gives 8 rows of pixels: the newest two images; the oldest is its note.
        let forms = |h: &[ChatMessage],
                     notes: &HashMap<String, String>,
                     pinned: &HashSet<String>,
                     sticky: &HashSet<String>,
                     shrunk: &HashSet<String>,
                     vision: Option<&ProjectorConfig>,
                     room: u32| {
            image_forms(h, &images, notes, pinned, sticky, shrunk, vision, room)
                .into_iter()
                .flatten()
                .collect::<Vec<bool>>()
        };
        let h = chat("and then?");
        assert_eq!(
            forms(&h, &all_notes, &none, &none, &no_shrink, Some(&v), 16),
            [true, false, false]
        );
        // Without a note the old image stays pixels: nothing is lost before a description exists.
        let mut notes = all_notes.clone();
        notes.remove("a");
        assert_eq!(
            forms(&h, &notes, &none, &none, &no_shrink, Some(&v), 16),
            [false, false, false]
        );
        // Pinned, it comes back as pixels; once a note it stays one in a roomier context.
        let pinned = HashSet::from(["a".to_string()]);
        assert_eq!(
            forms(&h, &all_notes, &pinned, &none, &no_shrink, Some(&v), 16),
            [false, false, false]
        );
        let sticky = HashSet::from(["a".to_string()]);
        assert_eq!(
            forms(&h, &all_notes, &none, &sticky, &no_shrink, Some(&v), 64),
            [true, false, false]
        );
        assert_eq!(
            forms(&h, &all_notes, &none, &none, &no_shrink, Some(&v), 64),
            [false, false, false]
        );
        // An overflow's shrink turns that image to text; the newest message's images never shrink here.
        let shrunk = HashSet::from(["b".to_string()]);
        assert_eq!(
            forms(&h, &all_notes, &none, &none, &shrunk, Some(&v), 64),
            [false, true, false]
        );
        // A text-only model sees every image as text (its note, or `[image]`).
        assert_eq!(
            forms(&h, &notes, &none, &none, &no_shrink, None, 16),
            [true, true, true]
        );
        // A missing image file shows its note when it has one.
        let mut gone = images.clone();
        gone.remove("a");
        let g = image_forms(
            &h,
            &gone,
            &all_notes,
            &none,
            &none,
            &no_shrink,
            Some(&v),
            64,
        );
        assert!(g[0][0], "a missing image shows its note");
        let g = image_forms(&h, &gone, &notes, &none, &none, &no_shrink, Some(&v), 64);
        assert!(g[0][0], "a missing image without a note shows `[image]`");
        // A question about a picture pins the newest image that is a note.
        let both = HashSet::from(["a".to_string(), "b".to_string()]);
        assert_eq!(
            asked_image(&chat("what did the screenshot say?"), &both),
            Some("b".into())
        );
        assert_eq!(asked_image(&chat("and then?"), &both), None);
        // The turn's own images count (`with_turn`, as `image_tokens` measures a turn): two images
        // fit 16 tokens of room, until a third sent with the turn pushes the oldest to its note.
        let inner = SessionInner {
            id: 0,
            history: h[..4].to_vec(),
            active: None,
            images: images.clone(),
            notes: all_notes.clone(),
            pinned: none.clone(),
            sticky: none.clone(),
            pixels: HashMap::new(),
        };
        let turn = |text: &str, incoming: &[Arc<Image>]| {
            let (h, imgs) = with_turn(&inner, text, incoming);
            image_forms(
                &h,
                &imgs,
                &all_notes,
                &none,
                &none,
                &no_shrink,
                Some(&v),
                16,
            )
            .concat()
        };
        assert_eq!(turn("and then?", &[]), [false, false]);
        assert_eq!(turn("and this?", &[picture()]), [true, false, false]);
    }

    #[test]
    fn a_question_about_a_picture_is_recognised_by_whole_words() {
        assert!(mentions_image("What does the Screenshot show?"));
        assert!(mentions_image("compare the two images"));
        assert!(!mentions_image("pick a topic about imagery"));
        assert!(!mentions_image("explain this paragraph"));
        assert!(!mentions_image("help me figure out why"));
    }
}
