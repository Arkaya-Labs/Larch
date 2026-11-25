//! Workers: one system prompt over many prompts, as plain sessions sent at the same time so the
//! engine batches them and each reuses the system prompt's KV from the prefix cache. Workers see
//! no tools.

use std::{
    panic,
    sync::{
        Arc, Mutex, PoisonError, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    time::{Duration, Instant},
};

use larch::{EngineError, Finish, GenParams, Larch, SessionOptions, TextEvent, Timings, Usage};

use crate::runtime::inference::{SeatAdmission, Seated};

/// One helper of a chat turn: a worker's share of a skill's work (a section, a paper), for the
/// chat to show beside its progress line (`Agent::helpers`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Helper {
    /// Its place among the turn's helpers, from 0: a second skill call's come after the first's.
    pub id: usize,
    /// What it works on, in a few words.
    pub task: String,
    /// Where it is.
    pub state: HelperState,
}

/// Where a helper is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HelperState {
    /// Waiting for a seat: the engine runs `Workers::at_once` at a time.
    Queued,
    Working,
    Done,
    Failed,
}

thread_local! {
    /// Where this thread's workers say how their helpers stand, and the id their next run's
    /// first helper takes: set on a chat's skill thread (`telling`), so a skill names its helpers
    /// without a sender through every call between.
    static HELPERS: std::cell::RefCell<Option<(mpsc::Sender<Helper>, usize)>> =
        const { std::cell::RefCell::new(None) };
}

/// `f`, with the workers it runs on this thread telling `to` how their helpers stand, numbered
/// from `from` (the helpers the turn already has), so each id names one helper of the turn.
pub fn telling<T>(to: mpsc::Sender<Helper>, from: usize, f: impl FnOnce() -> T) -> T {
    HELPERS.with(|h| *h.borrow_mut() = Some((to, from)));
    let out = f();
    HELPERS.with(|h| *h.borrow_mut() = None);
    out
}

/// Tells this thread's listener, if any, that the helpers `ids` of `tasks` are at `state`.
fn tell(tasks: &[String], ids: std::ops::Range<usize>, state: impl Fn(usize) -> HelperState) {
    HELPERS.with(|h| {
        let Some((to, from)) = &*h.borrow() else {
            return;
        };
        for id in ids.filter(|&id| id < tasks.len()) {
            let task = tasks[id].clone();
            // A chat that stopped listening still gets its answer.
            let _ = to.send(Helper {
                id: from + id,
                task,
                state: state(id),
            });
        }
    });
}

/// How often a read waiting on the engine looks at its stop.
const POLL: Duration = Duration::from_millis(5);

/// The reader threads a stop left running (`read`), alive while their `Arc` is: each holds its
/// stream's engine until the engine's pass under way ends.
// ponytail: process-wide, so one engine's stopped readers hold back another's unload; key them by
// engine once larch hands a stream's engine out
static STOPPED: Mutex<Vec<Weak<()>>> = Mutex::new(Vec::new());

/// Waits up to `limit` for the readers a stop left running to end, so their engine can shut down;
/// false when one still runs.
pub fn wait_stopped(limit: Duration) -> bool {
    let end = Instant::now() + limit;
    loop {
        let mut left = STOPPED.lock().unwrap_or_else(PoisonError::into_inner);
        left.retain(|r| r.strong_count() > 0);
        if left.is_empty() {
            return true;
        }
        drop(left);
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(POLL);
    }
}

/// Runs prompts as parallel sessions on one engine.
#[derive(Clone, Copy, Debug)]
pub struct Workers {
    /// Sequences the engine fits at once (the fit check's answer); workers use one less.
    pub max_workers: usize,
    /// Whether the engine's prefix cache keeps blocks: only then do workers anchor the system
    /// prompt and keep their KV for the group, which cost a prefill and KV the cache repays.
    pub prefix_cache: bool,
}

impl Default for Workers {
    fn default() -> Self {
        Self {
            max_workers: 8,
            prefix_cache: false,
        }
    }
}

/// One model turn of a worker: its session, its text, how it ended (`Finish::Length` is cut at
/// `max_tokens`) and the engine's counts and timings.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    /// The session's ID, for the audit log.
    pub session: u64,
    /// The visible text.
    pub text: String,
    /// Why the turn ended.
    pub finish: Finish,
    /// Prompt, generated and cached tokens.
    pub usage: Usage,
    /// Time to first token, decode speed.
    pub timings: Timings,
    /// The think block `hide_thinking` withheld, if any.
    pub thinking: Option<String>,
}

impl Workers {
    /// The sessions `run` sends at once: one less than `max_workers`, so the user's chat always
    /// has a batch slot; with `max_workers` 1 there is none to spare, and one worker still runs.
    pub fn at_once(&self) -> usize {
        self.max_workers.saturating_sub(1).max(1)
    }

    /// Each prompt's answers, in prompt order: the first, then, when `retry` returns feedback
    /// for it, the answer to that feedback sent as the next user message of the same session.
    /// A retry is never checked again. With the answers comes the engine error that stopped the
    /// session, if any; a failed retry keeps the first answer. Up to `max_workers - 1` sessions
    /// run at once, so the user's chat always has a batch slot. With `prefix_cache`, the system
    /// prompt goes to the cache first and each session keeps its KV, which the cache needs, until
    /// its group is done and the session is dropped; when that first request fails, the sessions
    /// prefill it.
    pub fn run(
        &self,
        larch: &Larch,
        system: &str,
        prompts: &[String],
        p: &GenParams,
        retry: impl Fn(&Answer) -> Option<String>,
    ) -> Vec<(Vec<Answer>, Option<EngineError>)> {
        self.run_until(
            larch,
            system,
            prompts,
            p,
            retry,
            |_, _| {},
            &AtomicBool::default(),
        )
    }

    /// `run`, which `stop` ends: once it is set, nothing more is sent (no anchor, no group), and
    /// each request is cancelled at once, its KV freed when the engine's pass under way ends
    /// (`wait_stopped`); each prompt left gets `EngineError::Cancelled`. `live` gets each first
    /// answer's events as they are read, with the prompt's index: in prompt order, the first while
    /// it is written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_until(
        &self,
        larch: &Larch,
        system: &str,
        prompts: &[String],
        p: &GenParams,
        retry: impl Fn(&Answer) -> Option<String>,
        live: impl FnMut(usize, &TextEvent),
        stop: &AtomicBool,
    ) -> Vec<(Vec<Answer>, Option<EngineError>)> {
        self.run_helpers(larch, system, prompts, &[], p, retry, live, stop)
    }

    /// `run_until`, where each prompt is a helper the chat sees (`telling`), named by `tasks`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_helpers(
        &self,
        larch: &Larch,
        system: &str,
        prompts: &[String],
        tasks: &[String],
        p: &GenParams,
        retry: impl Fn(&Answer) -> Option<String>,
        mut live: impl FnMut(usize, &TextEvent),
        stop: &AtomicBool,
    ) -> Vec<(Vec<Answer>, Option<EngineError>)> {
        tell(tasks, 0..prompts.len(), |_| HelperState::Queued);
        let seats = SeatAdmission::of(larch);
        let n = seats.group(p.class, self.at_once());
        if self.prefix_cache && !system.is_empty() && !prompts.is_empty() {
            // An engine that fails it fails the sessions too, and they say so. `stop` cancels it
            // as it does the sessions, and a stop already set sends it not at all: a run stopped
            // before it starts sends the engine nothing.
            let _ = crate::run::prefill(larch, system, "", p, stop);
        }
        // ponytail: each group's KV stays parked until its slowest answer; keep_kv false is the
        // lever if a long document's group runs the pool short
        let opts = SessionOptions {
            system: (!system.is_empty()).then(|| system.into()),
            keep_kv: self.prefix_cache,
            ..SessionOptions::default()
        };
        let mut answers: Vec<(Vec<Answer>, Option<EngineError>)> =
            Vec::with_capacity(prompts.len());
        // Each prompt of a group, done once its answers are in: failed when it has none.
        let ended = |answers: &[(Vec<Answer>, Option<EngineError>)], from: usize| {
            tell(tasks, from..answers.len(), |id| {
                match answers[id].0.is_empty() {
                    true => HelperState::Failed,
                    false => HelperState::Done,
                }
            })
        };
        // Streams buffer without bound, so reading them in order stalls no worker.
        // ponytail: a group of n starts after the previous group's slowest answer; refill each
        // slot as its answer ends when the review bench shows idle slots
        for chunk in prompts.chunks(n) {
            let done = answers.len();
            if stop.load(Ordering::Relaxed) {
                answers.extend(
                    chunk
                        .iter()
                        .map(|_| (Vec::new(), Some(EngineError::Cancelled))),
                );
                ended(&answers, done);
                continue;
            }
            let mut sessions: Vec<_> = chunk.iter().map(|_| larch.session(opts.clone())).collect();
            let held = match seats.take(chunk.len(), p.class, stop) {
                Ok(held) => held,
                Err(e) => {
                    answers.extend(chunk.iter().map(|_| (Vec::new(), Some(e))));
                    ended(&answers, done);
                    continue;
                }
            };
            tell(tasks, done..done + chunk.len(), |_| HelperState::Working);
            let streams: Vec<_> = (sessions.iter_mut().zip(chunk).zip(held))
                .map(|((session, prompt), seat)| {
                    let sent = seat.hold(session.send(prompt, p.clone()));
                    sent.map(|s| Reading::new(session.id(), s))
                })
                .collect();
            let firsts: Vec<_> = (streams.into_iter().enumerate())
                .map(|(j, s)| s.and_then(|s| read(s, stop, &mut |e| live(done + j, e))))
                .collect();
            // Every retry of the group is sent before any is read, so they batch too; the firsts
            // are read, so their seats are free for them.
            let feedback: Vec<_> = (firsts.iter())
                .map(|first| retry(first.as_ref().ok()?))
                .collect();
            let mut held = seats.take(feedback.iter().flatten().count(), p.class, stop);
            let retries: Vec<_> = (sessions.iter_mut().zip(feedback))
                .map(|(session, feedback)| {
                    let feedback = feedback?;
                    let sent = match &mut held {
                        Ok(held) => {
                            let seat = held.pop().expect("a seat for each retry");
                            seat.hold(session.send(&feedback, p.clone()))
                        }
                        Err(e) => Err(*e),
                    };
                    Some(sent.map(|s| Reading::new(session.id(), s)))
                })
                .collect();
            answers.extend(firsts.into_iter().zip(retries).map(|(first, again)| {
                let first = match first {
                    Ok(first) => first,
                    Err(e) => return (Vec::new(), Some(e)),
                };
                match again.map(|again| again.and_then(|s| read(s, stop, &mut |_| {}))) {
                    Some(Ok(again)) => (vec![first, again], None),
                    Some(Err(e)) => (vec![first], Some(e)),
                    None => (vec![first], None),
                }
            }));
            ended(&answers, done);
        }
        // The next run on this thread (a second skill call's, a later stage's) numbers after these.
        HELPERS.with(|h| {
            if let Some((_, from)) = &mut *h.borrow_mut() {
                *from += tasks.len();
            }
        });
        answers
    }
}

/// The answer `stream` brings in `session`, or the error that ended it; `stop` ends the wait at
/// once and cancels the request.
pub(crate) fn answer(
    session: u64,
    stream: Seated,
    stop: &AtomicBool,
) -> Result<Answer, EngineError> {
    read(Reading::new(session, stream), stop, &mut |_| {})
}

/// A stream read on a thread of its own from its send on, so its seat frees when the engine ends
/// it, not when its caller gets to it: a group reads its streams in turn.
pub(crate) struct Reading {
    session: u64,
    cancel: larch_engine::RequestCanceller,
    events: mpsc::Receiver<TextEvent>,
    thread: std::thread::JoinHandle<()>,
    reader: Weak<()>,
}

impl Reading {
    /// Starts reading `session`'s `stream`.
    pub(crate) fn new(session: u64, stream: Seated) -> Self {
        // The stream is read on its own thread, so a stop never waits on the engine: a prefill,
        // a withheld think block or one forward pass of a model that swaps sends nothing for
        // minutes. The cancel frees the request once the engine's pass under way ends; until
        // then the thread holds the engine, which `wait_stopped` waits for.
        let cancel = stream.canceller();
        let (tx, events) = mpsc::channel();
        let alive = Arc::new(());
        let reader = Arc::downgrade(&alive);
        let thread = std::thread::spawn(move || {
            let _alive = alive;
            for event in stream {
                if tx.send(event).is_err() {
                    break;
                }
            }
        });
        Self {
            session,
            cancel,
            events,
            thread,
            reader,
        }
    }
}

/// `answer` of a stream being read, handing `live` each event before it is read.
pub(crate) fn read(
    reading: Reading,
    stop: &AtomicBool,
    live: &mut dyn FnMut(&TextEvent),
) -> Result<Answer, EngineError> {
    let Reading {
        session,
        cancel,
        events,
        thread,
        reader,
    } = reading;
    let mut text = String::new();
    while !stop.load(Ordering::Relaxed) {
        let event = match events.recv_timeout(POLL) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            // A stream always ends on an event: only a reader that panicked hangs up first, and
            // its panic is the caller's, as when the caller read the stream itself.
            Err(RecvTimeoutError::Disconnected) => {
                panic::resume_unwind(thread.join().expect_err("a stream ends on an event"))
            }
        };
        live(&event);
        match event {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
            TextEvent::Text { delta, .. } => text.push_str(&delta),
            TextEvent::Done {
                finish,
                usage,
                timings,
                thinking,
            } => {
                return Ok(Answer {
                    session,
                    text,
                    finish,
                    usage,
                    timings,
                    thinking,
                });
            }
            TextEvent::Error(e) => return Err(e),
        }
    }
    cancel.cancel();
    let mut stopped = STOPPED.lock().unwrap_or_else(PoisonError::into_inner);
    stopped.retain(|r| r.strong_count() > 0);
    stopped.push(reader);
    Err(EngineError::Cancelled)
}

/// An opt-in worker path with a separate root-budget reservation for every model turn. The
/// default worker entry points retain their existing behavior until their host opts in.
#[cfg(feature = "runtime-admission")]
impl Workers {
    #[allow(clippy::too_many_arguments)]
    pub fn run_admitted<A: crate::runtime::inference::Admission>(
        &self,
        larch: &Larch,
        system: &str,
        prompts: &[String],
        p: &GenParams,
        retry: impl Fn(&Answer) -> Option<String>,
        admission: &A,
        root: &str,
        context_tokens: usize,
        stop: &AtomicBool,
    ) -> Vec<(Vec<Answer>, Option<String>)> {
        self.run_admitted_with_live(
            larch,
            system,
            prompts,
            p,
            retry,
            admission,
            root,
            context_tokens,
            stop,
            |_, _| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_admitted_with_live<A: crate::runtime::inference::Admission>(
        &self,
        larch: &Larch,
        system: &str,
        prompts: &[String],
        p: &GenParams,
        retry: impl Fn(&Answer) -> Option<String>,
        admission: &A,
        root: &str,
        context_tokens: usize,
        stop: &AtomicBool,
        mut live: impl FnMut(usize, &TextEvent),
    ) -> Vec<(Vec<Answer>, Option<String>)> {
        let options = SessionOptions {
            system: (!system.is_empty()).then(|| system.into()),
            keep_kv: self.prefix_cache,
            ..SessionOptions::default()
        };
        prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let mut session = larch.session(options.clone());
                let first = admitted_turn(
                    larch,
                    &mut session,
                    prompt,
                    p,
                    admission,
                    root,
                    context_tokens,
                    stop,
                    &mut |event| live(index, event),
                );
                match first {
                    Err(e) => (vec![], Some(e)),
                    Ok(first) => {
                        let again = retry(&first);
                        let mut answers = vec![first];
                        if let Some(feedback) = again {
                            match admitted_turn(
                                larch,
                                &mut session,
                                &feedback,
                                p,
                                admission,
                                root,
                                context_tokens,
                                stop,
                                &mut |_| {},
                            ) {
                                Ok(second) => answers.push(second),
                                Err(error) => return (answers, Some(error)),
                            }
                        }
                        (answers, None)
                    }
                }
            })
            .collect()
    }
}

#[cfg(feature = "runtime-admission")]
#[allow(clippy::too_many_arguments)]
fn admitted_turn<A: crate::runtime::inference::Admission>(
    larch: &Larch,
    session: &mut larch::Session,
    prompt: &str,
    params: &GenParams,
    admission: &A,
    root: &str,
    context_tokens: usize,
    stop: &AtomicBool,
    live: &mut dyn FnMut(&TextEvent),
) -> Result<Answer, String> {
    use crate::runtime::inference::AdmissionPermit;
    if stop.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    if params.max_tokens == 0 {
        return Err("finite output limit required".into());
    }
    let mut messages: Vec<larch_model::Message> = session
        .history()
        .into_iter()
        .map(|item| larch_model::Message {
            role: item.role.as_str().into(),
            content: item.content,
        })
        .collect();
    messages.push(larch_model::Message {
        role: "user".into(),
        content: prompt.into(),
    });
    let length = larch
        .prompt_len(
            larch::Prompt::Messages(messages),
            &params.tools,
            params.thinking,
        )
        .map_err(|e| e.to_string())?;
    if length
        .checked_add(params.max_tokens as usize)
        .is_none_or(|total| total > context_tokens)
    {
        return Err("worker prompt exceeds context budget".into());
    }
    let permit = admission
        .acquire(root, context_tokens as u64, stop)
        .map_err(|_| "worker admission failed")?;
    // The permit above is this request's admission.
    let stream = Seated::admitted(
        session
            .send(prompt, params.clone())
            .map_err(|e| e.to_string())?,
    );
    let result = read(Reading::new(session.id(), stream), stop, live).map_err(|e| e.to_string())?;
    let used = u64::from(result.usage.prompt_tokens) + u64::from(result.usage.completion_tokens);
    permit
        .settle(used)
        .map_err(|_| "worker accounting failed")?;
    Ok(result)
}
