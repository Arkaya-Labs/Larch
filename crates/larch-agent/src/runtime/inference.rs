//! A fresh session for each executor; restore only committed delivered messages.
use crate::workers::{self, Answer};
use larch::{
    ChatMessage, EngineError, GenParams, Larch, Session, SessionOptions, TextEvent, TextStream,
    api::Class,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, PoisonError};

pub struct ExecutorSession {
    session: Session,
}
impl ExecutorSession {
    pub fn new(
        larch: &Larch,
        options: SessionOptions,
        committed: Vec<ChatMessage>,
    ) -> Result<Self, EngineError> {
        let mut session = larch.session(options.clone());
        if !committed.is_empty() {
            session.restore(options, committed)?;
        }
        Ok(Self { session })
    }
    fn send(
        &mut self,
        text: &str,
        params: GenParams,
        stop: &AtomicBool,
    ) -> Result<Answer, EngineError> {
        self.turn(stop, |session| session.send(text, params))
    }
    fn turn(
        &mut self,
        stop: &AtomicBool,
        submit: impl FnOnce(&mut Session) -> Result<TextStream, EngineError>,
    ) -> Result<Answer, EngineError> {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(EngineError::Cancelled);
        }
        // The caller holds this request's admission permit (`send_admitted`, `send_reserved`).
        let stream = Seated::admitted(submit(&mut self.session)?);
        workers::answer(self.session.id(), stream, stop)
    }
    pub fn committed_history(&self) -> Vec<ChatMessage> {
        self.session.history()
    }
}

/// Admission must be shared by all callers of an engine. A permit charges the root even when a
/// request fails or is cancelled, and releases its capacity when settled.
pub trait Admission {
    type Permit: AdmissionPermit<Error = Self::Error>;
    /// Shown to the user as why a step could not get a seat.
    type Error: std::fmt::Display;
    fn acquire(
        &self,
        root: &str,
        required: u64,
        stop: &AtomicBool,
    ) -> Result<Self::Permit, Self::Error>;
}
pub trait AdmissionPermit {
    type Error;
    fn settle(self, used: u64) -> Result<(), Self::Error>;
}

impl ExecutorSession {
    /// Used only when the task actor has atomically reserved the full context for this attempt.
    pub(crate) fn send_reserved(
        &mut self,
        text: &str,
        params: GenParams,
        reserved: u64,
        stop: &AtomicBool,
    ) -> Result<Answer, AdmittedError<std::convert::Infallible>> {
        // A configuration error, said as one: not a cancellation.
        if params.max_tokens == 0 {
            return Err(AdmittedError::Invalid(
                "runtime requires a finite output limit",
            ));
        }
        if reserved < u64::from(params.max_tokens) {
            return Err(AdmittedError::Invalid(
                "context reservation is below output limit",
            ));
        }
        self.send(text, params, stop).map_err(AdmittedError::Engine)
    }
    pub(crate) fn send_admitted<A: Admission>(
        &mut self,
        admission: &A,
        root: &str,
        text: &str,
        params: GenParams,
        max_context_tokens: u64,
        stop: &AtomicBool,
    ) -> Result<Answer, AdmittedError<A::Error>> {
        let p = params.clone();
        self.admitted(admission, root, &params, max_context_tokens, stop, |s| {
            s.send(text, p)
        })
    }
    /// `send_admitted` for the result of the last answer's one call: `content` keeps `call`'s
    /// markup out of the history, which holds the call itself.
    #[allow(clippy::too_many_arguments)]
    /// The answer after the last one's calls, each with its result: several in one turn only on
    /// a model that takes them (`Session::set_several_tool_calls`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tool_results_admitted<A: Admission>(
        &mut self,
        admission: &A,
        root: &str,
        content: &str,
        calls: Vec<(larch::session::ToolCall, String)>,
        params: GenParams,
        max_context_tokens: u64,
        stop: &AtomicBool,
    ) -> Result<Answer, AdmittedError<A::Error>> {
        let results: Vec<(String, String)> = (calls.iter())
            .map(|(call, result)| (call.id.clone(), result.clone()))
            .collect();
        let calls = calls.into_iter().map(|(call, _)| call).collect();
        self.session
            .set_several_tool_calls(content, calls)
            .map_err(|e| match e.downcast::<EngineError>() {
                Ok(e) => AdmittedError::Engine(e),
                Err(_) => AdmittedError::Invalid("this model takes one tool call per turn"),
            })?;
        let p = params.clone();
        self.admitted(admission, root, &params, max_context_tokens, stop, |s| {
            s.send_tool_results(results, p)
        })
    }
    fn admitted<A: Admission>(
        &mut self,
        admission: &A,
        root: &str,
        params: &GenParams,
        max_context_tokens: u64,
        stop: &AtomicBool,
        submit: impl FnOnce(&mut Session) -> Result<TextStream, EngineError>,
    ) -> Result<Answer, AdmittedError<A::Error>> {
        if params.max_tokens == 0 {
            return Err(AdmittedError::Invalid(
                "runtime requires a finite output limit",
            ));
        }
        if max_context_tokens < u64::from(params.max_tokens) {
            return Err(AdmittedError::Invalid(
                "context reservation is below output limit",
            ));
        }
        let permit = admission
            .acquire(root, max_context_tokens, stop)
            .map_err(AdmittedError::Admission)?;
        let answer = self.turn(stop, submit).map_err(AdmittedError::Engine)?;
        let used =
            u64::from(answer.usage.prompt_tokens) + u64::from(answer.usage.completion_tokens);
        permit.settle(used).map_err(AdmittedError::Settle)?;
        Ok(answer)
    }
}
/// Why an admitted model request failed.
#[derive(Debug)]
pub enum AdmittedError<E> {
    /// No seat or permit was given.
    Admission(E),
    /// The engine failed the request.
    Engine(EngineError),
    /// The answer came, but its permit did not settle.
    Settle(E),
    /// The request's limits cannot run.
    Invalid(&'static str),
}
impl<E: std::fmt::Display> std::fmt::Display for AdmittedError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => write!(f, "model seat unavailable: {e}"),
            Self::Engine(e) => write!(f, "model inference failed: {e}"),
            Self::Settle(e) => write!(f, "model seat release failed: {e}"),
            Self::Invalid(e) => f.write_str(e),
        }
    }
}

/// Shared process scheduler backed by the task store's transactional root ledger. The store
/// lock is held only for admission and settlement, never during inference. A failed or cancelled
/// request charges the full reservation; a crash leaves the reservation for reconciliation.
#[derive(Clone)]
pub struct SharedAdmission {
    store: std::sync::Arc<std::sync::Mutex<larch_task::store::SqliteStore>>,
    capacity: usize,
    reserve: u64,
    next: std::sync::Arc<std::sync::atomic::AtomicU64>,
    instance: String,
}
impl SharedAdmission {
    pub fn new(
        store: std::sync::Arc<std::sync::Mutex<larch_task::store::SqliteStore>>,
        capacity: usize,
        reserve: u64,
        instance: String,
    ) -> Result<Self, String> {
        if capacity == 0 || reserve == 0 || instance.trim().is_empty() {
            return Err("invalid admission configuration".into());
        }
        Ok(Self {
            store,
            capacity,
            reserve,
            next: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            instance,
        })
    }
}
pub struct SharedPermit {
    store: std::sync::Arc<std::sync::Mutex<larch_task::store::SqliteStore>>,
    id: String,
    reserved: u64,
    settled: bool,
}
impl Admission for SharedAdmission {
    type Permit = SharedPermit;
    type Error = String;
    fn acquire(
        &self,
        root: &str,
        required: u64,
        stop: &AtomicBool,
    ) -> Result<Self::Permit, String> {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if required > self.reserve {
            return Err("reservation below request context limit".into());
        }
        let number = self
            .next
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| "permit IDs exhausted")?;
        let request = larch_task::admission::Request {
            id: format!("{}:{number}", self.instance),
            root: root.into(),
            reserve: self.reserve,
        };
        self.store
            .lock()
            .map_err(|_| "admission store lock poisoned")?
            .reserve_permit(&request, self.capacity)?;
        Ok(SharedPermit {
            store: self.store.clone(),
            id: request.id,
            reserved: self.reserve,
            settled: false,
        })
    }
}
impl AdmissionPermit for SharedPermit {
    type Error = String;
    fn settle(mut self, used: u64) -> Result<(), String> {
        self.store
            .lock()
            .map_err(|_| "admission store lock poisoned")?
            .settle_permit(&self.id, used)?;
        self.settled = true;
        Ok(())
    }
}
impl Drop for SharedPermit {
    fn drop(&mut self) {
        if !self.settled {
            // Unsettled means failed or cancelled: charge the full reservation, as the type says.
            // A crash never gets here and leaves the reservation for reconciliation.
            if let Ok(mut store) = self.store.lock() {
                let _ = store.settle_permit(&self.id, self.reserved);
            }
        }
    }
}

/// Shared, process-local model seats for requests without a durable token budget. This does not
/// represent token accounting or cross-process device ownership. An engine's callers share one
/// instance (`SeatAdmission::of`).
#[derive(Clone)]
pub struct SeatAdmission {
    state: std::sync::Arc<(Mutex<Pool>, std::sync::Condvar)>,
}
/// The seats' count: what runs, how much may, and the background takers waiting, first first.
struct Pool {
    active: usize,
    capacity: usize,
    waiting: std::collections::VecDeque<u64>,
    next: u64,
}
impl SeatAdmission {
    /// `capacity` seats of their own, which no engine's other callers share: a test's, or a host
    /// with no loaded engine. An engine's callers use `of`.
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("seat capacity must be positive".into());
        }
        let pool = Pool {
            active: 0,
            capacity,
            waiting: Default::default(),
            next: 0,
        };
        Ok(Self {
            state: std::sync::Arc::new((Mutex::new(pool), std::sync::Condvar::new())),
        })
    }
}
/// One taken seat, freed when settled or dropped.
pub struct SeatPermit {
    state: std::sync::Arc<(Mutex<Pool>, std::sync::Condvar)>,
    released: bool,
}
impl Admission for SeatAdmission {
    type Permit = SeatPermit;
    type Error = String;
    fn acquire(
        &self,
        _root: &str,
        _required: u64,
        stop: &AtomicBool,
    ) -> Result<Self::Permit, String> {
        // The task runtime's requests are background work; one seat never exceeds the capacity,
        // so a stop is the only refusal.
        let mut seats = (self.take(1, Class::Background, stop)).map_err(|_| "cancelled")?;
        Ok(seats.pop().expect("take(1) gives one seat"))
    }
}

/// The engines' seats by `Larch::engine_id`, so every caller of one engine in the process finds
/// the same handle (`SeatAdmission::of`) without each signature on the way carrying it. Tests
/// that each load their own engine never share seats.
// ponytail: one entry per engine the process loads, never removed; drop it with the engine if a
// host that switches models often shows the list grow
static SEATS: Mutex<Vec<(u64, SeatAdmission)>> = Mutex::new(Vec::new());

impl SeatAdmission {
    /// The seats of `larch`'s engine, shared by every model caller in the process: as many as
    /// its KV pool fits sequences, unless its loader set them (`install`).
    pub fn of(larch: &Larch) -> SeatAdmission {
        let engine = larch.engine_id();
        let mut all = SEATS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, seats)) = all.iter().find(|(e, _)| *e == engine) {
            return seats.clone();
        }
        let fit = usize::try_from(larch.deployment().fit_max_seqs).unwrap_or(usize::MAX);
        let seats = SeatAdmission::new(fit.max(1)).expect("a positive capacity");
        all.push((engine, seats.clone()));
        seats
    }

    /// `larch`'s engine with `capacity` seats, set by its loader while no seat is taken, also
    /// after a caller found them (`of`).
    pub fn install(larch: &Larch, capacity: usize) -> Result<SeatAdmission, String> {
        if capacity == 0 {
            return Err("seat capacity must be positive".into());
        }
        let seats = SeatAdmission::of(larch);
        let mut pool = seats.pool();
        if pool.active > 0 || !pool.waiting.is_empty() {
            return Err("the engine's seats are in use".into());
        }
        pool.capacity = capacity;
        drop(pool);
        Ok(seats)
    }

    /// The requests that may run at once.
    pub fn capacity(&self) -> usize {
        self.pool().capacity
    }

    /// The most of `want` requests of `class` one group may send together: a background group
    /// waits for all its seats at once, so it is never larger than the pool.
    pub fn group(&self, class: Class, want: usize) -> usize {
        match class {
            Class::Foreground => want,
            Class::Background => want.min(self.capacity()),
        }
        .max(1)
    }

    /// `n` seats together for requests of `class`; `stop` ends a wait. A foreground request (a
    /// person waits on it) never waits: the engine starts it ahead of background work, as it
    /// did before seats, and its seats only hold background back. Background takers wait in
    /// arrival order for all their seats at once: one holding some while it waited for the rest
    /// could hold up another doing the same, and one taking a seat a group ahead of it waits for
    /// could starve that group.
    pub fn take(
        &self,
        n: usize,
        class: Class,
        stop: &AtomicBool,
    ) -> Result<Vec<SeatPermit>, EngineError> {
        use std::sync::atomic::Ordering;
        let (_, ready) = &*self.state;
        let mut pool = self.pool();
        if class == Class::Background && n > pool.capacity {
            let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
            return Err(EngineError::Overloaded {
                waiting: count(n),
                limit: count(pool.capacity),
            });
        }
        if stop.load(Ordering::Acquire) {
            return Err(EngineError::Cancelled);
        }
        // No seat to take has no place in line either.
        if class == Class::Background && n > 0 {
            let ticket = pool.next;
            pool.next += 1;
            pool.waiting.push_back(ticket);
            loop {
                // Before the seat check: a stop that came while the seats freed still wins.
                if stop.load(Ordering::Acquire) {
                    pool.waiting.retain(|t| *t != ticket);
                    // The next in line may fit where this one did not.
                    ready.notify_all();
                    return Err(EngineError::Cancelled);
                }
                let first = pool.waiting.front() == Some(&ticket);
                if first && pool.capacity.saturating_sub(pool.active) >= n {
                    break;
                }
                // The stop is a flag nobody signals, so the wait looks at it now and then.
                pool = ready
                    .wait_timeout(pool, std::time::Duration::from_millis(10))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            pool.waiting.pop_front();
            ready.notify_all();
        }
        pool.active += n;
        Ok((0..n)
            .map(|_| SeatPermit {
                state: self.state.clone(),
                released: false,
            })
            .collect())
    }

    /// `send`'s stream of `class`, sent once `take` gives it a seat, holding it until it ends.
    pub fn seat(
        &self,
        class: Class,
        stop: &AtomicBool,
        send: impl FnOnce() -> Result<TextStream, EngineError>,
    ) -> Result<Seated, EngineError> {
        let seat = self
            .take(1, class, stop)?
            .pop()
            .expect("take(1) gives one seat");
        seat.hold(send())
    }

    /// The count, locked. Nothing panics while it is held, so a poisoned lock still holds a true
    /// count.
    fn pool(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.state.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A model stream holding its engine seat until its terminal event is read or it is dropped. A
/// drop cancels the request, then frees the seat while the engine may still end its pass under
/// way; a reader that stopped holds it until the stream ends (`workers::read`).
pub struct Seated {
    stream: TextStream,
    seat: Option<SeatPermit>,
}
impl Seated {
    /// A stream whose caller already holds its admission permit.
    pub(crate) fn admitted(stream: TextStream) -> Self {
        Self { stream, seat: None }
    }
}
impl Iterator for Seated {
    type Item = TextEvent;
    fn next(&mut self) -> Option<TextEvent> {
        let event = self.stream.next();
        if !matches!(event, Some(TextEvent::Text { .. })) {
            self.seat = None;
        }
        event
    }
}
impl std::ops::Deref for Seated {
    type Target = TextStream;
    fn deref(&self) -> &TextStream {
        &self.stream
    }
}
impl std::ops::DerefMut for Seated {
    fn deref_mut(&mut self) -> &mut TextStream {
        &mut self.stream
    }
}
impl SeatPermit {
    /// `sent`'s stream, holding this seat until it ends; a failed send frees the seat at once.
    pub fn hold(self, sent: Result<TextStream, EngineError>) -> Result<Seated, EngineError> {
        sent.map(|stream| Seated {
            stream,
            seat: Some(self),
        })
    }
    fn release(&mut self) -> Result<(), String> {
        if self.released {
            return Ok(());
        }
        let (lock, ready) = &*self.state;
        // As `SeatAdmission::pool`: a poisoned lock still holds a true count, and a seat lost to
        // it would never come back.
        let mut pool = lock.lock().unwrap_or_else(PoisonError::into_inner);
        pool.active = pool.active.checked_sub(1).ok_or("seat counter underflow")?;
        self.released = true;
        // Every waiter: the first in line may need more seats than this one frees.
        ready.notify_all();
        Ok(())
    }
}
impl AdmissionPermit for SeatPermit {
    type Error = String;
    fn settle(mut self, _used: u64) -> Result<(), String> {
        self.release()
    }
}
impl Drop for SeatPermit {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Waits until `seats` has `n` background takers in line.
    fn queued(seats: &SeatAdmission, n: usize) {
        while seats.pool().waiting.len() != n {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_waiting_group_goes_before_a_later_single_taker() {
        let (seats, never) = (SeatAdmission::new(2).unwrap(), AtomicBool::default());
        let first = seats.take(1, Class::Background, &never).unwrap();
        std::thread::scope(|s| {
            let group = s.spawn(|| seats.take(2, Class::Background, &never).map(|g| g.len()));
            queued(&seats, 1);
            let single = s.spawn(|| seats.take(1, Class::Background, &never).map(|g| g.len()));
            queued(&seats, 2);
            // A seat is free all along, but the group asked first.
            std::thread::sleep(Duration::from_millis(50));
            let (waited, active) = (!single.is_finished(), seats.pool().active);
            drop(first);
            assert!(waited && active == 1, "the single seat went past the group");
            assert_eq!(group.join().unwrap(), Ok(2));
            assert_eq!(single.join().unwrap(), Ok(1));
        });
        assert_eq!(seats.pool().active, 0);
    }

    #[test]
    fn foreground_never_waits_and_background_waits_for_it() {
        let (seats, never) = (SeatAdmission::new(1).unwrap(), AtomicBool::default());
        let background = seats.take(1, Class::Background, &never).unwrap();
        // Beyond the capacity too: the engine queues it ahead of background work.
        let chat = seats.take(2, Class::Foreground, &never).unwrap();
        drop(background);
        std::thread::scope(|s| {
            let next = s.spawn(|| seats.take(1, Class::Background, &never).map(drop));
            queued(&seats, 1);
            std::thread::sleep(Duration::from_millis(50));
            let waited = !next.is_finished();
            drop(chat);
            assert!(waited, "background ran beside foreground at capacity");
            assert_eq!(next.join().unwrap(), Ok(()));
        });
    }

    #[test]
    fn a_stop_ends_a_wait_and_its_place_in_line() {
        let seats = SeatAdmission::new(1).unwrap();
        let (never, stop) = (AtomicBool::default(), AtomicBool::default());
        let held = seats.take(1, Class::Background, &never).unwrap();
        std::thread::scope(|s| {
            let waiting = s.spawn(|| seats.take(1, Class::Background, &stop).map(drop));
            queued(&seats, 1);
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(waiting.join().unwrap(), Err(EngineError::Cancelled));
        });
        assert!(seats.pool().waiting.is_empty());
        drop(held);
        assert_eq!(seats.take(1, Class::Background, &never).map(drop), Ok(()));
    }

    #[test]
    fn a_stop_while_the_seat_frees_cancels_without_sending() {
        let seats = SeatAdmission::new(1).unwrap();
        let (never, stop) = (AtomicBool::default(), AtomicBool::default());
        let mut held = seats.take(1, Class::Background, &never).unwrap();
        let sent = AtomicBool::default();
        std::thread::scope(|s| {
            let waiting = s.spawn(|| {
                seats
                    .seat(Class::Background, &stop, || {
                        sent.store(true, std::sync::atomic::Ordering::Relaxed);
                        Err(EngineError::Cancelled)
                    })
                    .map(drop)
            });
            queued(&seats, 1);
            // Stop and free the seat under one lock, so the waiter sees both at once.
            let mut pool = seats.pool();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            pool.active -= 1;
            held[0].released = true;
            drop(pool);
            seats.state.1.notify_all();
            assert_eq!(waiting.join().unwrap(), Err(EngineError::Cancelled));
        });
        assert!(!sent.load(std::sync::atomic::Ordering::Relaxed));
        assert!(seats.pool().waiting.is_empty());
        assert_eq!(seats.pool().active, 0);
    }

    #[test]
    fn a_background_group_larger_than_the_pool_is_refused() {
        let seats = SeatAdmission::new(2).unwrap();
        let got = seats.take(3, Class::Background, &AtomicBool::default());
        assert!(matches!(
            got,
            Err(EngineError::Overloaded {
                waiting: 3,
                limit: 2
            })
        ));
        assert_eq!(seats.group(Class::Background, 3), 2);
        assert_eq!(seats.group(Class::Foreground, 3), 3);
    }

    #[test]
    fn a_failed_send_and_a_poisoned_lock_still_free_the_seat() {
        let (seats, never) = (SeatAdmission::new(1).unwrap(), AtomicBool::default());
        let seat = seats
            .take(1, Class::Background, &never)
            .unwrap()
            .pop()
            .unwrap();
        assert!(seat.hold(Err(EngineError::Cancelled)).is_err());
        assert_eq!(seats.pool().active, 0);
        let held = seats.take(1, Class::Background, &never).unwrap();
        let poisoner = seats.clone();
        let poisoned = std::thread::spawn(move || {
            let _pool = poisoner.state.0.lock().unwrap();
            panic!("poisons the seats' lock");
        });
        assert!(poisoned.join().is_err() && seats.state.0.is_poisoned());
        drop(held);
        assert_eq!(seats.pool().active, 0);
    }
}
