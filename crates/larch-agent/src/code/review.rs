//! What runs beside the session loop's worker while it works (the plan's "Using concurrency"): a
//! reviewer, one more engine sequence, reads a change the model says is done with no tests to
//! judge it against the request, and names its doubt in the done's stop, never sending it back;
//! and each change's
//! tests run while the model writes its next step, their result joining a later step's when they
//! ended in time. The cross-file checks run inline at a reply's end (`Code::calls`), so they are
//! not run here again.
// ponytail: no headless page run (idea I): nothing in the repo loads a page yet, so `findings`'
// static script checks stand in; add it here when a headless runner lands
// ponytail: the reviewer reads dones only, not each change; add a per-change review when an eval
// shows false dones it catches too late

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use larch::{GenParams, Larch, Prompt, TextStream, api::Grammar};

use crate::{
    code::{
        experts,
        r#loop::{
            Attempt, Code, Evidence, MIN_ROOM, NO_TESTS_DONE, NONE_NEAR_DONE, REVIEWS, Setup,
            findings, message, verdict,
        },
        session::changed,
        task::{CodeTask, TestRun, Verify},
    },
    parse::Parsed,
    run::AgentEvent,
    runtime::inference::Seated,
    skills::{
        ask_docs,
        read_papers::{KV_BLOCK_TOKENS, TEMPLATE_TOKENS},
    },
    tools::code::uncompared,
    workers::{self, Workers},
};

/// The most tokens a verdict takes: `ok`, or `not done: ` and a one-line reason.
const VERDICT_TOKENS: u32 = 160;

/// The reviewer's verdict starts so when the change is not done; its reason follows.
const NOT_DONE: &str = "not done: ";

/// What the reviewer is told to do.
const REVIEWER: &str = "You review a change another model made to a project, which it says is \
done. Read the request, what it said, what the checks found, and the changed lines of each file, \
numbered as they are now. Answer ok when the change does everything the request asks. Otherwise \
answer not done: and the one concrete thing still missing or broken, in one line the other model \
can act on, such as the reset button does nothing.";

/// What the reviewer is told when part of the files did not fit its prompt.
const CUT: &str = "\nParts of the files were left out for room: a part you cannot see is no \
reason for not done.\n";

/// The tools whose calls change nothing the tests read: tests running in the background go on
/// beside them.
pub(super) const QUIET: [&str; 4] = ["view", "search", "lookup", experts::NAME];

/// What a run of the tests in the background found: what the model is told, the user's line,
/// and the run, when the tests ran.
type Found = (String, String, Option<(TestRun, String)>);

/// Tests running in the background, which their own flag stops.
struct Job {
    handle: JoinHandle<Found>,
    stop: Arc<AtomicBool>,
}

impl Job {
    /// Stops the tests and waits for them to end, which a stop makes quick.
    fn end(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
    }
}

/// What runs beside a run's attempts: the reviewer, when the deployment holds it (`fits`), and
/// each attempt's tests in the background.
pub(super) struct Beside {
    /// The reviewer's parameters and prompt cap in tokens (`fits`); None when the deployment
    /// has no room for it.
    reviewer: Option<(GenParams, usize)>,
    /// Why the reviewer is off, until it is told: at the first done it would have read.
    off: Option<&'static str>,
    /// Each attempt's tests running in the background.
    jobs: Vec<Option<Job>>,
    /// Each attempt's files as they were before its first write, for the reviewer's changed
    /// lines.
    base: Vec<BTreeMap<String, String>>,
}

/// The reviewer's prompt cap in tokens on a deployment of `ctx_max` tokens, `kv_pool_blocks` KV
/// blocks (0 is not known, and caps nothing) and `fit_max_seqs` sequences, beside `k` attempts
/// within config.toml's `workers`: a quarter of the context, shrunk to what the pool leaves
/// after the attempts' whole contexts and the verdict; or why it is off, when that leaves less
/// than `MIN_ROOM`. It reads at a done, while the attempts are parked and no writer runs, so it
/// is sized by its own prompt, not a whole context, as the room sizes writers.
// ponytail: a hybrid model's state slots are not counted; the Mac's 8 hold k <= 4 and a reviewer
fn fits(
    ctx_max: u32,
    kv_pool_blocks: u32,
    fit_max_seqs: u32,
    workers: Workers,
    k: usize,
) -> Result<usize, &'static str> {
    let fit = usize::try_from(fit_max_seqs).unwrap_or(usize::MAX);
    if fit.max(workers.max_workers) <= k {
        return Err("Reviewer: off, config.toml's workers run no session beside the work");
    }
    let quarter = u64::from(ctx_max / 4);
    let cap = match u64::from(kv_pool_blocks) * KV_BLOCK_TOKENS {
        0 => quarter,
        pool => {
            let parked = u64::from(ctx_max).div_ceil(KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS * k as u64;
            let verdict = u64::from(VERDICT_TOKENS) + TEMPLATE_TOKENS;
            (pool.saturating_sub(parked + verdict) / KV_BLOCK_TOKENS * KV_BLOCK_TOKENS).min(quarter)
        }
    };
    match cap < u64::from(MIN_ROOM) {
        true => Err("Reviewer: off, the memory holds no session beside the work"),
        false => Ok(usize::try_from(cap).unwrap_or(usize::MAX)),
    }
}

/// The reason in a verdict `text` (`NOT_DONE`); None for ok, or a verdict with no reason.
fn reason(text: &str) -> Option<&str> {
    (text.trim().strip_prefix(NOT_DONE)).filter(|r| !r.trim().is_empty())
}

/// Whether `reply` says the change is done: no call, now or later, and no parse error. A plan's
/// writes are all later calls (`plan::Job::writes`), so a plan is never a done.
fn done(reply: &Parsed) -> bool {
    reply.call.is_none() && reply.later.is_empty() && reply.error.is_none()
}

/// Whether `e` is the stop of a done the reviewer may take back (`Beside::close`).
pub(super) fn held(e: &AgentEvent) -> bool {
    matches!(e, AgentEvent::Notice(n) if n.ends_with(NO_TESTS_DONE) || n.ends_with(NONE_NEAR_DONE))
}

impl Beside {
    /// What runs beside `k` attempts of `code` on `larch`, in `run`.
    pub(super) fn new(larch: &Larch, code: &Code, run: &Setup, k: usize) -> Beside {
        let d = larch.deployment();
        let fit = fits(d.ctx_max, d.kv_pool_blocks, d.fit_max_seqs, code.workers, k);
        let reviewer = fit.ok().map(|cap| {
            let params = GenParams {
                max_tokens: VERDICT_TOKENS,
                grammar: Some(Grammar {
                    trigger: None,
                    pattern: format!(r"(?:ok|{NOT_DONE}[^\n]+)"),
                }),
                ..ask_docs::params(Some(code.family), run.namespace)
            };
            (params, cap)
        });
        Beside {
            reviewer,
            off: fit.err(),
            jobs: (0..k).map(|_| None).collect(),
            base: vec![BTreeMap::new(); k],
        }
    }

    /// Starts the reviewer on attempt `i`'s change when its `reply` says it is done and only the
    /// model's word would end it: no tests are behind it, and the static review (`review` in
    /// loop.rs) would not send it back anyway. Its verdict comes while the done is taken
    /// (`close`). None when the reviewer is off (why is told at the first such done), cannot fit
    /// the changed lines in its prompt, or the engine refuses it: a reviewer never holds the
    /// worker.
    pub(super) fn open(
        &mut self,
        larch: &Larch,
        a: &Attempt,
        i: usize,
        reply: &Parsed,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Option<TextStream> {
        // Untested, after the static review's first say and where no build check passed;
        // tested, where no test is near; never in simple, whose done is unchecked. A check that
        // ran outranks a same-model judge with no ground truth.
        let unchecked = match &a.before {
            _ if a.task.verify == Verify::Simple => false,
            None => a.reviews > 0 && !a.check_passed(),
            Some(_) => a.task.none_near().unwrap_or(false),
        };
        if !done(reply) || !unchecked {
            return None;
        }
        let dir = a.task.dir().ok()?;
        let (found, broken) = findings(dir, &a.wrote, a.site);
        if (broken && a.reviews < REVIEWS) || !a.task.changed().unwrap_or(false) {
            return None;
        }
        let Some((p, cap)) = &self.reviewer else {
            if let Some(why) = self.off.take() {
                a.traced(serde_json::json!({ "reviewer": why }), sink);
                sink(AgentEvent::Progress(why.into()));
            }
            return None;
        };
        let tokens = |text: &str| larch.token_ids(text).len();
        let mut user = format!(
            "Request: {}\n\nIt said: {}\n",
            a.task.request,
            reply.text_before.trim()
        );
        if !found.is_empty() {
            let _ = writeln!(user, "\nThe checks found:\n{}", found.join("\n"));
        }
        // The whole prompt counts against the cap the pool was sized for: its fixed part first,
        // then the changed lines, then whole files while they fit.
        let room = cap
            .checked_sub(tokens(REVIEWER) + tokens(&user))
            .unwrap_or(0);
        let now: Vec<(&String, String)> = (a.wrote.iter())
            .map(|name| (name, fs::read_to_string(dir.join(name)).unwrap_or_default()))
            .collect();
        let hunks: Vec<String> = (now.iter())
            .filter_map(|(name, text)| {
                let was = self.base[i].get(*name).map_or("", String::as_str);
                changed(name, was, text, a.room.changed_tokens)
            })
            .collect();
        let hunks = hunks.join("\n\n");
        let mut left = room.checked_sub(tokens(&hunks)).or_else(|| {
            let why = "Reviewer: skipped, the change is too long to read beside the work";
            sink(AgentEvent::Progress(format!("{}{why}", a.label)));
            None
        })?;
        // `changed` cuts a long change's lines with this tail.
        let mut cut = hunks
            .lines()
            .any(|l| l.starts_with("... ") && l.ends_with(" more lines"));
        let _ = write!(user, "\nThe changed lines:\n{hunks}\n");
        for (name, text) in &now {
            let whole = format!("\n{name} whole:\n{text}\n");
            match left.checked_sub(tokens(&whole)) {
                Some(rest) => {
                    left = rest;
                    user += &whole;
                }
                None => cut = true,
            }
        }
        if cut {
            user += CUT;
        }
        let prompt = vec![message("system", REVIEWER), message("user", &user)];
        let stream = larch.generate(Prompt::Messages(prompt), p.clone()).ok()?;
        let line = "Reviewer: checking the change against the request";
        sink(AgentEvent::Progress(format!("{}{line}", a.label)));
        Some(stream)
    }

    /// Reads the verdict `review` opened on `a`, once the done was taken (`ok` when its calls did
    /// not fail), whose stop is `held`: a doubt joins the stop, then `held` goes to `sink`.
    pub(super) fn close(
        &mut self,
        review: Option<TextStream>,
        a: &mut Attempt,
        held: Option<AgentEvent>,
        ok: bool,
        sink: &mut dyn FnMut(AgentEvent),
        stop: &AtomicBool,
    ) {
        let label = a.label.clone();
        // No seat: the verdict is read only after the attempt's calls, whose writers may need
        // every seat, so a seat held here could wait on them forever.
        let review = review.map(Seated::admitted);
        let line = match review.filter(|_| ok).map(|r| workers::answer(0, r, stop)) {
            None => None,
            Some(Err(e)) => Some(format!("Reviewer: no verdict ({e})")),
            // A same-model judge with no ground truth: its doubt goes with the done, which stays
            // unverified, and never sends it back.
            Some(Ok(answer)) => match reason(&answer.text).map(str::trim) {
                Some(why) => {
                    match a.stopped.as_mut() {
                        Some(stop) => *stop += &format!("; a reviewer doubts it: {why}"),
                        // Sent back already, by a failing build check: the doubt joins that
                        // result.
                        None => {
                            if let Some(step) = a.steps.last_mut() {
                                step.1 += &format!("\na reviewer read the change: {why}");
                            }
                        }
                    }
                    Some(format!("Reviewer: not done: {why}"))
                }
                None => Some("Reviewer: the change does what was asked".into()),
            },
        };
        if let Some(line) = line {
            a.traced(serde_json::json!({ "reviewer": line }), sink);
            sink(AgentEvent::Progress(format!("{label}{line}")));
        }
        held.into_iter().for_each(sink);
    }

    /// Starts attempt `i`'s tests in the background when the task has them, its steps from
    /// `from` changed a file and it goes on, while the model writes its next step; never in
    /// simple mode, which runs only what the model calls.
    pub(super) fn start(
        &mut self,
        a: &Attempt,
        i: usize,
        from: usize,
        sink: &mut dyn FnMut(AgentEvent),
    ) {
        let wrote = (a.steps[from..].iter()).any(|s| matches!(s.2, Evidence::Wrote(_)));
        if !wrote
            || a.stopped.is_some()
            || a.task.test_cmd.is_none()
            || a.task.verify == Verify::Simple
        {
            return;
        }
        let line = "Tests running in the background";
        sink(AgentEvent::Progress(format!("{}{line}", a.label)));
        let task = a.task.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || tests(&task, &flag));
        self.jobs[i] = Some(Job { handle, stop });
    }

    /// Before attempt `i`'s `reply` runs, whose writes find their files as `was` says: what its
    /// tests in the background said, for the reply's result, when they ended and the reply
    /// changes nothing. A step never waits for them: ones still running go on beside a reply
    /// that only reads, and stop for one that writes or runs something, which makes them stale.
    /// A done reads their tests, run on the folder as it is, instead of running them again.
    pub(super) fn join(
        &mut self,
        a: &mut Attempt,
        i: usize,
        reply: &Parsed,
        was: &BTreeMap<String, String>,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> String {
        for (path, text) in was {
            self.base[i]
                .entry(path.clone())
                .or_insert_with(|| text.clone());
        }
        let Some(job) = self.jobs[i].take() else {
            return String::new();
        };
        let done = done(reply);
        let quiet = (reply.call.iter().chain(&reply.later)).all(|c| QUIET.contains(&&*c.name));
        let label = &a.label;
        if !job.handle.is_finished() {
            if quiet && !done {
                self.jobs[i] = Some(job);
            } else {
                job.end();
                // A write outdates them, and its own run is told; a done runs them again.
                if done {
                    let line = "Tests in the background stopped: running them again for the done";
                    sink(AgentEvent::Progress(format!("{label}{line}")));
                }
            }
            return String::new();
        }
        let (seen, line, tests) = job
            .handle
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p));
        if !quiet {
            return String::new();
        }
        sink(AgentEvent::Progress(format!(
            "{label}In the background: {line}"
        )));
        match done {
            true => {
                a.fresh = tests;
                String::new()
            }
            false => match tests {
                // A run that could not be compared says nothing of this state.
                Some((run, shown)) if !uncompared(&shown) => {
                    let before = a.before.clone();
                    let passed = a.record(&run, before.as_ref());
                    seen + a.passes(passed)
                }
                _ => seen,
            },
        }
    }
}

/// Every test run still going stops before the attempts' packets, which run the tests again and
/// commit, or before a failed run removes their worktrees.
impl Drop for Beside {
    fn drop(&mut self) {
        self.jobs
            .iter_mut()
            .filter_map(Option::take)
            .for_each(Job::end);
    }
}

/// `task`'s tests after a change, which `stop` ends.
fn tests(task: &CodeTask, stop: &Arc<AtomicBool>) -> Found {
    match task.test_until(Some(stop), false) {
        Ok((run, shown)) => {
            let line = verdict(&shown).map_or_else(|| "tests did not run".into(), |v| v.1);
            let seen = format!("\nthe tests after your previous change: {shown}");
            (seen, line, Some((run, shown)))
        }
        Err(e) => {
            let seen = format!("\nthe tests could not run after your previous change: {e}");
            (seen, "tests could not run".into(), None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKERS: Workers = Workers {
        max_workers: 8,
        prefix_cache: true,
    };

    const ONE: Workers = Workers {
        max_workers: 1,
        prefix_cache: false,
    };

    const MEMORY: Result<usize, &str> =
        Err("Reviewer: off, the memory holds no session beside the work");

    /// A pool that holds exactly the attempts' contexts leaves no room for the reviewer, and
    /// says it is the memory; one worker beside one attempt says it is the workers.
    #[test]
    fn the_reviewer_is_off_when_the_pool_fits_only_the_attempts() {
        let workers = Err("Reviewer: off, config.toml's workers run no session beside the work");
        assert_eq!(fits(32_768, 32_768 / 16, 8, WORKERS, 1), MEMORY);
        assert_eq!(fits(131_072, 131_072 / 16, 1, WORKERS, 1), MEMORY);
        assert_eq!(fits(8192, 1024, 2, WORKERS, 2), MEMORY);
        assert_eq!(fits(8192, 512, 1, ONE, 1), workers);
    }

    /// The Mac's 32k context in a pool of one and a half contexts, one attempt: the reviewer's
    /// prompt and verdict fit the half beside the attempt, at a quarter of the context; a pool
    /// with less to spare shrinks the prompt rather than turning the reviewer off.
    #[test]
    fn a_reviewer_fits_beside_one_attempt_in_a_pool_of_one_and_a_half() {
        assert_eq!(fits(32_768, 3 * 32_768 / 32, 8, WORKERS, 1), Ok(8192));
        let tight = fits(32_768, (32_768 + 2048) / 16, 8, WORKERS, 1);
        assert!(
            tight.is_ok_and(|cap| (256..2048).contains(&cap)),
            "{tight:?}"
        );
    }

    /// A laptop's 8k context in a pool of two, a big machine's 128k in a pool of two, eight
    /// attempts in a pool of sixteen, one worker in config.toml with memory to spare, and a pool
    /// not known keep the reviewer on at a quarter of the context.
    #[test]
    fn a_big_deployment_keeps_the_reviewer_on() {
        assert_eq!(fits(8192, 1024, 2, WORKERS, 1), Ok(2048));
        assert_eq!(fits(131_072, 2 * 131_072 / 16, 2, WORKERS, 1), Ok(32_768));
        assert_eq!(fits(131_072, 16 * 131_072 / 16, 16, WORKERS, 8), Ok(32_768));
        assert_eq!(fits(32_768, 16 * 32_768 / 16, 16, ONE, 1), Ok(8192));
        assert_eq!(fits(131_072, 0, 16, WORKERS, 4), Ok(32_768));
    }

    /// A plan's writes, all later calls with no call now, are work, not a done: the reviewer
    /// never opens on them and the tests in the background stop for them.
    #[test]
    fn a_plans_writes_are_not_a_done() {
        let write = crate::parse::ToolCall {
            name: "write_file".into(),
            args: serde_json::json!({"path": "a.js", "about": "the game"}),
        };
        let writes = Parsed {
            later: vec![write],
            ..Parsed::default()
        };
        assert!(!done(&writes));
        assert!(done(&Parsed::default()));
    }

    /// A verdict reads as not done only with a reason.
    #[test]
    fn a_verdict_reads_its_reason() {
        let why = "the reset button does nothing";
        assert_eq!(reason(&format!("{NOT_DONE}{why}\n")), Some(why));
        assert_eq!(reason("ok"), None);
        assert_eq!(reason("not done: "), None);
        assert_eq!(reason(""), None);
    }
}
