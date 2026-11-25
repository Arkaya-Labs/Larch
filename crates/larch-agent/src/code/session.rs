//! Code mode's step loop: one engine session per attempt whose transcript (the system prompt with
//! the repository map, the request, then each reply and what its calls returned) is only
//! appended to. Each step's prompt is the last one's, its reply and the new result, so the engine
//! reuses the KV the session parked and prefills only what is new; the model reads every
//! observation whole, as it came, instead of a view folded or gone stale.
//! A reply that is dropped (cut, unparsable, refused by its tool's schema) is never appended:
//! the next step sends the same prompt again, which the parked KV holds, held to a call; one
//! dropped while held is told why, so the greedy ask after it differs. Once the next prompt would
//! pass four fifths of the context, the transcript is compacted: the older steps folded to a line,
//! the files in play shown as they are now, in a fresh session. A model whose template does not
//! render a transcript's history as the start of the next prompt runs the same loop: the engine
//! reuses only the prefix that matches and reads the rest again.

use regex::Regex;
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    sync::{Arc, atomic::AtomicBool},
};

use larch::{EngineError, Larch, Session, SessionOptions};
use larch_model::Message;

use crate::{
    audit::Audit,
    code::{
        allow::{Ask, Decision},
        experts::{self, Experts},
        r#loop::{
            ANSWER_NOW, Attempt, CUT, Code, Evidence, edit_path, inner, live, message, messages,
            packets, relative, send, writes,
        },
        plan,
        review::{self, Beside},
        task::Packet,
    },
    parse::{Family, Parsed},
    run::{self, AgentEvent, Checked},
    tool::{self, ToolError},
    tools::code::numbered,
};

/// Lines shown on each side of a change (`changed`).
const AROUND: usize = 10;

/// The part of the context, in fifths, a prompt may reach before its transcript is compacted.
const COMPACT_FIFTHS: u32 = 4;

/// The part of the context, in fifths, a compacted transcript's prompt fills at least when the
/// first prompt leaves room (`kept`): the steps after it have the two fifths up to
/// `COMPACT_FIFTHS` before the next compaction.
// ponytail: a fixed share; tune it when the audit shows runs that compact more than once
const COMPACTED_FIFTHS: u32 = 2;

/// The prompt tokens within which a compaction keeps its last step whole, on a `ctx` context
/// whose first prompt took `base`: half of what the compaction point leaves above `base`, or
/// the `COMPACTED_FIFTHS` the files in play fill when that is more. Measured above `base`, so a
/// small context's system prompt and map never leave the last step no room.
pub(super) fn kept(ctx: u32, base: u32) -> u32 {
    let at = ctx / 5 * COMPACT_FIFTHS;
    (ctx / 5 * COMPACTED_FIFTHS).max(base + at.saturating_sub(base) / 2)
}

/// Replies dropped one after the other that stop an attempt: held to a call, a model that still
/// gives none will not.
const DROPS: u32 = 3;

/// One attempt's conversation with the model.
// ponytail: follow-ups start cold (idea G): parking a Talk between a chat's runs needs the app to
// hold it across `code_start` calls and the run's worktree or checkpoint to match it; add when a
// Mac follow-up's first-token time shows the full re-read
struct Talk {
    session: Session,
    /// The transcript, which each step sends whole and only appends to.
    messages: Vec<Message>,
    /// Replies dropped since the last one kept: the next one is held to a call.
    drops: u32,
    /// Compacted since its last step: a prompt still too long cannot be compacted again.
    compacted: bool,
    /// The first step's prompt tokens: the system prompt, its tools and the request.
    base: u32,
    /// The next reply is held to a plan (`plan::new_project`); only the first one is.
    plan: bool,
}

impl Talk {
    /// A conversation on `system` that starts with `user`, in a fresh session.
    fn new(larch: &Larch, system: &str, user: &str) -> Talk {
        Talk {
            session: larch.session(SessionOptions::default()),
            messages: messages(system, user),
            drops: 0,
            compacted: false,
            base: 0,
            plan: false,
        }
    }
}

/// Free mode: every attempt's steps until each stops, then each one's packet. The index and
/// packet of the attempt the tests pick, the replies and how many parsed.
pub(super) fn free(
    code: &Code,
    larch: &Larch,
    attempts: &mut [Attempt],
    audit: &Audit,
    sink: &mut dyn FnMut(AgentEvent),
    ask: &mut dyn FnMut(&Ask) -> Decision,
    stop: &Arc<AtomicBool>,
) -> Result<(usize, Packet, u32, u32), ToolError> {
    let ctx = larch.deployment().ctx_max;
    plan::offer(attempts, code.thinking.answer_max_tokens);
    let mut run = code.setup(larch, attempts, stop)?;
    let registry = &attempts[0].registry;
    let planned = plan::held(registry, code.family, code.xml_body, &run.p)?;
    // The reply that ends an attempt stops at a call's marker, so it is words, as a chat turn
    // without tools stops (`Agent::no_calls`).
    let mut ending = run.p.clone();
    ending.stop.push(code.family.marker().into());
    let mut talks = Vec::new();
    for a in attempts.iter() {
        let talk = Talk::new(larch, &run.system, &a.prompt(None));
        let plan = plan::new_project(a)?;
        talks.push(Talk { plan, ..talk });
    }
    let mut experts = Experts::new(code, larch, &attempts[0], &run, attempts.len(), stop)?;
    if let Some(experts) = &experts {
        attempts.iter_mut().for_each(|a| experts.offer(a));
    }
    let mut beside = Beside::new(larch, code, &run, attempts.len());
    let (mut steps, mut parsed) = (0, 0);
    loop {
        let live = live(attempts, &run, stop, sink);
        if live.is_empty() {
            break;
        }
        code.offer(&mut run, attempts)?;
        // One run per grammar, so with k at 1 a step is one run.
        let mut replies = Vec::new();
        for (hold, params) in [(0, &run.p), (1, &run.forced), (2, &planned), (3, &ending)] {
            let held = |i: usize| match (attempts[i].ending.is_some(), talks[i].plan) {
                (true, _) => 3,
                (_, true) => 2,
                _ => u8::from(attempts[i].force || talks[i].drops > 0),
            };
            let group: Vec<usize> = live.iter().copied().filter(|&i| held(i) == hold).collect();
            if group.is_empty() {
                continue;
            }
            let prompts: Vec<_> = (group.iter())
                .map(|&i| vec![talks[i].messages.clone()])
                .collect();
            let mut sessions: Vec<&mut Session> = (talks.iter_mut().enumerate())
                .filter(|(i, _)| group.contains(i))
                .map(|(_, t)| &mut t.session)
                .collect();
            let ran = send(larch, &code.workers, &mut sessions, &prompts, params, stop);
            replies.extend(group.into_iter().zip(ran));
        }
        for (i, answer) in replies {
            let (a, talk) = (&mut attempts[i], &mut talks[i]);
            // A plan that fails is not asked for again: the next step is free.
            talk.plan = false;
            let forced = a.force || talk.drops > 0;
            let answer = match answer {
                // The 80% check estimates the next prompt; one it missed is compacted now. A
                // first prompt too long has nothing to compact: the engine's error says so.
                Err(EngineError::PromptTooLong { .. })
                    if !talk.compacted && !a.steps.is_empty() =>
                {
                    compact(larch, a, talk, &run.system, ctx, sink);
                    continue;
                }
                answer => answer,
            };
            let Some(answer) = a.answered(answer, stop, sink) else {
                continue;
            };
            steps += 1;
            if talk.base == 0 {
                talk.base = answer.usage.prompt_tokens;
            }
            let (reply, cut, ok) = code.read(a, i, &answer, audit, sink)?;
            parsed += u32::from(ok);
            if let Some(why) = dropped(a, &reply, cut) {
                talk.drops += 1;
                // An edit cut or broken before its call parsed failed as one refused does, which
                // steers the file to write_file (`Attempt::failed_edit`).
                let steer = match reply.call.is_none().then(|| edit_path(&answer.text)) {
                    Some(Some(path)) => a.failed_edit(inner(&a.task.repo, &path))?,
                    _ => String::new(),
                };
                let note = format!(
                    "{}the reply was dropped and asked for again: {why}",
                    a.label
                );
                sink(AgentEvent::Notice(note));
                if talk.drops >= DROPS {
                    a.stop(
                        format!("{DROPS} replies in a row were dropped: {why}"),
                        sink,
                    );
                } else if forced {
                    // Held to a call, greedy, on the same prompt, the model would say it again:
                    // it is told why instead.
                    let said = if cut {
                        "(no call: cut)"
                    } else {
                        "(no valid call)"
                    };
                    let seen = format!("{why}{steer}");
                    talk.messages.push(message("assistant", said));
                    talk.messages.push(message("user", &seen));
                    a.steps.push((said.into(), seen, Evidence::Other));
                }
                a.spent(&run.budget, sink);
                continue;
            }
            talk.drops = 0;
            if a.last_words(&reply, sink) {
                continue;
            }
            let job = plan::job(a, &reply, &run.room);
            let writes = job.as_ref().map(plan::Job::writes);
            let (n, was) = (a.steps.len(), texts(a, writes.as_ref().unwrap_or(&reply)));
            let said = unnoted(code.family, &answer.text, &reply);
            // A plan's writes are the step's: the reviewer and the tests in the background see
            // its files' earlier text, as a reply's own writes.
            let shown = writes.as_ref().unwrap_or(&reply);
            let ready = beside.join(a, i, shown, &was, sink);
            let review = beside.open(larch, a, i, shown, sink);
            // Only the stop the reviewer may take back waits for its verdict.
            let mut held = None;
            let mut hold = |e| match review.is_some() && review::held(&e) {
                true => held = Some(e),
                false => sink(e),
            };
            // An ask runs its experts first; their places are then its call's result, and when
            // none holds, a lookup of the question runs after it, as the single session's would.
            let then = (experts.as_mut().zip(experts::asked(a, &reply)))
                .and_then(|(e, question)| e.ask(code, larch, a, &question, &mut hold, stop));
            let called = match job {
                Some(job) => plan::fan_out(
                    code, larch, a, job, &answer, &run, audit, &mut hold, ask, stop,
                ),
                None => {
                    let asked = code.calls(
                        larch, a, reply, true, &answer, &run, audit, &mut hold, ask, stop,
                    );
                    match then.filter(|_| a.stopped.is_none()) {
                        Some(lookup) if asked.is_ok() => code.calls(
                            larch, a, lookup, true, &answer, &run, audit, &mut hold, ask, stop,
                        ),
                        _ => asked,
                    }
                }
            };
            beside.close(review, a, held, called.is_ok(), sink, stop);
            called?;
            if a.steps.len() > n {
                // The tests in the background print paths too.
                let seen = observation(a, n, &was) + &relative(a.task.dir()?, &ready);
                let next = answer.usage.prompt_tokens
                    + answer.usage.completion_tokens
                    + larch.token_ids(&seen).len() as u32;
                talk.messages.push(message("assistant", &said));
                talk.messages.push(message("user", &seen));
                talk.compacted = false;
                // The next reply's room counts too: a thinking budget may need more than a fifth.
                let room = run.p.max_tokens.max(run.forced.max_tokens);
                let full = next >= ctx / 5 * COMPACT_FIFTHS || next + room >= ctx;
                if a.stopped.is_none() && full {
                    compact(larch, a, talk, &run.system, ctx, sink);
                }
            }
            a.spent(&run.budget, sink);
            beside.start(a, i, n, sink);
        }
    }
    // No check runs beside the packets' tests and commit.
    drop(beside);
    let (best, packet) = packets(attempts, stop)?;
    Ok((best, packet, steps, parsed))
}

/// `text` without its calls' notes, which are the user's (`Registry::noted`): kept, they would
/// tell the model nothing, as `shown` says.
/// The text as said when taking them out would change what the reply calls.
fn unnoted(family: Family, text: &str, reply: &Parsed) -> String {
    let string = r#""(?:[^"\\]|\\.)*""#;
    let note = format!(
        r#"(?s)<parameter=note>.*?</parameter>\n?|\s*,\s*"note"\s*:\s*{string}|"note"\s*:\s*{string}\s*,?\s*"#
    );
    let out = Regex::new(&note)
        .expect("a valid regex")
        .replace_all(text, "");
    let bare = |r: &Parsed| {
        let mut calls: Vec<_> = r.call.iter().chain(&r.later).cloned().collect();
        for c in &mut calls {
            if let Some(m) = c.args.as_object_mut() {
                m.remove("note");
            }
        }
        (calls, r.error.clone())
    };
    match bare(&run::read(Some(family), &out, false)) == bare(reply) {
        true => out.into_owned(),
        false => text.to_owned(),
    }
}

/// Why `reply` is dropped instead of appended: its token cap `cut` it before a call, it did not
/// parse, or its call fails its tool's schema. None for a reply that is kept.
fn dropped(a: &Attempt, reply: &Parsed, cut: bool) -> Option<String> {
    if cut {
        return Some(CUT.into());
    }
    match run::check(reply, &a.registry, &a.tools) {
        Checked::ParseError(e) => Some(e.to_string()),
        Checked::Invalid(_, e) => Some(e.to_string()),
        _ => None,
    }
}

/// The text of each file `reply`'s writing calls (`writes`) name, relative to the attempt's folder, as
/// it is before they run: empty for a file not there.
fn texts(a: &Attempt, reply: &Parsed) -> BTreeMap<String, String> {
    let Ok(dir) = a.task.dir() else {
        return BTreeMap::new();
    };
    (reply.call.iter().chain(&reply.later))
        .filter(|c| writes(&c.name))
        .filter_map(|c| c.args["path"].as_str())
        .map(|p| inner(&a.task.repo, p))
        .map(|p| {
            let text = fs::read_to_string(dir.join(&p)).unwrap_or_default();
            (p, text)
        })
        .collect()
}

/// What the model is told after a reply: what each of its calls returned since step `n`, then
/// the lines each changed file has now around its change (`changed`), from the texts `was`
/// before the calls; each write's result ends with what it leaves broken (`loop.rs`'s `findings_tail`).
fn observation(a: &Attempt, n: usize, was: &BTreeMap<String, String>) -> String {
    let seen: Vec<&str> = a.steps[n..].iter().map(|s| s.1.as_str()).collect();
    let mut out = seen.join("\n\n");
    let Ok(dir) = a.task.dir() else {
        return out;
    };
    for path in a.steps[n..].iter().filter_map(|s| match &s.2 {
        Evidence::Wrote(path) => Some(path),
        _ => None,
    }) {
        let now = fs::read_to_string(dir.join(path)).unwrap_or_default();
        let before = was.get(path).map_or("", String::as_str);
        if let Some(lines) = changed(path, before, &now, a.room.changed_tokens) {
            let _ = write!(out, "\n{lines}");
        }
    }
    out
}

/// The lines of `now`, the file at `path`, that differ from `before`, numbered, with `AROUND`
/// lines on each side, up to `tokens` (`Room::changed_tokens`): the model's next edit copies
/// from these, not from a view the change made stale. None when nothing changed.
pub(super) fn changed(path: &str, before: &str, now: &str, tokens: u32) -> Option<String> {
    let (old, new): (Vec<&str>, Vec<&str>) = (before.lines().collect(), now.lines().collect());
    if old == new {
        return None;
    }
    if new.is_empty() {
        return Some(format!("{path} is empty now"));
    }
    let head = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let tail = (old[head..].iter().rev())
        .zip(new[head..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let from = head.saturating_sub(AROUND);
    let to = (new.len() - tail + AROUND).min(new.len());
    let lines = numbered(new[from..to].iter().copied(), from + 1).join("\n");
    Some(format!(
        "{path} now, lines {}-{to} of {}:\n{}",
        from + 1,
        new.len(),
        tool::truncate(lines, tokens)
    ))
}

/// Starts `talk` again in a fresh session, one prompt read once: the request, the steps so far
/// folded to a line each but the last, whole within `kept`'s share of the `ctx` context, then
/// the other files in play (the ones the attempt viewed or wrote, last used first) as they are
/// now, up to `kept` too: each whole, or else the parts its last views showed that fit. A last
/// step past `kept` loses its last lines, never the whole: folded, the view the model just asked
/// for is asked for again, and folded again at the next compaction. The second compaction with
/// nothing new since the first (`Attempt::new_at`) makes the next reply the attempt's last
/// (`Attempt::end`): compacting again would only loop.
fn compact(
    larch: &Larch,
    a: &mut Attempt,
    talk: &mut Talk,
    system: &str,
    ctx: u32,
    sink: &mut dyn FnMut(AgentEvent),
) {
    let keep = kept(ctx, talk.base) as usize;
    let tokens = |text: &str| talk.base as usize + larch.token_ids(text).len();
    let seen = a.steps.last().map(|s| s.1.as_str());
    let lines: Vec<&str> = seen.unwrap_or_default().lines().collect();
    let whole = |n: usize| seen.map(|seen| head(seen, &lines, n));
    let fits = |n: &usize| tokens(&a.prompt(whole(*n).as_deref())) <= keep;
    let n = match fits(&lines.len()) {
        true => lines.len(),
        false => ((0..lines.len()).collect::<Vec<_>>().partition_point(fits)).saturating_sub(1),
    };
    let mut user = a.prompt(whole(n).as_deref());
    let mut room = keep.saturating_sub(tokens(&user));
    let dir = a.task.dir().ok();
    let last = a.steps.len();
    // Last used first; the last step's own view is whole above already.
    let mut shown: Vec<&String> = Vec::new();
    for (i, step) in a.steps.iter().enumerate().rev() {
        let (Evidence::View(name) | Evidence::Wrote(name)) = &step.2 else {
            continue;
        };
        if shown.contains(&name) || (i + 1 == last && matches!(step.2, Evidence::View(_))) {
            continue;
        }
        shown.push(name);
        let Some(text) = dir.and_then(|d| fs::read_to_string(d.join(name)).ok()) else {
            continue;
        };
        let all: Vec<&str> = text.lines().collect();
        let lines = numbered(all.iter().copied(), 1).join("\n");
        let n = larch.token_ids(&lines).len();
        if n <= room {
            room -= n;
            let _ = write!(user, "\n{name} as it is now:\n{lines}\n");
            continue;
        }
        // Too long whole: the lines its views showed, newest first, as they are now.
        let mut parts: Vec<(usize, usize)> = Vec::new();
        for (_, seen, _) in (a.steps[..last - 1].iter().rev())
            .filter(|s| matches!(&s.2, Evidence::View(v) if v == name))
        {
            let Some((from, to)) = span(seen) else {
                continue;
            };
            let to = to.min(all.len());
            if from > to || parts.contains(&(from, to)) {
                continue;
            }
            parts.push((from, to));
            let part = numbered(all[from - 1..to].iter().copied(), from).join("\n");
            let n = larch.token_ids(&part).len();
            if n <= room {
                room -= n;
                let _ = write!(
                    user,
                    "\n{name} lines {from}-{to} as they are now:\n{part}\n"
                );
            }
        }
    }
    // One that freed no room is followed by another at once, so this ends it too; with a write
    // or a new read in between, each compaction is the room for the next step.
    let full = a.compacted > 0 && a.new_at <= a.compacted;
    if full {
        let why = "the context filled twice with nothing new in between; ended with the model's \
                   answer";
        a.end(why.into());
    }
    // An ending set before (`STALE`) wrote ANSWER_NOW at its step's end, which `head` may cut.
    if a.ending.is_some() && !user.contains(ANSWER_NOW) {
        let full = if full { "The context is full. " } else { "" };
        let _ = write!(user, "\n{full}{ANSWER_NOW}\n");
    }
    a.compacted = last;
    *talk = Talk {
        compacted: true,
        base: talk.base,
        ..Talk::new(larch, system, &user)
    };
    let why = format!(
        "{}the transcript neared the context's end: compacted",
        a.label
    );
    sink(AgentEvent::Notice(why));
}

/// The first and last line numbers of a view's result (`numbered`); None for no numbered line.
fn span(seen: &str) -> Option<(usize, usize)> {
    let number = |l: &str| (l.split_once(": ")).and_then(|(k, _)| k.trim().parse::<usize>().ok());
    let mut numbers = seen.lines().filter_map(number);
    let first = numbers.next()?;
    Some((first.max(1), numbers.next_back().unwrap_or(first)))
}

/// `seen`, a step's result of `lines`, whole or cut to `n` lines: a view's first ones and where
/// a view of the rest starts, when its lines are numbered (`numbered`); any other result's
/// first and last halves, since a test run or a command ends on its failure.
fn head(seen: &str, lines: &[&str], n: usize) -> String {
    if n >= lines.len() {
        return seen.into();
    }
    let next = (lines[n].split_once(": ")).and_then(|(k, _)| k.trim().parse::<usize>().ok());
    let (rest, end) = match next {
        Some(k) => (
            format!("... more lines cut for room; view from start {k}"),
            n,
        ),
        None => ("... more lines cut for room".into(), n / 2),
    };
    let tail = &lines[lines.len() - (n - end)..];
    let kept: Vec<&str> = (lines[..end].iter().copied())
        .chain([rest.as_str()])
        .chain(tail.iter().copied())
        .collect();
    kept.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A change shows its lines with `AROUND` on each side, numbered as they are now; a new file
    /// shows from its start, and no change shows nothing.
    #[test]
    fn a_change_shows_its_lines_as_they_are_now() {
        let file: String = (1..=30).map(|n| format!("line {n}\n")).collect();
        let now = file.replace("line 15\n", "line 15 changed\nline 15b\n");
        let shown = changed("f.txt", &file, &now, 1000).unwrap();
        assert!(
            shown.starts_with("f.txt now, lines 5-26 of 31:\n5: line 5\n"),
            "{shown}"
        );
        assert!(shown.contains("\n15: line 15 changed\n16: line 15b\n17: line 16\n"));
        assert!(shown.ends_with("\n26: line 25"), "{shown}");
        let new = changed("n.txt", "", "a\nb\n", 1000).unwrap();
        assert_eq!(new, "n.txt now, lines 1-2 of 2:\n1: a\n2: b");
        assert_eq!(changed("f.txt", &file, &file, 1000), None);
        assert_eq!(
            changed("f.txt", &file, "", 1000).as_deref(),
            Some("f.txt is empty now")
        );
    }

    /// From 32k a compaction keeps what it kept before: the files in play fill two fifths, and
    /// the last step whole up to half the room past the first prompt.
    #[test]
    fn a_big_context_compacts_as_before() {
        assert_eq!(32_768 / 5 * COMPACTED_FIFTHS, 13_106);
        assert_eq!(kept(32_768, 5_000), 15_606);
        assert_eq!(kept(131_072, 5_000), 54_928);
        assert_eq!(kept(131_072, 60_000), 82_428);
    }

    /// A view cut for room keeps its first lines and says where a view of the rest starts; any
    /// other result keeps its start and its end.
    #[test]
    fn a_cut_step_says_where_the_rest_starts() {
        let seen = "1: a\n2: b\n3: c";
        let lines: Vec<&str> = seen.lines().collect();
        assert_eq!(head(seen, &lines, 3), seen);
        let cut = "1: a\n... more lines cut for room; view from start 2";
        assert_eq!(head(seen, &lines, 1), cut);
        let out = ["running", "ok", "ok", "FAILED a"];
        assert_eq!(
            head(&out.join("\n"), &out, 2),
            "running\n... more lines cut for room\nFAILED a"
        );
    }

    /// A note comes out of either call format; a text that would read as another call stays.
    #[test]
    fn unnoted_takes_out_the_note_alone() {
        let read = |t: &str| run::read(Some(Family::Qwen3), t, false);
        let json = "<tool_call>\n{\"name\": \"view\", \"arguments\": {\"path\": \"a.py\", \"note\": \"why \\\"x\\\"\"}}\n</tool_call>";
        let bare =
            "<tool_call>\n{\"name\": \"view\", \"arguments\": {\"path\": \"a.py\"}}\n</tool_call>";
        assert_eq!(unnoted(Family::Qwen3, json, &read(json)), bare);
        let xml = "<tool_call>\n<function=view>\n<parameter=path>\na.py\n</parameter>\n<parameter=note>\nwhy\n</parameter>\n</function>\n</tool_call>";
        let bare = "<tool_call>\n<function=view>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(unnoted(Family::Qwen3, xml, &read(xml)), bare);
        // A note inside a written text is the text's, not the call's.
        let text = r#"<tool_call>
{"name": "write_file", "arguments": {"path": "a.json", "text": "{\"note\": \"x\"}"}}
</tool_call>"#;
        assert_eq!(unnoted(Family::Qwen3, text, &read(text)), text);
    }
}
