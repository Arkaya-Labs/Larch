//! Fan-out by plan: the session's model splits work on several files into one job per file with
//! the `plan` call, one grammar-held reply that names each file's path and what it holds or what
//! changes in it, and the contract the files share (the names, ids and signatures that cross
//! them). Each file is then written by the writer (`body`), `Room::writers` at a time, every brief
//! carrying the contract: a new file from its about, a file there from its text as it is now and
//! the change. A new project's first step is held to a plan, so its files fan out by plan, not by
//! a reply that happens to make several write_file calls. After the files, the cross-file checks
//! run; a file left unchanged or a check that fails falls back to the session, which is told what
//! is left and makes the rest of the change itself. An attempt plans once. An editor's text
//! goes through write_file's checks as any rewrite does, `tools::code::dropped` among them: one
//! that drops what its change does not name is refused, and its file goes to the session.
// ponytail: an editor rewrites its file whole through the writer, so a file longer than
// `Room::rewrite_max_lines` or the writer's brief goes to the session; give editors a session with
// edit calls when a big repo's plans show that fallback often

use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use larch::{GenParams, Larch, toolcall::Mode as Hold};
use serde_json::{Value, json};

use crate::{
    audit::Audit,
    code::{
        allow::{Ask, Decision},
        checks::{self, Finding},
        r#loop::{Attempt, Code, Evidence, SAID_NO, Setup, inner},
        room::Room,
    },
    parse::{Family, Parsed, ToolCall},
    run::{self, AgentEvent, Checked},
    tool::{Permission, Registry, Tool, ToolCtx, ToolError, ToolSpec},
    tools::files::prepare_deep,
    workers::Answer,
};

/// The plan call's name.
pub(super) const NAME: &str = "plan";

/// The plan call, which the session loop runs itself (`fan_out`); the registry only gives its
/// schema to the grammar and the checks, and runs a plan that names no file, which fails.
struct Plan {
    /// The tokens its arguments may take: the step's whole reply room (`offer`), so a plan is
    /// cut only where any call would be, and offering it raises no step's cap.
    tokens: u32,
}

impl Tool for Plan {
    fn spec(&self) -> ToolSpec {
        let file = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "about": {"type": "string", "description":
                    "For a new file, everything it holds; for a file there, what changes in it"}
            },
            "required": ["path", "about"]
        });
        ToolSpec {
            name: NAME.into(),
            description: "Splits work on two or more files into one job per file, all done at \
                          the same time: first the contract the files share (the names, ids, \
                          classes and function signatures that cross files), then each file's \
                          path and about."
                .into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "contract": {"type": "string", "description":
                        "The names, ids, classes and function signatures every file uses exactly"},
                    "files": {"type": "array", "items": file}
                },
                "required": ["contract", "files"],
                "additionalProperties": false
            }),
            permission: Permission::Read,
            untrusted_results: false,
            max_result_tokens: 100,
            max_args_tokens: self.tokens,
        }
    }

    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        Err(ToolError::Invalid(
            "a plan names one file or more, each with its path and about".into(),
        ))
    }
}

/// Gives each of `attempts`' registries the plan call, which the session loop's tools offer
/// (`Code::setup`): its arguments get the room a step's reply has, the longest call's or
/// config.toml's `answer` cap, whichever is more (`run::params`).
// ponytail: a plan of ~30 files with real abouts passes the 2048 tokens of the longest call; size
// it from the room's free context when an eval shows big-site plans cut
pub(super) fn offer(attempts: &mut [Attempt], answer: u32) {
    for a in attempts {
        let specs = a.registry.specs(&a.tools);
        let call = specs.iter().map(|s| s.max_args_tokens).max().unwrap_or(0);
        let tokens = call.max(answer);
        a.registry.replace(Box::new(Plan { tokens }));
    }
}

/// Whether `a`'s first step is held to a plan: its folder has no file yet, or it builds a new
/// website, whose page, stylesheet and script are several files; never a read-only run.
pub(super) fn new_project(a: &Attempt) -> Result<bool, ToolError> {
    Ok(!a.task.read_only && (a.site || a.task.files()?.is_empty()))
}

/// `p` held to one plan call of `registry`'s, in `family`'s format (the XML body when `xml`);
/// with thinking, the call follows the think block.
pub(super) fn held(
    registry: &Registry,
    family: Family,
    xml: bool,
    p: &GenParams,
) -> Result<GenParams, ToolError> {
    let mut grammar = run::grammar(registry, &[NAME], Some(family), xml, Hold::Required)
        .map_err(|e| ToolError::Invalid(format!("the plan grammar: {e}")))?;
    if let Some(g) = grammar.as_mut().filter(|_| p.thinking_budget.is_some()) {
        g.trigger = Some("</think>".into());
    }
    Ok(GenParams {
        grammar,
        ..p.clone()
    })
}

/// A plan reply's work.
pub(super) struct Job {
    /// A write_file call for each file the writer takes.
    calls: Vec<ToolCall>,
    /// The planned files left to the session, relative to the folder: a path the write refuses,
    /// or a file there too long to rewrite whole.
    left: Vec<String>,
    /// The reply made calls after the plan, which do not run.
    more: bool,
}

impl Job {
    /// Its write_file calls as one reply.
    pub(super) fn writes(&self) -> Parsed {
        Parsed {
            later: self.calls.clone(),
            ..Parsed::default()
        }
    }

    /// The calls `writers` at a time, each group one reply whose files are written side by side.
    fn waves(&self, writers: usize) -> Vec<Parsed> {
        (self.calls.chunks(writers.max(1)))
            .map(|c| Parsed {
                call: Some(c[0].clone()),
                later: c[1..].to_vec(),
                ..Parsed::default()
            })
            .collect()
    }
}

/// The work of `reply` when it is a valid plan of `a`'s (its arguments as the schema types
/// them, so the XML body's list is one) that names a file. A file there is written from its
/// text as it is now, changed as its about says, when it has at most `room.rewrite_max_lines`
/// lines and fits the writer's brief with the contract; one longer, a path named twice after its
/// first, and a path the write refuses are left to the session. None for any other reply. A plan
/// sets `a`'s contract, which `fan_out` clears.
pub(super) fn job(a: &mut Attempt, reply: &Parsed, room: &Room) -> Option<Job> {
    let Checked::Call(call, _, more) = run::check(reply, &a.registry, &a.tools) else {
        return None;
    };
    if call.name != NAME {
        return None;
    }
    let dir = a.task.dir().ok()?.to_path_buf();
    let contract = call.args["contract"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    a.contract = Some(contract.clone());
    let (mut calls, mut left, mut seen) = (Vec::new(), Vec::new(), BTreeSet::new());
    for f in call.args["files"].as_array().into_iter().flatten() {
        let (Some(path), Some(about)) = (f["path"].as_str(), f["about"].as_str()) else {
            continue;
        };
        let rel = inner(&a.task.repo, path);
        if !seen.insert(rel.clone()) {
            continue;
        }
        let old = fs::read_to_string(dir.join(&rel)).ok();
        let about = match old {
            Some(_) => format!("its text as it is now, with this change and no other: {about}"),
            None => about.into(),
        };
        let args = json!({"path": path, "about": about});
        let write = ToolCall {
            name: "write_file".into(),
            args,
        };
        let fits = old.is_none_or(|t| {
            t.lines().count() <= room.rewrite_max_lines
                && a.briefs(&write, "").ok().flatten().is_some()
        });
        match fits && prepare_deep(&write.args, &a.ctx).is_ok() {
            true => calls.push(write),
            false => left.push(rel),
        }
    }
    if calls.is_empty() && left.is_empty() {
        a.contract = None;
        return None;
    }
    Some(Job { calls, left, more })
}

/// Runs `job` in `a`: one progress line for the plan, then its files `run.room.writers` at a
/// time through the writer (`Code::calls`), each brief with the contract; then the cross-file
/// checks on the files written. What is left (a file left unchanged, one the user declined, a
/// finding, calls after the plan) is told as one more step, and the session makes the rest of
/// the change. The plan is offered no more.
#[allow(clippy::too_many_arguments)]
pub(super) fn fan_out(
    code: &Code,
    larch: &Larch,
    a: &mut Attempt,
    job: Job,
    answer: &Answer,
    run: &Setup,
    audit: &Audit,
    sink: &mut dyn FnMut(AgentEvent),
    ask: &mut dyn FnMut(&Ask) -> Decision,
    stop: &Arc<AtomicBool>,
) -> Result<(), ToolError> {
    a.tools.retain(|t| *t != NAME);
    let path = |c: &ToolCall| c.args["path"].as_str().unwrap_or_default().to_string();
    let paths: Vec<String> = job
        .calls
        .iter()
        .map(|c| inner(&a.task.repo, &path(c)))
        .collect();
    let all: Vec<&str> = (paths.iter().chain(&job.left))
        .map(String::as_str)
        .collect();
    let files = match all.len() {
        1 => "1 file".into(),
        n => format!("{n} files"),
    };
    let line = format!("{}Plan: {files}: {}", a.label, all.join(", "));
    sink(AgentEvent::Progress(line));
    let n = a.steps.len();
    for wave in job.waves(run.room.writers) {
        code.calls(larch, a, wave, false, answer, run, audit, sink, ask, stop)?;
        if a.stopped.is_some() || stop.load(Ordering::Relaxed) {
            break;
        }
    }
    a.contract = None;
    if a.stopped.is_some() || stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    let steps = &a.steps[n..];
    let wrote: BTreeSet<String> = (steps.iter())
        .filter_map(|s| match &s.2 {
            Evidence::Wrote(path) => Some(path.clone()),
            _ => None,
        })
        .collect();
    // The step of a call the user said no to shows its path, as `shown` writes it.
    let declined = |c: &ToolCall| {
        let named = format!("\"path\":{}", Value::from(path(c)));
        (steps.iter()).any(|s| s.0.contains(&named) && s.1.contains(SAID_NO))
    };
    let (mut missed, mut no) = (Vec::new(), Vec::new());
    for (c, p) in job
        .calls
        .iter()
        .zip(&paths)
        .filter(|(_, p)| !wrote.contains(*p))
    {
        match declined(c) {
            true => no.push(p.clone()),
            false => missed.push(p.clone()),
        }
    }
    missed.extend(job.left);
    let found = checks::check(a.task.dir()?, &wrote);
    if let Some(told) = fallback(&missed, &no, &found, job.more) {
        a.steps.push(("(plan)".into(), told, Evidence::Other));
    }
    Ok(())
}

/// What the session is told after a plan that left the files `missed` unchanged, whose files
/// the user said `no` to, whose files do not fit together (`found`), or whose reply made `more`
/// calls after it: what is left, for it to do itself. None when nothing is.
fn fallback(missed: &[String], no: &[String], found: &[Finding], more: bool) -> Option<String> {
    if missed.is_empty() && no.is_empty() && found.is_empty() && !more {
        return None;
    }
    let mut out = String::from("The plan's files were done side by side");
    if !no.is_empty() {
        let _ = write!(out, "; the user declined {}: leave them", no.join(", "));
    }
    if more {
        out += "; only the plan ran, not the calls after it";
    }
    if !missed.is_empty() {
        let _ = write!(out, "; not changed: {}", missed.join(", "));
    }
    if !found.is_empty() {
        out += "; they do not fit together:";
        for f in found {
            let _ = write!(out, "\n{f}");
        }
    }
    if !missed.is_empty() || !found.is_empty() || more {
        out += "\nMake the rest of this change yourself, one call at a time.";
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        code::r#loop::{self, PROMPT_TOKENS},
        config::{MAX_THINKING_BUDGET, Thinking},
        workers::Workers,
    };

    /// A registry with the plan call, on an 8k context.
    fn registry() -> Registry {
        let mut registry = Registry::builtin().noted();
        registry.replace(Box::new(Plan { tokens: 1024 }));
        registry
    }

    /// `PROMPT_TOKENS` holds the system prompt and every tool's schema, plan and write_file
    /// included, at four bytes a token, with room for a short request: the room sizes a view
    /// past it (`Room::view_tokens`).
    #[test]
    fn the_first_prompt_fits_its_estimate() {
        let mut registry = registry();
        registry.replace(crate::tools::code::write_file());
        let specs = registry.specs(&r#loop::tools(true, false));
        let tools = Value::Array(specs.iter().map(ToolSpec::function).collect()).to_string();
        let system = r#loop::system(Family::Qwen3, true, true, false);
        let request = 400;
        let bytes = system.len() + tools.len() + request;
        assert!(bytes <= PROMPT_TOKENS as usize * 4, "{bytes}");
    }

    /// The plan's grammar, in the JSON body and the XML one, takes a plan of path and about
    /// lines with its contract, and refuses a file without its about or a reply that is not the
    /// call; checked against its schema, the reply's files are a list in either body.
    #[test]
    fn the_plan_grammar_holds_a_reply_to_one_plan_call() {
        let registry = registry();
        let files = r#"[{"path": "index.html", "about": "the page"}, {"path": "game.js", "about": "the rules"}]"#;
        for xml in [false, true] {
            let g = run::grammar(&registry, &[NAME], Some(Family::Qwen3), xml, Hold::Required)
                .unwrap()
                .unwrap();
            let said = |contract: &str, files: &str| match xml {
                false => format!(
                    "<tool_call>\n{{\"name\": \"{NAME}\", \"arguments\": {{\"contract\": \"{contract}\", \"files\": {files}}}}}\n</tool_call>"
                ),
                true => format!(
                    "<tool_call>\n<function={NAME}>\n<parameter=contract>\n{contract}\n</parameter>\n<parameter=files>\n{files}\n</parameter>\n</function>\n</tool_call>"
                ),
            };
            let matches = |t: &str| larch::api::grammar::matches(&g, t).unwrap();
            let ok = said("ids: board; play(col)", files);
            assert!(matches(&ok), "{ok}");
            assert!(!matches(&said("", r#"[{"path": "a.js"}]"#)), "xml {xml}");
            assert!(!matches("I will write three files."));
            let reply = run::read(Some(Family::Qwen3), &ok, false);
            let Checked::Call(call, ..) = run::check(&reply, &registry, &[NAME]) else {
                panic!("no valid call: {ok}");
            };
            assert_eq!(call.args["files"].as_array().map(Vec::len), Some(2), "{ok}");
            assert_eq!(call.args["contract"], "ids: board; play(col)");
        }
    }

    /// With thinking, the hold arms after the think block; without, from the first token.
    #[test]
    fn the_plan_hold_waits_for_the_think_block_only_when_thinking() {
        let thinking = GenParams {
            thinking_budget: Some(512),
            ..GenParams::default()
        };
        let trigger = |p: &GenParams| {
            let held = held(&registry(), Family::Qwen3, false, p).unwrap();
            held.grammar.unwrap().trigger
        };
        assert_eq!(trigger(&thinking).as_deref(), Some("</think>"));
        assert_eq!(trigger(&GenParams::default()), None);
    }

    /// A job's files go `Room::writers` at a time: a big made-up deployment writes eight files in
    /// one wave where a small one takes several.
    #[test]
    fn a_big_deployment_writes_more_files_at_once() {
        let calls: Vec<ToolCall> = (0..8)
            .map(|i| ToolCall {
                name: "write_file".into(),
                args: json!({"path": format!("f{i}.js"), "about": "x"}),
            })
            .collect();
        let job = Job {
            calls: calls.clone(),
            left: Vec::new(),
            more: false,
        };
        let full = Thinking {
            budget: MAX_THINKING_BUDGET,
            answer_max_tokens: 0,
        };
        let workers = Workers {
            max_workers: 16,
            prefix_cache: true,
        };
        let small = Room::new(8192, 1024, workers, full);
        let big = Room::new(131_072, 4 * 131_072 / 16, workers, full);
        let waves = |room: Room| job.waves(room.writers);
        assert_eq!(waves(big).len(), 1, "{big:?}");
        assert!(waves(small).len() > 1, "{small:?}");
        let back: Vec<ToolCall> = (waves(small).into_iter())
            .flat_map(|w| w.call.into_iter().chain(w.later))
            .collect();
        assert_eq!(back, calls);
    }

    /// Nothing left is no fallback; a miss or a finding tells the session to do the rest, and a
    /// file the user declined is named to leave, never to make.
    #[test]
    fn a_plan_falls_back_only_on_a_miss_or_a_finding() {
        assert_eq!(fallback(&[], &[], &[], false), None);
        let page = vec!["index.html".to_string()];
        let told = fallback(&page, &[], &[], false).unwrap();
        assert!(told.contains("not changed: index.html"), "{told}");
        let f = Finding {
            file: "game.js".into(),
            line: 3,
            message: "looks up the id board, which no element has".into(),
        };
        let told = fallback(&[], &[], std::slice::from_ref(&f), false).unwrap();
        assert!(told.contains(&f.to_string()), "{told}");
        assert!(told.ends_with("yourself, one call at a time."), "{told}");
        let told = fallback(&[], &page, &[], false).unwrap();
        assert!(
            told.contains("the user declined index.html: leave them"),
            "{told}"
        );
        assert!(!told.contains("yourself"), "{told}");
        let told = fallback(&[], &[], &[], true).unwrap();
        assert!(told.contains("not the calls after it"), "{told}");
    }
}
