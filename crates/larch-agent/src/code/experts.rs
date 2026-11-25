//! Area experts (the plan's "Using concurrency"): when the lead's own map (`outline`) cannot show
//! the repository in a quarter of the context, the session loop's model (the lead) gets
//! `ask_experts`. On its first ask the repository is cut into areas, folders whose map fits the
//! lead's map share (`Room::map_tokens`); each ask goes to the areas whose code matches the
//! question best (`lookup::lookup_areas`), as many as the room asks at once, each an engine
//! sequence whose system prompt is its area's map and whose question comes with that code. An
//! expert answers, held by a grammar, with the places that matter as `path:line: what` lines, or
//! `none`; a place must be a line of a named block of its own area's files, and one that is not is
//! dropped. Experts read; the lead views and edits itself. When no place is left, the step falls
//! back to the single session's lookup. An expert is one sequence per ask, which parks nothing,
//! unless the engine's prefix cache reuses its map and the KV pool holds every area's beside the
//! attempts: then it is a kept session, parked between asks.

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    ops::Range,
    path::Path,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::SystemTime,
};

use larch::{Finish, GenParams, Larch, Session, SessionOptions, api::Grammar};
use larch_model::Message;
use regex::Regex;
use serde_json::{Value, json};

use crate::{
    code::{
        lookup,
        r#loop::{Attempt, Code, Setup, file_map, messages, outline, send, source},
        map,
    },
    parse::{Parsed, ToolCall},
    run::{self, AgentEvent, Checked},
    skills::{ask_docs, read_papers},
    tool::{Permission, Tool, ToolCtx, ToolError, ToolSpec},
    workers::Workers,
};

/// The ask_experts call's name.
pub(super) const NAME: &str = "ask_experts";

/// The most tokens an expert's answer takes: four places of a line each.
const ANSWER_TOKENS: u32 = 200;

/// The most tokens an ask's result takes: the places of the experts asked at once.
const RESULT_TOKENS: u32 = 1000;

/// What each expert is told, before its area's map.
const EXPERT: &str = "You know one area of a code repository: its files with their lines and \
named blocks follow, and each question comes with the area's code that matches it best, \
numbered. Answer with the places the question is about, one per line as path:line: what is there \
and what it expects, at most four lines. Answer none when your area has nothing the question \
asks about. Name only places in your area's files.";

/// A place's path, line and what is there, as the grammar holds and `places` reads them.
const PATH: &str = r"[^\s:]+";
const LINE: &str = r"[1-9][0-9]{0,6}";
const WHAT: &str = r"\S[^\n]*";

/// A place in an answer, its path and line captured.
static PLACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"^({PATH}):({LINE}): {WHAT}$")).expect("a valid regex"));

/// The grammar of an expert's answer: none, or one to four places.
fn answer_pattern() -> String {
    let place = format!("{PATH}:{LINE}: {WHAT}");
    format!(r"(?:none|{place}(?:\n{place}){{0,3}})")
}

/// The ask_experts call. The session loop runs the experts (`Experts::ask`) and leaves their
/// result here; the call then runs as any other, audited and cut to its result tokens.
struct Ask(Arc<Mutex<Option<String>>>);

impl Tool for Ask {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: NAME.into(),
            description: "This repository is too big to read whole: asks the experts of the \
                          areas that match the question best where something is or what a \
                          module expects, and returns their places as path:line: what. View and \
                          edit the places yourself."
                .into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description":
                        "Where is X, or what does module Y expect, such as: where is the retry logic"}
                },
                "required": ["question"],
                "additionalProperties": false
            }),
            permission: Permission::Read,
            untrusted_results: false,
            max_result_tokens: RESULT_TOKENS,
            max_args_tokens: 200,
        }
    }

    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        slot.take()
            .ok_or_else(|| ToolError::Invalid("ask_experts runs in the session loop only".into()))
    }
}

/// A file of an area: its path, when it was read, and its map (`file_map`).
struct Mapped {
    path: String,
    stamp: Option<(Option<SystemTime>, u64)>,
    map: String,
}

/// The areas, built on the first ask: each file's map, path order, each area's range of them,
/// the area of each file, and each area's kept session when experts park.
struct Areas {
    files: Vec<Mapped>,
    ranges: Vec<Range<usize>>,
    of: BTreeMap<String, usize>,
    kept: Vec<Option<Session>>,
}

/// The experts of a run whose repository the lead's map cannot show whole.
pub(super) struct Experts {
    /// What `Ask` returns next.
    slot: Arc<Mutex<Option<String>>>,
    /// The tokens an area's map takes at most: the lead's map share.
    map_tokens: usize,
    ctx: usize,
    /// The experts asked at once (`at_once`).
    at_once: usize,
    /// Experts are kept sessions (`keeps`).
    keep: bool,
    /// The deployment's KV pool and the run's attempts, for `keeps` once the areas are known.
    pool: u32,
    attempts: usize,
    params: GenParams,
    /// None until the first ask; Err when the repository could not be cut into two areas.
    areas: Option<Result<Areas, String>>,
}

/// The tokens an expert takes at most: its area's map, its code and its answer.
fn each(map_tokens: usize, ctx: usize) -> usize {
    map_tokens + ctx / 8 + ANSWER_TOKENS as usize + read_papers::TEMPLATE_TOKENS as usize
}

/// Ranges of `files` (path, tokens), path order, each a folder, or a run of neighbouring ones,
/// whose map fits `budget`: a folder too big is cut by its subfolders and files, and a file
/// bigger alone is an area alone.
fn areas(files: &[(&str, usize)], budget: usize) -> Vec<Range<usize>> {
    let mut cut = Vec::new();
    split(files, 0, 0, budget, &mut cut);
    let size = |r: &Range<usize>| files[r.clone()].iter().map(|f| f.1).sum::<usize>();
    let mut packed: Vec<Range<usize>> = Vec::new();
    for r in cut {
        match packed.last_mut() {
            Some(last) if size(last) + size(&r) <= budget => last.end = r.end,
            _ => packed.push(r),
        }
    }
    packed
}

/// Cuts `files`, which start at `from` and share their first `depth` folders, into ranges whose
/// tokens fit `budget`, by the folder (or file) at `depth`.
fn split(
    files: &[(&str, usize)],
    from: usize,
    depth: usize,
    budget: usize,
    out: &mut Vec<Range<usize>>,
) {
    if files.len() == 1 || files.iter().map(|f| f.1).sum::<usize>() <= budget {
        out.push(from..from + files.len());
        return;
    }
    let key = |f: &(&str, usize)| f.0.split('/').nth(depth).map(str::to_owned);
    let mut start = 0;
    while start < files.len() {
        let k = key(&files[start]);
        let end = start + files[start..].iter().take_while(|f| key(f) == k).count();
        split(&files[start..end], from + start, depth + 1, budget, out);
        start = end;
    }
}

/// The experts asked at once: as many as config.toml's workers and a KV pool of `pool` blocks
/// (0 is not known) hold at `each` tokens an expert, as the research readers.
fn at_once(workers: Workers, pool: u32, each: usize) -> usize {
    let each = each as u64;
    read_papers::at_once(usize::MAX, workers, u64::from(pool), each).min(workers.at_once())
}

/// Whether experts are kept sessions, parked between asks: only when the engine's prefix cache
/// reuses a map the next ask repeats (a model without one, as a hybrid, reuses a parked state
/// only whole, and a new question changes it) and a known pool of `pool` blocks holds every one
/// of `areas` experts at `each` tokens beside `attempts` whole contexts of `ctx` and one more for
/// the writers and the reviewer, so a parked expert never pushes out the lead.
fn keeps(
    prefix_cache: bool,
    pool: u32,
    ctx: usize,
    attempts: usize,
    areas: usize,
    each: usize,
) -> bool {
    let pool = u64::from(pool) * read_papers::KV_BLOCK_TOKENS;
    let need = ((attempts + 1) * ctx + areas * each) as u64;
    prefix_cache && pool > 0 && need <= pool
}

/// The places of an expert's `answer` for area `k` (`of`) of `dir` that hold, each line whole,
/// and why each other one was dropped: a place must be a line of a named block of a file of its
/// own area, as the file is now. `./` before a path and `\` in it are read as git writes paths.
fn places(
    answer: &str,
    k: usize,
    dir: &Path,
    of: &BTreeMap<String, usize>,
) -> (Vec<String>, Vec<String>) {
    let (mut kept, mut dropped) = (Vec::new(), Vec::new());
    let answer = answer.trim();
    if answer == "none" {
        return (kept, dropped);
    }
    for l in answer.lines() {
        let checked = (|| {
            let c = PLACE.captures(l).ok_or("names no place")?;
            let path = c[1].replace('\\', "/");
            let path = path.strip_prefix("./").unwrap_or(&path);
            let line: usize = c[2].parse().map_err(|_| "names no line")?;
            if of.get(path) != Some(&k) {
                return Err("is no file of its area");
            }
            let text = source(&dir.join(path)).ok_or("cannot be read")?;
            let blank = text
                .lines()
                .nth(line - 1)
                .is_none_or(|t| t.trim().is_empty());
            let named = (map::symbols(path, &text).iter())
                .any(|s| (s.start_line..=s.end_line).contains(&line));
            match !blank && named {
                true => Ok(l.replacen(&c[1], path, 1)),
                false => Err("is no line of a named block"),
            }
        })();
        match checked {
            Ok(place) => kept.push(place),
            Err(why) => dropped.push(format!("{l} ({why})")),
        }
    }
    (kept, dropped)
}

/// The question of `reply` when it is a valid ask_experts call of `a`'s that its tool_calls
/// budget lets run.
pub(super) fn asked(a: &Attempt, reply: &Parsed) -> Option<String> {
    if a.calls >= a.task.budget.tool_calls {
        return None;
    }
    match run::check(reply, &a.registry, &a.tools) {
        Checked::Call(call, ..) if call.name == NAME => {
            Some(call.args["question"].as_str()?.to_owned())
        }
        _ => None,
    }
}

/// When a file was last changed and its size, to tell a map gone stale.
fn stamp(path: &Path) -> Option<(Option<SystemTime>, u64)> {
    fs::metadata(path)
        .ok()
        .map(|m| (m.modified().ok(), m.len()))
}

/// A file's map at `path` in `dir`, whole, its name alone when the map does not read it.
fn mapped(dir: &Path, path: &str) -> Mapped {
    let full = dir.join(path);
    let map = match source(&full) {
        Some(text) => file_map(path, &text, |_| usize::MAX).0,
        None => format!("{path}\n"),
    };
    Mapped {
        path: path.to_owned(),
        stamp: stamp(&full),
        map,
    }
}

impl Experts {
    /// The experts of `a`'s repository on `larch`, which `code` runs in `run`: None when
    /// config.toml turns them off or the lead's map (`outline`) shows the whole repository in a
    /// quarter of the context. The areas wait for the first ask.
    pub(super) fn new(
        code: &Code,
        larch: &Larch,
        a: &Attempt,
        run: &Setup,
        attempts: usize,
        stop: &AtomicBool,
    ) -> Result<Option<Experts>, ToolError> {
        let d = larch.deployment();
        let ctx = d.ctx_max as usize;
        if !code.experts || outline(&a.task, d.ctx_max / 4, stop)?.1 {
            return Ok(None);
        }
        let map_tokens = run.room.map_tokens as usize;
        let params = GenParams {
            max_tokens: ANSWER_TOKENS,
            grammar: Some(Grammar {
                trigger: None,
                pattern: answer_pattern(),
            }),
            ..ask_docs::params(Some(code.family), run.namespace)
        };
        Ok(Some(Experts {
            slot: Arc::default(),
            map_tokens,
            ctx,
            at_once: at_once(code.workers, d.kv_pool_blocks, each(map_tokens, ctx)),
            keep: code.workers.prefix_cache,
            pool: d.kv_pool_blocks,
            attempts,
            params,
            areas: None,
        }))
    }

    /// Offers ask_experts to `a`.
    pub(super) fn offer(&self, a: &mut Attempt) {
        a.registry.replace(Box::new(Ask(self.slot.clone())));
        a.tools.push(NAME);
    }

    /// The areas of `a`'s repository, cut on the first ask from its files: Err when their map
    /// does not make two areas, as one expert would hold only what the lead's map could.
    fn areas(
        &mut self,
        a: &Attempt,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<&mut Areas, String> {
        if self.areas.is_none() {
            let built = (|| {
                let dir = a.task.dir().map_err(|e| e.to_string())?;
                let mut paths = a.task.files().map_err(|e| e.to_string())?;
                paths.sort();
                let files: Vec<Mapped> = paths.iter().map(|p| mapped(dir, p)).collect();
                let sized: Vec<(&str, usize)> = (files.iter())
                    .map(|f| (f.path.as_str(), f.map.len().div_ceil(4)))
                    .collect();
                let ranges = areas(&sized, self.map_tokens);
                if ranges.len() < 2 {
                    return Err("the repository's map makes one area".to_string());
                }
                let of = (ranges.iter().enumerate())
                    .flat_map(|(k, r)| files[r.clone()].iter().map(move |f| (f.path.clone(), k)))
                    .collect();
                let n = ranges.len();
                let each = each(self.map_tokens, self.ctx);
                self.keep = keeps(self.keep, self.pool, self.ctx, self.attempts, n, each);
                Ok(Areas {
                    files,
                    ranges,
                    of,
                    kept: (0..n).map(|_| None).collect(),
                })
            })();
            if let Ok(areas) = &built {
                let parked = if self.keep {
                    "kept between asks"
                } else {
                    "one sequence an ask"
                };
                let line = format!("Area experts: {} areas, {parked}", areas.ranges.len());
                sink(AgentEvent::Progress(line));
            }
            self.areas = Some(built);
        }
        let built = self.areas.as_mut().expect("set above");
        built.as_mut().map_err(|e| e.clone())
    }

    /// Asks `question` for `a` of the areas whose code matches it best, as many as run at
    /// once, and leaves what they said for the ask_experts call (`Ask`), which the caller then
    /// runs: the places that hold, and the ones dropped. When none holds, or the experts could
    /// not run, the result says lookup follows, and the lookup call to run after it comes back.
    pub(super) fn ask(
        &mut self,
        code: &Code,
        larch: &Larch,
        a: &mut Attempt,
        question: &str,
        sink: &mut dyn FnMut(AgentEvent),
        stop: &AtomicBool,
    ) -> Option<Parsed> {
        let label = a.label.clone();
        let slot = self.slot.clone();
        let fall = |sink: &mut dyn FnMut(AgentEvent), why: String| {
            // A stopped ask has no lookup after it: the session stops at its next check.
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let line = "Area experts: falling back to lookup";
            sink(AgentEvent::Progress(format!("{label}{line}")));
            *slot.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("{}\nWhat lookup finds follows.", why.trim_end()));
            Some(Parsed {
                call: Some(ToolCall {
                    name: "lookup".into(),
                    args: json!({"query": question}),
                }),
                ..Parsed::default()
            })
        };
        let dir = match a.task.dir() {
            Ok(dir) => dir.to_path_buf(),
            Err(e) => return fall(sink, format!("The folder could not be read ({e})")),
        };
        // The first ask's cut decides whether experts park, so it comes before keep is read.
        if let Err(why) = self.areas(a, sink) {
            return fall(sink, format!("No area experts: {why}"));
        }
        let (at_once, keep, map_tokens, ctx) = (self.at_once, self.keep, self.map_tokens, self.ctx);
        let params = self.params.clone();
        let areas = self.areas(a, sink).expect("cut above");
        let of = &areas.of;
        let best = match lookup::lookup_areas(&dir, question, at_once, |f| of.get(f).copied(), stop)
        {
            Ok(best) if !best.is_empty() => best,
            Ok(_) => return fall(sink, "No area's code matches the question".into()),
            Err(e) => return fall(sink, format!("The areas could not be ranked ({e})")),
        };
        let n = best.len();
        sink(AgentEvent::Progress(format!(
            "{label}Asking {n} area experts: {question}"
        )));
        let tokens = |t: &str| larch.token_ids(t).len();
        let mut prompts: Vec<Vec<Vec<Message>>> = Vec::with_capacity(n);
        for (k, parts) in &best {
            let range = areas.ranges[*k].clone();
            // A map the lead's writes made stale is read again.
            for f in &mut areas.files[range.clone()] {
                if stamp(&dir.join(&f.path)) != f.stamp {
                    *f = mapped(&dir, &f.path);
                }
            }
            let mut map = String::new();
            for f in &areas.files[range] {
                if (map.len() + f.map.len()).div_ceil(4) > map_tokens {
                    break;
                }
                map += &f.map;
            }
            let system = format!("{EXPERT}\n\nThe area's files:\n{map}");
            // The code gets an eighth of the context, less when the map takes more tokens than
            // its bytes say; the answer and the template keep theirs.
            let mut left = (ctx / 8).min(ctx.saturating_sub(tokens(&system) + each(0, 0)));
            let mut shown = String::new();
            for part in parts {
                let Some(rest) = left.checked_sub(tokens(part) + 1) else {
                    break;
                };
                left = rest;
                let _ = writeln!(shown, "{part}");
            }
            let user = format!("The area's code that matches best:\n{shown}\nQuestion: {question}");
            prompts.push(vec![messages(&system, &user)]);
        }
        let one_shot = SessionOptions {
            keep_kv: false,
            ..SessionOptions::default()
        };
        let mut sessions: Vec<Session> = (best.iter())
            .map(|(k, _)| match keep {
                true => (areas.kept[*k].take())
                    .unwrap_or_else(|| larch.session(SessionOptions::default())),
                false => larch.session(one_shot.clone()),
            })
            .collect();
        let mut answers = Vec::with_capacity(n);
        for (wave, prompts) in sessions.chunks_mut(at_once).zip(prompts.chunks(at_once)) {
            let mut wave: Vec<&mut Session> = wave.iter_mut().collect();
            answers.extend(send(
                larch,
                &code.workers,
                &mut wave,
                prompts,
                &params,
                stop,
            ));
            if answers.len() < n {
                let line = format!("{} of {n} area experts answered", answers.len());
                sink(AgentEvent::Progress(format!("{label}{line}")));
            }
        }
        if keep {
            for ((k, _), session) in best.iter().zip(sessions) {
                areas.kept[*k] = Some(session);
            }
        }
        let (mut said, mut notes) = (Vec::new(), String::new());
        for ((k, _), answer) in best.iter().zip(answers) {
            // The experts' tokens are the run's, as its writers' are.
            if let Ok(answer) = &answer {
                a.tokens += u64::from(answer.usage.completion_tokens);
            }
            match answer {
                Ok(answer) if answer.finish != Finish::Length => {
                    let (kept, dropped) = places(&answer.text, *k, &dir, &areas.of);
                    said.extend(kept);
                    for d in dropped {
                        let _ = write!(notes, "\ndropped, not in its area's named blocks: {d}");
                    }
                }
                Ok(_) => notes += "\nan expert's answer was cut",
                Err(e) => {
                    let _ = write!(notes, "\nan expert failed: {e}");
                }
            }
        }
        if said.is_empty() {
            return fall(sink, format!("No area expert named a place{notes}"));
        }
        let line = format!("Area experts: {} places", said.len());
        sink(AgentEvent::Progress(format!("{label}{line}")));
        let seen = format!("The area experts' places:\n{}{notes}", said.join("\n"));
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(seen);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `folders` folders of 40 files each, every file's map 30 tokens.
    fn repo(folders: usize) -> Vec<(String, usize)> {
        (0..folders)
            .flat_map(|d| (0..40).map(move |f| (format!("pkg{d:02}/m{f:02}.py"), 30)))
            .collect()
    }

    fn sized(files: &[(String, usize)]) -> Vec<(&str, usize)> {
        files.iter().map(|(f, n)| (f.as_str(), *n)).collect()
    }

    /// Areas run in order over every file, each a run of whole folders whose maps fit the
    /// budget: ten folders of 1200 tokens at a budget of 4096 make areas of three.
    #[test]
    fn areas_are_whole_folders_that_fit_the_budget() {
        let big = repo(10);
        let cut = areas(&sized(&big), 4096);
        assert_eq!(cut, vec![0..120, 120..240, 240..360, 360..400]);
        let one = areas(&sized(&repo(3)), 4096);
        assert_eq!(one.len(), 1, "{one:?}");
    }

    /// A folder too big for one area is cut by its files; a file bigger alone is its own area.
    #[test]
    fn a_folder_too_big_is_cut_and_a_huge_file_is_alone() {
        let mut flat: Vec<(String, usize)> =
            (0..100).map(|f| (format!("src/m{f:03}.py"), 100)).collect();
        flat[50].1 = 9000;
        let cut = areas(&sized(&flat), 4096);
        assert!(cut.contains(&(50..51)), "{cut:?}");
        for r in cut.iter().filter(|r| r.len() > 1) {
            assert!(r.len() * 100 <= 4096, "{r:?}");
        }
    }

    const MANY: Workers = Workers {
        max_workers: 16,
        prefix_cache: true,
    };

    /// A big made-up deployment asks many experts at once; a laptop's pool of two 8k contexts
    /// fewer, and one worker one at a time.
    #[test]
    fn experts_are_asked_at_once_up_to_the_room() {
        let big = at_once(MANY, 16 * 131_072 / 16, each(16_384, 131_072));
        assert_eq!(big, 15);
        let small = at_once(MANY, 1024, each(2000, 8192));
        assert_eq!(small, 4);
        let one = Workers {
            max_workers: 1,
            prefix_cache: false,
        };
        assert_eq!(at_once(one, 0, each(2000, 8192)), 1);
    }

    /// Experts park only with a prefix cache and a pool that holds them all beside the
    /// attempts: never on a model without one (a hybrid), nor in a pool not known or too small.
    #[test]
    fn experts_park_only_when_the_pool_holds_them_beside_the_lead() {
        let (ctx, e) = (131_072, each(16_384, 131_072));
        let pool = (64 * ctx / 16) as u32;
        assert!(keeps(true, pool, ctx, 1, 8, e));
        assert!(!keeps(false, pool, ctx, 1, 8, e));
        assert!(!keeps(true, 0, ctx, 1, 8, e));
        assert!(!keeps(true, pool, ctx, 1, 400, e));
        assert!(!keeps(true, (4 * ctx / 16) as u32, ctx, 2, 8, e));
    }

    /// A place holds when it is a line of a named block of its own area's file: a good line
    /// is kept beside a bad one, and a line past the file, in another area, blank or outside
    /// every block is dropped with why. `./` and `\` read as git paths.
    #[test]
    fn a_place_must_be_a_named_line_of_its_own_area() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("net")).unwrap();
        fs::create_dir_all(dir.path().join("ui")).unwrap();
        let net = "import os\n\ndef a():\n\n    return 1\n";
        fs::write(dir.path().join("net/a.py"), net).unwrap();
        fs::write(dir.path().join("ui/b.py"), "def b():\n    return 2\n").unwrap();
        let of = BTreeMap::from([("net/a.py".to_string(), 0), ("ui/b.py".to_string(), 1)]);
        let at = |answer: &str| places(answer, 0, dir.path(), &of);
        let (kept, dropped) = at("net/a.py:3: def a\nnet/a.py:9: nothing");
        assert_eq!(kept, ["net/a.py:3: def a"]);
        assert_eq!(
            dropped,
            ["net/a.py:9: nothing (is no line of a named block)"]
        );
        assert_eq!(at("none"), (vec![], vec![]));
        let (kept, dropped) = at("ui/b.py:1: def b\nnet/a.py:4: blank\nnet/a.py:1: import");
        assert!(kept.is_empty(), "{kept:?}");
        assert_eq!(dropped.len(), 3, "{dropped:?}");
        assert!(
            dropped[0].ends_with("(is no file of its area)"),
            "{dropped:?}"
        );
        let (kept, _) = at("./net/a.py:3: def a\nnet\\a.py:5: returns 1");
        assert_eq!(kept, ["net/a.py:3: def a", "net/a.py:5: returns 1"]);
    }

    /// The answer grammar takes none or up to four places, each a place `places` reads, and
    /// nothing else.
    #[test]
    fn the_answer_grammar_and_the_place_check_agree() {
        let g = Grammar {
            trigger: None,
            pattern: answer_pattern(),
        };
        let matches = |t: &str| larch::api::grammar::matches(&g, t).unwrap();
        assert!(matches("none"));
        let two = "src/a.py:12: retry loop\nsrc/b.py:3: backoff";
        assert!(matches(two));
        assert!(two.lines().all(|l| PLACE.is_match(l)));
        for bad in [
            "The retry logic is in src/a.py.",
            "src/a.py:0: x",
            "src/a.py:1:  two spaces",
            "src/a.py:1: ",
        ] {
            assert!(!matches(bad), "{bad}");
            assert!(!PLACE.is_match(bad), "{bad}");
        }
        assert!(!matches(&["a.py:1: x"; 5].join("\n")));
    }
}
