//! `bench --memory-chats`: scripted chats through the real `Agent`, scored on what memory kept
//! (`memory::list` after each turn) and on probe questions asked later in new chats.
//!
//! A case runs in its own throwaway data folder: its seeded facts, its chats one after another
//! (each a new chat), the facts it forgets, then each probe in a new chat. Nothing is injected
//! into a turn; the only help a turn gets is a yes to a `remember` y/n (asked after an untrusted result), as
//! a user who said the fact would give, and a no to every other write and research plan.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

use anyhow::Context;
use larch::Larch;
use serde::{Deserialize, Serialize};

use super::{
    Heard, PromptsFile, Report, Routing, append_logs, chat_options, clear_logs, holds, lines,
    listen, nearest, settings_differ, steps_of, write_config,
};
use crate::{
    Agent, AgentOptions,
    audit::{AuditEvent, EventKind},
    config::Thinking,
    memory,
    skills::ask_docs::ms,
};

/// One user message of a case's chat.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryMessage {
    /// What the user says.
    pub user: String,
    /// The message holds nothing to keep: a save during its turn is a false save.
    #[serde(default)]
    pub no_save: bool,
    /// The user asks for the fact to be saved ("remember that ...").
    #[serde(default)]
    pub explicit: bool,
}

/// A fact a case's chats state, as the details a saved fact must hold. Each detail is a word or
/// phrase as `holds` reads it (`*` runs on), or alternatives split by `|` ("Lucía|Lucia").
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldFact {
    /// The details; the first is the key, which finds the saved fact the fact is in.
    pub details: Vec<String>,
    /// The user asked for it to be remembered; kept out of "remembers on its own".
    #[serde(default)]
    pub explicit: bool,
    /// Details a correction replaced: a fact holding one of these and some of `details`, but not
    /// all, is a stale fact left beside the new one. Such a fact is saved only through a fact
    /// holding its key and one more detail.
    #[serde(default)]
    pub stale: Vec<String>,
}

/// A question asked in a new chat after the case's chats, and how its answer is scored: it
/// holds every `answer` detail and none of `not`. With neither, a human checks it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryProbe {
    /// The question.
    pub user: String,
    /// Details the answer must hold, as `GoldFact::details` reads them.
    #[serde(default)]
    pub answer: Vec<String>,
    /// Details the answer must not hold.
    #[serde(default)]
    pub not: Vec<String>,
}

/// One line of bench/agent/memory-chats/chats.jsonl.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryCase {
    /// `mc-<nn>-<name>`.
    pub id: String,
    /// What it tests.
    #[serde(default)]
    pub tags: Vec<String>,
    /// The #436 fixture it was made from, if any.
    #[serde(default)]
    pub from: Option<String>,
    /// Facts in memory before the first chat, saved as the `remember` tool saves them.
    #[serde(default)]
    pub memory: Vec<String>,
    /// The chats, in order, each a new chat of user messages.
    pub chats: Vec<Vec<MemoryMessage>>,
    /// After the chats, every fact holding one of these is forgotten, as the app's Forget does.
    #[serde(default)]
    pub forget: Vec<String>,
    /// The facts the chats state.
    #[serde(default)]
    pub facts: Vec<GoldFact>,
    /// Detail sets that some fact must still hold at the end: a seeded fact a wrong merge would
    /// overwrite.
    #[serde(default)]
    pub keep: Vec<Vec<String>>,
    /// Detail sets no fact may hold at the end: a hypothetical, a quote, a "don't save".
    #[serde(default)]
    pub never: Vec<Vec<String>>,
    /// The questions asked in new chats after the chats.
    #[serde(default)]
    pub probes: Vec<MemoryProbe>,
    /// Why the case does not run yet; it is listed and counted as skipped.
    #[serde(default)]
    pub skip: Option<String>,
    /// The chats run with memory off (`memory::set_enabled`), and it is on again for the probes:
    /// nothing said while off is saved, or answered later.
    #[serde(default)]
    pub off: bool,
}

/// A memory chat set, ready for `run_memory_chats`.
#[derive(Debug)]
pub struct MemoryChatSet {
    /// The cases, in file order.
    pub cases: Vec<MemoryCase>,
    /// The file.
    pub file: PromptsFile,
    /// Where the chats' audit log is copied after an error and at the end; `None` keeps none.
    pub audit: Option<std::path::PathBuf>,
}

/// The memory chat set at `path`. A case whose ID is not `mc-<nn>-...` or comes twice, with no
/// chat or an empty one, an empty detail, or a probe with no question fails the set.
pub fn memory_chat_set(path: &Path) -> anyhow::Result<MemoryChatSet> {
    let (cases, file) = lines::<MemoryCase>(path)?;
    let mut seen = std::collections::HashSet::new();
    for c in &cases {
        let named = (c.id.strip_prefix("mc-")).and_then(|r| r.split_once('-'));
        anyhow::ensure!(
            named.is_some_and(|(n, rest)| n.len() == 2
                && n.bytes().all(|b| b.is_ascii_digit())
                && !rest.is_empty()),
            "{} is not named mc-<nn>-<name>",
            c.id
        );
        anyhow::ensure!(seen.insert(&c.id), "{} comes twice", c.id);
        anyhow::ensure!(
            !c.chats.is_empty() && c.chats.iter().all(|chat| !chat.is_empty()),
            "{} has a chat with no message",
            c.id
        );
        let details = (c
            .facts
            .iter()
            .flat_map(|f| f.details.iter().chain(&f.stale)))
        .chain(c.keep.iter().chain(&c.never).flatten())
        .chain(c.probes.iter().flat_map(|p| p.answer.iter().chain(&p.not)))
        .chain(&c.forget);
        for d in details {
            anyhow::ensure!(
                d.split('|').all(|alt| !alt.trim().is_empty()),
                "{} has an empty detail",
                c.id
            );
        }
        anyhow::ensure!(
            c.facts.iter().all(|f| !f.details.is_empty())
                && c.keep.iter().chain(&c.never).all(|set| !set.is_empty()),
            "{} has a fact or a detail set with no detail",
            c.id
        );
        anyhow::ensure!(
            c.probes.iter().all(|p| !p.user.trim().is_empty()),
            "{} has a probe with no question",
            c.id
        );
    }
    Ok(MemoryChatSet {
        cases,
        file,
        audit: None,
    })
}

/// Whether `text` holds `detail`, one of its `|` alternatives, ignoring case.
pub fn has(text: &str, detail: &str) -> bool {
    let text = text.to_lowercase();
    detail.split('|').any(|alt| holds(&text, alt.trim()))
}

/// Whether `text` looks like it holds a secret: an API key, a private key or a bearer token.
pub fn secret_shaped(text: &str) -> bool {
    crate::memory::capture::has_secret(text)
}

/// One turn of a case, a chat message's or a probe's.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryTurn {
    /// The chat it was in, from 0; a probe's chat counts on after the case's chats.
    pub chat: usize,
    /// What the user said.
    pub user: String,
    /// The message holds nothing to keep.
    pub no_save: bool,
    /// The user asked for the fact to be saved.
    pub explicit: bool,
    /// A probe's question, in a chat after the case's chats.
    pub probe: bool,
    /// The slugs of the facts saved or changed during the turn.
    pub saved: Vec<String>,
    /// The tools called, in order.
    pub tools: Vec<String>,
    /// The reply as it showed.
    pub answer: String,
    /// The first model step's time to first token; `None` when no step ran to its end.
    pub ttft_ms: Option<f64>,
    /// Prompt tokens over the turn's model steps.
    pub prompt_tokens: u32,
    /// Of those, the tokens reused from a cache.
    pub cached_tokens: u32,
    /// The engine error, or the failed turn's notice.
    pub error: Option<String>,
    /// The turn, end to end, its y/n answers included.
    pub wall_ms: f64,
}

/// How a gold fact came out.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldRow {
    /// The fact's details.
    pub details: Vec<String>,
    /// The user asked for it to be remembered.
    pub explicit: bool,
    /// A fact a turn of the case saved or changed holds its key, and with `stale`, one more
    /// detail (capture). Seeded facts and facts left as they were do not count.
    pub saved: bool,
    /// That fact, the one that holds most of its details, by slug.
    pub slug: Option<String>,
    /// The details that fact lacks; empty is a whole fact.
    pub missing: Vec<String>,
    /// A fact the correction should have replaced is still there (`GoldFact::stale`).
    pub stale: bool,
}

/// How a probe came out, scored apart for capture, retrieval and use.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRow {
    /// The question.
    pub user: String,
    /// The reply.
    pub answer: String,
    /// Capture: every gold fact of the case was saved.
    pub captured: bool,
    /// Retrieval: the new chat's memory block holds every `answer` detail, or the turn called
    /// `recall`.
    pub retrieved: bool,
    /// Use: the reply holds every `answer` detail and none of `not`; `None` for a human check.
    pub right: Option<bool>,
    /// Why it failed, each.
    pub misses: Vec<String>,
}

/// One case's run and how it scored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryChatRow {
    /// The case's ID.
    pub id: String,
    /// Its tags.
    pub tags: Vec<String>,
    /// Why it did not run; it has no other result.
    pub skipped: Option<String>,
    /// Every turn, the chats' then the probes'.
    pub turns: Vec<MemoryTurn>,
    /// The facts in memory after the chats, before the forgets, as `memory::list` reads them
    /// (the sentence alone, without a `Said:` quote: `sentence_of`).
    pub facts: Vec<SavedFact>,
    /// Each gold fact, scored on `facts`.
    pub gold: Vec<GoldRow>,
    /// Messages with nothing to keep whose turn saved a fact.
    pub false_saves: usize,
    /// Probes whose turn saved a fact.
    pub probe_saves: usize,
    /// `remember` calls on messages that asked for the save.
    pub remember_explicit: usize,
    /// `remember` calls on other messages that hold a fact: the model saved on its own.
    pub remember_unasked: usize,
    /// `remember` calls on messages with nothing to keep, and on probes.
    pub remember_negative: usize,
    /// `never` sets a fact holds at the end, joined by ` + `.
    pub never_held: Vec<String>,
    /// `keep` sets no fact holds at the end, joined by ` + `.
    pub lost: Vec<String>,
    /// Updates that took a gold fact's or a `keep` set's key out of a fact: one fact overwritten
    /// by another, as the near-duplicate update can do to two people.
    pub merges: usize,
    /// Facts at the end that look like a secret (`secret_shaped`).
    pub secrets: usize,
    /// The facts `forget` deleted.
    pub forgot: usize,
    /// `remember` y/n questions (after an untrusted result), each answered yes.
    pub approvals: usize,
    /// Other writes and research plans, each answered no.
    pub declined: usize,
    /// Each probe.
    pub probes: Vec<ProbeRow>,
    /// The first error: a turn's, or the memory folder's. The case's other turns still run, and
    /// its facts are scored all the same.
    pub error: Option<String>,
    /// The case, end to end.
    pub wall_ms: f64,
}

/// A fact in memory at a case's end.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedFact {
    /// Its slug.
    pub slug: String,
    /// Its whole text.
    pub text: String,
}

/// Nearest-rank p50 and p95.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct P50P95 {
    /// The median.
    pub p50: f64,
    /// The 95th percentile.
    pub p95: f64,
}

/// The run's numbers over the cases that ran, an errored one included (its facts count as they
/// are); each share is `None` when there was nothing to count.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryChatSummary {
    /// Cases in the set.
    pub cases: usize,
    /// Cases skipped (`MemoryCase::skip`).
    pub skipped: usize,
    /// Cases an error ended.
    pub errors: usize,
    /// Gold facts the user never asked to be remembered.
    pub unasked_facts: usize,
    /// Of those, the ones saved.
    pub unasked_saved: usize,
    /// Remembers on its own: `unasked_saved` over `unasked_facts`.
    pub remembered_share: Option<f64>,
    /// Gold facts the user asked to be remembered.
    pub explicit_facts: usize,
    /// Of those, the ones saved.
    pub explicit_saved: usize,
    /// Gold facts saved whole, of all those saved.
    pub whole_facts: usize,
    /// Whole facts: `whole_facts` over the gold facts saved.
    pub whole_share: Option<f64>,
    /// Chat messages with nothing to keep (the probes not included).
    pub negative_messages: usize,
    /// Of those, the ones whose turn saved a fact.
    pub false_saves: usize,
    /// No false saves: `false_saves` over `negative_messages`, the plan's 2% target.
    pub false_save_share: Option<f64>,
    /// Probe questions asked.
    pub probe_turns: usize,
    /// Of those, the ones whose turn saved a fact.
    pub probe_saves: usize,
    /// `never` sets held at the end.
    pub never_held: usize,
    /// Facts that look like a secret at the end.
    pub secrets: usize,
    /// Gold facts that correct another (`GoldFact::stale`).
    pub corrections: usize,
    /// Of those, the ones saved whole with no stale fact beside them.
    pub corrected: usize,
    /// Wrong merges: updates that overwrote a gold or kept fact, and `keep` sets lost.
    pub merges: usize,
    /// Probes scored in Rust.
    pub probes: usize,
    /// Of those, the ones answered right.
    pub probes_right: usize,
    /// Used later: `probes_right` over `probes`.
    pub used_share: Option<f64>,
    /// Probes left for a human check.
    pub probe_checks: usize,
    /// The time to first token of every turn after a chat's first (turn N+1); `None` with none.
    pub later_ttft_ms: Option<P50P95>,
    /// The cached prompt tokens of those turns, summed.
    pub later_cached_tokens: u64,
    /// `remember` y/n questions answered yes (asked after an untrusted result).
    pub approvals: usize,
    /// `remember` calls on messages that asked for the save.
    pub remember_explicit: usize,
    /// `remember` calls on other fact messages: saves the model made on its own.
    pub remember_unasked: usize,
    /// `remember` calls on messages with nothing to keep, and on probes.
    pub remember_negative: usize,
}

/// A `--memory-chats` run.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryChats {
    /// The set's file.
    pub chats: PromptsFile,
    /// One per case, in file order.
    pub rows: Vec<MemoryChatRow>,
    /// The numbers.
    pub summary: MemoryChatSummary,
}

fn share(n: usize, of: usize) -> Option<f64> {
    (of > 0).then(|| n as f64 / of as f64)
}

/// The numbers over `rows`, the set's `cases`.
pub fn memory_chat_summary(rows: &[MemoryChatRow]) -> MemoryChatSummary {
    let mut s = MemoryChatSummary {
        cases: rows.len(),
        ..MemoryChatSummary::default()
    };
    let mut later = Vec::new();
    for r in rows {
        if r.skipped.is_some() {
            s.skipped += 1;
            continue;
        }
        // Counted, not dropped: the cases that fail are often the hard ones.
        s.errors += usize::from(r.error.is_some());
        for g in &r.gold {
            match g.explicit {
                true => {
                    (s.explicit_facts, s.explicit_saved) = (
                        s.explicit_facts + 1,
                        s.explicit_saved + usize::from(g.saved),
                    )
                }
                false => {
                    (s.unasked_facts, s.unasked_saved) =
                        (s.unasked_facts + 1, s.unasked_saved + usize::from(g.saved))
                }
            }
            s.whole_facts += usize::from(g.saved && g.missing.is_empty());
        }
        s.negative_messages += r.turns.iter().filter(|t| t.no_save).count();
        s.false_saves += r.false_saves;
        s.probe_turns += r.turns.iter().filter(|t| t.probe).count();
        s.probe_saves += r.probe_saves;
        s.remember_explicit += r.remember_explicit;
        s.remember_unasked += r.remember_unasked;
        s.remember_negative += r.remember_negative;
        s.never_held += r.never_held.len();
        s.secrets += r.secrets;
        s.merges += r.merges + r.lost.len();
        for p in &r.probes {
            match p.right {
                Some(right) => {
                    (s.probes, s.probes_right) = (s.probes + 1, s.probes_right + usize::from(right))
                }
                None => s.probe_checks += 1,
            }
        }
        s.approvals += r.approvals;
        for pair in r.turns.windows(2) {
            if pair[0].chat == pair[1].chat {
                later.extend(pair[1].ttft_ms);
                s.later_cached_tokens += u64::from(pair[1].cached_tokens);
            }
        }
    }
    let saved = s.explicit_saved + s.unasked_saved;
    s.remembered_share = share(s.unasked_saved, s.unasked_facts);
    s.whole_share = share(s.whole_facts, saved);
    s.false_save_share = share(s.false_saves, s.negative_messages);
    s.used_share = share(s.probes_right, s.probes);
    later.sort_by(f64::total_cmp);
    s.later_ttft_ms = (!later.is_empty()).then(|| P50P95 {
        p50: nearest(&later, 50.0),
        p95: nearest(&later, 95.0),
    });
    s
}

/// Fills `MemoryChatSummary::corrections` and `corrected`, which need the cases' gold.
fn count_corrections(s: &mut MemoryChatSummary, cases: &[MemoryCase], rows: &[MemoryChatRow]) {
    for (c, r) in cases.iter().zip(rows) {
        if r.skipped.is_some() {
            continue;
        }
        for (f, g) in c.facts.iter().zip(&r.gold) {
            if !f.stale.is_empty() {
                s.corrections += 1;
                s.corrected += usize::from(g.saved && g.missing.is_empty() && !g.stale);
            }
        }
    }
}

/// A fact's text without the user's quoted sentence that background capture stores after it
/// (plan 2.1: a `Said: "..."` line): the bench scores what the model wrote, as the quote holds
/// every detail the user said.
pub fn sentence_of(text: &str) -> &str {
    let quote = (text.match_indices("Said:"))
        .find(|&(i, _)| i == 0 || text[..i].ends_with('\n'))
        .map(|(i, _)| i);
    text[..quote.unwrap_or(text.len())].trim()
}

/// The facts in `data` by slug, each its sentence (`sentence_of`).
fn facts(data: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    Ok(memory::list(data)?
        .into_iter()
        .map(|f| (f.slug, sentence_of(&f.text).to_owned()))
        .collect())
}

/// Scores the state after case `c`'s chats into `row`: each gold fact, through the facts in
/// `touched` (saved or changed by a turn of the case) alone, then `keep`, `never` and secrets.
fn score_facts(
    c: &MemoryCase,
    end: &BTreeMap<String, String>,
    touched: &BTreeSet<String>,
    row: &mut MemoryChatRow,
) {
    row.facts = (end.iter())
        .map(|(slug, text)| SavedFact {
            slug: slug.clone(),
            text: text.clone(),
        })
        .collect();
    for f in &c.facts {
        let key = &f.details[0];
        let held = |text: &str| f.details.iter().filter(|d| has(text, d)).count();
        // The key and one detail of this gold fact's own: a fact that names the key alone (the
        // dog's name) does not capture each gold fact about it (its breed, its age), and a
        // correction is captured only with what it changed.
        let enough = 2.min(f.details.len());
        let best = (end.iter())
            .filter(|(slug, text)| {
                touched.contains(*slug) && has(text, key) && held(text) >= enough
            })
            .max_by_key(|(_, text)| held(text));
        // Any fact, a seeded one too: about the same thing (one detail), holding the old value,
        // and short of the new one.
        let stale = (end.values()).any(|text| {
            f.stale.iter().any(|d| has(text, d)) && held(text) > 0 && held(text) < f.details.len()
        });
        row.gold.push(GoldRow {
            details: f.details.clone(),
            explicit: f.explicit,
            saved: best.is_some(),
            slug: best.map(|(slug, _)| slug.clone()),
            missing: (f.details.iter())
                .filter(|d| !best.is_some_and(|(_, text)| has(text, d)))
                .cloned()
                .collect(),
            stale,
        });
    }
    let held = |set: &[String]| end.values().any(|text| set.iter().all(|d| has(text, d)));
    row.lost = (c.keep.iter())
        .filter(|set| !held(set))
        .map(|set| set.join(" + "))
        .collect();
    row.never_held = (c.never.iter())
        .filter(|set| held(set))
        .map(|set| set.join(" + "))
        .collect();
    row.secrets = end.values().filter(|text| secret_shaped(text)).count();
}

/// The facts saved or changed between `before` and `after`, and how many of the changes took a
/// key of `keys` out of a fact.
fn changes(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
    keys: &[&String],
) -> (Vec<String>, usize) {
    let mut saved = Vec::new();
    let mut merges = 0;
    for (slug, text) in after {
        match before.get(slug) {
            Some(old) if old == text => {}
            Some(old) => {
                saved.push(slug.clone());
                merges += usize::from(keys.iter().any(|k| has(old, k) && !has(text, k)));
            }
            None => saved.push(slug.clone()),
        }
    }
    (saved, merges)
}

/// Runs each case of `set` in a throwaway data folder of its own, whose config.toml is
/// `routing.config`'s with the run's settings (`write_config`), with nothing attached (with
/// `routing.app`, the desktop app's options instead, their differences in `report.app`). Each
/// turn runs to its end. `save` writes the report after each case; `progress` gets a line for
/// each.
#[allow(clippy::too_many_arguments)]
pub fn run_memory_chats(
    larch: &Larch,
    thinking: Thinking,
    routing: &Routing,
    prefix_cache: bool,
    set: &MemoryChatSet,
    report: &mut Report,
    mut save: impl FnMut(&Report) -> anyhow::Result<()>,
    mut progress: impl FnMut(String),
) -> anyhow::Result<()> {
    report.memory_chats = Some(MemoryChats {
        chats: set.file.clone(),
        ..MemoryChats::default()
    });
    // Each case has a data folder of its own, so the run's logs add up case by case.
    if let Some(to) = set.audit.as_deref() {
        clear_logs(to)?;
    }
    for c in &set.cases {
        let started = Instant::now();
        let mut row = MemoryChatRow {
            id: c.id.clone(),
            tags: c.tags.clone(),
            skipped: c.skip.clone(),
            ..MemoryChatRow::default()
        };
        if c.skip.is_none() {
            let data = tempfile::tempdir().context("making a data folder for a memory chat")?;
            write_config(data.path(), routing, thinking, routing.config.as_deref())?;
            let (opts, mut differs) = chat_options(data.path(), Vec::new(), routing, prefix_cache)?;
            if routing.app && report.app.is_none() {
                differs.extend(settings_differ(routing, thinking)?);
                for line in &differs {
                    progress(format!("--app: {line}"));
                }
                report.app = Some(differs);
            }
            if let Err(e) = run_case(larch, &opts, routing.xml_body, c, &mut row) {
                row.error.get_or_insert(format!("{e:#}"));
            }
            if let Some(to) = set.audit.as_deref() {
                append_logs(data.path(), to)?;
            }
        }
        row.wall_ms = ms(started);
        progress(case_line(&row));
        if let Some(m) = &mut report.memory_chats {
            m.rows.push(row);
            m.summary = memory_chat_summary(&m.rows);
            count_corrections(&mut m.summary, &set.cases, &m.rows);
        }
        save(report)?;
    }
    Ok(())
}

/// A case's progress line.
fn case_line(row: &MemoryChatRow) -> String {
    if let Some(why) = &row.skipped {
        return format!("{} skipped: {why}", row.id);
    }
    let saved = row.gold.iter().filter(|g| g.saved).count();
    let whole = (row.gold.iter())
        .filter(|g| g.saved && g.missing.is_empty())
        .count();
    let right = row.probes.iter().filter(|p| p.right == Some(true)).count();
    let scored = row.probes.iter().filter(|p| p.right.is_some()).count();
    format!(
        "{} saved {saved}/{} (whole {whole}), false saves {}, merges {}, probes {right}/{scored}, {:.1} s{}",
        row.id,
        row.gold.len(),
        row.false_saves,
        row.merges + row.lost.len(),
        row.wall_ms / 1e3,
        row.error
            .as_ref()
            .map(|e| format!("; error: {e}"))
            .unwrap_or_default()
    )
}

/// Case `c` in the data folder of `opts`, into `row`. A failed turn is recorded and the case
/// goes on, so its facts are still scored.
fn run_case(
    larch: &Larch,
    opts: &AgentOptions,
    xml_body: bool,
    c: &MemoryCase,
    row: &mut MemoryChatRow,
) -> anyhow::Result<()> {
    let data = opts.data_dir.as_path();
    for fact in &c.memory {
        memory::remember(data, fact, "user", "bench")?;
    }
    let keys: Vec<&String> = (c.facts.iter().map(|f| &f.details[0]))
        .chain(c.keep.iter().map(|set| &set[0]))
        .collect();
    let mut known = facts(data)?;
    let mut touched = BTreeSet::new();
    if c.off {
        memory::set_enabled(data, false)?;
    }
    for (n, chat) in c.chats.iter().enumerate() {
        let mut agent = Agent::new(larch, opts.clone())?;
        agent.set_xml_body(xml_body);
        for m in chat {
            let mut turn = chat_turn(larch, &mut agent, n, &m.user, row);
            (turn.no_save, turn.explicit) = (m.no_save, m.explicit);
            // A save from a background capture lands after the turn: it is waited for.
            agent.wait_for_capture();
            let now = facts(data)?;
            let (saved, merges) = changes(&known, &now, &keys);
            row.merges += merges;
            row.false_saves += usize::from(m.no_save && !saved.is_empty());
            touched.extend(saved.iter().cloned());
            known = now;
            record(row, MemoryTurn { saved, ..turn });
        }
    }
    if c.off {
        memory::set_enabled(data, true)?;
    }
    // Scored before the forgets, so a fact saved and then forgotten counts as saved.
    score_facts(c, &known, &touched, row);
    for word in &c.forget {
        for (slug, text) in &known {
            if has(text, word) {
                memory::forget(data, slug)?;
                row.forgot += 1;
            }
        }
        known = facts(data)?;
    }
    let captured = row.gold.iter().all(|g| g.saved);
    for (i, p) in c.probes.iter().enumerate() {
        let shown = memory::system_prefix(data);
        let mut agent = Agent::new(larch, opts.clone())?;
        agent.set_xml_body(xml_body);
        let mut turn = chat_turn(larch, &mut agent, c.chats.len() + i, &p.user, row);
        turn.probe = true;
        agent.wait_for_capture();
        let now = facts(data)?;
        let (saved, merges) = changes(&known, &now, &keys);
        row.merges += merges;
        row.probe_saves += usize::from(!saved.is_empty());
        known = now;
        let mut misses: Vec<String> = (p.answer.iter())
            .filter(|d| !has(&turn.answer, d))
            .map(|d| format!("lacks {d}"))
            .collect();
        misses.extend(
            (p.not.iter())
                .filter(|d| has(&turn.answer, d))
                .map(|d| format!("holds {d}")),
        );
        let scored = !(p.answer.is_empty() && p.not.is_empty());
        let right = scored.then_some(misses.is_empty() && turn.error.is_none());
        row.probes.push(ProbeRow {
            user: p.user.clone(),
            answer: turn.answer.clone(),
            captured,
            retrieved: (!p.answer.is_empty() && p.answer.iter().all(|d| has(&shown, d)))
                || turn.tools.iter().any(|t| t == "recall"),
            right,
            misses,
        });
        record(row, MemoryTurn { saved, ..turn });
    }
    Ok(())
}

/// Adds `turn` to `row`: its `remember` calls by the kind of message, and its error as the
/// case's first.
fn record(row: &mut MemoryChatRow, turn: MemoryTurn) {
    let calls = turn.tools.iter().filter(|t| *t == "remember").count();
    match (turn.explicit, turn.no_save || turn.probe) {
        (true, _) => row.remember_explicit += calls,
        (false, true) => row.remember_negative += calls,
        (false, false) => row.remember_unasked += calls,
    }
    if row.error.is_none() {
        row.error.clone_from(&turn.error);
    }
    row.turns.push(turn);
}

/// One user message as a turn of `agent`'s chat `chat`, run to its end: a `remember` y/n is
/// answered yes, every other one no, each counted in `row`.
fn chat_turn(
    larch: &Larch,
    agent: &mut Agent,
    chat: usize,
    user: &str,
    row: &mut MemoryChatRow,
) -> MemoryTurn {
    let started = Instant::now();
    let before = steps_of(agent).map_or(0, |s| s.len());
    let mut heard = Heard::new();
    let mut ran = || -> anyhow::Result<()> {
        let turn = agent.turn_with(larch, user)?;
        let mut parked = listen(turn, true, &mut heard);
        while let Some(call) = parked {
            let yes = call.name == "remember";
            match yes {
                true => row.approvals += 1,
                false => row.declined += 1,
            }
            parked = listen(agent.confirm(yes)?, true, &mut heard);
        }
        anyhow::ensure!(
            heard.ended,
            "{}",
            (heard.notice.clone()).unwrap_or_else(|| "the turn ended without an answer".into())
        );
        Ok(())
    };
    let error = ran().err().map(|e| format!("{e:#}"));
    let steps: Vec<AuditEvent> = steps_of(agent)
        .map(|s| s.into_iter().skip(before).collect())
        .unwrap_or_default();
    let steps = steps.iter().filter(|s| s.kind == EventKind::Turn);
    let mut turn = MemoryTurn {
        chat,
        user: user.into(),
        tools: heard.calls.iter().map(|c| c.name.clone()).collect(),
        answer: heard.said,
        error,
        wall_ms: ms(started),
        ..MemoryTurn::default()
    };
    for s in steps {
        if turn.ttft_ms.is_none() {
            turn.ttft_ms = s.ttft_ms;
        }
        turn.prompt_tokens += s.prompt_tokens.unwrap_or_default();
        turn.cached_tokens += s.cached_tokens.unwrap_or_default();
    }
    turn
}

/// The run as the bench prints it at its end: one line per section 4 measure of the memory
/// plan, then each case's line.
pub fn memory_chats_table(m: &MemoryChats) -> String {
    let s = &m.summary;
    let pct = |v: Option<f64>| v.map_or("-".into(), |v| format!("{:.0}%", v * 100.0));
    let mut out = format!(
        "memory chats: {} cases, {} skipped, {} errors\n\
         remembers on its own  {} ({}/{})\n\
         explicit saves        {}/{} ({} approvals)\n\
         whole facts           {} ({}/{} saved)\n\
         false saves           {} ({}/{} messages), never-sets held {}, secrets {}\n\
         probe saves           {}/{} probes\n\
         remember calls        {} explicit, {} unasked, {} on negatives\n\
         corrections           {}/{}, wrong merges {}\n\
         used later            {} ({}/{}), {} for a human check\n",
        s.cases,
        s.skipped,
        s.errors,
        pct(s.remembered_share),
        s.unasked_saved,
        s.unasked_facts,
        s.explicit_saved,
        s.explicit_facts,
        s.approvals,
        pct(s.whole_share),
        s.whole_facts,
        s.unasked_saved + s.explicit_saved,
        pct(s.false_save_share),
        s.false_saves,
        s.negative_messages,
        s.never_held,
        s.secrets,
        s.probe_saves,
        s.probe_turns,
        s.remember_explicit,
        s.remember_unasked,
        s.remember_negative,
        s.corrected,
        s.corrections,
        s.merges,
        pct(s.used_share),
        s.probes_right,
        s.probes,
        s.probe_checks,
    );
    if let Some(t) = s.later_ttft_ms {
        out += &format!(
            "turn N+1 TTFT         p50 {:.0} ms, p95 {:.0} ms, {} cached tokens\n",
            t.p50, t.p95, s.later_cached_tokens
        );
    }
    for r in &m.rows {
        out += &case_line(r);
        out.push('\n');
    }
    out
}
