//! The Folio bench (`larch-agent folio-bench`): one chat on real documents filled to a context,
//! read once and pinned, then asked questions about the last document, with the numbers a GPU run
//! needs in one `larch-agent-folio/2` report: the first read, each answer's first token, what the
//! device holds (`Larch::document_fit`), the quote check's verdict on each answer, and the speed
//! of a plain chat before and after the pin. Then the two steps a pin is for: a switch to another
//! chat and back, and an engine restart. `--plain` runs the same steps with the documents as plain
//! text in the system prompt and no pin, on an engine with the same memory plan, as the baseline.
//! `--idle` first lets the pin go as the app does after a pause, waits for the plain plan to come
//! back, measures a plain chat on it, and asks the questions again, `--idle-cycles` times (`Idle`).

use std::{
    fs,
    path::PathBuf,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

use anyhow::Context;
use larch::{GenParams, Larch, Prompt, TextEvent, api::TURN_ROOM};
use larch_engine::{DOC_CONTEXT_STEP, DOC_POOL_FRACTION, EngineHandle, ReloadChange};
use larch_model::chat::Message;
use serde::{Deserialize, Serialize};

use crate::{
    Agent, AgentOptions,
    folio::{Documents, cite, map, source},
    router::Router,
    run::AgentEvent,
    skills::read_papers::MAX_PAPERS,
};

/// The report's schema name.
pub const SCHEMA: &str = "larch-agent-folio/2";
/// Tokens the block's headers, the agent's own system text and the question take, which the
/// documents' own counts leave out of the fill.
const FRAME: u32 = 2048;
/// Plain generations whose median speed is the chat's decode speed.
const DECODE_RUNS: usize = 3;
/// Tokens of each plain generation.
const DECODE_TOKENS: u32 = 128;
/// A prompt no model answers short, so each run decodes all its tokens.
const DECODE_PROMPT: &str = "Count upward from 1, one number per line, and do not stop.";
/// What the other chat is asked at a switch: a short answer that decodes at full speed.
const SWITCH_PROMPT: &str = "Count upward from 1 to 100, one number per line.";
/// The shortest quote or gold passage, normalized, that counts as the same text as another.
const OVERLAP: usize = 12;

/// One question of the set, about the last document.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    /// The question's name in the report.
    pub id: String,
    /// What the user asks.
    pub question: String,
    /// Passages of the document's last pages that hold the answer; a quote the check matched on
    /// its page counts when it is one of them or inside one.
    pub gold: Vec<String>,
    /// Texts of which the right answer holds any one, letters and digits compared.
    pub expect: Vec<String>,
}

/// The questions of a JSON Lines file.
pub fn questions(path: &std::path::Path) -> anyhow::Result<Vec<Question>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    (text.lines().filter(|l| !l.trim().is_empty()))
        .enumerate()
        .map(|(n, line)| {
            serde_json::from_str(line).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

/// The documents `fill` chose, or left out, with their tokens.
#[derive(Debug, Serialize)]
pub struct Sized {
    /// The file's name.
    pub name: String,
    /// Its tokens.
    pub tokens: u32,
}

/// A plain chat's decode speed.
#[derive(Debug, Serialize)]
pub struct Decode {
    /// Tokens a second after the first, of each run.
    pub tok_s: Vec<f64>,
    /// The median of `tok_s`.
    pub median: f64,
}

/// One question's answer and what was measured on it.
#[derive(Debug, Serialize)]
pub struct Asked {
    /// The question's id.
    pub id: String,
    /// The answer, with the notes the chat wrote under it.
    pub answer: String,
    /// The engine's time to first token (a turn's last step), milliseconds.
    pub ttft_ms: f64,
    /// The turn's last step's decode speed, tokens a second.
    pub tok_s: f64,
    /// Send to the first answer text, milliseconds; none when no text came.
    pub first_text_ms: Option<f64>,
    /// Send to the end of the turn, milliseconds.
    pub total_ms: f64,
    /// Prompt tokens the engine did not read again.
    pub cached_tokens: u32,
    /// Quotes of the answer with a citation, and of those the ones found on their cited page.
    pub cited: usize,
    /// Quotes matched on their cited page.
    pub matched: usize,
    /// A matched quote is one of the question's gold passages, or inside one.
    pub quote_ok: bool,
    /// The answer holds one of the expected texts.
    pub value_ok: bool,
    /// Why the turn gave no answer.
    pub error: Option<String>,
    /// The tools the turn called, in order: a call's result is read before the answer, which is
    /// what a slow last step's first token can be.
    pub calls: Vec<String>,
}

/// What the run came to, for the table.
#[derive(Debug, Serialize)]
pub struct Summary {
    /// Questions asked.
    pub asked: usize,
    /// Of them, with a gold quote matched.
    pub quote_ok: usize,
    /// Of them, holding an expected text.
    pub value_ok: usize,
    /// The longest engine time to first token of the second question on, milliseconds.
    pub later_ttft_max_ms: Option<f64>,
    /// The longest send-to-first-text of the second question on, milliseconds.
    pub later_first_text_max_ms: Option<f64>,
}

/// The `larch-agent-folio/2` report.
#[derive(Debug, Serialize)]
pub struct Report {
    /// `SCHEMA`.
    pub schema: &'static str,
    /// The model's alias.
    pub model: String,
    /// `"pinned"` (the documents pinned in the engine) or `"plain"` (the baseline: plain text in
    /// the system prompt, no pin, the same memory plan).
    pub mode: &'static str,
    /// The context asked for: the documents fill it less a turn's room.
    pub context: u32,
    /// The context the engine loaded for the documents (0 when nothing was pinned).
    pub plan_context: u32,
    /// The documents read, in letter order, with their tokens.
    pub documents: Vec<Sized>,
    /// The documents that did not fit with them.
    pub left_out: Vec<Sized>,
    /// Tokens the block holds, as the read counted them.
    pub tokens: u32,
    /// The set fits the documents plan (`DocFit::fits`).
    pub fits: bool,
    /// The longest document the plan holds, tokens.
    pub max_tokens: u32,
    /// Expert layers the plan keeps in host memory for this set.
    pub host_layers: u32,
    /// The seats the documents plan gives this set (`DocFit::seats`).
    pub seats: u32,
    /// The chip the read ended on ("Ready", "Fits up to 12 pages on this computer").
    pub chip: String,
    /// The attach-to-ready time, milliseconds; none when the set was not read whole.
    pub first_read_ms: Option<f64>,
    /// A plain chat's speed before the documents were read, on the default plan.
    pub decode_default: Decode,
    /// The same, with the documents held.
    pub decode_documents: Decode,
    /// Each question, in order.
    pub questions: Vec<Asked>,
    /// The totals.
    pub summary: Summary,
    /// The first question asked again in the document chat after another chat was used; none
    /// when there were no questions.
    pub after_switch: Option<Asked>,
    /// The other chat's decode speed at the switch, tokens a second.
    pub other_chat_tok_s: Option<f64>,
    /// Pinned: the time `read_documents` took when focus returned, milliseconds (near 0 while the
    /// pin stays; an import when it gave way to the other chat).
    pub switch_back_ms: Option<f64>,
    /// The first question asked again after the engine restarted and the chat was opened again.
    pub after_restart: Option<Asked>,
    /// The restart's engine load, with the plan's reload in plain mode, milliseconds.
    pub restart_load_ms: Option<f64>,
    /// From opening the chat after the restart to its first answer text, milliseconds.
    pub restart_open_ms: Option<f64>,
    /// The restart's open, step by step: the chat opened, then each chip state its documents
    /// went through, milliseconds from the start of the open.
    pub restart_stamps: Vec<Stamp>,
    /// The document chat's id, which `reopen` opens again from a kept data folder.
    pub chat: String,
    /// `--idle`: the pin let go after the questions and what came after; none without it.
    pub idle: Option<Idle>,
}

/// The engine's memory plan as a reload left it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Plan {
    /// The loaded context, tokens.
    pub context: u32,
    /// The KV pool's blocks.
    pub kv_pool_blocks: u32,
    /// The seats (`max_batch_seqs`).
    pub seats: u32,
    /// Device memory free when the load began (`EngineHandle::free_before`), bytes: flat across
    /// reloads when each finds the device as the first load did.
    pub free_before: u64,
}

impl Plan {
    /// The plan loaded now.
    fn of(larch: &Larch, engine: &EngineHandle) -> Plan {
        let seats = larch::knob::spec("max_batch_seqs")
            .map_or(0.0, |k| larch.knobs().effective().values[k.id as usize]);
        Plan {
            context: crate::max_context(larch),
            kv_pool_blocks: larch.deployment().kv_pool_blocks,
            seats: seats as u32,
            free_before: engine.free_before(),
        }
    }
}

/// `--idle`: after the questions, `--idle-cycles` times: the chat's pin let go with its file
/// written, as the app's idle release does, the wait for the engine to restore the plain plan, a
/// plain chat on it, and the questions asked again (an import and a documents reload first).
#[derive(Debug, Serialize)]
pub struct Idle {
    /// The longest wait for each restore, seconds (`--idle`).
    pub wait_secs: u64,
    /// Each cycle, in order.
    pub cycles: Vec<IdleCycle>,
}

/// One `--idle` cycle.
#[derive(Debug, Serialize)]
pub struct IdleCycle {
    /// What the release did (`Released`, as it debugs): `Covered` when the pin file held the pin.
    pub released: String,
    /// The engine restored the plain plan within the wait.
    pub restored: bool,
    /// From the release to the restore, or to the end of the wait, milliseconds.
    pub waited_ms: f64,
    /// The plan the pin was held on, before the release.
    pub plan_before: Plan,
    /// The plan after the wait: the restored plain plan when `restored`.
    pub plan_after: Plan,
    /// A plain chat's speed after the wait, measured as `Report::decode_default`.
    pub decode_restored: Decode,
    /// `read_documents` before the questions: the import and its documents reload, milliseconds.
    pub read_ms: f64,
    /// That read, step by step, as `Report::restart_stamps`.
    pub stamps: Vec<Stamp>,
    /// The plan that read loaded.
    pub plan_reread: Plan,
    /// Each question asked again, in order.
    pub questions: Vec<Asked>,
}

/// The `larch-agent-folio-reopen/1` report: one restart of a kept run's document chat, its pin
/// imported, then a plain chat's speed with the documents held (`Report::decode_documents`).
#[derive(Debug, Serialize)]
pub struct Reopened {
    /// `REOPEN_SCHEMA`.
    pub schema: &'static str,
    /// The engine's load, milliseconds.
    pub load_ms: f64,
    /// The load went straight into the chat's recorded documents plan (else the default plan, and
    /// the open reloads for the pin).
    pub planned: bool,
    /// From opening the chat to its documents held, milliseconds.
    pub open_ms: f64,
    /// The open, step by step, as `Report::restart_stamps`.
    pub stamps: Vec<Stamp>,
    /// A plain chat's speed with the documents held.
    pub decode_documents: Decode,
}

/// The reopen report's schema.
pub const REOPEN_SCHEMA: &str = "larch-agent-folio-reopen/1";

/// One step of an open and when it came.
#[derive(Debug, Serialize)]
pub struct Stamp {
    /// `"opened"` (the chat and its documents loaded), else the chip state, as `Documents` debugs.
    pub event: String,
    /// Milliseconds from the start of the open.
    pub ms: f64,
}

/// [`Run::reload`]: loads an engine, straight into the plan of the chat given when it holds a pin.
pub type Reload<'a> = dyn Fn(Option<&str>) -> anyhow::Result<(Larch, EngineHandle)> + 'a;

/// What a run needs.
pub struct Run<'a> {
    /// The model's alias, for the report.
    pub model: String,
    /// The context asked for, tokens.
    pub context: u32,
    /// The engine's data folder (the chat and its extract cache).
    pub data_dir: PathBuf,
    /// The model's architecture.
    pub arch: String,
    /// The model's tool-call body is XML (`load`'s fourth value), as `chat` tells its agent.
    pub xml_body: bool,
    /// The engine keeps a prefix cache.
    pub prefix_cache: bool,
    /// The baseline: hold the documents as plain text, pin nothing.
    pub plain: bool,
    /// `--idle`: after the questions, let the pin go and wait at most this long for the plain
    /// plan (`Idle`).
    pub idle: Option<Duration>,
    /// `--idle-cycles`: how many times `idle` runs, 1 or more.
    pub idle_cycles: u32,
    /// Loads the engine again for the restart step: as at the start of the run, or straight into
    /// the documents plan of the chat it will reopen when that chat holds a pin.
    pub reload: &'a Reload<'a>,
    /// The candidate documents; the last by file name is the one the questions are about.
    pub docs: &'a [PathBuf],
    /// The questions.
    pub questions: &'a [Question],
}

/// Which of `sizes` (tokens, documents in name order) to read: the last always, then the others
/// largest first while the total stays within `budget` and `MAX_PAPERS`, as indices in order.
pub fn fill(sizes: &[u32], budget: u32) -> Vec<usize> {
    let last = sizes.len() - 1;
    let mut total = sizes[last];
    let mut picked = vec![last];
    let mut order: Vec<usize> = (0..last).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sizes[i]));
    for i in order {
        if picked.len() < MAX_PAPERS && total.saturating_add(sizes[i]) <= budget {
            total += sizes[i];
            picked.push(i);
        }
    }
    picked.sort_unstable();
    picked
}

/// `text` as its lower-case letters and digits, so line breaks, hyphenation marks and spacing
/// differ without changing a comparison.
fn norm(text: &str) -> String {
    (text.chars().filter(|c| c.is_alphanumeric()))
        .flat_map(char::to_lowercase)
        .collect()
}

/// One answer's quote and value verdicts: whether a quote the check matched on its page is a gold
/// passage or inside one, and whether the answer holds an expected text.
pub fn score(q: &Question, answer: &str, checked: &cite::Checked) -> (bool, bool) {
    let gold: Vec<String> = q.gold.iter().map(|g| norm(g)).collect();
    let quote = checked.results.iter().any(|v| {
        let quote = norm(&v.quote);
        v.result == Ok(cite::CiteResult::Matched)
            && quote.len() >= OVERLAP
            && gold.iter().any(|g| g.contains(&quote) || quote.contains(g))
    });
    let said = norm(answer);
    let value = q.expect.iter().any(|e| said.contains(&norm(e)));
    (quote, value)
}

/// The context the documents plan loads for `tokens` of documents: what `Larch::pin` reloads to
/// (the document and a turn's room, to a step), which the engine's `document_context` keeps
/// private.
pub fn plan_context(tokens: u32) -> u32 {
    tokens
        .saturating_add(TURN_ROOM)
        .next_multiple_of(DOC_CONTEXT_STEP)
}

/// Reloads `engine` to the documents plan a pin would get (`Larch::prepare_document`), without
/// the pin or its way back to the old plan, for the plain baseline.
fn apply_plan(engine: &EngineHandle, context: u32, seats: u32) -> anyhow::Result<()> {
    engine
        .reload(ReloadChange {
            max_context: Some(context),
            kv_pool_fraction: Some(DOC_POOL_FRACTION),
            max_batch_seqs: Some(seats),
            drafter_off: true,
            ..ReloadChange::default()
        })
        .map_err(|e| anyhow::anyhow!("reloading for the documents plan: {e:?}"))
}

/// The median speed of `DECODE_RUNS` plain generations on the loaded plan.
fn decode(larch: &Larch) -> anyhow::Result<Decode> {
    let mut tok_s = Vec::new();
    for _ in 0..DECODE_RUNS {
        let prompt = Prompt::Messages(vec![Message {
            role: "user".into(),
            content: DECODE_PROMPT.into(),
        }]);
        let params = GenParams {
            max_tokens: DECODE_TOKENS,
            temperature: Some(0.0),
            thinking: Some(false),
            prefix_cache: false,
            ..GenParams::default()
        };
        for event in larch.generate(prompt, params)? {
            match event {
                TextEvent::Done { timings, .. } => tok_s.push(timings.tok_s),
                TextEvent::Error(e) => anyhow::bail!("a plain generation failed: {e:?}"),
                _ => {}
            }
        }
    }
    let mut sorted = tok_s.clone();
    sorted.sort_by(f64::total_cmp);
    Ok(Decode {
        median: sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        tok_s,
    })
}

/// The documents `r` reads: the files it may open, then the read and the left-out ones with their
/// tokens, as counted from the extract cache the chat's own read fills.
fn choose(
    larch: &Larch,
    r: &Run,
    stop: &AtomicBool,
) -> anyhow::Result<(Vec<PathBuf>, Vec<Sized>, Vec<Sized>)> {
    let mut files = r.docs.to_vec();
    files.sort_by_key(|p| p.file_name().map(ToOwned::to_owned));
    // Counted from the extract cache the chat's own read fills, so each file is extracted once.
    let mut sized = Vec::new();
    for file in &files {
        let read = source::text(&r.data_dir, file, stop, None)
            .map_err(|e| anyhow::anyhow!("{}: {e}", file.display()))?;
        let name = file.file_name().unwrap_or_default().to_string_lossy();
        let tokens = u32::try_from(larch.token_ids(&read.text).len()).unwrap_or(u32::MAX);
        sized.push(Sized {
            name: name.into(),
            tokens,
        });
    }
    let counts: Vec<u32> = sized.iter().map(|s| s.tokens).collect();
    let picked = fill(&counts, r.context.saturating_sub(TURN_ROOM + FRAME));
    let allowed: Vec<PathBuf> = picked.iter().map(|&i| files[i].clone()).collect();
    let (mut documents, mut left_out) = (Vec::new(), Vec::new());
    for (i, s) in sized.into_iter().enumerate() {
        match picked.contains(&i) {
            true => documents.push(s),
            false => left_out.push(s),
        }
    }
    Ok((allowed, documents, left_out))
}

/// Opens `chat` of a kept run's data folder again on a fresh engine, as the run's restart does,
/// and measures a plain chat with its documents held.
/// `first` is the engine as loaded, in `load_ms`, straight into the chat's plan when `planned`.
pub fn reopen(
    first: (Larch, EngineHandle),
    (load_ms, planned): (f64, bool),
    r: &Run,
    chat: &str,
) -> anyhow::Result<Reopened> {
    let stop = AtomicBool::new(false);
    let (larch, _engine) = first;
    let (allowed, ..) = choose(&larch, r, &stop)?;
    let (mut stamps, opened) = (Vec::new(), Instant::now());
    let (agent, ..) = open_documents(&larch, r, &allowed, Some(chat), &stop, &mut stamps)?;
    let open_ms = opened.elapsed().as_secs_f64() * 1e3;
    let decode_documents = decode(&larch)?;
    drop(agent);
    Ok(Reopened {
        schema: REOPEN_SCHEMA,
        load_ms,
        planned,
        open_ms,
        stamps,
        decode_documents,
    })
}

/// Runs the bench: a plain decode, the documents chosen and read, each question asked in the one
/// chat, and a second plain decode with the documents held.
pub fn run(first: (Larch, EngineHandle), r: &Run) -> anyhow::Result<Report> {
    anyhow::ensure!(!r.docs.is_empty(), "no documents given");
    let stop = AtomicBool::new(false);
    let (mut larch, mut engine) = first;
    let decode_default = decode(&larch)?;
    let (allowed, documents, left_out) = choose(&larch, r, &stop)?;
    // The plain baseline gets the memory plan the pin would, loaded before its chat opens.
    let plan = (r.plain).then(|| {
        let tokens = documents
            .iter()
            .map(|d| d.tokens)
            .sum::<u32>()
            .saturating_add(FRAME);
        (plan_context(tokens), larch.document_fit(tokens))
    });
    if let Some((context, fit)) = &plan {
        anyhow::ensure!(fit.fits, "the documents do not fit the documents plan");
        apply_plan(&engine, *context, fit.seats)?;
    }
    let (mut agent, mut tokens, chip, first_read_ms) =
        open_documents(&larch, r, &allowed, None, &stop, &mut Vec::new())?;
    if tokens == 0 {
        tokens = documents.iter().map(|d| d.tokens).sum();
    }
    let fit = larch.document_fit(tokens);
    let mut asked = Vec::new();
    for q in r.questions {
        asked.push(ask(&larch, &mut agent, q));
    }
    let decode_documents = decode(&larch)?;
    let idle = match r.idle {
        Some(wait) => {
            let mut cycles = Vec::new();
            for n in 1..=r.idle_cycles.max(1) {
                cycles.push(idle(&larch, &engine, &mut agent, r, wait, &stop, n)?);
            }
            Some(Idle {
                wait_secs: wait.as_secs(),
                cycles,
            })
        }
        None => None,
    };
    let later = |f: fn(&Asked) -> Option<f64>| {
        (asked.iter().skip(1).filter_map(f))
            .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))))
    };
    let summary = Summary {
        asked: asked.len(),
        quote_ok: asked.iter().filter(|a| a.quote_ok).count(),
        value_ok: asked.iter().filter(|a| a.value_ok).count(),
        later_ttft_max_ms: later(|a| Some(a.ttft_ms)),
        later_first_text_max_ms: later(|a| a.first_text_ms),
    };
    // The switch: another chat is used, then the first question is asked again.
    let (mut other_chat_tok_s, mut after_switch, mut switch_back_ms) = (None, None, None);
    let mut other = Agent::new(&larch, options(r, Vec::new()))?;
    other.set_xml_body(r.xml_body);
    if let Some(q) = r.questions.first() {
        let plain_q = Question {
            id: "other".into(),
            question: SWITCH_PROMPT.into(),
            gold: Vec::new(),
            expect: Vec::new(),
        };
        other_chat_tok_s = Some(ask(&larch, &mut other, &plain_q).tok_s);
        if !r.plain {
            let began = Instant::now();
            agent.read_documents(&[], &stop, &mut |_| {}, &mut || {})?;
            switch_back_ms = Some(began.elapsed().as_secs_f64() * 1e3);
        }
        after_switch = Some(ask(&larch, &mut agent, q));
    }
    // The restart: everything dropped, the engine loaded as at the start, the chat opened again.
    let (mut after_restart, mut restart_load_ms, mut restart_open_ms) = (None, None, None);
    let mut restart_stamps = Vec::new();
    let chat = agent.chat_id().to_owned();
    if let Some(q) = r.questions.first() {
        drop((agent, other));
        let began = Instant::now();
        drop((larch, engine));
        // The plain baseline applies its plan below; a pinned chat's plan was recorded with it.
        (larch, engine) = (r.reload)((!r.plain).then_some(chat.as_str()))?;
        if let Some((context, fit)) = &plan {
            apply_plan(&engine, *context, fit.seats)?;
        }
        restart_load_ms = Some(began.elapsed().as_secs_f64() * 1e3);
        let opened = Instant::now();
        let id = Some(chat.as_str());
        let (mut agent, ..) = open_documents(&larch, r, &allowed, id, &stop, &mut restart_stamps)?;
        let open = opened.elapsed().as_secs_f64() * 1e3;
        let row = ask(&larch, &mut agent, q);
        restart_open_ms = Some(open + row.first_text_ms.unwrap_or(row.total_ms));
        after_restart = Some(row);
    }
    Ok(Report {
        schema: SCHEMA,
        model: r.model.clone(),
        mode: if r.plain { "plain" } else { "pinned" },
        context: r.context,
        plan_context: crate::max_context(&larch),
        documents,
        left_out,
        tokens,
        fits: fit.fits,
        max_tokens: fit.max_tokens,
        host_layers: fit.host_layers,
        seats: fit.seats,
        chip,
        first_read_ms,
        decode_default,
        decode_documents,
        questions: asked,
        summary,
        after_switch,
        other_chat_tok_s,
        switch_back_ms,
        after_restart,
        restart_load_ms,
        restart_open_ms,
        restart_stamps,
        chat,
        idle,
    })
}

/// One `--idle` cycle ([`IdleCycle`], the `n`th): lets `agent`'s pin go as the app's idle release
/// does, waits at most `wait` for the engine to restore the plain plan, then measures a plain chat
/// and asks the questions again. The restore is a reload: the plan it leaves differs from the
/// documents plan.
fn idle(
    larch: &Larch,
    engine: &EngineHandle,
    agent: &mut Agent,
    r: &Run,
    wait: Duration,
    stop: &AtomicBool,
    n: u32,
) -> anyhow::Result<IdleCycle> {
    let plan_before = Plan::of(larch, engine);
    let released = agent.release_documents(stop, &mut |_| {}, true);
    let began = Instant::now();
    let restored = loop {
        if Plan::of(larch, engine) != plan_before {
            break true;
        }
        if began.elapsed() >= wait {
            break false;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let waited_ms = began.elapsed().as_secs_f64() * 1e3;
    let plan_after = Plan::of(larch, engine);
    eprintln!(
        "folio-bench idle {n}: released {released:?}; {} after {:.1} s",
        if restored {
            "the plain plan came back"
        } else {
            "no restore"
        },
        waited_ms / 1e3
    );
    let decode_restored = decode(larch)?;
    let (mut stamps, opened) = (Vec::new(), Instant::now());
    agent.read_documents(
        &[],
        stop,
        &mut |e| {
            if let AgentEvent::Documents(d) = e {
                stamps.push(Stamp {
                    event: format!("{d:?}"),
                    ms: opened.elapsed().as_secs_f64() * 1e3,
                });
            }
        },
        &mut || {},
    )?;
    let read_ms = opened.elapsed().as_secs_f64() * 1e3;
    let plan_reread = Plan::of(larch, engine);
    let questions: Vec<Asked> = r.questions.iter().map(|q| ask(larch, agent, q)).collect();
    let first = questions.first().map_or(0.0, |a| a.ttft_ms);
    let gb = |p: &Plan| p.free_before as f64 / 1e9;
    println!(
        "idle {} {n}: restored {restored} in {:.1} s, free before {:.3} / restored {:.3} / read again {:.3} GB, plain {:.1} tok/s, read {:.1} s, first answer's first token {:.0} ms",
        r.model,
        waited_ms / 1e3,
        gb(&plan_before),
        gb(&plan_after),
        gb(&plan_reread),
        decode_restored.median,
        read_ms / 1e3,
        first
    );
    Ok(IdleCycle {
        released: format!("{released:?}"),
        restored,
        waited_ms,
        plan_before,
        plan_after,
        decode_restored,
        read_ms,
        stamps,
        plan_reread,
        questions,
    })
}

/// The options of a bench chat that may read `allowed`.
fn options(r: &Run, allowed: Vec<PathBuf>) -> AgentOptions {
    AgentOptions {
        data_dir: r.data_dir.clone(),
        allowed,
        arch: r.arch.clone(),
        router: Router::Off,
        prefix_cache: r.prefix_cache,
        memory_capture: false,
    }
}

/// The document chat, new or `chat` reopened, with its documents held: pinned (read, or imported
/// from the disk) or, for the baseline, plain text. Also the tokens the read counted, the chip it
/// ended on and the attach-to-ready time; `stamps` gets the open's steps.
fn open_documents(
    larch: &Larch,
    r: &Run,
    allowed: &[PathBuf],
    chat: Option<&str>,
    stop: &AtomicBool,
    stamps: &mut Vec<Stamp>,
) -> anyhow::Result<(Agent, u32, String, Option<f64>)> {
    let began = Instant::now();
    let stamp = |event: String| Stamp {
        event,
        ms: began.elapsed().as_secs_f64() * 1e3,
    };
    let opts = options(r, allowed.to_vec());
    let mut agent = match chat {
        Some(id) => Agent::open_attached(larch, id, opts)?,
        None => Agent::new(larch, opts)?,
    };
    stamps.push(stamp("opened".into()));
    agent.set_xml_body(r.xml_body);
    let (mut tokens, mut chip, mut first_read_ms) = (0, String::new(), None);
    if r.plain {
        agent.hold_plain_documents(stop)?;
        chip = "Plain".into();
    } else {
        let started = Instant::now();
        agent.read_documents(
            &[],
            stop,
            &mut |e| {
                let AgentEvent::Documents(d) = e else { return };
                stamps.push(stamp(format!("{d:?}")));
                match &d {
                    Documents::Reading { total, .. } => tokens = tokens.max(*total),
                    Documents::Answered { .. } => {}
                    // The read ends on one of these; MakingRoom comes before it, with the reload.
                    Documents::Ready
                    | Documents::FitsSlower { .. }
                    | Documents::FitsUpTo { .. } => {
                        first_read_ms = Some(started.elapsed().as_secs_f64() * 1e3);
                        chip = d.to_string();
                    }
                    _ => chip = d.to_string(),
                }
            },
            &mut || {},
        )?;
    }
    Ok((agent, tokens, chip, first_read_ms))
}

/// One question in the chat: its answer, its times and the quote check's verdict.
fn ask(larch: &Larch, agent: &mut Agent, q: &Question) -> Asked {
    let started = Instant::now();
    let mut row = Asked {
        id: q.id.clone(),
        answer: String::new(),
        ttft_ms: 0.0,
        tok_s: 0.0,
        first_text_ms: None,
        total_ms: 0.0,
        cached_tokens: 0,
        cited: 0,
        matched: 0,
        quote_ok: false,
        value_ok: false,
        error: None,
        calls: Vec::new(),
    };
    let mut notices = Vec::new();
    match agent.turn_with(larch, &q.question) {
        Err(e) => row.error = Some(e.to_string()),
        Ok(turn) => {
            for event in turn {
                match event {
                    AgentEvent::Text(text) => {
                        if !text.trim().is_empty() {
                            row.first_text_ms
                                .get_or_insert(started.elapsed().as_secs_f64() * 1e3);
                        }
                        row.answer += &text;
                    }
                    AgentEvent::Done(t) => {
                        (row.ttft_ms, row.tok_s, row.cached_tokens) =
                            (t.ttft_ms, t.tok_s, t.cached_tokens)
                    }
                    // A document question reads; a write is not what was asked, and a turn dropped
                    // here is cancelled.
                    AgentEvent::NeedsConfirm { call } => {
                        row.error = Some(format!("the turn asked to run {}", call.name));
                        break;
                    }
                    AgentEvent::Notice(n) => notices.push(n),
                    AgentEvent::ToolCall { name, .. } => row.calls.push(name),
                    _ => {}
                }
            }
        }
    }
    // A turn the engine refused has a notice and no text: not an answer, an error.
    if row.answer.trim().is_empty() && row.error.is_none() && !notices.is_empty() {
        row.error = Some(notices.join("; "));
    }
    row.total_ms = started.elapsed().as_secs_f64() * 1e3;
    if let Some(docs) = agent.pinned_documents() {
        let checked = cite::check(&row.answer, docs, &map::labels(docs));
        (row.cited, row.matched) = (checked.cited, checked.matched);
        (row.quote_ok, row.value_ok) = score(q, &row.answer, &checked);
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folio::{Rep, source::doc};

    #[test]
    fn the_plan_context_is_the_document_and_a_turn_rounded_up_to_a_step() {
        assert_eq!(
            plan_context(0),
            TURN_ROOM.next_multiple_of(DOC_CONTEXT_STEP)
        );
        assert_eq!(
            plan_context(DOC_CONTEXT_STEP * 3 - TURN_ROOM),
            DOC_CONTEXT_STEP * 3
        );
        assert_eq!(
            plan_context(DOC_CONTEXT_STEP * 3 - TURN_ROOM + 1),
            DOC_CONTEXT_STEP * 4
        );
    }

    #[test]
    fn fill_keeps_the_last_and_takes_the_largest_that_fit() {
        assert_eq!(fill(&[30, 10, 20, 5], 40), [0, 3]);
        assert_eq!(fill(&[30, 10, 20, 5], 100), [0, 1, 2, 3]);
        // The last stays when it alone passes the budget: the report then says it does not fit.
        assert_eq!(fill(&[1, 50], 10), [1]);
        // At most MAX_PAPERS documents.
        assert_eq!(fill(&[1; 20], u32::MAX).len(), MAX_PAPERS);
    }

    #[test]
    fn a_matched_quote_inside_gold_and_an_expected_text_are_what_scores() {
        let q = Question {
            id: "q".into(),
            question: "Price?".into(),
            gold: vec!["a rental price of 2 USD per GPU hour".into()],
            expect: vec!["2 USD".into()],
        };
        let text = "Intro.\n\nThe cost assumes a rental price of 2 USD per GPU hour.\n";
        let docs = vec![doc(0, "a.md", "sha", Rep::Text, text.into())];
        let said = "It is 2 USD. \"rental price of 2 USD per GPU hour\" [p. 1]";
        let checked = cite::check(said, &docs, &map::labels(&docs));
        assert_eq!(checked.matched, 1, "{checked:?}");
        assert_eq!(score(&q, said, &checked), (true, true));
        // The same quote cited on a page the text does not have is no match, and a wrong value fails.
        let off = "It is 3 USD. \"rental price of two dollars\" [p. 1]";
        let checked = cite::check(off, &docs, &map::labels(&docs));
        assert_eq!(score(&q, off, &checked), (false, false));
    }
}
