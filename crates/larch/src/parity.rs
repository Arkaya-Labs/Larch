//! `larch parity` (architecture 15, D16): llama.cpp's teacher-forced dump of a GGUF
//! (`tools/llama-logits`) against the production engine on the same GGUF, fed through
//! [`EngineHandle::force`] in each mode, one engine per mode, each compared with
//! [`oracle::compare`]. Exit 0 when every mode passes, 1 when a gate is missed or a run fails, 2 when
//! it could not run as asked (not run is never 0), 3 when the engine faults (9.11).

use std::cmp::Reverse;
use std::fs::File;
use std::io::{BufReader, Write};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Args, ValueEnum};
use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Forced, ForcedHandle, OnFault, Position, Request,
    RequestHandle, Usage,
};
use larch_model::oracle::{self, Dump, Prompt, Top};
use larch_records::{
    EngineError, EngineSample, LoadError, Read, ReadFrom, Record, RejectReason, RingHandle,
    knob_id, knob_spec,
};
use serde_json::json;

/// Which path feeds the dump's tokens; `all` is the other four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// One batch per prompt: the prompt-reading path.
    Prefill,
    /// One token per step: the production decode path.
    Decode,
    /// 64-token chunks, each step with a decode run's row; those decode runs are compared too.
    Chunked,
    /// The second half of each prompt on a session's parked first half (D14).
    Cached,
    /// Prefill, decode, chunked and cached.
    All,
}

/// `on` starts each engine with `decode_graphs=1` (refused, and reported, where the device has no
/// replay), `off` with 0: the control for graph replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Graphs {
    /// Captured decode graphs replayed where the device has them.
    On,
    /// No captured decode graphs.
    Off,
}

/// `larch parity` options.
#[derive(Args, Debug)]
pub struct Options {
    /// GGUF the dump was made from.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// llama-logits dump, plain JSONL (`<(zstd -dc x.jsonl.zst)` works; only a regular file's line
    /// carries its dump_sha256).
    #[arg(long = "ref")]
    pub reference: PathBuf,
    /// Mean-KL bound: the model's calibrated bound.
    #[arg(long, value_parser = finite(true))]
    pub bound: f64,
    /// Tie margin (nats): the model's calibrated margin; top-1 is gated where the reference's top two
    /// differ by more.
    #[arg(long, value_parser = finite(false))]
    pub margin: f64,
    /// Which path feeds the tokens.
    #[arg(long, value_enum, default_value_t = Mode::All)]
    pub mode: Mode,
    /// Decode-mode runs in flight: the decode engine's max_batch_seqs.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub parallel: u32,
    /// Captured decode graphs, or the graphs-off control.
    #[arg(long, value_enum, default_value_t = Graphs::On)]
    pub graphs: Graphs,
    /// The CPU reference.
    #[arg(long, conflicts_with = "device")]
    pub cpu: bool,
    /// GPU ordinal (default zero).
    #[arg(long)]
    pub device: Option<i32>,
    /// Startup knob assignment, repeatable. Refused: prefill_chunk_tokens, which each mode sets (one
    /// batch, or chunked's 64; any budget at or above a prompt computes the same one batch);
    /// max_batch_seqs, which --parallel sets; decode_graphs, which --graphs sets; telemetry_tier,
    /// as the path is read from every step's sample.
    #[arg(long = "set")]
    pub set: Vec<String>,
    /// Only the dump's first N prompts.
    #[arg(long)]
    pub prompts: Option<NonZeroUsize>,
    /// The GGUF the dump was made from, when it is not `-m`: scores `-m` (such as a `larch
    /// repack` of it, which llama.cpp cannot load) against that file's reference. Its sha256 binds
    /// the dump; each line records both files' hashes.
    #[arg(long = "ref-model")]
    pub ref_model: Option<PathBuf>,
}

/// Parser for a finite number above 0 (`strict`) or from 0 up.
fn finite(strict: bool) -> impl Fn(&str) -> Result<f64, String> + Clone + Send + Sync + 'static {
    move |s| match s.parse::<f64>() {
        Ok(v) if v.is_finite() && if strict { v > 0.0 } else { v >= 0.0 } => Ok(v),
        _ => Err(format!(
            "{s}: not a finite number {} 0",
            if strict { ">" } else { ">=" }
        )),
    }
}

/// `rate`, or none when it is over `of` nothing: 0 would read as a total miss.
fn over(of: usize, rate: f64) -> Option<f64> {
    (of > 0).then_some(rate)
}

/// Why there is no verdict.
enum Stop {
    /// Exit 2: inputs missing or wrong, or the engine could not start as asked.
    NotRun(anyhow::Error),
    /// Exit 1: a run failed.
    Failed(anyhow::Error),
}

/// Runs every requested mode on engines from `start` ([`larch_engine::start`]; tests wrap the model),
/// one JSON line each on `out`, and one `parity: ...` verdict on `err`; returns the exit code.
pub fn run(
    o: &Options,
    start: impl Fn(EngineOptions) -> Result<EngineHandle, LoadError>,
    out: &mut impl Write,
    err: &mut impl Write,
) -> i32 {
    let mut pass = true;
    let (code, verdict) = match gate(o, &start, out, &mut pass) {
        Ok(()) if pass => (0, "every mode passed".to_owned()),
        Ok(()) => (1, "a gate was missed".to_owned()),
        Err(Stop::Failed(e)) => (1, format!("a run failed: {e:#}")),
        // A miss already printed is not hidden behind a later mode that could not run.
        Err(Stop::NotRun(e)) if !pass => (1, format!("a gate was missed; then not run: {e:#}")),
        Err(Stop::NotRun(e)) => (2, format!("not run: {e:#}")),
    };
    // The exit code carries the verdict when stderr is gone.
    let _ = writeln!(err, "parity: {verdict}");
    code
}

/// Runs the modes in turn; a mode that misses its gate clears `pass`.
fn gate(
    o: &Options,
    start: &impl Fn(EngineOptions) -> Result<EngineHandle, LoadError>,
    out: &mut impl Write,
    pass: &mut bool,
) -> Result<(), Stop> {
    let mut sets = o
        .set
        .iter()
        .map(|s| crate::knob::assignment(s))
        .collect::<Result<Vec<_>>>()
        .map_err(Stop::NotRun)?;
    let own = [
        (knob_id::PREFILL_CHUNK_TOKENS, "each mode sets it"),
        (knob_id::MAX_BATCH_SEQS, "--parallel sets it"),
        (knob_id::DECODE_GRAPHS, "--graphs sets it"),
        (
            knob_id::TELEMETRY_TIER,
            "the path is read from every step's sample",
        ),
    ];
    if let Some(&(k, why)) = own.iter().find(|k| sets.iter().any(|s| s.0 == k.0)) {
        let name = knob_spec(k).map_or("?", |s| s.name.as_str());
        return Err(Stop::NotRun(anyhow!("--set {name}: {why}")));
    }
    // The engine's default is 0: replay is opt-in.
    let on = o.graphs == Graphs::On;
    sets.push((knob_id::DECODE_GRAPHS, f64::from(u8::from(on))));
    let hash = |p: &PathBuf| {
        oracle::sha256(p)
            .with_context(|| format!("hashing {}", p.display()))
            .map_err(Stop::NotRun)
    };
    let sha = hash(&o.model)?;
    let ref_sha = o.ref_model.as_ref().map(hash).transpose()?;
    // The dump's digest names the reference run; only of a regular file, as the hasher would drain a
    // pipe before the read below. Canonical, so `/dev/stdin < x` is x, not the hasher's own (null)
    // stdin; a path that does not resolve (a pipe, or a missing file, which fails at the read) has none.
    let dump_sha = match std::fs::canonicalize(&o.reference) {
        Ok(p) if p.is_file() => Some(
            oracle::sha256(&p)
                .with_context(|| format!("hashing {}", o.reference.display()))
                .map_err(Stop::NotRun)?,
        ),
        _ => None,
    };
    let mut d = File::open(&o.reference)
        .map_err(anyhow::Error::from)
        .and_then(|f| {
            Ok(Dump::read(
                BufReader::new(f),
                ref_sha.as_deref().unwrap_or(&sha),
            )?)
        })
        .with_context(|| format!("reading {}", o.reference.display()))
        .map_err(Stop::NotRun)?;
    if let Some(n) = o.prompts {
        d.prompts.truncate(n.get());
    }
    if let Some(i) = d.prompts.iter().position(|p| p.tokens.is_empty()) {
        return Err(Stop::NotRun(anyhow!("prompt {i} has no tokens")));
    }
    if let Some(i) = d.prompts.iter().position(|p| !p.images.is_empty()) {
        let e = anyhow!("prompt {i} has images: the image gate is larch-model's `vision` test");
        return Err(Stop::NotRun(e));
    }
    let Some(longest) = d.prompts.iter().map(|p| p.tokens.len()).max() else {
        let e = anyhow!("{}: no prompts", o.reference.display());
        return Err(Stop::NotRun(e));
    };
    // A context past u32 fails the context check below.
    let longest = u32::try_from(longest).unwrap_or(u32::MAX);
    let modes = match o.mode {
        Mode::All => &[Mode::Prefill, Mode::Decode, Mode::Chunked, Mode::Cached][..],
        _ => std::slice::from_ref(&o.mode),
    };
    let chunk = knob_spec(knob_id::PREFILL_CHUNK_TOKENS).expect("registered");
    let one_batch = (f64::from(longest) / chunk.step).ceil() * chunk.step;
    let one_batch = one_batch.max(chunk.min);
    if modes.contains(&Mode::Prefill) && one_batch > chunk.max {
        let e = anyhow!(
            "the longest prompt ({longest} tokens) is over prefill_chunk_tokens' max {}: no \
             one-batch prefill",
            chunk.max
        );
        return Err(Stop::NotRun(e));
    }
    let device = if o.cpu {
        larch_kernels::CPU
    } else {
        o.device.unwrap_or(0)
    };
    // One engine at a time: two would hold two models.
    for &mode in modes {
        let mut opts = EngineOptions::new(o.model.clone(), device);
        // The CLI's policy (9.11, D2): a fault flushes the ring and exits 3, not a failed run's 1.
        opts.on_fault = OnFault::Exit;
        opts.ring_capacity = RING;
        // ponytail: max_context = the longest prompt, so each sequence fits and the ring is not read
        // for preemptions; on a card whose pool holds fewer than --parallel contexts, fewer run in
        // flight than asked. Check RequestSpan.preemptions from the ring when such a card is run.
        opts.max_context = longest;
        // Decode's prompt is one token: a larger budget only grows the forced run's logits scratch.
        let chunk_tokens = match mode {
            Mode::Chunked | Mode::Decode => chunk.min,
            _ => one_batch.min(chunk.max),
        };
        opts.knob_overrides =
            [&sets[..], &[(knob_id::PREFILL_CHUNK_TOKENS, chunk_tokens)]].concat();
        if mode == Mode::Decode {
            // The engine's continuous batching keeps --parallel runs in flight.
            let seqs = (knob_id::MAX_BATCH_SEQS, f64::from(o.parallel));
            opts.knob_overrides.push(seqs);
        }
        let e = start(opts)
            .with_context(|| format!("starting {} on device {device}", o.model.display()))
            .map_err(Stop::NotRun)?;
        let ring = e.ring();
        let ran = check(&e, longest).map_err(Stop::NotRun).and_then(|()| {
            let rows = feed(&e, mode, &d).map_err(Stop::Failed)?;
            let graphs = e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize] != 0.0;
            Ok((rows, e.capabilities(), graphs, e.eager_decode_steps()))
        });
        e.shutdown();
        let (rows, caps, graphs, eager) = ran?;
        // One run per dump prompt, then chunked mode's decode runs.
        let companions = rows.len() - d.prompts.len();
        let path = samples(&ring).and_then(|s| {
            let n = replayed(mode, graphs, &s).map_err(|e| {
                anyhow!("{e}: {eager} decode steps ran eager on a refused capture (LARCH_FIT_DIAGNOSTICS=1 prints why)")
            })?;
            Ok((n, mixed(mode, companions, &s)?))
        });
        let index: Vec<usize> = rows.iter().map(|r| r.0).collect();
        // Each run's reference rows: the positions it computed.
        let (prompts, run): (Vec<_>, Vec<_>) = rows
            .into_iter()
            .map(|(i, at, top)| {
                let p = &d.prompts[i];
                let tokens = p.tokens[at.clone()].to_vec();
                let reference = Prompt {
                    text: String::new(),
                    tokens,
                    images: Vec::new(),
                    top: p.top[at].to_vec(),
                };
                (reference, top)
            })
            .unzip();
        let reference = Dump {
            header: d.header.clone(),
            prompts,
        };
        let r = oracle::compare(&reference, &run, o.margin).map_err(|e| Stop::Failed(e.into()))?;
        let ok = r.passes(o.bound);
        // The engine computed these rows, so a miss on them stands whatever else is unverified: mean
        // KL over the bound, or decisive positions under the top-1 gate. Rows that pass count only on
        // the path asked for, with a decisive position to gate top-1 on; else not run.
        let missed = !ok && (r.all.decisive > 0 || r.all.mean_kl > o.bound);
        let path = match path {
            Err(e) if !missed => return Err(Stop::NotRun(e)),
            path => path.ok(),
        };
        if !missed && r.all.decisive == 0 {
            let e = anyhow!(
                "{mode:?}: no decisive positions at margin {}: top-1 cannot be gated",
                o.margin
            );
            return Err(Stop::NotRun(e));
        }
        *pass &= ok;
        let per_prompt: Vec<_> = r
            .prompts
            .iter()
            .zip(index)
            .map(|(s, i)| {
                json!({
                    "prompt": i, "top1": over(s.positions, s.top1_rate()), "decisive": s.decisive,
                    "decisive_top1": over(s.decisive, s.decisive_top1_rate()), "mean_kl": s.mean_kl,
                    "max_kl": s.max_kl,
                })
            })
            .collect();
        let line = json!({
            "mode": format!("{mode:?}").to_lowercase(),
            "backend": format!("{:?}", caps.backend),
            "device": device,
            // Only decode mode runs several prompts in flight.
            "parallel": if mode == Mode::Decode { o.parallel } else { 1 },
            "decode_graphs": graphs,
            "graph_replay": caps.graph_replay,
            // Null where a miss left the path unverified or not taken.
            "replayed_steps": path.map(|p| p.0),
            "eager_decode_steps": eager,
            "mixed_steps": path.map(|p| p.1),
            "prompts": d.prompts.len(),
            "positions": r.all.positions,
            "top1": r.all.top1_rate(),
            "margin": o.margin,
            "decisive": r.all.decisive,
            "decisive_top1": r.all.decisive_top1_rate(),
            "mean_kl": r.all.mean_kl,
            "p99_kl": r.all.p99_kl,
            "max_kl": r.all.max_kl,
            "bound": o.bound,
            "pass": ok,
            "per_prompt": per_prompt,
            "model_sha256": sha,
            // The file the dump was made from, when not the one scored (--ref-model).
            "ref_model_sha256": ref_sha,
            "reference": {
                "backend": d.header.backend, "device": d.header.raw["device"],
                "n_batch": d.header.raw["n_batch"], "n_ubatch": d.header.raw["n_ubatch"],
                "dump_sha256": dump_sha,
            },
            "llama_cpp": d.header.llama_cpp_tag,
        });
        writeln!(out, "{line}").map_err(|e| Stop::Failed(e.into()))?;
    }
    Ok(())
}

/// The engine runs what was asked: no startup override rejected but `--graphs on`'s where the device
/// has no replay (the line reports it), and its context holds the longest prompt.
fn check(e: &EngineHandle, longest: u32) -> Result<()> {
    let no_replay = (knob_id::DECODE_GRAPHS, 1.0, RejectReason::Unsupported);
    let rejected = e.rejected_overrides().iter().find(|&r| *r != no_replay);
    if let Some((k, v, reason)) = rejected {
        let name = knob_spec(*k).map_or("?", |s| s.name.as_str());
        bail!("startup override {name}={v} rejected: {reason:?}");
    }
    let ctx = e.knobs().effective().values[knob_id::MAX_CONTEXT as usize];
    ensure!(
        ctx >= f64::from(longest),
        "max_context {ctx} is below the longest prompt ({longest} tokens)"
    );
    Ok(())
}

/// Ring slots per engine: a mode's step samples (one per step at T1) and spans.
// ponytail: 1 << 18 records (64 MB); a mode past ~250k steps laps it and is not run. Drain the ring
// during feed when a dump that long is run.
const RING: usize = 1 << 18;

/// Every step's sample (T1 writes one each), from the engine's first step; not run when the ring
/// lapped: the path is then unverifiable.
pub(crate) fn samples(ring: &RingHandle) -> Result<Vec<EngineSample>> {
    // Seq 0, not Oldest: a lapped ring then reads Lagged.
    let (mut r, mut out) = (ring.reader(ReadFrom::Seq(0)), Vec::new());
    loop {
        match r.next() {
            Read::Record(Record::EngineSample(_, s)) => out.push(s),
            Read::Record(_) => {}
            Read::Empty => break,
            Read::Lagged { dropped } => {
                bail!("the ring lost {dropped} records: the path is unverifiable")
            }
        }
    }
    Ok(out)
}

/// Steps that replayed a captured decode graph; not run when decode mode had graphs in force and none
/// replayed: `--graphs on` tests the replay path wherever there is one.
fn replayed(mode: Mode, graphs: bool, steps: &[EngineSample]) -> Result<usize> {
    let n = steps.iter().filter(|s| s.graph_replayed != 0).count();
    ensure!(
        n > 0 || !graphs || mode != Mode::Decode,
        "graphs on, but no step replayed"
    );
    Ok(n)
}

/// Steps with both prefill and decode rows. In chunked mode every step with prefill rows has a decode
/// row (D16) but each decode run's own one-token first step (`companions` of them); anything else is
/// not run: the path asked for was not taken.
fn mixed(mode: Mode, companions: usize, steps: &[EngineSample]) -> Result<usize> {
    let prefill = steps.iter().filter(|s| s.n_prefill_tokens > 0);
    let (mixed, bare): (Vec<&EngineSample>, Vec<_>) = prefill.partition(|s| s.n_decode_seqs > 0);
    let ones = bare.iter().filter(|s| s.n_prefill_tokens == 1).count();
    let n = bare.len() - ones.min(companions);
    if mode == Mode::Chunked && n > 0 {
        // A decode run that ended early, was preempted or was not yet admitted reads differently.
        for s in bare.iter().filter(|s| s.n_prefill_tokens > 1) {
            eprintln!(
                "parity: chunk step {} without a decode row: prefill_tokens={} running={} \
                 waiting={} preempted={}",
                s.step, s.n_prefill_tokens, s.n_running, s.n_waiting, s.n_preempted
            );
        }
    }
    ensure!(
        mode != Mode::Chunked || n == 0,
        "chunked: {n} chunk steps ran without a decode row"
    );
    Ok(mixed.len())
}

/// A compared run: the dump prompt, the positions it computed (its reference rows) and their top 64.
type Run = (usize, Range<usize>, Vec<Top>);

/// The steps a decode run keeps in hand past the next chunked run's chunks, for those it decodes
/// before that run is admitted.
// ponytail: a caller stalled for more engine steps than this between the check and the chunked run's
// submission lets the decode run end first, which the ring check reports as not run. 32 showed it on
// the M5's small model (a ~3 ms step): the last prompt's 3 chunks ran bare. Raise it again if it shows.
const MARGIN: usize = 128;

/// One run per dump prompt, in order, from its first computed position; then, in chunked mode, the
/// decode runs that shared its steps.
fn feed(e: &EngineHandle, mode: Mode, d: &Dump) -> Result<Vec<Run>> {
    // `prompt` tokens of `p` are the prompt, the rest are fed one per decode step.
    let force = |p: &Prompt, prompt: usize, session: u64| -> Result<ForcedHandle> {
        let f = Forced {
            tokens: p.tokens.clone(),
            prompt: prompt as u32,
            session,
            keep_kv: false,
        };
        Ok(e.force(f)?)
    };
    let mut rows = Vec::new();
    match mode {
        Mode::Prefill => {
            for (i, p) in d.prompts.iter().enumerate() {
                let n = p.tokens.len();
                rows.push((i, 0..n, finish(force(p, n, 0)?, 0)?));
            }
        }
        Mode::Decode => {
            // ponytail: every prompt is admitted at once (max_batch_seqs bounds the running set),
            // so a dump of more prompts than max_waiting (256 by default) is refused at admission.
            // Admit in waves of max_waiting when a dump of more than 256 prompts is run.
            let hs = d.prompts.iter().map(|p| force(p, 1, 0));
            for (i, h) in hs.collect::<Result<Vec<_>>>()?.into_iter().enumerate() {
                rows.push((i, 0..d.prompts[i].tokens.len(), finish(h, 0)?));
            }
        }
        Mode::Chunked => {
            // Decode runs of the dump's prompts, longest first, cycled; each stays in flight across
            // chunked runs while it has their chunks and MARGIN left.
            let budget = e.knobs().effective().values[knob_id::PREFILL_CHUNK_TOKENS as usize];
            let mut order: Vec<usize> = (0..d.prompts.len()).collect();
            order.sort_by_key(|&i| Reverse(d.prompts[i].tokens.len()));
            let longest = order[0];
            let mut order = order.into_iter().cycle();
            // The decode run in flight: its prompt, handle and positions so far.
            let mut decoding: Option<(usize, ForcedHandle, Vec<Top>)> = None;
            let mut decoded = Vec::new();
            for (i, p) in d.prompts.iter().enumerate() {
                let n = p.tokens.len();
                let left = match &mut decoding {
                    Some((j, h, top)) => {
                        take(h.request.id, 0, top, h.positions.try_iter())?;
                        d.prompts[*j].tokens.len().saturating_sub(top.len())
                    }
                    None => 0,
                };
                let need = n.div_ceil(budget as usize) + MARGIN;
                if left < need {
                    decoded.extend(decoding.take().map(retire).transpose()?);
                    // The cycle's next if it has `need` past its first position, else the longest.
                    let j = order.next().expect("a prompt");
                    let j = if d.prompts[j].tokens.len() > need {
                        j
                    } else {
                        longest
                    };
                    let h = force(&d.prompts[j], 1, 0)?;
                    // Decoding (past its first position) before the chunked run is admitted, so its
                    // row shares the chunk steps (D16, `Forced`).
                    let (id, mut top) = (h.request.id, Vec::new());
                    let first = h.positions.recv();
                    let first = first.with_context(|| {
                        format!("request {id}: ended before its first position")
                    })?;
                    take(id, 0, &mut top, [first])?;
                    decoding = Some((j, h, top));
                }
                rows.push((i, 0..n, finish(force(p, n, 0)?, 0)?));
            }
            decoded.extend(decoding.map(retire).transpose()?);
            rows.append(&mut decoded);
        }
        Mode::Cached => {
            for (i, p) in d.prompts.iter().enumerate() {
                let (n, h) = (p.tokens.len(), p.tokens.len() / 2);
                if h == 0 {
                    rows.push((i, n..n, Vec::new())); // nothing to park
                    continue;
                }
                // The first half as a session turn that samples one token and parks.
                let session = i as u64 + 1;
                let first = Request {
                    session,
                    keep_kv: true,
                    ..Request::new(p.tokens[..h].to_vec(), 1)
                };
                ended(&e.submit(first)?)?;
                rows.push((i, h..n, finish(force(p, n, session)?, h)?));
                e.release_session(session);
            }
        }
        Mode::All => unreachable!("gate runs all as the four"),
    }
    Ok(rows)
}

/// Waits for a forced run's end: it reused exactly `from` positions and reported `from..` in order.
fn finish(h: ForcedHandle, from: usize) -> Result<Vec<Top>> {
    let usage = ended(&h.request)?;
    ensure!(
        usage.cached_tokens as usize == from,
        "request {}: {} positions reused, expected {from}",
        h.request.id,
        usage.cached_tokens
    );
    // All of them arrived before Done (`ForcedHandle`).
    let mut top = Vec::new();
    take(h.request.id, from, &mut top, h.positions.try_iter())?;
    Ok(top)
}

/// Cancels decode run `(j, h, top)` and returns it as prompt `j`'s computed prefix.
fn retire((j, h, mut top): (usize, ForcedHandle, Vec<Top>)) -> Result<Run> {
    h.request.cancel();
    // Done if it ran out first; every position has arrived by its terminal event.
    if let Err(e) = ended(&h.request)
        && e.downcast_ref() != Some(&EngineError::Cancelled)
    {
        return Err(e);
    }
    take(h.request.id, 0, &mut top, h.positions.try_iter())?;
    Ok((j, 0..top.len(), top))
}

/// Appends request `id`'s `positions` to `top`, which holds its positions from `from` on, each the
/// next due.
fn take(
    id: u64,
    from: usize,
    top: &mut Vec<Top>,
    positions: impl IntoIterator<Item = Position>,
) -> Result<()> {
    for p in positions {
        let due = from + top.len();
        ensure!(
            p.pos as usize == due,
            "request {id}: position {} where {due} was due",
            p.pos
        );
        top.push(p.top);
    }
    Ok(())
}

/// A request's terminal event: Done's usage, or its error (a cached first turn samples one token,
/// which is skipped).
fn ended(r: &RequestHandle) -> Result<Usage> {
    for event in r.events.iter() {
        match event {
            EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
            EngineEvent::Done { usage, .. } => return Ok(usage),
            EngineEvent::Error(e) => return Err(anyhow!(e).context(format!("request {}", r.id))),
        }
    }
    bail!(
        "request {}: the engine ended it without Done or Error",
        r.id
    )
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};

    use clap::Parser;
    use larch_engine::{EngineOptions, Loader};
    use larch_kernels::{KError, KvPool, Stream, Tensor};
    use larch_model::{Arch, FakeModel, Model, ModelConfig, StepBatch, Transformer, oracle, tiny};
    use larch_records::knob_id;
    use serde_json::{Value, json};

    use super::*;

    fn parse(args: &[&str]) -> Result<Options, clap::Error> {
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            o: Options,
        }
        Cli::try_parse_from(["parity"].iter().chain(args)).map(|c| c.o)
    }

    /// Swaps the ids of a row's first two entries: a confident wrong top-1.
    fn swap(t: &mut Top) {
        (t[0].0, t[1].0) = (t[1].0, t[0].0);
    }

    /// Exit code, stdout's JSON lines, stderr.
    fn exec(args: &[&str]) -> (i32, Vec<Value>, String) {
        exec_with(args, larch_engine::start)
    }

    /// [`exec`] on engines from `start`.
    fn exec_with(
        args: &[&str],
        start: impl Fn(EngineOptions) -> Result<EngineHandle, LoadError>,
    ) -> (i32, Vec<Value>, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(&parse(args).unwrap(), start, &mut out, &mut err);
        let lines = String::from_utf8(out).unwrap();
        let lines = lines.lines().map(|l| serde_json::from_str(l).unwrap());
        (code, lines.collect(), String::from_utf8(err).unwrap())
    }

    /// Engines whose model is `wrap` around the GGUF's.
    fn wrapped<M: Model + 'static>(
        wrap: fn(Transformer) -> M,
    ) -> impl Fn(EngineOptions) -> Result<EngineHandle, LoadError> {
        move |opts| {
            let path = opts.model.clone();
            let load: Loader = Box::new(move |dev, s, rows| {
                Ok(Box::new(wrap(Transformer::load(&path, dev, s, rows)?)) as Box<dyn Model>)
            });
            larch_engine::start_with(opts, load)
        }
    }

    /// A per-test dir with the tiny model and a dump of it from the engine's own one-batch prefill
    /// on CPU (prompts of 40, 150 and 7 tokens), each prompt's rows passed through `edit`. Returns
    /// (dir, model, dump).
    fn fixture(name: &str, edit: fn(&mut [Top])) -> (PathBuf, String, String) {
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("larch-parity-{pid}-{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let (model, dump) = (dir.join("model.gguf"), dir.join("dump.jsonl"));
        std::fs::write(&model, tiny::gguf(Arch::Llama, 3)).unwrap();
        let mut opts = EngineOptions::new(model.clone(), larch_kernels::CPU);
        opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 192.0)];
        let e = larch_engine::start(opts).unwrap();
        let header = json!({
            "larch_oracle": 1, "top_k": 64, "n_vocab": tiny::VOCAB, "prompts": 3,
            "llama_cpp_tag": "test", "llama_cpp_commit": "test",
            "model_sha256": oracle::sha256(&model).unwrap(), "prompts_sha256": "0".repeat(64),
            "backend": "CPU", "device": "CPU", "n_batch": 2048, "n_ubatch": 512,
        });
        let mut lines = vec![header.to_string()];
        for (j, n) in [40u32, 150, 7].into_iter().enumerate() {
            let tokens: Vec<u32> = (0..n).map(|i| (i * 7 + j as u32 * 31 + 3) % 250).collect();
            let f = Forced {
                tokens: tokens.clone(),
                prompt: n,
                session: 0,
                keep_kv: false,
            };
            let mut top = finish(e.force(f).unwrap(), 0).unwrap();
            edit(&mut top);
            let line = json!({"index": j, "prompt": "", "tokens": tokens, "top64": top});
            lines.push(line.to_string());
        }
        e.shutdown();
        std::fs::write(&dump, lines.join("\n")).unwrap();
        let s = |p: PathBuf| p.into_os_string().into_string().unwrap();
        (dir, s(model), s(dump))
    }

    #[test]
    fn options_parse() {
        let o = parse(&["-m", "m", "--ref", "d", "--bound", "0.5", "--margin", "0"]).unwrap();
        assert_eq!(
            (o.mode, o.parallel, o.graphs, o.cpu, o.device, o.prompts),
            (Mode::All, 1, Graphs::On, false, None, None)
        );
        assert_eq!(o.margin, 0.0);
        assert!(o.set.is_empty());
        let o = parse(&[
            "--model",
            "m",
            "--ref",
            "d",
            "--bound",
            "1e-3",
            "--margin",
            "0.25",
            "--mode",
            "cached",
            "--parallel",
            "64",
            "--graphs",
            "off",
            "--device",
            "1",
            "--set",
            "max_batch_seqs=8",
            "--set",
            "decode_graphs=0",
            "--prompts",
            "2",
        ])
        .unwrap();
        assert_eq!(
            (
                &o.model,
                &o.reference,
                o.bound,
                o.mode,
                o.parallel,
                o.graphs
            ),
            (
                &PathBuf::from("m"),
                &PathBuf::from("d"),
                1e-3,
                Mode::Cached,
                64,
                Graphs::Off
            )
        );
        assert_eq!((o.device, o.prompts), (Some(1), NonZeroUsize::new(2)));
        assert_eq!(o.margin, 0.25);
        assert_eq!(o.set, ["max_batch_seqs=8", "decode_graphs=0"]);
        assert!(
            parse(&[
                "-m", "m", "--ref", "d", "--bound", "1", "--margin", "0", "--cpu"
            ])
            .unwrap()
            .cpu
        );
        for (bad, rule) in [
            (&["--margin=0", "--bound=0"][..], "> 0"),
            (&["--bound=1", "--margin=-1"], ">= 0"),
        ] {
            let e = parse(&[&["-m", "m", "--ref", "d"][..], bad].concat()).unwrap_err();
            assert!(
                e.to_string()
                    .contains(&format!("not a finite number {rule}")),
                "{e}"
            );
        }
        for bad in [
            &["--margin=0"][..],
            &["--bound=1"],
            &["--margin=0", "--bound=0"],
            &["--margin=0", "--bound=-1"],
            &["--margin=0", "--bound=nan"],
            &["--margin=0", "--bound=inf"],
            &["--bound=1", "--margin=-1"],
            &["--bound=1", "--margin=nan"],
            &["--bound=1", "--margin=inf"],
            &["--bound=1", "--margin=0", "--parallel=0"],
            &["--bound=1", "--margin=0", "--parallel=65"],
            &["--bound=1", "--margin=0", "--cpu", "--device=0"],
            &["--bound=1", "--margin=0", "--prompts=0"],
        ] {
            let e = parse(&[&["-m", "m", "--ref", "d"][..], bad].concat()).unwrap_err();
            assert_eq!(e.exit_code(), 2, "{bad:?}");
        }
    }

    #[test]
    fn not_run_is_exit_2() {
        let (dir, m, d) = fixture("not-run", |_| {});
        let other = dir.join("other.gguf");
        std::fs::write(&other, tiny::gguf(Arch::Llama, 4)).unwrap();
        let (o, x) = (other.to_str().unwrap(), dir.join("missing"));
        let x = x.to_str().unwrap();
        // A prompt with no tokens, which `Dump::read` takes.
        let empty = dir.join("empty.jsonl");
        let text = std::fs::read_to_string(&d).unwrap();
        let line = json!({"index": 3, "prompt": "", "tokens": [], "top64": []});
        let text = text.replacen("\"prompts\":3", "\"prompts\":4", 1) + &format!("\n{line}");
        std::fs::write(&empty, text).unwrap();
        let empty = empty.to_str().unwrap();
        let cases: [(&[&str], &str); 10] = [
            (&["-m", x, "--ref", &d, "--cpu"], x),
            (&["-m", &m, "--ref", x, "--cpu"], x),
            (&["-m", o, "--ref", &d, "--cpu"], &d),
            (
                &[
                    "-m",
                    &m,
                    "--ref",
                    &d,
                    "--cpu",
                    "--set",
                    "prefill_chunk_tokens=64",
                ],
                "prefill_chunk_tokens",
            ),
            (
                &[
                    "-m",
                    &m,
                    "--ref",
                    &d,
                    "--cpu",
                    "--set",
                    "attn_decode_splits=4",
                ],
                "attn_decode_splits",
            ),
            (
                &["-m", &m, "--ref", &d, "--cpu", "--set", "max_batch_seqs=8"],
                "max_batch_seqs: --parallel sets it",
            ),
            (
                &["-m", &m, "--ref", &d, "--cpu", "--set", "telemetry_tier=0"],
                "telemetry_tier: the path is read from every step's sample",
            ),
            (
                &[
                    "-m",
                    &m,
                    "--ref",
                    &d,
                    "--cpu",
                    "--graphs",
                    "off",
                    "--set",
                    "decode_graphs=1",
                ],
                "decode_graphs: --graphs sets it",
            ),
            (
                &["-m", &m, "--ref", empty, "--cpu"],
                "prompt 3 has no tokens",
            ),
            (&["-m", &m, "--ref", &d, "--device", "99"], "device 99"),
        ];
        for (args, names) in cases {
            let (code, lines, err) = exec(&[args, &["--bound", "1", "--margin", "0"]].concat());
            assert_eq!(code, 2, "{args:?}: {err}");
            assert!(err.starts_with("parity: not run:"), "{err}");
            assert!(err.contains(names), "{err}");
            assert!(lines.is_empty());
        }
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1", "--margin", "1000",
        ];
        let (code, lines, err) = exec(&args);
        assert_eq!((code, lines.len()), (2, 0), "{err}");
        assert!(err.starts_with("parity: not run:"), "{err}");
        assert!(
            err.contains("no decisive positions at margin 1000"),
            "{err}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A dump binds the file it was made from: another file (a repack, here other weights) is not
    /// run against it, unless `--ref-model` names that file, and then each line records both.
    #[test]
    fn ref_model_scores_another_file_against_a_dump() {
        let (dir, m, d) = fixture("ref-model", |_| {});
        let other = dir.join("other.gguf");
        std::fs::write(&other, tiny::gguf(Arch::Llama, 4)).unwrap();
        let other = other.to_str().unwrap();
        let base = [
            "-m", other, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0", "--mode",
            "prefill",
        ];
        let (code, lines, err) = exec(&base);
        assert_eq!((code, lines.len()), (2, 0), "{err}");
        let (code, lines, err) = exec(&[&base[..], &["--ref-model", &m]].concat());
        assert_ne!(code, 2, "{err}");
        let sha = |p: &str| oracle::sha256(Path::new(p)).unwrap();
        assert_eq!(lines[0]["model_sha256"], sha(other));
        assert_eq!(lines[0]["ref_model_sha256"], sha(&m));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_dump_of_the_engine_passes_every_mode() {
        let (dir, m, d) = fixture("pass", |_| {});
        let base = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0",
        ];
        let (code, lines, err) = exec(&[&base[..], &["--mode", "all"]].concat());
        assert_eq!(code, 0, "{err}");
        // Per run: its prompt and decisive positions, the dump's prompts in order.
        let run = |p: &Value| {
            (
                p["prompt"].as_u64().unwrap(),
                p["decisive"].as_u64().unwrap(),
            )
        };
        let mut got: Vec<_> = lines
            .iter()
            .map(|l| {
                let per_prompt = l["per_prompt"].as_array().unwrap().iter();
                (
                    l["mode"].as_str().unwrap(),
                    l["positions"].as_u64().unwrap(),
                    per_prompt.map(run).collect::<Vec<_>>(),
                    &l["pass"],
                )
            })
            .collect();
        // Then chunked mode's decode runs, the longest prompt's first.
        let decoded = got[2].2.split_off(3);
        assert!(!decoded.is_empty() && decoded[0].0 == 1, "{decoded:?}");
        got[2].1 -= decoded.iter().map(|r| r.1).sum::<u64>();
        let yes = &Value::Bool(true);
        let all = vec![(0, 40), (1, 150), (2, 7)];
        let cached = vec![(0, 40 - 20), (1, 150 - 75), (2, 7 - 3)];
        let want = [
            ("prefill", 197, all.clone(), yes),
            ("decode", 197, all.clone(), yes),
            ("chunked", 197, all, yes),
            ("cached", 20 + 75 + 4, cached, yes),
        ];
        assert_eq!(got, want);
        // larch's own dump has no exact ties at the top: every position is decisive at margin 0.
        let decisive = |s: &Value| {
            s["decisive_top1"] == 1.0
                && s["top1"] == 1.0
                && s["mean_kl"].is_f64()
                && s["max_kl"].is_f64()
        };
        let line = |l: &Value| {
            l["margin"] == 0.0
                && l["decisive"] == l["positions"]
                && decisive(l)
                && l["per_prompt"].as_array().unwrap().iter().all(decisive)
        };
        assert!(lines.iter().all(line), "{lines:?}");
        // --graphs on where the device has no replay: decode_graphs=1 is refused and the run is
        // direct decode, honestly reported.
        let direct = |l: &Value| {
            (
                &l["decode_graphs"],
                &l["graph_replay"],
                &l["replayed_steps"],
            ) == (&json!(false), &json!(false), &json!(0))
        };
        assert!(lines.iter().all(direct), "{lines:?}");
        // No graphs, so no decode step ran eager on a refused capture.
        assert!(
            lines.iter().all(|l| l["eager_decode_steps"] == 0),
            "{lines:?}"
        );
        // The reference run by its identity, and the dump file by its digest.
        let reference = json!({
            "backend": "CPU", "device": "CPU", "n_batch": 2048, "n_ubatch": 512,
            "dump_sha256": oracle::sha256(d.as_ref()).unwrap(),
        });
        assert!(
            lines.iter().all(|l| l["reference"] == reference),
            "{lines:?}"
        );
        let (code, lines, err) =
            exec(&[&base[..], &["--mode", "decode", "--parallel", "8"]].concat());
        assert_eq!((code, lines.len()), (0, 1), "{err}");
        assert_eq!((&lines[0]["parallel"], &lines[0]["pass"]), (&json!(8), yes));
        let (code, lines, err) =
            exec(&[&base[..], &["--mode", "decode", "--graphs", "off"]].concat());
        assert_eq!((code, lines.len()), (0, 1), "{err}");
        assert_eq!(lines[0]["decode_graphs"], false);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// `--graphs on` opts every engine in to replay (`decode_graphs=1`; the engine's default is 0),
    /// `off` sets 0.
    #[test]
    fn graphs_sets_decode_graphs() {
        let (dir, m, d) = fixture("graphs", |_| {});
        for (graphs, want) in [("on", 1.0), ("off", 0.0)] {
            let seen = std::sync::Mutex::new(Vec::new());
            let start = |o: EngineOptions| {
                seen.lock().unwrap().push(o.knob_overrides.clone());
                larch_engine::start(o)
            };
            let args = [
                "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0", "--graphs",
                graphs,
            ];
            let (code, _, err) = exec_with(&args, start);
            assert_eq!(code, 0, "{err}");
            let seen = seen.into_inner().unwrap();
            let set = |k: &Vec<(u16, f64)>| k.contains(&(knob_id::DECODE_GRAPHS, want));
            assert!(
                seen.len() == 4 && seen.iter().all(set),
                "{graphs}: {seen:?}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A step sample of a fake model's engine, to edit into synthetic step sequences.
    fn sample() -> EngineSample {
        let load: Loader = Box::new(|_, _, _| Ok(Box::new(FakeModel::new(256)) as Box<dyn Model>));
        let e =
            larch_engine::start_with(EngineOptions::new(PathBuf::new(), larch_kernels::CPU), load)
                .unwrap();
        ended(&e.submit(Request::new(vec![1], 1)).unwrap()).unwrap();
        let ring = e.ring();
        e.shutdown();
        samples(&ring).unwrap()[0]
    }

    /// Decode mode with graphs in force replays: no replayed step is not run, never a pass.
    #[test]
    fn graphs_on_must_replay() {
        let s = sample();
        let step = |r| EngineSample {
            graph_replayed: r,
            ..s
        };
        let (none, one) = ([step(0), step(0)], [step(0), step(1)]);
        assert_eq!(replayed(Mode::Decode, true, &one).unwrap(), 1);
        let e = replayed(Mode::Decode, true, &none).unwrap_err();
        assert_eq!(e.to_string(), "graphs on, but no step replayed");
        // Nothing to require: graphs not in force, or not the decode mode.
        assert_eq!(replayed(Mode::Decode, false, &none).unwrap(), 0);
        assert_eq!(replayed(Mode::Chunked, true, &none).unwrap(), 0);
    }

    /// Chunked mode: every step with prefill rows has a decode row but each decode run's one-token
    /// first step; anything else is not run.
    #[test]
    fn chunk_steps_need_a_decode_row() {
        let s = sample();
        let step = |prefill, decode| EngineSample {
            n_prefill_tokens: prefill,
            n_decode_seqs: decode,
            ..s
        };
        // A decode run's first step, a chunked run's two chunks beside it, decode alone.
        let ok = [step(1, 0), step(64, 1), step(3, 1), step(0, 1)];
        assert_eq!(mixed(Mode::Chunked, 1, &ok).unwrap(), 2);
        // A chunk without a decode row; a one-token chunk without one past the decode run's first.
        for bad in [[step(1, 0), step(64, 0)], [step(1, 0), step(1, 0)]] {
            let e = mixed(Mode::Chunked, 1, &bad).unwrap_err();
            assert_eq!(
                e.to_string(),
                "chunked: 1 chunk steps ran without a decode row"
            );
        }
        // Elsewhere the count only.
        assert_eq!(
            mixed(Mode::Decode, 0, &[step(1, 0), step(1, 3)]).unwrap(),
            1
        );
    }

    /// Every chunk step carries a decode row (D16), a one-token prompt's too: a long decode run
    /// stays in flight across the chunked runs. A dump of a one-token prompt alone has no decode
    /// run to share its step with: not run.
    #[test]
    fn chunk_steps_carry_a_decode_row() {
        let (dir, m, d) = fixture("chunked", |_| {});
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0", "--mode", "chunked",
        ];
        let (four, one) = (add_one_token(&d, 3, 1), add_one_token(&d, 0, 1));
        std::fs::write(&d, four).unwrap();
        let (code, lines, err) = exec(&args);
        assert_eq!(code, 0, "{err}");
        // 40, 150 (64 + 64 + 22), 7 and 1 tokens.
        assert_eq!(lines[0]["mixed_steps"], 6, "{lines:?}");
        std::fs::write(&d, one).unwrap();
        let (code, lines, err) = exec(&args);
        assert_eq!((code, lines.len()), (2, 0), "{err}");
        assert!(
            err.ends_with("not run: chunked: 1 chunk steps ran without a decode row\n"),
            "{err}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A dump whose long prompts run out: after them come 1-token prompts, too short to decode beside
    /// a chunked run, so the longest decodes again. The 150-, 40- and 7-token prompts have 194
    /// positions past their first, fewer than the 203 chunked runs' steps.
    #[test]
    fn short_prompts_leave_the_longest_decoding() {
        let (dir, m, d) = fixture("short-prompts", |_| {});
        std::fs::write(&d, add_one_token(&d, 3, 200)).unwrap();
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0", "--mode", "chunked",
        ];
        let (code, lines, err) = exec(&args);
        assert_eq!(code, 0, "{err}");
        // 40, 150 (64 + 64 + 22), 7 and 200 one-token chunks.
        assert_eq!(lines[0]["mixed_steps"], 205);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The dump at `d`'s first `keep` prompts and `ones` 1-token prompts: the first prompt's first
    /// token.
    fn add_one_token(d: &str, keep: usize, ones: usize) -> String {
        let text = std::fs::read_to_string(d).unwrap();
        let mut lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let (tokens, top) = (&lines[1]["tokens"][0], &lines[1]["top64"][0]);
        let one = |i| json!({"index": i, "prompt": "", "tokens": [tokens], "top64": [top]});
        let ones: Vec<_> = (keep..keep + ones).map(one).collect();
        lines.truncate(keep + 1);
        lines[0]["prompts"] = json!(keep + ones.len());
        lines.extend(ones);
        lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A rate over nothing is null, not 0 (a total miss): a prompt with no compared positions has no
    /// top1, one with no decisive position has no decisive_top1. The 7-token prompt is tied at the top
    /// throughout and a 1-token prompt is added (the cached mode has no second half of it to compare).
    #[test]
    fn a_rate_over_nothing_is_null() {
        let (dir, m, d) = fixture("null-rates", |rows| {
            if rows.len() == 7 {
                rows.iter_mut().for_each(|t| t[0].1 = t[1].1);
            }
        });
        std::fs::write(&d, add_one_token(&d, 3, 1)).unwrap();
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0",
        ];
        let per_prompt = |mode: &str| {
            let (code, lines, err) = exec(&[&args[..], &["--mode", mode]].concat());
            assert_eq!(code, 0, "{err}");
            assert!(lines[0]["top1"].is_f64() && lines[0]["decisive_top1"].is_f64());
            let rows = lines[0]["per_prompt"].as_array().unwrap().iter();
            rows.map(|p| (p["top1"].clone(), p["decisive_top1"].clone()))
                .collect::<Vec<_>>()
        };
        let (one, null) = (json!(1.0), Value::Null);
        let want = [(&one, &one), (&one, &one), (&one, &null), (&one, &one)];
        assert_eq!(
            per_prompt("prefill"),
            want.map(|(a, b)| (a.clone(), b.clone()))
        );
        let want = [(&one, &one), (&one, &one), (&one, &null), (&null, &null)];
        assert_eq!(
            per_prompt("cached"),
            want.map(|(a, b)| (a.clone(), b.clone()))
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_wrong_reference_misses_the_gate() {
        let (dir, m, d) = fixture("wrong", |rows| rows.iter_mut().for_each(swap));
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0",
        ];
        let (code, lines, err) = exec(&args);
        assert_eq!(code, 1, "{err}");
        assert!(err.starts_with("parity: a gate was missed"), "{err}");
        assert_eq!(lines.len(), 4);
        assert!(lines.iter().all(|l| l["pass"] == false), "{lines:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Loses its device at its first step.
    struct Lost(Transformer);

    impl Model for Lost {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }

        fn forward(
            &mut self,
            _: &Stream,
            _: &StepBatch,
            _: &KvPool,
            _: &Tensor,
        ) -> Result<(), KError> {
            Err(KError::DeviceLost)
        }
    }

    /// The CLI's fault policy (9.11, D2): a fault exits the process with 3 for `larch supervise`,
    /// not 1 as a failed run. The test runs itself again as a child whose model loses its device.
    #[test]
    fn a_fault_exits_3() {
        const CHILD: &str = "LARCH_PARITY_FAULT_CHILD";
        if let Some(dir) = std::env::var_os(CHILD) {
            let dir = PathBuf::from(dir);
            let s = |f: &str| dir.join(f).into_os_string().into_string().unwrap();
            let (m, d) = (s("model.gguf"), s("dump.jsonl"));
            let args = [
                "-m", &m, "--ref", &d, "--cpu", "--bound", "1", "--margin", "0",
            ];
            let (code, _, err) = exec_with(&args, wrapped(Lost));
            eprint!("{err}");
            std::process::exit(code);
        }
        let (dir, ..) = fixture("fault", |_| {});
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "parity::tests::a_fault_exits_3", "--nocapture"])
            .env(CHILD, &dir)
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&child.stderr);
        assert_eq!(child.status.code(), Some(3), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Negates the logits of decode rows in steps that also carry prefill rows, and only there.
    struct MixedWrong(Transformer);

    impl Model for MixedWrong {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.0.forward(s, b, kv, logits)?;
            if b.decode_rows > 0 && b.prefill_rows > 0 {
                // The decode rows' logits come first.
                let n = b.decode_rows as usize * self.0.config().vocab as usize;
                let mut l = vec![0f32; n];
                s.download(logits.buffer(), logits.offset(), &mut l)?;
                l.iter_mut().for_each(|x| *x = -*x);
                s.upload(logits.buffer(), logits.offset(), &l)?;
            }
            Ok(())
        }
    }

    /// Chunked mode compares its decode runs' rows too: a defect only decode rows that share a step
    /// with prefill rows carry misses chunked mode, while the modes without such rows pass.
    #[test]
    fn chunked_compares_its_decode_rows() {
        let (dir, m, d) = fixture("mixed-wrong", |_| {});
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0",
        ];
        let (code, lines, err) = exec_with(&args, wrapped(MixedWrong));
        let pass: Vec<_> = lines
            .iter()
            .map(|l| (l["mode"].as_str().unwrap(), &l["pass"]))
            .collect();
        let (yes, no) = (&json!(true), &json!(false));
        let want = [
            ("prefill", yes),
            ("decode", yes),
            ("chunked", no),
            ("cached", yes),
        ];
        assert_eq!((code, pass), (1, want.to_vec()), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Rows that miss are a miss even where the run could not count otherwise: wrong rows of a
    /// one-token prompt whose chunk step had no decode row beside it, and rows tied at the top (no
    /// decisive position) with mean KL over the bound, each give exit 1 and a failed line, not 2.
    /// The tiny model's tops are nearly flat, so the tie moves KL by little: its bound is tiny.
    #[test]
    fn a_miss_where_it_could_not_count_is_exit_1() {
        // Exit code, each line's pass and the verdict of `mode` on a fixture whose rows `edit` made.
        let run = |mode: &str, bound: &str, edit: fn(&mut [Top])| {
            let (dir, m, d) = fixture(&format!("miss-{mode}"), edit);
            if mode == "chunked" {
                std::fs::write(&d, add_one_token(&d, 0, 1)).unwrap();
            }
            let args = [
                "-m", &m, "--ref", &d, "--cpu", "--bound", bound, "--margin", "0", "--mode", mode,
            ];
            let (code, lines, err) = exec(&args);
            std::fs::remove_dir_all(dir).unwrap();
            let pass: Vec<_> = lines.iter().map(|l| l["pass"].clone()).collect();
            (code, pass, err)
        };
        let got = [
            run("chunked", "1e-4", |rows| rows.iter_mut().for_each(swap)),
            run("prefill", "1e-300", |rows| {
                rows.iter_mut().for_each(|t| t[0].1 = t[1].1)
            }),
        ];
        let miss = (
            1,
            vec![json!(false)],
            "parity: a gate was missed\n".to_owned(),
        );
        assert_eq!(got, [miss.clone(), miss]);
    }

    /// A miss is not hidden by a later mode that cannot run: the first halves are wrong (prefill,
    /// decode and chunked miss) and the second halves tied at the top, so cached, which compares
    /// only those, has no decisive position.
    #[test]
    fn a_miss_then_not_run_is_exit_1() {
        let (dir, m, d) = fixture("miss-then-not-run", |rows| {
            let h = rows.len() / 2;
            rows[..h].iter_mut().for_each(swap);
            rows[h..].iter_mut().for_each(|t| t[0].1 = t[1].1);
        });
        let args = [
            "-m", &m, "--ref", &d, "--cpu", "--bound", "1e-4", "--margin", "0",
        ];
        let (code, lines, err) = exec(&args);
        assert_eq!((code, lines.len()), (1, 3), "{err}");
        assert!(lines.iter().all(|l| l["pass"] == false), "{lines:?}");
        assert!(
            err.starts_with("parity: a gate was missed; then not run: Cached: no decisive"),
            "{err}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
