//! `larch routes-check -m <gguf> [--ref <dump> --bound B --margin M] [--cpu | --device N]`:
//! checks each architecture-gated route on this card at the model's shapes, then proves the
//! candidates with `larch parity` and writes them to the card profile ([`larch_model::card`])
//! for this model, keeping other models' routes. A projection route is a candidate when it matches
//! the CPU reference under kernels/tests/ops.cpp's gemm tolerance (the quant_gemm ctest's) on every
//! case and beats the route it replaces by more than 5 % over them; decode graphs, measured with
//! those candidates on, when they replay in both the one- and the eight-sequence runs, sample the
//! tokens eager decode samples (tests/graphs.rs's check), beat it by more than 5 % and keep the
//! context a graphs-off load gets. The candidates are written only when `larch parity --mode all`
//! and `--mode decode --parallel 32` both pass with them on: the op check compares a lossy route
//! with the CPU form of the same op, so only the model's own reference shows its accuracy. Without
//! `--ref` nothing is written. One line per route, then the verdict; off CUDA every route is
//! unavailable and nothing is written.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use clap::Args;
use larch_engine::{EngineEvent, EngineHandle, EngineOptions, Request};
use larch_records::RingHandle;

use crate::parity::{Graphs, Mode};
use larch_kernels::{Backend, Buffer, CPU, DType, Device, DeviceProps, Stream, Tensor};
use larch_model::card::{self, Card, Profile, Route};
use larch_model::{RouteCase, Transformer};
use larch_records::knob_id;

/// A route turns on only past this speedup over the route it replaces.
const WIN: f64 = 1.05;
/// kernels/tests/ops.cpp's gemm row for f32 outputs (rtol, atol), which quant_gemm.cpp reuses.
const TOLERANCE: (f64, f64) = (1e-4, 1e-4);
/// Decode-step rows (the INT8 rules switch at 14 and 28) and prefill chunk rows.
const ROWS: [i64; 13] = [8, 12, 14, 16, 20, 24, 28, 32, 48, 64, 128, 256, 512];
/// Timed passes of a case, after one warm-up.
const PASSES: u32 = 5;

/// `larch routes-check` options.
#[derive(Args, Debug)]
pub struct Options {
    /// GGUF whose shapes the routes are checked at.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// The CPU reference, which has none of the routes.
    #[arg(long, conflicts_with = "device")]
    pub cpu: bool,
    /// GPU ordinal (default zero).
    #[arg(long)]
    pub device: Option<i32>,
    /// The model's llama-logits dump, which `larch parity` proves the candidates against.
    #[arg(long = "ref", requires_all = ["bound", "margin"])]
    pub reference: Option<PathBuf>,
    /// The model's calibrated mean-KL bound (`larch parity --bound`).
    #[arg(long, requires = "reference")]
    pub bound: Option<f64>,
    /// The model's calibrated tie margin (`larch parity --margin`).
    #[arg(long, requires = "reference")]
    pub margin: Option<f64>,
}

/// Checks every gated route on the device and writes the ones parity proved to `profile`.
pub fn run(o: &Options, profile: &Path, out: &mut impl Write) -> anyhow::Result<()> {
    // Every load here takes exactly the routes being measured, never an earlier profile's.
    card::set_checking(Some(Vec::new()));
    let ran = check(o, profile, out);
    card::set_checking(None);
    ran
}

fn check(o: &Options, profile: &Path, out: &mut impl Write) -> anyhow::Result<()> {
    let device = if o.cpu { CPU } else { o.device.unwrap_or(0) };
    let dev = Device::open(device).with_context(|| format!("opening device {device}"))?;
    let props = dev.props()?;
    if props.backend != Backend::Cuda {
        for route in Route::ALL {
            writeln!(out, "{}: unavailable on {:?}", route.name(), props.backend)?;
        }
        return Ok(());
    }
    let s = Stream::new(&dev)?;
    let mut model = Transformer::load(&o.model, &dev, &s, 512)
        .map_err(|e| anyhow::anyhow!("loading {}: {e:?}", o.model.display()))?;
    let cases = model.route_cases(&ROWS);
    let mut on = Vec::new();
    for route in Route::ALL.into_iter().filter(|&r| r != Route::DecodeGraphs) {
        let cases: Vec<_> = cases.iter().filter(|c| c.route == route).collect();
        let (passed, why) = match cases.is_empty() {
            true if props.arch_code == 120 => (false, "on: the measured sm_120 default".into()),
            true => (false, "off: no shape of this model changes route".into()),
            false => projection(&mut model, &dev, &s, &cases)?,
        };
        let held = match route {
            Route::I8StreamK if props.arch_code != 120 => card::EXCLUDED
                .iter()
                .map(|e| format!("; holds {}-wide at {:?} rows: {}", e.1, e.2, e.3))
                .collect(),
            _ => String::new(),
        };
        writeln!(out, "{}: {why}{held}", route.name())?;
        if passed {
            on.push(route);
        }
    }
    drop(model);
    // Graphs are measured on the projection routes they would replay.
    card::set_checking(Some(on.clone()));
    let (passed, why) = graphs(o, device, &props);
    writeln!(out, "{}: {why}", Route::DecodeGraphs.name())?;
    if passed {
        on.push(Route::DecodeGraphs);
    }
    let names: Vec<_> = on.iter().map(|r| r.name()).collect();
    let names = match names.is_empty() {
        true => "none".to_owned(),
        false => names.join(", "),
    };
    let Some(reference) = &o.reference else {
        writeln!(
            out,
            "candidates: {names}; nothing written: pass --ref, --bound and --margin to prove them \
             with larch parity"
        )?;
        return Ok(());
    };
    card::set_checking(Some(on.clone()));
    if let Some(missed) = parity(o, device, reference)? {
        writeln!(out, "candidates: {names}; nothing written: {missed}")?;
        return Ok(());
    }
    let model =
        card::model_key(&o.model).with_context(|| format!("reading {}", o.model.display()))?;
    let earlier = card::read(profile).unwrap_or(None);
    let written = Profile::with(earlier, Card::of(&props), model, on);
    card::write(profile, &written).with_context(|| format!("writing {}", profile.display()))?;
    writeln!(
        out,
        "proved by larch parity: {names}; written to {}",
        profile.display()
    )?;
    Ok(())
}

/// `larch parity --mode all` and `--mode decode --parallel 32` with the candidates on: None when
/// both pass, else what missed.
// ponytail: a miss writes none of the candidates; run parity per route if a card keeps missing
// on one.
fn parity(o: &Options, device: i32, reference: &Path) -> anyhow::Result<Option<String>> {
    for (mode, parallel) in [(Mode::All, 1), (Mode::Decode, 32)] {
        let p = crate::parity::Options {
            model: o.model.clone(),
            reference: reference.to_path_buf(),
            bound: o.bound.context("--bound")?,
            margin: o.margin.context("--margin")?,
            mode,
            parallel,
            graphs: Graphs::On,
            cpu: false,
            device: Some(device),
            set: Vec::new(),
            prompts: None,
            ref_model: None,
        };
        let mut verdict = Vec::new();
        let code = crate::parity::run(&p, larch_engine::start, &mut std::io::sink(), &mut verdict);
        if code != 0 {
            let verdict = String::from_utf8_lossy(&verdict);
            return Ok(Some(format!(
                "larch parity --mode {mode:?} --parallel {parallel} exited {code}: {}",
                verdict.trim()
            )));
        }
    }
    Ok(None)
}

/// One projection route over its cases: whether it passed, and its line.
fn projection(
    model: &mut Transformer,
    dev: &Device,
    s: &Stream,
    cases: &[&RouteCase],
) -> anyhow::Result<(bool, String)> {
    let (mut on_secs, mut off_secs) = (0.0, 0.0);
    for c in cases {
        let ([n, k], rows) = (c.shape(), c.rows);
        // f16 activations in [0.125, 1) of either sign, seeded by the case.
        let mut seed = (rows * n + k) as u64;
        let x: Vec<u16> = (0..rows * k)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (seed >> 33) as u16;
                (r & 0x8000) | (0x3000 + r % 0x0c00)
            })
            .collect();
        let xb = Buffer::alloc(dev, x.len() as u64 * 2)?;
        s.upload(&xb, 0, &x)?;
        let ob = Buffer::alloc(dev, (rows * n) as u64 * 4)?;
        let xt = Tensor::new(&xb, 0, DType::F16, &[rows, k])?;
        let ot = Tensor::new(&ob, 0, DType::F32, &[rows, n])?;
        off_secs += time(model, s, c, false, &xt, &ot)?;
        on_secs += time(model, s, c, true, &xt, &ot)?;
        let mut dut = vec![0f32; (rows * n) as usize];
        s.download(&ob, 0, &mut dut)?;
        let edge = n.min(128);
        for cols in [0..edge, n - edge..n] {
            let reference = model.route_reference(s, c, &x, cols.clone())?;
            let dut: Vec<f32> = (0..rows)
                .flat_map(|r| &dut[(r * n + cols.start) as usize..(r * n + cols.end) as usize])
                .copied()
                .collect();
            if !within(&reference, &dut) {
                let what = format!("{:?} {:?} at {rows} rows", c.dtype(), c.shape());
                return Ok((false, format!("off: {what} misses the CPU reference")));
            }
        }
    }
    let speedup = off_secs / on_secs;
    let verdict = if speedup > WIN {
        "on"
    } else {
        "off: not 5 % faster"
    };
    let line = format!("{verdict} ({speedup:.3}x over {} cases)", cases.len());
    Ok((speedup > WIN, line))
}

/// Seconds for [`PASSES`] passes of `case`, after a warm-up; `out` keeps the last pass's.
fn time(
    model: &mut Transformer,
    s: &Stream,
    case: &RouteCase,
    on: bool,
    x: &Tensor,
    out: &Tensor,
) -> anyhow::Result<f64> {
    model.route_case(s, case, on, x, out)?;
    s.sync()?;
    let t = Instant::now();
    for _ in 0..PASSES {
        model.route_case(s, case, on, x, out)?;
    }
    s.sync()?;
    Ok(t.elapsed().as_secs_f64())
}

/// `dut` against `reference`: |dut - ref| <= rtol |ref| + atol max|ref|, per element.
fn within(reference: &[f32], dut: &[f32]) -> bool {
    let (rtol, atol) = TOLERANCE;
    let peak = reference
        .iter()
        .filter(|r| r.is_finite())
        .fold(0f64, |m, &r| m.max(f64::from(r).abs()));
    reference.iter().zip(dut).all(|(&r, &d)| {
        let (r, d) = (f64::from(r), f64::from(d));
        (d - r).abs() <= rtol * r.abs() + atol * peak
    })
}

/// One decode run's measure ([`decode`] twice on one engine, the second timed).
struct Run {
    tokens: Vec<Vec<u32>>,
    secs: f64,
    /// The effective decode_graphs and max_context.
    kept: [f64; 2],
    /// Steps of the timed pass that replayed a graph: its one-sequence run, its eight.
    replays: [usize; 2],
}

/// Decode graphs against eager decode, each on a fresh engine at the default context; a failed
/// run is the route's line, not the command's error, so the projection routes still count.
fn graphs(o: &Options, device: i32, props: &DeviceProps) -> (bool, String) {
    if props.flags & DeviceProps::CAP_GRAPHS == 0 {
        return (
            false,
            "unavailable: the device does not capture graphs".into(),
        );
    }
    let run = |graphs: bool| -> anyhow::Result<Run> {
        let mut opts = EngineOptions::new(o.model.clone(), device);
        opts.knob_overrides = vec![
            (knob_id::DECODE_GRAPHS, f64::from(u8::from(graphs))),
            (knob_id::MAX_BATCH_SEQS, 8.0),
        ];
        let e = larch_engine::start(opts).map_err(|e| anyhow::anyhow!("loading: {e:?}"))?;
        let v = e.knobs().effective().values;
        let kept = [
            v[knob_id::DECODE_GRAPHS as usize],
            v[knob_id::MAX_CONTEXT as usize],
        ];
        // The first pass captures the graphs; the second is timed.
        let ring = e.ring();
        let ran = decode(&e, &ring).and_then(|(_, [_, before])| {
            let t = Instant::now();
            let (tokens, [one, eight]) = decode(&e, &ring)?;
            let secs = t.elapsed().as_secs_f64();
            Ok((tokens, secs, [one - before, eight - one]))
        });
        e.shutdown();
        let (tokens, secs, replays) = ran?;
        Ok(Run {
            tokens,
            secs,
            kept,
            replays,
        })
    };
    match (run(false), run(true)) {
        (Err(e), _) => (false, format!("off: eager decode failed: {e:#}")),
        (_, Err(e)) => (false, format!("off: graphs decode failed: {e:#}")),
        (Ok(eager), Ok(on)) => verdict(&eager, &on),
    }
}

/// Whether the graphs run `on` proves decode graphs against `eager`, and the route's line.
fn verdict(eager: &Run, on: &Run) -> (bool, String) {
    let context = eager.kept[1];
    if on.kept != [1.0, context] {
        return (
            false,
            format!(
                "off: graphs do not load at the full {context} context (decode_graphs {}, \
                 context {})",
                on.kept[0], on.kept[1]
            ),
        );
    }
    // A refused capture runs eager with the knob on: only replayed steps measure graphs.
    if on.replays.contains(&0) {
        return (
            false,
            format!(
                "off: graphs did not replay ({} and {} steps of the 1 and 8 sequences)",
                on.replays[0], on.replays[1]
            ),
        );
    }
    if on.tokens != eager.tokens {
        return (
            false,
            "off: graphs sample other tokens than eager decode".into(),
        );
    }
    let speedup = eager.secs / on.secs;
    let verdict = if speedup > WIN {
        "on"
    } else {
        "off: not 5 % faster"
    };
    (
        speedup > WIN,
        format!(
            "{verdict} ({speedup:.3}x, 1 and 8 sequences, {} and {} steps replayed)",
            on.replays[0], on.replays[1]
        ),
    )
}

/// One greedy request alone, then eight together, 64 tokens each: their tokens, and the steps
/// the engine had replayed a graph on after each group.
fn decode(e: &EngineHandle, ring: &RingHandle) -> anyhow::Result<(Vec<Vec<u32>>, [usize; 2])> {
    let replayed = || -> anyhow::Result<usize> {
        let steps = crate::parity::samples(ring)?;
        Ok(steps.iter().filter(|s| s.graph_replayed != 0).count())
    };
    let mut tokens = Vec::new();
    let mut after = [0; 2];
    for (i, group) in [1u32, 8].into_iter().enumerate() {
        let reqs = (0..group)
            .map(|i| Request::new(vec![i + 1; 8], 64))
            .collect();
        for h in e.submit_group(reqs).map_err(|e| anyhow::anyhow!("{e:?}"))? {
            let mut t = Vec::new();
            for ev in h.events.iter() {
                match ev {
                    EngineEvent::Token { id, .. } => t.push(id),
                    EngineEvent::Done { .. } => break,
                    ev => bail!("decode ended with {ev:?}"),
                }
            }
            tokens.push(t);
        }
        after[i] = replayed()?;
    }
    Ok((tokens, after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        o: Options,
    }

    #[test]
    fn the_cpu_backend_reports_every_route_unavailable_and_writes_nothing() {
        let dir = std::env::temp_dir().join(format!("larch-routes-check-{}", std::process::id()));
        let profile = dir.join(larch_model::card::FILE);
        let o = Cli::parse_from(["x", "-m", "absent.gguf", "--cpu"]).o;
        let mut out = Vec::new();
        run(&o, &profile, &mut out).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert_eq!(out.lines().count(), Route::ALL.len(), "{out}");
        for (line, route) in out.lines().zip(Route::ALL) {
            assert!(line.starts_with(route.name()), "{line}");
            assert!(line.contains("unavailable"), "{line}");
        }
        assert!(!dir.exists());
    }

    /// Graphs that loaded, sampled eager's tokens and ran faster, but replayed no step of one
    /// group (a refused capture runs eager with the knob on), prove nothing.
    #[test]
    fn graphs_qualify_only_where_they_replayed() {
        let run = |kept, secs, replays| Run {
            tokens: vec![vec![1, 2]],
            secs,
            kept,
            replays,
        };
        let eager = run([0.0, 4096.0], 1.0, [0, 0]);
        assert!(verdict(&eager, &run([1.0, 4096.0], 0.5, [60, 60])).0);
        for refused in [[0, 60], [60, 0], [0, 0]] {
            let (passed, line) = verdict(&eager, &run([1.0, 4096.0], 0.5, refused));
            assert!(!passed && line.contains("did not replay"), "{line}");
        }
        let (passed, line) = verdict(&eager, &run([1.0, 2048.0], 0.5, [60, 60]));
        assert!(!passed && line.contains("full 4096 context"), "{line}");
        assert!(!verdict(&eager, &run([1.0, 4096.0], 0.99, [60, 60])).0);
    }

    /// Without a dump to prove them against, nothing is written; the dump needs its bound and
    /// margin.
    #[test]
    fn parity_needs_its_dump_bound_and_margin() {
        assert!(Cli::try_parse_from(["x", "-m", "m.gguf", "--ref", "d.jsonl"]).is_err());
        assert!(Cli::try_parse_from(["x", "-m", "m.gguf", "--bound", "0.1"]).is_err());
        let o = Cli::parse_from([
            "x", "-m", "m.gguf", "--ref", "d.jsonl", "--bound", "0.1", "--margin", "0",
        ])
        .o;
        assert_eq!((o.bound, o.margin), (Some(0.1), Some(0.0)));
    }

    /// The gemm rows of kernels/tests/ops.cpp: |dut - ref| <= rtol |ref| + atol max|ref|.
    #[test]
    fn within_holds_the_gemm_tolerance_against_the_largest_reference() {
        let reference = [100.0, 1.0, -2.0];
        assert!(within(&reference, &[100.0, 1.01, -2.0]));
        assert!(!within(&reference, &[100.0, 1.0103, -2.0]));
        assert!(!within(&reference, &[100.0, 1.0, f32::NAN]));
    }
}
