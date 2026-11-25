//! Static registry listing and human knob commands over the existing control API.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use larch_records::{
    AbCommand, Allow, Authority, KnobControl, KnobDelta, KnobRequest, KnobSpec, KnobType, REGISTRY,
    RevertTarget,
};

/// Finds a knob by its registered name; unknown names are rejected.
pub fn spec(name: &str) -> Result<&'static KnobSpec> {
    REGISTRY
        .iter()
        .find(|k| k.name.as_str() == name)
        .with_context(|| format!("unknown knob: {name}"))
}

/// Parses a registered value, accepting enum names and true/false as well as numbers.
pub fn value(k: &KnobSpec, raw: &str) -> Result<f64> {
    let v = match k.ty() {
        KnobType::Bool if raw == "true" => 1.0,
        KnobType::Bool if raw == "false" => 0.0,
        KnobType::Enum => match k.enum_names().position(|name| name == raw) {
            Some(index) => index as f64,
            None => raw
                .parse()
                .with_context(|| format!("invalid value for {}", k.name.as_str()))?,
        },
        _ => raw
            .parse()
            .with_context(|| format!("invalid value for {}", k.name.as_str()))?,
    };
    Ok(v)
}

/// Parses one startup or A/B assignment using the shared registry.
pub fn assignment(raw: &str) -> Result<(u16, f64)> {
    let (name, raw) = raw.split_once('=').context("expected knob=value")?;
    let k = spec(name)?;
    Ok((k.id, value(k, raw)?))
}

/// Prints the contract registry without opening a model or creating an engine.
pub fn list(out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "name\tdefault\trange\tactivation\trisk\tbackends\tenum"
    )?;
    for k in REGISTRY {
        writeln!(
            out,
            "{}\t{}\t{}..={}\t{:?}\t{:?}\t{:#x}\t{}",
            k.name.as_str(),
            k.default,
            k.min,
            k.max,
            k.activation(),
            k.risk(),
            k.backends,
            k.enum_names().collect::<Vec<_>>().join(",")
        )?;
    }
    Ok(())
}

/// Executes human list/get/set/revert commands and prints the engine's actual reply.
pub fn command(control: &dyn KnobControl, words: &[&str], out: &mut impl Write) -> Result<()> {
    match words {
        ["list"] => list(out),
        ["get", name] => {
            let k = spec(name)?;
            let cfg = control.effective();
            writeln!(
                out,
                "{}={} (rev {})",
                name, cfg.values[k.id as usize], cfg.rev
            )?;
            Ok(())
        }
        ["set", name, raw] => {
            let k = spec(name)?;
            let reply = control.set(KnobRequest::new(k.id, value(k, raw)?, Authority::Human));
            writeln!(out, "{reply:?}")?;
            Ok(())
        }
        ["revert", rev] => {
            let reply = control.revert(
                RevertTarget::Rev(rev.parse().context("invalid revision")?),
                Authority::Human,
            );
            writeln!(out, "{reply:?}")?;
            Ok(())
        }
        _ => bail!("usage: /knob list|get <name>|set <name> <value>|revert <revision>"),
    }
}

/// Human A/B trial ids, unique in this process: the top half, since gait numbers its trials in the
/// low half (0 means no trial).
static NEXT_TRIAL: AtomicU64 = AtomicU64::new(1 << 63);

/// Starts a human A/B under a new trial id, or stops one, and prints the id with the engine's reply.
pub fn ab(control: &dyn KnobControl, words: &[&str], out: &mut impl Write) -> Result<()> {
    let cmd = match words {
        ["stop", trial] => AbCommand::Stop {
            trial_id: trial.parse().context("invalid trial id")?,
        },
        [raw] | [raw, _] => {
            let period = match words.get(1) {
                Some(period) => period.parse::<u32>().context("invalid A/B period")?,
                None => 200,
            };
            ensure!(period > 0, "A/B period must be positive");
            let (knob, value) = assignment(raw)?;
            let mut b = [KnobDelta::default(); 4];
            b[0] = KnobDelta { knob, value };
            // max_steps 0: it runs until `/ab stop`, a guardrail trip or a fault (9.9).
            AbCommand::Start {
                b,
                period_steps: period,
                max_steps: 0,
                trial_id: NEXT_TRIAL.fetch_add(1, Ordering::Relaxed),
                authority: Authority::Human,
                allow: Allow::NONE,
            }
        }
        _ => bail!("usage: /ab <knob>=<value> [period_steps] | /ab stop <trial>"),
    };
    let (AbCommand::Start { trial_id, .. } | AbCommand::Stop { trial_id }) = cmd;
    writeln!(out, "trial {trial_id}: {:?}", control.ab(cmd))?;
    Ok(())
}
