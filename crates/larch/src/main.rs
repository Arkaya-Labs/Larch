//! The `larch` binary.

use std::{ffi::OsString, io, path::PathBuf};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version = larch::VERSION)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Stream a conversation through the loaded engine.
    Chat(larch::chat::Options),
    /// List real devices and backend versions.
    Info(larch::info::Options),
    /// List the static knob registry; live changes are made inside chat.
    Knob {
        #[command(subcommand)]
        cmd: KnobCmd,
    },
    /// Serve the model over OpenAI's and Anthropic's HTTP APIs.
    Serve(larch::serve::Options),
    /// The fixed benchmark against llama.cpp and Ollama (architecture 16).
    Bench(larch::bench::Options),
    /// llama.cpp's teacher-forced dump against the engine in each D16 mode; exit 0 pass, 1 gate
    /// missed, 2 not run (architecture 15).
    Parity(larch::parity::Options),
    /// Rewrite the routed experts of some layers to Q2_K, as a GGUF llama.cpp loads.
    Repack(larch::repack::Options),
    /// Make the faster copy of a model this GPU holds only in part (the 35B on a 16 GB card).
    Faster(larch::faster::Options),
    /// Prove each architecture-gated route on this card and write the card profile with those that
    /// passed.
    RoutesCheck(larch::routes_check::Options),
    /// larch's tokens against a llama.cpp tokenisation dump; exit 1 on any mismatch (10.2).
    Tokcheck {
        #[arg(short = 'm')]
        gguf: PathBuf,
        #[arg(long = "ref")]
        reference: PathBuf,
    },
    /// Run a command, restart it when it dies (architecture 10.3).
    Supervise {
        #[arg(last = true, required = true)]
        command: Vec<OsString>,
    },
}

#[derive(Subcommand)]
enum KnobCmd {
    List,
}

// Commands get an unlocked `io::stderr()`: a StderrLock held for the whole run blocks the engine's
// fault thread, so its `larch fault=` line never prints before the exit (#74).
fn main() -> anyhow::Result<()> {
    larch::supervise::exit_with_supervisor()?;
    match Cli::parse().cmd {
        Cmd::Supervise { command } => larch::supervise::run(&command, &mut io::stderr()),
        Cmd::Tokcheck { gguf, reference } => larch::tokcheck::run(&gguf, &reference),
        Cmd::Chat(opts) => {
            let _install_lock = larch_fetch::hold_install_lock()?;
            larch::chat::run(
                opts,
                io::BufReader::new(io::stdin()),
                &mut io::stdout().lock(),
                &mut io::stderr(),
            )
        }
        Cmd::Info(opts) => larch::info::run(opts, &mut io::stdout().lock()),
        Cmd::Repack(opts) => larch::repack::run(&opts),
        Cmd::Faster(opts) => larch::faster::run(&opts),
        Cmd::Knob { cmd: KnobCmd::List } => larch::knob::list(&mut io::stdout().lock()),
        Cmd::Parity(opts) => {
            let _install_lock = larch_fetch::hold_install_lock()?;
            std::process::exit(larch::parity::run(
                &opts,
                larch_engine::start,
                &mut io::stdout().lock(),
                &mut io::stderr(),
            ))
        }
        Cmd::RoutesCheck(opts) => {
            let _install_lock = larch_fetch::hold_install_lock()?;
            let profile = larch_model::card::path()
                .ok_or_else(|| anyhow::anyhow!("no per-user data directory"))?;
            larch::routes_check::run(&opts, &profile, &mut io::stdout().lock())
        }
        Cmd::Serve(opts) => {
            let _install_lock = larch_fetch::hold_install_lock()?;
            larch::serve::run(opts, &mut io::stderr())
        }
        Cmd::Bench(opts) => {
            let _install_lock = larch_fetch::hold_install_lock()?;
            larch::bench::run(opts, &mut io::stdout().lock())
        }
    }
}
