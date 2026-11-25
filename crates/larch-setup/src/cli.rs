//! Command-line coordination keeps planning pure and commits components before optional models.
pub(crate) mod files;
mod models;
mod running;
mod window;
use crate::{
    detect::{self, Os},
    effects::{self, Effects, Roots},
    fetch::Client,
    manifest::{self, Error, Manifest, Result},
    plan::{self, Paths, Request},
    selftest::{Limits, SystemHooks},
    txn::{self, Journal, Receipt, Transaction},
    unpack,
};
use clap::{Parser, Subcommand};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
#[derive(Parser)]
#[command(
    name = "larch-setup",
    version,
    about = "Install and maintain Larch for this user"
)]
struct Args {
    #[arg(long, global = true)]
    yes: bool,
    #[arg(long, global = true)]
    json: bool,
    /// Show the plan and stop before changing anything; an interrupted change is still recovered.
    #[arg(long, global = true)]
    dry_run: bool,
    #[arg(long, global = true, hide = true)]
    bootstrap: bool,
    #[arg(long, global = true, hide = true)]
    txn_child: bool,
    #[arg(long, global = true, hide = true)]
    txn_id: Option<String>,
    #[arg(long, global = true, hide = true)]
    interactive_install: bool,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    Install {
        #[arg(long)]
        cli: bool,
        #[arg(long)]
        app: bool,
        #[arg(long)]
        flavour: Option<String>,
        /// Release channel: stable, or candidate for testers.
        #[arg(long, env = "LARCH_SETUP_CHANNEL", default_value = "stable")]
        channel: String,
        #[arg(long)]
        version: Option<String>,
    },
    Modify {
        #[arg(long,value_parser=["cli","app"],num_args=1..)]
        add: Vec<String>,
        #[arg(long,value_parser=["cli","app"],num_args=1..)]
        remove: Vec<String>,
    },
    Repair,
    Update {
        #[arg(long)]
        check: bool,
        #[arg(long)]
        relaunch_app: bool,
    },
    Rollback {
        #[arg(value_parser=["cli","app","setup"])]
        component: String,
    },
    Uninstall {
        #[arg(value_parser=["cli","app"])]
        component: Option<String>,
        #[arg(long)]
        purge: bool,
    },
    Status,
    /// Open the installer window from this release's signed setup package.
    Window {
        /// Release channel: stable, or candidate for testers.
        #[arg(long, env = "LARCH_SETUP_CHANNEL", default_value = "stable")]
        channel: String,
    },
    Pin {
        version: String,
    },
    Unpin,
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}
#[derive(Subcommand)]
enum ModelCommand {
    List,
    Get { file: String },
}
/// One line for the installer window under --json, whose stdout carries nothing else.
fn emit(json: bool, value: serde_json::Value) {
    if json {
        println!("{value}");
    }
}
/// Download progress as `event` lines, at most one per whole percent.
fn progress(json: bool, component: &str, total: u64) -> impl FnMut(u64, u64) -> bool {
    let component = component.to_owned();
    let mut shown = None;
    move |done, _| {
        let percent = done.saturating_mul(100).checked_div(total).unwrap_or(100);
        if shown != Some(percent) {
            shown = Some(percent);
            emit(
                json,
                serde_json::json!({"event": "download", "component": component, "done": done, "total": total}),
            );
        }
        true
    }
}
/// A default-No answer is safe for EOF, redirected input and unattended runs.
fn ask(question: &str) -> Result<bool> {
    eprint!("{question} [y/N] ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}
fn id() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}
fn request(command: &Command) -> Result<Request> {
    Ok(match command {
        Command::Install {
            cli,
            app,
            channel,
            version,
            ..
        } => Request::Install {
            cli: *cli,
            app: *app,
            channel: channel.clone(),
            version: version.clone(),
        },
        Command::Modify { add, remove } => Request::Modify {
            add: add.clone(),
            remove: remove.clone(),
        },
        Command::Repair => Request::Repair,
        Command::Update { .. } => Request::Update,
        Command::Rollback { component } => Request::Rollback {
            component: component.clone(),
        },
        Command::Uninstall { component, purge } => Request::Uninstall {
            component: component.clone(),
            purge: *purge,
        },
        Command::Pin { version } => Request::Pin(version.clone()),
        Command::Unpin => Request::Unpin,
        _ => {
            return Err(Error::Refused(
                "this command does not mutate an installation".into(),
            ));
        }
    })
}
fn acquire(root: &Path, wait: bool) -> Result<Transaction> {
    let mut shown = false;
    loop {
        match Transaction::acquire(root) {
            Err(Error::Busy) if wait => {
                if !shown {
                    eprintln!("waiting for the app and installation lock to close");
                    shown = true;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(Error::Busy) => {
                let holders = effects::holders(root, Os::current()?)?;
                if holders.is_empty() || holders.iter().any(|h| h.name.starts_with("larch-setup")) {
                    return Err(Error::Busy);
                }
                return Err(Error::InUse(
                    holders
                        .iter()
                        .map(|h| format!("{} (pid {})", h.name, h.pid))
                        .collect::<Vec<_>>()
                        .join(", "),
                ));
            }
            other => return other,
        }
    }
}
fn hooks(
    to: Receipt,
    previous: Option<Receipt>,
    txn: String,
    roots: &Roots,
) -> Result<SystemHooks> {
    Ok(SystemHooks::new(
        to,
        previous,
        txn,
        roots.clone(),
        Effects::from_env()?,
        Limits::default(),
    ))
}
fn recover(t: &Transaction, roots: &Roots) -> Result<()> {
    let path = t.root().join("journal.json");
    if !path.exists() {
        return Ok(());
    }
    let j: Journal = serde_json::from_slice(&fs::read(path)?)?;
    let mut h = hooks(j.to.clone(), t.receipt()?, j.txn.clone(), roots)?;
    let direction = t.recover(&mut h, unpack::staged_ok)?;
    h.report_warnings()?;
    eprintln!("recovered transaction {}: {direction:?}", j.txn);
    Ok(())
}
fn saved(client: &Client, root: &Path) -> Result<Manifest> {
    client.saved(
        &fs::read(root.join("manifest.json"))?,
        &fs::read_to_string(root.join("manifest.minisig"))?,
    )
}
fn store_release(
    paths: &Paths,
    release: &crate::fetch::Release,
    plan: &mut plan::Plan,
) -> Result<()> {
    for (name, bytes) in [
        ("manifest.json", release.body.clone()),
        ("manifest.minisig", release.signature.as_bytes().to_vec()),
    ] {
        let live = paths.root.join(name);
        if paths.existing.contains(&live) {
            plan.steps.push(txn::Step::Rename {
                from: live.clone(),
                to: paths
                    .root
                    .join(".trash")
                    .join(&paths.txn)
                    .join(format!("release-cache-{name}")),
            });
        }
        let staged = paths
            .root
            .join(".staging")
            .join(&paths.txn)
            .join("release")
            .join(name);
        plan.metadata.push((staged.clone(), bytes));
        plan.steps.push(txn::Step::Rename {
            from: staged,
            to: live,
        });
    }
    Ok(())
}
fn archive_path(p: &Paths, name: &str, c: &txn::Component) -> Result<PathBuf> {
    let file = format!("{name}-{}-{}-{}.tar.zst", p.target, c.flavour, c.version);
    manifest::relative_path(&file)?;
    Ok(p.root.join(".staging").join(&p.txn).join(file))
}
fn stage(
    t: &Transaction,
    p: &Paths,
    plan: &plan::Plan,
    client: &mut Client,
    os: Os,
    json: bool,
) -> Result<Vec<PathBuf>> {
    let mut archives = Vec::new();
    for (name, a) in &plan.downloads {
        let c = &plan.to.components[name];
        let archive = archive_path(p, name, c)?;
        t.check_path(&archive)?;
        txn::create_dir_all(archive.parent().unwrap())?;
        files::resume(t, &archive, a)?;
        if !fs::metadata(&archive).is_ok_and(|m| m.len() == a.size)
            || !larch_fetch::sha256_file(&archive).is_ok_and(|h| h.eq_ignore_ascii_case(&a.sha256))
        {
            client.download(a, &archive, progress(json, name, a.size))?;
        }
        emit(
            json,
            serde_json::json!({"event": "unpack", "component": name}),
        );
        let dest = p.stage(name, os);
        t.check_path(&dest)?;
        unpack::unpack(&archive, a, &dest)?;
        archives.push(archive);
    }
    for (path, bytes) in &plan.metadata {
        t.check_path(path)?;
        txn::create_dir_all(path.parent().unwrap())?;
        txn::atomic_write(path, bytes)?;
    }
    Ok(archives)
}
fn cleanup_install(t: &Transaction, old: &Receipt, roots: &Roots) -> Result<()> {
    for c in old.components.values() {
        files::empty_dirs(&c.dest)?;
    }
    for name in [".staging", ".previous", ".trash"] {
        files::empty_dirs(&t.root().join(name))?;
    }
    for name in [
        "manifest.json",
        "manifest.minisig",
        "check-cache.json",
        "check-cache.lock",
        "setup.log",
        "setup.log.1",
        "setup.log.2",
        "setup.log.3",
        "setup.log.lock",
        "receipt.json.tmp",
        "journal.json.tmp",
    ] {
        let p = t.root().join(name);
        t.check_path(&p)?;
        match fs::symlink_metadata(&p) {
            Ok(m) if !m.is_dir() => {
                fs::remove_file(&p)?;
                txn::sync_parent(&p)?;
            }
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    if cfg!(target_os = "macos") {
        files::empty_dirs(&roots.home.join("Applications/.larch-staging"))?;
    }
    Ok(())
}
/// Prints the plan `mutate` would run and, under --json, one line with its sizes, the hardware
/// choice and the models that fit: the installer window's review screen.
fn dry_run(
    args: &Args,
    plan: &plan::Plan,
    choice: &detect::Choice,
    manifest: &Manifest,
    memory: u64,
    roots: &Roots,
    root: &Path,
) -> Result<()> {
    for line in &plan.lines {
        eprintln!("{line}");
    }
    let components: serde_json::Map<_, _> = plan
        .downloads
        .iter()
        .map(|(name, a)| {
            let unpack = a.files.iter().map(|f| f.2).sum::<u64>();
            (
                name.clone(),
                serde_json::json!({"download": a.size, "unpack": unpack}),
            )
        })
        .collect();
    emit(
        args.json,
        serde_json::json!({
            "version": manifest.version,
            "flavour": choice.flavour,
            "gpu": choice.gpu,
            "why": choice.why,
            "components": components,
            "download": plan.download_size,
            "required": plan.required_space()?,
            "free": detect::free_space(root)?,
            "root": root,
            "data": roots.data_dir(),
            "data_size": models::data_size(roots)?,
            "models": models::fitting(manifest, memory),
        }),
    );
    Ok(())
}
fn mutate(
    args: &Args,
    command: &Command,
    roots: &Roots,
    root: &Path,
    flavour: Option<&str>,
    interactive: bool,
) -> Result<()> {
    let mut req = request(command)?;
    let relaunch = matches!(
        command,
        Command::Update {
            relaunch_app: true,
            ..
        }
    );
    let t = acquire(root, relaunch || args.txn_child)?;
    let txn = args.txn_id.clone().unwrap_or_else(id);
    let os = Os::current()?;
    if cfg!(windows) && !args.txn_child && !args.dry_run {
        let child = running::restart(
            &t,
            &txn,
            os,
            if interactive { Some(command) } else { None },
            args,
        )?;
        drop(t);
        return running::forward(child);
    }
    running::clean_old(&t)?;
    recover(&t, roots)?;
    let old = t.receipt()?;
    let app = old.as_ref().and_then(|r| r.components.get("app")).cloned();
    let outcome = (|| -> Result<()> {
        let machine = detect::detect_machine()?;
        let mut client = Client::from_env()?;
        let channel = match command {
            Command::Install { channel, .. } => channel.as_str(),
            _ => old.as_ref().map_or("stable", |r| r.channel.as_str()),
        };
        let offline = matches!(
            req,
            Request::Uninstall { .. } | Request::Pin(_) | Request::Unpin | Request::Rollback { .. }
        ) || matches!(&req,Request::Modify{add,..} if add.is_empty());
        let release = if offline {
            None
        } else {
            let selected = match command {
                Command::Install { version, .. } => version.as_deref(),
                _ => None,
            };
            Some(client.release(channel, selected)?)
        };
        let manifest = match &release {
            Some(r) => r.manifest.clone(),
            None => saved(&client, t.root())?,
        };
        if manifest.channel != channel {
            return Err(Error::Refused(
                "stored release channel differs from the receipt".into(),
            ));
        }
        if args.bootstrap {
            let r = release
                .as_ref()
                .ok_or_else(|| Error::Refused("bootstrap requires install".into()))?;
            client.bootstrap(&r.head, &machine.target, &env::current_exe()?)?;
        }
        let choice = if matches!(
            req,
            Request::Uninstall { .. } | Request::Pin(_) | Request::Unpin
        ) {
            detect::Choice {
                flavour: "cpu".into(),
                gpu: None,
                why: "maintenance of recorded files; existing backends are preserved".into(),
            }
        } else {
            detect::choose(&machine, &manifest, flavour)?
        };
        if let Request::Install { cli, app, .. } = &mut req
            && !*cli
            && !*app
        {
            if !cfg!(windows) && !args.txn_child && !args.dry_run {
                let key = format!("{}-cpu", machine.target);
                let a = manifest
                    .components
                    .get("setup")
                    .and_then(|m| m.get(&key))
                    .ok_or_else(|| {
                        Error::Refused("release has no setup archive for this target".into())
                    })?;
                let version =
                    plan::artifact_version(a, "setup", &machine.target, "cpu", &manifest.version)?;
                let files = manifest::inventory(&a.files)?;
                let changed = match old.as_ref().and_then(|r| r.components.get("setup")) {
                    Some(c) => {
                        c.version != version
                            || c.flavour != "cpu"
                            || manifest::inventory(&c.files)? != files
                    }
                    None => true,
                };
                if changed {
                    return running::restart(&t, &txn, os, None, args).map(drop);
                }
            }
            if args.yes {
                *cli = true;
                *app = true;
            } else {
                eprint!("Install CLI, app, or both? [both] ");
                io::stderr().flush()?;
                let mut line = String::new();
                io::stdin().read_line(&mut line)?;
                (*cli, *app) = match line.trim() {
                    "cli" => (true, false),
                    "app" => (false, true),
                    "both" | "" => (true, true),
                    _ => return Err(Error::Refused("choose cli, app, or both".into())),
                };
            }
        }
        let paths = files::snapshot(t.root(), roots, &machine.target, &txn, os)?;
        let mut plan = plan::plan(old.as_ref(), &manifest, &req, &choice, &paths, os)?;
        if args.dry_run {
            return dry_run(args, &plan, &choice, &manifest, machine.memory, roots, root);
        }
        let changes_setup = plan
            .steps
            .iter()
            .any(|s| matches!(s,txn::Step::Rename{to,..} if to==&paths.dest("setup",os)));
        if !cfg!(windows) && changes_setup && !args.txn_child {
            return running::restart(
                &t,
                &txn,
                os,
                if interactive { Some(command) } else { None },
                args,
            )
            .map(drop);
        }
        for line in &plan.lines {
            eprintln!("{line}");
        }
        if let Request::Uninstall {
            component: None, ..
        } = &req
            && let Some(r) = &old
        {
            for (name, c) in &r.components {
                eprintln!(
                    "remove {name}: {} bytes at {}",
                    c.files.iter().map(|f| f.2).sum::<u64>(),
                    c.dest.display()
                );
            }
        }
        let purge = if matches!(
            req,
            Request::Uninstall {
                component: None,
                purge: true
            }
        ) {
            true
        } else if matches!(
            req,
            Request::Uninstall {
                component: None,
                purge: false
            }
        ) && !args.yes
        {
            models::ask_purge(roots)?
        } else {
            false
        };
        if !args.yes && !ask("Apply this plan?")? {
            return Ok(());
        }
        let model = if interactive {
            models::pick(&manifest, machine.memory)?
        } else {
            None
        };
        if plan.required_space()? > detect::free_space(t.root())? {
            return Err(Error::Refused(
                "not enough free space including the ten percent safety margin".into(),
            ));
        }
        if os == Os::Macos
            && plan.downloads.iter().any(|(n, _)| n == "app")
            && plan.required_space()? > detect::free_space(&paths.dest("app", os))?
        {
            return Err(Error::Refused(
                "not enough free space on the Applications volume".into(),
            ));
        }
        if let Some(r) = &release {
            store_release(&paths, r, &mut plan)?;
        }
        let archives = stage(&t, &paths, &plan, &mut client, os, args.json)?;
        emit(args.json, serde_json::json!({"event": "apply"}));
        let mut h = hooks(plan.to.clone(), old.clone(), txn.clone(), roots)?;
        let journal = Journal::new(
            txn.clone(),
            t.receipt_digest()?,
            plan.to.clone(),
            plan.steps,
        );
        let result = t.run(journal, &mut h);
        if let Err(error) = result {
            if h.failed_gpu() {
                emit(args.json, serde_json::json!({"event": "gpu_failed"}));
                eprintln!(
                    "GPU acceptance was refused and rolled back; install the cpu flavour as a new transaction with larch-setup install --flavour cpu"
                );
            }
            if h.failed_gpu()
                && !args.yes
                && ask(
                    "GPU acceptance failed and was rolled back. Install the cpu flavour as a new transaction?",
                )?
            {
                drop(t);
                let retry = Args {
                    txn_child: true,
                    txn_id: None,
                    bootstrap: false,
                    yes: args.yes,
                    json: args.json,
                    dry_run: false,
                    command: None,
                    interactive_install: false,
                };
                let command = Command::Install {
                    cli: plan.to.components.contains_key("cli"),
                    app: plan.to.components.contains_key("app"),
                    flavour: Some("cpu".into()),
                    channel: channel.into(),
                    version: None,
                };
                return mutate(&retry, &command, roots, root, Some("cpu"), false);
            }
            return Err(error);
        }
        h.report_warnings()?;
        files::clean_downloads(&archives)?;
        files::empty_dirs(&t.root().join(".staging"))?;
        if matches!(
            req,
            Request::Uninstall {
                component: None,
                ..
            }
        ) {
            if purge {
                models::purge(roots, old.as_ref())?;
            }
            cleanup_install(&t, old.as_ref().unwrap_or(&plan.to), roots)?;
            fs::remove_file(t.root().join("receipt.json"))?;
            txn::sync_parent(&t.root().join("receipt.json"))?;
            running::finish_uninstall(&t)?;
            let root = t.root().to_owned();
            fs::remove_file(root.join(".lock"))?;
            drop(t);
            files::empty_dirs(&root)?;
            files::leftovers(&root)?;
        } else {
            if let Some(model) = model
                && let Err(e) = models::get(&t, roots, &model, true, args.json)
            {
                return Err(Error::Refused(format!(
                    "components committed; model download failed: {e}; retry model get"
                )));
            }
            if args.json {
                println!("{}", serde_json::to_string(&t.receipt()?)?);
            } else {
                eprintln!("committed transaction {txn}");
            }
        }
        Ok(())
    })();
    if relaunch && let Some(app) = app {
        running::launch(&app, outcome.is_err().then_some(txn.as_str()), roots)?;
    }
    outcome
}
/// Runs the documented CLI; executable boundaries report a single sanitized failure line.
pub fn run() -> Result<()> {
    let mut args = Args::parse();
    let roots = Roots::from_env()?;
    let root = larch_fetch::install_root().map_err(|e| Error::Refused(e.to_string()))?;
    let interactive = args.command.is_none()
        || args.interactive_install
        || matches!(
            args.command,
            Some(Command::Install {
                cli: false,
                app: false,
                ..
            })
        ) && !args.yes;
    if args.command.is_none() {
        // A double-click installs with install's own defaults, its channel variable included.
        args.command = Args::parse_from(["larch-setup", "install"]).command;
    }
    let command = args.command.as_ref().unwrap();
    match command {
        Command::Status => {
            let _guard = if root.join(".lock").exists() {
                Some(files::shared(&root)?)
            } else {
                None
            };
            let r = files::receipt(&root)?;
            if args.json {
                println!("{}", serde_json::to_string(&r)?);
            } else if let Some(r) = r {
                eprintln!(
                    "channel: {}; pin: {:?}; flavour: {}; why: {}; last_txn: {}",
                    r.channel, r.pinned, r.detect.flavour, r.detect.why, r.last_txn
                );
                for (n, c) in r.components {
                    println!("{n} {} {} {}", c.version, c.flavour, c.dest.display());
                }
            } else {
                println!("not installed");
            }
            Ok(())
        }
        Command::Update { check: true, .. } => {
            let result = crate::check()?;
            if args.json {
                println!("{}", serde_json::to_string(&result)?);
            } else if let Some(v) = result.version {
                println!(
                    "{v}: {} bytes for {}",
                    result.download_size,
                    result.components.join(", ")
                );
            } else {
                println!("up to date");
            }
            Ok(())
        }
        Command::Model { command } => models::command(command, &roots, &root, &args),
        Command::Window { channel } => window::open(&args, channel, &roots),
        Command::Install { flavour, .. } => mutate(
            &args,
            command,
            &roots,
            &root,
            flavour.as_deref(),
            interactive,
        ),
        _ => mutate(&args, command, &roots, &root, None, interactive),
    }
}
/// Starts the installed app, as `update --relaunch-app` does: the installer window's last button.
pub fn open_app() -> Result<()> {
    let root = larch_fetch::install_root().map_err(|e| Error::Refused(e.to_string()))?;
    let app = files::receipt(&root)?
        .and_then(|r| r.components.get("app").cloned())
        .ok_or_else(|| Error::Refused("the app is not installed".into()))?;
    running::launch(&app, None, &Roots::from_env()?)
}
/// The larch-setup program: one actionable failure line, and under --json an `error` line for
/// the installer window. It never loads an engine.
pub fn main() {
    let interactive =
        env::args_os().len() == 1 || env::args_os().any(|a| a == "--interactive-install");
    let child = env::args_os().any(|a| a == "--txn-child");
    let result = run();
    if let Err(error) = &result {
        let token = env::var("LARCH_RELEASES_TOKEN").ok();
        let busy = matches!(error, Error::Busy | Error::InUse(_));
        emit(
            env::args_os().any(|a| a == "--json"),
            serde_json::json!({"event": "error", "message": crate::log::clean(&error.to_string(), token.as_deref()), "busy": busy}),
        );
        if busy {
            eprintln!(
                "{}; state: unchanged; next: close the holder and retry",
                crate::log::clean(&error.to_string(), token.as_deref())
            );
        } else {
            let root =
                larch_fetch::install_root().unwrap_or_else(|_| env::temp_dir().join("larch-setup"));
            crate::log::Log::new(&root).failure(
                &error.to_string(),
                "inspect status; any in-flight journal is retained for recovery",
                "run larch-setup repair, or retry the command",
            );
        }
    }
    if cfg!(windows) && interactive && (child || result.is_err()) {
        eprintln!("Press Enter to close");
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
    }
    if result.is_err() {
        std::process::exit(1);
    }
}
