use super::*;
use manifest::CatalogModel;
use std::collections::BTreeSet;
fn index(roots: &Roots) -> Result<Vec<larch_fetch::IndexEntry>> {
    let data = roots.data_dir();
    if !data.join("models").exists() {
        return Ok(vec![]);
    }
    larch_fetch::index_read(&data).map_err(|e| Error::Refused(format!("model index: {e}")))
}
/// The catalog models this machine's memory can run.
pub(super) fn fitting(manifest: &Manifest, memory: u64) -> Vec<&CatalogModel> {
    manifest
        .models
        .iter()
        .filter(|m| m.min_mem_gb.saturating_mul(1024 * 1024 * 1024) <= memory)
        .collect()
}
/// Lists a fitting catalog with an explicit skip and license acknowledgement.
pub(super) fn pick(manifest: &Manifest, memory: u64) -> Result<Option<CatalogModel>> {
    let fitting = fitting(manifest, memory);
    if fitting.is_empty() {
        return Ok(None);
    }
    for (n, m) in fitting.iter().enumerate() {
        eprintln!(
            "{}: {} ({} bytes, {} GiB memory)",
            n + 1,
            m.display,
            m.size,
            m.min_mem_gb
        );
    }
    eprint!("0: Skip\nModel [Skip]: ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    if matches!(line.trim(), "" | "0" | "Skip" | "skip") {
        return Ok(None);
    }
    let n = line
        .trim()
        .parse::<usize>()
        .map_err(|_| Error::Refused("choose a model number or Skip".into()))?;
    let m = fitting
        .get(n.wrapping_sub(1))
        .ok_or_else(|| Error::Refused("model choice is out of range".into()))?;
    if ask(&format!(
        "License: {}. Accept and download {} after commit?",
        m.license_url, m.display
    ))? {
        Ok(Some((*m).clone()))
    } else {
        Ok(None)
    }
}
/// Downloads only a revision-pinned licensed catalog file, then updates index and receipt.
pub(super) fn get(
    t: &Transaction,
    roots: &Roots,
    m: &CatalogModel,
    yes: bool,
    json: bool,
) -> Result<()> {
    manifest::relative_path(&m.file)?;
    if m.revision.len() != 40 || !m.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Refused(
            "catalog revision is not pinned; ask the release owner for a pinned catalog".into(),
        ));
    }
    if !m.repo.split('/').all(|s| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_ .".contains(&b))
            && !s.contains(' ')
    }) || m.repo.split('/').count() != 2
    {
        return Err(Error::Refused("invalid catalog repository".into()));
    }
    eprintln!("License: {}", m.license_url);
    if !yes && !ask("Accept this license and download?")? {
        return Ok(());
    }
    let data = roots.data_dir();
    let dest = data.join("models").join(manifest::relative_path(&m.file)?);
    txn::create_dir_all(dest.parent().unwrap())?;
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        m.repo, m.revision, m.file
    );
    larch_fetch::download(
        &url,
        None,
        &dest,
        &m.sha256,
        m.size,
        progress(json, "model", m.size),
    )
    .map_err(|e| Error::Refused(format!("model download: {e}")))?;
    larch_fetch::index_add(
        &data,
        larch_fetch::IndexEntry {
            file: dest.to_string_lossy().into_owned(),
            repo: m.repo.clone(),
            revision: m.revision.clone(),
            sha256: m.sha256.clone(),
            size: m.size,
            alias: None,
        },
    )
    .map_err(|e| Error::Refused(format!("model index: {e}")))?;
    if let Some(mut receipt) = t.receipt()? {
        receipt.models.retain(|e| e.file != dest.to_string_lossy());
        receipt.models.push(txn::Model {
            file: dest.to_string_lossy().into_owned(),
            repo: m.repo.clone(),
            revision: m.revision.clone(),
            sha256: m.sha256.clone(),
        });
        txn::atomic_write(
            &t.root().join("receipt.json"),
            &serde_json::to_vec(&receipt)?,
        )?;
    }
    Ok(())
}
/// Handles model commands under the installation lock; components never roll back on download errors.
pub(super) fn command(
    command: &ModelCommand,
    roots: &Roots,
    root: &Path,
    args: &Args,
) -> Result<()> {
    match command {
        ModelCommand::List => {
            let _guard = files::shared(root)?;
            let client = Client::from_env()?;
            for m in saved(&client, root)?.models {
                println!(
                    "{} {} {} bytes license={}",
                    m.file, m.display, m.size, m.license_url
                );
            }
            Ok(())
        }
        ModelCommand::Get { file } => {
            let t = acquire(root, args.txn_child)?;
            if cfg!(windows) && !args.txn_child {
                let child = running::restart(&t, &id(), Os::current()?, None, args)?;
                drop(t);
                return running::forward(child);
            }
            running::clean_old(&t)?;
            recover(&t, roots)?;
            let _machine = detect::detect_machine()?;
            let client = Client::from_env()?;
            let manifest = saved(&client, t.root())?;
            let m = manifest
                .models
                .iter()
                .find(|m| &m.file == file)
                .ok_or_else(|| Error::Refused("file is not in the signed model catalog".into()))?;
            get(&t, roots, m, args.yes, args.json)
        }
    }
}
/// Bytes of models recorded in the data directory's index.
pub(super) fn data_size(roots: &Roots) -> Result<u64> {
    Ok(index(roots)?.iter().map(|e| e.size).sum())
}
/// Asks the one installation-data question with a default of keeping everything.
pub(super) fn ask_purge(roots: &Roots) -> Result<bool> {
    let size = data_size(roots)?;
    ask(&format!(
        "Also delete your data: models {:.2} GB, chats, memory, logs at {}?",
        size as f64 / 1e9,
        roots.data_dir().display()
    ))
}
fn known_tree(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => {
            fs::remove_dir_all(path)?;
            txn::sync_parent(path)?;
        }
        Ok(_) => {
            fs::remove_file(path)?;
            txn::sync_parent(path)?;
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
fn unknown(dir: &Path, known: &BTreeSet<PathBuf>, out: &mut Vec<PathBuf>) -> Result<()> {
    if !fs::symlink_metadata(dir).is_ok_and(|m| m.is_dir()) {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let e = entry?;
        let p = e.path();
        if known.contains(&p) {
            continue;
        }
        if e.file_type()?.is_dir() {
            unknown(&p, known, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}
/// Purges recorded model files and known data trees, asking separately about unknown files.
pub(super) fn purge(roots: &Roots, receipt: Option<&Receipt>) -> Result<()> {
    let data = roots.data_dir();
    if !data.exists() {
        return Ok(());
    }
    let entries = index(roots)?;
    let models = data.join("models");
    let mut files = BTreeSet::new();
    for e in &entries {
        let p = PathBuf::from(&e.file);
        files.insert(if p.is_absolute() {
            p
        } else {
            models.join(manifest::relative_path(&e.file)?)
        });
    }
    if let Some(r) = receipt {
        for m in &r.models {
            let p = PathBuf::from(&m.file);
            files.insert(if p.is_absolute() {
                p
            } else {
                models.join(manifest::relative_path(&m.file)?)
            });
        }
    }
    let canonical = fs::canonicalize(&data)?;
    for p in files {
        let inside = p.starts_with(&models)
            && p.parent()
                .and_then(|p| fs::canonicalize(p).ok())
                .is_some_and(|p| p.starts_with(&canonical));
        if !inside {
            eprintln!("left model outside the data directory: {}", p.display());
            continue;
        }
        if fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir()) {
            eprintln!("left unexpected model directory: {}", p.display());
            continue;
        }
        match fs::remove_file(&p) {
            Ok(()) => txn::sync_parent(&p)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        for e in entries.iter().filter(|e| {
            let entry = Path::new(&e.file);
            if entry.is_absolute() {
                entry == p
            } else {
                models.join(entry) == p
            }
        }) {
            larch_fetch::index_remove(&data, &e.file)
                .map_err(|e| Error::Refused(format!("model index: {e}")))?;
        }
    }
    let mut known = BTreeSet::from([models.join("index.json"), models.join(".index.lock")]);
    for name in ["chat", "agent", "memory", "logs"] {
        let p = data.join(name);
        known_tree(&p)?;
        known.insert(p);
    }
    let mut other = Vec::new();
    unknown(&data, &known, &mut other)?;
    for p in other {
        if ask(&format!(
            "Unrecognized data file {}. Delete this file?",
            p.display()
        ))? {
            fs::remove_file(&p)?;
            txn::sync_parent(&p)?;
        } else {
            eprintln!("left unrecognized data file: {}", p.display());
        }
    }
    files::empty_dirs(&data)?;
    Ok(())
}
