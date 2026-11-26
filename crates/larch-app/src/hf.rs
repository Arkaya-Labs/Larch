//! Models from the Hugging Face Hub: a repo's GGUFs, a remote header read with range requests, and
//! resumable, verified downloads into the shared Hugging Face cache (the layout huggingface_hub and
//! llama.cpp read, so a model downloads once for every tool), or into a Larch library the user
//! picks. The Hub is `LARCH_HF_ENDPOINT` (default
//! `https://huggingface.co`; tests point it at a local server); `HF_TOKEN`, when set, is sent as a
//! bearer token (never across a redirect). These commands use only the disk and the network, so they
//! work with no model loaded and after a GPU fault.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, BufReader},
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use larch_fetch::{Ranged, agent, agent_unfollowed};
use larch_model::{
    ModelConfig,
    gguf::{Gguf, Value},
};
use serde::Deserialize;

use crate::{
    App, Event, Inner, contract::*, known_types, quant_label, runnable, trunk, unloadable,
};

const TICK: Duration = Duration::from_millis(250);
const AUTH: &str = "This model needs a Hugging Face account: accept its license on huggingface.co and set HF_TOKEN.";

pub(crate) struct Hub {
    endpoint: Mutex<String>,
    agent: Mutex<ureq::Agent>,
    /// The agent for the one request whose redirect is read, not followed ([`Hub::pin`]).
    unfollowed: Mutex<ureq::Agent>,
    /// Each repo's GGUFs as last listed, with the part sizes and LFS hashes downloads verify.
    trees: Mutex<HashMap<String, Vec<Remote>>>,
    inspected: Mutex<HashMap<(String, String), RemoteModel>>,
    /// Whether each repo projector read so far is one the app loads ([`Hub::loads`]).
    projectors: Mutex<HashMap<(String, String), bool>>,
    jobs: Mutex<Vec<Job>>,
    next_id: AtomicU64,
}

#[derive(Clone)]
struct Remote {
    file: HfFile,
    parts: Vec<Part>,
    /// The repo's vision projector for this model, downloaded beside it.
    projector: Option<Part>,
}

impl Remote {
    /// The files a download writes: the parts, then the projector.
    fn files(&self) -> impl Iterator<Item = &Part> {
        self.parts.iter().chain(&self.projector)
    }
}

#[derive(Clone)]
struct Part {
    path: String,
    size: u64,
    /// The LFS oid, hex.
    sha256: Option<String>,
    /// Its name in `<data>/models` when not the repo's (a projector named for its model).
    local: Option<String>,
}

impl Part {
    /// Its file name in `<data>/models`.
    fn name(&self) -> &str {
        match &self.local {
            Some(local) => local,
            None => self.path.rsplit('/').next().unwrap_or_default(),
        }
    }

    /// The Hugging Face cache can hold it: an LFS hash to name its blob, and a path of plain names
    /// (no `..`, root or drive) to file it under in a snapshot. Both come from the Hub.
    fn caches(&self) -> bool {
        self.sha256
            .as_deref()
            .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
            && Path::new(&self.path)
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
    }
}

/// Where a download writes.
enum Target {
    /// A Larch library, each file by its name.
    Library(PathBuf),
    /// The Hugging Face cache's folder for the repo ([`repo_dir`]) and the commit its snapshot is
    /// named for ([`Hub::pin`]): each file is a blob named by its hash, linked from
    /// `snapshots/<commit>/<path>`.
    Cache(PathBuf, String),
}

impl Target {
    /// Where the download of `p` writes, and where the model is then listed.
    fn paths(&self, p: &Part) -> (PathBuf, PathBuf) {
        match self {
            Target::Library(dir) => (dir.join(p.name()), dir.join(p.name())),
            Target::Cache(dir, commit) => (
                dir.join("blobs")
                    .join(p.sha256.as_deref().unwrap_or_default()),
                // The Hub's path a part at a time: one `/` joined whole stays in a Windows path.
                (p.path.split('/'))
                    .fold(dir.join("snapshots").join(commit), |at, part| at.join(part)),
            ),
        }
    }
}

/// Where a download is to go, before its thread asks the Hub which commit it fetches.
struct Plan {
    /// The library asked for, or Larch's own.
    library: PathBuf,
    /// The repo's folder in the Hugging Face cache, when the download goes there.
    cache: Option<PathBuf>,
    /// Why a download that would go in the cache goes to the library.
    note: Option<String>,
}

/// The Hugging Face cache's folder for `repo` (`owner/name`, as [`Hub::list`] checked it).
fn repo_dir(hub: &Path, repo: &str) -> PathBuf {
    hub.join(format!("models--{}", repo.replace('/', "--")))
}

struct Job {
    info: Download,
    stop: Arc<AtomicBool>,
    /// The file names it writes in `<data>/models`.
    names: Vec<String>,
}

/// Why a download ended without the model.
enum Stop {
    Cancelled,
    Failed(String),
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Hub {
    pub(crate) fn new() -> Hub {
        let endpoint =
            std::env::var("LARCH_HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
        let endpoint = endpoint.trim_end_matches('/').to_owned();
        Hub {
            agent: Mutex::new(agent(&endpoint)),
            unfollowed: Mutex::new(agent_unfollowed(&endpoint)),
            endpoint: Mutex::new(endpoint),
            trees: Mutex::default(),
            inspected: Mutex::default(),
            projectors: Mutex::default(),
            jobs: Mutex::default(),
            next_id: AtomicU64::new(1),
        }
    }

    /// One GET, tried again while the Hub or the network says "later" (a dropped or reset
    /// connection, 429, a 5xx): after 0.5, 1.5 and 4 s, or as Retry-After asks, at most 5 s.
    fn get(&self, url: &str, range: Option<String>) -> Result<ureq::Response, Box<ureq::Error>> {
        let range = range.as_deref().map(|r| ("Range", r));
        let token = std::env::var("HF_TOKEN").ok();
        larch_fetch::get(&self.agent(), url, token.as_deref(), range.as_slice())
    }

    /// The agent, cloned out of its lock: a guard held across a request made every header read,
    /// listing and download start wait for the one before it.
    fn agent(&self) -> ureq::Agent {
        lock(&self.agent).clone()
    }

    fn resolve_url(&self, repo: &str, revision: &str, path: &str) -> String {
        format!("{}/{repo}/resolve/{revision}/{path}", lock(&self.endpoint))
    }

    /// Every GGUF in `repo`, fetched fresh, smallest first.
    fn list(&self, repo: &str) -> Result<Vec<Remote>, String> {
        self.list_at(repo, "main")
    }

    /// Every GGUF in `repo` at `revision` (`main`, or a commit [`Hub::pin`] checked), fetched
    /// fresh, smallest first; `main`'s is kept for [`Hub::remote`].
    fn list_at(&self, repo: &str, revision: &str) -> Result<Vec<Remote>, String> {
        let valid = repo.split_once('/').is_some_and(|(owner, name)| {
            [owner, name].iter().all(|s| {
                !s.is_empty()
                    && !s.starts_with('.')
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            })
        });
        if !valid {
            return Err(format!(
                "{repo} is not a Hugging Face model name (owner/name)."
            ));
        }
        #[derive(Deserialize)]
        struct Entry {
            #[serde(rename = "type")]
            kind: String,
            path: String,
            #[serde(default)]
            size: u64,
            lfs: Option<Lfs>,
        }
        #[derive(Deserialize)]
        struct Lfs {
            oid: String,
            size: u64,
        }
        let mut url = Some(format!(
            "{}/api/models/{repo}/tree/{revision}?recursive=true",
            lock(&self.endpoint)
        ));
        let mut parts = Vec::new();
        while let Some(at) = url.take() {
            let resp = self.get(&at, None).map_err(|e| match *e {
                // The Hub answers 401 for a repo that does not exist, too.
                ureq::Error::Status(401 | 403, _) => format!(
                    "Hugging Face has no model {repo} you can open: check its name, or set HF_TOKEN if it is private or gated."
                ),
                ureq::Error::Status(404, _) => format!("Hugging Face has no such model: {repo}."),
                e => http_error(Box::new(e), repo),
            })?;
            url = resp.header("link").and_then(next_link);
            let page: Vec<Entry> = serde_json::from_reader(resp.into_reader()).map_err(|e| {
                format!("Hugging Face sent an unreadable file list for {repo}: {e}")
            })?;
            parts.extend(page.into_iter().filter(|e| e.kind == "file").map(|e| Part {
                size: e.lfs.as_ref().map_or(e.size, |l| l.size),
                sha256: e.lfs.map(|l| l.oid),
                path: e.path,
                local: None,
            }));
        }
        let mut remotes = group(parts);
        // A projector the app does not load (another kind of vision model) neither downloads nor
        // says the model reads images.
        for r in &mut remotes {
            if let Some(p) = r.projector.take_if(|p| !self.loads(repo, &p.path)) {
                r.file.size_bytes -= p.size;
                r.file.mmproj = None;
            }
        }
        if revision == "main" {
            lock(&self.trees).insert(repo.into(), remotes.clone());
        }
        Ok(remotes)
    }

    /// Whether the repo's projector at `path` is one the app loads ([`crate::reads_images`], from
    /// its header: one range request, remembered). Unreadable is no, and asked again next listing.
    fn loads(&self, repo: &str, path: &str) -> bool {
        let key = (repo.to_owned(), path.to_owned());
        if let Some(&ok) = lock(&self.projectors).get(&key) {
            return ok;
        }
        let Ok((g, _)) = self.header(repo, path) else {
            return false;
        };
        let ok = crate::reads_images(&g);
        lock(&self.projectors).insert(key, ok);
        ok
    }

    /// The commit `main` of `repo` is at, from the Hub's own answer to resolving `remote`'s first
    /// file (its `x-repo-commit`, as huggingface_hub reads it: the CDN that answer redirects to
    /// sends none), and `remote` as listed at that commit, so every file comes from that one commit
    /// and checks against its hashes. None when the Hub names no commit, as a mirror may not.
    fn pin(&self, repo: &str, remote: &Remote) -> Result<Option<(String, Remote)>, String> {
        let first = &remote.parts[0].path;
        let what = format!("{repo}/{first}");
        let token = std::env::var("HF_TOKEN").ok();
        let agent = lock(&self.unfollowed).clone();
        let mut url = self.resolve_url(repo, "main", first);
        let mut commit = None;
        // A renamed repo redirects to its new name on the Hub itself (a path), followed here; a
        // redirect to another host is the CDN's.
        for _ in 0..3 {
            let resp = larch_fetch::get(&agent, &url, token.as_deref(), &[("Range", "bytes=0-0")])
                .map_err(|e| http_error(e, &what))?;
            commit = resp.header("x-repo-commit").map(str::to_owned);
            match resp.header("location") {
                Some(to) if commit.is_none() && to.starts_with('/') => {
                    url = format!("{}{to}", lock(&self.endpoint));
                }
                _ => break,
            }
        }
        // It names a folder: a commit hash and nothing else.
        let Some(commit) =
            commit.filter(|c| c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            return Ok(None);
        };
        let pinned = (self.list_at(repo, &commit)?.into_iter())
            .find(|r| r.file.file == remote.file.file)
            .ok_or_else(|| {
                lock(&self.trees).remove(repo);
                format!(
                    "{} is no longer in {repo} on Hugging Face: list its files again.",
                    remote.file.file
                )
            })?;
        Ok(Some((commit, pinned)))
    }

    /// `file` of `repo` as last listed, listing the repo if it is not there.
    fn remote(&self, repo: &str, file: &str) -> Result<Remote, String> {
        let find = |list: &[Remote]| list.iter().find(|r| r.file.file == file).cloned();
        let cached = lock(&self.trees).get(repo).and_then(|l| find(l));
        match cached {
            Some(r) => Ok(r),
            None => find(&self.list(repo)?).ok_or_else(|| no_file(repo, file)),
        }
    }

    /// The header of one remote GGUF and the file's size.
    fn header(&self, repo: &str, path: &str) -> Result<(Gguf, u64), String> {
        let url = self.resolve_url(repo, "main", path);
        let what = format!("{repo}/{path}");
        let mut r = Ranged::new(
            |range| {
                self.get(&url, Some(range))
                    .map_err(|e| io::Error::other(http_error(e, &what)))
            },
            "Hugging Face",
            what.clone(),
        )
        .map_err(|e| e.to_string())?;
        let len = r.size();
        let g = Gguf::read(BufReader::new(&mut r), len);
        match (g, r.error().map(str::to_owned)) {
            (Ok(g), _) => Ok((g, len)),
            (Err(_), Some(e)) => Err(e),
            (Err(e), None) => Err(format!("{path} is not a GGUF Larch can read: {e:?}")),
        }
    }
}

/// The `rel="next"` URL of a `Link` header.
fn next_link(link: &str) -> Option<String> {
    link.split(',')
        .find(|l| l.contains("rel=\"next\""))
        .and_then(|l| Some(l.split_once('<')?.1.split_once('>')?.0.to_owned()))
}

fn no_file(repo: &str, file: &str) -> String {
    format!("Hugging Face has no such file: {repo}/{file}.")
}

fn http_error(e: Box<ureq::Error>, what: &str) -> String {
    match *e {
        ureq::Error::Status(401 | 403, _) => AUTH.into(),
        ureq::Error::Status(404, _) => format!("Hugging Face has no such file: {what}."),
        ureq::Error::Status(code, _) => format!("Hugging Face answered {code} for {what}."),
        ureq::Error::Transport(t) => format!("Cannot reach Hugging Face: {t}"),
    }
}

/// larch runs the architecture `arch`: [`ModelConfig::from_gguf`], the one list of them, refuses
/// any other with `UnsupportedArch` before it reads a key.
/// A mixture-of-experts model, by its GGUF architecture (`qwen35moe`) or its name: "MoE", or an
/// active-parameter word such as `A3B` (`Qwen3.5-35B-A3B`).
pub fn is_moe(name: &str, arch: &str) -> bool {
    arch.ends_with("moe")
        || name.to_lowercase().contains("moe")
        || name.split([' ', '-', '_', '/']).any(|w| {
            w.len() > 2
                && w.starts_with(['A', 'a'])
                && w.ends_with(['B', 'b'])
                && w[1..w.len() - 1]
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.')
        })
}

fn arch_runs(arch: &str) -> bool {
    let g = Gguf {
        kv: [("general.architecture".into(), Value::Str(arch.into()))].into(),
        tensors: Vec::new(),
        data_offset: 0,
    };
    !arch.is_empty()
        && !matches!(
            ModelConfig::from_gguf(&g),
            Err(larch_records::LoadError::UnsupportedArch)
        )
}

/// `name-00001-of-00003.gguf` → (`name`, 1, 3).
pub(crate) fn split_part(path: &str) -> Option<(&str, u32, u32)> {
    let (rest, n) = path[..path.len() - ".gguf".len()].rsplit_once("-of-")?;
    let (stem, i) = rest.rsplit_once('-')?;
    let num = |s: &str| {
        Some(s)
            .filter(|s| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))?
            .parse()
            .ok()
    };
    Some((stem, num(i)?, num(n)?))
}

/// The GGUFs among a repo's files, a split set as one entry (only when every part is there),
/// smallest first. A vision projector is not one: each model takes the one that matches it
/// ([`pick_projector`](crate::pick_projector); any of the repo's, as a GGUF repo is one model in
/// several quants), named so the app finds it beside the model.
fn group(files: Vec<Part>) -> Vec<Remote> {
    let mut sets: Vec<Vec<Part>> = Vec::new();
    let mut splits: BTreeMap<(String, u32), Vec<(u32, Part)>> = BTreeMap::new();
    let mut projectors = Vec::new();
    for p in files {
        let name = p.name().to_ascii_lowercase();
        if !name.ends_with(".gguf") || p.path.contains('\\') {
            continue;
        }
        if name.contains("mmproj") {
            if split_part(&p.path).is_none() {
                projectors.push(p);
            }
            continue;
        }
        match split_part(&p.path) {
            Some((stem, i, n)) => splits.entry((stem.into(), n)).or_default().push((i, p)),
            None => sets.push(vec![p]),
        }
    }
    for ((_, n), mut ps) in splits {
        ps.sort_by_key(|&(i, _)| i);
        if ps.iter().map(|&(i, _)| i).eq(1..=n) {
            sets.push(ps.into_iter().map(|(_, p)| p).collect());
        }
    }
    let mut remotes: Vec<_> = sets
        .into_iter()
        .map(|parts| {
            let projector = projector_for(parts[0].name(), &projectors);
            Remote {
                file: HfFile {
                    file: parts[0].path.clone(),
                    parts: parts.iter().map(|p| p.path.clone()).collect(),
                    size_bytes: parts.iter().chain(&projector).map(|p| p.size).sum(),
                    quant: quant(parts[0].name()),
                    downloaded: false,
                    mmproj: projector.as_ref().map(|p| p.path.clone()),
                },
                parts,
                projector,
            }
        })
        .collect();
    remotes
        .sort_by(|a, b| (a.file.size_bytes, &a.file.file).cmp(&(b.file.size_bytes, &b.file.file)));
    remotes
}

/// The projector of `projectors` for the model file `model` ([`pick_projector`](crate::pick_projector):
/// any of the repo's, f16 first), named to sit beside it: its own name when that already pairs, else
/// `mmproj-<model stem>-f16.gguf` (or no precision when its name has none), which every quant of the
/// model shares and `mmproj_beside` pairs with each.
fn projector_for(model: &str, projectors: &[Part]) -> Option<Part> {
    let pick = crate::pick_projector(model, projectors.iter().map(Part::name), true)?;
    let mut p = projectors.iter().find(|p| p.name() == pick)?.clone();
    if crate::pick_projector(model, [pick], false).is_none() {
        let precision = if quant(pick).is_empty() { "" } else { "-f16" };
        p.local = Some(format!(
            "mmproj-{}{precision}.gguf",
            crate::model_stem(model)
        ));
    }
    Some(p)
}

/// The quant type in a GGUF file name (`model-Q4_K_M.gguf` → `Q4_K_M`), or "".
pub(crate) fn quant(name: &str) -> String {
    let stem = name[..name.len() - ".gguf".len()].to_ascii_uppercase();
    let is_quant = |s: &str| {
        if matches!(s, "F16" | "F32" | "BF16" | "F64" | "MXFP4" | "MXFP4_MOE") {
            return true;
        }
        let Some(rest) = ["IQ", "TQ", "Q"].iter().find_map(|p| s.strip_prefix(p)) else {
            return false;
        };
        let mut groups = rest.split('_');
        groups
            .next()
            .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
            && groups.all(|g| !g.is_empty() && g.bytes().all(|b| b.is_ascii_alphanumeric()))
    };
    // The last `-`/`.` separated token that is a quant, or ends in one after a `_`.
    stem.rsplit(['-', '.'])
        .find_map(|token| {
            let starts = std::iter::once(0).chain(token.match_indices('_').map(|(i, _)| i + 1));
            starts.map(|i| &token[i..]).find(|s| is_quant(s))
        })
        .unwrap_or_default()
        .into()
}

/// Every part is in `dir` with its exact size.
fn downloaded<'a>(dir: &Path, mut parts: impl Iterator<Item = &'a Part>) -> bool {
    parts.all(|p| fs::metadata(dir.join(p.name())).is_ok_and(|m| m.len() == p.size))
}

/// Every part is in a snapshot of the repo's cache folder `dir` ([`holds`]), whichever tool
/// downloaded it.
fn cached<'a>(dir: &Path, mut parts: impl Iterator<Item = &'a Part>) -> bool {
    let snapshots: Vec<PathBuf> = (fs::read_dir(dir.join("snapshots")).into_iter().flatten())
        .flatten()
        .map(|e| e.path())
        .collect();
    parts.all(|p| p.caches() && snapshots.iter().any(|s| holds(&s.join(&p.path), p)))
}

/// The snapshot entry `f` is `p`: its size, and when it is a link (not a copy, where links could
/// not be made) one to the blob named by `p`'s hash, not an older revision's of the same path.
fn holds(f: &Path, p: &Part) -> bool {
    fs::metadata(f).is_ok_and(|m| m.len() == p.size)
        && fs::read_link(f).map_or(true, |t| {
            t.file_name().and_then(|n| n.to_str()) == p.sha256.as_deref()
        })
}

/// Holds huggingface_hub's lock on the blob of `p` in the repo's cache folder `dir`
/// (`<cache>/.locks/models--<owner>--<name>/<sha256>.lock`, which its filelock takes too) until the
/// file is dropped. Another download of the blob, by Larch or another tool, is waited for, with
/// Cancel heard meanwhile: two writers would share one partial file.
fn lock_blob(dir: &Path, p: &Part, stop: &AtomicBool) -> Result<fs::File, Stop> {
    let locks = dir
        .with_file_name(".locks")
        .join(dir.file_name().unwrap_or_default());
    let path = locks.join(format!("{}.lock", p.sha256.as_deref().unwrap_or_default()));
    let failed = |e: io::Error| Stop::Failed(format!("cannot lock {}: {e}", path.display()));
    let file = fs::create_dir_all(&locks)
        .and_then(|()| {
            fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&path)
        })
        .map_err(failed)?;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(fs::TryLockError::WouldBlock) if !stop.load(Ordering::Relaxed) => {
                std::thread::sleep(TICK);
            }
            Err(fs::TryLockError::WouldBlock) => return Err(Stop::Cancelled),
            Err(fs::TryLockError::Error(e)) => return Err(failed(e)),
        }
    }
}

/// A temporary file's name, `<stem>-<pid>-<n>.tmp`: no other download, in this process or another,
/// uses it at the same time (one's `File::create` would truncate the other's, and its rename fail).
fn temp_name(stem: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{stem}-{}-{n}.tmp", std::process::id())
}

/// Larch can write in the folder `dir`, made if it is not there, as `add_library` checks a folder.
fn writable(dir: &Path) -> io::Result<()> {
    let probe = dir.join(temp_name(".larch-write-check"));
    fs::create_dir_all(dir)
        .and_then(|()| fs::write(&probe, b""))
        .and_then(|()| fs::remove_file(&probe))
}

/// Puts the blob `blob` at `path` in a cache snapshot, `depth` folders below the commit's, as
/// huggingface_hub does: a relative link, or [`place`] where none can be made.
fn link(blob: &Path, path: &Path, depth: usize, new: bool) -> io::Result<()> {
    link_with(blob, path, depth, new, |target, path| {
        #[cfg(unix)]
        return std::os::unix::fs::symlink(target, path);
        #[cfg(windows)]
        return std::os::windows::fs::symlink_file(target, path);
    })
}

/// [`link`] with `symlink` making the link (a test's one fails, as Windows' does without Developer
/// Mode).
fn link_with(
    blob: &Path,
    path: &Path,
    depth: usize,
    new: bool,
    symlink: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    // A file of the wrong size, or a link to a blob that is gone or another revision's.
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    // Up from the file's folder past the commit's folder and `snapshots`, to the repo's.
    let mut target: PathBuf = std::iter::repeat_n("..", depth + 2).collect();
    target.push("blobs");
    target.push(blob.file_name().unwrap_or_default());
    if symlink(&target, path).is_ok() {
        return Ok(());
    }
    place(blob, path, new)
}

/// Where the system makes no links (Windows without Developer Mode, the common case there, and
/// some disks), the file itself goes in the snapshot, as huggingface_hub puts it: moved when this
/// download fetched it (`new`), else copied, since another snapshot may link to the blob.
fn place(blob: &Path, path: &Path, new: bool) -> io::Result<()> {
    if new {
        fs::rename(blob, path)
    } else {
        fs::copy(blob, path).map(drop)
    }
}

impl App {
    /// Points the Hugging Face commands at another Hub (`LARCH_HF_ENDPOINT` sets it at start).
    pub fn set_hf_endpoint(&self, url: &str) {
        let hub = &self.inner.hub;
        *lock(&hub.endpoint) = url.trim_end_matches('/').into();
        *lock(&hub.agent) = agent(url);
        *lock(&hub.unfollowed) = agent_unfollowed(url);
        lock(&hub.trees).clear();
        lock(&hub.inspected).clear();
    }

    /// Chat repos with GGUF files matching `query` (all of them when empty), by `sort`:
    /// `downloads`, `likes` or `lastModified`, most first. A chat repo is tagged text-generation,
    /// image-text-to-text (Qwen3.5, Gemma 3: the most downloaded GGUFs) or nothing; the Hub's
    /// pipeline_tag filter takes one tag only, so the others are dropped here. The Hub matches
    /// words of the repo name, so "moe" and "gguf" are not sent: every result is a GGUF repo, and
    /// "moe" keeps the mixture-of-experts repos ([`is_moe`]), most of which (`-A3B`) do not say MoE.
    pub fn hf_search(&self, query: &str, sort: &str, limit: u32) -> Result<Vec<HfModel>, String> {
        if !["downloads", "likes", "lastModified"].contains(&sort) {
            return Err(format!("{sort} is not a way to sort models."));
        }
        let (moe, words): (Vec<&str>, Vec<&str>) = query
            .split_whitespace()
            .filter(|w| !w.eq_ignore_ascii_case("gguf"))
            .partition(|w| w.eq_ignore_ascii_case("moe"));
        let query = words.join(" ");
        let moe = !moe.is_empty();
        let hub = &self.inner.hub;
        // An MoE search asks for the most the Hub returns, then keeps `limit` of the MoE ones.
        let ask = if moe { 100 } else { limit };
        let mut url = format!(
            "{}/api/models?filter=gguf&sort={sort}&direction=-1&limit={}&expand%5B%5D=downloads&expand%5B%5D=tags&expand%5B%5D=gguf&expand%5B%5D=pipeline_tag",
            lock(&hub.endpoint),
            ask.clamp(1, 100)
        );
        if !query.trim().is_empty() {
            url += "&search=";
            // Percent-encoding by hand: the query is the only free text in any Hub URL.
            for b in query.trim().bytes() {
                if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                    url.push(b as char);
                } else {
                    url += &format!("%{b:02X}");
                }
            }
        }
        #[derive(Deserialize)]
        struct Found {
            id: String,
            #[serde(default)]
            downloads: u64,
            #[serde(default)]
            tags: Vec<String>,
            #[serde(default)]
            gguf: Summary,
            pipeline_tag: Option<String>,
        }
        #[derive(Default, Deserialize)]
        struct Summary {
            #[serde(default)]
            architecture: String,
            total: Option<u64>,
            context_length: Option<u64>,
        }
        let resp = hub
            .get(&url, None)
            .map_err(|e| http_error(e, "the model search"))?;
        let found: Vec<Found> = serde_json::from_reader(resp.into_reader())
            .map_err(|e| format!("Hugging Face sent an unreadable search result: {e}"))?;
        Ok(found
            .into_iter()
            .filter(|f| {
                f.pipeline_tag
                    .as_deref()
                    .is_none_or(|t| ["text-generation", "image-text-to-text"].contains(&t))
                    && (!moe || is_moe(&f.id, &f.gguf.architecture))
            })
            .take(limit as usize)
            .map(|mut f| {
                // The Hub's tags need not name the pipeline: a repo whose pipeline is
                // image-text-to-text says so, which is how the app knows it reads images before
                // listing its files.
                const VISION: &str = "image-text-to-text";
                if f.pipeline_tag.as_deref() == Some(VISION) && !f.tags.iter().any(|t| t == VISION)
                {
                    f.tags.push(VISION.into());
                }
                f
            })
            .map(|f| HfModel {
                repo: f.id,
                downloads: f.downloads,
                tags: f.tags,
                arch_runs: arch_runs(&f.gguf.architecture),
                arch: f.gguf.architecture,
                params: f.gguf.total,
                context_length: f.gguf.context_length,
            })
            .collect())
    }

    /// The GGUFs in `repo`, smallest first.
    pub fn hf_files(&self, repo: &str) -> Result<Vec<HfFile>, String> {
        Ok(self
            .inner
            .hub
            .list(repo)?
            .into_iter()
            .map(|r| HfFile {
                downloaded: self.inner.has(repo, &r),
                ..r.file
            })
            .collect())
    }

    /// What `file` of `repo` holds, from the headers of all its parts (one range request each, for
    /// a few MiB at most). Parts after the first carry only their tensor tables.
    pub fn hf_inspect(&self, repo: &str, file: &str) -> Result<RemoteModel, String> {
        let hub = &self.inner.hub;
        let key = (repo.to_owned(), file.to_owned());
        if let Some(m) = lock(&hub.inspected).get(&key) {
            return Ok(m.clone());
        }
        let remote = hub.remote(repo, file)?;
        let heads = remote
            .parts
            .iter()
            .map(|p| hub.header(repo, &p.path))
            .collect::<Result<Vec<_>, _>>()?;
        let g = &heads[0].0;
        let arch = g
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let u = |k: &str| g.get(&format!("{arch}.{k}")).and_then(Value::as_u64);
        let kv_bytes_per_token = (|| {
            let n_heads = u("attention.head_count")?;
            // A per-layer array is not read here; no estimate rather than a wrong one.
            let kv_heads = match g.get(&format!("{arch}.attention.head_count_kv")) {
                Some(v) => v.as_u64()?,
                None => n_heads,
            };
            let head_dim = u("attention.key_length")
                .or_else(|| Some(u("embedding_length")? / n_heads.max(1)))?;
            // The layers holding KV: a hybrid's attention layers only, without its MTP block.
            let layers =
                ModelConfig::from_gguf(g).map_or(u("block_count"), |c| Some(c.kv_layers.into()));
            [layers?, kv_heads, head_dim]
                .into_iter()
                .try_fold(4u64, u64::checked_mul)
        })();
        let tensors = heads.iter().flat_map(|(g, _)| &g.tensors);
        let trunk = trunk(g);
        let loaded = tensors.clone().filter(|t| trunk(t));
        let m = RemoteModel {
            supported: runnable(g) && known_types(loaded.clone()),
            quant: quant_label(g, tensors.clone()),
            // A tensor of a type larch does not know has no size here: such a part counts as its
            // data section (its size after the header), which is its tensors plus alignment padding.
            weights_bytes: heads
                .iter()
                .map(|(g, len)| {
                    if known_types(&g.tensors) {
                        g.tensors.iter().map(|t| t.bytes).sum()
                    } else {
                        len.saturating_sub(g.data_offset)
                    }
                })
                .sum(),
            params: tensors
                .clone()
                .map(|t| t.dims.iter().product::<u64>())
                .fold(0, u64::saturating_add),
            kv_bytes_per_token,
            context_length: u("context_length"),
            expert_bytes: tensors
                .clone()
                .filter(|t| t.name.ends_with("_exps.weight"))
                .map(|t| t.bytes)
                .sum(),
            unloadable: unloadable(loaded),
            arch,
        };
        lock(&hub.inspected).insert(key, m.clone());
        Ok(m)
    }

    /// Starts downloading `file` of `repo` (every part) into `library`, or with None into the
    /// shared Hugging Face cache; returns its id. Its progress and end arrive as `download:*`
    /// events. A download for the cache goes to Larch's own library instead, and its
    /// `download:done` says why, when there is no cache, Larch cannot write there, the Hub gave a
    /// file no LFS hash, or it names no commit for the download (a mirror).
    pub fn download_model(
        &self,
        repo: &str,
        file: &str,
        library: Option<&str>,
    ) -> Result<String, String> {
        let dir = self.inner.library(library)?;
        let hub = &self.inner.hub;
        let remote = hub.remote(repo, file)?;
        if self.inner.has(repo, &remote) {
            return Err(format!("{file} is already downloaded."));
        }
        let cache = (self.inner.hf_cache())
            .filter(|_| library.is_none() && remote.files().all(Part::caches));
        let plan = match cache.map(|c| writable(&c).map(|()| c.clone()).map_err(|e| (c, e))) {
            Some(Ok(cache)) => Plan {
                library: dir,
                cache: Some(repo_dir(&cache, repo)),
                note: None,
            },
            Some(Err((cache, e))) => Plan {
                library: dir,
                cache: None,
                note: Some(format!(
                    "Larch cannot write to the Hugging Face cache at {} ({e}), so it went to Larch's own library.",
                    cache.display()
                )),
            },
            None => Plan {
                library: dir,
                cache: None,
                note: None,
            },
        };
        let names: Vec<String> = remote.files().map(|p| p.name().into()).collect();
        let mut jobs = lock(&hub.jobs);
        let busy = jobs
            .iter()
            .any(|j| j.info.state == "running" && j.names.iter().any(|n| names.contains(n)));
        if busy {
            return Err(format!("{file} is already downloading."));
        }
        let id = hub.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let stop = Arc::new(AtomicBool::new(false));
        jobs.push(Job {
            info: Download {
                id: id.clone(),
                repo: repo.into(),
                file: file.into(),
                received_bytes: 0,
                total_bytes: remote.file.size_bytes,
                state: "running".into(),
            },
            stop: stop.clone(),
            names,
        });
        drop(jobs);
        // Before the thread exists, so a window shows the row ahead of an end that comes at once.
        (self.inner.emit)(Event::DownloadProgress(DownloadProgress {
            id: id.clone(),
            repo: repo.into(),
            file: file.into(),
            received_bytes: 0,
            total_bytes: remote.file.size_bytes,
            bytes_per_sec: 0.,
        }));
        let (inner, job_id, owned) = (self.inner.clone(), id.clone(), repo.to_owned());
        std::thread::Builder::new()
            .name("larch-download".into())
            .spawn(move || inner.download(&job_id, &owned, remote, plan, &stop))
            .map_err(|e| {
                // No thread will end the job: end it here as a failed download ends, or the row the
                // window already shows reads as downloading for good.
                let message = format!("The download could not start: {e}");
                if let Some(j) = lock(&self.inner.hub.jobs)
                    .iter_mut()
                    .find(|j| j.info.id == id)
                {
                    j.info.state = "failed".into();
                }
                (self.inner.emit)(Event::DownloadFailed(DownloadFailed {
                    id: id.clone(),
                    repo: repo.into(),
                    file: file.into(),
                    message: message.clone(),
                    cancelled: false,
                }));
                message
            })?;
        Ok(id)
    }

    /// Stops a running download and deletes what it fetched; `download:failed` (cancelled) follows.
    /// A download that already ended is left as it is.
    pub fn cancel_download(&self, id: &str) -> Result<(), String> {
        let jobs = lock(&self.inner.hub.jobs);
        let job = jobs
            .iter()
            .find(|j| j.info.id == id)
            .ok_or("No such download.")?;
        job.stop.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// The downloads of this session, running and ended.
    pub fn list_downloads(&self) -> Result<Vec<Download>, String> {
        Ok(lock(&self.inner.hub.jobs)
            .iter()
            .map(|j| j.info.clone())
            .collect())
    }
}

/// Bytes received and when to say so.
struct Progress {
    received: u64,
    /// Bytes fetched in this run (not resumed ones), for the rate.
    fetched: u64,
    started: Instant,
    last: Instant,
}

impl Inner {
    /// The Hugging Face cache, where downloads go unless the user picks a library: the folder
    /// list_models reads it from.
    pub(crate) fn hf_cache(&self) -> Option<PathBuf> {
        (self.others.lock().unwrap_or_else(PoisonError::into_inner))
            .iter()
            .find(|(app, ..)| app == "Hugging Face")
            .map(|(_, dir, _)| dir.clone())
    }

    /// `remote` of `repo` is whole in a library or in the Hugging Face cache.
    fn has(&self, repo: &str, remote: &Remote) -> bool {
        self.libraries()
            .iter()
            .any(|d| downloaded(d, remote.files()))
            || (self.hf_cache()).is_some_and(|c| cached(&repo_dir(&c, repo), remote.files()))
    }

    /// Runs one download on its own thread and sends how it ended.
    fn download(&self, id: &str, repo: &str, remote: Remote, plan: Plan, stop: &AtomicBool) {
        let file = remote.file.file.clone();
        let mut note = plan.note;
        let for_cache = plan.cache.is_some();
        // In the cache, every file comes from the one commit the Hub resolves the first one to.
        let pinned = match plan.cache {
            Some(dir) => (self.hub.pin(repo, &remote))
                .map(|pin| pin.map(|(commit, at)| (Target::Cache(dir, commit), at))),
            None => Ok(None),
        };
        let result = pinned.map_err(Stop::Failed).and_then(|pinned| {
            let (to, remote) = pinned.unwrap_or_else(|| {
                if for_cache {
                    note = Some(
                        "Hugging Face named no commit for it, so it went to Larch's own library."
                            .into(),
                    );
                }
                (Target::Library(plan.library), remote)
            });
            let mut fetched = Vec::new();
            let got = self.fetch(id, repo, &remote, &to, stop, &mut fetched);
            if let Err(Stop::Cancelled) = got {
                // A cache blob's partial went in `fetch`, under its lock: past it, another tool
                // may be writing it.
                if let Target::Library(_) = to {
                    for p in remote.files() {
                        let _ = larch_fetch::remove_partial(&to.paths(p).0);
                    }
                }
                // A split model's finished parts too: alone, part 1 is listed and cannot load.
                for f in &fetched {
                    let _ = crate::remove_model_file(f);
                    let _ = larch_fetch::index_remove(&self.data, &f.to_string_lossy());
                }
            }
            got
        });
        let (state, event) = match result {
            Ok(path) => (
                "done",
                Event::DownloadDone(DownloadDone {
                    id: id.into(),
                    repo: repo.into(),
                    file,
                    path: path.to_string_lossy().into(),
                    note,
                }),
            ),
            Err(stop) => {
                let cancelled = matches!(stop, Stop::Cancelled);
                let message = match stop {
                    Stop::Cancelled => "The download was cancelled.".into(),
                    Stop::Failed(m) => m,
                };
                let state = if cancelled { "cancelled" } else { "failed" };
                let event = Event::DownloadFailed(DownloadFailed {
                    id: id.into(),
                    repo: repo.into(),
                    file,
                    message,
                    cancelled,
                });
                (state, event)
            }
        };
        if let Some(j) = lock(&self.hub.jobs).iter_mut().find(|j| j.info.id == id) {
            j.info.state = state.into();
            if state == "done" {
                j.info.received_bytes = j.info.total_bytes;
            }
        }
        (self.emit)(event);
    }

    /// Fetches every part not yet at `to`, and the projector; returns the first part's path.
    /// `fetched` gets each part this run wrote whole, for a cancel to delete.
    fn fetch(
        &self,
        id: &str,
        repo: &str,
        remote: &Remote,
        to: &Target,
        stop: &AtomicBool,
        fetched: &mut Vec<PathBuf>,
    ) -> Result<PathBuf, Stop> {
        let now = Instant::now();
        let mut progress = Progress {
            received: 0,
            fetched: 0,
            started: now,
            last: now,
        };
        if let Target::Cache(dir, _) = to {
            // A repo folder with no `snapshots` is one `hf cache scan` calls corrupt.
            fs::create_dir_all(dir.join("blobs"))
                .and_then(|()| fs::create_dir_all(dir.join("snapshots")))
                .map_err(|e| Stop::Failed(format!("cannot write {}: {e}", dir.display())))?;
        }
        for p in remote.files() {
            let (dest, path) = to.paths(p);
            if let Target::Cache(dir, commit) = to {
                // Not indexed: the cache is every tool's, and uninstalling Larch leaves it.
                let at = (dir.as_path(), commit.as_str());
                if self.cache_part(id, repo, remote, p, at, (&dest, &path), stop, &mut progress)? {
                    fetched.push(path);
                }
                continue;
            }
            let (hash, revision) = if !fs::metadata(&dest).is_ok_and(|m| m.len() == p.size) {
                let got =
                    self.fetch_part(id, repo, remote, p, &dest, "main", stop, &mut progress)?;
                fetched.push(dest.clone());
                got
            } else {
                // Hashing a part already there takes seconds a GB: Cancel is heard first.
                if stop.load(Ordering::Relaxed) {
                    return Err(Stop::Cancelled);
                }
                progress.received += p.size;
                (
                    larch_fetch::sha256_file(&dest).map_err(|e| Stop::Failed(e.to_string()))?,
                    "main".into(),
                )
            };
            larch_fetch::index_add(
                &self.data,
                larch_fetch::IndexEntry {
                    file: dest
                        .to_str()
                        .ok_or_else(|| {
                            Stop::Failed("cannot record a non-Unicode model path".into())
                        })?
                        .into(),
                    repo: repo.into(),
                    revision,
                    sha256: hash,
                    size: p.size,
                    alias: None,
                },
            )
            .map_err(|e| Stop::Failed(format!("cannot record model download: {e}")))?;
        }
        if let Target::Cache(dir, commit) = to {
            // Only once every file is in place, and whole: until then `main` names the commit
            // whose files the tools that resolve it offline still find.
            let refs = dir.join("refs");
            let temp = dir.join(temp_name(".refs-main"));
            fs::create_dir_all(&refs)
                .map_err(|e| e.to_string())
                .and_then(|()| crate::write_synced(&refs.join("main"), &temp, commit.as_bytes()))
                .map_err(|e| Stop::Failed(format!("cannot write {}: {e}", refs.display())))?;
        }
        Ok(to.paths(&remote.parts[0]).1)
    }

    /// Puts `p` at `path` in the snapshot of the repo's cache folder and commit `at`, from its blob
    /// `dest`, under the blob's lock: nothing when the snapshot holds it, a link when the blob is
    /// there and hashes right, else a fetch first. True when this run fetched the blob.
    #[allow(clippy::too_many_arguments)]
    fn cache_part(
        &self,
        id: &str,
        repo: &str,
        remote: &Remote,
        p: &Part,
        (dir, commit): (&Path, &str),
        (dest, path): (&Path, &Path),
        stop: &AtomicBool,
        progress: &mut Progress,
    ) -> Result<bool, Stop> {
        let _held = lock_blob(dir, p, stop)?;
        if holds(path, p) {
            progress.received += p.size;
            return Ok(false);
        }
        // A blob this run did not fetch (another snapshot's, or a download stopped before its
        // link) is hashed before it goes live: its name is the hash it must have.
        let whole = fs::metadata(dest).is_ok_and(|m| m.len() == p.size);
        let good = whole && {
            if stop.load(Ordering::Relaxed) {
                return Err(Stop::Cancelled);
            }
            let hash = larch_fetch::sha256_file(dest).map_err(|e| Stop::Failed(e.to_string()))?;
            p.sha256
                .as_deref()
                .is_some_and(|h| h.eq_ignore_ascii_case(&hash))
        };
        if good {
            progress.received += p.size;
        } else {
            // A damaged blob goes: a fetch would resume past its end.
            if whole {
                fs::remove_file(dest)
                    .map_err(|e| Stop::Failed(format!("cannot replace {}: {e}", dest.display())))?;
            }
            let got = self.fetch_part(id, repo, remote, p, dest, commit, stop, progress);
            // Its partial is this run's only while the lock is held.
            if let Err(Stop::Cancelled) = got {
                let _ = larch_fetch::remove_partial(dest);
            }
            got?;
        }
        link(dest, path, p.path.matches('/').count(), !good)
            .map_err(|e| Stop::Failed(format!("cannot save {}: {e}", path.display())))?;
        Ok(!good)
    }

    /// Fetches one part at revision `at` into `dest`'s partial file, resuming what is there, and
    /// checks its size and hash.
    #[allow(clippy::too_many_arguments)]
    fn fetch_part(
        &self,
        id: &str,
        repo: &str,
        remote: &Remote,
        p: &Part,
        dest: &Path,
        at: &str,
        stop: &AtomicBool,
        progress: &mut Progress,
    ) -> Result<(String, String), Stop> {
        use larch_fetch::DownloadError as E;
        let path = larch_fetch::part_path(dest);
        let what = format!("{repo}/{}", p.path);
        let token = std::env::var("HF_TOKEN").ok();
        let (received, fetched) = (progress.received, progress.fetched);
        // ponytail: retain the symbolic revision when the server omits x-repo-commit;
        // require an immutable revision when the Hub contract guarantees one.
        let revision = Mutex::new("main".to_owned());
        // Where resolve redirected (the CDN's signed URL): the ranges after the first go there
        // straight, since a Hub round trip per 8 MiB piece would cost about as much as the piece.
        // It takes no token, and one the CDN refuses (it expired) sends the next range via resolve.
        let direct = Mutex::new(None::<String>);
        let url = self.hub.resolve_url(repo, at, &p.path);
        let request = |start: u64, end: Option<u64>, etag: Option<&str>| {
            let range = match end {
                Some(end) => format!("bytes={start}-{end}"),
                None => format!("bytes={start}-"),
            };
            let mut headers = vec![("Range", range.as_str())];
            if let Some(etag) = etag {
                headers.push(("If-Range", etag));
            }
            let agent = self.hub.agent();
            let cdn = lock(&direct).clone();
            if let Some(cdn) = cdn {
                match larch_fetch::get(&agent, &cdn, None, &headers) {
                    Err(e) if matches!(*e, ureq::Error::Status(401 | 403 | 404 | 410, _)) => {
                        *lock(&direct) = None;
                    }
                    r => return r,
                }
            }
            let response = larch_fetch::get(&agent, &url, token.as_deref(), &headers)?;
            if response.get_url() != url {
                *lock(&direct) = Some(response.get_url().to_owned());
            }
            if let Some(commit) = response.header("x-repo-commit") {
                *lock(&revision) = commit.into();
            }
            Ok(response)
        };
        larch_fetch::download_parallel(dest, p.sha256.as_deref(), p.size, request, |have, got| {
            progress.received = received + have;
            progress.fetched = fetched + got;
            if progress.last.elapsed() >= TICK {
                progress.last = Instant::now();
                self.progress(id, repo, remote, progress);
            }
            !stop.load(Ordering::Relaxed)
        }).map(|hash| (hash, revision.into_inner().unwrap_or_else(PoisonError::into_inner))).map_err(|e| match e {
            E::Cancelled => Stop::Cancelled,
            E::Http(e) => Stop::Failed(http_error(e, &what)),
            E::Write(e) => Stop::Failed(format!("cannot write {}: {e}", path.display())),
            E::Save(e) => Stop::Failed(format!("cannot save {}: {e}", dest.display())),
            E::Read(e) => Stop::Failed(format!("The download of {what} stopped: {e}. Try again to resume it.")),
            E::Short => Stop::Failed(format!("The download of {what} stopped early. Try again to resume it.")),
            E::Range => Stop::Failed(format!("Hugging Face sent the wrong part of {what}.")),
            E::Oversize => Stop::Failed(format!("Hugging Face sent more of {what} than its size; the download was deleted. Try again.")),
            E::Checksum => {
                // The hash it was checked against may be an older listing's: the next try lists again.
                lock(&self.hub.trees).remove(repo);
                Stop::Failed(format!("{what} does not match its checksum on Hugging Face; the download was deleted. Try again."))
            }
        })
    }

    fn progress(&self, id: &str, repo: &str, remote: &Remote, p: &Progress) {
        let total_bytes = remote.file.size_bytes;
        if let Some(j) = lock(&self.hub.jobs).iter_mut().find(|j| j.info.id == id) {
            j.info.received_bytes = p.received;
        }
        (self.emit)(Event::DownloadProgress(DownloadProgress {
            id: id.into(),
            repo: repo.into(),
            file: remote.file.file.clone(),
            received_bytes: p.received,
            total_bytes,
            bytes_per_sec: p.fetched as f64 / p.started.elapsed().as_secs_f64().max(1e-3),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finished_parts_are_recorded_in_the_model_index() {
        let data = std::env::temp_dir().join(format!("larch-model-index-{}", std::process::id()));
        fs::create_dir(&data).unwrap();
        let app = App::new(data.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let dest = data.join("models").join("model.gguf");
        fs::write(&dest, b"model bytes").unwrap();
        let remote = Remote {
            file: HfFile {
                file: "model.gguf".into(),
                parts: vec!["model.gguf".into()],
                size_bytes: 11,
                quant: String::new(),
                downloaded: false,
                mmproj: None,
            },
            parts: vec![Part {
                path: "model.gguf".into(),
                size: 11,
                sha256: None,
                local: None,
            }],
            projector: None,
        };
        assert!(
            app.inner
                .fetch(
                    "job",
                    "org/model",
                    &remote,
                    &Target::Library(data.join("models")),
                    &AtomicBool::new(false),
                    &mut Vec::new()
                )
                .is_ok()
        );
        // A part already there is not hashed after Cancel: it takes seconds a GB.
        let stopped = &AtomicBool::new(true);
        let models = Target::Library(data.join("models"));
        let got = (app.inner).fetch(
            "job",
            "org/model",
            &remote,
            &models,
            stopped,
            &mut Vec::new(),
        );
        assert!(matches!(got, Err(Stop::Cancelled)));
        assert_eq!(
            larch_fetch::index_read(&data).unwrap(),
            [larch_fetch::IndexEntry {
                file: dest.to_string_lossy().into(),
                repo: "org/model".into(),
                revision: "main".into(),
                sha256: larch_fetch::sha256_file(&dest).unwrap(),
                size: 11,
                alias: None,
            }]
        );
        drop(app);
        fs::remove_dir_all(data).unwrap();
    }

    /// Where no link can be made, as on Windows without Developer Mode, a blob this download
    /// fetched moves into the snapshot and one there before is copied; deleting the snapshot's
    /// copy leaves the blob, which another snapshot may hold.
    #[test]
    fn without_links_the_file_goes_in_the_snapshot() {
        let dir = std::env::temp_dir().join(format!("larch-no-links-{}", std::process::id()));
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let no_links = |_: &Path, _: &Path| Err(io::Error::other("no links here"));
        let (new, old) = (blobs.join("aa"), blobs.join("bb"));
        fs::write(&new, b"new").unwrap();
        fs::write(&old, b"old").unwrap();
        let snapshot = dir.join("snapshots/c0ffee");
        let (moved, copied) = (snapshot.join("Q4/m.gguf"), snapshot.join("o.gguf"));
        link_with(&new, &moved, 1, true, no_links).unwrap();
        link_with(&old, &copied, 0, false, no_links).unwrap();
        assert_eq!(fs::read(&moved).unwrap(), b"new");
        assert!(!new.exists());
        assert_eq!(fs::read(&copied).unwrap(), b"old");
        assert!(old.exists());
        crate::remove_model_file(&copied).unwrap();
        assert!(!copied.exists() && old.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn quant_from_file_names() {
        for (name, want) in [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model.Q8_0.gguf", "Q8_0"),
            ("model-f16.gguf", "F16"),
            ("model-bf16-00001-of-00002.gguf", "BF16"),
            ("Model-IQ4_XS.gguf", "IQ4_XS"),
            ("model-UD-Q4_K_XL.gguf", "Q4_K_XL"),
            ("model_q5_k_s.gguf", "Q5_K_S"),
            ("model-7B-instruct.gguf", ""),
        ] {
            assert_eq!(quant(name), want, "{name}");
        }
    }

    #[test]
    fn hub_architectures_larch_runs() {
        assert!(arch_runs("qwen35") && arch_runs("llama"));
        assert!(!arch_runs("dflash") && !arch_runs("clip") && !arch_runs(""));
    }

    #[test]
    fn link_header_next() {
        let link =
            r#"<https://hub/api?cursor=a>; rel="prev", <https://hub/api?cursor=b>; rel="next""#;
        assert_eq!(next_link(link).as_deref(), Some("https://hub/api?cursor=b"));
        assert_eq!(next_link(r#"<https://hub/x>; rel="prev""#), None);
    }
}
