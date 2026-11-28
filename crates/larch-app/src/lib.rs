//! The desktop app's core: what the UI calls and what it is sent (`contract`), over the larch facade.
//! Tauri-free, so it is tested on the CPU reference like every other crate.

pub mod connections;
pub mod contract;
mod faster;
pub mod handoff;
mod hf;
pub use hf::is_moe;
pub mod mcp;
pub mod task;
mod telemetry;

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    fs::{self, File},
    io::{self, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use contract::*;
use larch::{
    ChatMessage, EngineError, GenParams, Larch, LoadOptions, Session, SessionOptions, TextEvent,
    api::{Class, Grammar, invalid},
    chat::{Metrics, Terminal, Transcript, transcript_path},
};
use larch_model::{
    ModelConfig,
    gguf::{self, Gguf, TensorInfo, Value},
};
use larch_records::{
    Authority, HealthState, KnobReply, KnobRequest, REGISTRY, Read, ReadFrom, Record, RevertTarget,
    RingHandle,
};

use larch_agent::{
    Agent, AgentEvent, AgentOptions,
    code::{
        allow::{self, Allowed, Ask, Decision},
        r#loop::{self, Code, Mode, Outcome},
        record::{self, TaskRecord},
        task::{Budget, CodeTask, Overrides, TestRun},
    },
    config, memory,
    router::{self, Intent},
    runtime::inference::{SeatAdmission, Seated},
};

use handoff::fit_estimate;

/// What the core sends the UI; [`Event::name`] and [`Event::payload`] are what Tauri emits.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// `chat:<id>`.
    Chat { id: String, event: ChatEvent },
    /// `engine:sample`.
    Sample(EngineSample),
    /// `engine:config`.
    Config(ConfigChange),
    /// `engine:failed`.
    Failed(Failed),
    /// `engine:health`.
    Health(Health),
    /// `download:progress`.
    DownloadProgress(DownloadProgress),
    /// `download:done`.
    DownloadDone(DownloadDone),
    /// `download:failed`.
    DownloadFailed(DownloadFailed),
    /// `faster:progress`.
    FasterProgress(FasterProgress),
    /// `faster:done`.
    FasterDone(FasterDone),
    /// `faster:failed`.
    FasterFailed(FasterFailed),
    /// `code:allowed`.
    Allowed(FolderAllowed),
    /// `mcp:status`.
    Mcp(McpChanged),
}

impl Event {
    /// The event name in contract.rs.
    pub fn name(&self) -> String {
        match self {
            Event::Chat { id, .. } => format!("chat:{id}"),
            Event::Sample(_) => "engine:sample".into(),
            Event::Config(_) => "engine:config".into(),
            Event::Failed(_) => "engine:failed".into(),
            Event::Health(_) => "engine:health".into(),
            Event::DownloadProgress(_) => "download:progress".into(),
            Event::DownloadDone(_) => "download:done".into(),
            Event::DownloadFailed(_) => "download:failed".into(),
            Event::FasterProgress(_) => "faster:progress".into(),
            Event::FasterDone(_) => "faster:done".into(),
            Event::FasterFailed(_) => "faster:failed".into(),
            Event::Allowed(_) => "code:allowed".into(),
            Event::Mcp(_) => "mcp:status".into(),
        }
    }

    /// The JSON payload in contract.rs.
    pub fn payload(&self) -> serde_json::Value {
        match self {
            Event::Chat { event, .. } => serde_json::to_value(event),
            Event::Sample(s) => serde_json::to_value(s),
            Event::Config(c) => serde_json::to_value(c),
            Event::Failed(f) => serde_json::to_value(f),
            Event::Health(h) => serde_json::to_value(h),
            Event::DownloadProgress(p) => serde_json::to_value(p),
            Event::DownloadDone(d) => serde_json::to_value(d),
            Event::DownloadFailed(f) => serde_json::to_value(f),
            Event::FasterProgress(p) => serde_json::to_value(p),
            Event::FasterDone(d) => serde_json::to_value(d),
            Event::FasterFailed(f) => serde_json::to_value(f),
            Event::Allowed(a) => serde_json::to_value(a),
            Event::Mcp(m) => serde_json::to_value(m),
        }
        .expect("contract types serialize")
    }
}

type Emit = Arc<dyn Fn(Event) + Send + Sync>;

const STOPPED: &str = "The GPU stopped. Restart Larch.";
/// The error of a call a chat refuses while a reply is being written in it, such as an attach:
/// the desktop keeps the attach for the chat's next message.
pub const BUSY: &str = "a reply is still being written in this chat";
const NO_MODEL: &str = "Load a model first.";
/// `NO_MODEL` while one loads: a message sent then was told to load the model just loaded.
const LOADING: &str = "The model is still loading. Try again once it is ready.";
const PARKED: &str = "Approve or discard this chat's code packet first.";
/// Why images cannot go to the loaded model: its projector failed to load (`why`), or it has none
/// (the agent's own answer to an image on a text-only model).
fn no_vision(why: Option<&str>) -> String {
    match why {
        Some(why) => {
            format!("This model's vision projector did not load ({why}), so it can't see images.")
        }
        None => larch_agent::NO_VISION.into(),
    }
}
/// Code mode runs its tests and commands only through the sandbox, which Linux, macOS and Windows
/// have. It is built only with the test-only `code-mode` feature: it scores below the plain model
/// on Polyglot (14/30 vs 17/30, 15 Nov) and waits on a redo, so no release build can turn it on.
/// There, it is still off unless settings.json says `"code_mode": true` (`Inner::code_mode`).
const SANDBOXED: bool = cfg!(all(
    feature = "code-mode",
    any(target_os = "linux", target_os = "macos", windows)
));
/// What the app's code tasks are tagged with: `chat-<id>`, so their worktrees under
/// `<data>/agent/code` are told apart from the CLI's, which the sweep at start must not touch.
const CHAT_TAG: &str = "chat-";
/// Why a code run is refused when settings.json leaves code mode off.
const CODE_MODE_OFF: &str =
    "Code mode is off. Set \"code_mode\": true in settings.json and restart to turn it on.";
/// The answer to a change asked in a chat with no folder to make it in.
const NO_FOLDER: &str = "Pick a project folder and send it again, and I'll make the change there.";
/// The prefix cache's blocks of 16 tokens when LARCH_PREFIX_CACHE_BLOCKS is unset: larch-agent's
/// default, 16,384 tokens, the chat's anchors and a reading prompt of 5,000 tokens with room to
/// spare. The engine keeps at most a quarter of the KV pool for it, which ends it lower on a small
/// pool.
const PREFIX_CACHE_BLOCKS: usize = 1024;

use larch_agent::runtime::service::DeviceLease;

/// The device leases this process holds, by canonical data folder. The lock is per open file, so
/// a second App over the same folder (a resumed window) shares the lease instead of being refused
/// by its own process.
struct Leases {
    /// Every lease alive in the process, a load's tentative one included.
    live: Vec<(PathBuf, std::sync::Weak<DeviceLease>)>,
    /// Successful loads' leases: a loaded engine can outlive its window through tuner or worker
    /// handles, so another process must not load while a clone may exist.
    // ponytail: kept until process exit; release on unload once tuner and worker handles join.
    kept: Vec<Arc<DeviceLease>>,
}
static LEASES: Mutex<Leases> = Mutex::new(Leases {
    live: Vec::new(),
    kept: Vec::new(),
});
fn leases() -> MutexGuard<'static, Leases> {
    LEASES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The data folder's device lease, shared with every other App over that folder in this process.
/// It guards the folder's `runtime-service/device.lock`, not the GPU: a process using another
/// data folder is not refused.
fn device_lease(data: &Path) -> Result<Arc<DeviceLease>, String> {
    // `data`, `data/` and a symlink to it name one lock file.
    let data =
        fs::canonicalize(data).map_err(|e| format!("cannot open {}: {e}", data.display()))?;
    // Held from lookup to insert, so two loads at once never both open the lock file.
    let mut leases = leases();
    leases.live.retain(|(_, lease)| lease.strong_count() > 0);
    let shared = leases.live.iter().find(|(dir, _)| *dir == data);
    if let Some(lease) = shared.and_then(|(_, lease)| lease.upgrade()) {
        return Ok(lease);
    }
    runtime_dir(&data)?;
    let lease = Arc::new(DeviceLease::acquire(&data)?);
    leases.live.push((data, Arc::downgrade(&lease)));
    Ok(lease)
}

/// Keeps a loaded engine's lease until process exit (`Leases::kept`).
fn keep_lease(lease: Arc<DeviceLease>) {
    let mut leases = leases();
    if !leases.kept.iter().any(|kept| Arc::ptr_eq(kept, &lease)) {
        leases.kept.push(lease);
    }
}

/// Drops a failed load's lease under the table's lock: a lookup that then misses it finds the lock
/// file closed, not still closing.
fn release_lease(lease: Arc<DeviceLease>) {
    let _table = leases();
    drop(lease);
}

/// Creates `<data>/runtime-service` private to this user, as the service does, before anything
/// opens a file there: task goals, results and decisions are stored in it.
// ponytail: repeats host.rs's create-then-chmod; call larch-task's helper once it exports one.
// On Windows the data folder's inherited per-user ACL guards it, as the service relies on.
fn runtime_dir(data: &Path) -> Result<(), String> {
    let dir = data.join("runtime-service");
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Also tightens a folder an older build or the device lease made with the umask's mode.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("cannot make {} private: {e}", dir.display()))?;
    }
    Ok(())
}

/// How long an unload waits for stopped replies to let their engine go (`State::unload`), with the
/// app's lock held: a swapping model's one pass can take longer, and the user then tries again.
const STOPPING: Duration = Duration::from_secs(5);

/// How long a chat may go without a turn before it lets its pinned documents go (written to their
/// pin file first while `docs_keep_ready` is on): while any pin is held the engine stays on the
/// documents plan, where plain chats decode about a fifth slower (a Mac bench). The chat's next
/// question imports the file again, or reads the documents again without one.
// ponytail: fixed at 5 minutes, the import (about 13 s on an M5) is the cost of a question after
// it; revisit if users ask again after a short pause and wait for it.
const PIN_IDLE: Duration = Duration::from_secs(5 * 60);

/// How often the chats are looked at for [`PIN_IDLE`] (`Inner::release_idle`).
const PIN_IDLE_TICK: Duration = Duration::from_secs(30);

/// The app: at most one engine, the chats open on it, and the data folder. Every command returns
/// `Err` with a message for the user; after a GPU fault every command returns [`STOPPED`], except
/// the Hugging Face ones (`hf.rs`), which use only the disk and the network.
pub struct App {
    inner: Arc<Inner>,
}

struct Inner {
    data: PathBuf,
    device: i32,
    emit: Emit,
    /// The task service this app hosts: None until `task_service` finds none to connect to, then
    /// the one it started, kept after it failed or ended so the next `task_service` restarts it
    /// rather than connecting to another process's. Model loads and deletes hold it throughout.
    /// Before `state`, so on drop the service stops before the engine it holds a clone of.
    tasks: Mutex<Option<task::Hosted>>,
    state: Mutex<State>,
    hub: hf::Hub,
    /// The faster copies being made ([`faster`]): each original's path and its stop flag.
    building: Mutex<Vec<(PathBuf, Arc<AtomicBool>)>>,
    /// Other apps' model folders that list_models also reads: (app, folder, levels below it).
    others: Mutex<Vec<(String, PathBuf, usize)>>,
    /// The GPU's name, total and free memory, read the first time the UI asks (before a model
    /// takes any); None on the CPU reference or when the device cannot be read.
    gpu: OnceLock<Option<(String, u64, u64)>>,
    /// Larch's model libraries: `<data>/models` first, then the folders the user added
    /// (`<data>/settings.json`). Models download into one of them.
    libraries: Mutex<Vec<PathBuf>>,
    /// "Let tasks use the web", kept in settings.json (on by default): a task saved while it is
    /// on may have its steps call web_search and fetch_page, while it stays on.
    web: Arc<AtomicBool>,
    /// `"code_mode"` in settings.json (default off), read once at start: a change in a chat with a
    /// folder starts a code run only with it on, on an OS with the sandbox (`Inner::coding_on`).
    code_mode: bool,
    /// Worktrees of an earlier run of the app that the sweep at start could not remove; the
    /// next code run's chat is told once.
    orphans: Mutex<Vec<String>>,
    /// Why the vision projector beside a model did not load the last time it loaded, by the
    /// model's path: its listing then says it reads no images.
    projector_errors: Mutex<HashMap<String, String>>,
    /// The chats being named (`Inner::name_chat`). Its own lock, not the state's: unload waits
    /// for the jobs while holding the state.
    names: Mutex<Names>,
    /// The MCP servers the user installed (`<data>/mcp.json`); their tools join each new chat.
    mcp: mcp::Hub,
    /// Dropped with the app, which stops its idle pin watch (`watch_idle_pins`).
    _idle_watch: mpsc::Sender<()>,
}

/// The naming jobs, and the chats already told their naming failed.
#[derive(Default)]
struct Names {
    /// Each running job by chat, which the job alone holds strongly: a dead entry is a job that
    /// ended.
    jobs: HashMap<String, Weak<Naming>>,
    /// Told once per run: a failure that repeats (a template that refuses the request) is
    /// retried after each answer without a toast each time.
    told: HashSet<String>,
}

/// One chat's naming: its stop, and its request's canceller once sent, so a stop also ends a
/// prefill under way.
#[derive(Default)]
struct Naming {
    stop: AtomicBool,
    request: Mutex<Option<larch_engine::RequestCanceller>>,
}

impl Naming {
    /// Stops the job: its request now, the job at its next look.
    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(request) = lock(&self.request).as_ref() {
            request.cancel();
        }
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

#[derive(Default)]
struct State {
    loading: bool,
    /// Models loaded so far: the last one's number is the loaded engine's.
    loads: u64,
    model: Option<Loaded>,
    /// Chats with a session on the loaded engine; the rest live only in their transcripts.
    chats: HashMap<String, Chat>,
    /// Chats open as larch-agent agents; a chat is open in one of the two maps, never both.
    agents: HashMap<String, AgentChat>,
    /// Chats with a code run: None while it runs, then its outcome until `code_decide`. Kept
    /// across loads, since a parked outcome's worktree stays on disk until it is decided.
    codes: HashMap<String, Option<Outcome>>,
    /// Code runs waiting for the user's answer to a `CodeAsk`, by chat: `code_answer` sends it.
    asks: HashMap<String, mpsc::Sender<Decision>>,
    /// Each running code run's stop, by chat, which `cancel` sets (`Code::run_until`).
    code_stops: HashMap<String, Arc<AtomicBool>>,
    /// The real folder each chat's last code run changed in place, and the chat's title: it
    /// holds the folder while that run's entry stays in `codes` (`held`).
    in_place: HashMap<String, (PathBuf, String)>,
    /// Images `image_add` took for a message not sent yet, by reference, with the order they came
    /// in; `agent_send` takes them.
    pending_images: HashMap<String, (Arc<larch_model::vision::Image>, u64)>,
    pending_seq: u64,
}

/// The bytes of images waiting to be sent, in every chat together: past it the oldest go first
/// (a dropped draft's), never the image just added.
const PENDING_IMAGE_BYTES: usize = 256 << 20;
/// Why a message whose images take `rows` tokens cannot go to a model with a `ctx`-token context,
/// or None when it can: the images may fill at most three quarters of it, so the system prompt,
/// the words and the reply keep a quarter.
fn too_many_rows(rows: u64, ctx: u64) -> Option<String> {
    (ctx > 0 && rows * 4 > ctx * 3).then(|| {
        format!(
            "These images take about {rows} tokens, more than this model's {ctx}-token context \
             holds beside a reply. Send fewer or smaller images."
        )
    })
}

/// Removes from `pending` the additions a sent message took (`(reference, pending_seq)`): an image
/// added again since, under a newer sequence number, stays for the next message.
fn release_taken(
    pending: &mut HashMap<String, (Arc<larch_model::vision::Image>, u64)>,
    taken: &[(String, u64)],
) {
    for (r, seq) in taken {
        if pending.get(r).is_some_and(|p| p.1 == *seq) {
            pending.remove(r);
        }
    }
}

/// The largest image file `image_add` reads (larch_model::vision::MAX_BYTES), checked before any
/// copy of the request body.
const MAX_IMAGE_BYTES: usize = 32 << 20;

struct Loaded {
    larch: Larch,
    /// Which load this is (`State::loads`), so a change is undone only on the engine that made it.
    engine: u64,
    path: String,
    /// GGUF `general.architecture`, which picks the agent's tool-call format.
    arch: String,
    /// The chat template asks for the XML tool-call body.
    xml_body: bool,
    /// The engine's prefix cache keeps blocks.
    prefix_cache: bool,
    /// A vision projector beside the model loaded with it (`mmproj_beside`): chats take images.
    vision: bool,
    /// Why the projector beside the model did not load, when one is there and did not.
    vision_error: Option<String>,
    /// Encodes attached images ahead of their turn, with a vision model.
    prepare: Option<Preparer>,
    device_name: String,
    /// The model file's SHA-256, which names its probe (config.toml `router = "probe"`): hashed
    /// once, when the first chat with that router opens.
    sha256: OnceLock<Option<String>>,
    /// Stops this engine's telemetry thread.
    stop: Arc<AtomicBool>,
}

impl State {
    /// Closes every chat (their transcripts stay) and hands back the engine for the caller to drop;
    /// refused while a model loads, a reply is being written, or a stopped one or a chat's naming
    /// still holds the engine past `STOPPING`.
    fn unload(&mut self, names: &Mutex<Names>) -> Result<Option<Loaded>, String> {
        if self.loading {
            return Err("A model is already loading.".into());
        }
        if self.chats.values().any(|c| c.turn.is_some())
            || self.agents.values().any(|c| c.turn.is_some())
            || self.codes.values().any(Option::is_none)
        {
            return Err("Stop the replies being written first.".into());
        }
        // A stopped reply's reader holds the engine until the engine's pass under way ends.
        if !larch_agent::workers::wait_stopped(STOPPING) {
            return Err("The stopped replies are still ending: try again in a moment.".into());
        }
        // A naming job holds the engine too: its request is cancelled, and a chat it left
        // unnamed is named after its next answer.
        if !stop_naming(names, STOPPING) {
            return Err("The chats being named are still ending: try again in a moment.".into());
        }
        // A capture never outlives its engine: each closing agent joins its own, cancelled
        // here first, so no stream holds the old engine while the next one loads.
        memory::capture::cancel_all();
        self.chats.clear();
        self.agents.clear();
        Ok(self.model.take())
    }

    /// [`State::unload`], then the hosted task service stops, which holds a clone of the engine:
    /// the engine, and true when a service was hosted. A service that does not stop in time
    /// refuses it, and the engine stays loaded (its chats reopen from their transcripts).
    fn unload_hosted(
        &mut self,
        tasks: &mut Option<task::Hosted>,
        names: &Mutex<Names>,
    ) -> Result<(Option<Loaded>, bool), String> {
        let old = self.unload(names)?;
        match unhost(tasks) {
            Ok(hosted) => Ok((old, hosted)),
            Err(e) => {
                self.model = old;
                Err(e)
            }
        }
    }
}

/// Stops the task service in `tasks`; true when there was one. One that does not stop in time
/// stays there, with its engine.
fn unhost(tasks: &mut Option<task::Hosted>) -> Result<bool, String> {
    let Some(hosted) = tasks.as_mut() else {
        return Ok(false);
    };
    hosted.stop()?;
    *tasks = None;
    Ok(true)
}

impl Drop for Loaded {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Joined before the engine goes: no prepare holds it past an unload.
        self.prepare.take();
    }
}

/// Encodes images as they are attached (`Larch::prepare_image`, S5), one at a time on its own thread,
/// so pastes never decode in parallel (a 40 MP image takes most of a GB while it is cut). Dropping it
/// skips what still waits and joins the thread, so its clone of the engine is gone when an unload
/// returns (a new model then gets the whole device; Windows can delete the old model's file).
struct Preparer {
    tx: Option<std::sync::mpsc::SyncSender<Arc<larch_model::vision::Image>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Preparer {
    fn new(larch: Larch) -> Option<Preparer> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Arc<larch_model::vision::Image>>(
            larch::session::MAX_MESSAGE_IMAGES,
        );
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = std::thread::Builder::new()
            .name("larch-image-prepare".into())
            .spawn(move || {
                for image in rx {
                    if stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    // A failure only means the turn encodes it.
                    let _ = larch.prepare_image(&image);
                }
            })
            .ok()?;
        Some(Preparer {
            tx: Some(tx),
            stop,
            thread: Some(thread),
        })
    }

    /// Queues `image`; past a message's worth waiting it is left to its turn.
    fn send(&self, image: Arc<larch_model::vision::Image>) {
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(image);
        }
    }
}

impl Drop for Preparer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct AgentChat {
    agent: Arc<Mutex<Agent>>,
    /// Set while a turn runs; storing true stops it.
    turn: Option<Arc<AtomicBool>>,
    /// The history after the last turn, for `open_chat` and `list_chats` while a turn holds the agent.
    history: Vec<ChatMessage>,
    /// The project folder the agent's file tools start in, when the chat has one (`Inner::agent`).
    folder: Option<PathBuf>,
    /// The MCP hub's version when it opened: an idle chat at an older one reopens with the tools
    /// as they are now.
    mcp_version: u64,
    /// When its last turn ended, until [`PIN_IDLE`] later its pin is let go
    /// (`Inner::release_idle`); none before its first turn (a pin is only taken in one) and after
    /// that release.
    ended: Option<Instant>,
}

/// What an agent turn starts from: the user's text and the mode they picked, if any, or a confirm.
enum Step {
    Turn(String, Option<Intent>, Vec<Arc<larch_model::vision::Image>>),
    Confirm(bool),
}

struct Chat {
    session: Session,
    transcript: Transcript,
    /// Set while a reply is written; storing true stops it.
    turn: Option<Arc<AtomicBool>>,
}

/// GPU 0 when this build has GPUs (the v1 platforms, Windows CUDA and macOS Metal, have one),
/// else the CPU reference: a CPU-only build, or a GPU build on a machine without one, refuses GPU 0.
pub fn default_device() -> Result<i32, String> {
    let gpus = larch_kernels::device_count()
        .map_err(|e| format!("Cannot check GPUs: {e}: {}", larch_kernels::last_error()))?;
    Ok(if gpus > 0 { 0 } else { larch_kernels::CPU })
}

/// Why this machine's GPU goes unused, when `default_device` fell back to the CPU over one it has:
/// a driver older than the kernels' CUDA, or a GPU this build has no code for. Both count as no GPU
/// (`usable_devices` in kernels/src/cuda/backend.cu), so without this the CPU runs in silence.
pub fn unused_gpu() -> Option<String> {
    if larch_kernels::device_count().ok()? > 0 {
        return None;
    }
    match larch_kernels::Device::open(0) {
        // No driver at all: no GPU to miss.
        Ok(_) | Err(larch_kernels::KError::Driver { have: 0, .. }) => None,
        Err(e @ larch_kernels::KError::Driver { .. }) => Some(e.to_string()),
        Err(larch_kernels::KError::Unsupported) => Some(larch_kernels::last_error()),
        Err(_) => None,
    }
}

impl App {
    /// Opens the service's task database to save paused drafts and read committed records. It
    /// never starts, controls or answers a run: those go through [`App::task_service`], whose
    /// service applies its start checks.
    pub fn task_api(&self) -> Result<task::TaskApi, String> {
        runtime_dir(&self.inner.data)?;
        let path = larch_task::store::database_path(&self.inner.data);
        let store = larch_task::store::SqliteStore::open(path)?;
        Ok(task::TaskApi::new(store))
    }

    /// Connect to the workflow service: one already running on the data folder, or else one this
    /// app hosts in its own process on its loaded engine. Get a client for each use: a model load
    /// restarts the hosted service, and an older client's token ends with it.
    pub fn task_service(&self) -> Result<task::ServiceClient, String> {
        // Held while connecting: two first calls host one service, and a model swap, which holds
        // it, has no service start on the old engine or on none meanwhile.
        let mut tasks = self.inner.tasks();
        if let Err(stopped) = self.inner.ready().map(drop) {
            // Every step would fail on the faulted engine.
            unhost(&mut tasks)?;
            return Err(stopped);
        }
        // A task saved now may use every trusted MCP server that is on, and the web if allowed.
        let mut grants = self.inner.mcp.grants();
        if self.inner.web.load(Ordering::Relaxed) {
            grants.extend(WEB_GRANTS.map(String::from));
        }
        if tasks.is_none()
            && let Ok(client) = task::ServiceClient::connect(&self.inner.data)
        {
            return Ok(client.with_grants(grants));
        }
        self.host_tasks(&mut tasks)?;
        task::ServiceClient::connect(&self.inner.data).map(|c| c.with_grants(grants))
    }

    /// Task steps' tools as each step starts: trusted MCP servers', then the web's while the
    /// switch is on, so turning it off takes them from every later step.
    fn step_tools(&self) -> larch_agent::runtime::driver::StepToolSource {
        let (mcp, web) = (self.inner.mcp.step_tools(), self.inner.web.clone());
        Arc::new(move || {
            let mut tools = mcp();
            if web.load(Ordering::Relaxed) {
                use larch_agent::runtime::{driver::StepTool, effects};
                let [search, fetch] = WEB_GRANTS.map(String::from);
                tools.push(StepTool {
                    tool: effects::web_search_tool(),
                    capability: search,
                });
                tools.push(StepTool {
                    tool: effects::fetch_page_tool(),
                    capability: fetch,
                });
            }
            tools
        })
    }

    /// Turns "Let tasks use the web" on or off and keeps it in settings.json. Tasks saved while
    /// it was off never gain the web; turning it off takes it from every task's next step.
    pub fn set_tasks_web(&self, on: bool) -> Result<(), String> {
        let libs = self
            .inner
            .libraries
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let was = self.inner.web.swap(on, Ordering::Relaxed);
        self.inner.save_settings(&libs).inspect_err(|_| {
            self.inner.web.store(was, Ordering::Relaxed);
        })
    }

    /// Stops the task service this app hosts, which a quit does not wait for otherwise: a step
    /// running then is recovered and run again at the next start.
    pub fn stop_tasks(&self) -> Result<(), String> {
        unhost(&mut self.inner.tasks()).map(drop)
    }

    /// Hosts the task service on the loaded engine, or on none, unless the one this app hosts
    /// still runs: then it only waits for that one to answer.
    fn host_tasks(&self, tasks: &mut Option<task::Hosted>) -> Result<(), String> {
        if let Some(hosted) = tasks.as_mut().filter(|h| h.alive()) {
            return hosted.ready();
        }
        let larch = self.inner.ready()?.model.as_ref().map(|m| m.larch.clone());
        *tasks = None;
        tasks
            .insert(task::Hosted::start(
                &self.inner.data,
                larch,
                self.step_tools(),
            )?)
            .ready()
    }

    /// Hosts the task service again after a model change when `hosted`. Best effort: the change
    /// stands either way, and a service that did not start stays in `tasks`, so the next
    /// `task_service` restarts it and returns its error.
    fn rehost_tasks(&self, tasks: &mut Option<task::Hosted>, hosted: bool) {
        if hosted {
            let _ = self.host_tasks(tasks);
        }
    }

    /// Opens the data folder (`<data>/models`, `<data>/chat`, created if missing, never deleted) on
    /// `device` (a GPU ordinal or `larch_kernels::CPU`), with no model loaded.
    pub fn new(
        data: PathBuf,
        device: i32,
        emit: impl Fn(Event) + Send + Sync + 'static,
    ) -> Result<App, String> {
        for dir in ["models", "chat"] {
            std::fs::create_dir_all(data.join(dir))
                .map_err(|e| format!("cannot create {}: {e}", data.join(dir).display()))?;
        }
        // An added folder that is gone (an unplugged drive) stays listed, with nothing in it, until
        // it is back or removed. `modelsDir` is the single folder an earlier version kept.
        let file = fs::read(data.join(SETTINGS));
        let settings = (file.as_ref().ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .unwrap_or_default();
        let mut libraries = vec![data.join("models")];
        for d in settings["libraries"]
            .as_array()
            .into_iter()
            .flatten()
            .chain([&settings["modelsDir"]])
            .filter_map(|v| v.as_str().map(PathBuf::from))
        {
            if !libraries.contains(&d) {
                libraries.push(d);
            }
        }
        // A parked packet lives only in memory, so what an earlier run of the app left under
        // agent/code is nobody's: it goes, or the same request could not run again.
        let orphans = Mutex::new(sweep_orphans(&data));
        let emit: Emit = Arc::new(emit);
        let mcp = mcp::Hub::load(&data, emit.clone());
        let (idle_watch, idle_stop) = mpsc::channel();
        let app = App {
            inner: Arc::new(Inner {
                data,
                device,
                emit,
                state: Mutex::default(),
                hub: hf::Hub::new(),
                building: Mutex::default(),
                others: Mutex::new(other_model_dirs()),
                gpu: OnceLock::new(),
                libraries: Mutex::new(libraries),
                web: Arc::new(AtomicBool::new(tasks_web(&file))),
                code_mode: settings["code_mode"].as_bool().unwrap_or_default(),
                orphans,
                tasks: Mutex::default(),
                projector_errors: Mutex::default(),
                names: Mutex::default(),
                mcp,
                _idle_watch: idle_watch,
            }),
        };
        watch_idle_pins(Arc::downgrade(&app.inner), idle_stop)
            .map_err(|e| format!("cannot start the idle documents watch: {e}"))?;
        Ok(app)
    }

    /// Never fails, so the UI can always show why nothing else works.
    pub fn app_state(&self) -> AppState {
        let st = self.inner.lock();
        let health = match &st.model {
            _ if st.loading => Health::Loading,
            None => Health::Empty,
            Some(m) if m.larch.health().state == HealthState::Failed => Health::Failed,
            Some(_) => Health::Ready,
        };
        let gpu = self.inner.gpu();
        AppState {
            health,
            model: st.model.as_ref().map(|m| m.path.clone()),
            // Named by the engine once a model is loaded, else by the device itself.
            device: st.model.as_ref().map_or_else(
                || gpu.as_ref().map_or_else(String::new, |g| g.0.clone()),
                |m| m.device_name.clone(),
            ),
            data_dir: self.inner.data.display().to_string(),
            gpu_total_bytes: gpu.as_ref().map(|g| g.1),
            gpu_free_bytes: gpu.as_ref().map(|g| g.2),
            libraries: self.libraries(),
            hf_cache: self.inner.hf_cache().map(|d| d.display().to_string()),
            // A config.toml that does not read shows the default, which a chat's own load reports.
            docs_keep_ready: config::load(&self.inner.data).map_or(true, |c| c.docs_keep_ready),
            tasks_web: self.inner.web.load(Ordering::Relaxed),
            // A chat's relative paths start at its first attachment, which the UI has.
            agent_folder: None,
            code_mode: self.inner.coding_on(),
            vision: st.model.as_ref().is_some_and(|m| m.vision),
            vision_error: st.model.as_ref().and_then(|m| m.vision_error.clone()),
        }
    }

    /// The loaded engine's telemetry ring.
    pub fn ring(&self) -> Result<RingHandle, String> {
        Ok(self.inner.ready()?.larch()?.telemetry())
    }

    /// The loaded engine's seats, which every model caller in the process shares: a test's way
    /// to hold them.
    #[doc(hidden)]
    pub fn seats(&self) -> Result<SeatAdmission, String> {
        Ok(SeatAdmission::of(self.inner.ready()?.larch()?))
    }

    /// The GGUFs in Larch's libraries, then the GGUFs other apps on this machine downloaded
    /// (LM Studio, Ollama, the Hugging Face cache, llama.cpp, Jan, GPT4All), read in place and never
    /// copied or changed. Read from their headers: files that are not GGUF are left out; a GGUF larch
    /// cannot run is listed as unsupported. A file reached twice (a link) is listed once.
    pub fn list_models(&self) -> Result<Vec<ModelEntry>, String> {
        drop(self.inner.ready()?);
        let mut found = Vec::new();
        for dir in self.inner.libraries() {
            let mut paths = Vec::new();
            model_files(&dir, 0, &mut paths);
            found.extend(
                paths
                    .into_iter()
                    .map(|p| ("Larch".to_owned(), dir.clone(), p)),
            );
        }
        for (app, dir, depth) in self
            .inner
            .others
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
        {
            let mut paths = Vec::new();
            if app == "Ollama" {
                ollama_models(dir, &mut paths);
            } else {
                model_files(dir, *depth, &mut paths);
            }
            found.extend(paths.into_iter().map(|p| (app.clone(), dir.clone(), p)));
        }
        // A file reached twice (a link) is listed once, and so is a Hugging Face snapshot's copy
        // (where the cache could make no links) of a file another snapshot of the repo has at the
        // same path with the same size: two downloads of one model, not two models.
        let key = |p: &PathBuf| match snapshot_of(p) {
            Some(s) if !p.is_symlink() => (
                s.parent().unwrap_or(s).join(p.strip_prefix(s).unwrap_or(p)),
                fs::metadata(p).map_or(0, |m| m.len()),
            ),
            _ => (fs::canonicalize(p).unwrap_or_else(|_| p.clone()), 0),
        };
        let mut seen = HashSet::new();
        Ok(found
            .into_iter()
            .filter(|(_, _, p)| seen.insert(key(p)))
            .filter_map(|(app, dir, p)| {
                self.model_entry(p).map(|m| ModelEntry {
                    source: app,
                    library: dir.display().to_string(),
                    ..m
                })
            })
            .collect())
    }

    /// Deletes a model from one of Larch's libraries or from a snapshot of the Hugging Face cache
    /// (where Larch downloads, shared with other tools), every part of a split one. Files other
    /// apps keep are never touched; the loaded model is shut down first (refused while a reply is
    /// being written).
    pub fn delete_model(&self, path: &str) -> Result<(), String> {
        let file = Path::new(path);
        let parent = file.parent().and_then(|p| fs::canonicalize(p).ok());
        // `<cache>/models--<owner>--<name>/snapshots/<commit>/`, or a folder in it.
        let snapshot = || {
            let cache = fs::canonicalize(self.inner.hf_cache()?).ok()?;
            let inside = parent.as_ref()?.strip_prefix(cache).ok()?;
            let parts: Vec<_> = inside.components().collect();
            (parts.len() >= 3 && parts[1].as_os_str() == "snapshots")
                .then(|| file.parent().map(Path::to_owned))?
        };
        let dir = self
            .inner
            .libraries()
            .into_iter()
            .find(|d| parent.is_some() && fs::canonicalize(d).ok() == parent)
            .or_else(snapshot);
        // In a snapshot, a link whose blob is gone too: nothing else would ever remove it.
        let there = file.is_file() || (snapshot_of(file).is_some() && file.is_symlink());
        let (Some(dir), Some(name)) = (dir, file.file_name().filter(|_| there)) else {
            return Err(format!(
                "{path} is not in an Larch library: remove it in the app that downloaded it."
            ));
        };
        // The library's own spelling, so the paths below compare with the folder's listing.
        let name = name.to_string_lossy().into_owned();
        let file = dir.join(&name);
        // Its faster version goes with it: no library lists that copy, so nothing else could
        // remove it. (Another library's file of the same name shares it; its card makes it again.)
        let copy = self.inner.faster_path(&file);
        // The loaded model is this file (or its copy) only when both resolve: two missing files are
        // not the same.
        let in_use = |st: &State| {
            let loaded = st
                .model
                .as_ref()
                .and_then(|m| fs::canonicalize(&m.path).ok());
            loaded.is_some()
                && (loaded == fs::canonicalize(&file).ok()
                    || loaded == fs::canonicalize(&copy).ok())
        };
        // Held from the check to the delete: no task service starts on the engine being unloaded.
        let mut tasks = self.inner.tasks();
        let mut st = self.inner.lock();
        // A load under way may still be shutting down the old engine, which maps its file.
        if st.loading {
            return Err("A model is already loading.".into());
        }
        if in_use(&st) {
            // The engine maps the file, and Windows refuses to delete a mapped file: shut it down first.
            let (old, hosted) = st.unload_hosted(&mut tasks, &self.inner.names)?;
            drop(st);
            drop(old);
            (self.inner.emit)(Event::Health(Health::Empty));
            self.rehost_tasks(&mut tasks, hosted);
            st = self.inner.lock();
            // A load that started meanwhile may be mapping this very file.
            if st.loading {
                return Err("A model is already loading.".into());
            }
            if in_use(&st) {
                return Err(format!("{name} was loaded again meanwhile: try again."));
            }
        }
        // The files go with the state lock held, so no load starts on them meanwhile.
        // Its projector goes with it, unless another model of the folder pairs with it too: a stray
        // projector would pair with the next model that takes the folder alone.
        let projectors = projectors_named_for(&file);
        let mut doomed: Vec<PathBuf> = match hf::split_part(&name) {
            Some((stem, _, n)) => (1..=n)
                .map(|i| dir.join(format!("{stem}-{i:05}-of-{n:05}.gguf")))
                .collect(),
            None => vec![file.clone()],
        };
        self.inner.stop_build(&file);
        larch::faster::clean(&copy);
        doomed.push(copy.clone());
        for f in doomed {
            match remove_model_file(&f) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => {
                    return Err(format!("cannot delete {}: {e}", f.display()));
                }
                _ => {}
            }
        }
        // The model's files are gone: a lone generic projector now has no model beside it. In a
        // snapshot that means the whole snapshot, whose root projector its folders' models share.
        let mut others = Vec::new();
        match snapshot_of(&file) {
            Some(root) => model_files(root, 1, &mut others),
            None => model_files(&dir, 0, &mut others),
        }
        for projector in projectors {
            if others
                .iter()
                .any(|m| projectors_named_for(m).contains(&projector))
            {
                continue;
            }
            match remove_model_file(&projector) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => {
                    return Err(format!("cannot delete {}: {e}", projector.display()));
                }
                _ => {}
            }
        }
        drop(st);
        Ok(())
    }

    /// Larch's libraries, the default `<data>/models` first.
    pub fn libraries(&self) -> Vec<String> {
        self.inner
            .libraries()
            .iter()
            .map(|d| d.display().to_string())
            .collect()
    }

    /// Adds a folder as a library, after checking Larch can write there, and remembers it; returns
    /// the libraries. Models already in it are listed from then on.
    pub fn add_library(&self, path: &str) -> Result<Vec<String>, String> {
        let dir = PathBuf::from(path.trim());
        if !dir.is_absolute() {
            return Err(format!(
                "{} is not a full folder path, such as D:\\Models.",
                dir.display()
            ));
        }
        let probe = dir.join(".larch-write-check");
        fs::create_dir_all(&dir)
            .and_then(|()| fs::write(&probe, b""))
            .and_then(|()| fs::remove_file(&probe))
            .map_err(|e| format!("Larch cannot write to {}: {e}", dir.display()))?;
        let mut libs = self.inner.libraries();
        let same = |d: &PathBuf| fs::canonicalize(d).ok() == fs::canonicalize(&dir).ok();
        if libs.iter().any(same) {
            return Err(format!("{} is already a library.", dir.display()));
        }
        libs.push(dir);
        self.inner.save_libraries(libs)
    }

    /// Stops listing an added library; its files stay on disk. The default one stays, and none is
    /// removed while a download runs, since it may be writing there.
    pub fn remove_library(&self, path: &str) -> Result<Vec<String>, String> {
        let mut libs = self.inner.libraries();
        let i = libs
            .iter()
            .position(|d| d.as_os_str() == path)
            .ok_or_else(|| format!("{path} is not a library."))?;
        if i == 0 {
            return Err("Larch's own library stays.".into());
        }
        if self.list_downloads()?.iter().any(|d| d.state == "running") {
            return Err(
                "Wait for the downloads to finish, or cancel them, then remove the library.".into(),
            );
        }
        libs.remove(i);
        self.inner.save_libraries(libs)
    }

    /// Replaces the other apps' folders list_models reads (tests point it at a temporary folder).
    pub fn set_model_dirs(&self, dirs: Vec<(String, PathBuf, usize)>) {
        *self
            .inner
            .others
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = dirs;
    }

    fn model_entry(&self, path: PathBuf) -> Option<ModelEntry> {
        let file = File::open(&path).ok()?;
        let size_bytes = file.metadata().ok()?.len();
        let g = Gguf::read(BufReader::new(file), size_bytes).ok()?;
        let text = |key: &str| g.get(key).and_then(Value::as_str).map(str::to_owned);
        let supported = runnable(&g);
        let fit = match supported
            .then(|| fit_estimate(&LoadOptions::new(path.clone(), self.inner.device)))
        {
            Some(Ok(e)) => Fit::Fits {
                kv_bytes: e.kv_bytes,
                max_seqs: e.max_seqs_fit,
            },
            Some(Err(larch::LoadError::DoesNotFit { need, have })) => Fit::TooBig {
                need_bytes: need,
                have_bytes: have,
            },
            _ => Fit::Unknown,
        };
        let quant = quant_label(&g, &g.tensors);
        Some(ModelEntry {
            name: text("general.name").unwrap_or_else(|| {
                path.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into()
            }),
            arch: text("general.architecture").unwrap_or_default(),
            path: path.to_string_lossy().into(),
            quant,
            size_bytes,
            supported,
            fit,
            source: String::new(),
            library: String::new(),
            vision: mmproj_beside(&path).is_some()
                && !(self.inner.projector_errors.lock())
                    .unwrap_or_else(PoisonError::into_inner)
                    .contains_key(path.to_string_lossy().as_ref()),
            faster: (self.inner).faster_offer(&path, self.inner.gpu().as_ref().map(|g| g.1)),
        })
    }

    /// Loads `path` in place of the current model. Open chats are closed (their transcripts stay)
    /// and reopen on the new model; refused while a reply is being written.
    pub fn load_model(&self, path: &str) -> Result<(), String> {
        // Held through the swap: no task service starts on the old engine, or on none, meanwhile.
        let mut tasks = self.inner.tasks();
        let (old, hosted) = {
            let mut st = self.inner.ready()?;
            let (old, hosted) = st.unload_hosted(&mut tasks, &self.inner.names)?;
            st.loading = true;
            (old, hosted)
        };
        // The old engine shuts down before the new one takes the device, outside the state lock: its
        // prepare worker may be finishing an image, and other commands answer meanwhile.
        drop(old);
        let loaded = self.load_engine(path);
        // Saved schedules run without the Tasks page ever opened: once there are tasks, a loaded
        // model hosts the service.
        let saved = larch_task::store::database_path(&self.inner.data).exists();
        self.rehost_tasks(&mut tasks, hosted || (loaded.is_ok() && saved));
        loaded
    }

    /// Loads `path` once the old engine is gone and `loading` is set.
    fn load_engine(&self, path: &str) -> Result<(), String> {
        // Tentative until the load succeeds: a failed load releases it.
        let lease = match device_lease(&self.inner.data) {
            Ok(lease) => lease,
            Err(error) => {
                self.inner.lock().loading = false;
                (self.inner.emit)(Event::Health(Health::Empty));
                return Err(error);
            }
        };
        let emit = &self.inner.emit;
        emit(Event::Health(Health::Loading));
        let (arch, xml_body) = gguf_tools(Path::new(path));
        let mut opts = LoadOptions::new(path.into(), self.inner.device);
        // LoadOptions::new has read the variable already, and an unparsable one as 0.
        if !std::env::var("LARCH_PREFIX_CACHE_BLOCKS").is_ok_and(|v| v.parse::<usize>().is_ok()) {
            opts.prefix_cache_blocks = PREFIX_CACHE_BLOCKS;
        }
        // The engine gives a GGUF model's prefix cache these blocks.
        let prefix_cache = opts.prefix_cache_blocks > 0;
        // A vision projector beside the model loads with it (the engine makes room for a few
        // images); one that will not (a backend without the vision ops, a device without the room)
        // leaves the model to load alone, as it did before projectors were looked for, and the app
        // says why (`AppState::vision_error`).
        let plain = opts.clone();
        opts.mmproj = mmproj_beside(Path::new(path));
        let with_mmproj = opts.mmproj.is_some();
        let mut loaded = larch_agent::load_fitting(opts, Larch::load).map(|(larch, _)| larch);
        let mut vision_error = None;
        if with_mmproj && let Err(e) = &loaded {
            vision_error = Some(e.to_string());
            loaded = larch_agent::load_fitting(plain, Larch::load).map(|(larch, _)| larch);
        }
        {
            let mut failed =
                (self.inner.projector_errors.lock()).unwrap_or_else(PoisonError::into_inner);
            match &vision_error {
                Some(e) => failed.insert(path.into(), e.clone()),
                None => failed.remove(path),
            };
        }
        let mut st = self.inner.lock();
        st.loading = false;
        let larch = match loaded {
            Ok(larch) => larch,
            Err(e) => {
                drop(st);
                release_lease(lease);
                emit(Event::Health(Health::Empty));
                let file = Path::new(path)
                    .file_name()
                    .map_or(path.into(), |f| f.to_string_lossy());
                return Err(match e {
                    larch::LoadError::DoesNotFit { need, have } => format!(
                        "{file} needs {:.1} GB of GPU memory and {:.1} GB is free. Close what else uses the GPU, or pick a smaller quant.",
                        // GiB, as the app's "GB" everywhere else.
                        need as f64 / f64::from(1u32 << 30),
                        have as f64 / f64::from(1u32 << 30)
                    ),
                    e => format!("Cannot load {path}: {e}"),
                });
            }
        };
        keep_lease(lease);
        let ring = larch.telemetry();
        let mut reader = ring.reader(ReadFrom::Oldest);
        // The snapshot is written at load, before Ready.
        let device_name = loop {
            match reader.next() {
                Read::Record(Record::DeploymentSnapshot(_, d)) => {
                    break d.device_name.as_str().to_owned();
                }
                Read::Record(_) | Read::Lagged { .. } => {}
                Read::Empty => break String::new(),
            }
        };
        let stop = Arc::new(AtomicBool::new(false));
        st.loads += 1;
        let engine = st.loads;
        telemetry::spawn(ring, engine, emit.clone(), stop.clone());
        let vision = larch.sees_images();
        let prepare = vision.then(|| Preparer::new(larch.clone())).flatten();
        st.model = Some(Loaded {
            larch,
            engine,
            path: path.into(),
            arch,
            xml_body,
            prefix_cache,
            vision,
            vision_error,
            prepare,
            device_name,
            sha256: OnceLock::new(),
            stop,
        });
        drop(st);
        emit(Event::Health(Health::Ready));
        Ok(())
    }

    /// Every transcript in `<data>/chat`, newest first.
    /// A page of the official MCP Registry's servers (the MCP tools page's search).
    pub fn mcp_search(&self, query: &str, cursor: Option<&str>) -> Result<McpSearch, String> {
        mcp::search(query, cursor)
    }

    /// The installed MCP servers and their connections, secrets masked.
    pub fn mcp_servers(&self) -> Vec<McpServerStatus> {
        self.inner.mcp.list()
    }

    /// Installs an MCP server; on, it connects in the background (`mcp:status`). A chat takes the
    /// new tools at its next turn (`Inner::agent` reopens it).
    pub fn mcp_add(&self, server: McpServer) -> Result<(), String> {
        self.inner.mcp.add(server)
    }

    pub fn mcp_remove(&self, id: &str) -> Result<(), String> {
        self.inner.mcp.remove(id)
    }

    pub fn mcp_set_enabled(&self, id: &str, enabled: bool) -> Result<(), String> {
        self.inner.mcp.set_enabled(id, enabled)
    }

    /// Lets unattended workflow steps call a server's read tools, or stops them.
    pub fn mcp_set_trusted(&self, id: &str, trusted: bool) -> Result<(), String> {
        self.inner.mcp.set_trusted(id, trusted)
    }

    /// Connects to the server if it is not, and lists its tools.
    pub fn mcp_test(&self, id: &str) -> Result<Vec<McpToolInfo>, String> {
        self.inner.mcp.test(id)
    }

    /// Starts signing in to a remote MCP server: the page to open in the browser, and its host.
    pub fn mcp_sign_in(&self, id: &str) -> Result<McpSignIn, String> {
        self.inner.mcp.sign_in(id)
    }

    /// Finishes a waiting sign-in with the address the browser ended on.
    pub fn mcp_sign_in_finish(&self, id: &str, url: &str) -> Result<(), String> {
        self.inner.mcp.sign_in_finish(id, url)
    }

    pub fn mcp_sign_out(&self, id: &str) -> Result<(), String> {
        self.inner.mcp.sign_out(id)
    }

    /// Stops a waiting sign-in and keeps the sign-in the server already has.
    pub fn mcp_cancel_sign_in(&self, id: &str) -> Result<(), String> {
        self.inner.mcp.cancel_sign_in(id)
    }

    pub fn list_chats(&self) -> Result<Vec<ChatEntry>, String> {
        let st = self.inner.ready()?;
        let dir = self.inner.data.join("chat");
        let mut chats = Vec::new();
        for entry in
            std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().is_none_or(|x| x != "jsonl") {
                continue;
            }
            let id = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let (title, turns) = match st.history(&id, &path) {
                Ok(h) => {
                    // The loop's own notes (a retry's, the call cap's) are no turn: a chat of one
                    // message whose call was malformed listed "2 turns".
                    let users: Vec<_> = h.iter().filter(|m| larch_agent::is_turn(m)).collect();
                    // A first message of images alone has no words to name the chat.
                    let first = users.first().map(|m| match m.content.trim() {
                        "" if !m.images.is_empty() => "Image",
                        _ => m.content.as_str(),
                    });
                    // A compaction's summary carries the turns it sums up.
                    (shown_name(&path, first), larch_agent::turns(&h) as u32)
                }
                // Shown, not hidden: opening it gives the same error.
                Err(e) => (format!("Unreadable: {e:#}"), 0),
            };
            let updated_at = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_millis() as u64);
            chats.push(ChatEntry {
                id,
                title,
                updated_at,
                turns,
            });
        }
        chats.sort_by_key(|c| std::cmp::Reverse(c.updated_at));
        Ok(chats)
    }

    /// Creates and saves an empty chat; it opens on the engine at its first send.
    pub fn new_chat(&self, system: Option<String>) -> Result<String, String> {
        let history = system
            .iter()
            .map(|s| ChatMessage {
                role: larch::Role::System,
                content: s.clone(),
                ..Default::default()
            })
            .collect();
        let options = larch::SessionOptions {
            system,
            ..Default::default()
        };
        self.create_chat(options, history).map(|(id, _)| id)
    }

    /// Saves a chat of `history` under a fresh id; returns the id and its transcript's path.
    fn create_chat(
        &self,
        options: larch::SessionOptions,
        history: Vec<ChatMessage>,
    ) -> Result<(String, PathBuf), String> {
        drop(self.inner.ready()?);
        let id = format!(
            "{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos(),
            std::process::id()
        );
        let path = self.inner.path(&id)?;
        Transcript::create(&path, options, history).map_err(|e| format!("{e:#}"))?;
        Ok((id, path))
    }

    /// A new chat that picks up task `run_id`: its goal as the first message and its steps'
    /// results as the answer, so a follow-up question reads them. Named after the goal.
    pub fn chat_from_task(&self, run_id: &str) -> Result<String, String> {
        let service = self.task_service()?;
        let run = service.run(run_id)?.ok_or("unknown task")?;
        let names = service.step_instructions(run_id)?;
        let mut answer = String::from("Here is where this task got to.");
        let steps = run.graph.as_ref().or(run.definition.initial_graph.as_ref());
        // Joins and conditions only route; their results are their inputs again.
        let work = steps.iter().flat_map(|g| &g.nodes).filter(|node| {
            !matches!(
                node.kind,
                larch_task::graph::NodeKind::Join | larch_task::graph::NodeKind::Condition { .. }
            )
        });
        for (n, node) in work.enumerate() {
            let result = run.node_results.get(&node.id).map_or(
                if run.skipped_nodes.contains(&node.id) {
                    "Skipped."
                } else {
                    "Not finished."
                },
                String::as_str,
            );
            let name = names.get(&node.id).unwrap_or(&node.id);
            answer.push_str(&format!("\n\n**Step {}: {name}**\n\n{result}", n + 1));
        }
        if let Some(result) = &run.result {
            answer.push_str(&format!("\n\n**Outcome**\n\n{result}"));
        }
        let history = [
            (larch::Role::User, run.definition.goal.clone()),
            (larch::Role::Assistant, answer),
        ]
        .map(|(role, content)| ChatMessage {
            role,
            content,
            ..Default::default()
        });
        let (id, path) = self.create_chat(larch::SessionOptions::default(), history.into())?;
        save_name(&named_path(&path), &title(&run.definition.goal))?;
        Ok(id)
    }

    /// The chat's history; with a model loaded the chat also opens on it.
    pub fn open_chat(&self, id: &str) -> Result<Vec<Message>, String> {
        let mut st = self.inner.ready()?;
        let history = if let Some(a) = st.agents.get(id) {
            a.history.clone()
        } else if st.model.is_some() {
            self.inner.open(&mut st, id)?.session.history()
        } else {
            Transcript::resume(&self.inner.path(id)?)
                .map_err(|e| format!("{e:#}"))?
                .1
                .history
        };
        // Its newest images are encoded while the user reads (encode on paste, for a reopened
        // chat): the next turn reads their rows from the engine's cache instead of encoding them.
        // Read with the state lock released; the prepare worker takes them after.
        let warm = st.model.as_ref().is_some_and(|m| m.prepare.is_some());
        drop(st);
        if warm && let Ok(path) = self.inner.path(id) {
            let mut newest: Vec<&String> = Vec::new();
            for r in history.iter().rev().flat_map(|m| m.images.iter().rev()) {
                if newest.len() == WARM_IMAGES {
                    break;
                }
                if larch::session::image_ref(r) && !newest.contains(&r) {
                    newest.push(r);
                }
            }
            // Oldest first: the engine keeps the newest tiles when they are more than its cache.
            let images: Vec<Arc<larch_model::vision::Image>> = (newest.into_iter().rev())
                .filter_map(|r| {
                    let bytes = read_image(&larch::chat::images_dir(&path).join(r)).ok()?;
                    let image = larch_model::vision::Image::new(bytes.into()).ok()?;
                    (image.hex() == *r).then(|| Arc::new(image))
                })
                .collect();
            let st = self.inner.lock();
            if let Some(p) = st.model.as_ref().and_then(|m| m.prepare.as_ref()) {
                images.into_iter().for_each(|i| p.send(i));
            }
        }
        Ok(history.into_iter().map(message).collect())
    }

    /// Deletes the chat's transcript; refused while a reply is being written.
    pub fn delete_chat(&self, id: &str) -> Result<(), String> {
        let mut st = self.inner.ready()?;
        if st.busy(id) {
            return Err(BUSY.into());
        }
        // Its worktree would stay on disk with nothing left to decide it; a change made in
        // place is kept.
        if st.packet_waits(id) {
            return Err(PARKED.into());
        }
        st.codes.remove(id);
        // Closing unlocks the transcript first.
        st.chats.remove(id);
        st.agents.remove(id);
        let path = self.inner.path(id)?;
        // Before its title goes: a stopped job writes nothing (`Inner::name_chat`).
        {
            let mut names = lock(&self.inner.names);
            if let Some(job) = names.jobs.get(id).and_then(Weak::upgrade) {
                job.stop();
            }
            names.told.remove(id);
        }
        let attached = self.inner.attached(id)?;
        match fs::remove_file(attached_path(&path)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                return Err(format!("cannot delete what is attached to the chat: {e}"));
            }
            _ => {}
        }
        // With the list gone, `forget` keeps only what another chat still attaches.
        for file in &attached {
            larch_agent::folio::forget(&self.inner.data, file)
                .map_err(|e| format!("cannot delete what the chat read of its documents: {e}"))?;
        }
        larch_agent::folio::source::forget_chat(&self.inner.data, id)
            .map_err(|e| format!("cannot delete the chat's documents record: {e}"))?;
        match fs::remove_file(named_path(&path)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                return Err(format!("cannot delete the chat's title: {e}"));
            }
            _ => {}
        }
        match fs::remove_file(TaskRecord::path(&self.inner.data, id)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                return Err(format!("cannot delete the chat's code task: {e}"));
            }
            _ => {}
        }
        larch::chat::delete_images(&path)
            .map_err(|e| format!("cannot delete the chat's images: {e:#}"))?;
        std::fs::remove_file(&path).map_err(|e| format!("cannot delete {}: {e}", path.display()))
    }

    /// Names the chat `title` (trimmed), in place of the model's or the first message's name.
    pub fn rename_chat(&self, id: &str, title: &str) -> Result<(), String> {
        let title = title.trim();
        if title.is_empty() {
            return Err("A chat needs a name.".into());
        }
        let path = self.inner.path(id)?;
        // Stopped and written under the lock jobs start and write under: a running job would
        // write over the user's name, and one started before the write would see no name yet.
        let names = lock(&self.inner.names);
        if let Some(job) = names.jobs.get(id).and_then(Weak::upgrade) {
            job.stop();
        }
        save_name(&named_path(&path), title)
    }

    /// The folders and files attached to the chat, which its agent alone may read.
    pub fn attachments(&self, chat_id: &str) -> Result<Vec<String>, String> {
        Ok(shown(&self.inner.attached(chat_id)?))
    }

    /// Attaches the folder or file at `path` to the chat: its agent reopens on the same history
    /// at the next turn, with the file tools on that path. The line says it took.
    pub fn attach(&self, chat_id: &str, path: &str) -> Result<Attached, String> {
        let real = fs::canonicalize(path).map_err(|e| format!("cannot attach {path}: {e}"))?;
        // Counted first: a folder that cannot be listed is not attached.
        let ctx = larch_agent::tool::ToolCtx {
            allowed: vec![real.clone()],
            data_dir: self.inner.data.clone(),
            source: String::new(),
        };
        let at = real.to_str().unwrap_or(path);
        let n = larch_agent::skills::ask_docs::count(&ctx, at).map_err(|e| e.to_string())?;
        let mut list = self.inner.attached(chat_id)?;
        if !list.contains(&real) {
            list.push(real.clone());
        }
        self.inner.save_attached(chat_id, &list)?;
        let s = if n == 1 { "" } else { "s" };
        let line = format!("Attached {} ({n} file{s})", tilde(&real));
        Ok(Attached {
            attached: shown(&list),
            line,
        })
    }

    /// Takes `path` off the chat's attachments; its agent reopens without it at the next turn.
    pub fn detach(&self, chat_id: &str, path: &str) -> Result<Vec<String>, String> {
        let mut list = self.inner.attached(chat_id)?;
        // As `shown` gives it: a folder's trailing separator is no part of the path.
        list.retain(|p| p != Path::new(path));
        self.inner.save_attached(chat_id, &list)?;
        // After the list is saved, which `forget` reads to keep what another chat still holds.
        larch_agent::folio::forget(&self.inner.data, Path::new(path))
            .map_err(|e| format!("cannot forget what the chat read of {path}: {e}"))?;
        Ok(shown(&list))
    }

    /// Starts a plain reply, with no tool, on a foreground seat, which never waits; its text and
    /// end arrive as `chat:<id>` events. A cancel that lands before the reply starts is this
    /// call's cancelled error, with no event; one after it ends with `Done { finish: "cancelled" }`.
    pub fn send(&self, chat_id: &str, text: &str, params: SendParams) -> Result<(), String> {
        // Captures not yet started give way to the reply; its foreground seat never waits for
        // the seats background work holds.
        memory::capture::cancel_all();
        // The turn is reserved before its seat is taken, so Cancel and another send see it.
        let stop = Arc::new(AtomicBool::new(false));
        let (seats, engine) = {
            let mut st = self.inner.ready()?;
            if st.coding(chat_id) {
                return Err(BUSY.into());
            }
            let larch = st.larch()?;
            let (seats, engine) = (SeatAdmission::of(larch), larch.engine_id());
            let chat = self.inner.open(&mut st, chat_id)?;
            if chat.turn.is_some() {
                return Err(BUSY.into());
            }
            chat.turn = Some(stop.clone());
            (seats, engine)
        };
        let seat = seats.take(1, larch::api::Class::Foreground, &stop);
        let started = self.inner.ready().and_then(|mut st| {
            let seat = seat
                .map_err(engine_error)?
                .pop()
                .expect("take(1) gives one seat");
            // A guard: `State::unload` refuses while this reservation is held, so no load swaps
            // the engine in between.
            if st.larch()?.engine_id() != engine {
                return Err("The model changed before this reply started: send it again.".into());
            }
            let chat = st
                .chats
                .get_mut(chat_id)
                .filter(|c| c.turn.as_ref().is_some_and(|t| Arc::ptr_eq(t, &stop)))
                .filter(|_| !stop.load(Ordering::Acquire))
                .ok_or_else(|| engine_error(EngineError::Cancelled))?;
            let before = chat.session.history().len();
            let p = GenParams {
                max_tokens: params.max_tokens.unwrap_or(GenParams::default().max_tokens),
                temperature: params.temperature,
                ..GenParams::default()
            };
            let stream = seat
                .hold(chat.session.send(text, p))
                .map_err(engine_error)?;
            let (inner, id, stop) = (self.inner.clone(), chat_id.to_owned(), stop.clone());
            std::thread::Builder::new()
                .name("larch-chat".into())
                .spawn(move || inner.reply(id, stream, stop, before))
                .map(drop)
                .map_err(|e| e.to_string())
        });
        if started.is_err() {
            // Only this send's reservation: a later send may hold the chat's turn by now.
            if let Some(chat) = self.inner.lock().chats.get_mut(chat_id)
                && chat.turn.as_ref().is_some_and(|t| Arc::ptr_eq(t, &stop))
            {
                chat.turn = None;
            }
        }
        started
    }

    /// Stops the chat's reply; its text so far is kept, marked truncated. A code run stops at its
    /// next safe point (`Code::run_until`), a write it waits to ask about answered Stop, and ends
    /// at its packet as at any stop.
    pub fn cancel(&self, chat_id: &str) -> Result<(), String> {
        // `lock`, not `ready`: a code run is stopped after a GPU fault too.
        let mut st = self.inner.lock();
        if st.coding(chat_id) {
            if let Some(stop) = st.code_stops.get(chat_id) {
                stop.store(true, Ordering::Relaxed);
            }
            if let Some(asked) = st.asks.remove(chat_id) {
                // A run that ended meanwhile hears nothing.
                let _ = asked.send(Decision::Stop);
            }
            return Ok(());
        }
        // Still `lock`: a reply or turn under way when the GPU failed must hear its stop.
        let turns = [
            st.chats.get(chat_id).and_then(|c| c.turn.as_ref()),
            st.agents.get(chat_id).and_then(|c| c.turn.as_ref()),
        ];
        for stop in turns.into_iter().flatten() {
            stop.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Starts an agent turn: larch-agent's loop with its file and memory tools. Its text, tool
    /// calls and end arrive as `chat:<id>` events; a write stops the turn at `Confirm`. `intent`
    /// is the mode the user picked (the composer's chips, the Iris panel's modes); with none the
    /// turn has every tool and the model decides. A change starts a code run instead, where code
    /// mode is on (`AppState::code_mode`), on the chat's project folder `repo` or the checkout the
    /// text names, and a change with none is answered with a notice to pick one; a run
    /// (`Intent::Run`) is a chat turn, as any question is. Code is answered with no tool, chat
    /// with no web skill. A turn's file tools start in that folder (`Inner::agent`).
    pub fn agent_send(
        &self,
        chat_id: &str,
        text: &str,
        repo: Option<&Path>,
        intent: Option<Intent>,
        images: &[String],
    ) -> Result<(), String> {
        let (images, taken) = self.pending_images(images)?;
        let repo = (repo.map(Path::to_path_buf)).or_else(|| router::named_repo(text));
        // With code mode off the folder is only where the turn's file tools start: a change is a
        // turn with no tool, as code is.
        let coding = self.inner.coding_on();
        let sent = match (intent, repo) {
            // A code run reads files, not pictures: a message with images is a chat turn.
            (Some(Intent::Change), Some(repo)) if coding && images.is_empty() => {
                self.code_start(chat_id, text, &repo, None, 1, false)
            }
            // Code written into the chat would be one more copy to paste by hand. Kept out of
            // the history, so the answer before it is still the one a run drafts from.
            (Some(Intent::Change), None) if coding && images.is_empty() => {
                self.inner.path(chat_id)?;
                if self.inner.ready()?.busy(chat_id) {
                    return Err(BUSY.into());
                }
                self.code_end(chat_id, NO_FOLDER.into(), None)
            }
            (_, repo) => {
                let step = Step::Turn(text.into(), intent, images);
                self.agent_step(chat_id, step, repo.as_deref())
            }
        };
        // Sent, the images are the chat's: they no longer wait. Only the additions this send took
        // go: the same image added again while the send ran is a newer chip, still to send.
        if sent.is_ok() {
            release_taken(&mut self.inner.lock().pending_images, &taken);
        }
        sent
    }

    /// The images `refs` names that `image_add` took, in order, and which addition of each (its
    /// reference and `pending_seq`); an unknown one is an error (the UI adds it again). Images for
    /// a model that reads none are refused here, before any step.
    #[allow(clippy::type_complexity)]
    fn pending_images(
        &self,
        refs: &[String],
    ) -> Result<(Vec<Arc<larch_model::vision::Image>>, Vec<(String, u64)>), String> {
        if refs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if refs.len() > larch::session::MAX_MESSAGE_IMAGES {
            return Err(format!(
                "A message can carry at most {} images.",
                larch::session::MAX_MESSAGE_IMAGES
            ));
        }
        let st = self.inner.lock();
        if !st.model.as_ref().is_some_and(|m| m.vision) {
            let why = st.model.as_ref().and_then(|m| m.vision_error.as_deref());
            return Err(no_vision(why));
        }
        let (images, taken): (Vec<_>, Vec<_>) = (refs.iter())
            .map(|r| {
                (st.pending_images.get(r))
                    .map(|p| (p.0.clone(), (r.clone(), p.1)))
                    .ok_or_else(|| "An image of this message is gone; add it again.".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        // The newest message's images always go as pixels, and no overflow can shrink them: a
        // message whose images alone fill the context would only fail as the model reads it.
        if let Some(m) = st.model.as_ref() {
            let rows: u64 = (images.iter())
                .map(|i| u64::from(m.larch.image_rows(i).unwrap_or(0)))
                .sum();
            let ctx = u64::from(m.larch.deployment().ctx_max);
            if let Some(no) = too_many_rows(rows, ctx) {
                return Err(no);
            }
        }
        Ok((images, taken))
    }

    /// Takes an image for the next message (the UI calls this as it is pasted, dropped or picked,
    /// so a bad file is refused on its chip): checked and sized now, kept until `agent_send` names
    /// it by the reference returned.
    pub fn image_add(&self, bytes: &[u8]) -> Result<ImageInfo, String> {
        // Before any copy: a body past the limit is refused as it came.
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err("That image is larger than 32 MB.".into());
        }
        let image =
            Arc::new(larch_model::vision::Image::new(Arc::from(bytes)).map_err(|e| e.to_string())?);
        let mut st = self.inner.lock();
        let tokens = (st.model.as_ref()).and_then(|m| m.larch.image_rows(&image));
        // Encoded now, while the user types (S5): the send then reads its rows from the engine's
        // cache. On the model's prepare thread, so the chip shows at once.
        if let Some(p) = st.model.as_ref().and_then(|m| m.prepare.as_ref()) {
            p.send(image.clone());
        }
        let mut info = ImageInfo {
            reference: image.hex(),
            width: image.width(),
            height: image.height(),
            tokens,
            evicted: Vec::new(),
        };
        st.pending_seq += 1;
        let seq = st.pending_seq;
        st.pending_images
            .insert(info.reference.clone(), (image, seq));
        // Unsent drafts let go of their oldest images past the budget; the new one stays.
        let size = |st: &State| {
            st.pending_images
                .values()
                .map(|p| p.0.bytes().len())
                .sum::<usize>()
        };
        while size(&st) > PENDING_IMAGE_BYTES && st.pending_images.len() > 1 {
            let oldest = (st.pending_images.iter())
                .min_by_key(|(_, p)| p.1)
                .map(|(r, _)| r.clone())
                .expect("images");
            st.pending_images.remove(&oldest);
            info.evicted.push(oldest);
        }
        Ok(info)
    }

    /// Image tiles the loaded model's engine has encoded (`Larch::image_tiles_encoded`): attached,
    /// sent and reopened chats' images, a cache hit not counted. 0 with no model.
    pub fn image_tiles_encoded(&self) -> u64 {
        let st = self.inner.lock();
        st.model
            .as_ref()
            .map_or(0, |m| m.larch.image_tiles_encoded())
    }

    /// The bytes of an image the chat `chat_id` holds, by its reference: what a message's
    /// thumbnail shows. A reference is 64 hex digits and names a file in that chat's folder only;
    /// a file past [`MAX_IMAGE_BYTES`] (no image the app stored) is refused, not read.
    pub fn chat_image(&self, chat_id: &str, reference: &str) -> Result<Vec<u8>, String> {
        if !larch::session::image_ref(reference) {
            return Err("not an image reference".into());
        }
        let path = self.inner.path(chat_id)?;
        read_image(&larch::chat::images_dir(&path).join(reference))
    }

    /// Answers the write an agent turn stopped at: `yes` runs it; the turn then goes on.
    pub fn agent_confirm(&self, chat_id: &str, yes: bool) -> Result<(), String> {
        self.agent_step(chat_id, Step::Confirm(yes), None)
    }

    /// `step` on the chat's agent, on its project `folder` when the chat has one (`Inner::agent`).
    fn agent_step(&self, chat_id: &str, step: Step, folder: Option<&Path>) -> Result<(), String> {
        let mut st = self.inner.ready()?;
        if st.coding(chat_id) {
            return Err(BUSY.into());
        }
        let chat = self.inner.agent(&mut st, chat_id, folder)?;
        if chat.turn.is_some() {
            return Err(BUSY.into());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let (inner, id, agent) = (self.inner.clone(), chat_id.to_owned(), chat.agent.clone());
        let turn = stop.clone();
        std::thread::Builder::new()
            .name("larch-agent".into())
            .spawn(move || inner.agent_turn(id, agent, turn, step))
            .map_err(|e| e.to_string())?;
        // After the thread exists: a failed spawn must not leave the chat busy for good. The state
        // lock is held, so the turn cannot end before this.
        chat.turn = Some(stop);
        Ok(())
    }

    /// Starts a code run in the chat: larch-agent's code mode on `repo`, as `larch-agent code`
    /// runs it. It changes the folder in place, any folder, after a checkpoint commit on a branch
    /// of its own; each write the folder's allowances do not cover arrives as `CodeAsk` and waits
    /// for `code_answer`. `isolated`, or `k` above 1, runs k attempts in worktrees under
    /// `<data>/agent/code` instead, in the checkout of a repository's top. Its calls, results and
    /// notices arrive as `chat:<id>` events, then `Packet`, which `code_decide` answers; refused
    /// while a worktree's packet waits in the chat, and on an OS without the sandbox. A change
    /// made in place that still waits is kept first, as `code_decide` keeps it: the next run
    /// starts from it, and its checkpoint branch stays for an undo by hand. The folder is checked
    /// before the run starts; one whose tests neither `test_cmd` nor a manifest names runs
    /// untested, and the model's word ends it. One chat at a time changes a folder in place
    /// (`State::held`). The run is asked `request` within the chat's task
    /// (`record::request`); a chat's first run in place starts from the files its last answer
    /// drafted (`record::drafts`) that the folder lacks and allows (`CodeTask::drafts`).
    pub fn code_start(
        &self,
        chat_id: &str,
        request: &str,
        repo: &Path,
        test_cmd: Option<String>,
        k: u32,
        isolated: bool,
    ) -> Result<(), String> {
        self.start_code(chat_id, request, repo, test_cmd, k, isolated, false)
    }

    /// `code_start`, `read_only` for a run (`Intent::Run`): the run offers no tool that writes
    /// (`CodeTask::read_only`), writes no drafts, keeps no checkpoint branch, leaves the chat's
    /// code task as it was and ends with its answer and `Done`, no packet.
    #[allow(clippy::too_many_arguments)]
    fn start_code(
        &self,
        chat_id: &str,
        request: &str,
        repo: &Path,
        test_cmd: Option<String>,
        k: u32,
        isolated: bool,
        read_only: bool,
    ) -> Result<(), String> {
        if !SANDBOXED {
            return Err("Code mode is off in this release.".into());
        }
        if !self.inner.code_mode {
            return Err(CODE_MODE_OFF.into());
        }
        // The record's file is named after the chat: a name that is no chat's is refused first.
        self.inner.path(chat_id)?;
        // Started over it, the next run's outcome would replace this one, and its Undo with it.
        let in_place = |o: &Option<Outcome>| o.as_ref().is_some_and(|o| o.task.folder.is_some());
        if self.inner.lock().codes.get(chat_id).is_some_and(in_place) {
            // A notice, not an end: this message's reply goes on as the new run.
            self.decide(chat_id, true, false)?;
        }
        // A task in another folder is no task here; an unreadable record is no first run: its
        // drafts may be in the folder already.
        let here = real_folder(repo)?;
        let loaded = TaskRecord::load_in(&self.inner.data, chat_id, &here);
        let fresh = matches!(loaded, Ok(None));
        let record = loaded.unwrap_or_else(|e| {
            let why = format!("this run starts without the chat's code task, unreadable: {e}");
            self.notice(chat_id, why);
            None
        });
        let in_folder = !isolated && k <= 1;
        let folder = in_folder.then(|| here.clone());
        let (code, larch, named, drafts) = {
            let st = self.inner.ready()?;
            if st.busy(chat_id) {
                return Err(BUSY.into());
            }
            if st.packet_waits(chat_id) {
                return Err(PARKED.into());
            }
            // Before git makes a checkpoint in a folder another chat is changing.
            st.held(chat_id, folder.as_deref())?;
            // The chat list's title, or what the chat is about to be named: this request.
            let transcript = self.inner.path(chat_id)?;
            let history = st.history(chat_id, &transcript);
            let first = (history.iter().flatten()).find(|m| m.role == larch::Role::User);
            let named = shown_name(&transcript, Some(first.map_or(request, |m| &m.content)));
            let m = st.model.as_ref().ok_or(st.no_model())?;
            let config = config::load(&self.inner.data).map_err(|e| e.to_string())?;
            let code = Code::new(&m.arch, m.xml_body, &config, m.prefix_cache, Mode::Free)
                .map_err(|e| e.to_string())?;
            let drafts = match fresh && !read_only {
                false => Vec::new(),
                true => {
                    let history = history.map_err(|e| format!("{e:#}"))?;
                    let answer = history
                        .iter()
                        .rev()
                        .find(|m| m.role == larch::Role::Assistant);
                    answer.map_or_else(Vec::new, |m| record::drafts(&m.content))
                }
            };
            (code, m.larch.clone(), named, drafts)
        };
        // git on the checkout runs outside the state lock, as nothing here needs it.
        let overrides = Overrides {
            test_cmd,
            allowed_paths: None,
            check_cmds: vec![],
            budget: Budget {
                k,
                ..r#loop::BUDGET
            },
        };
        // A run answers this message alone: the task's objective and asks read as work to do.
        let asked = match (read_only, record.as_ref()) {
            (true, Some(r)) if !r.changed.is_empty() => format!(
                "{request}\n\nChanged earlier in this chat: {}",
                r.changed.join(", ")
            ),
            (true, _) => request.to_owned(),
            (false, r) => record::request(r, request),
        };
        let mut task = match in_folder {
            true => CodeTask::in_folder(&asked, repo, &self.inner.data, overrides),
            false => CodeTask::new(&asked, repo, overrides),
        }
        .map_err(|e| e.to_string())?;
        task.tag = Some(format!("{CHAT_TAG}{chat_id}"));
        task.drafts = drafts;
        task.read_only = read_only;
        let mut st = self.inner.ready()?;
        // Checked again: another command may have started a run while git ran.
        if st.busy(chat_id) {
            return Err(BUSY.into());
        }
        if st.packet_waits(chat_id) {
            return Err(PARKED.into());
        }
        st.held(chat_id, folder.as_deref())?;
        let (inner, id) = (self.inner.clone(), chat_id.to_owned());
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let words = request.to_owned();
        std::thread::Builder::new()
            .name("larch-code".into())
            .spawn(move || inner.code_run(id, larch, code, task, words, record, here, &stopped))
            .map_err(|e| e.to_string())?;
        st.codes.insert(chat_id.into(), None);
        st.code_stops.insert(chat_id.into(), stop);
        match folder {
            Some(folder) => st.in_place.insert(chat_id.into(), (folder, named)),
            None => st.in_place.remove(chat_id),
        };
        Ok(())
    }

    /// Answers the chat's packet: `yes` merges it into its checkout, no discards it; either way
    /// its worktree and branch go, then `Notice` and `Done` arrive as `chat:<id>` events. A merge
    /// git refuses leaves the packet waiting, so the user can set the checkout right and approve
    /// it again. Once merged, the packet is decided whatever its cleanup does: a worktree or
    /// branch that stays is a second notice, never a packet to answer again. A change made in
    /// place is in the folder already: `yes` keeps it, no undoes it (`CodeTask::undo`), which an
    /// undo refused leaves to answer again.
    pub fn code_decide(&self, chat_id: &str, yes: bool) -> Result<(), String> {
        self.decide(chat_id, yes, true)
    }

    /// `code_decide`; with `end` false, a change made in place is decided with a notice only,
    /// so the reply under way (`code_start`'s new run) isn't ended by a `Done`.
    fn decide(&self, chat_id: &str, yes: bool, end: bool) -> Result<(), String> {
        // `lock`, not `ready`: deciding runs only git, so a packet is decided after a GPU fault.
        let mut st = self.inner.lock();
        if !matches!(st.codes.get(chat_id), Some(Some(_))) {
            return Err("No packet waits in this chat.".into());
        }
        let mut outcome = (st.codes.insert(chat_id.into(), None))
            .flatten()
            .expect("checked above");
        // git runs outside the state lock: a merge, or removing a worktree the tests filled,
        // takes seconds. The chat reads as running meanwhile.
        drop(st);
        if outcome.task.folder.is_some() {
            let text = match yes {
                true => format!("kept in {}", outcome.task.repo.display()),
                false => match outcome.task.undo() {
                    Ok(put) if put.is_empty() => "Nothing to undo: the run changed no file.".into(),
                    Ok(put) => format!("undone; put back as it was: {}", put.join(", ")),
                    Err(e) => {
                        let mut st = self.inner.lock();
                        st.codes.insert(chat_id.into(), Some(outcome));
                        return Err(format!("Not undone: {e}"));
                    }
                },
            };
            if !yes {
                self.undone(chat_id);
            }
            let asked = if yes {
                "Keep the change."
            } else {
                "Undo the change."
            };
            // While the chat still reads as running, as in `code_run`.
            self.inner
                .keep_run(chat_id, &outcome.task.repo, asked, &text);
            self.inner.lock().codes.remove(chat_id);
            if !end {
                self.notice(chat_id, text);
                return Ok(());
            }
            return self.code_end(chat_id, text, None);
        }
        if yes {
            let approval = outcome.packet.approve(true).expect("a yes approves");
            if let Err(e) = outcome.task.apply(&outcome.packet, approval) {
                self.inner
                    .lock()
                    .codes
                    .insert(chat_id.into(), Some(outcome));
                return Err(format!("Not merged: {e}"));
            }
        }
        let left = outcome.task.discard().err();
        if !yes {
            self.undone(chat_id);
        }
        let repo = outcome.task.repo.display().to_string();
        let decided = match yes {
            true => format!("merged into {repo}"),
            false => format!("discarded; {repo} untouched"),
        };
        let asked = if yes {
            "Merge the change."
        } else {
            "Discard the change."
        };
        // While the chat still reads as running, as in `code_run`.
        self.inner
            .keep_run(chat_id, &outcome.task.repo, asked, &decided);
        self.inner.lock().codes.remove(chat_id);
        self.code_end(
            chat_id,
            decided,
            left.map(|e| format!("its worktree or branch stays: {e}")),
        )
    }

    /// Notes in the chat's task record that its last change is gone; a record that cannot say so
    /// is a notice.
    fn undone(&self, chat_id: &str) {
        let data = &self.inner.data;
        let noted = TaskRecord::load(data, chat_id).and_then(|r| match r {
            Some(mut r) => {
                r.undone();
                r.save(data, chat_id)
            }
            None => Ok(()),
        });
        if let Err(e) = noted {
            self.notice(
                chat_id,
                format!("the chat's code task still names the change: {e}"),
            );
        }
    }

    /// Answers the `CodeAsk` the chat's code run waits on; refused when none waits.
    pub fn code_answer(&self, chat_id: &str, decision: Decision) -> Result<(), String> {
        let asked = self.inner.lock().asks.remove(chat_id);
        let asked = asked.ok_or("No question waits in this chat.")?;
        // A run that ended meanwhile hears nothing, which is no answer to give.
        let _ = asked.send(decision);
        Ok(())
    }

    /// What code mode may change in the folder `repo` without asking.
    pub fn code_allowed(&self, repo: &Path) -> Result<Allowed, String> {
        allow::load(&self.inner.data, &real_folder(repo)?).map_err(|e| e.to_string())
    }

    /// Sets what code mode may change in the folder `repo` without asking: silence on or off, a
    /// reset to nothing (every write asks again); `code:allowed` tells the UI.
    pub fn code_allow(&self, repo: &Path, allowed: Allowed) -> Result<(), String> {
        let folder = real_folder(repo)?;
        allow::save(&self.inner.data, &folder, &allowed).map_err(|e| e.to_string())?;
        (self.inner.emit)(Event::Allowed(FolderAllowed {
            folder: folder.display().to_string(),
            allowed,
        }));
        Ok(())
    }

    /// Tells the chat `text`.
    fn notice(&self, chat_id: &str, text: String) {
        (self.inner.emit)(Event::Chat {
            id: chat_id.into(),
            event: ChatEvent::Notice { text },
        });
    }

    /// Ends a reply with no model step (a decided packet, a change with no folder): `text`, then
    /// what was left behind (`left`), then `Done`.
    fn code_end(&self, chat_id: &str, text: String, left: Option<String>) -> Result<(), String> {
        self.notice(chat_id, text);
        if let Some(left) = left {
            self.notice(chat_id, left);
        }
        (self.inner.emit)(Event::Chat {
            id: chat_id.into(),
            event: eos(),
        });
        Ok(())
    }

    /// The remembered facts, in `MEMORY.md` order (`memory::list`).
    pub fn memory_list(&self) -> Result<Vec<Fact>, String> {
        let facts =
            memory::list(&self.inner.data).map_err(|e| format!("cannot read your memory: {e}"))?;
        let dir = self.inner.data.join("agent/memory/facts");
        Ok(facts
            .into_iter()
            .map(|f| Fact {
                updated_at: fs::metadata(dir.join(format!("{}.md", f.slug)))
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_millis() as u64),
                slug: f.slug,
                text: f.text,
                source: f.source,
            })
            .collect())
    }

    /// Remembers `text` as the user's fact. A near-duplicate is updated instead, and the receipt
    /// says what it replaced (`old`), so the UI can say so and Undo puts it back.
    pub fn memory_add(&self, text: &str) -> Result<memory::Receipt, String> {
        memory::remember(&self.inner.data, text, "user", "the desktop app")
            .map(|saved| saved.receipt(text))
            .map_err(|e| e.to_string())
    }

    /// Puts `text` in place of the fact `slug`'s sentence, keeping the user's words a capture
    /// saved after it; open chats show it from their next turn. Refused when the fact is no
    /// longer `expected`, the text the edit started from. Whether it changed anything: the same
    /// sentence writes nothing.
    pub fn memory_update(&self, slug: &str, expected: &str, text: &str) -> Result<bool, String> {
        memory::edit(&self.inner.data, slug, expected, text).map_err(|e| e.to_string())
    }

    /// Forgets the fact `slug`, returning it for an Undo (`memory_restore`).
    pub fn memory_forget(&self, slug: &str) -> Result<memory::Gone, String> {
        memory::forget_kept(&self.inner.data, slug).map_err(|e| e.to_string())
    }

    /// Puts a forgotten fact back as it was: its slug, type, source and `created`.
    pub fn memory_restore(&self, gone: &memory::Gone) -> Result<(), String> {
        memory::restore(&self.inner.data, gone).map_err(|e| e.to_string())
    }

    /// Undoes the save `receipt` names; refused, and nothing changed, when the fact was edited,
    /// saved again or forgotten since.
    pub fn memory_undo(&self, receipt: &memory::Receipt) -> Result<(), String> {
        memory::undo(&self.inner.data, receipt).map_err(|e| e.to_string())
    }

    /// Whether memory is on.
    pub fn memory_enabled(&self) -> bool {
        memory::enabled(&self.inner.data)
    }

    /// Turns memory on or off: off, chats show no memory, offer no memory tools and save
    /// nothing, from each open chat's next turn.
    pub fn memory_set_enabled(&self, on: bool) -> Result<(), String> {
        memory::set_enabled(&self.inner.data, on).map_err(|e| e.to_string())
    }

    /// Sets `docs_keep_ready` in config.toml (a pinned document is exported to disk and read back
    /// after a restart), from the next chat turn; off deletes the pin files already kept.
    pub fn set_docs_keep_ready(&self, on: bool) -> Result<(), String> {
        let path = config::path(&self.inner.data);
        let old = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
        };
        let dir = path.parent().unwrap_or(&self.inner.data);
        fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
        let new = with_docs_keep_ready(&old, on);
        write_synced(&path, &path.with_extension("tmp"), new.as_bytes())?;
        // Off frees the disk the kept documents took (#590 F10); a chat reads them again.
        if on {
            return Ok(());
        }
        let cache = larch_agent::folio::cache::dir(&self.inner.data);
        match larch_agent::folio::cache::evict(&cache, 0) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(format!(
                "cannot delete the kept documents in {}: {e}",
                cache.display()
            )),
            _ => Ok(()),
        }
    }

    /// The chat's documents as its next question reads them (from the agent's extract cache), for
    /// checking an answer's cites and showing the page one names. Empty when the chat has none.
    pub fn documents(&self, chat_id: &str) -> Result<Vec<larch_agent::folio::Doc>, String> {
        let ctx = larch_agent::tool::ToolCtx {
            allowed: self.inner.attached(chat_id)?,
            data_dir: self.inner.data.clone(),
            source: String::new(),
        };
        let (docs, _) = larch_agent::folio::load(&ctx, chat_id, &AtomicBool::new(false))
            .map_err(|e| e.to_string())?;
        Ok(docs.unwrap_or_default())
    }

    /// The knob registry with the values in force.
    pub fn knobs(&self) -> Result<Vec<Knob>, String> {
        let cfg = self.inner.ready()?.larch()?.knobs().effective();
        Ok(REGISTRY
            .iter()
            .map(|k| Knob {
                id: k.id,
                name: k.name.as_str().into(),
                value: cfg.values[k.id as usize],
                min: k.min,
                max: k.max,
                default: k.default,
                activation: camel(k.activation()),
            })
            .collect())
    }

    /// Sets a knob as a human; the engine's own answer comes back.
    pub fn knob_set(&self, name: &str, value: f64) -> Result<KnobResult, String> {
        let spec = larch::knob::spec(name).map_err(|e| e.to_string())?;
        let knobs = self.inner.ready()?.larch()?.knobs();
        Ok(knob_result(knobs.set(KnobRequest::new(
            spec.id,
            value,
            Authority::Human,
        ))))
    }

    /// Reverts revision `rev` of engine `engine` (its `ConfigChange`) to its declared inverse;
    /// rejected once another engine is loaded, whose revisions reuse the numbers.
    pub fn knob_revert(&self, engine: u64, rev: u64) -> Result<KnobResult, String> {
        // Checked under the lock that keeps this engine loaded until the revert is done.
        let st = self.inner.ready()?;
        if st.model.as_ref().map(|m| m.engine) != Some(engine) {
            return Ok(KnobResult::Rejected {
                reason: "that change was made on a model that is no longer loaded".into(),
            });
        }
        let knobs = st.larch()?.knobs();
        Ok(knob_result(
            knobs.revert(RevertTarget::Rev(rev), Authority::Human),
        ))
    }
}

impl Inner {
    fn libraries(&self) -> Vec<PathBuf> {
        self.libraries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The library named `path`, or the default one for None.
    fn library(&self, path: Option<&str>) -> Result<PathBuf, String> {
        let libs = self.libraries();
        match path {
            None => Ok(libs[0].clone()),
            Some(p) => libs
                .into_iter()
                .find(|d| d.as_os_str() == p)
                .ok_or_else(|| format!("{p} is not a library.")),
        }
    }

    /// Writes the added libraries to settings.json, then uses them.
    fn save_libraries(&self, libs: Vec<PathBuf>) -> Result<Vec<String>, String> {
        let mut held = self
            .libraries
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.save_settings(&libs)?;
        let shown = libs.iter().map(|d| d.display().to_string()).collect();
        *held = libs;
        Ok(shown)
    }

    /// Writes settings.json: the libraries after the default one, "Let tasks use the web", and the
    /// code mode setting as it was read. Callers hold the libraries lock, so a
    /// library change and a switch never write over each other or share the temporary file. It is
    /// synced, then renamed over settings.json, so a crash leaves the old settings, never a torn file
    /// that `App::new` reads as none.
    fn save_settings(&self, libs: &[PathBuf]) -> Result<(), String> {
        let settings = self.data.join(SETTINGS);
        let temp = self.data.join(format!("{SETTINGS}.tmp"));
        let added: Vec<_> = libs[1..].iter().map(|d| d.display().to_string()).collect();
        let tasks_web = self.web.load(Ordering::Relaxed);
        let mut json = serde_json::json!({ "libraries": added, "tasks_web": tasks_web });
        if self.code_mode {
            json["code_mode"] = true.into();
        }
        write_synced(&settings, &temp, json.to_string().as_bytes())
    }

    /// Whether code mode is on: the setting, on an OS with the sandbox.
    fn coding_on(&self) -> bool {
        SANDBOXED && self.code_mode
    }

    fn tasks(&self) -> MutexGuard<'_, Option<task::Hosted>> {
        self.tasks.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The GPU's name, total and free memory, read once ([`Inner::gpu`]'s field).
    fn gpu(&self) -> &Option<(String, u64, u64)> {
        let device = self.device;
        self.gpu.get_or_init(|| {
            let p = (device != larch_kernels::CPU).then(|| {
                larch_kernels::Device::open(device)
                    .and_then(|d| d.props())
                    .ok()
            })??;
            Some((p.name, p.vram_total, p.vram_free))
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panicked command must not take every later one with it.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The state, unless the engine failed (9.11): then only a restart helps.
    fn ready(&self) -> Result<MutexGuard<'_, State>, String> {
        let st = self.lock();
        match &st.model {
            Some(m) if m.larch.health().state == HealthState::Failed => Err(STOPPED.into()),
            _ => Ok(st),
        }
    }

    fn path(&self, id: &str) -> Result<PathBuf, String> {
        transcript_path(&self.data, id).map_err(|e| e.to_string())
    }

    /// What is attached to the chat `id`: `chat/<id>.attached.json`, beside its transcript, which
    /// the engine's journal format has no field for. None for a new chat.
    fn attached(&self, id: &str) -> Result<Vec<PathBuf>, String> {
        let path = attached_path(&self.path(id)?);
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| format!("cannot read {}: {e}", path.display())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("cannot read {}: {e}", path.display())),
        }
    }

    /// Saves the chat's attachments, and closes its agent, which reopens with them at its next
    /// turn; refused while a reply is being written in it.
    fn save_attached(&self, id: &str, list: &[PathBuf]) -> Result<(), String> {
        let mut st = self.ready()?;
        if st.agents.get(id).is_some_and(|c| c.turn.is_some()) {
            return Err(BUSY.into());
        }
        let path = attached_path(&self.path(id)?);
        let json = serde_json::to_vec(list).map_err(|e| e.to_string())?;
        // Every send and open reads it: a torn file would refuse the chat until deleted by hand.
        write_synced(&path, &path.with_extension("json.tmp"), &json)?;
        st.agents.remove(id);
        Ok(())
    }

    /// The chat on the loaded engine, resumed from its transcript if it is not open yet.
    fn open<'a>(&self, st: &'a mut State, id: &str) -> Result<&'a mut Chat, String> {
        if !st.chats.contains_key(id) {
            if st.agents.get(id).is_some_and(|c| c.turn.is_some()) {
                return Err(BUSY.into());
            }
            st.agents.remove(id); // its transcript lock goes with it
            let larch = st.larch()?;
            let (transcript, snapshot) =
                Transcript::resume(&self.path(id)?).map_err(|e| format!("{e:#}"))?;
            let mut session = larch.session(snapshot.options.clone());
            session
                .restore(snapshot.options, snapshot.history)
                .map_err(engine_error)?;
            st.chats.insert(
                id.into(),
                Chat {
                    session,
                    transcript,
                    turn: None,
                },
            );
        }
        Ok(st.chats.get_mut(id).expect("inserted"))
    }

    /// The chat as an agent on the loaded engine, reopened from its transcript if it is not open
    /// yet. A reopened chat that ends at an unanswered write says so with `Confirm`. With the
    /// chat's project `folder`, the file tools start there: a relative path is the folder's, and
    /// the system prompt lists its files before the attachments'. An agent open on another folder
    /// is reopened on this one, its history kept; None keeps the agent that is open.
    fn agent<'a>(
        &self,
        st: &'a mut State,
        id: &str,
        folder: Option<&Path>,
    ) -> Result<&'a mut AgentChat, String> {
        let open = st.agents.get(id);
        if let (Some(a), Some(folder)) = (open, folder)
            && a.folder.as_deref() != Some(folder)
        {
            // Another holder (a turn ending, `keep_run`) would keep its transcript claimed, and
            // the reopen below would fail on it.
            if a.turn.is_some() || Arc::strong_count(&a.agent) > 1 {
                return Err(BUSY.into());
            }
            st.agents.remove(id); // its transcript lock goes with it
        }
        // MCP servers changed (connected, signed in, turned off) since it opened: an idle chat
        // reopens with their tools as they are now, on the folder it had. Not one parked at a
        // write: reopened, its call was prepared again (a path resolved anew, the folder gone for
        // a confirm, which names none) and Allow ran a call the card did not show. Nothing else
        // holds the agent at a strong count of 1, so its lock is free.
        let version = self.mcp.version();
        let mut kept = None;
        if st.agents.get(id).is_some_and(|a| {
            a.mcp_version != version
                && a.turn.is_none()
                && Arc::strong_count(&a.agent) == 1
                && a.agent.try_lock().is_ok_and(|g| g.parked().is_none())
        }) {
            kept = st.agents.remove(id).and_then(|a| a.folder);
        }
        let folder = folder.or(kept.as_deref());
        if !st.agents.contains_key(id) {
            if st.chats.get(id).is_some_and(|c| c.turn.is_some()) {
                return Err(BUSY.into());
            }
            st.chats.remove(id); // its transcript lock goes with it
            let m = st.model.as_ref().ok_or(st.no_model())?;
            // The chat's project folder, then what the user attached to it, and nothing else:
            // with neither, it offers no file tools.
            let mut allowed: Vec<PathBuf> = folder.map(Path::to_path_buf).into_iter().collect();
            let attached = self.attached(id)?;
            allowed.extend(attached.into_iter().filter(|a| Some(a.as_path()) != folder));
            let mut opts = AgentOptions::app(self.data.clone(), allowed, m.arch.clone());
            // On only while the engine's cache keeps blocks, as in the CLI.
            opts.prefix_cache &=
                (m.larch.prefix_cache_stats()).is_some_and(|s| s.capacity_blocks > 0);
            let outside = self.mcp.chat_tools();
            let mut agent = Agent::open_attached_with(&m.larch, id, opts, outside)
                .map_err(|e| e.to_string())?;
            // A small model stuck repeating one short piece, or going round whole paragraphs,
            // ends with a notice, not at the cap.
            agent.stop_loops(true);
            // Each fact the chat saves unasked, its captures after a turn too, reaches the UI
            // with its receipt, for Undo. The thread ends when the agent and its captures do.
            let to = forward_receipts(Arc::clone(&self.emit), id).map_err(|e| e.to_string())?;
            agent.set_memory_receipts(to);
            // Qwen 3.5's template writes the XML call body: a grammar held to JSON had it write
            // "</function>" past its call until the token cap (90-100 s a tool step, 5 Nov 2026).
            agent.set_xml_body(m.xml_body);
            // The probe router finds the model's probe by the file's hash (toolcall-opt item 2):
            // without it every turn fell back, as the app never named the model.
            if agent.wants_model_sha256() {
                let sha = m
                    .sha256
                    .get_or_init(|| larch_model::oracle::sha256(Path::new(&m.path)).ok());
                agent.set_model_sha256(sha.clone());
            }
            if let Some(call) = agent.parked() {
                (self.emit)(Event::Chat {
                    id: id.into(),
                    event: ChatEvent::Confirm {
                        name: call.name.clone(),
                        args: call.args.clone(),
                    },
                });
            }
            let history = agent.history();
            st.agents.insert(
                id.into(),
                AgentChat {
                    agent: Arc::new(Mutex::new(agent)),
                    turn: None,
                    history,
                    folder: folder.map(Path::to_path_buf),
                    mcp_version: version,
                    ended: None,
                },
            );
        }
        Ok(st.agents.get_mut(id).expect("inserted"))
    }

    /// Lets go of every other chat's pin, as the engine's pool is full.
    fn unpin_others(&self, keep: &str) {
        let others: Vec<String> = (self.lock().agents.keys())
            .filter(|c| c.as_str() != keep)
            .cloned()
            .collect();
        for c in &others {
            self.release(c);
        }
    }

    /// Lets go of the pin of each chat whose last turn ended [`PIN_IDLE`] or more before `now`
    /// (the watch's clock; a test's later one), once its file holds it whole: one no file holds
    /// ("Keep documents ready" off, no room on the volume) stays, as the next question would read
    /// the set again (#591 review). A chat answering now, or whose agent something else holds,
    /// keeps it until the next look; any other is looked at once a pause. The app's lock is held
    /// throughout, so an attach or a folder change never finds the agent held by this (#591
    /// review): the export of a pin already written is a few engine steps, which wait out a
    /// reload the engine is running for another chat.
    fn release_idle(&self, now: Instant) {
        let mut st = self.lock();
        for (id, chat) in &mut st.agents {
            let due = chat.turn.is_none()
                && (chat.ended).is_some_and(|e| now.saturating_duration_since(e) >= PIN_IDLE);
            if !due || Arc::strong_count(&chat.agent) > 1 {
                continue;
            }
            let Ok(mut a) = chat.agent.try_lock() else {
                continue;
            };
            self.release_agent(id, &mut a, true);
            drop(a);
            chat.ended = None;
        }
    }

    /// Exports chat `id`'s pin and drops it; a chat that is answering keeps it. False when it
    /// did not run: the chat is gone or its agent is busy.
    fn release(&self, id: &str) -> bool {
        let agent = self.lock().agents.get(id).map(|c| c.agent.clone());
        let Some(agent) = agent else {
            return false;
        };
        let Ok(mut a) = agent.try_lock() else {
            return false;
        };
        self.release_agent(id, &mut a, false);
        true
    }

    /// `Agent::release_documents` for chat `id`, with its notices forwarded, and `Released` sent
    /// when a pin went, so the chip stops saying the documents are ready (#591 review).
    fn release_agent(&self, id: &str, a: &mut Agent, covered_only: bool) {
        let stop = AtomicBool::new(false);
        let send = |event| {
            (self.emit)(Event::Chat {
                id: id.into(),
                event,
            })
        };
        let released = a.release_documents(
            &stop,
            &mut |e| {
                if let AgentEvent::Notice(text) = e {
                    send(ChatEvent::Notice { text });
                }
            },
            covered_only,
        );
        use larch_agent::folio::held::Released as R;
        if matches!(released, R::Covered | R::Uncovered) {
            send(ChatEvent::Documents {
                state: DocState::Released,
                text: "Released; the next question loads it again".into(),
            });
        }
    }

    /// Runs one agent turn on its own thread: forwards its events and its helpers, then sends its
    /// end (`Done`, `Confirm` or `Error`). The agent saves its transcript after every step itself.
    fn agent_turn(&self, id: String, agent: Arc<Mutex<Agent>>, stop: Arc<AtomicBool>, step: Step) {
        let send = |event| {
            (self.emit)(Event::Chat {
                id: id.clone(),
                event,
            })
        };
        let mut a = agent.lock().unwrap_or_else(PoisonError::into_inner);
        // A new question reads the chat's documents (small ones as text, medium ones pinned; other
        // chats keep theirs, which give way when this one needs the room) and tells their state.
        if let Step::Turn(text, ..) = &step {
            let links = larch_agent::folio::links(text);
            let told = |e| match e {
                AgentEvent::Documents(d) => send(d.into()),
                AgentEvent::Notice(text) => send(ChatEvent::Notice { text }),
                _ => {}
            };
            let read = a.read_documents(&links, &stop, &mut |e| told(e), &mut || {
                self.unpin_others(&id)
            });
            if let Err(e) = read {
                send(ChatEvent::Notice {
                    text: e.to_string(),
                });
            }
        }
        // Read between events, as the turn holds the agent; sent when they change. The agent
        // empties the board as a turn starts, so none of an earlier turn's is sent as this one's.
        let board = a.helpers();
        let mut told = Vec::new();
        let mut tell_helpers = || {
            let now = board.lock().unwrap_or_else(PoisonError::into_inner).clone();
            if now != told {
                send(ChatEvent::Helpers {
                    helpers: now.iter().map(chat_helper).collect(),
                });
                told = now;
            }
        };
        let mut end = None;
        // A turn that fails before answering says why in its last notice.
        let mut reason = None;
        // The turn borrows the agent; it ends (or is dropped, cancelling) before the history is read.
        {
            let started = match step {
                // Notes are written lazily, before the turn whose prompt needs one (the agent).
                Step::Turn(text, intent, images) if !images.is_empty() => {
                    a.turn_images(&text, images, intent)
                }
                Step::Turn(text, Some(intent), _) => a.turn_for(&text, intent),
                Step::Turn(text, None, _) => a.turn(&text),
                Step::Confirm(yes) => a.confirm(yes),
            };
            match started {
                Err(e) => {
                    end = Some(ChatEvent::Error {
                        message: e.to_string(),
                    })
                }
                // Dropping the turn mid-answer cancels it, keeping the text so far.
                Ok(turn) => {
                    for event in turn {
                        tell_helpers();
                        match event {
                            AgentEvent::Text(text) if !text.is_empty() => {
                                send(ChatEvent::Delta { text })
                            }
                            AgentEvent::Text(_) => {}
                            AgentEvent::ToolCall { name, args } => {
                                send(ChatEvent::ToolCall { name, args })
                            }
                            AgentEvent::ToolResult { name, summary } => {
                                send(ChatEvent::ToolResult { name, summary })
                            }
                            AgentEvent::Notice(text) => {
                                reason = Some(text.clone());
                                send(ChatEvent::Notice { text })
                            }
                            AgentEvent::Progress(text) => send(ChatEvent::Progress { text }),
                            AgentEvent::Thinking(text) => send(ChatEvent::Thinking { text }),
                            AgentEvent::Documents(d) => send(d.into()),
                            AgentEvent::NeedsConfirm { call } => {
                                end = Some(ChatEvent::Confirm {
                                    name: call.name,
                                    args: call.args,
                                })
                            }
                            // The agent reports timings, not token counts, for a turn.
                            AgentEvent::Done(t) => {
                                end = Some(ChatEvent::Done {
                                    finish: "eos".into(),
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                    cached_tokens: t.cached_tokens,
                                    ttft_ms: t.ttft_ms,
                                    tok_s: t.tok_s,
                                    dropped_exchanges: t.dropped_exchanges,
                                })
                            }
                        }
                        if stop.load(Ordering::Relaxed) {
                            end = Some(ChatEvent::Done {
                                finish: "cancelled".into(),
                                prompt_tokens: 0,
                                completion_tokens: 0,
                                cached_tokens: 0,
                                ttft_ms: 0.0,
                                tok_s: 0.0,
                                dropped_exchanges: 0,
                            });
                            break;
                        }
                    }
                }
            }
        }
        // A Stop seen just as the turn parked a write (its call, then the confirm, came in one
        // step) ended it with no card while the write still waited: every later message failed
        // "the turn waits for a confirm". Its card is the end instead; nothing runs without Allow.
        if matches!(&end, Some(ChatEvent::Done { finish, .. }) if finish == "cancelled")
            && let Some(call) = a.parked()
        {
            end = Some(ChatEvent::Confirm {
                name: call.name.clone(),
                args: call.args.clone(),
            });
        }
        let history = a.history();
        drop(a);
        // Let go before the end is sent: a send right after it may reopen the agent on another
        // folder, which `Inner::agent` refuses while anything else holds it.
        drop(agent);
        let mut st = self.lock();
        let failed = st
            .model
            .as_ref()
            .is_some_and(|m| m.larch.health().state == HealthState::Failed);
        // Any answer names a chat that has no name yet, from its first message: a stopped first
        // turn, or a naming that failed, is named after the next answer.
        let first = history.iter().find(|m| m.role == larch::Role::User);
        let naming = match (&end, first, st.model.as_ref()) {
            (Some(ChatEvent::Done { finish, .. }), Some(first), Some(m))
                if !failed && finish == "eos" =>
            {
                (self.start_naming(&id)).map(|job| (job, m.larch.clone(), first.content.clone()))
            }
            _ => None,
        };
        if let Some(chat) = st.agents.get_mut(&id) {
            chat.turn = None;
            chat.history = history;
            chat.ended = Some(Instant::now());
        }
        drop(st);
        send(match end {
            _ if failed => ChatEvent::Error {
                message: STOPPED.into(),
            },
            Some(event) => event,
            None => ChatEvent::Error {
                message: reason.unwrap_or_else(|| "The turn ended without an answer.".into()),
            },
        });
        // After the end is sent, so the answer is never held for its title.
        match naming {
            Some((Ok((path, job)), larch, first)) => self.name_chat(&id, path, job, larch, &first),
            Some((Err(message), ..)) if lock(&self.names).told.insert(id.clone()) => {
                send(ChatEvent::TitleFailed { message })
            }
            _ => {}
        }
    }

    /// Registers a naming job for the chat, unless it has a name or one is being given: where
    /// the name goes and the job, which only the job may hold strongly.
    fn start_naming(&self, id: &str) -> Option<Result<(PathBuf, Arc<Naming>), String>> {
        let path = match self.path(id) {
            Ok(transcript) => named_path(&transcript),
            Err(e) => return Some(Err(format!("could not name this chat: {e}"))),
        };
        let mut names = lock(&self.names);
        if path.exists() || names.jobs.get(id).is_some_and(|w| w.strong_count() > 0) {
            return None;
        }
        let job = Arc::new(Naming::default());
        names.jobs.insert(id.into(), Arc::downgrade(&job));
        Some(Ok((path, job)))
    }

    /// Asks the model to name the chat from its `first` message and saves the name at `path`,
    /// unless `job` was stopped (the chat deleted or the model unloaded), then sends it, or why
    /// not the first time it fails in this run.
    fn name_chat(&self, id: &str, path: PathBuf, job: Arc<Naming>, larch: Larch, first: &str) {
        let named = name(&larch, larch_agent::user_text(first), &job);
        // The engine is let go before anything waits on a lock: unload waits for it.
        drop(larch);
        let event = {
            let mut names = lock(&self.names);
            // Checked and written under the lock delete_chat stops the job under, so a deleted
            // chat's name is never written after its files are gone.
            let event = match named {
                _ if job.stopped() => None,
                Ok(None) => None,
                Ok(Some(title)) => Some(match save_name(&path, &title) {
                    Ok(()) => ChatEvent::Title { title },
                    Err(message) => ChatEvent::TitleFailed { message },
                }),
                Err(e) => Some(ChatEvent::TitleFailed {
                    message: format!("could not name this chat: {e}"),
                }),
            };
            if (names.jobs.get(id)).is_some_and(|w| w.as_ptr() == Arc::as_ptr(&job)) {
                names.jobs.remove(id);
            }
            match event {
                Some(ChatEvent::TitleFailed { .. }) if !names.told.insert(id.into()) => None,
                Some(ChatEvent::Title { .. }) => {
                    names.told.remove(id);
                    event
                }
                _ => event,
            }
        };
        // Last: a dead entry is a job that has ended.
        drop(job);
        if let Some(event) = event {
            (self.emit)(Event::Chat {
                id: id.into(),
                event,
            });
        }
    }

    /// Runs a code task on its own thread: forwards its calls, results and notices, then parks
    /// its outcome for `code_decide` and sends its `Packet`, or sends why it failed. `words` is
    /// what the user asked, which the chat and its task `record` keep, as the task in `here`.
    #[allow(clippy::too_many_arguments)]
    fn code_run(
        &self,
        id: String,
        larch: Larch,
        code: Code,
        task: CodeTask,
        words: String,
        record: Option<TaskRecord>,
        here: PathBuf,
        stop: &Arc<AtomicBool>,
    ) {
        let send = |event| {
            (self.emit)(Event::Chat {
                id: id.clone(),
                event,
            })
        };
        let orphans =
            std::mem::take(&mut *self.orphans.lock().unwrap_or_else(PoisonError::into_inner));
        for text in orphans {
            send(ChatEvent::Notice { text });
        }
        // An ask answered for a file, a folder or everything widens the folder's allowances,
        // which the run saves before the call's result arrives: the UI is told them then.
        let widened = Cell::new(false);
        let shown = &mut |e| match e {
            // Code mode's write_file has the chat's name, whose relative paths the UI reads from
            // the agent's folder: this one's are the code folder's.
            AgentEvent::ToolCall { name, mut args } => {
                if let Some(path) = args["path"].as_str().filter(|_| name == "write_file") {
                    args["path"] = task.repo.join(path).display().to_string().into();
                }
                send(ChatEvent::ToolCall { name, args })
            }
            AgentEvent::ToolResult { name, summary } => {
                // Read back, so a save that failed tells what is kept.
                match widened.take().then(|| allow::load(&self.data, &task.repo)) {
                    Some(Ok(allowed)) => (self.emit)(Event::Allowed(FolderAllowed {
                        folder: task.repo.display().to_string(),
                        allowed,
                    })),
                    Some(Err(e)) => send(ChatEvent::Notice {
                        text: format!("cannot read what the folder allows: {e}"),
                    }),
                    None => {}
                }
                send(ChatEvent::ToolResult { name, summary })
            }
            AgentEvent::Notice(text) => send(ChatEvent::Notice { text }),
            AgentEvent::Progress(text) => send(ChatEvent::Progress { text }),
            AgentEvent::Thinking(text) => send(ChatEvent::Thinking { text }),
            AgentEvent::Documents(d) => send(d.into()),
            // Code mode's loop writes no answer text, parks no write and ends no turn: its end
            // is the outcome.
            AgentEvent::Text(_) | AgentEvent::NeedsConfirm { .. } | AgentEvent::Done(..) => {}
        };
        // An ask waits on this thread for `code_answer`; a chat that goes, or an app that
        // closes, drops the sender, which stops the run.
        let asked = &mut |ask: &Ask| {
            let (tx, rx) = mpsc::channel();
            {
                // Under the lock `cancel` stops in, so a stop is never missed in between.
                let mut st = self.lock();
                if stop.load(Ordering::Relaxed) {
                    return Decision::Stop;
                }
                st.asks.insert(id.clone(), tx);
            }
            send(ChatEvent::CodeAsk {
                name: ask.name.clone(),
                args: ask.args.clone(),
                path: ask.path.clone(),
                folder: ask.path.as_ref().map(|_| ask.folder().to_string()),
            });
            let decision = rx.recv().unwrap_or(Decision::Stop);
            widened.set(matches!(
                decision,
                Decision::File | Decision::Folder | Decision::All
            ));
            decision
        };
        let ran = code.run_until(&larch, &task, &self.data, shown, asked, stop);
        let mut st = self.lock();
        st.asks.remove(&id);
        st.code_stops.remove(&id);
        drop(st);
        drop(larch);
        let failed = (self.lock().model.as_ref())
            .is_some_and(|m| m.larch.health().state == HealthState::Failed);
        // Kept while the chat still reads as running (`codes` holds None), so a message the user
        // sends meanwhile is refused instead of landing in the history before the run; and
        // before its end is sent, so a question the user asks next reads it.
        let said = match &ran {
            Ok(outcome) => Some(told(outcome)),
            Err(e) if !failed => Some(format!("The code run failed: {e}")),
            Err(_) => None,
        };
        if let Ok(outcome) = ran.as_ref()
            && !task.read_only
        {
            let kept =
                TaskRecord::ran(record, &words, &outcome.packet, &here).save(&self.data, &id);
            if let Err(e) = kept {
                let text = format!("this chat will not remember its code task: {e}");
                send(ChatEvent::Notice { text });
            }
        }
        if let Some(said) = &said {
            self.keep_run(&id, &task.repo, &words, said);
        }
        let mut st = self.lock();
        let end = match ran {
            Ok(outcome) => {
                for text in &outcome.notices {
                    send(ChatEvent::Notice { text: text.clone() });
                }
                // A read-only run's answer is its notice: no packet, nothing to keep or undo.
                if task.read_only {
                    if outcome.notices.is_empty() {
                        let why = outcome.packet.stopped.as_deref().unwrap_or("it stopped");
                        let text = format!("The run ended with no answer ({why}).");
                        send(ChatEvent::Notice { text });
                    }
                    st.codes.remove(&id);
                    eos()
                } else {
                    let packet = packet(&outcome);
                    st.codes.insert(id.clone(), Some(outcome));
                    packet
                }
            }
            Err(e) => {
                st.codes.remove(&id);
                let message = match failed {
                    true => STOPPED.into(),
                    false => e.to_string(),
                };
                ChatEvent::Error { message }
            }
        };
        drop(st);
        send(end);
    }

    /// Keeps in chat `id`'s history what a code run in it was asked and did, `user` then
    /// `answer`, as the agent's own turns are kept: its next turn reads them, on the run's
    /// `folder`, and so does the chat opened again. A chat that cannot keep them says so.
    fn keep_run(&self, id: &str, folder: &Path, user: &str, answer: &str) {
        let kept = (|| {
            let agent = {
                let mut st = self.lock();
                self.agent(&mut st, id, Some(folder))?.agent.clone()
            };
            // Outside the state lock: a turn that holds the agent ends first.
            let mut a = agent.lock().unwrap_or_else(PoisonError::into_inner);
            a.add_exchange(user, answer).map_err(|e| e.to_string())?;
            let history = a.history();
            drop(a);
            if let Some(chat) = self.lock().agents.get_mut(id) {
                chat.history = history;
            }
            Ok::<_, String>(())
        })();
        if let Err(e) = kept {
            (self.emit)(Event::Chat {
                id: id.into(),
                event: ChatEvent::Notice {
                    text: format!("this chat will not remember the code run: {e}"),
                },
            });
        }
    }

    /// Streams one reply on its own thread, saves the turn, then sends its terminal event.
    fn reply(&self, id: String, mut stream: Seated, stop: Arc<AtomicBool>, before: usize) {
        let mut end = None;
        // ponytail: a stop lands at the stream's next event (a token, or the end of the prefill);
        // TextStream's canceller is crate-private in larch. Stop mid-prefill once F4 exposes it.
        while let Some(event) = stream.next() {
            match event {
                TextEvent::Text { delta, .. } if !delta.is_empty() => (self.emit)(Event::Chat {
                    id: id.clone(),
                    event: ChatEvent::Delta { text: delta },
                }),
                TextEvent::Text { .. }
                | TextEvent::Prefilled { .. }
                | TextEvent::Thinking { .. } => {}
                TextEvent::Done {
                    finish,
                    usage,
                    timings,
                    ..
                } => {
                    let metrics = Metrics {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        cached_tokens: usage.cached_tokens,
                        ttft_ms: timings.ttft_ms,
                        tpot_ms: timings.tpot_ms,
                        tok_s: timings.tok_s,
                    };
                    let done = ChatEvent::Done {
                        finish: camel(finish),
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        cached_tokens: usage.cached_tokens,
                        ttft_ms: timings.ttft_ms,
                        tok_s: timings.tok_s,
                        dropped_exchanges: timings.dropped_exchanges,
                    };
                    end = Some((Terminal::from(finish), Some(metrics), done));
                }
                TextEvent::Error(EngineError::Cancelled) => {
                    // Its dropped_exchanges is filled in below, once the history says.
                    let stopped = ChatEvent::Done {
                        finish: "cancelled".into(),
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        cached_tokens: 0,
                        ttft_ms: 0.0,
                        tok_s: 0.0,
                        dropped_exchanges: 0,
                    };
                    end = Some((Terminal::Cancelled, None, stopped));
                }
                TextEvent::Error(e) => {
                    let message = engine_error(e);
                    end = Some((Terminal::Error, None, ChatEvent::Error { message }));
                }
            }
            if stop.load(Ordering::Relaxed) {
                stream.cancel();
            }
        }
        let (terminal, metrics, mut event) = end.expect("a stream ends with a terminal event");
        let mut st = self.lock();
        let failed = st
            .model
            .as_ref()
            .is_some_and(|m| m.larch.health().state == HealthState::Failed);
        if let Some(chat) = st.chats.get_mut(&id) {
            chat.turn = None;
            // A failed turn leaves the history as it was; any other adds one exchange after
            // dropping the oldest ones the context could not hold.
            let (dropped, exchange) = match terminal {
                Terminal::Error => (0, None),
                _ => {
                    let history = chat.session.history();
                    let dropped = ((before + 2 - history.len()) / 2) as u32;
                    if let ChatEvent::Done {
                        dropped_exchanges, ..
                    } = &mut event
                    {
                        *dropped_exchanges = dropped;
                    }
                    (dropped, history.last_chunk::<2>().cloned())
                }
            };
            if let Err(e) = chat
                .transcript
                .append_turn(terminal, metrics, dropped, exchange)
            {
                event = ChatEvent::Error {
                    message: format!("The reply was not saved: {e:#}"),
                };
            }
        }
        if failed && terminal == Terminal::Error {
            event = ChatEvent::Error {
                message: STOPPED.into(),
            };
        }
        drop(st);
        (self.emit)(Event::Chat { id, event });
    }
}

impl State {
    /// A code run is running in chat `id`.
    fn coding(&self, id: &str) -> bool {
        self.codes.get(id).is_some_and(Option::is_none)
    }

    /// A worktree's packet waits in chat `id` for `code_decide`; a change made in place, which
    /// waits only to be undone, does not hold the chat.
    fn packet_waits(&self, id: &str) -> bool {
        let isolated = |o: &Outcome| o.task.folder.is_none();
        self.codes
            .get(id)
            .is_some_and(|o| o.as_ref().is_none_or(isolated))
    }

    /// Refused when a chat other than `id` changes `folder` in place: its run, or its change
    /// still waiting to be kept or undone, holds the folder until its `codes` entry goes (a
    /// decided packet, a failed run, a deleted chat). A run that changed nothing holds nothing.
    fn held(&self, id: &str, folder: Option<&Path>) -> Result<(), String> {
        let holds = |o: &&Option<Outcome>| o.as_ref().is_none_or(|o| !o.packet.diff.is_empty());
        let holder = (self.in_place.iter())
            .filter(|(chat, (f, _))| *chat != id && Some(f.as_path()) == folder)
            .find_map(|(chat, (_, title))| Some((self.codes.get(chat).filter(holds)?, title)));
        match holder {
            Some((None, title)) => Err(format!(
                "Another chat ({title}) is changing this folder; wait for it or run this one \
                 isolated."
            )),
            Some((Some(_), title)) => Err(format!(
                "Another chat ({title}) has a change waiting to be kept or undone in this \
                 folder; wait for it or run this one isolated."
            )),
            None => Ok(()),
        }
    }

    /// Chat `id`'s history: open, or read from its transcript at `path`.
    fn history(&self, id: &str, path: &Path) -> anyhow::Result<Vec<ChatMessage>> {
        match (self.agents.get(id), self.chats.get(id)) {
            (Some(a), _) => Ok(a.history.clone()),
            (_, Some(chat)) => Ok(chat.session.history()),
            _ => Transcript::resume(path).map(|(_, snapshot)| snapshot.history),
        }
    }

    /// A reply, an agent turn or a code run is being written in chat `id`.
    fn busy(&self, id: &str) -> bool {
        self.chats.get(id).is_some_and(|c| c.turn.is_some())
            || self.agents.get(id).is_some_and(|c| c.turn.is_some())
            || self.coding(id)
    }

    fn larch(&self) -> Result<&Larch, String> {
        self.model
            .as_ref()
            .map(|m| &m.larch)
            .ok_or_else(|| self.no_model().into())
    }

    /// Why nothing runs without a model: one is loading, or none is.
    fn no_model(&self) -> &'static str {
        match self.loading {
            true => LOADING,
            false => NO_MODEL,
        }
    }
}

/// Removes the worktrees and branches of the app's code tasks (`CHAT_TAG`) that an earlier run
/// of the app left under `<data>/agent/code`; what could not be removed, each in one line.
fn sweep_orphans(data: &Path) -> Vec<String> {
    let mut left = Vec::new();
    let code = data.join("agent").join("code");
    let tasks = fs::read_dir(&code).into_iter().flatten().flatten();
    for task in tasks.map(|e| e.path()).filter(|p| p.is_dir()) {
        let name = task.file_name().unwrap_or_default().to_string_lossy();
        if !name.contains(&format!("-{CHAT_TAG}")) {
            continue;
        }
        for attempt in fs::read_dir(&task).into_iter().flatten().flatten() {
            let n = attempt.file_name().to_string_lossy().into_owned();
            // The worktree is named as the project inside its attempt's folder
            // (`CodeTask::add_worktree`); an older run's is the attempt's folder itself.
            let inner = (fs::read_dir(attempt.path()).into_iter().flatten().flatten())
                .map(|e| e.path())
                .find(|p| p.join(".git").is_file());
            let path = inner.unwrap_or_else(|| attempt.path());
            if let Err(e) = remove_orphan(&path, &format!("iris/{name}-{n}")) {
                left.push(format!("'{}' could not be removed: {e}", path.display()));
            }
            let _ = fs::remove_dir(attempt.path());
        }
        // Empty once its attempts are gone.
        let _ = fs::remove_dir(&task);
    }
    left
}

/// Removes the worktree at `path` and its `branch` from the checkout its `.git` file names;
/// a folder whose `.git` names no checkout just goes.
fn remove_orphan(path: &Path, branch: &str) -> Result<(), String> {
    let gitfile = fs::read_to_string(path.join(".git")).unwrap_or_default();
    // `gitdir: <repo>/.git/worktrees/<name>`, as `git worktree add` writes it.
    let repo = (gitfile.trim().strip_prefix("gitdir:"))
        .map(|p| Path::new(p.trim()))
        .and_then(|p| p.ancestors().nth(3))
        .filter(|repo| repo.join(".git").exists())
        .map(Path::to_path_buf);
    let Some(repo) = repo else {
        return fs::remove_dir_all(path).map_err(|e| e.to_string());
    };
    let mut task = CodeTask {
        request: String::new(),
        repo,
        base_sha: String::new(),
        branch: Some(branch.into()),
        worktree: Some(path.into()),
        allowed_paths: vec![],
        test_cmd: None,
        check_cmds: vec![],
        budget: r#loop::BUDGET,
        tag: None,
        folder: None,
        drafts: vec![],
        base_tests: None,
        verify: Default::default(),
        read_only: false,
    };
    task.discard().map_err(|e| e.to_string())
}

/// A chat's title as the chat list shows it: the first words of its first message.
fn title(first: &str) -> String {
    larch_agent::user_text(first).chars().take(40).collect()
}

/// What the model is told when it names a chat.
const NAME: &str = "You name chats. You are shown the first message of a chat, and you reply \
with a short title for it: two to five words in Title Case, as a chat list shows it, such as \
\"Two-Player Connect-4 Game\" or \"Mount Fuji Essay\". Never answer the message or do what it \
asks. Reply with the title alone.";

/// A title's words: letters and digits in any script, joined by spaces, hyphens and a few signs.
/// At most five words of at most 20 characters each: an unbounded last word let a model that ran
/// out of words glue the rest of a sentence into it ("dormantstratovolcanoinJapan…").
const NAME_PATTERN: &str = r"[\p{L}\p{N}][\p{L}\p{N}'&+#.-]{0,19}( [\p{L}\p{N}'&+#.-]{1,20}){0,4}";

/// The longest title kept, in characters: longer ones lose their last words. The chat list's
/// fallback (`title`) is as long.
const NAME_CHARS: usize = 40;

/// The most of a first message the model reads to name its chat.
const NAME_BYTES: usize = 2000;

/// The user's turn that asks for the title: the message is quoted rather than sent as is, so a
/// small model names it instead of answering it ("write an essay about Mount Fuji" was answered).
fn name_request(message: &str) -> String {
    let message = message[..message.floor_char_boundary(NAME_BYTES)].trim();
    format!(
        "Give a short title (two to five words) for a chat that starts with the message below. \
         Do not answer it.\n\n<message>\n{message}\n</message>\n\nTitle:"
    )
}

/// `text`, the model's title, as kept: cut to `NAME_CHARS` at a word, then trimmed of its loose
/// end (punctuation-only words, a trailing `.`, `-`, `&` or `'`). Words keep their own signs:
/// "C++ vs C# Migration", "Salt & Pepper". None when nothing but punctuation is left.
fn short_title(text: &str) -> Option<String> {
    let mut words: Vec<&str> = Vec::new();
    let mut len = 0;
    for word in text.split_whitespace() {
        let more = usize::from(!words.is_empty()) + word.chars().count();
        if len + more > NAME_CHARS {
            if words.is_empty() {
                // One word past the limit: cut it, the list shows no more.
                words.push(
                    &word[..word
                        .char_indices()
                        .nth(NAME_CHARS)
                        .map_or(word.len(), |c| c.0)],
                );
            }
            break;
        }
        len += more;
        words.push(word);
    }
    while words
        .last()
        .is_some_and(|w| !w.chars().any(char::is_alphanumeric))
    {
        words.pop();
    }
    let title = words.join(" ");
    let title = title.trim_end_matches(['.', '-', '&', '\'']);
    (!title.is_empty()).then(|| title.to_owned())
}

/// Where the model's name for the chat at `transcript` is kept.
fn named_path(transcript: &Path) -> PathBuf {
    transcript.with_extension("title")
}

/// The chat's name as the chat list shows it: the model's, or else the first words of its
/// `first` message, or "New chat".
fn shown_name(transcript: &Path, first: Option<&str>) -> String {
    match fs::read_to_string(named_path(transcript)) {
        Ok(named) => named,
        Err(e) if e.kind() == io::ErrorKind::NotFound => first.map_or("New chat".into(), title),
        Err(e) => format!("Unreadable title: {e}"),
    }
}

/// Writes `title` beside `path` and renames it there once whole: a write cut short (a full disk)
/// leaves no `path` behind, which would be listed as the name and stop every retry
/// (`Inner::start_naming`). The half-written file is removed.
fn save_name(path: &Path, title: &str) -> Result<(), String> {
    let part = path.with_extension("title.part");
    let saved = fs::write(&part, title).and_then(|()| fs::rename(&part, path));
    saved.map_err(|e| match fs::remove_file(&part) {
        Err(gone) if gone.kind() != io::ErrorKind::NotFound => format!(
            "could not save this chat's name: {e}; nor remove {}: {gone}",
            part.display()
        ),
        _ => format!("could not save this chat's name: {e}"),
    })
}

/// `m`, locked: a panicked command must not take every later one with it.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Stops every naming job in `names` and waits up to `limit` for them to end, and so to let go
/// of their engine; false when one still runs.
fn stop_naming(names: &Mutex<Names>, limit: Duration) -> bool {
    let jobs: Vec<_> = lock(names).jobs.values().cloned().collect();
    for job in jobs.iter().filter_map(Weak::upgrade) {
        job.stop();
    }
    let end = Instant::now() + limit;
    while jobs.iter().any(|j| j.strong_count() > 0) {
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

/// The model's title for a chat that starts with `message`, held to `NAME_PATTERN`, greedy, not
/// thinking and behind any person's turn; None when it gives nothing but punctuation, or when
/// `job` is stopped, which cancels the request.
fn name(larch: &Larch, message: &str, job: &Naming) -> Result<Option<String>, EngineError> {
    let p = GenParams {
        // Five words and the end of turn; a cut word is trimmed below.
        max_tokens: 20,
        temperature: Some(0.0),
        thinking: Some(false),
        class: Class::Background,
        grammar: Some(Grammar {
            trigger: None,
            pattern: NAME_PATTERN.into(),
        }),
        ..GenParams::default()
    };
    let mut session = larch.session(SessionOptions {
        system: Some(NAME.into()),
        // One request, never continued: nothing to park.
        keep_kv: false,
        ..SessionOptions::default()
    });
    let stream = session.send(&name_request(message), p)?;
    // Set under the lock `Naming::stop` reads it under: a stop after this cancels the request,
    // one before it is seen here (dropping the stream cancels it).
    *lock(&job.request) = Some(stream.canceller());
    if job.stopped() {
        return Ok(None);
    }
    let mut text = String::new();
    for event in stream {
        if job.stopped() {
            return Ok(None);
        }
        match event {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
            TextEvent::Text { delta, .. } => text.push_str(&delta),
            TextEvent::Done { .. } => break,
            TextEvent::Error(e) => return Err(e),
        }
    }
    Ok(short_title(&text))
}

/// `repo`'s real path, as allowances are kept by.
fn real_folder(repo: &Path) -> Result<PathBuf, String> {
    let real = fs::canonicalize(repo).map_err(|e| format!("'{}': {e}", repo.display()))?;
    Ok(larch_agent::tool::plain(real))
}

/// The end of a reply with no model stats: a code run's that has no packet, or a decided one's.
fn eos() -> ChatEvent {
    ChatEvent::Done {
        finish: "eos".into(),
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        ttft_ms: 0.0,
        tok_s: 0.0,
        dropped_exchanges: 0,
    }
}

/// The outcome's packet as the UI shows it.
fn packet(o: &Outcome) -> ChatEvent {
    let p = &o.packet;
    let tests = |run: &Option<TestRun>| {
        run.as_ref().map(|t| Tests {
            passed: t.passed,
            failed: t.failed,
            failing: t.failing.clone(),
        })
    };
    ChatEvent::Packet {
        // A follow-up's commit is named after the task (`record::request`); its card, the ask.
        title: r#loop::title(record::now(&o.task.request)).into(),
        body: p.pr_body.clone(),
        diff: p.diff.clone(),
        tests_before: tests(&p.tests_before),
        tests_after: tests(&p.tests_after),
        scope: (p.scope.iter())
            .map(|s| ScopeEntry {
                path: s.path.clone(),
                inside: s.inside,
            })
            .collect(),
        outside: p.scope.iter().filter(|s| !s.inside).count() as u32,
        stop: p.stopped.clone(),
        steps: o.steps,
        parsed: o.parsed,
        checkpoint: o.task.folder.as_ref().and_then(|f| f.branch.clone()),
    }
}

/// What the chat keeps of a code run that ended with `o`, as its answer to the request: the
/// folder and the files changed in it, the model's last words, why the run stopped and where the
/// change is. The chat's next turn reads it, and can read the files by their paths.
fn told(o: &Outcome) -> String {
    /// Files named one by one; the rest are counted.
    const NAMED: usize = 20;
    let (p, repo) = (&o.packet, &o.task.repo);
    let files: Vec<String> = (p.scope.iter().take(NAMED))
        .map(|s| repo.join(&s.path).display().to_string())
        .collect();
    let mut text = match (files.is_empty(), p.scope.len().saturating_sub(NAMED)) {
        (true, _) => format!(
            "I worked in the folder {} and changed no file.",
            repo.display()
        ),
        (false, 0) => format!(
            "I worked in the folder {} and changed {}.",
            repo.display(),
            files.join(", ")
        ),
        (false, more) => format!(
            "I worked in the folder {} and changed {} and {more} more files.",
            repo.display(),
            files.join(", ")
        ),
    };
    let words = p.pr_body.trim();
    if !words.is_empty() {
        text = format!("{text} {}", &words[..words.floor_char_boundary(1000)]);
    }
    if let Some(why) = &p.stopped {
        text = format!("{text} The run stopped: {}.", why.trim_end_matches('.'));
    }
    if let Some(t) = &p.tests_after {
        text = format!(
            "{text} After it {} tests passed and {} failed.",
            t.passed, t.failed
        );
    }
    // With no change there is nothing to vouch for; the tests' counts above say what they say.
    if !p.scope.is_empty() {
        text = format!("{text} The change is {}.", p.verdict);
    }
    text + match o.task.folder {
        _ if p.scope.is_empty() => "",
        Some(_) => " The change is in the folder; the user can keep it or undo it.",
        None => {
            " The change waits in a worktree of its own for the user to merge or discard; the \
             folder is as it was."
        }
    }
}

fn message(m: ChatMessage) -> Message {
    let (role, content) = match m.role {
        larch::Role::System => (Role::System, m.content),
        // An agent chat's user message ends with its turn's tool note, which is the model's.
        larch::Role::User => (Role::User, larch_agent::user_text(&m.content).to_owned()),
        larch::Role::Assistant => (Role::Assistant, m.content),
        larch::Role::Tool => (Role::Tool, m.content),
    };
    Message {
        role,
        content,
        truncated: m.truncated,
        images: m.images,
    }
}

/// The images of a reopened chat encoded ahead of its next turn: about as many as the engine's tile
/// cache keeps (16 tiles; a screenshot is 7).
const WARM_IMAGES: usize = 3;

/// A stored image's bytes, refused past [`MAX_IMAGE_BYTES`] (no image the app stored is larger)
/// before more than that is read.
fn read_image(file: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    fs::File::open(file)
        .and_then(|f| f.take(MAX_IMAGE_BYTES as u64 + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("cannot read the image: {e}"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err("That image is larger than 32 MB.".into());
    }
    Ok(bytes)
}

/// The vision projector beside the model at `model`: the `mmproj*.gguf` in its folder that
/// [`pick_projector`] pairs with it, or the folder's one projector beside its one model when that
/// projector's name names no model (`mmproj-model-f16.gguf`, as some repos ship it). Only a projector
/// larch reads by its header (InternVL's or Qwen3-VL's), and only where the backend has the vision
/// ops (not Metal).
fn mmproj_beside(model: &Path) -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        return None;
    }
    let base = model.file_name()?.to_str()?;
    let (projectors, models) = ggufs_beside(model);
    let names = || projectors.iter().filter_map(|p| p.file_name()?.to_str());
    // A lone generic projector (unsloth's `mmproj-F16.gguf`) goes with the folder's one model, in
    // as many quants as it has.
    let lone = (models.len() == 1 && projectors.len() == 1)
        .then(|| names().next())
        .flatten()
        .filter(|p| matches!(model_stem(p).as_str(), "" | "model") && loadable_projector(p));
    let name = pick_projector(base, names(), false).or(lone)?;
    let path = projectors
        .iter()
        .find(|p| p.file_name() == Some(name.as_ref()))?;
    vision_projector(path).then(|| path.clone())
}

/// The projectors that may go with the model at `model`: those in its folder, then, for a model in
/// a folder of a Hugging Face snapshot (a split quant's), those at the snapshot's root, where the
/// repo keeps them. And the models of its folder, each split one once, by [`model_stem`].
fn ggufs_beside(model: &Path) -> (Vec<PathBuf>, HashSet<String>) {
    let dir = model.parent().unwrap_or(Path::new(""));
    let root = snapshot_of(model).filter(|root| *root != dir);
    let (mut projectors, mut models) = (Vec::new(), HashSet::new());
    for (folder, own) in std::iter::once((dir, true)).chain(root.map(|r| (r, false))) {
        let mut found: Vec<_> = (fs::read_dir(folder).into_iter().flatten().flatten())
            .map(|e| e.path())
            .collect();
        found.sort();
        for p in found {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let lower = name.to_lowercase();
            if !lower.ends_with(".gguf") {
                continue;
            }
            if lower.contains("mmproj") {
                projectors.push(p);
            } else if own && hf::split_part(&lower).is_none_or(|(_, part, _)| part == 1) {
                models.insert(model_stem(name));
            }
        }
    }
    (projectors, models)
}

/// The projectors beside the model at `model` ([`ggufs_beside`]) that go with it, by name only (any
/// kind and precision, on any platform): each whose name names that model, or the lone projector
/// with a generic name beside the folder's only model (in any number of quants). What deleting the
/// model deletes.
fn projectors_named_for(model: &Path) -> Vec<PathBuf> {
    let Some(name) = model.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let stem = model_stem(name);
    let (projectors, models) = ggufs_beside(model);
    let stem_of = |p: &PathBuf| model_stem(&p.file_name().unwrap_or_default().to_string_lossy());
    let mut named: Vec<PathBuf> = projectors
        .iter()
        .filter(|p| stem_of(p) == stem)
        .cloned()
        .collect();
    if named.is_empty()
        && models.len() == 1
        && let [lone] = projectors.as_slice()
        && matches!(stem_of(lone).as_str(), "" | "model")
    {
        named.push(lone.clone());
    }
    named
}

/// The snapshot of the Hugging Face cache the file at `file` is in
/// (`models--<owner>--<name>/snapshots/<commit>`), in a folder of it or at its root. A library's
/// own folder named `snapshots` is not one.
fn snapshot_of(file: &Path) -> Option<&Path> {
    (file.ancestors().skip(1)).find(|a| {
        let snapshots = a
            .parent()
            .filter(|s| s.file_name() == Some("snapshots".as_ref()));
        (snapshots.and_then(Path::parent).and_then(Path::file_name))
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("models--"))
    })
}

/// A GGUF file name reduced to the model it names, so a projector pairs with its model: lower case,
/// less `.gguf`, a split part's `-0000n-of-0000m`, a leading `mmproj` and the trailing quant or
/// precision (`OpenGVLab_InternVL3_5-1B-Q8_0.gguf` and `mmproj-OpenGVLab_InternVL3_5-1B-f16.gguf` are
/// both `opengvlab_internvl3_5-1b`).
pub(crate) fn model_stem(name: &str) -> String {
    let lower = name.to_lowercase();
    let Some(mut stem) = lower.strip_suffix(".gguf") else {
        return lower;
    };
    if let Some((s, ..)) = hf::split_part(&lower) {
        stem = s;
    }
    if let Some(rest) = stem.strip_prefix("mmproj") {
        stem = rest.trim_start_matches(['-', '_', '.']);
    }
    // The quant and what follows it (`-imat`, `-v2`: a build's tag, not another model).
    let q = hf::quant(&lower).to_lowercase();
    if !q.is_empty()
        && let Some(at) = stem.rfind(q.as_str())
    {
        stem = stem[..at].trim_end_matches(['-', '_', '.']);
    }
    stem.to_owned()
}

/// Whether the projector file `name` is one the loader reads: f16 by its name, or no precision in it
/// (a Q8_0 or bf16 projector would fail the load).
pub(crate) fn loadable_projector(name: &str) -> bool {
    !name.to_lowercase().ends_with(".gguf") || matches!(hf::quant(name).as_str(), "F16" | "")
}

/// Which of `projectors` (file names) goes with the model file `model`: a loadable one whose
/// [`model_stem`] is the model's (a library holding the 1B's and the 2B's pairs each with its own),
/// else with `any` (a repo's projectors all belong to its model) the first loadable one.
pub(crate) fn pick_projector<'a>(
    model: &str,
    projectors: impl IntoIterator<Item = &'a str>,
    any: bool,
) -> Option<&'a str> {
    let stem = model_stem(model);
    let mut usable: Vec<&str> = projectors
        .into_iter()
        .filter(|p| loadable_projector(p))
        .collect();
    usable.sort_unstable();
    (usable.iter().copied())
        .find(|p| model_stem(p) == stem)
        .or_else(|| usable.first().copied().filter(|_| any))
}

/// Whether the GGUF at `path` is a projector with keys larch can use, by its header.
fn vision_projector(path: &Path) -> bool {
    let read = || {
        let file = File::open(path).ok()?;
        let len = file.metadata().ok()?.len();
        Some(reads_images(&Gguf::read(BufReader::new(file), len).ok()?))
    };
    read().unwrap_or(false)
}

/// Whether the GGUF header `g` is a vision projector larch loads (InternVL's or Qwen3-VL's, which
/// Qwen 3.5's mmproj is).
pub(crate) fn reads_images(g: &Gguf) -> bool {
    larch_model::projector::ProjectorConfig::from_gguf(g).is_ok()
}

/// The header parses as a model larch runs and the file holds its first layer: an MTP draft file
/// (llama.cpp's `-draft`) keeps its model's header but only the extra layer, and fails at load.
// ponytail: blk.0 only, since a split model's first part holds the first layers; check every layer
// once a model's parts are read together here.
fn runnable(g: &Gguf) -> bool {
    let trunk = trunk(g);
    ModelConfig::from_gguf(g).is_ok()
        && g.tensors.iter().any(|t| t.name.starts_with("blk.0."))
        && known_types(g.tensors.iter().filter(|t| trunk(t)))
}

/// Whether the main pass loads a tensor: not one of the MTP layers past the trunk
/// (`blk.{n_layers}` on, `nextn_predict_layers`), whose bf16 routers the MTP drafter converts
/// itself (bartowski's Qwen3.5 GGUFs carry two).
fn trunk(g: &Gguf) -> impl Fn(&TensorInfo) -> bool + use<> {
    let n = ModelConfig::from_gguf(g).map_or(u32::MAX, |c| c.n_layers);
    move |t| {
        t.name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next()?.parse::<u32>().ok())
            .is_none_or(|i| i < n)
    }
}

/// Every tensor is of a type the engine loads; one other (an IQ quant, or Q2_K outside the routed
/// experts) fails the load.
fn known_types<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> bool {
    tensors.into_iter().all(loads)
}

/// Whether the engine loads tensor `t`'s type on this build's backend ([`loads_on`]).
fn loads(t: &TensorInfo) -> bool {
    loads_on(t, larch_kernels::BACKEND)
}

/// Whether the engine built for `backend` loads tensor `t`'s type: Q2_K only as routed experts
/// (`larch repack`'s) and only where CUDA or the CPU runs them; no GPU has a Q2_K kernel for the
/// other matrices, and Metal and HIP refuse Q2_K at load (`no_weights`). NVFP4 only on the CPU,
/// whose devices alone decode it so far.
fn loads_on(t: &TensorInfo, backend: &str) -> bool {
    gguf::dtype(t.ggml_type).is_ok_and(|d| match d {
        larch_kernels::DType::Q2K => {
            t.name.ends_with("_exps.weight") && matches!(backend, "cuda" | "cpu")
        }
        larch_kernels::DType::Nvfp4 => backend == "cpu",
        _ => true,
    })
}

/// GGML's type names, by id; "" for retired ids.
#[rustfmt::skip]
const GGML_NAMES: [&str; 41] = [
    "F32", "F16", "Q4_0", "Q4_1", "", "", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S",
    "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "", "", "", "TQ1_0", "TQ2_0", "",
    "", "", "MXFP4", "NVFP4",
];

/// The tensor types among `tensors` the engine cannot load, by GGML name, each once in id order;
/// an id GGML does not name (a fork's quant) as `type <id>`.
fn unloadable<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> Vec<String> {
    let ids: BTreeSet<u32> = tensors
        .into_iter()
        .filter(|t| !loads(t))
        .map(|t| t.ggml_type)
        .collect();
    ids.into_iter()
        .map(|t| match GGML_NAMES.get(t as usize) {
            Some(n) if !n.is_empty() => (*n).to_owned(),
            _ => format!("type {t}"),
        })
        .collect()
}

/// llama.cpp's `general.file_type` names (llama_ftype), by id; "" for retired ids.
#[rustfmt::skip]
const FILE_TYPES: [&str; 39] = [
    "F32", "F16", "Q4_0", "Q4_1", "", "", "", "Q8_0", "Q5_0", "Q5_1", "Q2_K", "Q3_K_S", "Q3_K_M",
    "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "IQ2_XXS", "IQ2_XS", "Q2_K_S",
    "IQ3_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ3_M", "IQ2_S", "IQ2_M", "IQ4_XS", "IQ1_M",
    "BF16", "", "", "", "TQ1_0", "TQ2_0", "MXFP4_MOE",
];

/// The quant a file was made as (`general.file_type`, as its file name spells it), else the type
/// holding the most tensor bytes: a Q2_K file keeps some tensors in Q3_K, so the bytes alone
/// mislabel it.
fn quant_label<'a>(g: &Gguf, tensors: impl IntoIterator<Item = &'a TensorInfo>) -> String {
    g.get("general.file_type")
        .and_then(Value::as_u64)
        .and_then(|t| FILE_TYPES.get(t as usize))
        .filter(|n| !n.is_empty())
        .map_or_else(|| main_quant(tensors), |n| (*n).to_owned())
}

/// The type holding the most tensor bytes, or "" when that type is not one larch knows.
// ponytail: the engine's `Weights::quant` is private; the type holding the most bytes is
// recomputed here until larch-engine exposes it.
fn main_quant<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> String {
    let mut by_type = BTreeMap::new();
    for t in tensors {
        let n: &mut u64 = by_type.entry(t.ggml_type).or_default();
        *n = n.saturating_add(t.bytes);
    }
    by_type
        .into_iter()
        .max_by_key(|&(_, b)| b)
        .and_then(|(t, _)| gguf::dtype(t).ok())
        .map_or_else(String::new, |d| format!("{d:?}"))
}

/// GGUF `general.architecture`, empty when the header cannot be read (the load then says why), and
/// whether the chat template asks for the XML tool-call body, as `larch-agent code` reads them.
fn gguf_tools(path: &Path) -> (String, bool) {
    let read = || {
        let file = File::open(path).ok()?;
        let len = file.metadata().ok()?.len();
        let g = Gguf::read(BufReader::new(file), len).ok()?;
        let arch = g.get("general.architecture").and_then(Value::as_str)?;
        let template = g.get("tokenizer.chat_template").and_then(Value::as_str);
        Some((
            arch.to_owned(),
            template.is_some_and(larch::toolcall::xml_body),
        ))
    };
    read().unwrap_or_default()
}

/// The user's home folder, the one the agent's file tools may use.
/// The app's own settings, in the data folder.
const SETTINGS: &str = "settings.json";
/// "Let tasks use the web" at start, from settings.json as read: on in a new profile (no file,
/// or settings without the key), as written when it says true or false, and off when the file
/// cannot be read, is not settings or holds anything else. On by default since a task saved with
/// it off made up the listings it was asked to find (23 Nov); off on a damaged file, as a user who
/// turned it off would get it back, and with it every saved task's web grants.
fn tasks_web(file: &std::io::Result<Vec<u8>>) -> bool {
    match file {
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
        Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(serde_json::Value::Object(settings)) => {
                (settings.get("tasks_web")).is_none_or(|v| v.as_bool() == Some(true))
            }
            _ => false,
        },
    }
}

/// The capabilities "Let tasks use the web" grants: web_search's and fetch_page's.
const WEB_GRANTS: [&str; 2] = ["network.web_search", "network.fetch_page"];

/// Where a chat's attachments are kept, beside its transcript `chat/<id>.jsonl`.
fn attached_path(transcript: &Path) -> PathBuf {
    transcript.with_extension("attached.json")
}

/// The paths as the UI shows them.
/// `list` as the UI shows it: a folder ends with the separator, so its chip is a folder's.
fn shown(list: &[PathBuf]) -> Vec<String> {
    let shown = |p: &PathBuf| match p.is_dir() && !p.ends_with(std::path::MAIN_SEPARATOR_STR) {
        true => format!("{}{}", p.display(), std::path::MAIN_SEPARATOR),
        false => p.display().to_string(),
    };
    list.iter().map(shown).collect()
}

/// `path` with the home folder written as `~`, as the chat's line shows it.
fn tilde(path: &Path) -> String {
    match home().and_then(|h| path.strip_prefix(h).ok().map(Path::to_path_buf)) {
        Some(rest) if rest.as_os_str().is_empty() => "~".into(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Writes `bytes` to `path` through `temp`, synced, then renamed over it: a crash leaves the old
/// file, never a torn one.
fn write_synced(path: &Path, temp: &Path, bytes: &[u8]) -> Result<(), String> {
    File::create(temp)
        .and_then(|mut f| {
            f.write_all(bytes)?;
            f.sync_all()
        })
        .and_then(|()| fs::rename(temp, path))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// config.toml's text with `docs_keep_ready` set: its line replaced where there is one, else added.
fn with_docs_keep_ready(text: &str, on: bool) -> String {
    let line = format!("docs_keep_ready = {on}");
    let mut found = false;
    let mut out: Vec<&str> = text
        .lines()
        .map(|l| match l.split_once('=') {
            Some((k, _)) if k.trim() == "docs_keep_ready" => {
                found = true;
                line.as_str()
            }
            _ => l,
        })
        .collect();
    if !found {
        out.push(&line);
    }
    out.join("\n") + "\n"
}

fn engine_error(e: EngineError) -> String {
    match e {
        EngineError::InvalidRequest(invalid::SESSION_BUSY) => BUSY.into(),
        EngineError::NotReady(HealthState::Failed) | EngineError::DeviceLost => STOPPED.into(),
        e => e.to_string(),
    }
}

fn knob_result(reply: KnobReply) -> KnobResult {
    match reply {
        KnobReply::Applied { rev, .. } => KnobResult::Applied { rev },
        KnobReply::Rejected { reason } => KnobResult::Rejected {
            reason: format!("{reason:?}"),
        },
        KnobReply::Indeterminate { reason, .. } => KnobResult::Pending {
            reason: format!("{reason:?}"),
        },
    }
}

/// A Rust variant name as the contract spells it: `StopSeq` → `stopSeq`.
fn camel(v: impl Debug) -> String {
    let s = format!("{v:?}");
    s[..1].to_lowercase() + &s[1..]
}

/// Where other apps keep the GGUFs they download, from their documented defaults and overrides:
/// (app, folder, how many folder levels below it the files sit).
fn other_model_dirs() -> Vec<(String, PathBuf, usize)> {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    // A HOME set on Windows (MSYS2, Emacs) is not where these apps keep their models.
    let home = home();
    let at = |p: &str| home.as_ref().map(|h| h.join(p));
    let cache = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        at("Library/Caches")
    } else {
        var("XDG_CACHE_HOME").or_else(|| at(".cache"))
    };
    let app_data = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        at("Library/Application Support")
    } else {
        var("XDG_DATA_HOME").or_else(|| at(".local/share"))
    };
    [
        ("LM Studio", at(".lmstudio/models"), 2),
        ("LM Studio", at(".cache/lm-studio/models"), 2),
        (
            "Ollama",
            var("OLLAMA_MODELS").or_else(|| at(".ollama/models")),
            0,
        ),
        // Where huggingface_hub keeps it, and Larch downloads to: `models--<owner>--<name>/
        // snapshots/<commit>/`, and one folder more for a split model's parts.
        ("Hugging Face", larch_fetch::hf_cache(), 4),
        (
            "llama.cpp",
            var("LLAMA_CACHE").or_else(|| cache.map(|c| c.join("llama.cpp"))),
            0,
        ),
        ("Jan", at("jan/models"), 2),
        (
            "Jan",
            var("APPDATA")
                .or_else(|| app_data.clone())
                .map(|d| d.join("Jan/data/models")),
            2,
        ),
        ("GPT4All", app_data.map(|d| d.join("nomic.ai/GPT4All")), 0),
    ]
    .into_iter()
    .filter_map(|(app, dir, depth)| Some((app.to_owned(), dir?, depth)))
    .collect()
}

/// The model GGUFs in `dir` and up to `depth` folder levels below it, sorted. Vision projectors
/// and the second and later parts of a split model are not models on their own.
fn model_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
    paths.sort();
    for p in paths {
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        if p.is_dir() {
            if depth > 0 {
                model_files(&p, depth - 1, out);
            }
        } else if name.ends_with(".gguf") && !name.contains("mmproj") && whole_split(&p) {
            out.push(p);
        }
    }
}

/// Whether `p` is a model on its own: not a split model's later part, and its first part only
/// with every part beside it. A download that failed (or still runs) on part 2 listed part 1,
/// which loaded only to fail, after unloading the model that was running.
fn whole_split(p: &Path) -> bool {
    let name = p.file_name().unwrap_or_default().to_string_lossy();
    match hf::split_part(&name) {
        None => true,
        Some((stem, part, n)) => {
            part == 1
                && (2..=n).all(|i| {
                    p.with_file_name(format!("{stem}-{i:05}-of-{n:05}.gguf"))
                        .is_file()
                })
        }
    }
}

/// Deletes a model file. A link in a Hugging Face cache snapshot (`<repo>/snapshots/<commit>/<path>`)
/// takes the blob it links to (`<repo>/blobs/<hash>`) with it, and the links to that blob at the
/// same path in the repo's other snapshots: left dangling, `hf cache scan` calls the repo corrupt.
// ponytail: the same path only; a link to the blob from another path of the repo would dangle.
fn remove_model_file(f: &Path) -> io::Result<()> {
    if let Some(snapshot) = snapshot_of(f)
        && let Some(snapshots) = snapshot.parent()
        && fs::symlink_metadata(f)?.is_symlink()
        && let Ok(blob) = fs::canonicalize(f)
        && blob.parent()
            == fs::canonicalize(snapshots.with_file_name("blobs"))
                .ok()
                .as_deref()
        && let Ok(rel) = f.strip_prefix(snapshot)
    {
        for s in fs::read_dir(snapshots)?.flatten() {
            let link = s.path().join(rel);
            if fs::canonicalize(&link).is_ok_and(|b| b == blob) {
                fs::remove_file(&link)?;
            }
        }
        return fs::remove_file(&blob);
    }
    fs::remove_file(f)
}

/// Ollama's models: each manifest names its weights blob (a GGUF without the extension).
fn ollama_models(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut manifests = Vec::new();
    let mut stack = vec![(dir.join("manifests"), 0)];
    while let Some((d, level)) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            // host/namespace/model/tag
            if p.is_dir() && level < 3 {
                stack.push((p, level + 1));
            } else if p.is_file() {
                manifests.push(p);
            }
        }
    }
    manifests.sort();
    for m in manifests {
        let Some(json) = std::fs::read(&m)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        else {
            continue;
        };
        let layers = json["layers"].as_array().into_iter().flatten();
        for l in layers.filter(|l| l["mediaType"] == "application/vnd.ollama.image.model") {
            if let Some(digest) = l["digest"].as_str() {
                out.push(dir.join("blobs").join(digest.replace(':', "-")));
            }
        }
    }
}

/// The sender a chat's agent hands its memory receipts to (its `remember`'s and its captures'):
/// each reaches the UI as `chat:<id>`'s `MemorySaved`, for Undo. The thread ends when every
/// sender is gone, with the agent and its captures.
fn forward_receipts(emit: Emit, chat: &str) -> std::io::Result<mpsc::Sender<memory::Receipt>> {
    let (to, receipts) = mpsc::channel::<memory::Receipt>();
    let chat = chat.to_owned();
    std::thread::Builder::new()
        .name("memory receipts".into())
        .spawn(move || {
            for receipt in receipts {
                emit(Event::Chat {
                    id: chat.clone(),
                    event: ChatEvent::MemorySaved { receipt },
                });
            }
        })?;
    Ok(to)
}

/// Looks at the app's chats every [`PIN_IDLE_TICK`] and lets an idle one's pin go
/// (`Inner::release_idle`), until the app is gone: its sender of `stop` goes with it.
fn watch_idle_pins(app: Weak<Inner>, stop: mpsc::Receiver<()>) -> io::Result<()> {
    std::thread::Builder::new()
        .name("larch-pin-idle".into())
        .spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = stop.recv_timeout(PIN_IDLE_TICK) {
                let Some(app) = app.upgrade() else {
                    break;
                };
                app.release_idle(Instant::now());
            }
        })
        .map(drop)
}

/// A helper as the chat event carries it.
fn chat_helper(h: &larch_agent::workers::Helper) -> ChatHelper {
    use larch_agent::workers::HelperState::*;
    let state = match h.state {
        Queued => "queued",
        Working => "working",
        Done => "done",
        Failed => "failed",
    };
    ChatHelper {
        task: h.task.clone(),
        state: state.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_api_creates_the_service_data_directory() {
        let dir = std::env::temp_dir().join(format!(
            "larch-app-task-new-path-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        assert!(!dir.join("runtime-service").exists());
        let api = app.task_api().unwrap();
        assert!(api.list().unwrap().is_empty());
        assert!(dir.join("runtime-service/tasks.sqlite").exists());
        drop(api);
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }

    /// Saved goals stay unreadable to other users even when the app opens the store before any
    /// service ran, under a readable data root and a runtime folder left at the umask's mode.
    #[cfg(unix)]
    #[test]
    fn task_api_keeps_the_runtime_folder_private_before_any_service_ran() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "larch-app-task-private-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        drop(app.task_api().unwrap());
        assert_eq!(mode(&dir.join("runtime-service")), 0o700, "created private");
        // As the device lease of an older build left it.
        fs::set_permissions(
            dir.join("runtime-service"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        drop(app.task_api().unwrap());
        assert_eq!(mode(&dir.join("runtime-service")), 0o700, "tightened");
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }

    /// Loads racing over one folder, however it is spelled, share one lease instead of the loser
    /// being refused by its own process; the lease goes when its last loader fails.
    #[test]
    fn loads_over_one_folder_share_its_device_lease() {
        let dir = std::env::temp_dir().join(format!(
            "larch-app-lease-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let start = std::sync::Barrier::new(8);
        let spelled = [
            dir.clone(),
            dir.join("."),
            dir.join("../").join(dir.file_name().unwrap()),
        ];
        let held: Vec<_> = std::thread::scope(|s| {
            let racers: Vec<_> = (0..8)
                .map(|i| {
                    let (start, path) = (&start, &spelled[i % 3]);
                    s.spawn(move || {
                        start.wait();
                        device_lease(path)
                    })
                })
                .collect();
            racers
                .into_iter()
                .map(|r| r.join().unwrap().unwrap())
                .collect()
        });
        assert!(held.iter().all(|lease| Arc::ptr_eq(lease, &held[0])));
        assert!(DeviceLease::acquire(&dir).is_err(), "the folder is leased");
        held.into_iter().for_each(release_lease);
        assert!(
            DeviceLease::acquire(&dir).is_ok(),
            "the last failed load let it go"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_web_switch_is_on_by_default_kept_and_gives_task_steps_the_web_only_while_on() {
        let dir = std::env::temp_dir().join(format!(
            "larch-app-tasks-web-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let tools = app.step_tools();
        // On in a new data folder: a task saved with it off made up the listings it was asked
        // to find (23 Nov).
        assert!(app.app_state().tasks_web);
        let names: Vec<_> = tools().iter().map(|t| t.tool.spec().name).collect();
        assert_eq!(names, ["web_search", "fetch_page"]);
        app.set_tasks_web(false).unwrap();
        assert!(tools().is_empty(), "off takes the web from the next step");
        drop(app);
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        assert!(!app.app_state().tasks_web, "off is kept across a restart");
        let tools = app.step_tools();
        assert!(tools().is_empty());
        app.set_tasks_web(true).unwrap();
        assert_eq!(tools().len(), 2);
        drop(app);
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        assert!(app.app_state().tasks_web, "on is kept across a restart");
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_web_switch_is_on_only_in_a_new_profile_or_as_written() {
        use std::io::{Error, ErrorKind};
        let file = |s: &str| Ok(s.as_bytes().to_vec());
        assert!(super::tasks_web(&Err(Error::from(ErrorKind::NotFound))));
        assert!(super::tasks_web(&file("{}")));
        assert!(super::tasks_web(&file(
            r#"{"libraries": [], "auto_tune": true}"#
        )));
        assert!(super::tasks_web(&file(r#"{"tasks_web": true}"#)));
        assert!(!super::tasks_web(&file(r#"{"tasks_web": false}"#)));
        // A file that cannot be read, is not settings or says something else keeps it off: a user
        // who turned it off would get it back, and with it every saved task's web grants.
        assert!(!super::tasks_web(&Err(Error::from(
            ErrorKind::PermissionDenied
        ))));
        assert!(!super::tasks_web(&file(r#"{"tasks_web": false"#)));
        assert!(!super::tasks_web(&file("[]")));
        assert!(!super::tasks_web(&file(r#"{"tasks_web": "yes"}"#)));
        assert!(!super::tasks_web(&file(r#"{"tasks_web": null}"#)));

        let dir = std::env::temp_dir().join(format!(
            "larch-app-tasks-web-broken-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(super::SETTINGS), r#"{"tasks_web": false"#).unwrap();
        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        assert!(!app.app_state().tasks_web && app.step_tools()().is_empty());
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn task_api_reads_records_from_the_service_data_directory() {
        use larch_task::store::SqliteStore;
        use larch_task::types::{PlanningPolicy, RunState, SCHEMA_VERSION, WorkflowDefinition};

        let dir = std::env::temp_dir().join(format!(
            "larch-app-task-path-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let service_dir = dir.join("runtime-service");
        fs::create_dir_all(&service_dir).unwrap();
        let mut store = SqliteStore::open(service_dir.join("tasks.sqlite")).unwrap();
        let definition = WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "Find evidence".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        };
        let run = RunState::new("service-run".into(), definition).unwrap();
        store.create_idempotent("create", &run).unwrap();
        drop(store);

        let app = App::new(dir.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let api = app.task_api().unwrap();
        let card = api.inspect("service-run").unwrap().unwrap();
        assert_eq!(card.definition.goal, "Find evidence");
        assert_eq!(api.list().unwrap().len(), 1);
        assert!(!dir.join("tasks.sqlite").exists());
        drop(api);
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }

    /// A title is at most five words of at most 20 characters: the model cannot glue a sentence
    /// into its last word, as it did ("Mount Fuji is a majestic dormantstratovolcanoinJapan…").
    #[test]
    fn a_title_is_a_few_short_words() {
        let held = |t: &str| {
            let g = larch_engine::grammar::Grammar {
                trigger: None,
                pattern: NAME_PATTERN.into(),
            };
            larch_engine::grammar::matches(&g, t).unwrap()
        };
        assert!(held("Two-Player Connect-4 Game"));
        assert!(held("Mount Fuji Essay"));
        assert!(held("Écrire un Essai"));
        assert!(held("Internationalization Bug"));
        assert!(!held("Mount Fuji is a majestic dormant"));
        assert!(!held(
            "Mount Fuji is a majestic dormantstratovolcanoinJapan"
        ));
        assert!(!held(" Leading Space"));
    }

    /// The kept title loses words past 40 characters and its loose ends; punctuation alone is
    /// no title.
    #[test]
    fn a_long_title_is_cut_at_a_word() {
        assert_eq!(
            short_title(" Two-Player Connect-4 Game.\n").as_deref(),
            Some("Two-Player Connect-4 Game")
        );
        let cut = short_title("Understanding Internationalization Problems Everywhere").unwrap();
        assert_eq!(cut, "Understanding Internationalization");
        assert!(cut.chars().count() <= NAME_CHARS);
        assert_eq!(short_title(&"x".repeat(60)).unwrap().len(), NAME_CHARS);
        assert_eq!(short_title(" - . "), None);
        // Signs that are part of a word stay (review of #514: "C++ vs C#" lost them).
        for kept in [
            "C++ vs C# Migration",
            "Learn C++",
            "Salt & Pepper Recipes",
            "F# Basics",
        ] {
            assert_eq!(short_title(kept).as_deref(), Some(kept));
        }
        assert_eq!(
            short_title("Salt & Pepper &").as_deref(),
            Some("Salt & Pepper")
        );
        assert_eq!(short_title("Rust Tips -").as_deref(), Some("Rust Tips"));
        assert_eq!(short_title("Two-").as_deref(), Some("Two"));
        let cut = short_title("Comparing C++ and C# for Game Engines Today").unwrap();
        assert_eq!(cut, "Comparing C++ and C# for Game Engines");
        assert!(cut.chars().count() <= NAME_CHARS);
    }

    /// The message is quoted for naming, never sent as the request, and cut to `NAME_BYTES`.
    #[test]
    fn the_message_is_quoted_to_be_named() {
        let asked = name_request("write a 1000 word essay about mount fuji\n");
        assert!(asked.contains("<message>\nwrite a 1000 word essay about mount fuji\n</message>"));
        assert!(asked.contains("Do not answer it."));
        assert!(name_request(&"é".repeat(NAME_BYTES)).len() < NAME_BYTES + 300);
    }

    /// A message's images may fill three quarters of the context, and no more.
    #[test]
    fn a_message_of_images_past_the_context_is_refused() {
        assert_eq!(too_many_rows(6144, 8192), None);
        let no = too_many_rows(6145, 8192).unwrap();
        assert!(no.contains("6145") && no.contains("8192"), "{no}");
        assert_eq!(too_many_rows(0, 0), None);
    }

    /// A send's cleanup removes the additions it took, not the same image added again while it
    /// ran (its chip still waits, and the next send must find it).
    #[test]
    fn a_send_releases_only_the_additions_it_took() {
        let png = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../larch-model/tests/data/vision/palette.png"),
        )
        .unwrap();
        let image = Arc::new(larch_model::vision::Image::new(Arc::from(&png[..])).unwrap());
        let r = image.hex();
        let mut pending = HashMap::new();
        pending.insert(r.clone(), (image.clone(), 1));
        let taken = vec![(r.clone(), 1)];
        // Added again while the send ran: a newer addition under the same reference.
        pending.insert(r.clone(), (image.clone(), 2));
        release_taken(&mut pending, &taken);
        assert_eq!(pending.get(&r).map(|p| p.1), Some(2));
        release_taken(&mut pending, &[(r.clone(), 2)]);
        assert!(pending.is_empty());
    }

    /// Both projectors larch loads read images by their header, another kind does not; and Qwen
    /// 3.5's repo layout (unsloth's `mmproj-F16.gguf`, a generic name beside the one model) pairs.
    #[test]
    fn either_projector_reads_images() {
        use larch_model::gguf::Value;
        use larch_model::tiny::Projector;
        let header = |b: Vec<u8>| Gguf::parse(&b).unwrap();
        assert!(reads_images(&header(Projector::new(32, 1).gguf())));
        assert!(reads_images(&header(Projector::qwen(32, 1).gguf())));
        let mut other = Projector::qwen(32, 1);
        other
            .kv
            .iter_mut()
            .find(|e| e.0 == "clip.projector_type")
            .unwrap()
            .1 = Value::Str("gemma3".into());
        assert!(!reads_images(&header(other.gguf())));
        let dir = std::env::temp_dir().join(format!("larch-qwen-pair-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("Qwen3.5-0.8B-Q8_0.gguf");
        fs::write(&model, b"GGUF").unwrap();
        fs::write(dir.join("mmproj-F16.gguf"), Projector::qwen(32, 1).gguf()).unwrap();
        let paired = mmproj_beside(&model);
        fs::remove_dir_all(&dir).unwrap();
        if !cfg!(target_os = "macos") {
            assert_eq!(paired, Some(dir.join("mmproj-F16.gguf")));
        }
    }

    #[test]
    fn a_split_model_is_listed_once_all_its_parts_are_there() {
        let dir = std::env::temp_dir().join(format!("larch-split-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let part = |i: u32| dir.join(format!("Big-Q4_K_M-{i:05}-of-00003.gguf"));
        let listed = || {
            let mut out = Vec::new();
            model_files(&dir, 0, &mut out);
            out
        };
        fs::write(part(1), b"").unwrap();
        fs::write(part(2), b"").unwrap();
        assert!(listed().is_empty(), "{:?}", listed());
        fs::write(part(3), b"").unwrap();
        assert_eq!(listed(), [part(1)]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Only a Hugging Face cache repo's `snapshots` holds snapshots: a library may have a folder
    /// of that name.
    #[test]
    fn a_snapshot_is_one_of_a_cache_repo() {
        let repo = Path::new("hub/models--o--n/snapshots/abc");
        assert_eq!(snapshot_of(&repo.join("m.gguf")), Some(repo));
        assert_eq!(snapshot_of(&repo.join("Q4_K_M/m.gguf")), Some(repo));
        assert_eq!(snapshot_of(Path::new("D/snapshots/2024/m.gguf")), None);
        assert_eq!(snapshot_of(Path::new("hub/models--o--n/blobs/abc")), None);
    }

    #[test]
    fn a_projector_pairs_with_the_model_its_name_names() {
        assert_eq!(
            model_stem("OpenGVLab_InternVL3_5-1B-Q8_0.gguf"),
            "opengvlab_internvl3_5-1b"
        );
        assert_eq!(
            model_stem("mmproj-OpenGVLab_InternVL3_5-1B-f16.gguf"),
            "opengvlab_internvl3_5-1b"
        );
        assert_eq!(model_stem("x-Q4_K_M-00001-of-00002.gguf"), "x");
        // A build's tag after the quant (imatrix, a revision) names no other model.
        assert_eq!(
            model_stem("Zed-Q4_K_M-imat.gguf"),
            model_stem("mmproj-Zed-f16.gguf")
        );
        assert_eq!(model_stem("Zed-Q8_0-v2.gguf"), "zed");
        assert_eq!(
            (
                model_stem("mmproj.gguf"),
                model_stem("mmproj-model-f16.gguf")
            ),
            ("".into(), "model".into())
        );
        let library = [
            "mmproj-OpenGVLab_InternVL3_5-1B-f16.gguf",
            "mmproj-OpenGVLab_InternVL3_5-2B-f16.gguf",
            "mmproj-OpenGVLab_InternVL3_5-2B-bf16.gguf",
        ];
        let pick = |m, any| pick_projector(m, library, any);
        assert_eq!(
            pick("OpenGVLab_InternVL3_5-2B-Q4_K_M.gguf", false),
            Some(library[1])
        );
        assert_eq!(
            pick("OpenGVLab_InternVL3_5-1B-Q8_0.gguf", false),
            Some(library[0])
        );
        // A text model whose name pairs with none takes none (a stray projector made it "see").
        assert_eq!(pick("Qwen2.5-0.5B-Instruct-Q8_0.gguf", false), None);
        // A repo's projectors all belong to its model: the f16 one, never Q8_0 or bf16 (the loader
        // refuses both).
        let repo = ["mmproj-model-Q8_0.gguf", "mmproj-model-f16.gguf"];
        assert_eq!(
            pick_projector("gemma-3-4b-it-Q8_0.gguf", repo, true),
            Some(repo[1])
        );
        assert_eq!(
            pick_projector("m-Q8_0.gguf", ["mmproj-m-bf16.gguf"], true),
            None
        );
    }

    /// A file with its model's header but only a later layer (an MTP `-draft` GGUF) is not a model.
    #[test]
    fn a_draft_layer_alone_is_not_runnable() {
        let mut g = Gguf::parse(&larch_model::tiny::gguf(larch_model::Arch::Llama, 3)).unwrap();
        assert!(runnable(&g));
        g.tensors.retain(|t| !t.name.starts_with("blk.0."));
        assert!(!runnable(&g));
    }

    /// A Q2_K file is refused before Load (one Q2_K tensor fails the engine's load) and labelled by
    /// its file type, not by the Q3_K tensors that hold most of its bytes.
    #[test]
    fn a_q2_k_file_is_labelled_and_not_runnable() {
        let mut g = Gguf::parse(&larch_model::tiny::gguf(larch_model::Arch::Llama, 3)).unwrap();
        g.kv.insert("general.file_type".into(), Value::U32(10));
        assert_eq!(quant_label(&g, &g.tensors), "Q2_K");
        g.tensors[0].ggml_type = 10;
        assert!(!runnable(&g));
        g.tensors[1].ggml_type = 30;
        g.tensors[2].ggml_type = 139;
        assert_eq!(unloadable(&g.tensors), ["Q2_K", "BF16", "type 139"]);
    }

    /// Q2_K routed experts (`larch repack`'s) load on CUDA and the CPU, so the file stays runnable
    /// there; a Metal or HIP build refuses them, so it lists the file as unsupported.
    #[test]
    fn q2_k_experts_are_runnable() {
        let mut g = Gguf::parse(&larch_model::tiny::gguf(larch_model::Arch::Qwen3Moe, 3)).unwrap();
        for t in g
            .tensors
            .iter_mut()
            .filter(|t| t.name.ends_with("_exps.weight"))
        {
            t.ggml_type = 10;
        }
        // This build's backend decides, as for any file.
        let here = matches!(larch_kernels::BACKEND, "cuda" | "cpu");
        assert_eq!(runnable(&g), here);
        let refused: &[&str] = if here { &[] } else { &["Q2_K"] };
        assert_eq!(unloadable(&g.tensors), refused);
        let exps = g.tensors.iter().find(|t| t.ggml_type == 10).unwrap();
        for (backend, runs) in [
            ("cuda", true),
            ("cpu", true),
            ("metal", false),
            ("hip", false),
        ] {
            assert_eq!(loads_on(exps, backend), runs, "{backend}");
        }
    }

    /// The MTP layer's bf16 routers (bartowski's Qwen3.5 35B) do not make the model unrunnable;
    /// a bf16 tensor in the trunk does.
    #[test]
    fn bf16_in_the_mtp_layer_still_runs() {
        let arch = larch_model::Arch::Qwen35Moe;
        let mut g = Gguf::parse(&larch_model::tiny::gguf_mtp(arch, 3)).unwrap();
        assert!(g.tensors.iter().any(|t| t.ggml_type == 30));
        assert!(runnable(&g));
        let i = g
            .tensors
            .iter()
            .position(|t| t.name.starts_with("blk.0."))
            .unwrap();
        g.tensors[i].ggml_type = 30;
        assert!(!runnable(&g));
    }

    /// A receipt an agent sends reaches the UI as the chat's `memorySaved` event, with the
    /// receipt the UI's Undo hands back.
    #[test]
    fn a_receipt_reaches_the_ui_as_the_chats_memory_saved_event() {
        let (tx, rx) = mpsc::channel();
        let emit: Emit = Arc::new(move |e| {
            let _ = tx.send(e);
        });
        let to = forward_receipts(emit, "c1").unwrap();
        let receipt = memory::Receipt {
            slug: "the-user-likes-tea".into(),
            text: "The user likes tea.\nSaid: \"I like tea a lot\"".into(),
            old: None,
            was: None,
            created: "2026-11-06T10:00:00Z".into(),
        };
        to.send(receipt.clone()).unwrap();
        let e = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert_eq!(
            (e.name(), e.payload()["kind"].as_str()),
            ("chat:c1".into(), Some("memorySaved"))
        );
        let back: memory::Receipt = serde_json::from_value(e.payload()["receipt"].clone()).unwrap();
        assert_eq!(back, receipt);
        // The thread ends with its last sender.
        drop(to);
        assert!(rx.recv_timeout(std::time::Duration::from_secs(5)).is_err());
    }

    /// The setting replaces its line, or is added, and leaves the other lines as they are.
    #[test]
    fn docs_keep_ready_is_set_in_config_text() {
        assert_eq!(with_docs_keep_ready("", false), "docs_keep_ready = false\n");
        let kept = "# mine\nthinking = \"off\"\n  docs_keep_ready=true # old\nweb_pages = 3\n";
        assert_eq!(
            with_docs_keep_ready(kept, false),
            "# mine\nthinking = \"off\"\ndocs_keep_ready = false\nweb_pages = 3\n"
        );
    }

    /// Unload stops a naming job and waits for it to let go of its engine, refusing while it
    /// holds on past the limit.
    #[test]
    fn stop_naming_stops_each_job_and_waits_for_it_to_end() {
        let names = Mutex::new(Names::default());
        let job = Arc::new(Naming::default());
        lock(&names)
            .jobs
            .insert("a".to_owned(), Arc::downgrade(&job));
        assert!(!stop_naming(&names, Duration::from_millis(30)));
        assert!(job.stopped());
        drop(job);
        assert!(stop_naming(&names, Duration::from_millis(30)));
    }

    /// The tiny model trained for `ctx` tokens, with a byte vocabulary and a template that writes
    /// each message as it is (tests/app.rs's `tiny_gguf_ctx`).
    fn tiny_model(ctx: u32) -> Vec<u8> {
        let bytes = larch_model::tiny::gguf(larch_model::Arch::Llama, 3);
        let g = Gguf::parse(&bytes).unwrap();
        let tensors: Vec<_> = (g.tensors.iter())
            .map(|t| {
                let at = (g.data_offset + t.offset) as usize;
                let data = bytes[at..at + t.bytes as usize].to_vec();
                (t.name.clone(), t.dims.clone(), t.ggml_type, data)
            })
            .collect();
        let mut kv: Vec<_> = (g.kv.into_iter())
            .filter(|(k, _)| k != "llama.context_length")
            .collect();
        let template = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";
        let tokens = (0..256).map(|b| Value::Str(format!("<0x{b:02X}>")));
        kv.extend([
            ("llama.context_length".into(), Value::U32(ctx)),
            ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
            ("tokenizer.ggml.tokens".into(), Value::Arr(tokens.collect())),
            (
                "tokenizer.ggml.token_type".into(),
                Value::Arr(vec![Value::I32(6); 256]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), Value::U32(97)),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(255)),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
            ("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)),
            (
                "tokenizer.chat_template".into(),
                Value::Str(template.into()),
            ),
        ]);
        gguf::write(&kv, &tensors)
    }

    /// A chat that holds a pin keeps it right after its answer and lets it go once it has had no
    /// turn for `PIN_IDLE`, written to its pin file, so plain chats get their plan back. With
    /// "keep documents ready" off no file holds it, and it stays until another chat needs the
    /// room.
    #[test]
    fn a_chat_with_no_turn_for_pin_idle_lets_its_pin_go() {
        let dir = std::env::temp_dir().join(format!(
            "larch-app-pin-idle-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("models")).unwrap();
        let (tx, rx) = mpsc::channel();
        let app = App::new(dir.clone(), larch_kernels::CPU, move |e| {
            let _ = tx.send(e);
        })
        .unwrap();
        app.set_model_dirs(Vec::new());
        let model = dir.join("models").join("tiny-32k.gguf");
        fs::write(&model, tiny_model(32_768)).unwrap();
        app.load_model(model.to_str().unwrap()).unwrap();
        let id = app.new_chat(None).unwrap();
        // Past what the loaded context holds as plain text beside a turn: it is pinned.
        let lease = dir.join("lease.md");
        let line = "Rent is $1,200 a month. The lease ends 3 March 2027.\n";
        fs::write(&lease, line.repeat(45)).unwrap();
        app.attach(&id, lease.to_str().unwrap()).unwrap();
        let pinned = || {
            let st = app.inner.lock();
            let stats = st.model.as_ref().and_then(|m| m.larch.prefix_cache_stats());
            stats.map_or(0, |s| s.pinned_blocks)
        };
        let pin_files = || {
            (fs::read_dir(dir.join("doc-cache"))
                .into_iter()
                .flatten()
                .flatten())
            .filter(|e| e.path().extension().is_some_and(|x| x == "pin"))
            .count()
        };
        let ask = |text: &str| {
            app.agent_send(&id, text, None, Some(Intent::Chat), &[])
                .unwrap();
            let (mut events, end) = (Vec::new(), Instant::now() + Duration::from_secs(300));
            let mut heard = Instant::now();
            loop {
                let left = end.saturating_duration_since(Instant::now());
                // On a miss, whether the turn was still going (a slow runner) or had stopped.
                let e = rx.recv_timeout(left).unwrap_or_else(|e| {
                    let (n, quiet) = (events.len(), heard.elapsed());
                    let tail = &events[n.saturating_sub(3)..];
                    panic!(
                        "an end of {text:?}: {e}; {n} events, none for {quiet:?}, the last {tail:?}"
                    )
                });
                if let Event::Chat { id: chat, event } = e {
                    heard = Instant::now();
                    assert_eq!(chat, id);
                    let last = matches!(
                        event,
                        ChatEvent::Done { .. }
                            | ChatEvent::Error { .. }
                            | ChatEvent::Confirm { .. }
                    );
                    events.push(event);
                    if last {
                        return events;
                    }
                }
            }
        };
        let read = |events: &[ChatEvent]| {
            (events.iter()).any(|e| {
                matches!(
                    e,
                    ChatEvent::Documents {
                        state: DocState::Reading { .. },
                        ..
                    }
                )
            })
        };
        // The chip hears that the pin went (#591 review).
        let released = || {
            (rx.try_iter()).any(|e| {
                matches!(
                    e,
                    Event::Chat {
                        event: ChatEvent::Documents {
                            state: DocState::Released,
                            ..
                        },
                        ..
                    }
                )
            })
        };
        let events = ask("When does the lease end?");
        assert!(read(&events) && pinned() > 0, "{events:?}");
        app.inner.release_idle(Instant::now());
        assert!(pinned() > 0, "it just answered");
        assert!(!released());
        // An agent something else holds (a turn ending, an attach) is left to the next look.
        let other = (app.inner.lock().agents.get(&id)).map(|c| Arc::clone(&c.agent));
        app.inner.release_idle(Instant::now() + PIN_IDLE);
        assert!(pinned() > 0, "its agent is held elsewhere");
        drop(other);
        app.inner.release_idle(Instant::now() + PIN_IDLE);
        assert_eq!(pinned(), 0, "idle for PIN_IDLE");
        assert_eq!(pin_files(), 1, "written to its file");
        assert!(released());
        // Kept off: with no file to import, the next question pins the documents by reading them
        // again (here from the blocks the prefix cache still has). No file holds that pin, so it
        // stays through the pause, as letting it go would cost the next question a full read
        // (#591 review); another chat that needs the room still takes it.
        app.set_docs_keep_ready(false).unwrap();
        assert_eq!(pin_files(), 0);
        ask("What is the rent?");
        let held = pinned();
        assert!(held > 0);
        app.inner.release_idle(Instant::now() + PIN_IDLE);
        assert_eq!((pinned(), pin_files()), (held, 0), "kept with no file");
        assert!(!released());
        app.inner.unpin_others("another chat");
        assert_eq!((pinned(), pin_files()), (0, 0));
        assert!(released());
        drop(app);
        fs::remove_dir_all(dir).unwrap();
    }
}
