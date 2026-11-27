//! Commands the UI calls and events the core sends, serialized as JSON (camelCase). The UI's
//! `apps/desktop/src/demo.rs` fills the same shapes for `--demo`.
//!
//! Commands: `app_state() -> AppState`, `list_models() -> Vec<ModelEntry>`, `load_model(path)`,
//! `delete_model(path)` (only from a Larch library; unloads it if loaded),
//! `add_library(path) -> Vec<String>`, `remove_library(path) -> Vec<String>` (the files stay),
//! `list_chats() -> Vec<ChatEntry>`, `new_chat(system: Option<String>) -> String` (the chat id),
//! `open_chat(id) -> Vec<Message>`, `delete_chat(id)`, `rename_chat(id, title)`, `send(chat_id, text, params: SendParams)`,
//! `cancel(chat_id)`, `knobs() -> Vec<Knob>`, `knob_set(name, value) -> KnobResult`,
//! `knob_revert(engine, rev) -> KnobResult` (`engine` from the change's `ConfigChange`), `restart_app()` (the shell relaunches the app; the GPU-stopped
//! screen calls it), `agent_send(chat_id, text, repo: Option<String>, intent: Option<Intent>, images:
//! Option<Vec<String>>)` (a
//! turn with larch-agent's tools, ending in `Done` or `Confirm`; `intent` is what the text asks
//! for when the UI knows it, else the model reads it; a change starts code mode with k 1 instead,
//! on the chat's project folder `repo` or the checkout the text names), `agent_confirm(chat_id,
//! yes)` (answers a `Confirm`), `code_start(chat_id, request, repo, test_cmd: Option<String>, k,
//! isolated: Option<bool>)` (larch-agent's code mode on the folder `repo`: in place, any folder,
//! after a checkpoint branch, each write not allowed yet a `CodeAsk`; isolated or with k above 1,
//! k attempts in worktrees of a repository's top; its calls, results and notices as an agent turn's, ending in `Packet`),
//! `code_answer(chat_id, decision: Decision)` (answers a `CodeAsk`; the run goes on),
//! `code_decide(chat_id, yes)` (answers the `Packet`: merges it into the checkout or discards it,
//! or, made in place, keeps it or undoes it; then `Notice` and `Done`; all three only where
//! `AppState.code_mode` is true), `code_allowed(repo) -> Allowed`, `code_allow(repo, allowed:
//! Allowed)` (what code mode may change in the folder without asking), `attachments(chat_id) -> Vec<String>`,
//! `attach(chat_id, path) -> Attached` (a folder or file this chat's agent may read; its next
//! turn reopens it with them; a chat starts with none), `detach(chat_id, path) -> Vec<String>`,
//! `image_add(<raw body: the image's bytes>) -> ImageInfo` (an image for the next message, checked
//! now; `agent_send` names it by `reference`), `chat_image(chat_id, reference) -> <raw bytes>` (a
//! chat image, for its thumbnail),
//! `memory_list() -> Vec<Fact>`,
//! `memory_add(text) -> Receipt` (with `old` when it updated a fact like it), `memory_update(slug,
//! expected, text) -> bool` (the sentence only, a capture's `Said:` kept; refused when the fact
//! is no longer `expected`; false when nothing changed), `memory_forget(slug) -> Gone`,
//! `memory_restore(gone)` (a Forget's Undo), `memory_undo(receipt)` (refused when the fact changed since),
//! `memory_enabled() -> bool`, `memory_set_enabled(on)`, `hf_search(query, sort, limit) -> Vec<HfModel>`,
//! `hf_files(repo) -> Vec<HfFile>`,
//! `hf_inspect(repo, file) -> RemoteModel`, `download_model(repo, file, library: Option<String>) -> String`
//! (into that library, None: the shared Hugging Face cache, else the default library; the download id),
//! `cancel_download(id)`, `list_downloads() -> Vec<Download>`,
//! `mcp_search(query, cursor: Option<String>) -> McpSearch` (the official MCP Registry),
//! `mcp_servers() -> Vec<McpServerStatus>` (secrets masked), `mcp_add(server: McpServer)`,
//! `mcp_remove(id)`, `mcp_set_enabled(id, enabled)`, `mcp_set_trusted(id, trusted)` (workflow steps
//! may call its read tools), `mcp_test(id) -> Vec<McpToolInfo>` (connects
//! if it is not), `mcp_sign_in(id) -> McpSignIn` (a remote server behind OAuth: the page to open;
//! the rest ends in `mcp:status`), `mcp_sign_in_finish(id, url)` (the address the browser ended on,
//! when its return could not reach this computer), `mcp_sign_out(id)`. Every command returns `Result<_, String>`; the error is shown to the user as it is.
//!
//! Events: `chat:<id>` carries [`ChatEvent`], `engine:sample` [`EngineSample`] (about 4 a second),
//! `engine:config` [`ConfigChange`], `engine:failed` [`Failed`], `engine:health` [`Health`] (on every
//! change), `download:progress` [`DownloadProgress`], `download:done` [`DownloadDone`],
//! `download:failed` [`DownloadFailed`], `code:allowed` [`FolderAllowed`] (on every save of a folder's allowances), `mcp:status`
//! [`McpChanged`] (a server connected, failed or was turned off).

use serde::{Deserialize, Serialize};

/// What a message asks for: `"change"`, `"code"`, `"web"` or `"chat"` (`agent_send`).
pub use larch_agent::router::Intent;

/// A folder's allowances, `{"silent":false,"files":["calc.py"],"folders":["tests"]}`, and the
/// answer to a `CodeAsk`: `"once"`, `"file"`, `"folder"`, `"all"` (silent from now on), `"no"`
/// or `"stop"`.
pub use larch_agent::code::allow::{Allowed, Decision};

/// A save the user can undo: `{"slug","text","old","was":{"kind","source"},"created"}`.
pub use larch_agent::memory::{Gone, Receipt};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Health {
    /// No model loaded yet.
    Empty,
    Loading,
    Ready,
    /// A GPU fault (architecture 9.11): every command fails until the app restarts.
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppState {
    pub health: Health,
    /// Path of the loaded GGUF.
    pub model: Option<String>,
    pub device: String,
    pub data_dir: String,
    /// The GPU's memory, read once when the app first asks, before a model takes any. `None` on the
    /// CPU reference or when the device cannot be read (never estimated).
    pub gpu_total_bytes: Option<u64>,
    /// What other programs left free then: the room a model has on this machine.
    pub gpu_free_bytes: Option<u64>,
    /// Where the agent's file tools start (the home folder): a relative path in a tool call is in it.
    pub agent_folder: Option<String>,
    /// Larch's model libraries: `<data>/models` first, then the folders the user added.
    pub libraries: Vec<String>,
    /// The shared Hugging Face cache, where `download_model` with no library downloads to; None
    /// when there is no home folder to find it in, and downloads go to the first library.
    #[serde(default)]
    pub hf_cache: Option<String>,
    /// "Keep documents ready after restart" (config.toml `docs_keep_ready`, on by default).
    pub docs_keep_ready: bool,
    /// "Let tasks use the web" (on by default): task steps may call web_search and fetch_page.
    pub tasks_web: bool,
    /// Code mode is on (`code_start`): settings.json says `"code_mode": true` (default off) and
    /// this OS has the sandbox the tests run in.
    pub code_mode: bool,
    /// The loaded model reads images: a vision projector (`mmproj*.gguf`) beside it loaded too.
    pub vision: bool,
    /// Why the vision projector beside the loaded model did not load (it then reads no images).
    #[serde(default)]
    pub vision_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub path: String,
    pub name: String,
    pub arch: String,
    pub quant: String,
    pub size_bytes: u64,
    /// The architecture is one larch runs (architecture 8).
    pub supported: bool,
    pub fit: Fit,
    /// The app whose folder holds it: `Larch` for its libraries, else `LM Studio`, `Ollama`,
    /// `Hugging Face`, `llama.cpp`, `Jan` or `GPT4All`. Those files are read in place.
    pub source: String,
    /// The folder it was found under: a Larch library, or the other app's model folder.
    pub library: String,
    /// A vision projector (`mmproj*.gguf`) is beside it, so it loads with it and reads images;
    /// false once that projector failed to load with it.
    #[serde(default)]
    pub vision: bool,
    /// "Faster on this computer": offered for this model here, and its copy once made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faster: Option<Faster>,
}

/// A model's faster copy on this computer (`build_faster`, `use_faster`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Faster {
    /// The copy's path once it is made.
    pub copy: Option<String>,
    /// It is being made now.
    pub building: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum Fit {
    Fits {
        kv_bytes: u64,
        max_seqs: u32,
    },
    TooBig {
        need_bytes: u64,
        have_bytes: u64,
    },
    /// Not checked: unsupported, or the device could not be read.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatEntry {
    pub id: String,
    /// The model's name for the chat (`ChatEvent::Title`), or else its first user message,
    /// shortened.
    pub title: String,
    /// Unix milliseconds.
    pub updated_at: u64,
    pub turns: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool's result; the UI does not show these.
    Tool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// The reply was stopped before it finished.
    pub truncated: bool,
    /// A user message's images, by reference (`chat_image` reads one); none is omitted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// An image `image_add` took for the next message: its reference (the sha256 of its bytes, what
/// `agent_send` names it by), its size after EXIF rotation, and the prompt rows it takes on the
/// loaded model (None when the model reads no images).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageInfo {
    pub reference: String,
    pub width: u32,
    pub height: u32,
    pub tokens: Option<u32>,
    /// Earlier unsent images this one pushed out of the bytes kept for messages not sent yet: their
    /// chips go, and a send naming them is refused.
    #[serde(default)]
    pub evicted: Vec<String>,
}

/// Unset fields follow the engine's knob defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SendParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum ChatEvent {
    Delta {
        text: String,
    },
    /// The turn ended and was saved. `cancelled`: stopped, its text so far kept as truncated; its
    /// stats may be zero.
    Done {
        /// `eos`, `length`, `stopSeq` or `cancelled`.
        finish: String,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        ttft_ms: f64,
        tok_s: f64,
        /// Oldest exchanges dropped to fit the context limit.
        dropped_exchanges: u32,
    },
    /// The turn did not happen, or stopped on a fault; the user message is removed.
    Error {
        message: String,
    },
    /// An agent turn called a tool.
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    /// What the call returned, in one line.
    ToolResult {
        name: String,
        summary: String,
    },
    /// The agent turn stopped at a write; `agent_confirm` answers it and the turn goes on.
    Confirm {
        name: String,
        args: serde_json::Value,
    },
    /// A code run made in place waits on a write its folder does not allow yet: `code_answer`
    /// answers it, and the run goes on. `path` is the file, relative to the folder, and `folder`
    /// the one a `"folder"` answer allows; both None for a command, which may write any file.
    CodeAsk {
        name: String,
        args: serde_json::Value,
        path: Option<String>,
        folder: Option<String>,
    },
    /// Something the user should know: no tools for this model, a failure, a compaction.
    Notice {
        text: String,
    },
    /// The chat's name in the chat list, which the model gives after the chat's first finished
    /// answer (after its `Done`). `{"kind":"title","title":"Two-Player Connect-4 Game"}`
    Title {
        title: String,
    },
    /// Why the chat could not be named, after its answer's `Done` and outside any reply; the
    /// next finished answer tries again. `{"kind":"titleFailed","message":"…"}`
    TitleFailed {
        message: String,
    },
    /// The chat saved a fact without asking: a `remember` the user asked for, or a background
    /// capture after a turn (which can come after `Done`). `memory_undo(receipt)` undoes it.
    /// `{"kind":"memorySaved","receipt":{"slug":"…","text":"…","old":null,"was":null,"created":"…"}}`
    MemorySaved {
        receipt: Receipt,
    },
    /// A code run's end: the attempt the tests picked, which `code_decide` merges or discards.
    /// `{"kind":"packet","title":"Fix add","body":"add adds now","diff":"diff --git …",
    /// "testsBefore":{"passed":0,"failed":1,"failing":["test_add"]},"testsAfter":{"passed":1,
    /// "failed":0,"failing":[]},"scope":[{"path":"calc.py","inside":true}],"outside":0,
    /// "stop":"the model said it is done and the tests agree","steps":4,"parsed":4}`
    Packet {
        /// The commit's subject: the request's first line.
        title: String,
        /// The model's words when it said it was done.
        body: String,
        /// The change against the checkout's HEAD when the run began.
        diff: String,
        /// The tests on the checkout's HEAD; null when the folder has none.
        tests_before: Option<Tests>,
        /// The tests on the change; null when the folder has none.
        tests_after: Option<Tests>,
        /// Every changed path, in git's order.
        scope: Vec<ScopeEntry>,
        /// The changed paths outside the allowed ones.
        outside: u32,
        /// Why the attempt stopped: done with the tests agreeing, or the budget that ran out.
        stop: Option<String>,
        /// The model's replies, over every attempt.
        steps: u32,
        /// The replies that parsed.
        parsed: u32,
        /// Made in place: the branch that keeps the folder as it was, and `code_decide` keeps or
        /// undoes the change; None for a worktree's packet, which it merges or discards.
        checkpoint: Option<String>,
    },
    /// A piece of the model's reasoning for a step of the reply, as it comes, before that step's
    /// words and apart from them. `{"kind":"thinking","text":"…"}`
    Thinking {
        text: String,
    },
    /// What a running tool is doing ("Reading 3 of 4 pages"): one line, which the next one or
    /// the answer replaces.
    Progress {
        text: String,
    },
    /// The chat's pinned documents as they stand (the chip), or a follow-up's speed; the next one
    /// replaces it. `text` is the chip's words ("Reading p. 40 of 120").
    /// `{"kind":"documents","state":{"kind":"ready"},"text":"Ready"}`
    Documents {
        state: DocState,
        text: String,
    },
    /// The helpers a running tool started (a research report's section writers, the papers it
    /// reads), each as it stands now; the next one replaces it.
    /// `{"kind":"helpers","helpers":[{"task":"Writing Methods","state":"working"}]}`
    Helpers {
        helpers: Vec<ChatHelper>,
    },
}

/// `larch_agent::folio::Documents` on the wire, which has no serde of its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum DocState {
    /// The engine reloads to make room for the documents, about `secs` seconds (0: not known yet).
    MakingRoom { secs: u32 },
    /// The engine has read `done` of `total` tokens of the documents' `pages` (0 without pages).
    Reading { done: u32, total: u32, pages: u32 },
    /// Read, and kept: a question costs only its own tokens.
    Ready,
    /// Read and kept, with `hostLayers` layers of the model in host memory to make room: other
    /// chats are slower while it is.
    FitsSlower { host_layers: u32 },
    /// Too big to read whole: the pages that would fit on this computer.
    FitsUpTo { pages: u32 },
    /// A question was answered: its time to first token and the prompt tokens not read again.
    Answered { ttft_ms: f64, cached_tokens: u32 },
    /// The app let the pin go (the chat sat idle, or another chat needed the room): nothing is
    /// ready, and the next question loads the documents again. The app's own state, which the
    /// agent never reports.
    Released,
}

impl From<larch_agent::folio::Documents> for ChatEvent {
    fn from(d: larch_agent::folio::Documents) -> ChatEvent {
        use larch_agent::folio::Documents as D;
        let state = match d {
            D::Reading { done, total, pages } => DocState::Reading { done, total, pages },
            D::MakingRoom { secs } => DocState::MakingRoom { secs },
            D::Ready => DocState::Ready,
            D::FitsSlower { host_layers } => DocState::FitsSlower { host_layers },
            D::FitsUpTo { pages } => DocState::FitsUpTo { pages },
            D::Answered {
                ttft_ms,
                cached_tokens,
            } => DocState::Answered {
                ttft_ms,
                cached_tokens,
            },
        };
        ChatEvent::Documents {
            state,
            text: d.to_string(),
        }
    }
}

/// One helper of a running tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatHelper {
    /// What it works on, in a few words.
    pub task: String,
    /// `queued` (waiting for a seat), `working`, `done` or `failed`.
    pub state: String,
}

/// A folder's allowances as just saved, by `code_allow` or by a `CodeAsk` answered for a file, a
/// folder or everything: `{"folder":"/home/me/toy","allowed":{"silent":true,"files":[],
/// "folders":[]}}`. `folder` is the real path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderAllowed {
    pub folder: String,
    pub allowed: Allowed,
}

/// One run of a code task's tests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tests {
    /// Tests that passed.
    pub passed: u32,
    /// Tests that failed.
    pub failed: u32,
    /// The failing tests' names, when the output names them.
    pub failing: Vec<String>,
}

/// A path a code run changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeEntry {
    /// Relative to the checkout.
    pub path: String,
    /// Inside the paths the change may touch.
    pub inside: bool,
}

/// What `attach` returns: the chat's attachments now, and the grey line that says it took
/// ("Attached ~/papers-ml (12 files)").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attached {
    pub attached: Vec<String>,
    pub line: String,
}

/// A remembered fact, from larch-agent's memory (`<data>/agent/memory`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fact {
    pub slug: String,
    pub text: String,
    /// Where it came from: a chat and turn, or the desktop app.
    pub source: String,
    /// Unix milliseconds of the fact file's last write.
    pub updated_at: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineSample {
    pub tok_s: f64,
    /// Sequences running now ("voices").
    pub running: u32,
    pub waiting: u32,
    /// Mean queue wait over the last sample window.
    pub queue_wait_ms: f64,
    pub weights_bytes: u64,
    pub kv_used_bytes: u64,
    pub kv_total_bytes: u64,
    /// Measured device memory; `None` where it is not measured (never estimated).
    pub gpu_used_bytes: Option<u64>,
    pub gpu_total_bytes: Option<u64>,
    /// Set while the driver holds the GPU's power limit below the highest seen since load (a thin
    /// laptop's firmware budget), so decode runs slower; `None` otherwise and where not measured.
    pub power_cap: Option<PowerCap>,
}

/// A power limit below the GPU's full one: the SM clock, and so decode speed, drops with it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PowerCap {
    /// The enforced limit now, watts.
    pub limit_w: f32,
    /// The highest enforced limit seen since load, watts.
    pub full_w: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Knob {
    /// The registry id, which gait's records name a knob by.
    pub id: u16,
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    /// `live`, `requestScoped`, `modelReload` or `processRestart`.
    pub activation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum KnobResult {
    Applied {
        rev: u64,
    },
    Rejected {
        reason: String,
    },
    /// Queued, not yet confirmed; its `engine:config` event follows when it lands.
    Pending {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigChange {
    /// Which load of a model made it: revisions number afresh on each engine.
    pub engine: u64,
    pub rev: u64,
    pub knob: String,
    pub old: f64,
    pub new: f64,
    /// `human`, `gait`, `guardrail`, `inverse` …, as the ring records it.
    pub cause: String,
    /// Unix milliseconds.
    pub at: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Failed {
    pub message: String,
}

/// A Hugging Face model repo with GGUF files, as `hf_search` finds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfModel {
    /// `owner/name`.
    pub repo: String,
    pub downloads: u64,
    /// The Hub's tags: architecture hints, licence, languages. Hints only: the header decides.
    pub tags: Vec<String>,
    /// What the Hub read from the repo's main GGUF, so filters work before a header is read here:
    /// its architecture ("" when the Hub has none), parameters and training context.
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub params: Option<u64>,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// larch runs `arch`; the header read decides for the file.
    #[serde(default)]
    pub arch_runs: bool,
}

/// One downloadable GGUF in a Hugging Face repo. Split GGUFs (`name-00001-of-00003.gguf`) are one entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfFile {
    /// Path in the repo of the file (the first part for a split set).
    pub file: String,
    /// Every part, in order; one element for a single file.
    pub parts: Vec<String>,
    /// Exact total bytes from the Hub (sum of parts, and of the projector when there is one).
    pub size_bytes: u64,
    /// Quant type parsed from the file name (e.g. "Q4_K_M", "Q8_0", "F16", "IQ4_XS", "BF16"), or "" if none.
    pub quant: String,
    /// Already in <data>/models (all parts present with the exact size, and the projector).
    pub downloaded: bool,
    /// The repo's vision projector for this model (its path in the repo), downloaded with it and
    /// counted in `size_bytes`; None for a text-only repo.
    #[serde(default)]
    pub mmproj: Option<String>,
}

/// What a remote GGUF holds, read from its header with HTTP range requests (no full download).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteModel {
    pub arch: String,
    /// larch runs this architecture (ModelConfig::from_gguf succeeds).
    pub supported: bool,
    /// The type holding the most tensor bytes, as list_models reports it.
    pub quant: String,
    /// Sum of tensor bytes (what the weights take on the device).
    pub weights_bytes: u64,
    /// Parameter count (sum of tensor element counts).
    pub params: u64,
    /// KV cache bytes per token of context at f16 (2 * n_layer * n_head_kv * head_dim * 2), when the header has the fields.
    pub kv_bytes_per_token: Option<u64>,
    /// The model's trained context (`<arch>.context_length`).
    pub context_length: Option<u64>,
    /// Bytes of the routed experts (`*_exps` tensors), which the engine keeps in host memory
    /// when the model does not fit whole.
    #[serde(default)]
    pub expert_bytes: u64,
    /// The tensor types the engine cannot load, by GGML name ("BF16", "IQ3_S"); empty when it
    /// loads them all.
    #[serde(default)]
    pub unloadable: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Download {
    pub id: String,
    pub repo: String,
    pub file: String,
    pub received_bytes: u64,
    pub total_bytes: u64,
    /// `running`, `done`, `failed` or `cancelled`.
    pub state: String,
}

/// `download:progress`, about 4 a second per running download.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub id: String,
    pub repo: String,
    pub file: String,
    pub received_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_sec: f64,
}

/// `download:done`: the model is in the Hugging Face cache or a library; list_models shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadDone {
    pub id: String,
    pub repo: String,
    pub file: String,
    pub path: String,
    /// Why a download meant for the shared Hugging Face cache went to Larch's own library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// `faster:progress`, about 4 a second while a faster copy is made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FasterProgress {
    /// The original model's path.
    pub path: String,
    /// How far it got, 0 to 1.
    pub fraction: f64,
    /// Seconds since it started.
    pub seconds: f64,
}

/// `faster:done`: the copy is made and passed its check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FasterDone {
    pub path: String,
    pub copy: String,
}

/// `faster:failed`: nothing was kept; `cancelled` true when cancel_faster stopped it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FasterFailed {
    pub path: String,
    pub message: String,
    pub cancelled: bool,
}

/// `download:failed`: `cancelled` true when cancel_download stopped it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadFailed {
    pub id: String,
    pub repo: String,
    pub file: String,
    pub message: String,
    pub cancelled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON the UI's mock.js mirrors.
    #[test]
    fn wire_shapes() {
        let e = ChatEvent::Delta { text: "hi".into() };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"delta","text":"hi"}"#
        );
        let e = ChatEvent::Progress {
            text: "Reading 1 of 4 pages".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"progress","text":"Reading 1 of 4 pages"}"#
        );
        use larch_agent::folio::Documents as D;
        let e = ChatEvent::from(D::reading(1, 3, 120));
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"documents","state":{"kind":"reading","done":1,"total":3,"pages":120},"text":"Reading p. 40 of 120"}"#
        );
        assert_eq!(
            serde_json::to_string(&ChatEvent::from(D::ready(2))).unwrap(),
            r#"{"kind":"documents","state":{"kind":"fitsSlower","hostLayers":2},"text":"Ready; other chats may be slower"}"#
        );
        let e = ChatEvent::from(D::Answered {
            ttft_ms: 1200.,
            cached_tokens: 9,
        });
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"documents","state":{"kind":"answered","ttftMs":1200.0,"cachedTokens":9},"text":"First word in 1.2 s"}"#
        );
        let e = ChatEvent::Documents {
            state: DocState::Released,
            text: "Released".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"documents","state":{"kind":"released"},"text":"Released"}"#
        );
        let e = ChatEvent::Thinking {
            text: "Pin 19 is LFXOUT.".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"thinking","text":"Pin 19 is LFXOUT."}"#
        );
        let e = ChatEvent::Helpers {
            helpers: vec![ChatHelper {
                task: "Writing Methods".into(),
                state: "working".into(),
            }],
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"helpers","helpers":[{"task":"Writing Methods","state":"working"}]}"#
        );
        let e = ChatEvent::Title {
            title: "Two-Player Connect-4 Game".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"title","title":"Two-Player Connect-4 Game"}"#
        );
        let e = ChatEvent::TitleFailed {
            message: "could not name this chat".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"titleFailed","message":"could not name this chat"}"#
        );
        let f = Fit::TooBig {
            need_bytes: 2,
            have_bytes: 1,
        };
        assert_eq!(
            serde_json::to_string(&f).unwrap(),
            r#"{"kind":"tooBig","needBytes":2,"haveBytes":1}"#
        );
        assert_eq!(serde_json::to_string(&Health::Ready).unwrap(), r#""ready""#);
        let tests = |failing: Vec<String>| Tests {
            passed: 1 - failing.len() as u32,
            failed: failing.len() as u32,
            failing,
        };
        let p = ChatEvent::Packet {
            title: "Fix add".into(),
            body: "add adds now".into(),
            diff: "diff --git …".into(),
            tests_before: Some(tests(vec!["test_add".into()])),
            tests_after: Some(tests(vec![])),
            scope: vec![ScopeEntry {
                path: "calc.py".into(),
                inside: true,
            }],
            outside: 0,
            stop: Some("the model said it is done and the tests agree".into()),
            steps: 4,
            parsed: 4,
            checkpoint: Some("larch/checkpoint-fix-add-1".into()),
        };
        assert_eq!(
            serde_json::to_string(&p).unwrap(),
            r#"{"kind":"packet","title":"Fix add","body":"add adds now","diff":"diff --git …","testsBefore":{"passed":0,"failed":1,"failing":["test_add"]},"testsAfter":{"passed":1,"failed":0,"failing":[]},"scope":[{"path":"calc.py","inside":true}],"outside":0,"stop":"the model said it is done and the tests agree","steps":4,"parsed":4,"checkpoint":"larch/checkpoint-fix-add-1"}"#
        );
        let ask = ChatEvent::CodeAsk {
            name: "edit".into(),
            args: serde_json::json!({"path": "C:/p/src/a.py"}),
            path: Some("src/a.py".into()),
            folder: Some("src".into()),
        };
        assert_eq!(
            serde_json::to_string(&ask).unwrap(),
            r#"{"kind":"codeAsk","name":"edit","args":{"path":"C:/p/src/a.py"},"path":"src/a.py","folder":"src"}"#
        );
        let answer: Decision = serde_json::from_str(r#""all""#).unwrap();
        assert_eq!(answer, Decision::All);
        let f = DownloadFailed {
            id: "1".into(),
            repo: "org/model".into(),
            file: "model.gguf".into(),
            message: "no".into(),
            cancelled: true,
        };
        assert_eq!(
            serde_json::to_string(&f).unwrap(),
            r#"{"id":"1","repo":"org/model","file":"model.gguf","message":"no","cancelled":true}"#
        );
        let p = DownloadProgress {
            id: "1".into(),
            repo: "r".into(),
            file: "f".into(),
            received_bytes: 1,
            total_bytes: 2,
            bytes_per_sec: 0.5,
        };
        assert_eq!(
            serde_json::to_string(&p).unwrap(),
            r#"{"id":"1","repo":"r","file":"f","receivedBytes":1,"totalBytes":2,"bytesPerSec":0.5}"#
        );
    }
}

/// How an MCP server is reached, and the name a chat gives its tools (`<server>__<tool>`): the UI
/// builds servers and previews names with these.
pub use larch_agent::mcp::client::{Transport as McpTransport, tool_name as mcp_tool_name};

/// An installed MCP server, as `<data>/mcp.json` keeps it (`mcp_add`). Its tools join every new
/// chat while it is `enabled`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServer {
    /// Short and unique: letters, digits and `-`; the tools' names start with it.
    pub id: String,
    pub name: String,
    /// The registry name it was installed from, or `custom`.
    pub source: String,
    pub transport: larch_agent::mcp::client::Transport,
    pub enabled: bool,
    /// The user trusts its read-only labels: unattended workflow steps may call its read tools.
    /// Off until the user turns it on, since a server labels its own tools.
    #[serde(default)]
    pub trusted: bool,
    /// This install's own identity, never reused: a workflow's grant names it with the id, so a
    /// server installed later under a removed one's id never inherits that one's grants. Set by
    /// `mcp_add`; whatever the caller passes is replaced.
    #[serde(default)]
    pub install: String,
}

/// One tool of a server, as `mcp_test` and `mcp_servers` list it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolInfo {
    /// The server's own name for it (the chat sees `<server id>__<name>`).
    pub name: String,
    pub description: String,
    /// It says it only reads; any other tool asks before it runs.
    pub read_only: bool,
    /// Why no chat takes it (`larch_agent::left_out`): chats open without it.
    #[serde(default)]
    pub left_out: Option<String>,
}

/// A server and its connection, with every env and header value masked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStatus {
    pub id: String,
    pub name: String,
    pub source: String,
    pub transport: larch_agent::mcp::client::Transport,
    pub enabled: bool,
    /// Workflow steps may call its read tools (`McpServer::trusted`).
    pub trusted: bool,
    /// `"off"`, `"connecting"`, `"connected"`, `"needs-sign-in"` or `"error"`.
    pub status: String,
    pub tools: Vec<McpToolInfo>,
    pub error: Option<String>,
    /// A remote server signed in to with OAuth (`mcp_sign_in`).
    pub signed_in: bool,
    /// A remote server without the user's own Authorization header: it can sign in.
    pub can_sign_in: bool,
    /// A sign-in waits for the browser's return or a pasted address (`mcp_sign_in`); once it
    /// fails or ends, the card's sign-in panel closes and Sign in shows again.
    #[serde(default)]
    pub signing: bool,
}

/// A sign-in to start in the browser: the page, and its host (shown before it opens).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpSignIn {
    pub url: String,
    pub host: String,
}

/// A page of the MCP Registry's `/v0/servers`: each entry as the registry sends it
/// (`{server, _meta}`), latest versions only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpSearch {
    pub servers: Vec<serde_json::Value>,
    pub next_cursor: Option<String>,
}

/// `mcp:status`: which server changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpChanged {
    pub id: String,
}
