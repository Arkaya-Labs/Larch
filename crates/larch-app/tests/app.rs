//! The app core on the CPU reference with the generated tiny model (architecture 15).

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, channel},
    },
    time::{Duration, Instant},
};

use larch_agent::runtime::inference::SeatAdmission;
#[cfg(feature = "code-mode")]
use larch_app::contract::{Decision, Tests};
use larch_app::{
    App, Event,
    contract::{
        Allowed, ChatEvent, Fit, FolderAllowed, Health, Intent, KnobResult, Role, SendParams,
    },
};
use larch_kernels::CPU;
use larch_model::{
    Arch,
    gguf::{Gguf, Value, write},
    tiny,
};
use larch_records::{Read, ReadFrom, Record};

const DEADLINE: Duration = Duration::from_secs(30);
/// An agent turn prefills its whole system prompt on the CPU reference in a debug build: ~8 s on
/// a fast desktop, over 30 s on the Windows CI laptop.
const AGENT_DEADLINE: Duration = Duration::from_secs(120);
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-app-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Re-renders the assistant text after a separator token, so a parked turn's tokens are the next
/// prompt's prefix (as tests/facade.rs).
const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";

/// The tiny model with a byte vocabulary and a chat template, under `arch`'s name.
fn tiny_gguf(arch: &str) -> Vec<u8> {
    tiny_gguf_ctx(arch, None)
}

/// [`tiny_gguf`] with its training context set to `ctx`.
fn tiny_gguf_ctx(arch: &str, ctx: Option<u32>) -> Vec<u8> {
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv: Vec<_> = g.kv.into_iter().collect();
    kv.retain(|(k, _)| k != "general.architecture");
    if let Some(ctx) = ctx {
        kv.retain(|(k, _)| k != "llama.context_length");
        kv.push(("llama.context_length".into(), Value::U32(ctx)));
    }
    let tokens = (0..256).map(|b| Value::Str(format!("<0x{b:02X}>")));
    kv.extend([
        ("general.architecture".into(), Value::Str(arch.into())),
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
            Value::Str(TEMPLATE.into()),
        ),
    ]);
    write(&kv, &tensors)
}

/// An app over `dir` with the tiny model in its models folder, and its events. Code mode is on
/// unless `dir` already has settings.json: the code tests run through it.
fn app(dir: &Path) -> (App, Receiver<Event>, String) {
    let settings = dir.join("settings.json");
    if !settings.exists() {
        fs::write(settings, r#"{"code_mode": true}"#).unwrap();
    }
    let (tx, rx) = channel();
    let app = App::new(dir.to_owned(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    // Only this test's folders: not the models other apps keep on the machine running the tests.
    app.set_model_dirs(Vec::new());
    let model = dir.join("models").join("tiny.gguf");
    if !model.exists() {
        fs::write(&model, tiny_gguf("llama")).unwrap();
    }
    let model = model.to_str().unwrap().to_owned();
    (app, rx, model)
}

fn loaded(dir: &Path) -> (App, Receiver<Event>) {
    let (app, rx, model) = app(dir);
    app.load_model(&model).unwrap();
    assert_eq!(app.app_state().health, Health::Ready);
    let health: Vec<_> = rx
        .try_iter()
        .filter_map(|e| match e {
            Event::Health(h) => Some(h),
            _ => None,
        })
        .collect();
    assert_eq!(health, [Health::Loading, Health::Ready]);
    (app, rx)
}

/// Every chat event of `id` up to and including its terminal one.
fn turn(rx: &Receiver<Event>, id: &str) -> Vec<ChatEvent> {
    let mut events = Vec::new();
    let end = Instant::now() + DEADLINE;
    loop {
        let e = rx
            .recv_timeout(end - Instant::now())
            .expect("a terminal chat event");
        if let Event::Chat { id: chat, event } = e {
            assert_eq!(chat, id);
            let last = !matches!(event, ChatEvent::Delta { .. });
            events.push(event);
            if last {
                return events;
            }
        }
    }
}

fn text(events: &[ChatEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::Delta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Whether every one of `seats` comes free within the deadline: nothing kept one.
fn all_free(seats: &SeatAdmission) -> bool {
    let (seats, (tx, rx)) = (seats.clone(), channel());
    std::thread::spawn(move || {
        let class = larch::api::Class::Background;
        let taken = seats.take(seats.capacity(), class, &Default::default());
        let _ = tx.send(taken.is_ok());
    });
    rx.recv_timeout(DEADLINE).unwrap_or(false)
}

fn params(max_tokens: u32) -> SendParams {
    SendParams {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
    }
}

#[test]
fn chat_send_done_resume() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(Some("be brief".into())).unwrap();
    app.send(&id, "hello", params(8)).unwrap();
    let events = turn(&rx, &id);
    let Some(ChatEvent::Done {
        completion_tokens, ..
    }) = events.last()
    else {
        panic!("{events:?}")
    };
    assert_eq!(*completion_tokens, 8);
    let reply = text(&events);
    let chats = app.list_chats().unwrap();
    assert_eq!(chats.len(), 1);
    assert_eq!((chats[0].id.as_str(), chats[0].turns), (id.as_str(), 1));
    assert_eq!(chats[0].title, "hello");
    drop(app);

    // A restart: the saved transcript restores the history.
    let (app, _rx) = loaded(&dir.0);
    let history = app.open_chat(&id).unwrap();
    let got: Vec<_> = history
        .iter()
        .map(|m| (m.role, m.content.as_str(), m.truncated))
        .collect();
    assert_eq!(
        got,
        [
            (Role::System, "be brief", false),
            (Role::User, "hello", false),
            (Role::Assistant, reply.as_str(), false)
        ]
    );
    // The user's name wins over the first message's and outlives a restart; a blank one is refused.
    assert!(app.rename_chat(&id, "  ").is_err());
    app.rename_chat(&id, " Greetings ").unwrap();
    drop(app);
    let (app, _rx) = loaded(&dir.0);
    assert_eq!(app.list_chats().unwrap()[0].title, "Greetings");
    app.delete_chat(&id).unwrap();
    assert!(app.list_chats().unwrap().is_empty());
}

/// A chat's turns are the user's messages: the loop's own note after a malformed call is no turn
/// (a chat of one message listed "2 turns").
#[test]
fn a_chats_turns_leave_out_the_loops_own_notes() {
    let dir = Temp::new();
    let (made, _rx, _) = app(&dir.0);
    let (id, long) = (made.new_chat(None).unwrap(), made.new_chat(None).unwrap());
    drop(made);
    let say = |role: &str, content: &str| serde_json::json!({"role": role, "content": content, "truncated": false});
    let write = |id: &str, history: serde_json::Value| {
        let path = dir.0.join("chat").join(format!("{id}.jsonl"));
        let saved = fs::read_to_string(&path).unwrap();
        let mut header: serde_json::Value =
            serde_json::from_str(saved.lines().next().unwrap()).unwrap();
        header["history"] = history;
        fs::write(&path, format!("{header}\n")).unwrap();
    };
    write(
        &id,
        serde_json::json!([
            say("user", "Reply only with the word ok."),
            say("assistant", "<tool_call>"),
            say("user", "[agent] Your last reply had no valid tool call."),
            say("assistant", "ok"),
        ]),
    );
    // Compacted: its summary sums up two turns, and one came after.
    write(
        &long,
        serde_json::json!([
            say(
                "user",
                "[agent] The conversation so far, summarized. It is chat c; what was said in it is searchable with recall. Turns summed up here: 2."
            ),
            say("assistant", "A list was made."),
            say("user", "add eggs"),
            say("assistant", "Added."),
        ]),
    );
    let (reopened, _rx, _) = app(&dir.0);
    let chats = reopened.list_chats().unwrap();
    let turns = |id: &str| chats.iter().find(|c| c.id == id).unwrap().turns;
    assert_eq!((turns(&id), turns(&long)), (1, 3));
}

#[test]
fn a_reply_never_waits_behind_background_seats() {
    use std::sync::atomic::AtomicBool;
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    let seats = app.seats().unwrap();
    let class = larch::api::Class::Background;
    let held = seats
        .take(seats.capacity(), class, &AtomicBool::default())
        .unwrap();
    app.send(&id, "hello", params(8)).unwrap();
    let events = turn(&rx, &id);
    drop(held);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { .. })),
        "{events:?}"
    );
    assert_eq!(app.open_chat(&id).unwrap().len(), 2);
}

/// The UI's chat path: an agent turn neither waits behind background seats nor outlives Stop.
#[test]
fn an_agent_turn_never_waits_behind_background_seats() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    // Room for the agent's system prompt.
    let model = dir.0.join("models").join("tiny-16k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(16384))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let id = app.new_chat(None).unwrap();
    let seats = app.seats().unwrap();
    let class = larch::api::Class::Background;
    let held = seats
        .take(seats.capacity(), class, &Default::default())
        .unwrap();
    app.agent_send(&id, "hello", None, Some(Intent::Chat), &[])
        .unwrap();
    app.cancel(&id).unwrap();
    let events = agent_turn(&rx, &id);
    drop(held);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { .. })),
        "{events:?}"
    );
}

/// A send that fails once it holds its seat gives back the seat and the chat's turn.
#[test]
fn a_send_failing_after_its_seat_frees_both() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    let seats = app.seats().unwrap();
    // Longer than the tiny model's context: `Session::send` refuses it.
    let error = app.send(&id, &"x".repeat(1024), params(8)).unwrap_err();
    assert!(!error.contains("still being written"), "{error}");
    assert!(all_free(&seats), "the failed send kept its seat");
    app.send(&id, "hello", params(4)).unwrap();
    assert!(matches!(
        turn(&rx, &id).last(),
        Some(ChatEvent::Done { .. })
    ));
    assert!(all_free(&seats), "a finished reply gives its seat back");
}

#[test]
fn cancel_keeps_partial_text_truncated() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    app.send(&id, "hello", params(400)).unwrap();
    // One reply per chat at a time.
    let busy = app.send(&id, "again", params(4)).unwrap_err();
    assert!(busy.contains("still being written"), "{busy}");
    // Some text first, so the kept partial is not trivially empty.
    let first = loop {
        if let Event::Chat { event, .. } = rx.recv_timeout(DEADLINE).unwrap() {
            break event;
        }
    };
    assert!(matches!(&first, ChatEvent::Delta { text } if !text.is_empty()));
    app.cancel(&id).unwrap();
    let mut events = vec![first];
    events.extend(turn(&rx, &id));
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { finish, .. }) if finish == "cancelled"),
        "{events:?}"
    );
    let history = app.open_chat(&id).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[1].content, text(&events));
    assert!(history[1].truncated);
    drop(app);
    let (app, _rx) = loaded(&dir.0);
    assert_eq!(app.open_chat(&id).unwrap(), history);
}

#[test]
fn second_turn_reuses_cached_tokens() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    let cached = |text: &str| {
        app.send(&id, text, params(4)).unwrap();
        match turn(&rx, &id).pop() {
            Some(ChatEvent::Done { cached_tokens, .. }) => cached_tokens,
            e => panic!("{e:?}"),
        }
    };
    assert_eq!(cached("hello"), 0);
    assert!(cached("more") > 0);
}

#[test]
fn telemetry_matches_the_ring() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let ring = app.ring().unwrap();
    let id = app.new_chat(None).unwrap();
    // Long enough to span several 250 ms samples.
    app.send(&id, "hello", params(400)).unwrap();
    let mut samples = Vec::new();
    let end = Instant::now() + DEADLINE;
    // The turn, then a quiet second, so the last sample reports the engine at rest.
    let mut done = None;
    while done.is_none_or(|t: Instant| t.elapsed() < Duration::from_secs(1)) {
        match rx.recv_timeout(end - Instant::now()).unwrap() {
            Event::Sample(s) => samples.push(s),
            Event::Chat {
                event: ChatEvent::Done { .. },
                ..
            } => done = Some(Instant::now()),
            _ => {}
        }
    }
    let (mut deployment, mut last) = (None, None);
    let mut reader = ring.reader(ReadFrom::Oldest);
    loop {
        match reader.next() {
            Read::Record(Record::DeploymentSnapshot(_, d)) => deployment = Some(d),
            Read::Record(Record::EngineSample(_, s)) => last = Some(s),
            Read::Record(_) => {}
            Read::Empty => break,
            Read::Lagged { .. } => panic!("lagged"),
        }
    }
    let (d, last) = (deployment.unwrap(), last.unwrap());
    let block = 2
        * u64::from(d.n_layers)
        * u64::from(last.kv_block_tokens)
        * u64::from(d.n_kv_heads)
        * u64::from(d.head_dim)
        * 2;
    assert!(samples.iter().any(|s| s.tok_s > 0.0 && s.running == 1));
    let rest = samples.last().unwrap();
    assert_eq!((rest.tok_s, rest.running, rest.waiting), (0.0, 0, 0));
    assert_eq!(rest.weights_bytes, d.weights_bytes);
    assert_eq!(rest.kv_used_bytes, u64::from(last.kv_blocks_used) * block);
    assert_eq!(rest.kv_total_bytes, u64::from(last.kv_blocks_total) * block);
    // The CPU reference measures no device memory: none is reported, none invented.
    assert_eq!((rest.gpu_used_bytes, rest.gpu_total_bytes), (None, None));
    assert_eq!(rest.power_cap, None);

    // A knob change reaches the timeline, and reverting it does too.
    let knob = app
        .knobs()
        .unwrap()
        .into_iter()
        .find(|k| k.name == "max_batch_seqs")
        .unwrap();
    let KnobResult::Applied { rev } = app.knob_set("max_batch_seqs", 4.0).unwrap() else {
        panic!("not applied")
    };
    let config = |rx: &Receiver<Event>| loop {
        if let Event::Config(c) = rx.recv_timeout(DEADLINE).unwrap() {
            return c;
        }
    };
    let c = config(&rx);
    assert_eq!((c.rev, c.knob.as_str()), (rev, "max_batch_seqs"));
    assert_eq!((c.old, c.new, c.cause.as_str()), (knob.value, 4.0, "human"));
    assert!(matches!(
        app.knob_revert(c.engine, rev).unwrap(),
        KnobResult::Applied { .. }
    ));
    let c = config(&rx);
    assert_eq!((c.new, c.cause.as_str()), (knob.value, "inverse"));
    let rejected = app.knob_set("max_batch_seqs", 1e9).unwrap();
    assert!(
        matches!(rejected, KnobResult::Rejected { .. }),
        "{rejected:?}"
    );
    assert!(app.knob_set("no_such_knob", 1.0).is_err());
}

#[test]
fn undo_after_a_reload_never_reverts_the_new_engines_change() {
    let dir = Temp::new();
    let (app, rx, model) = app(&dir.0);
    let config = || loop {
        if let Event::Config(c) = rx.recv_timeout(DEADLINE).unwrap() {
            return c;
        }
    };
    let set = |name: &str, v: f64| {
        let KnobResult::Applied { rev } = app.knob_set(name, v).unwrap() else {
            panic!("not applied")
        };
        let c = config();
        assert_eq!((c.rev, c.knob.as_str()), (rev, name));
        c
    };
    let value = |name: &str| {
        app.knobs()
            .unwrap()
            .into_iter()
            .find(|k| k.name == name)
            .unwrap()
            .value
    };
    app.load_model(&model).unwrap();
    let a = set("max_batch_seqs", 4.0);
    app.load_model(&model).unwrap();
    let b = set("prefill_chunk_tokens", 256.0);
    // The new engine numbers its changes afresh, so only the engine tells them apart.
    assert_eq!(a.rev, b.rev);
    assert_ne!(a.engine, b.engine);
    let stale = app.knob_revert(a.engine, a.rev).unwrap();
    assert!(matches!(stale, KnobResult::Rejected { .. }), "{stale:?}");
    assert_eq!(value("prefill_chunk_tokens"), 256.0);
    assert!(matches!(
        app.knob_revert(b.engine, b.rev).unwrap(),
        KnobResult::Applied { .. }
    ));
    assert_eq!(value("prefill_chunk_tokens"), b.old);
}

#[test]
fn model_list_marks_unsupported_and_skips_other_files() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let models = dir.0.join("models");
    fs::write(models.join("other.gguf"), tiny_gguf("gemma3")).unwrap();
    fs::write(models.join("notes.txt"), "not a model").unwrap();
    fs::write(models.join("broken.gguf"), "not a model either").unwrap();
    let list = app.list_models().unwrap();
    let got: Vec<_> = list
        .iter()
        .map(|m| (m.name.as_str(), m.arch.as_str(), m.supported))
        .collect();
    assert_eq!(got, [("other", "gemma3", false), ("tiny", "llama", true)]);
    let tiny = &list[1];
    let size = fs::metadata(models.join("tiny.gguf")).unwrap().len();
    assert_eq!(tiny.size_bytes, size);
    assert_eq!(list[0].fit, Fit::Unknown);
    // ponytail: BLOCKED on #123; the stand-in fit check answers Unknown. Assert `Fits` once it lands.
    assert_eq!(tiny.fit, Fit::Unknown);
    assert!(!tiny.quant.is_empty());
    assert!(app.load_model(&list[0].path).is_err());
    assert_eq!(app.app_state().health, Health::Empty);
    let health: Vec<_> = rx.try_iter().collect();
    assert_eq!(
        health,
        [Event::Health(Health::Loading), Event::Health(Health::Empty)]
    );
}

#[test]
fn model_list_finds_other_apps_models_in_place() {
    let dir = Temp::new();
    let (app, _rx, _) = app(&dir.0);
    let other = dir.0.join("elsewhere");
    let studio = other.join("lmstudio");
    let repo = studio.join("pub").join("repo-GGUF");
    fs::create_dir_all(&repo).unwrap();
    fs::write(repo.join("studio.gguf"), tiny_gguf("qwen3")).unwrap();
    fs::write(repo.join("split-00001-of-00002.gguf"), tiny_gguf("llama")).unwrap();
    fs::write(repo.join("split-00002-of-00002.gguf"), tiny_gguf("llama")).unwrap();
    fs::write(repo.join("mmproj-f16.gguf"), tiny_gguf("llama")).unwrap();
    // Three levels down is past LM Studio's depth.
    let deep = repo.join("a").join("b");
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("deep.gguf"), tiny_gguf("llama")).unwrap();
    let ollama = other.join("ollama");
    let manifest = ollama.join("manifests/registry.ollama.ai/library/tiny");
    fs::create_dir_all(&manifest).unwrap();
    fs::create_dir_all(ollama.join("blobs")).unwrap();
    fs::write(ollama.join("blobs/sha256-ab12"), tiny_gguf("llama")).unwrap();
    fs::write(ollama.join("blobs/sha256-cd34"), "a licence").unwrap();
    fs::write(
        manifest.join("latest"),
        r#"{"layers":[{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:cd34"},
            {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:ab12"}]}"#,
    )
    .unwrap();
    app.set_model_dirs(vec![
        ("LM Studio".into(), studio.clone(), 2),
        ("Ollama".into(), ollama.clone(), 0),
        // The same folder twice: each file is listed once.
        ("LM Studio".into(), studio, 2),
    ]);
    let got: Vec<_> = app
        .list_models()
        .unwrap()
        .into_iter()
        .map(|m| {
            let file = Path::new(&m.path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (m.source, file, m.arch)
        })
        .collect();
    let row = |s: &str, f: &str, a: &str| (s.to_owned(), f.to_owned(), a.to_owned());
    assert_eq!(
        got,
        [
            row("Larch", "tiny.gguf", "llama"),
            row("LM Studio", "split-00001-of-00002.gguf", "llama"),
            row("LM Studio", "studio.gguf", "qwen3"),
            row("Ollama", "sha256-ab12", "llama"),
        ]
    );
}

/// The fault runs in a child process: LARCH_FAULT is read once per engine thread.
#[test]
fn fault_sends_engine_failed() {
    if std::env::var_os("LARCH_APP_FAULT_CHILD").is_some() {
        return fault_child();
    }
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fault_sends_engine_failed", "--nocapture"])
        .env("LARCH_APP_FAULT_CHILD", "1")
        .env("LARCH_FAULT", "device_lost@step:3")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("fault_child_verified"), "{stderr}");
}

fn fault_child() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    app.send(&id, "hello", params(32)).unwrap();
    let (mut failed, mut chat_error, mut health) = (None, None, None);
    while failed.is_none() || chat_error.is_none() || health.is_none() {
        match rx.recv_timeout(DEADLINE).unwrap() {
            Event::Failed(f) => failed = Some(f.message),
            Event::Health(h) => health = Some(h),
            Event::Chat {
                event: ChatEvent::Error { message },
                ..
            } => chat_error = Some(message),
            _ => {}
        }
    }
    let failed = failed.unwrap();
    assert!(failed.contains("Restart Larch"), "{failed}");
    assert_eq!(health, Some(Health::Failed));
    assert_eq!(app.app_state().health, Health::Failed);
    // Stop still reaches the chat's reply or turn after the fault.
    app.cancel(&id).unwrap();
    for result in [
        app.list_chats().map(drop),
        app.list_models().map(drop),
        app.new_chat(None).map(drop),
        app.open_chat(&id).map(drop),
        app.send(&id, "again", params(4)),
        app.knobs().map(drop),
        app.load_model("x.gguf"),
    ] {
        assert!(result.unwrap_err().contains("Restart Larch"));
    }
    eprintln!("fault_child_verified");
}

/// The shell's device loads a model on this build; GPU 0 alone is refused on a CPU-only one.
#[test]
fn default_device_loads_a_model() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_app::default_device().unwrap(), |_| {}).unwrap();
    let model = dir.0.join("models").join("tiny.gguf");
    fs::write(&model, tiny_gguf("llama")).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    assert_eq!(app.app_state().health, Health::Ready);
}

/// Every chat event of `id` until an agent turn ends: `Done`, `Error` or `Confirm`.
fn agent_turn(rx: &Receiver<Event>, id: &str) -> Vec<ChatEvent> {
    until(rx, id, |e| {
        matches!(
            e,
            ChatEvent::Done { .. } | ChatEvent::Error { .. } | ChatEvent::Confirm { .. }
        )
    })
}

/// Every chat event of `id` up to and including the one `last` picks.
fn until(rx: &Receiver<Event>, id: &str, last: impl Fn(&ChatEvent) -> bool) -> Vec<ChatEvent> {
    let mut events = Vec::new();
    let end = Instant::now() + AGENT_DEADLINE;
    loop {
        let e = rx
            .recv_timeout(end - Instant::now())
            .expect("an agent turn's end");
        if let Event::Chat { id: chat, event } = e {
            assert_eq!(chat, id);
            let last = last(&event);
            events.push(event);
            if last {
                return events;
            }
        }
    }
}

/// unsloth's repos ship one `mmproj-F16.gguf` beside several quants: copied into one folder, every
/// quant of the one model reads images, and the projector goes with the last of them.
#[test]
fn a_generic_projector_serves_every_quant_of_its_model() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    app.set_model_dirs(Vec::new());
    let models = dir.0.join("models");
    let model = tiny::gguf(Arch::Llama, 3);
    fs::write(models.join("InternVL3_5-1B-Q8_0.gguf"), &model).unwrap();
    fs::write(models.join("InternVL3_5-1B-Q4_K_M.gguf"), &model).unwrap();
    fs::write(
        models.join("mmproj-F16.gguf"),
        tiny::Projector::new(64, 5).gguf(),
    )
    .unwrap();
    let list = app.list_models().unwrap();
    assert_eq!(list.len(), 2);
    assert!(list.iter().all(|m| m.vision), "{list:?}");
    let projector = models.join("mmproj-F16.gguf");
    app.delete_model(&list[0].path).unwrap();
    assert!(projector.exists(), "the other quant still reads with it");
    app.delete_model(&list[1].path).unwrap();
    assert!(!projector.exists());
}

/// Unsent images past the bytes kept for drafts push out the oldest, and the add that did it names
/// them, so the UI drops their chips instead of failing the send later.
#[test]
fn an_image_that_pushes_out_an_older_one_names_it() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    let png = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../larch-model/tests/data/vision/palette.png"),
    )
    .unwrap();
    // 30 MB each (the bytes after IEND are kept, not read): 9 of them pass the 256 MiB budget.
    let mut refs = Vec::new();
    let mut last = None;
    for i in 0..9u8 {
        let mut big = png.clone();
        big.resize(30 << 20, i);
        let info = app.image_add(&big).unwrap();
        refs.push(info.reference.clone());
        last = Some(info);
    }
    assert_eq!(last.unwrap().evicted, refs[..1]);
}

/// A projector beside the model that does not load with it (another model's width): the model loads
/// alone, the app says why, and the listing stops saying the model reads images.
#[test]
fn a_projector_that_does_not_load_says_why() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    app.set_model_dirs(Vec::new());
    let models = dir.0.join("models");
    let model = models.join("tinyvl-Q8_0.gguf");
    let bytes = tiny_gguf_ctx("llama", Some(16384));
    let d = match Gguf::parse(&bytes)
        .unwrap()
        .kv
        .get("llama.embedding_length")
    {
        Some(Value::U32(d)) => *d,
        v => panic!("embedding length: {v:?}"),
    };
    fs::write(&model, bytes).unwrap();
    fs::write(
        models.join("mmproj-tinyvl-f16.gguf"),
        tiny::Projector::new(d + 8, 5).gguf(),
    )
    .unwrap();
    let path = model.to_str().unwrap().to_owned();
    let listed = |app: &App| app.list_models().unwrap()[0].vision;
    assert!(listed(&app), "a projector by its header, before any load");
    app.load_model(&path).unwrap();
    let state = app.app_state();
    assert!(!state.vision);
    let why = state.vision_error.expect("the reason");
    assert!(!why.is_empty());
    assert!(!listed(&app));
    let png = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../larch-model/tests/data/vision/palette.png"),
    )
    .unwrap();
    let info = app.image_add(&png).unwrap();
    let id = app.new_chat(None).unwrap();
    let refused = app.agent_send(
        &id,
        "what is this?",
        None,
        Some(Intent::Chat),
        &[info.reference],
    );
    assert!(refused.unwrap_err().contains(&why));
}

/// A chat with a project folder works in it: the agent's file tools start there and its system
/// prompt lists the folder's files, so "read index.html" is the project's. The chat opened
/// before without the folder reopens on it, its history kept.
/// Image paste end to end on the CPU: a projector beside the model turns images on; an image is taken
/// as it is added, goes with its message, stays with the chat (its reference, its bytes), and goes
/// with the chat. Without a projector, images are refused before any step.
#[test]
fn images_go_with_a_message_and_stay_with_the_chat() {
    let dir = Temp::new();
    let (tx, rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    app.set_model_dirs(Vec::new());
    let models = dir.0.join("models");
    let model = models.join("tinyvl-Q8_0.gguf");
    let bytes = tiny_gguf_ctx("llama", Some(16384));
    let d = match Gguf::parse(&bytes)
        .unwrap()
        .kv
        .get("llama.embedding_length")
    {
        Some(Value::U32(d)) => *d,
        v => panic!("embedding length: {v:?}"),
    };
    fs::write(&model, bytes).unwrap();
    let png = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../larch-model/tests/data/vision/palette.png"),
    )
    .unwrap();
    let model = model.to_str().unwrap().to_owned();
    app.load_model(&model).unwrap();
    assert!(!app.app_state().vision);
    let info = app.image_add(&png).unwrap();
    assert_eq!(info.tokens, None);
    let id = app.new_chat(None).unwrap();
    let refused = app.agent_send(
        &id,
        "what is this?",
        None,
        Some(Intent::Chat),
        &[info.reference],
    );
    assert!(refused.unwrap_err().contains("can't see images"));

    fs::write(
        models.join("mmproj-tinyvl-f16.gguf"),
        tiny::Projector::new(d, 5).gguf(),
    )
    .unwrap();
    app.load_model(&model).unwrap();
    assert!(app.app_state().vision);
    assert!(app.image_add(b"not an image").is_err());
    let info = app.image_add(&png).unwrap();
    assert_eq!(
        info.tokens,
        Some(4),
        "one 56-pixel tile of the generated projector"
    );
    let id = app.new_chat(None).unwrap();
    let images = [info.reference.clone()];
    app.agent_send(&id, "what is this?", None, Some(Intent::Chat), &images)
        .unwrap();
    app.cancel(&id).unwrap();
    agent_turn(&rx, &id);
    // Sent, the image is the chat's: it no longer waits for a message.
    let again = app.agent_send(&id, "and this?", None, Some(Intent::Chat), &images);
    assert!(again.unwrap_err().contains("gone"));
    let history = app.open_chat(&id).unwrap();
    let user = history.iter().find(|m| m.role == Role::User).unwrap();
    assert_eq!(user.images, images);
    assert_eq!(app.chat_image(&id, &info.reference).unwrap(), png);
    assert!(app.chat_image(&id, "../../etc/passwd").is_err());
    // A fresh engine (its cache empty): reopening the chat encodes its image before any turn.
    app.load_model(&model).unwrap();
    assert_eq!(app.image_tiles_encoded(), 0);
    app.open_chat(&id).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while app.image_tiles_encoded() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the reopened chat's image was not encoded"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    app.delete_chat(&id).unwrap();
    assert!(app.chat_image(&id, &info.reference).is_err());
}

#[test]
fn an_agent_turn_in_a_project_folder_starts_its_file_tools_there() {
    let dir = Temp::new();
    let (tx, rx) = channel();
    // A turn's end is held after it is sent, as a busy UI thread holds it: the chat is free by
    // then, so the next send, on another folder, is not refused as busy.
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let end = matches!(
            &e,
            Event::Chat {
                event: ChatEvent::Done { .. } | ChatEvent::Error { .. },
                ..
            }
        );
        let _ = tx.send(e);
        if end {
            std::thread::sleep(Duration::from_millis(300));
        }
    })
    .unwrap();
    app.set_model_dirs(Vec::new());
    // Room for both turns, whose replies are the tiny model's noise.
    let model = dir.0.join("models").join("tiny-16k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(16384))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let work = Temp::new();
    let project = work.0.join("game");
    fs::create_dir(&project).unwrap();
    fs::write(project.join("connect4.js"), "").unwrap();
    let id = app.new_chat(None).unwrap();
    let system = || {
        let history = app.open_chat(&id).unwrap();
        let system = history.into_iter().find(|m| m.role == Role::System);
        system.map(|m| m.content).unwrap_or_default()
    };
    // Each turn is stopped at its first token: what the test reads is saved when the agent opens.
    // A whole reply of the tiny model's noise, read again as history, pushes the folder's turn
    // past the context: minutes of prefill on the Windows CI laptop.
    app.agent_send(&id, "hello", None, Some(Intent::Chat), &[])
        .unwrap();
    app.cancel(&id).unwrap();
    agent_turn(&rx, &id);
    assert!(!system().contains("connect4.js"));
    (app.agent_send(
        &id,
        "what is in it?",
        Some(&project),
        Some(Intent::Chat),
        &[],
    ))
    .unwrap();
    app.cancel(&id).unwrap();
    agent_turn(&rx, &id);
    let prompt = system();
    assert!(prompt.contains("connect4.js"), "{prompt}");
    let history = app.open_chat(&id).unwrap();
    assert!(history.iter().any(|m| m.content == "hello"));
}

#[test]
fn agent_turn_streams_and_saves() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    // The agent's system prompt (its tools and rules) needs more than the tiny model's 512 tokens.
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let id = app.new_chat(None).unwrap();
    app.agent_send(&id, "hello", None, None, &[]).unwrap();
    // One turn per chat at a time, agent or plain.
    assert_eq!(
        app.agent_send(&id, "again", None, None, &[]).unwrap_err(),
        "a reply is still being written in this chat"
    );
    let events = agent_turn(&rx, &id);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { .. })),
        "{events:?}"
    );
    let reply = text(&events);
    // The first answer names the chat after its end, in the title's words.
    let named = until(&rx, &id, |e| {
        matches!(e, ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. })
    });
    let Some(ChatEvent::Title { title }) = named.last() else {
        panic!("{named:?}")
    };
    assert!(
        (title.split(' ').count() <= 6) && title.starts_with(char::is_alphanumeric),
        "{title}"
    );
    // Listed by that name and reopened from the agent's history while it stays open.
    let chats = app.list_chats().unwrap();
    assert_eq!(
        (
            chats[0].id.as_str(),
            chats[0].turns,
            chats[0].title.as_str()
        ),
        (id.as_str(), 1, title.as_str())
    );
    let history = app.open_chat(&id).unwrap();
    assert!(
        history
            .iter()
            .any(|m| m.role == Role::User && m.content == "hello")
    );
    let last = history.last().unwrap();
    assert_eq!(
        (last.role, last.content.as_str()),
        (Role::Assistant, reply.as_str())
    );
    // Nothing waits for a confirm, so confirming ends the step with the agent's error.
    app.agent_confirm(&id, true).unwrap();
    let events = agent_turn(&rx, &id);
    assert_eq!(
        events,
        [ChatEvent::Error {
            message: "no write waits for a confirm".into()
        }]
    );
    // The name goes with the chat.
    let saved = dir.0.join("chat").join(format!("{id}.title"));
    assert!(saved.exists());
    app.delete_chat(&id).unwrap();
    assert!(!saved.exists());
}

/// A chat whose first answer was stopped is named after its next answer, from its first message.
#[test]
fn a_stopped_first_turn_is_named_after_the_next_answer() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let id = app.new_chat(None).unwrap();
    app.agent_send(&id, "hello", None, Some(Intent::Chat), &[])
        .unwrap();
    app.cancel(&id).unwrap();
    let events = agent_turn(&rx, &id);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { finish, .. }) if finish == "cancelled"),
        "{events:?}"
    );
    assert!(!events.iter().any(|e| matches!(e, ChatEvent::Title { .. })));
    (app.agent_send(&id, "again", None, Some(Intent::Chat), &[])).unwrap();
    agent_turn(&rx, &id);
    let named = until(&rx, &id, |e| {
        matches!(e, ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. })
    });
    let Some(ChatEvent::Title { title }) = named.last() else {
        panic!("{named:?}")
    };
    assert_eq!(app.list_chats().unwrap()[0].title, *title);
}

/// A name that cannot be saved leaves no file behind, so a later answer names the chat; the user
/// is told once, not after every answer. The half-written file's place is taken by a folder,
/// which makes the save fail.
#[test]
fn a_name_that_cannot_be_saved_is_tried_again_after_the_next_answer() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let id = app.new_chat(None).unwrap();
    let saved = dir.0.join("chat").join(format!("{id}.title"));
    let part = dir.0.join("chat").join(format!("{id}.title.part"));
    fs::create_dir(&part).unwrap();
    app.agent_send(&id, "hello", None, Some(Intent::Chat), &[])
        .unwrap();
    agent_turn(&rx, &id);
    let named = until(&rx, &id, |e| {
        matches!(e, ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. })
    });
    assert!(
        matches!(named.last(), Some(ChatEvent::TitleFailed { .. })),
        "{named:?}"
    );
    assert!(!saved.exists());
    assert_eq!(app.list_chats().unwrap()[0].title, "hello");
    // The same failure after the next answer is retried quietly: the user was told once. The
    // tiny model names a chat in well under the wait.
    (app.agent_send(&id, "still", None, Some(Intent::Chat), &[])).unwrap();
    agent_turn(&rx, &id);
    let end = Instant::now() + Duration::from_secs(3);
    while let Ok(e) = rx.recv_timeout(end.saturating_duration_since(Instant::now())) {
        assert!(
            !matches!(
                e,
                Event::Chat {
                    event: ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. },
                    ..
                }
            ),
            "{e:?}"
        );
    }
    assert!(!saved.exists());
    fs::remove_dir(&part).unwrap();
    (app.agent_send(&id, "again", None, Some(Intent::Chat), &[])).unwrap();
    agent_turn(&rx, &id);
    let named = until(&rx, &id, |e| {
        matches!(e, ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. })
    });
    let Some(ChatEvent::Title { title }) = named.last() else {
        panic!("{named:?}")
    };
    assert_eq!(fs::read_to_string(&saved).unwrap(), *title);
    assert!(!part.exists());
}

/// A chat deleted while it is being named keeps no name on disk: the job is stopped, and one
/// that ends after the delete writes nothing. Reloading the model waits for the job to end.
#[test]
fn a_chat_deleted_while_it_is_named_leaves_no_title() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    let model = model.to_str().unwrap();
    app.load_model(model).unwrap();
    let id = app.new_chat(None).unwrap();
    app.agent_send(&id, "hello", None, Some(Intent::Chat), &[])
        .unwrap();
    let events = agent_turn(&rx, &id);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { .. })),
        "{events:?}"
    );
    // The job is registered before Done is sent, so it is running or about to.
    app.delete_chat(&id).unwrap();
    app.load_model(model).unwrap();
    assert!(!dir.0.join("chat").join(format!("{id}.title")).exists());
    assert!(app.list_chats().unwrap().is_empty());
}

/// Turning "keep documents ready" off deletes the pin files already kept; on keeps them
/// (#590 F10).
#[test]
fn keep_ready_off_deletes_the_kept_pins() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    let cache = larch_agent::folio::cache::dir(&dir.0);
    fs::create_dir_all(&cache).unwrap();
    let pin = cache.join("k.pin");
    fs::write(&pin, b"pin").unwrap();
    app.set_docs_keep_ready(true).unwrap();
    assert!(pin.exists());
    app.set_docs_keep_ready(false).unwrap();
    assert!(!pin.exists());
}

#[test]
fn memory_add_list_forget() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    assert!(app.memory_list().unwrap().is_empty());
    let added = app.memory_add("Mornings are for focus work").unwrap();
    assert_eq!(added.old, None);
    let slug = added.slug;
    let facts = app.memory_list().unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(
        (
            facts[0].slug.as_str(),
            facts[0].text.as_str(),
            facts[0].source.as_str()
        ),
        (
            slug.as_str(),
            "Mornings are for focus work",
            "the desktop app"
        )
    );
    assert!(facts[0].updated_at > 0);
    let gone = app.memory_forget(&slug).unwrap();
    assert!(app.memory_list().unwrap().is_empty());
    // Undo of the Forget puts back the same fact: its slug, type, source and created.
    app.memory_restore(&gone).unwrap();
    let back = larch_agent::memory::list(&dir.0).unwrap();
    assert_eq!(
        (
            back[0].slug.as_str(),
            back[0].kind.as_str(),
            back[0].source.as_str()
        ),
        (slug.as_str(), "user", "the desktop app")
    );
    assert_eq!(back[0].created, gone.created);
    // A second Undo finds the slug taken and says so.
    assert!(
        app.memory_restore(&gone)
            .unwrap_err()
            .contains("already in memory")
    );
}

#[test]
fn undo_of_a_forget_restores_a_captured_fact_as_it_was() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    let data = &dir.0;
    let said = ["I like green tea a lot.".to_string()];
    let item = r#"{"facts":[{"text":"The user likes green tea.","quote":"I like green tea","type":"preference","replaces":null}]}"#;
    larch_agent::memory::capture::save(data, item, false, &said, &[], "chat 7 turn 1").unwrap();
    let f = larch_agent::memory::list(data).unwrap().remove(0);
    let gone = app.memory_forget(&f.slug).unwrap();
    app.memory_restore(&gone).unwrap();
    let back = larch_agent::memory::list(data).unwrap().remove(0);
    assert_eq!(
        (
            &back.slug,
            &back.text,
            &back.kind,
            &back.source,
            &back.created
        ),
        (&f.slug, &f.text, &f.kind, &f.source, &f.created)
    );
    assert_eq!(back.source, "chat 7 turn 1");
}

#[test]
fn an_edit_changes_the_sentence_alone_and_refuses_a_fact_changed_since() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    let data = &dir.0;
    let said = ["I like green tea a lot.".to_string()];
    let item = r#"{"facts":[{"text":"The user likes green tea.","quote":"I like green tea","type":"user","replaces":null}]}"#;
    larch_agent::memory::capture::save(data, item, false, &said, &[], "chat 7").unwrap();
    let f = app.memory_list().unwrap().remove(0);
    let md = || fs::read_to_string(data.join("agent/memory/MEMORY.md")).unwrap();
    let (rev, before) = (larch_agent::memory::rev(data), md());
    // Save with nothing changed: no write, no rev.
    assert!(
        !app.memory_update(&f.slug, &f.text, "The user likes green tea.")
            .unwrap()
    );
    assert_eq!((larch_agent::memory::rev(data), md()), (rev, before));
    // A new sentence keeps the user's words after it, out of MEMORY.md.
    assert!(
        app.memory_update(&f.slug, &f.text, "The user likes jasmine green tea.")
            .unwrap()
    );
    let now = app.memory_list().unwrap().remove(0);
    assert_eq!(
        now.text,
        "The user likes jasmine green tea.\nSaid: \"I like green tea a lot.\""
    );
    assert!(
        md().contains("The user likes jasmine green tea.") && !md().contains("Said"),
        "{}",
        md()
    );
    assert_eq!(larch_agent::memory::rev(data), rev + 1);
    // An edit opened before that one is refused, and changes nothing.
    let e = app
        .memory_update(&f.slug, &f.text, "The user likes oolong.")
        .unwrap_err();
    assert!(e.contains("changed since you opened it"), "{e}");
    assert_eq!(app.memory_list().unwrap()[0].text, now.text);
    // The field's text merged onto one line with its Said is not a sentence.
    let merged = "The user likes tea.Said: \"I like green tea a lot.\"";
    assert!(
        app.memory_update(&f.slug, &now.text, merged)
            .unwrap_err()
            .contains("Said")
    );
}

#[test]
fn memory_add_says_what_it_replaced_and_undo_puts_it_back() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    app.memory_add("The user drinks green tea").unwrap();
    // A near-duplicate updates the fact: the receipt says what it replaced.
    let again = app.memory_add("the user drinks green tea!").unwrap();
    assert_eq!(again.old.as_deref(), Some("The user drinks green tea"));
    app.memory_undo(&again).unwrap();
    assert_eq!(
        app.memory_list().unwrap()[0].text,
        "The user drinks green tea"
    );
    // An edit since a save: its Undo is refused and says why.
    let third = app.memory_add("the user drinks green tea!").unwrap();
    app.memory_update(&third.slug, &third.text, "The user drinks black coffee")
        .unwrap();
    assert_eq!(
        app.memory_list().unwrap()[0].text,
        "The user drinks black coffee"
    );
    let e = app.memory_undo(&third).unwrap_err();
    assert!(e.contains("changed since"), "{e}");
    assert_eq!(
        app.memory_list().unwrap()[0].text,
        "The user drinks black coffee"
    );
}

#[test]
fn memory_switches_off_and_on() {
    let dir = Temp::new();
    let (tx, _rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    assert!(app.memory_enabled());
    app.memory_set_enabled(false).unwrap();
    assert!(!app.memory_enabled());
    app.memory_set_enabled(true).unwrap();
    assert!(app.memory_enabled());
}

#[test]
fn memory_off_takes_a_fact_out_of_an_open_chats_next_turn() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    app.memory_add("The user drinks green tea").unwrap();
    let id = app.new_chat(None).unwrap();
    let system = |app: &App| {
        let h = app.open_chat(&id).unwrap();
        assert_eq!(h[0].role, Role::System);
        h[0].content.clone()
    };
    // Each turn is stopped at its first token: the system prompt the test reads is set when the
    // turn starts. A whole reply of the tiny model's noise, read again as history, makes each
    // next turn's prefill longer: minutes on the Windows CI laptop.
    let turn = |message: &str| {
        app.agent_send(&id, message, None, Some(Intent::Chat), &[])
            .unwrap();
        app.cancel(&id).unwrap();
        agent_turn(&rx, &id);
    };
    turn("hello");
    let on = system(&app);
    assert!(
        on.contains("green tea") && on.contains("memory tools"),
        "{on}"
    );
    app.memory_set_enabled(false).unwrap();
    turn("hello again");
    let off = system(&app);
    assert!(
        !off.contains("green tea") && !off.contains("memory tools"),
        "{off}"
    );
    assert!(off.contains("Memory is off"), "{off}");
    app.memory_set_enabled(true).unwrap();
    turn("and again");
    assert_eq!(system(&app), on);
}

/// A loaded model's memory shows before its first reply: a sample arrives with nothing running,
/// and its KV pool is the one the engine reports once it steps. The CPU reference has no GPU to read.
#[test]
fn sample_arrives_before_the_first_reply() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let sample = |rx: &Receiver<Event>| loop {
        if let Event::Sample(s) = rx.recv_timeout(DEADLINE).unwrap() {
            break s;
        }
    };
    let first = sample(&rx);
    assert_eq!(
        (first.tok_s, first.running, first.kv_used_bytes),
        (0.0, 0, 0)
    );
    assert!(first.weights_bytes > 0 && first.kv_total_bytes > 0);
    let id = app.new_chat(None).unwrap();
    app.send(&id, "hi", params(4)).unwrap();
    loop {
        if let Event::Chat {
            event: ChatEvent::Done { .. },
            ..
        } = rx.recv_timeout(DEADLINE).unwrap()
        {
            break;
        }
    }
    let after = sample(&rx);
    assert_eq!(
        (after.weights_bytes, after.kv_total_bytes),
        (first.weights_bytes, first.kv_total_bytes)
    );
    let state = app.app_state();
    assert_eq!((state.gpu_total_bytes, state.gpu_free_bytes), (None, None));
}

/// Uninstall removes a model from Larch's own folder, every part of a split one, and nothing else:
/// not a file outside that folder; the loaded model is shut down first.
#[test]
fn delete_model_removes_only_larchs_files_unloading_the_one_in_use() {
    let dir = Temp::new();
    let (app, _rx, model) = app(&dir.0);
    let models = dir.0.join("models");
    for i in 1..=2 {
        fs::write(
            models.join(format!("big-{i:05}-of-00002.gguf")),
            tiny_gguf("llama"),
        )
        .unwrap();
    }
    let outside = dir.0.join("elsewhere.gguf");
    fs::write(&outside, tiny_gguf("llama")).unwrap();
    let sneaky = models.join("..").join("elsewhere.gguf");
    for refused in [&outside, &sneaky] {
        assert!(app.delete_model(refused.to_str().unwrap()).is_err());
    }
    assert!(outside.exists());
    // Its projector (named for it, whatever its kind) goes with it; another model's stays.
    let (projector, other) = (
        models.join("mmproj-big-f16.gguf"),
        models.join("mmproj-small-f16.gguf"),
    );
    fs::write(&projector, tiny_gguf("llama")).unwrap();
    fs::write(&other, tiny_gguf("llama")).unwrap();
    let first = models.join("big-00001-of-00002.gguf");
    app.delete_model(first.to_str().unwrap()).unwrap();
    assert!(!first.exists() && !models.join("big-00002-of-00002.gguf").exists());
    assert!(!projector.exists() && other.exists());
    // Two quants of one model: its projectors (f16 and Q8_0) stay while either quant does, however
    // the path is spelled, and go with the last one.
    let quants = ["pair-Q8_0.gguf", "pair-Q4_K_M.gguf"].map(|q| models.join(q));
    let pair = ["mmproj-pair-f16.gguf", "mmproj-pair-Q8_0.gguf"].map(|p| models.join(p));
    for f in quants.iter().chain(&pair) {
        fs::write(f, tiny_gguf("llama")).unwrap();
    }
    // Through a link to the library: other path components than the folder listing's.
    #[cfg(unix)]
    let spelled = {
        let link = dir.0.join("models-link");
        std::os::unix::fs::symlink(&models, &link).unwrap();
        link.join("pair-Q8_0.gguf")
    };
    #[cfg(not(unix))]
    let spelled = quants[0].clone();
    app.delete_model(spelled.to_str().unwrap()).unwrap();
    assert!(!quants[0].exists() && pair.iter().all(|p| p.exists()));
    app.delete_model(quants[1].to_str().unwrap()).unwrap();
    assert!(pair.iter().all(|p| !p.exists()) && other.exists());
    // A folder's lone projector with a generic name goes with the folder's only model.
    let solo = dir.0.join("solo");
    fs::create_dir_all(&solo).unwrap();
    let (alone, generic) = (
        solo.join("solo-Q8_0.gguf"),
        solo.join("mmproj-model-f16.gguf"),
    );
    fs::write(&alone, tiny_gguf("llama")).unwrap();
    fs::write(&generic, tiny_gguf("llama")).unwrap();
    app.add_library(solo.to_str().unwrap()).unwrap();
    app.delete_model(alone.to_str().unwrap()).unwrap();
    assert!(!alone.exists() && !generic.exists());
    // The model in use is shut down first, then deleted.
    app.load_model(&model).unwrap();
    app.delete_model(&model).unwrap();
    assert!(!Path::new(&model).exists());
    assert_eq!(app.app_state().health, Health::Empty);
    assert!(app.app_state().model.is_none());
}

/// Libraries: a relative path or a second copy is refused; an added folder's models are listed under
/// it and it is remembered across a restart; uninstall works in it; the default library stays;
/// removing an added one stops listing it and leaves its files.
#[test]
fn libraries_are_added_listed_and_removed() {
    let dir = Temp::new();
    let (first, _rx, _) = app(&dir.0);
    assert!(first.add_library("relative/models").is_err());
    let default = dir.0.join("models").display().to_string();
    assert!(first.add_library(&default).is_err());
    let added = dir.0.join("elsewhere").join("models");
    let libs = first.add_library(added.to_str().unwrap()).unwrap();
    assert_eq!(libs, [default.clone(), added.display().to_string()]);
    fs::write(added.join("new.gguf"), tiny_gguf("qwen3")).unwrap();
    let listed = |app: &App| -> Vec<(String, String)> {
        app.list_models()
            .unwrap()
            .into_iter()
            .map(|m| (m.library, m.arch))
            .collect()
    };
    let want = vec![
        (default.clone(), "llama".to_owned()),
        (added.display().to_string(), "qwen3".to_owned()),
    ];
    assert_eq!(listed(&first), want);
    drop(first);
    let (again, _rx, _) = app(&dir.0);
    assert_eq!(again.app_state().libraries, libs);
    assert_eq!(listed(&again), want);
    let copy = added.join("copy.gguf");
    fs::write(&copy, tiny_gguf("qwen3")).unwrap();
    again.delete_model(copy.to_str().unwrap()).unwrap();
    assert!(!copy.exists());
    assert!(again.remove_library(&default).is_err());
    assert_eq!(
        again.remove_library(&libs[1]).unwrap(),
        std::slice::from_ref(&default)
    );
    assert_eq!(listed(&again), [(default, "llama".to_owned())]);
    assert!(added.join("new.gguf").exists());
    // Once it is not a library, Larch no longer deletes from it.
    assert!(
        again
            .delete_model(added.join("new.gguf").to_str().unwrap())
            .is_err()
    );
}

/// An added library survives a restart, and writing settings.json keeps the code mode setting.
#[test]
fn an_added_library_survives_a_restart() {
    let dir = Temp::new();
    let (first, _rx, _) = app(&dir.0);
    let libs = first
        .add_library(dir.0.join("elsewhere").to_str().unwrap())
        .unwrap();
    drop(first);
    let (again, _rx, _) = app(&dir.0);
    assert_eq!(again.app_state().libraries, libs);
    let saved = fs::read_to_string(dir.0.join("settings.json")).unwrap();
    assert!(saved.contains(r#""code_mode":true"#), "{saved}");
}

/// Code mode is off unless settings.json turns it on: plain chat is the baseline, and a code run
/// is refused by name.
#[test]
fn code_mode_is_off_by_default() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    assert!(!app.app_state().code_mode);
    let repo = toy(&dir.0);
    let id = app.new_chat(None).unwrap();
    let refused = app
        .code_start(&id, "fix add", &repo, None, 1, false)
        .unwrap_err();
    assert!(refused.starts_with("Code mode is off"), "{refused}");
}

/// Runs git for a test's own setup, isolated from the host's configuration.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .current_dir(dir)
        .env_clear()
        .envs(std::env::var_os("PATH").map(|p| ("PATH", p)))
        .env("HOME", dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// A toy checkout in `dir`, committed: add subtracts, and its one test says so; pytest.ini names
/// its tests for a run that names none.
fn toy(dir: &Path) -> PathBuf {
    let repo = dir.join("toy");
    fs::create_dir_all(&repo).unwrap();
    fs::write(repo.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
    fs::write(
        repo.join("test_calc.py"),
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5, add(2, 3)\n\n\n\
         if __name__ == \"__main__\":\n    test_add()\n    print(\"ok\")\n",
    )
    .unwrap();
    fs::write(repo.join("pytest.ini"), "[pytest]\n").unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "start"]);
    repo
}

#[cfg(feature = "code-mode")]
/// Every chat event of `id` until a code run ends, at its `Packet` or `Error`; None, with the
/// reason printed, when this host cannot sandbox the tests and LARCH_SANDBOX_REQUIRED=1 does not
/// say it must (as larch-agent's tests/code_loop.rs).
fn code_turn(rx: &Receiver<Event>, id: &str) -> Option<Vec<ChatEvent>> {
    let events = until(rx, id, |e| {
        matches!(
            e,
            ChatEvent::Packet { .. } | ChatEvent::Error { .. } | ChatEvent::Done { .. }
        )
    });
    match events.last() {
        Some(ChatEvent::Error { message }) if message.starts_with("the sandbox is unavailable") => {
            let required = std::env::var_os("LARCH_SANDBOX_REQUIRED");
            assert!(
                required.is_none_or(|v| v != "1"),
                "LARCH_SANDBOX_REQUIRED=1 but {message}"
            );
            println!("skipped: {message}");
            None
        }
        _ => Some(events),
    }
}

/// The tiny model cannot make a call: its prompt does not even fit its 512-token context, so the
/// run stops at once, by name, and the packet has nothing in it.
#[cfg(feature = "code-mode")]
#[test]
fn code_start_parks_a_packet_that_discard_drops_leaving_the_checkout_as_it_was() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let repo = toy(&dir.0);
    let head = git(&repo, &["rev-parse", "HEAD"]);
    let id = app.new_chat(None).unwrap();
    // Windows' Python has no python3 outside the Store's aliases, which the sandbox cannot start.
    let python = if cfg!(windows) { "python" } else { "python3" };
    let test = || Some(format!("{python} test_calc.py"));
    app.code_start(&id, "fix add in calc.py", &repo, test(), 1, true)
        .unwrap();
    // One code run per chat: refused while it runs (or, had it ended already, while it waits).
    assert!(
        app.code_start(&id, "again", &repo, test(), 1, true)
            .is_err()
    );
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Packet {
        stop,
        scope,
        tests_before,
        tests_after,
        steps,
        ..
    }) = events.last()
    else {
        panic!("{events:?}")
    };
    let stop = stop.as_deref().unwrap();
    assert!(stop.starts_with("the engine failed: prompt is "), "{stop}");
    assert!(events.contains(&ChatEvent::Notice { text: stop.into() }));
    assert_eq!(scope, &[]);
    let failed = |t: &Option<Tests>| t.as_ref().unwrap().failed;
    assert_eq!(
        (failed(tests_before), failed(tests_after), *steps),
        (1, 1, 0)
    );
    assert!(!events.iter().any(|e| matches!(e, ChatEvent::Delta { .. })));
    assert_eq!(
        app.code_start(&id, "again", &repo, test(), 1, true)
            .unwrap_err(),
        "Approve or discard this chat's code packet first."
    );
    // The chat stays while its packet waits: its worktree would have nothing left to decide it.
    assert!(app.delete_chat(&id).is_err());
    // Named after the chat too, so two chats with the same words never collide.
    let worktree = dir
        .0
        .join(format!("agent/code/fix-add-in-calc-py-chat-{id}/0/toy"));
    assert!(worktree.join(".git").exists(), "{}", worktree.display());

    app.code_decide(&id, false).unwrap();
    let events = agent_turn(&rx, &id);
    let [ChatEvent::Notice { text }, ChatEvent::Done { finish, .. }] = &events[..] else {
        panic!("{events:?}")
    };
    assert!(
        text.starts_with("discarded; ") && text.ends_with(" untouched"),
        "{text}"
    );
    assert_eq!(finish, "eos");
    assert!(!worktree.exists());
    assert_eq!(git(&repo, &["rev-parse", "HEAD"]), head);
    assert_eq!(git(&repo, &["status", "--porcelain"]), "");
    assert_eq!(git(&repo, &["branch", "--list", "iris/*"]), "");
    let listed = git(&repo, &["worktree", "list", "--porcelain"]);
    assert_eq!(listed.matches("worktree ").count(), 1, "{listed}");
    assert_eq!(
        app.code_decide(&id, false).unwrap_err(),
        "No packet waits in this chat."
    );
    app.delete_chat(&id).unwrap();
}

/// Stop during a code run stops it at its next safe point, here once the tests it runs first
/// end: its packet says the user stopped it, and the model took no step.
#[cfg(feature = "code-mode")]
#[test]
fn cancel_stops_a_code_run_at_its_next_safe_point() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let repo = toy(&dir.0);
    let id = app.new_chat(None).unwrap();
    // Tests that take a while, so the stop lands while they run.
    let python = if cfg!(windows) { "python" } else { "python3" };
    let test = format!("{python} -c \"import time; time.sleep(2)\"");
    (app.code_start(&id, "fix add in calc.py", &repo, Some(test), 1, true)).unwrap();
    app.cancel(&id).unwrap();
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Packet { stop, steps, .. }) = events.last() else {
        panic!("{events:?}")
    };
    assert_eq!(
        (stop.as_deref(), *steps),
        (Some("the user stopped the run"), 0)
    );
    app.code_decide(&id, false).unwrap();
    agent_turn(&rx, &id);
}

/// A change asked for in a checkout the text names, or in the chat's project folder, runs code
/// mode in place, after a checkpoint branch; Cargo.toml names the tests. The intent is given, as
/// the tiny model's own reading of the text is noise. Undone, the folder is as it was, and a
/// change made in place never holds the chat: the next request runs at once.
#[cfg(feature = "code-mode")]
#[test]
fn a_change_request_naming_a_checkout_runs_code_mode() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let repo = toy(&work.0);
    // cargo's output stays out of the change.
    fs::write(repo.join(".gitignore"), "/target\n/Cargo.lock\n").unwrap();
    // Cargo.toml comes before pytest.ini: every host that runs these tests has cargo, and a test
    // command the host lacks stops the run before its packet.
    fs::create_dir(repo.join("src")).unwrap();
    fs::write(repo.join("src/lib.rs"), "").unwrap();
    fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"toy\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    // The toolchain these tests run on: rustup names it to cargo in a variable the sandbox
    // does not pass.
    if let Ok(toolchain) = std::env::var("RUSTUP_TOOLCHAIN") {
        fs::write(repo.join("rust-toolchain"), toolchain).unwrap();
    }
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "cargo"]);
    let id = app.new_chat(None).unwrap();
    let named = format!("Fix the failing test in {}", repo.display());
    for folder in [false, true] {
        match folder {
            false => app.agent_send(&id, &named, None, Some(Intent::Change), &[]),
            true => app.agent_send(
                &id,
                "Fix the failing test",
                Some(&repo),
                Some(Intent::Change),
                &[],
            ),
        }
        .unwrap();
        let Some(events) = code_turn(&rx, &id) else {
            return;
        };
        let Some(ChatEvent::Packet { checkpoint, .. }) = events.last() else {
            panic!("{events:?}")
        };
        let branch = checkpoint.as_deref().unwrap();
        assert!(
            branch.starts_with("larch/checkpoint-fix-the-failing-test-"),
            "{branch}"
        );
        assert!(git(&repo, &["branch", "--list", branch]).contains(branch));
        assert!(!events.iter().any(|e| matches!(e, ChatEvent::Delta { .. })));
        app.code_decide(&id, false).unwrap();
        let events = agent_turn(&rx, &id);
        let [ChatEvent::Notice { text }, ChatEvent::Done { .. }] = &events[..] else {
            panic!("{events:?}")
        };
        // The tiny model's run may change a file or none; either way the folder is as it was.
        let nothing = text == "Nothing to undo: the run changed no file.";
        assert!(text.starts_with("undone; ") || nothing, "{text}");
        assert_eq!(git(&repo, &["status", "--porcelain"]), "");
        assert_eq!(
            app.code_answer(&id, Decision::Once).unwrap_err(),
            "No question waits in this chat."
        );
    }
}

/// Approve merges the packet into the checkout, then its worktree and branch go. A cleanup git
/// cannot finish leaves the merge decided: the chat hears "merged into", then what stays.
#[cfg(unix)]
#[cfg(feature = "code-mode")]
#[test]
fn code_decide_true_merges_and_a_failed_cleanup_is_a_notice_not_a_packet() {
    use std::os::unix::fs::PermissionsExt;
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let repo = toy(&dir.0);
    let head = git(&repo, &["rev-parse", "HEAD"]);
    let id = app.new_chat(None).unwrap();
    // The tests on the base leave a nested repository, which `git clean -d` keeps and the packet
    // commits as a gitlink: a change to merge. Their second run, after the commit, leaves a
    // folder nothing can be removed from, so the worktree will not go.
    let test = "if [ -e sub ]; then mkdir -p stuck && touch stuck/f && chmod 555 stuck; \
                else git init -q sub && git -C sub -c user.name=x -c user.email=x \
                -c commit.gpgsign=false commit -q --allow-empty -m x; fi; exit 1";
    app.code_start(&id, "fix add in calc.py", &repo, Some(test.into()), 1, true)
        .unwrap();
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Packet { diff, scope, .. }) = events.last() else {
        panic!("{events:?}")
    };
    assert!(diff.contains("+Subproject commit "), "{diff}");
    assert_eq!(scope.len(), 1, "{scope:?}");
    let worktree = dir
        .0
        .join(format!("agent/code/fix-add-in-calc-py-chat-{id}/0/toy"));
    app.code_decide(&id, true).unwrap();
    let events = agent_turn(&rx, &id);
    let [
        ChatEvent::Notice { text: merged },
        ChatEvent::Notice { text: left },
        ChatEvent::Done { .. },
    ] = &events[..]
    else {
        panic!("{events:?}")
    };
    assert_eq!(merged, &format!("merged into {}", repo.display()));
    assert!(left.starts_with("its worktree or branch stays: "), "{left}");
    assert_ne!(git(&repo, &["rev-parse", "HEAD"]), head);
    assert!(git(&repo, &["ls-tree", "HEAD"]).contains("\tsub\n"));
    // Decided: nothing waits, and the chat can run again.
    assert_eq!(
        app.code_decide(&id, true).unwrap_err(),
        "No packet waits in this chat."
    );
    assert!(worktree.join("stuck").exists());
    fs::set_permissions(worktree.join("stuck"), fs::Permissions::from_mode(0o755)).unwrap();
    // A later start of the app sweeps what this one left (the worktree is gone from git's
    // list already; the branch is not).
    drop(app);
    let (_again, _rx) = loaded(&dir.0);
    assert!(!worktree.exists());
    assert_eq!(git(&repo, &["branch", "--list", "iris/*"]), "");
}

/// A change made in place that still waits to be kept or undone is kept when the chat's next
/// run starts: the chat hears "kept in" for it, and its history holds that decision between the
/// two runs. Started over it, the next run's outcome would replace it and its Undo with it.
#[cfg(feature = "code-mode")]
#[test]
fn a_new_run_in_place_keeps_the_change_still_waiting() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let empty = work.0.join("new");
    fs::create_dir(&empty).unwrap();
    let id = app.new_chat(None).unwrap();
    (app.code_start(&id, "write a program that says hi", &empty, None, 1, false)).unwrap();
    if code_turn(&rx, &id).is_none() {
        return;
    }
    (app.code_start(&id, "make it say bye", &empty, None, 1, false)).unwrap();
    // The keep is a notice in the new run's reply, never its end: a Done there ended the UI's
    // reply, so the new run's asks and packet were dropped unseen.
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Notice { text }) = events.first() else {
        panic!("{events:?}")
    };
    assert_eq!(text, &format!("kept in {}", empty.display()));
    assert!(
        !events.iter().any(|e| matches!(e, ChatEvent::Done { .. })),
        "{events:?}"
    );
    let asked: Vec<_> = (app.open_chat(&id).unwrap().into_iter())
        .filter(|m| m.role == Role::User)
        .map(|m| m.content)
        .collect();
    assert_eq!(
        asked,
        [
            "write a program that says hi",
            "Keep the change.",
            "make it say bye"
        ]
    );
}

/// A chat's code task lasts across its runs: its record keeps the first request and the asks
/// since, and a follow-up's run is asked the task, so its checkpoint is named after it, while its
/// card names the ask. Undone, a run that changed nothing says so, and the record says the change
/// is gone; the chat's record goes with the chat.
#[cfg(feature = "code-mode")]
#[test]
fn a_follow_up_run_in_a_chat_goes_on_with_its_task() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let empty = work.0.join("new");
    fs::create_dir(&empty).unwrap();
    let id = app.new_chat(None).unwrap();
    (app.code_start(&id, "write a page about cats", &empty, None, 1, false)).unwrap();
    if code_turn(&rx, &id).is_none() {
        return;
    }
    (app.code_start(&id, "continue", &empty, None, 1, false)).unwrap();
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Packet {
        checkpoint, title, ..
    }) = events.last()
    else {
        panic!("{events:?}")
    };
    assert_eq!(title, "continue");
    let branch = checkpoint.as_deref().unwrap();
    assert!(
        branch.starts_with("larch/checkpoint-write-a-page-about-cats-"),
        "{branch}"
    );
    let path = dir.0.join(format!("agent/code/tasks/{id}.json"));
    let record =
        || -> serde_json::Value { serde_json::from_slice(&fs::read(&path).unwrap()).unwrap() };
    let r = record();
    assert_eq!(r["objective"], "write a page about cats");
    assert_eq!(r["asks"], serde_json::json!(["continue"]));
    let verdict = r["verdict"].as_str().unwrap();
    assert!(verdict.starts_with("the engine failed: "), "{verdict}");
    app.code_decide(&id, false).unwrap();
    let events = agent_turn(&rx, &id);
    let notice = ChatEvent::Notice {
        text: "Nothing to undo: the run changed no file.".into(),
    };
    assert!(events.contains(&notice), "{events:?}");
    assert_eq!(record()["verdict"], "the user undid the last run's change");
    app.delete_chat(&id).unwrap();
    assert!(!path.exists());
}

/// Code the chat's last answer drafted, in blocks whose fence names a file, starts the chat's
/// first run: each file the folder lacks and allows is written after its checkpoint, and the chat
/// hears which; Undo takes them out again. A file the folder has is the user's and stays.
#[cfg(feature = "code-mode")]
#[test]
fn a_drafted_page_in_the_chat_starts_the_run() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let folder = work.0.join("cafe");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("style.css"), "mine\n").unwrap();
    let said = |role, content: &str| larch::ChatMessage {
        role,
        content: content.into(),
        ..Default::default()
    };
    let answer = "Here it is:\n```html index.html\n<h1>Cafe</h1>\n```\n```style.css\nh1 {}\n```\n";
    let history = vec![
        said(larch::Role::User, "code a cafe page"),
        said(larch::Role::Assistant, answer),
    ];
    let id = "drafted";
    let path = larch::chat::transcript_path(&dir.0, id).unwrap();
    larch::chat::Transcript::create(&path, larch::SessionOptions::default(), history).unwrap();
    let silent = Allowed {
        silent: true,
        ..Default::default()
    };
    app.code_allow(&folder, silent).unwrap();
    (app.code_start(id, "put it in the folder", &folder, None, 1, false)).unwrap();
    let Some(events) = code_turn(&rx, id) else {
        return;
    };
    let read = |name| fs::read_to_string(folder.join(name)).ok();
    assert_eq!(read("index.html").as_deref(), Some("<h1>Cafe</h1>\n"));
    assert_eq!(read("style.css").as_deref(), Some("mine\n"));
    let text = "started from the code drafted in this chat: index.html";
    let notice = ChatEvent::Notice { text: text.into() };
    assert!(events.contains(&notice), "{events:?}");
    app.code_decide(id, false).unwrap();
    agent_turn(&rx, id);
    assert_eq!(read("index.html"), None);
    assert_eq!(read("style.css").as_deref(), Some("mine\n"));
}

/// A run asked in a chat with a project folder is a chat turn: no code run starts, so nothing
/// is checkpointed and the chat gets no code task.
#[test]
fn a_run_request_in_a_project_is_a_chat_turn() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    // The agent's system prompt (its tools and rules) needs more than the tiny model's 512 tokens.
    let model = dir.0.join("models").join("tiny-4k.gguf");
    fs::write(&model, tiny_gguf_ctx("llama", Some(4096))).unwrap();
    app.load_model(model.to_str().unwrap()).unwrap();
    let work = Temp::new();
    let folder = work.0.join("prog");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("main.py"), "print('hi')\n").unwrap();
    let id = app.new_chat(None).unwrap();
    (app.agent_send(
        &id,
        "how's the performance?",
        Some(&folder),
        Some(Intent::Run),
        &[],
    ))
    .unwrap();
    app.cancel(&id).unwrap();
    let events = agent_turn(&rx, &id);
    assert!(
        matches!(events.last(), Some(ChatEvent::Done { .. })),
        "{events:?}"
    );
    assert!(!events.iter().any(|e| matches!(e, ChatEvent::Packet { .. })));
    assert!(!dir.0.join("agent/folders").exists());
    assert!(
        !dir.0
            .join("agent/code/tasks")
            .join(format!("{id}.json"))
            .exists()
    );
}

/// A change asked in a chat with no folder is answered with a notice to pick one, not with code
/// in the chat; the chat keeps nothing of it, so its last answer is still the one before.
#[cfg(feature = "code-mode")]
#[test]
fn a_change_with_no_folder_asks_for_one() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    let id = app.new_chat(None).unwrap();
    (app.agent_send(&id, "make the page blue", None, Some(Intent::Change), &[])).unwrap();
    let events = agent_turn(&rx, &id);
    let [ChatEvent::Notice { text }, ChatEvent::Done { .. }] = &events[..] else {
        panic!("{events:?}")
    };
    assert_eq!(
        text,
        "Pick a project folder and send it again, and I'll make the change there."
    );
    assert!(app.open_chat(&id).unwrap().is_empty());
}

/// One chat at a time changes a folder in place: while the first one's run or its change waits,
/// a second chat's in-place run is refused by the first one's name, and runs isolated instead;
/// once the first change is decided, the second chat may change the folder. The first one's
/// change here is a file its chat drafted.
#[cfg(feature = "code-mode")]
#[test]
fn a_second_chat_cannot_change_a_folder_in_place_at_once() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let repo = toy(&work.0);
    let python = if cfg!(windows) { "python" } else { "python3" };
    let test = || Some(format!("{python} test_calc.py"));
    let silent = Allowed {
        silent: true,
        ..Default::default()
    };
    app.code_allow(&repo, silent).unwrap();
    let said = |role, content: &str| larch::ChatMessage {
        role,
        content: content.into(),
        ..Default::default()
    };
    let history = vec![
        said(larch::Role::User, "fix add in calc.py"),
        said(larch::Role::Assistant, "```notes.txt\nadd adds\n```\n"),
    ];
    let a = "drafted".to_string();
    let path = larch::chat::transcript_path(&dir.0, &a).unwrap();
    larch::chat::Transcript::create(&path, larch::SessionOptions::default(), history).unwrap();
    let b = app.new_chat(None).unwrap();
    (app.code_start(&a, "fix add in calc.py", &repo, test(), 1, false)).unwrap();
    if code_turn(&rx, &a).is_none() {
        return;
    }
    assert_eq!(
        app.code_start(&b, "make add faster", &repo, test(), 1, false)
            .unwrap_err(),
        "Another chat (fix add in calc.py) has a change waiting to be kept or undone in this \
         folder; wait for it or run this one isolated."
    );
    app.code_start(&b, "make add faster", &repo, test(), 1, true)
        .unwrap();
    code_turn(&rx, &b).unwrap();
    app.code_decide(&b, false).unwrap();
    agent_turn(&rx, &b);
    app.code_decide(&a, true).unwrap();
    agent_turn(&rx, &a);
    app.code_start(&b, "make add faster", &repo, test(), 1, false)
        .unwrap();
    code_turn(&rx, &b).unwrap();
    app.code_decide(&b, false).unwrap();
    agent_turn(&rx, &b);
}

/// A run in place that changed nothing holds the folder from no other chat.
#[cfg(feature = "code-mode")]
#[test]
fn a_run_that_changed_nothing_holds_no_folder() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let empty = work.0.join("new");
    fs::create_dir(&empty).unwrap();
    let (a, b) = (app.new_chat(None).unwrap(), app.new_chat(None).unwrap());
    (app.code_start(&a, "write a program that says hi", &empty, None, 1, false)).unwrap();
    code_turn(&rx, &a).unwrap();
    (app.code_start(&b, "write a program that says bye", &empty, None, 1, false)).unwrap();
    code_turn(&rx, &b).unwrap();
}

/// A folder no test command names, an empty one outside git too, runs in place untested: its
/// packet has no tests, and no repository is made in it. Attempts side by side, which only tests
/// could pick between, are refused before any worktree is made.
#[cfg(feature = "code-mode")]
#[test]
fn a_folder_without_tests_runs_untested_but_not_side_by_side() {
    let dir = Temp::new();
    let (app, rx) = loaded(&dir.0);
    // Outside the data folder, which code mode never changes in place.
    let work = Temp::new();
    let empty = work.0.join("new");
    fs::create_dir(&empty).unwrap();
    let id = app.new_chat(None).unwrap();
    (app.code_start(&id, "write a program that says hi", &empty, None, 1, false)).unwrap();
    let Some(events) = code_turn(&rx, &id) else {
        return;
    };
    let Some(ChatEvent::Packet {
        tests_before: None,
        tests_after: None,
        checkpoint: Some(_),
        ..
    }) = events.last()
    else {
        panic!("{events:?}")
    };
    assert_eq!(fs::read_dir(&empty).unwrap().count(), 0);
    app.code_decide(&id, true).unwrap();
    agent_turn(&rx, &id);
    // The chat keeps what the run was asked and did, then what the user decided: its next turn
    // reads them.
    let kept: Vec<_> = (app.open_chat(&id).unwrap().into_iter())
        .filter(|m| m.role != Role::System)
        .map(|m| (m.role, m.content))
        .collect();
    let [
        (Role::User, asked),
        (Role::Assistant, did),
        (Role::User, keep),
        (Role::Assistant, kept),
    ] = &kept[..]
    else {
        panic!("{kept:?}")
    };
    assert_eq!(asked, "write a program that says hi");
    let start = format!(
        "I worked in the folder {} and changed no file. The run stopped: the engine failed: ",
        empty.display()
    );
    assert!(did.starts_with(&start), "{did}");
    // Nothing changed: there is no change to call unverified.
    assert!(!did.contains("unverified"), "{did}");
    assert_eq!(
        (keep.as_str(), kept),
        ("Keep the change.", &format!("kept in {}", empty.display()))
    );
    let repo = toy(&work.0);
    fs::remove_file(repo.join("pytest.ini")).unwrap();
    git(&repo, &["commit", "-q", "-am", "no manifest"]);
    let id = app.new_chat(None).unwrap();
    app.code_start(&id, "fix add", &repo, None, 2, true)
        .unwrap();
    let events = code_turn(&rx, &id).unwrap();
    let Some(ChatEvent::Error { message }) = events.last() else {
        panic!("{events:?}")
    };
    assert!(message.ends_with("name the test command"), "{message}");
    // No worktree: only the first chat's task record is there.
    let made: Vec<_> = (fs::read_dir(dir.0.join("agent/code")).unwrap())
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(made, ["tasks"]);
}

/// The permission menu's change is told to the UI as saved, by the folder's real path, so the
/// menu repaints from what the app keeps rather than from what it asked for.
#[test]
fn code_allow_tells_the_ui_what_the_folder_now_allows() {
    let dir = Temp::new();
    let (app, rx, _) = app(&dir.0);
    let folder = dir.0.join("project");
    fs::create_dir(&folder).unwrap();
    let allowed = Allowed {
        silent: true,
        ..Default::default()
    };
    app.code_allow(&folder.join("."), allowed.clone()).unwrap();
    let told = rx.try_iter().find_map(|e| match e {
        Event::Allowed(a) => Some(a),
        _ => None,
    });
    // Without Windows' verbatim prefix, as allowances are kept by.
    let real = fs::canonicalize(&folder).unwrap().display().to_string();
    let real = real.trim_start_matches(r"\\?\").to_string();
    assert_eq!(
        told,
        Some(FolderAllowed {
            folder: real,
            allowed
        })
    );
}

#[test]
fn a_chat_starts_with_nothing_attached_and_keeps_what_is_attached_to_it() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    let id = app.new_chat(None).unwrap();
    assert!(app.attachments(&id).unwrap().is_empty());
    let papers = dir.0.join("papers-ml");
    fs::create_dir(&papers).unwrap();
    for name in ["a.md", "b.txt", "c.docx"] {
        fs::write(papers.join(name), "text").unwrap();
    }
    let real = fs::canonicalize(&papers).unwrap().display().to_string();
    let got = app.attach(&id, papers.to_str().unwrap()).unwrap();
    // A folder is shown with its separator, so the UI shows a folder's chip.
    let shown = format!("{real}{}", std::path::MAIN_SEPARATOR);
    assert_eq!(got.attached, std::slice::from_ref(&shown));
    assert_eq!(got.line, format!("Attached {real} (2 files)"));
    // A file is attached alone, and attaching it again changes nothing.
    let file = papers.join("a.md");
    let file_real = fs::canonicalize(&file).unwrap().display().to_string();
    let got = app.attach(&id, file.to_str().unwrap()).unwrap();
    assert_eq!(got.line, format!("Attached {file_real} (1 file)"));
    app.attach(&id, file.to_str().unwrap()).unwrap();
    // Kept with the chat, so a new App (a restart) reads them back; another chat has none.
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    assert_eq!(
        app.attachments(&id).unwrap(),
        [shown.clone(), file_real.clone()]
    );
    assert!(
        app.attachments(&app.new_chat(None).unwrap())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        app.detach(&id, &file_real).unwrap(),
        std::slice::from_ref(&shown)
    );
    assert!(
        app.attach(&id, &dir.0.join("gone").display().to_string())
            .is_err()
    );
    // What cannot be listed is not attached: a FIFO canonicalizes but is no folder to walk.
    #[cfg(unix)]
    {
        let fifo = dir.0.join("fifo");
        let made = std::process::Command::new("mkfifo").arg(&fifo).status();
        if made.is_ok_and(|s| s.success()) {
            assert!(app.attach(&id, fifo.to_str().unwrap()).is_err());
            assert_eq!(app.attachments(&id).unwrap(), std::slice::from_ref(&shown));
        }
    }
    // A folder is detached by the path the UI shows.
    assert!(app.detach(&id, &shown).unwrap().is_empty());
    app.delete_chat(&id).unwrap();
    assert!(app.attachments(&id).unwrap().is_empty());
}

#[test]
fn detaching_a_file_forgets_what_the_chat_read_of_it_and_deleting_the_chat_its_record() {
    use larch_agent::folio::{source, store};
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), CPU, |_| {}).unwrap();
    let id = app.new_chat(None).unwrap();
    let file = dir.0.join("lease.md");
    fs::write(&file, "Rent is $1,200 a month.\n").unwrap();
    let real = fs::canonicalize(&file).unwrap();
    app.attach(&id, file.to_str().unwrap()).unwrap();
    // What a turn on the file leaves: its text in the extract cache and the chat's record.
    let stop = std::sync::atomic::AtomicBool::new(false);
    let doc = store(&dir.0, &id, "lease.md", &real, &stop)
        .unwrap()
        .unwrap();
    let stamp = source::Stamp::of(&fs::metadata(&real).unwrap()).unwrap();
    assert!(source::cached(&dir.0, &real, stamp).is_some());
    assert_eq!(doc.letter(), "A");
    app.detach(&id, &real.display().to_string()).unwrap();
    assert!(source::cached(&dir.0, &real, stamp).is_none());
    // The chat's letters stay while it lives, and go with it.
    assert_eq!(source::load(&dir.0, &id).unwrap().len(), 1);
    app.delete_chat(&id).unwrap();
    assert!(!source::record_at(&dir.0, &id).exists());
    assert!(real.exists());
}

/// The faster version of a model: one that fails to load is deleted and the original loaded again;
/// one that loads replaces it, and the original comes back on request. A file with no faster version
/// offers none and refuses a build (on the CPU reference no file offers one).
#[test]
fn a_faster_version_that_fails_or_runs_without_graphs_falls_back_to_the_original() {
    let tmp = Temp::new();
    let dir = tmp.0.clone();
    let (app, _rx) = loaded(&dir);
    let model = dir.join("models").join("tiny.gguf");
    let model = model.to_str().unwrap();
    let copy = dir.join("faster").join("tiny.faster.gguf");
    assert!(app.use_faster(model, true).is_err(), "no copy made yet");
    assert_eq!(app.faster_for(model), None);
    assert!(app.build_faster(model).is_err());
    assert!(!dir.join("faster").exists());

    fs::create_dir_all(copy.parent().unwrap()).unwrap();
    fs::write(&copy, b"not a model").unwrap();
    let e = app.use_faster(model, true).unwrap_err();
    assert!(e.contains("went back to the original"), "{e}");
    assert!(!copy.exists(), "the copy that failed is deleted");
    let st = app.app_state();
    assert_eq!(
        (st.model.as_deref(), st.health),
        (Some(model), Health::Ready)
    );

    // A copy that loads but runs without decode graphs (the CPU reference has none; on a GPU, a
    // load short of room) is not faster: back to the original, the copy kept. On a GPU that
    // replays graphs the tiny model loads whole with them, and the copy runs.
    fs::copy(model, &copy).unwrap();
    match app.use_faster(model, true) {
        Err(e) => {
            assert!(e.contains("too little free GPU memory"), "{e}");
            let st = app.app_state();
            assert_eq!(
                (st.model.as_deref(), st.health),
                (Some(model), Health::Ready)
            );
        }
        Ok(()) => {
            assert_ne!(
                larch_kernels::BACKEND,
                "cpu",
                "the CPU reference replays no graphs"
            );
            assert_eq!(app.app_state().model.as_deref(), copy.to_str());
        }
    }
    assert!(copy.exists(), "a copy that loaded is kept");
    app.use_faster(model, false).unwrap();
    assert_eq!(app.app_state().model.as_deref(), Some(model));
    // The copy's folder is not a library: the list shows the original only.
    let listed: Vec<String> = app
        .list_models()
        .unwrap()
        .into_iter()
        .map(|m| m.path)
        .collect();
    assert_eq!(listed, [model]);

    // Deleting the model deletes its faster version, which no library lists, with it.
    assert!(copy.exists());
    app.delete_model(model).unwrap();
    assert!(!copy.exists(), "the faster version goes with its model");
}
