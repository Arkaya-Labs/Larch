//! F4U1 CPU evidence: real engine with its existing fake, plus a real generated Transformer GGUF.

use std::{
    fs,
    io::{self, Cursor, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use larch::{
    ChatMessage, EngineError, Finish, GenParams, Larch, LoadOptions, Overflow, PinError, Prompt,
    Role, SessionOptions, TextEvent, TokenLogprob,
    api::{ModelClass, invalid},
    chat::{Metrics, Terminal, Transcript},
};
use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, ReloadChange, Request, RequestHandle, RouterDeltas,
    Usage, start, start_with,
};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{
    Arch, ChatTemplate, FakeModel, Message, Model, ModelConfig, StepBatch, Tokenizer, ToolCall,
    gguf::{Gguf, Value, write},
    tiny,
};
use larch_records::{
    AbCommand, AbReply, Authority, EffectiveConfig, Health, HealthState, KnobControl, KnobReply,
    KnobRequest, KnobSpec, Read, ReadFrom, Record, RejectReason, RevertTarget, knob_id,
};

static NEXT: AtomicU64 = AtomicU64::new(1);
// A test holding a `Temp` shares the CPU; a timed test runs alone, so parallel tests cannot stretch
// its wall-clock bound. The share is a count, not a lock guard, because a `Temp` may move threads.
static SHARES: AtomicU64 = AtomicU64::new(0);
static ALONE: Mutex<()> = Mutex::new(());

struct CpuShare;
impl Drop for CpuShare {
    fn drop(&mut self) {
        SHARES.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Waits out every shared `Temp`; new ones wait until the guard drops. A leaked share fails here
/// by name instead of hanging the run.
fn cpu_alone() -> MutexGuard<'static, ()> {
    let alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let shares = SHARES.load(Ordering::Acquire);
        if shares == 0 {
            return alone;
        }
        assert!(
            Instant::now() < deadline,
            "{shares} `Temp` shares still held; a test leaked one"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

struct Temp(PathBuf, #[allow(dead_code)] Option<CpuShare>);
impl Temp {
    fn new() -> Self {
        let _alone = ALONE.lock().unwrap_or_else(PoisonError::into_inner);
        SHARES.fetch_add(1, Ordering::AcqRel);
        Self::with(Some(CpuShare))
    }
    fn with(share: Option<CpuShare>) -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-facade-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p, share)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn metadata(tails: &[(&str, i32)], template: &str) -> Vec<(String, Value)> {
    let mut tokens: Vec<_> = (0..256)
        .map(|b| Value::Str(format!("<0x{b:02X}>")))
        .collect();
    tokens.push(Value::Str("<cue>".into()));
    tokens.extend(tails.iter().map(|(s, _)| Value::Str((*s).into())));
    tokens.push(Value::Str("<end>".into()));
    let mut types = vec![Value::I32(6); 256];
    types.push(Value::I32(3));
    types.extend(tails.iter().map(|(_, k)| Value::I32(*k)));
    types.push(Value::I32(3));
    vec![
        ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Value::Arr(tokens)),
        ("tokenizer.ggml.token_type".into(), Value::Arr(types)),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(256)),
        (
            "tokenizer.ggml.eos_token_id".into(),
            Value::U32(257 + tails.len() as u32),
        ),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)),
        (
            "tokenizer.chat_template".into(),
            Value::Str(template.into()),
        ),
    ]
}

const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";

fn fixture_with(tails: &[(&str, i32)], context: u32, template: &str) -> (Larch, EngineHandle) {
    fixture_model(
        tails,
        context,
        template,
        Box::new(FakeModel::new(tails.len() as u32 + 258)),
    )
}

fn fixture_model(
    tails: &[(&str, i32)],
    context: u32,
    template: &str,
    model: Box<dyn Model>,
) -> (Larch, EngineHandle) {
    let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let g = Gguf::parse(&write(&metadata(tails, template), &tensors)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let mut opts = EngineOptions::new("fake".into(), CPU);
    opts.max_context = context;
    opts.ring_capacity = 256;
    let e = start_with(opts, Box::new(move |_, _, _| Ok(model))).unwrap();
    (Larch::from_engine(e.clone(), tok, template), e)
}

fn fixture(context: u32) -> (Larch, EngineHandle) {
    fixture_with(&[("b", 1), ("c", 1), ("d", 1)], context, TEMPLATE)
}

fn params(tokens: u32) -> GenParams {
    GenParams {
        max_tokens: tokens,
        temperature: Some(0.0),
        ..GenParams::default()
    }
}

fn collect(stream: impl IntoIterator<Item = TextEvent>) -> (String, TextEvent) {
    let mut text = String::new();
    let mut terminal = None;
    for event in stream {
        match event {
            TextEvent::Text { delta, .. } => text.push_str(&delta),
            end => {
                assert!(terminal.is_none());
                terminal = Some(end);
            }
        }
    }
    (text, terminal.expect("terminal event"))
}

fn output(stream: larch::TextStream) -> (String, Vec<TokenLogprob>, TextEvent) {
    let mut probabilities = Vec::new();
    let (text, end) = collect(stream.inspect(|event| {
        if let TextEvent::Text {
            tokens, logprobs, ..
        } = event
        {
            let logprobs = logprobs.as_ref().expect("requested logprobs");
            assert_eq!(*tokens as usize, logprobs.len());
            probabilities.extend_from_slice(logprobs);
        }
    }));
    (text, probabilities, end)
}

fn same_output_tokens(cached: &[TokenLogprob], fresh: &[TokenLogprob]) {
    assert!(!cached.is_empty());
    assert_eq!(cached.len(), fresh.len());
    for (a, b) in cached.iter().zip(fresh) {
        assert_eq!(a.id, b.id);
        // The CPU bound established in larch-engine/tests/engine.rs::same_distribution.
        assert!((a.logprob - b.logprob).abs() < 1e-2, "{a:?} vs {b:?}");
        assert_eq!(a.top.len(), 5);
        assert_eq!(a.top.len(), b.top.len());
        for ((id, x), (other, y)) in a.top.iter().zip(&b.top) {
            assert_eq!(id, other);
            assert!((x - y).abs() < 1e-2, "{a:?} vs {b:?}");
        }
    }
}

fn usage(end: TextEvent, expected: Finish) -> larch::Usage {
    match end {
        TextEvent::Done {
            finish,
            usage: u,
            timings: t,
            ..
        } => {
            assert_eq!(finish, expected);
            assert_eq!(t.cached_tokens, u.cached_tokens);
            assert!(t.ttft_ms > 0.0 && t.tpot_ms >= 0.0 && t.tok_s >= 0.0);
            u
        }
        other => panic!("expected natural completion, got {other:?}"),
    }
}

#[test]
fn public_prompts_natural_eos_length_and_engine_telemetry() {
    let (v, e) = fixture(512);
    assert_eq!(v.health().state, HealthState::Ready);
    let mut s = v.session(SessionOptions::default());
    let (text, end) = collect(s.send("hello", params(2)).unwrap());
    assert_eq!(text, "bc");
    let u = usage(end, Finish::Length);
    assert_eq!((u.completion_tokens, u.cached_tokens), (2, 0));
    assert_eq!(s.history()[1].content, "bc");
    assert!(!s.history()[1].truncated);
    assert_eq!(s.tokens_in_kv(), u.prompt_tokens + 1);
    let (text, end) = collect(s.send("again", params(10)).unwrap());
    assert_eq!(text, "bcd");
    let u = usage(end, Finish::Eos);
    assert_eq!(u.completion_tokens, 4);
    assert_eq!(s.history()[3].content, text);
    assert!(!s.history()[3].truncated);
    assert_eq!(s.tokens_in_kv(), u.prompt_tokens + 3);
    let prompt = Prompt::Messages(vec![Message {
        role: "user".into(),
        content: "hello".into(),
    }]);
    assert_eq!(collect(v.generate(prompt, params(1)).unwrap()).0, "b");
    assert_eq!(
        usage(
            collect(v.generate(Prompt::Tokens(vec![256]), params(1)).unwrap()).1,
            Finish::Length
        )
        .cached_tokens,
        0
    );
    assert_eq!(
        collect(v.generate(Prompt::Text("<cue>".into()), params(1)).unwrap()).0,
        "b"
    );
    let mut reader = v.telemetry().reader(ReadFrom::Oldest);
    let mut span = None;
    loop {
        match reader.next() {
            Read::Record(Record::RequestSpan(_, r)) if r.session_id == s.id() => span = Some(r),
            Read::Empty => break,
            _ => {}
        }
    }
    assert!(span.is_some());
    assert_eq!(e.parked_tokens(0), 0);
}

#[test]
fn scripted_session_reuses_only_its_exact_prompt_without_generated_history() {
    let (v, _) = fixture(512);
    let mut session = v.session(SessionOptions::default());
    let first = vec![256, 100, 101, 102, 103];
    let (answer, end) = collect(session.generate(Prompt::Tokens(first), params(2)).unwrap());
    assert_eq!(answer, "hi");
    assert_eq!(usage(end, Finish::Length).cached_tokens, 0);
    assert!(session.history().is_empty());
    let scripted = vec![256, 100, 101, 102, 103, 120];
    let u = usage(
        collect(
            session
                .generate(Prompt::Tokens(scripted.clone()), params(2))
                .unwrap(),
        )
        .1,
        Finish::Length,
    );
    assert_eq!((u.prompt_tokens, u.cached_tokens), (6, 5));
    assert!(session.history().is_empty());
    let mut independent = v.session(SessionOptions::default());
    let u = usage(
        collect(
            independent
                .generate(Prompt::Tokens(scripted), params(2))
                .unwrap(),
        )
        .1,
        Finish::Length,
    );
    assert_eq!(u.cached_tokens, 0);
    let stream = session
        .generate(Prompt::Text("caller supplied".into()), params(10))
        .unwrap();
    assert!(matches!(
        session.generate(Prompt::Text("busy".into()), params(1)),
        Err(EngineError::InvalidRequest(invalid::SESSION_BUSY))
    ));
    drop(stream);
    assert!(session.history().is_empty());
    collect(
        session
            .generate(Prompt::Text("after cancellation".into()), params(1))
            .unwrap(),
    );
}

#[test]
fn template_roundtrip_cached_length_never_includes_unfed_last_token() {
    let (v, _) = fixture(512);
    let mut s = v.session(SessionOptions {
        system: Some("system".into()),
        ..SessionOptions::default()
    });
    usage(
        collect(s.send("first", params(2)).unwrap()).1,
        Finish::Length,
    );
    let cached = s.tokens_in_kv();
    let u = usage(
        collect(s.send("second", params(1)).unwrap()).1,
        Finish::Length,
    );
    assert_eq!(u.cached_tokens, cached);
    assert!(u.cached_tokens < u.prompt_tokens);
    assert_eq!(
        s.history()
            .iter()
            .map(|m| m.role.as_str())
            .collect::<Vec<_>>(),
        ["system", "user", "assistant", "user", "assistant"]
    );
}

#[test]
fn template_join_divergence_uses_engine_common_prefix() {
    let template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";
    let (v, _) = fixture_with(&[("b", 1), ("c", 1)], 512, template);
    let mut s = v.session(SessionOptions::default());
    collect(s.send("first", params(2)).unwrap());
    let parked = s.tokens_in_kv();
    let u = usage(
        collect(s.send("second", params(1)).unwrap()).1,
        Finish::Length,
    );
    assert!(
        u.cached_tokens < parked,
        "template no longer reproduces the assistant header"
    );
    assert_eq!(u.cached_tokens, 5);
}

#[test]
fn approved_template_fixtures_roundtrip_the_fully_tokenized_first_turn() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larch-model/tests/data");
    let files: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jinja"))
        .collect();
    assert!(!files.is_empty());
    for file in files {
        let template = fs::read_to_string(&file).unwrap();
        let (v, _) = fixture_with(&[("b", 1)], 1024, &template);
        let mut s = v.session(SessionOptions {
            system: Some("system".into()),
            ..SessionOptions::default()
        });
        collect(s.send("first", params(1)).unwrap());
        let parked = s.tokens_in_kv();
        let u = usage(
            collect(s.send("second", params(1)).unwrap()).1,
            Finish::Length,
        );
        assert_eq!(
            u.cached_tokens, parked,
            "{file:?}: the final sampled token was not fed"
        );

        // Give the unmodified template's assistant header one control token. FakeModel then
        // generates printable b/c instead of whitespace that a template may trim on re-render.
        let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
        let g = Gguf::parse(&write(&metadata(&[], &template), &tensors)).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let chat = ChatTemplate::from_gguf(&g, &tok).unwrap();
        let messages = [Message {
            role: "user".into(),
            content: "first".into(),
        }];
        let rendered = chat.render(&messages, true).unwrap();
        let prefix = chat.render(&messages, false).unwrap();
        let header = rendered.strip_prefix(&prefix).unwrap();
        assert!(!header.is_empty());
        let (v, _) = fixture_with(&[(header, 3), ("b", 1), ("c", 1)], 1024, &template);
        let mut s = v.session(SessionOptions {
            system: Some("system".into()),
            ..SessionOptions::default()
        });
        let (text, end) = collect(s.send("first", params(2)).unwrap());
        assert_eq!(text, "bc", "{file:?}");
        let u = usage(end, Finish::Length);
        let parked = s.tokens_in_kv();
        assert_eq!(parked, u.prompt_tokens + 1);
        let u = usage(
            collect(s.send("second", params(1)).unwrap()).1,
            Finish::Length,
        );
        assert_eq!(
            u.cached_tokens, parked,
            "{file:?}: one answer token was fed"
        );
    }
}

#[test]
fn template_rejection_restores_previous_history_and_parked_kv() {
    let template = "{% for m in messages %}{% if m.content == 'reject' %}{{ raise_exception('refused') }}{% endif %}{{ m.content }}{% endfor %}{{ bos_token }}";
    let (v, _) = fixture_with(&[("b", 1)], 512, template);
    let mut s = v.session(SessionOptions::default());
    collect(s.send("first", params(1)).unwrap());
    let before = s.history();
    let parked = s.tokens_in_kv();
    assert_eq!(
        s.send("reject", params(1)).err(),
        Some(EngineError::InvalidRequest(invalid::CHAT_TEMPLATE))
    );
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), parked);
}

#[test]
fn busy_rejection_invalid_request_rollback_and_drop_after_queued_done() {
    let (v, e) = fixture(512);
    let mut s = v.session(SessionOptions::default());
    let bad = GenParams {
        logprobs: 21,
        ..params(1)
    };
    assert_eq!(
        s.send("bad", bad).err(),
        Some(EngineError::InvalidRequest(invalid::LOGPROBS))
    );
    assert!(s.history().is_empty());
    let id = s.id();
    let stream = s.send("first", params(1)).unwrap();
    assert_eq!(
        s.send("second", params(1)).err(),
        Some(EngineError::InvalidRequest(invalid::SESSION_BUSY))
    );
    // An engine query processes behind submit. The stream remains busy even if Done is queued.
    e.parked_tokens(id);
    drop(stream);
    assert_eq!(e.parked_tokens(id), 0);
    assert_ne!(s.id(), id);
    assert!(s.history()[1].truncated);
    assert_eq!(s.history()[1].content, "");
    let u = usage(
        collect(s.send("next", params(1)).unwrap()).1,
        Finish::Length,
    );
    assert_eq!(u.cached_tokens, 0);
    assert!(
        s.tokens_in_kv() > 0,
        "a new ID can retain future natural turns"
    );
}

struct FailOnThirdForward(FakeModel, u8);
impl Model for FailOnThirdForward {
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
        self.1 += 1;
        if self.1 == 3 {
            return Err(KError::DeviceLost);
        }
        self.0.forward(s, b, kv, logits)
    }
}

#[test]
fn session_failed_turn_rolls_back_history() {
    let (v, e) = fixture_model(
        &[("b", 1), ("c", 1), ("d", 1)],
        20,
        TEMPLATE,
        Box::new(FailOnThirdForward(FakeModel::new(261), 0)),
    );
    let mut s = v.session(SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    });
    collect(s.send("12345", params(1)).unwrap());
    let before = s.history();
    let id = s.id();
    let parked = s.tokens_in_kv();
    assert!(parked > 0);
    // This prompt tentatively drops the previous exchange before engine intake rejects it.
    assert_eq!(
        s.send(
            "6789012345",
            GenParams {
                logprobs: 21,
                ..params(3)
            }
        )
        .err(),
        Some(EngineError::InvalidRequest(invalid::LOGPROBS))
    );
    assert_eq!(s.history(), before);
    assert_eq!(s.id(), id);
    assert_eq!(s.tokens_in_kv(), parked);
    let mut stream = s.send("6789012345", params(3)).unwrap();
    assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "b"));
    assert_eq!(
        stream.next(),
        Some(TextEvent::Error(EngineError::DeviceLost))
    );
    assert_eq!(stream.next(), None);
    drop(stream);
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), 0);
    assert_eq!(e.parked_tokens(id), 0);
    assert_eq!(v.health().state, HealthState::Failed);
    assert_eq!(
        s.send("next", params(1)).err(),
        Some(EngineError::NotReady(HealthState::Failed))
    );
    assert_eq!(s.history(), before);
}

#[test]
fn device_loss_transcript_resume_reprefills_once_then_reuses_kv() {
    let dir = Temp::new();
    let path = larch::chat::transcript_path(&dir.0, "fault").unwrap();
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let (v, e) = fixture_model(
        &[("b", 1), ("c", 1), ("d", 1)],
        512,
        TEMPLATE,
        Box::new(FailOnThirdForward(FakeModel::new(261), 0)),
    );
    let mut s = v.session(options.clone());
    let mut transcript = Transcript::create(&path, options.clone(), s.history()).unwrap();
    let (text, end) = collect(s.send("first", params(1)).unwrap());
    assert_eq!(text, "b");
    let TextEvent::Done {
        finish: Finish::Length,
        usage: first_usage,
        timings,
        ..
    } = end
    else {
        panic!("expected completed first turn");
    };
    transcript
        .append_turn(
            Terminal::Length,
            Some(Metrics {
                prompt_tokens: first_usage.prompt_tokens,
                completion_tokens: first_usage.completion_tokens,
                cached_tokens: first_usage.cached_tokens,
                ttft_ms: timings.ttft_ms,
                tpot_ms: timings.tpot_ms,
                tok_s: timings.tok_s,
            }),
            0,
            pair(&s.history()),
        )
        .unwrap();
    let before = s.history();
    assert!(s.tokens_in_kv() > 0);
    let (partial, end) = collect(s.send("failed", params(3)).unwrap());
    assert_eq!(partial, "b");
    assert_eq!(end, TextEvent::Error(EngineError::DeviceLost));
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), 0);
    assert_eq!(v.health().state, HealthState::Failed);
    transcript
        .append_turn(Terminal::Error, None, 0, None)
        .unwrap();
    drop(transcript);
    drop(s);
    drop(v);
    drop(e);

    let (_transcript, snapshot) = Transcript::resume(&path).unwrap();
    assert_eq!(snapshot.history, before);
    assert_eq!(snapshot.options, options);
    assert_eq!(snapshot.terminal, Some(Terminal::Error));
    assert!(snapshot.metrics.is_none());
    assert_eq!(snapshot.dropped_exchanges, 0);
    let (v, _) = fixture(512);
    assert_eq!(v.health().state, HealthState::Ready);
    let mut s = v.session(SessionOptions::default());
    s.restore(snapshot.options, snapshot.history).unwrap();
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), 0);
    assert_eq!(
        usage(
            collect(s.send("resumed", params(1)).unwrap()).1,
            Finish::Length
        )
        .cached_tokens,
        0
    );
    let parked = s.tokens_in_kv();
    assert!(parked > 0);
    let reused = usage(
        collect(s.send("again", params(1)).unwrap()).1,
        Finish::Length,
    );
    assert_eq!(reused.cached_tokens, parked);
    assert!(reused.cached_tokens < reused.prompt_tokens);
    assert_eq!(s.history()[..before.len()], before);
}

#[test]
fn cancel_and_drop_preserve_only_delivered_partial_text() {
    for explicit in [false, true] {
        let (v, e) = fixture(512);
        let mut s = v.session(SessionOptions::default());
        let old = s.id();
        let mut stream = s.send("first", params(3)).unwrap();
        assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "b"));
        if explicit {
            stream.cancel();
            assert_eq!(
                stream.next(),
                Some(TextEvent::Error(EngineError::Cancelled))
            );
            assert_eq!(stream.next(), None);
        }
        drop(stream);
        assert_eq!(s.history()[1].content, "b");
        assert!(s.history()[1].truncated);
        assert_eq!(e.parked_tokens(old), 0);
        assert_eq!(s.tokens_in_kv(), 0);
        assert_eq!(
            usage(
                collect(s.send("next", params(1)).unwrap()).1,
                Finish::Length
            )
            .cached_tokens,
            0
        );
    }
}

#[test]
fn reset_race_and_session_drop_cannot_resurrect_old_ids() {
    let (v, e) = fixture(512);
    let mut s = v.session(SessionOptions {
        system: Some("system".into()),
        ..SessionOptions::default()
    });
    let old = s.id();
    let mut stream = s.send("old", params(3)).unwrap();
    stream.next().unwrap();
    s.reset();
    let new = s.id();
    assert_ne!(old, new);
    collect(s.send("new", params(1)).unwrap());
    let history = s.history();
    assert_eq!(
        stream.next(),
        Some(TextEvent::Error(EngineError::Cancelled))
    );
    drop(stream);
    assert_eq!(s.history(), history);
    assert_eq!(history[1].content, "new");
    assert_eq!(e.parked_tokens(old), 0);
    assert!(s.tokens_in_kv() > 0);
    drop(s);
    assert_eq!(e.parked_tokens(new), 0);
    let mut s = v.session(SessionOptions::default());
    let id = s.id();
    let active = s.send("drop", params(3)).unwrap();
    drop(s);
    assert_eq!(
        usage(collect(active).1, Finish::Length).completion_tokens,
        3
    );
    assert_eq!(
        collect(
            v.session(SessionOptions::default())
                .send("temporary", params(3))
                .unwrap()
        )
        .0,
        "bcd"
    );
    assert_eq!(e.parked_tokens(id), 0);
}

#[test]
fn cancel_and_drop_with_pending_natural_done_keep_completed_turn() {
    for explicit in [false, true] {
        let (v, e) = fixture_with(&[("<0xC3>part", 6)], 512, TEMPLATE);
        let mut s = v.session(SessionOptions::default());
        let id = s.id();
        let mut stream = s.send("user", params(1)).unwrap();
        assert!(
            matches!(stream.next(), Some(TextEvent::Text { delta, tokens: 1, .. })
            if delta == "�")
        );
        let parked = s.tokens_in_kv();
        assert!(parked > 0);
        if explicit {
            stream.cancel();
            usage(stream.next().unwrap(), Finish::Length);
            assert_eq!(stream.next(), None);
        }
        drop(stream);
        assert_eq!(
            s.history()[1],
            ChatMessage {
                role: Role::Assistant,
                content: "�".into(),
                truncated: false,
                ..ChatMessage::default()
            }
        );
        assert_eq!(s.id(), id);
        assert_eq!(e.parked_tokens(id), parked);
    }
}

#[test]
fn reset_with_pending_done_cannot_complete_old_turn() {
    for (max_tokens, stop, delta) in [(1, "", "�"), (2, "é!", "é"), (3, "é!", "")] {
        let (v, e) = fixture_with(
            &[("<0xC3>part", 6), ("<0xA9>part", 6), ("!", 1)],
            512,
            TEMPLATE,
        );
        let mut s = v.session(SessionOptions {
            system: Some("system".into()),
            ..SessionOptions::default()
        });
        let old_id = s.id();
        let mut old = s
            .send(
                "old",
                GenParams {
                    stop: vec![stop.into()],
                    ..params(max_tokens)
                },
            )
            .unwrap();
        // UTF-8 flush, unmatched stop prefix, and matched StopSeq each leave Done pending.
        assert!(
            matches!(old.next(), Some(TextEvent::Text { delta: text, tokens, .. })
            if text == delta && tokens == max_tokens)
        );
        assert!(s.tokens_in_kv() > 0);
        s.reset();
        assert_ne!(s.id(), old_id);
        usage(collect(s.send("new", params(3)).unwrap()).1, Finish::Length);
        let history = s.history();
        let parked = s.tokens_in_kv();
        assert_eq!(history[1].content, "new");
        assert_eq!(history[2].content, "é!");
        assert!(parked > 0);
        assert_eq!(
            old.next(),
            Some(TextEvent::Error(EngineError::Cancelled)),
            "{max_tokens} tokens, stop {stop:?}"
        );
        assert_eq!(old.next(), None);
        drop(old);
        assert_eq!(s.history(), history);
        assert_eq!(s.tokens_in_kv(), parked);
        assert_eq!(e.parked_tokens(old_id), 0);
    }
}

#[test]
fn uncapped_answers_run_to_the_context_and_keep_an_eighth_free_on_overflow() {
    let (v, _) = fixture(32);
    let end = collect(v.generate(Prompt::Text("12345".into()), params(0)).unwrap()).1;
    assert!(
        matches!(end, TextEvent::Done { finish: Finish::Length, usage: u, .. } if u.prompt_tokens + u.completion_tokens == 32),
        "{end:?}"
    );
    let mut s = v.session(SessionOptions {
        on_overflow: Overflow::Fail,
        ..SessionOptions::default()
    });
    // 28 prompt tokens leave 4 = 32 / 8 for the answer; 29 do not.
    assert!(s.send(&"x".repeat(27), params(0)).is_ok());
    let mut s = v.session(SessionOptions {
        on_overflow: Overflow::Fail,
        ..SessionOptions::default()
    });
    assert!(matches!(
        s.send(&"x".repeat(28), params(0)).err(),
        Some(EngineError::PromptTooLong { limit: 32, .. })
    ));
}

#[test]
fn overflow_drops_whole_exchanges_preserves_system_and_fail_rolls_back() {
    for overflow in [Overflow::DropOldestExchanges, Overflow::Fail] {
        let (v, _) = fixture(20);
        let mut s = v.session(SessionOptions {
            system: Some("sys".into()),
            on_overflow: overflow,
            ..SessionOptions::default()
        });
        collect(s.send("12345", params(2)).unwrap());
        let before = s.history();
        let parked = s.tokens_in_kv();
        let result = s.send("6789012345", params(2));
        if overflow == Overflow::Fail {
            assert!(matches!(
                result.err(),
                Some(EngineError::PromptTooLong { limit: 20, .. })
            ));
            assert_eq!(s.history(), before);
            assert_eq!(s.tokens_in_kv(), parked);
        } else {
            let end = collect(result.unwrap()).1;
            assert!(
                matches!(end, TextEvent::Done { usage: u, timings: t, .. } if t.dropped_exchanges == 1 && u.cached_tokens == 3)
            );
            assert_eq!(s.history().len(), 3);
            assert_eq!(s.history()[0].content, "sys");
            assert_eq!(s.history()[1].content, "6789012345");
        }
        let before = s.history();
        let parked = s.tokens_in_kv();
        assert!(matches!(
            s.send(&"x".repeat(30), params(1)).err(),
            Some(EngineError::PromptTooLong { limit: 20, .. })
        ));
        assert_eq!(s.history(), before);
        assert_eq!(s.tokens_in_kv(), parked);
    }
}

#[test]
fn overflow_drops_multiple_oldest_exchanges_and_shrinks_cached_prefix() {
    let (v, _) = fixture(20);
    let mut s = v.session(SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    });
    for user in ["1111", "2222"] {
        usage(collect(s.send(user, params(2)).unwrap()).1, Finish::Length);
    }
    assert_eq!(s.history().len(), 5);
    let parked = s.tokens_in_kv();
    let (text, end) = collect(s.send("ZZZZZZZZZZ", params(2)).unwrap());
    assert!(matches!(&end, TextEvent::Done { timings, .. } if timings.dropped_exchanges == 2));
    let u = usage(end, Finish::Length);
    assert_eq!(u.cached_tokens, 3);
    assert!(u.cached_tokens < parked);
    assert_eq!(u.prompt_tokens, 14);
    assert_eq!(s.tokens_in_kv(), 15);
    let history = s.history();
    assert_eq!(
        history
            .iter()
            .map(|m| (m.role, m.content.as_str(), m.truncated))
            .collect::<Vec<_>>(),
        [
            (Role::System, "sys", false),
            (Role::User, "ZZZZZZZZZZ", false),
            (Role::Assistant, "bc", false)
        ]
    );
    assert_eq!(text, "bc");
}

#[test]
fn keep_kv_false_and_runtime_lifetime() {
    let (v, e) = fixture(512);
    let mut s = v.session(SessionOptions {
        keep_kv: false,
        ..SessionOptions::default()
    });
    collect(s.send("one", params(1)).unwrap());
    assert_eq!(s.tokens_in_kv(), 0);
    for (p, finish, text) in [
        (params(10), Finish::Eos, "bcd"),
        (
            GenParams {
                stop: vec!["cd".into()],
                ..params(10)
            },
            Finish::StopSeq,
            "b",
        ),
    ] {
        let (delivered, end) = collect(s.send("next", p).unwrap());
        assert_eq!(delivered, text);
        assert_eq!(usage(end, finish).cached_tokens, 0);
        assert_eq!(s.tokens_in_kv(), 0);
        let history = s.history();
        assert_eq!(history.last().unwrap().content, text);
        assert!(!history.last().unwrap().truncated);
    }
    let stream = v.generate(Prompt::Tokens(vec![256]), params(1)).unwrap();
    drop(v);
    drop(s);
    assert_eq!(collect(stream).0, "b");
    assert_ne!(
        e.health().state,
        HealthState::Ready,
        "last facade owner joins engine shutdown"
    );
}

#[test]
fn a_clone_serves_after_the_original_drops_and_the_last_clone_shuts_down() {
    let (v, e) = fixture(512);
    let clone = v.clone();
    drop(v);
    assert_eq!(
        e.health().state,
        HealthState::Ready,
        "a clone keeps the engine"
    );
    let stream = clone
        .generate(Prompt::Tokens(vec![256]), params(1))
        .unwrap();
    assert_eq!(collect(stream).0, "b");
    drop(clone);
    assert_ne!(
        e.health().state,
        HealthState::Ready,
        "the last clone joins engine shutdown"
    );
}

#[test]
fn defaults_follow_effective_configuration_and_explicit_options_override() {
    // Unset sampling values follow the knobs: greedy at temperature_default 0, sampled at 2 (the
    // tiny model's random weights are nearly flat). top_p_default 0, the registry minimum, runs.
    let dir = Temp::new();
    let tiny = Larch::load(LoadOptions::new(tiny_file(&dir), CPU)).unwrap();
    let ids = |p: GenParams| -> Vec<u32> {
        let p = GenParams {
            max_tokens: 8,
            seed: Some(1),
            logprobs: 1,
            ..p
        };
        let stream = tiny.generate(Prompt::Tokens(vec![1]), p).unwrap();
        stream
            .flat_map(|e| match e {
                TextEvent::Text { logprobs, .. } => logprobs.unwrap_or_default(),
                _ => vec![],
            })
            .map(|t| t.id)
            .collect()
    };
    let set = |knob, value| {
        let reply = tiny
            .knobs()
            .set(KnobRequest::new(knob, value, Authority::Human));
        assert!(matches!(reply, KnobReply::Applied { .. }), "{reply:?}");
    };
    let greedy = ids(GenParams {
        temperature: Some(0.0),
        ..GenParams::default()
    });
    set(knob_id::TEMPERATURE_DEFAULT, 0.0);
    assert_eq!(ids(GenParams::default()), greedy);
    set(knob_id::TEMPERATURE_DEFAULT, 2.0);
    set(knob_id::TOP_K_DEFAULT, 0.0);
    set(knob_id::TOP_P_DEFAULT, 1.0);
    assert_ne!(ids(GenParams::default()), greedy);
    set(knob_id::TOP_P_DEFAULT, 0.0);
    assert_eq!(ids(GenParams::default()), greedy);

    let (v, _) = fixture(512);
    let p = GenParams {
        top_p: Some(1.0),
        ..params(1)
    };
    assert_eq!(
        collect(v.generate(Prompt::Tokens(vec![256]), p).unwrap()).0,
        "b"
    );
    let p = GenParams {
        stop: vec![String::new()],
        top_p: Some(1.0),
        ..params(1)
    };
    assert_eq!(
        collect(v.generate(Prompt::Tokens(vec![256]), p).unwrap()).0,
        "b"
    );
    let p = GenParams {
        logprobs: 21,
        ..params(1)
    };
    assert!(matches!(
        v.generate(Prompt::Tokens(vec![256]), p).err(),
        Some(EngineError::InvalidRequest(invalid::LOGPROBS))
    ));
}

#[test]
fn utf8_pieces_and_stop_holdback_use_pr12_detokenizer() {
    let (v, _) = fixture_with(
        &[("<0xC3>part", 6), ("<0xA9>part", 6), ("!", 1)],
        512,
        TEMPLATE,
    );
    let p = GenParams {
        logprobs: 3,
        ..params(3)
    };
    let mut stream = v.generate(Prompt::Tokens(vec![256]), p).unwrap();
    let first = stream.next().unwrap();
    assert!(
        matches!(first, TextEvent::Text { delta, tokens: 2, logprobs: Some(probs) } if delta == "é" && probs.len() == 2 && probs.iter().all(|p| p.top.len() == 3))
    );
    assert!(
        matches!(stream.next(), Some(TextEvent::Text { delta, tokens: 1, .. }) if delta == "!")
    );
    usage(stream.next().unwrap(), Finish::Length);
    assert!(stream.next().is_none());
    assert_eq!(
        collect(v.generate(Prompt::Tokens(vec![256]), params(1)).unwrap()).0,
        "�",
        "natural termination flushes an incomplete UTF-8 piece"
    );
    let p = GenParams {
        stop: vec!["é!".into()],
        ..params(3)
    };
    let (text, end) = collect(v.generate(Prompt::Tokens(vec![256]), p).unwrap());
    assert!(text.is_empty());
    assert!(matches!(
        end,
        TextEvent::Done {
            finish: Finish::StopSeq,
            ..
        }
    ));

    let (v, e) = fixture_with(&[("aS", 1), ("TO", 1), ("Ptail", 1)], 512, TEMPLATE);
    let mut s = v.session(SessionOptions::default());
    let id = s.id();
    let p = GenParams {
        stop: vec!["STOP".into()],
        ..params(3)
    };
    let stream = s.send("user", p).unwrap();
    let started = std::time::Instant::now();
    while e.parked_tokens(id) == 0 {
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        std::thread::yield_now();
    }
    // Length is already queued; finish_stop must still acknowledge StopSeq with that usage.
    let (text, end) = collect(stream);
    assert_eq!(text, "a");
    let stopped = usage(end, Finish::StopSeq);
    assert_eq!(stopped.completion_tokens, 3);
    assert_eq!(s.history()[1].content, "a");
    assert!(!s.history()[1].truncated);
    let parked = e.parked_tokens(id);
    assert_eq!(parked, stopped.prompt_tokens + 2);
    assert_eq!(id, s.id());
    let mut fresh = v.session(SessionOptions::default());
    fresh
        .restore(SessionOptions::default(), s.history())
        .unwrap();
    let p = GenParams {
        seed: Some(7),
        logprobs: 5,
        ..params(2)
    };
    let (a, a_tokens, a_end) = output(s.send("next", p.clone()).unwrap());
    let (b, b_tokens, b_end) = output(fresh.send("next", p).unwrap());
    assert_eq!(a, b);
    same_output_tokens(&a_tokens, &b_tokens);
    let reused = usage(a_end, Finish::Length);
    assert_eq!(usage(b_end, Finish::Length).cached_tokens, 0);
    assert_eq!(reused.cached_tokens, 5);
    assert!(reused.cached_tokens > 0 && reused.cached_tokens <= parked);
    assert_eq!(s.history(), fresh.history());
    let p = GenParams {
        stop: vec!["STOP".into()],
        ..params(2)
    };
    assert_eq!(
        collect(v.generate(Prompt::Tokens(vec![256]), p).unwrap()).0,
        "aSTO",
        "natural terminal flushes an unmatched stop prefix"
    );
}

#[test]
fn cancellation_does_not_flush_undelivered_holdback_and_empty_stop_prefix() {
    let (v, _) = fixture_with(&[("aS", 1), ("TO", 1), ("P", 1)], 512, TEMPLATE);
    let mut s = v.session(SessionOptions::default());
    let mut stream = s
        .send(
            "user",
            GenParams {
                stop: vec!["STOP".into()],
                ..params(3)
            },
        )
        .unwrap();
    assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "a"));
    drop(stream);
    assert_eq!(s.history()[1].content, "a");
    assert!(s.history()[1].truncated);
    let p = GenParams {
        stop: vec!["aSTOP".into()],
        ..params(3)
    };
    let (text, end) = collect(v.generate(Prompt::Tokens(vec![256]), p).unwrap());
    assert_eq!(text, "");
    assert!(matches!(
        end,
        TextEvent::Done {
            finish: Finish::StopSeq,
            ..
        }
    ));
    // Dropped after the last text, with Done pending: the turn is complete, not cancelled.
    let mut s = v.session(SessionOptions::default());
    let id = s.id();
    let p = GenParams {
        stop: vec!["STOP".into()],
        ..params(3)
    };
    let mut stream = s.send("user", p).unwrap();
    assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "a"));
    assert!(
        matches!(stream.next(), Some(TextEvent::Text { delta, tokens: 2, .. }) if delta.is_empty())
    );
    drop(stream);
    assert!(!s.history()[1].truncated);
    assert_eq!(s.id(), id);
    assert!(s.tokens_in_kv() > 0);
}

/// A policy with a system message, and a history of the system then one cancelled exchange.
fn conversation() -> (SessionOptions, Vec<ChatMessage>) {
    let options = SessionOptions {
        system: Some("system".into()),
        ..SessionOptions::default()
    };
    let history = vec![
        ChatMessage {
            role: Role::System,
            content: "system".into(),
            truncated: false,
            ..ChatMessage::default()
        },
        ChatMessage {
            role: Role::User,
            content: "user\n\"é\"".into(),
            truncated: false,
            ..ChatMessage::default()
        },
        ChatMessage {
            role: Role::Assistant,
            content: "partial".into(),
            truncated: true,
            ..ChatMessage::default()
        },
    ];
    (options, history)
}

fn pair(history: &[ChatMessage]) -> Option<[ChatMessage; 2]> {
    history.last_chunk::<2>().cloned()
}

#[test]
fn transcripts_roundtrip_partial_tail_locking_and_resume_fresh_kv() {
    let dir = Temp::new();
    let path = dir.0.join("turns.jsonl");
    let (options, history) = conversation();
    let mut transcript = Transcript::create(&path, options.clone(), history[..1].to_vec()).unwrap();
    transcript
        .append_turn(Terminal::Cancelled, None, 0, pair(&history))
        .unwrap();
    assert!(Transcript::create(&path, options, vec![]).is_err());
    assert!(
        Transcript::resume(&path)
            .err()
            .unwrap()
            .to_string()
            .contains("is in use by another larch chat"),
        "one active transcript writer"
    );
    drop(transcript);
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"partial\"")
        .unwrap();
    let (transcript, latest) = Transcript::resume(&path).unwrap();
    assert_eq!(latest.history, history);
    drop(transcript);
    assert!(fs::read(&path).unwrap().ends_with(b"\n"));
    let (mut transcript, _) = Transcript::resume(&path).unwrap();
    let (v, _) = fixture(512);
    let mut s = v.session(SessionOptions::default());
    s.restore(latest.options.clone(), latest.history.clone())
        .unwrap();
    assert_eq!(s.history(), history);
    assert_eq!(s.tokens_in_kv(), 0);
    collect(s.send("next", params(1)).unwrap());
    let next = s.history();
    let metrics = Metrics {
        prompt_tokens: 20,
        completion_tokens: 1,
        cached_tokens: 0,
        ttft_ms: 1.0,
        tpot_ms: 0.0,
        tok_s: 0.0,
    };
    transcript
        .append_turn(Terminal::Length, Some(metrics), 0, pair(&next))
        .unwrap();
    drop(transcript);
    assert_eq!(Transcript::resume(&path).unwrap().1.history, next);
    assert!(larch::chat::transcript_path(&dir.0, "../escape").is_err());
}

#[test]
fn transcript_corruption_is_rejected_without_truncating_complete_lines() {
    let dir = Temp::new();
    // create makes the missing directories too.
    let path = dir.0.join("new/bad.jsonl");
    let (options, history) = conversation();
    drop(Transcript::create(&path, options, history.clone()).unwrap());
    let header = fs::read_to_string(&path).unwrap();
    assert!(!header.contains("terminal"), "the header records no turn");
    let (_, fresh) = Transcript::resume(&path).unwrap();
    assert_eq!((fresh.history, fresh.terminal), (history, None));
    let mut bad: serde_json::Value = serde_json::from_str(&header).unwrap();
    bad["history"][1]["role"] = "tool".into();
    let bytes = format!("{bad}\npartial");
    fs::write(&path, &bytes).unwrap();
    assert!(Transcript::resume(&path).is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), bytes);
}

fn tiny_file(dir: &Temp) -> PathBuf {
    tiny_file_with_template(dir, TEMPLATE)
}

fn tiny_file_with_template(dir: &Temp, template: &str) -> PathBuf {
    tiny_arch_file(dir, Arch::Llama, template)
}

fn tiny_arch_file(dir: &Temp, arch: Arch, template: &str) -> PathBuf {
    tiny_arch_file_ctx(dir, arch, template, None)
}

/// [`tiny_arch_file`] trained for `context` tokens, where given.
fn tiny_arch_file_ctx(dir: &Temp, arch: Arch, template: &str, context: Option<u32>) -> PathBuf {
    let bytes = tiny::gguf(arch, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            (
                t.name.clone(),
                t.dims.clone(),
                t.ggml_type,
                bytes[at..at + t.bytes as usize].to_vec(),
            )
        })
        .collect();
    let mut kv: Vec<_> = g.kv.into_iter().collect();
    for (k, v) in &mut kv {
        if let (true, Some(c)) = (k.ends_with(".context_length"), context) {
            *v = Value::U32(c);
        }
    }
    kv.extend(metadata(&[], template).into_iter().filter(|(k, _)| {
        !matches!(
            k.as_str(),
            "tokenizer.ggml.tokens"
                | "tokenizer.ggml.token_type"
                | "tokenizer.ggml.bos_token_id"
                | "tokenizer.ggml.eos_token_id"
        )
    }));
    kv.extend([
        (
            "tokenizer.ggml.tokens".into(),
            Value::Arr(
                (0..256)
                    .map(|b| Value::Str(format!("<0x{b:02X}>")))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Arr(vec![Value::I32(6); 256]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(97)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(255)),
    ]);
    let path = dir.0.join("tiny.gguf");
    fs::write(&path, write(&kv, &tensors)).unwrap();
    path
}

/// What `ModelClass::active_bytes` is for a GGUF: its tensors, less the embedding table, whose
/// rows a step gathers, when the output projection is a tensor of its own.
fn read_per_step(path: &Path) -> u64 {
    let g = Gguf::parse(&fs::read(path).unwrap()).unwrap();
    assert!(g.tensor("output.weight").is_some(), "untied output");
    let all: u64 = g.tensors.iter().map(|t| t.bytes).sum();
    all - g.tensor("token_embd.weight").unwrap().bytes
}

fn dense(active_bytes: u64) -> ModelClass {
    ModelClass {
        dense: true,
        active_bytes,
        experts_in_ram: false,
    }
}

#[test]
fn model_class_of_the_tiny_gguf() {
    let dir = Temp::new();
    let path = tiny_file(&dir);
    let v = Larch::load(LoadOptions::new(path.clone(), CPU)).unwrap();
    assert_eq!(v.model_class(), dense(read_per_step(&path)));
    assert!(v.model_class().active_bytes < v.deployment().weights_bytes);
    // A test model on the CPU took no device memory to load: 0 is unknown.
    let (fake, _) = fixture(64);
    assert_eq!(fake.model_class(), dense(0));
}

#[test]
fn model_class_follows_reload() {
    let dir = Temp::new();
    let llama = tiny_file(&dir);
    let qwen2 = dir.0.join("qwen2.gguf");
    fs::write(&qwen2, tiny::gguf(Arch::Qwen2, 3)).unwrap();
    let (a, b) = (read_per_step(&llama), read_per_step(&qwen2));
    assert_ne!(a, b, "the two models read different bytes");
    let g = Gguf::parse(&fs::read(&llama).unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let e = start(EngineOptions::new(llama.clone(), CPU)).unwrap();
    let v = Larch::from_engine(e.clone(), tok, template);
    assert_eq!(v.model_class(), dense(a));
    let to = |model: &PathBuf| ReloadChange {
        model: Some(model.clone()),
        ..ReloadChange::default()
    };
    e.reload(to(&qwen2)).unwrap();
    assert_eq!(v.model_class(), dense(b));
    let fake = |_: &_, _: &_, _| Ok(Box::new(FakeModel::new(tiny::VOCAB)) as Box<dyn Model>);
    e.reload_with(ReloadChange::default(), Box::new(fake))
        .unwrap();
    assert_eq!(v.model_class(), dense(0));
    e.reload(to(&llama)).unwrap();
    assert_eq!(v.model_class(), dense(a));
}

/// A Qwen 3.5 session's next turn runs on from the recurrent state at the end of its last prompt
/// without the generation prompt, which the next prompt repeats where the rendered answer does not,
/// and answers as a fresh session does.
#[test]
fn hybrid_session_runs_on_from_its_history() {
    let template = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}>{% endif %}";
    let dir = Temp::new();
    let path = tiny_arch_file(&dir, Arch::Qwen35, template);
    let g = Gguf::parse(&fs::read(&path).unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let chat = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let first = Message {
        role: "user".into(),
        content: "first turn".into(),
    };
    let history = tok.encode(&chat.render(&[first], false).unwrap(), true);
    let v = Larch::load(LoadOptions::new(path, CPU)).unwrap();
    let p = GenParams {
        logprobs: 5,
        ..params(4)
    };
    let options = SessionOptions::default();
    let mut cached = v.session(options.clone());
    output(cached.send("first turn", p.clone()).unwrap());
    let mut fresh = v.session(options.clone());
    fresh.restore(options, cached.history()).unwrap();
    let (a, a_tokens, a_end) = output(cached.send("second", p.clone()).unwrap());
    let (b, b_tokens, b_end) = output(fresh.send("second", p).unwrap());
    assert_eq!(a, b);
    same_output_tokens(&a_tokens, &b_tokens);
    let a_usage = usage(a_end, Finish::Length);
    assert_eq!(a_usage.cached_tokens as usize, history.len());
    assert_eq!(usage(b_end, Finish::Length).cached_tokens, 0);
}

/// A kept session's `generate` parks a hybrid model's state at the history end, as `send` does:
/// the template renders a past reply unlike its generation, so a snapshot at the reply's end
/// would never be reused.
#[test]
fn hybrid_generate_parks_at_the_history_end() {
    let template = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}>{% endif %}";
    let dir = Temp::new();
    let path = tiny_arch_file(&dir, Arch::Qwen35, template);
    let g = Gguf::parse(&fs::read(&path).unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let chat = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let message = |role: &str, content: &str| Message {
        role: role.into(),
        content: content.into(),
    };
    let first = vec![message("user", "first turn")];
    let history = tok.encode(&chat.render(&first, false).unwrap(), true).len();
    let v = Larch::load(LoadOptions::new(path, CPU)).unwrap();
    let mut session = v.session(SessionOptions::default());
    let mut cached = |messages: &[Message]| {
        let stream = session.generate(Prompt::Messages(messages.to_vec()), params(4));
        let (reply, end) = collect(stream.unwrap());
        (reply, usage(end, Finish::Length).cached_tokens as usize)
    };
    let (reply, _) = cached(&first);
    // A dropped reply, asked again, resumes from the snapshot.
    assert_eq!(cached(&first).1, history);
    let mut next = first;
    next.extend([message("assistant", &reply), message("user", "second")]);
    assert_eq!(cached(&next).1, history);
}

#[test]
fn cpu_session_cached_and_fresh_output_identical() {
    for template in [
        TEMPLATE,
        "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}",
    ] {
        let dir = Temp::new();
        let path = tiny_file_with_template(&dir, template);
        let g = Gguf::parse(&fs::read(&path).unwrap()).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let chat = ChatTemplate::from_gguf(&g, &tok).unwrap();
        let prompt = |history: &[ChatMessage], user: &str| {
            let mut messages: Vec<_> = history
                .iter()
                .map(|m| Message {
                    role: m.role.as_str().into(),
                    content: m.content.clone(),
                })
                .collect();
            messages.push(Message {
                role: "user".into(),
                content: user.into(),
            });
            tok.encode(&chat.render(&messages, true).unwrap(), true)
        };
        let v = Larch::load(LoadOptions::new(path, CPU)).unwrap();
        let options = SessionOptions::default();
        let mut cached = v.session(options.clone());
        let p = GenParams {
            seed: Some(7),
            logprobs: 5,
            ..params(4)
        };
        let mut fed = prompt(&[], "first turn crosses the block boundary");
        let (text, tokens, end) = output(
            cached
                .send("first turn crosses the block boundary", p.clone())
                .unwrap(),
        );
        let u = usage(end, Finish::Length);
        assert_eq!(tokens.len(), 4);
        assert_eq!(cached.history()[1].content, text);
        fed.extend(tokens.iter().take(tokens.len() - 1).map(|t| t.id));
        let parked = cached.tokens_in_kv();
        assert_eq!(parked, u.prompt_tokens + u.completion_tokens - 1);
        assert_eq!(parked as usize, fed.len());
        assert!(parked > 16 && !parked.is_multiple_of(16));
        let history = cached.history();
        let next = prompt(&history, "second");
        let lcp = fed
            .iter()
            .zip(&next)
            .take_while(|(a, b)| a == b)
            .count()
            .min(parked as usize)
            .min(next.len() - 1);
        assert!(lcp > 0);
        if template != TEMPLATE {
            assert!(!lcp.is_multiple_of(16), "reuse must end inside a block");
            assert!(
                lcp < parked as usize,
                "whole-render divergence must re-prefill"
            );
        }
        let mut fresh = v.session(options.clone());
        fresh.restore(options, history).unwrap();
        assert_eq!(fresh.tokens_in_kv(), 0);
        let (a, a_tokens, a_end) = output(cached.send("second", p.clone()).unwrap());
        let (b, b_tokens, b_end) = output(fresh.send("second", p).unwrap());
        assert_eq!(a, b);
        same_output_tokens(&a_tokens, &b_tokens);
        let a_usage = usage(a_end, Finish::Length);
        let b_usage = usage(b_end, Finish::Length);
        assert_eq!(a_usage.cached_tokens as usize, lcp);
        assert_eq!(a_usage.prompt_tokens as usize, next.len());
        assert_eq!(b_usage.cached_tokens, 0);
        assert_eq!(a_usage.completion_tokens as usize, a_tokens.len());
        assert_eq!(a_usage.prompt_tokens, b_usage.prompt_tokens);
        assert_eq!(cached.history(), fresh.history());
    }
}

#[test]
fn cpu_session_cancelled_continuation_matches_fresh() {
    let dir = Temp::new();
    let v = Larch::load(LoadOptions::new(tiny_file(&dir), CPU)).unwrap();
    for explicit in [false, true] {
        let options = SessionOptions::default();
        let mut s = v.session(options.clone());
        let mut stream = s.send("first", params(16)).unwrap();
        let Some(TextEvent::Text { delta, .. }) = stream.next() else {
            panic!("expected partial text");
        };
        assert!(!delta.is_empty());
        if explicit {
            stream.cancel();
            assert_eq!(
                stream.next(),
                Some(TextEvent::Error(EngineError::Cancelled))
            );
        }
        drop(stream);
        assert_eq!(s.history()[1].content, delta);
        assert!(s.history()[1].truncated);
        assert_eq!(s.tokens_in_kv(), 0);
        let mut fresh = v.session(options.clone());
        fresh.restore(options, s.history()).unwrap();
        let p = GenParams {
            seed: Some(7),
            logprobs: 5,
            ..params(4)
        };
        let (a, a_tokens, a_end) = output(s.send("next", p.clone()).unwrap());
        let (b, b_tokens, b_end) = output(fresh.send("next", p).unwrap());
        assert_eq!(a, b);
        same_output_tokens(&a_tokens, &b_tokens);
        assert_eq!(usage(a_end, Finish::Length).cached_tokens, 0);
        assert_eq!(usage(b_end, Finish::Length).cached_tokens, 0);
        assert_eq!(s.history(), fresh.history());
    }
}

fn chat_options(dir: &Temp) -> larch::chat::Options {
    larch::chat::Options {
        model: tiny_file(dir),
        drafter: None,
        drafter_auto: None,
        mmproj: None,
        cpu: true,
        device: None,
        max_tokens: Some(1),
        temperature: Some(0.0),
        stop: vec![],
        set: vec!["max_context=512".into()],
        system: None,
        no_keep_kv: false,
        fail_overflow: false,
        resume: None,
        data_dir: Some(dir.0.clone()),
    }
}

/// Runs on a zero-GPU host only; `validate_gpu`'s unit tests cover the decision everywhere.
#[test]
fn zero_gpu_chat_falls_back_once_but_explicit_device_still_errors() {
    if larch_kernels::device_count().unwrap() != 0 {
        return;
    }
    let dir = Temp::new();
    let (model, dir) = (&tiny_file(&dir), &dir.0);
    let warning = "larch: no usable GPU found; running on the CPU reference (slow)";
    for args in [
        &[][..],
        &["--cpu"][..],
        &["--device", "0"][..],
        &["--device", "1"][..],
    ] {
        let data = dir.join(format!("run-{}", args.join("-")));
        let output = Command::new(env!("CARGO_BIN_EXE_larch"))
            .arg("chat")
            .args(args)
            .arg("-m")
            .arg(model)
            .arg("--data-dir")
            .arg(&data)
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        if args.first() == Some(&"--device") {
            assert!(!output.status.success(), "{stderr}");
            assert!(!stderr.contains(warning), "{stderr}");
            assert!(!data.exists());
        } else {
            assert!(output.status.success(), "{stderr}");
            assert_eq!(
                stderr.lines().filter(|line| *line == warning).count(),
                usize::from(args.is_empty()),
                "{stderr}"
            );
            assert_eq!(
                stderr.contains("CPU reference/debug mode"),
                !args.is_empty(),
                "{stderr}"
            );
            assert_eq!(fs::read_dir(data.join("chat")).unwrap().count(), 1);
        }
    }
}

// Integration-test dependencies are not compiled with cfg(test).
const FAULT_INJECT: bool = cfg!(debug_assertions) || cfg!(feature = "fault-inject");

/// #74: a fault in chat prints the engine's fault line and the caller's error, then exits 3
/// within 9.11's 100 ms of detection.
#[test]
fn chat_cli_fault_reports_fault_line_and_caller_error_before_exit() {
    for (trigger, caller) in [
        ("panic@step:1", "internal error"),
        ("device_lost@step:1", "device lost"),
    ] {
        let _alone = cpu_alone();
        let dir = Temp::with(None);
        let model = tiny_file(&dir);
        let mut child = Command::new(env!("CARGO_BIN_EXE_larch"))
            .args(["chat", "--cpu", "--max-tokens", "2", "-m"])
            .arg(&model)
            .arg("--data-dir")
            .arg(&dir.0)
            .env("LARCH_FAULT", trigger)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        // stdin stays open, so only the fault (or /quit when injection is compiled out) ends chat.
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(if FAULT_INJECT {
                b"hi\n"
            } else {
                b"hi\n/quit\n"
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                let _ = child.kill();
                panic!("{trigger}: chat did not exit");
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let ended = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        drop(stdin);
        let result = child.wait_with_output().unwrap();
        let err = String::from_utf8_lossy(&result.stderr);
        let out = String::from_utf8_lossy(&result.stdout);
        if !FAULT_INJECT {
            assert!(result.status.success(), "{trigger}: {err}");
            continue;
        }
        assert_eq!(result.status.code(), Some(3), "{trigger}: {err}\n{out}");
        let detected = err
            .lines()
            .find_map(|line| {
                line.strip_prefix("larch fault=")?
                    .split_once(" detected_ns=")?
                    .1
                    .parse::<u128>()
                    .ok()
            })
            .unwrap_or_else(|| panic!("{trigger}: no fault line: {err}"));
        assert!(
            ended >= detected && ended - detected <= 100_000_000,
            "{trigger}: exit {} ms after detection: {err}",
            (ended.saturating_sub(detected)) / 1_000_000
        );
        assert!(
            err.lines().any(|line| line.starts_with(caller)),
            "{trigger}: {err}"
        );
        assert!(
            out.contains("Error · engine usage/timings unavailable"),
            "{trigger}: {out}"
        );
        let path = fs::read_dir(dir.0.join("chat"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let journal = fs::read_to_string(&path).unwrap();
        let last: serde_json::Value =
            serde_json::from_str(journal.lines().last().unwrap()).unwrap();
        assert_eq!(last["terminal"], "Error", "{trigger}: {journal}");
    }
}

#[test]
fn gpu_preflight_fails_before_model_io_or_journal_creation() {
    let count = larch_kernels::device_count().unwrap();
    let opening_error = if count == 0 {
        match larch_kernels::Device::open(0) {
            Ok(device) => {
                drop(device);
                None
            }
            // No driver, or one that cannot report its version, is no GPU (chat.rs validate_gpu).
            Err(
                larch_kernels::KError::InvalidArg | larch_kernels::KError::Driver { have: 0, .. },
            ) => None,
            Err(error) => Some(error),
        }
    } else {
        None
    };
    for ordinal in [-1, count] {
        let dir = Temp::new();
        let options = larch::chat::Options {
            cpu: false,
            device: Some(ordinal),
            ..chat_options(&dir)
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        let error =
            larch::chat::run(options, Cursor::new(b"/quit\n"), &mut out, &mut err).unwrap_err();
        // `{:#}` keeps anyhow's cause chain; `to_string` prints only the outermost context.
        let text = format!("{error:#}");
        if ordinal < 0 {
            assert!(text.contains("invalid GPU ordinal -1"), "{text}");
            assert!(text.contains("--cpu"), "{text}");
        } else if count == 0 {
            match &opening_error {
                Some(larch_kernels::KError::Driver { have, need }) if *have > 0 => assert_eq!(
                    error.downcast_ref::<larch::LoadError>(),
                    Some(&larch::LoadError::DriverTooOld {
                        have: *have,
                        need: *need
                    })
                ),
                Some(opening_error) => {
                    assert!(text.contains("opening GPU device"), "{text}");
                    assert!(text.contains(&opening_error.to_string()), "{text}");
                }
                None => {
                    assert!(text.contains("no GPU device"), "{text}");
                    assert!(text.contains("--cpu"), "{text}");
                }
            }
        } else {
            assert!(
                text.contains(&format!("invalid GPU ordinal {count}: {count} available")),
                "{text}"
            );
        }
        assert!(out.is_empty() && err.is_empty());
        assert!(!dir.0.join("chat").exists());
    }
}

#[test]
fn real_cpu_transformer_chat_cli_transcript_reset_resume_and_registration() {
    let dir = Temp::new();
    let model = tiny_file(&dir);
    let v = Larch::load(LoadOptions::new(model.clone(), CPU)).unwrap();
    let end = collect(v.generate(Prompt::Text("hi".into()), params(2)).unwrap()).1;
    assert_eq!(usage(end, Finish::Length).completion_tokens, 2);
    drop(v);
    let mut child = Command::new(env!("CARGO_BIN_EXE_larch"))
        .args([
            "chat",
            "--cpu",
            "--max-tokens",
            "2",
            "--temperature",
            "0",
            "-m",
        ])
        .arg(&model)
        .arg("--data-dir")
        .arg(&dir.0)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // A fresh process: its first human trial is 1 << 63, so /ab stop can name it.
    child.stdin.take().unwrap().write_all(b"hi\n/status\n/knob get temperature_default\n/ab max_batch_seqs=2\n/ab stop 9223372036854775808\n/reset\nagain\n/quit\n").unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = String::from_utf8(result.stdout).unwrap();
    assert!(
        out.contains("TTFT")
            && out.contains("cached")
            && out.contains("Ready")
            && out.contains("conversation reset"),
        "{out}"
    );
    // The real engine's A/B (9.9) starts under the human trial id and stops by it.
    assert!(out.contains("trial 9223372036854775808: Started"), "{out}");
    assert!(out.contains("trial 9223372036854775808: Stopped"), "{out}");
    let err = String::from_utf8(result.stderr).unwrap();
    assert_eq!(
        err.lines()
            .filter(|line| *line == larch::supervise::READY_LINE)
            .count(),
        1
    );
    let path = fs::read_dir(dir.0.join("chat"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let (_, last) = Transcript::resume(&path).unwrap();
    assert_eq!(last.history.len(), 2);
    assert_eq!(last.history[0].content, "again");
    // One header, then a record per turn or reset. hi is never sent when the worker has read the
    // /reset after it first; otherwise its turn is recorded, cancelled or complete.
    let records: Vec<serde_json::Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let terminals: Vec<_> = records
        .iter()
        .map(|r| r["terminal"].as_str().unwrap())
        .collect();
    assert!(
        matches!(
            terminals[..],
            ["Reset", "Length" | "Eos"] | [_, "Reset", "Length" | "Eos"]
        ),
        "{terminals:?}"
    );
    let id = path.file_stem().unwrap().to_str().unwrap();
    let options = larch::chat::Options {
        model,
        resume: Some(id.into()),
        ..chat_options(&dir)
    };
    let mut out = Vec::new();
    let mut err = Vec::new();
    larch::chat::run(options, Cursor::new(b"next\n/quit\n"), &mut out, &mut err).unwrap();
    let (_, resumed) = Transcript::resume(&path).unwrap();
    assert_eq!(resumed.history.len(), 4);
    assert_eq!(resumed.metrics.unwrap().cached_tokens, 0);
    for args in [
        &["knob", "list"][..],
        &["info"][..],
        &["tokcheck", "--help"][..],
    ] {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_larch"))
                .args(args)
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    {
        let unsupported = Command::new(env!("CARGO_BIN_EXE_larch"))
            .args(["chat", "--tune", "-m"])
            .arg(tiny_file(&dir))
            .output()
            .unwrap();
        assert!(!unsupported.status.success());
        assert!(String::from_utf8_lossy(&unsupported.stderr).contains("'--tune'"));
    }
}

#[test]
fn rejected_startup_never_prints_ready_and_unknown_controls_reject() {
    let dir = Temp::new();
    let options = larch::chat::Options {
        set: vec![
            "max_context=512".into(),
            "decode_graphs=true".into(), // valid in the registry, no graphs on the CPU
            "max_context=100".into(),
        ],
        ..chat_options(&dir)
    };
    let mut err = Vec::new();
    let result = larch::chat::run(options, Cursor::new(b""), &mut Vec::new(), &mut err);
    assert_eq!(
        result.unwrap_err().to_string(),
        "--set decode_graphs=1 rejected: Unsupported; \
         --set max_context=100 rejected: OutOfRange { min: 512.0, max: 131072.0 }"
    );
    assert!(
        !String::from_utf8(err)
            .unwrap()
            .contains(larch::supervise::READY_LINE)
    );
    assert!(larch::knob::assignment("unknown=1").is_err());
    assert_eq!(
        larch::knob::assignment("schedule_policy=balanced").unwrap(),
        (knob_id::SCHEDULE_POLICY, 1.0)
    );
    let (v, _) = fixture(512);
    let mut output = Vec::new();
    larch::knob::command(
        v.knobs().as_ref(),
        &["set", "max_context", "1024"],
        &mut output,
    )
    .unwrap();
    assert!(String::from_utf8(output).unwrap().contains("NotLive"));
}

/// `/ab` gives every start its own non-zero trial id (0 is no trial, which the engine rejects), runs
/// it until `/ab stop <trial>`, and prints the id that stop needs.
#[test]
fn ab_starts_carry_a_trial_id_and_stop_names_it() {
    struct Recorder(std::sync::Mutex<Vec<AbCommand>>);
    impl KnobControl for Recorder {
        fn registry(&self) -> &'static [KnobSpec] {
            unreachable!()
        }
        fn effective(&self) -> std::sync::Arc<EffectiveConfig> {
            unreachable!()
        }
        fn set(&self, _: KnobRequest) -> KnobReply {
            unreachable!()
        }
        fn revert(&self, _: RevertTarget, _: Authority) -> KnobReply {
            unreachable!()
        }
        fn ab(&self, cmd: AbCommand) -> AbReply {
            self.0.lock().unwrap().push(cmd);
            AbReply::Rejected {
                reason: RejectReason::Unsupported,
            }
        }
        fn health(&self) -> Health {
            unreachable!()
        }
    }
    let control = Recorder(std::sync::Mutex::new(Vec::new()));
    let mut out = Vec::new();
    larch::knob::ab(&control, &["max_batch_seqs=2"], &mut out).unwrap();
    larch::knob::ab(&control, &["max_batch_seqs=4", "50"], &mut out).unwrap();
    let starts = control.0.lock().unwrap().clone();
    let [
        AbCommand::Start {
            trial_id: first,
            period_steps: 200,
            max_steps: 0,
            ..
        },
        AbCommand::Start {
            trial_id: second,
            period_steps: 50,
            max_steps: 0,
            ..
        },
    ] = starts[..]
    else {
        panic!("{starts:?}")
    };
    assert!(
        first >= 1 << 63 && second != first,
        "human trials stay out of gait's range"
    );
    larch::knob::ab(&control, &["stop", &second.to_string()], &mut out).unwrap();
    assert_eq!(
        control.0.lock().unwrap()[2],
        AbCommand::Stop { trial_id: second }
    );
    assert!(
        String::from_utf8(out)
            .unwrap()
            .contains(&format!("trial {second}: Rejected"))
    );
    assert!(larch::knob::ab(&control, &["max_batch_seqs=2", "0"], &mut Vec::new()).is_err());
}

#[test]
fn public_load_preserves_engine_errors() {
    let dir = Temp::new();
    let model = tiny_file(&dir);
    for kind in 0..2 {
        let mut opts = LoadOptions::new(model.clone(), CPU);
        match kind {
            0 => opts.ring_capacity = 3,
            1 => opts.device = 99, // No such device on any backend.
            _ => unreachable!(),
        }
        let expected = larch_engine::start(opts.clone()).err().unwrap();
        assert!(
            matches!(expected, larch::LoadError::Backend(_)),
            "rejected before the weights load: {expected:?}"
        );
        assert_eq!(Larch::load(opts).err().unwrap(), expected);
    }
}

struct BrokenOutput {
    remaining: usize,
    accepted: Vec<u8>,
}
impl Write for BrokenOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        let n = self.remaining.min(bytes.len());
        self.remaining -= n;
        self.accepted.extend_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn output_disconnect_persists_a_cancelled_partial_turn() {
    for remaining in [2, 3, 4, 6] {
        let dir = Temp::new();
        let options = larch::chat::Options {
            max_tokens: Some(128),
            ..chat_options(&dir)
        };
        let mut output = BrokenOutput {
            remaining,
            accepted: Vec::new(),
        };
        let result = larch::chat::run(options, Cursor::new(b"hi\n"), &mut output, &mut Vec::new());
        assert!(result.is_err());
        let path = fs::read_dir(dir.0.join("chat"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let (_, snapshot) = Transcript::resume(&path).unwrap();
        assert_eq!(snapshot.terminal, Some(Terminal::Cancelled));
        assert!(snapshot.history[1].truncated);
        let bytes = &output.accepted[2..]; // Exclude the REPL prompt.
        let valid = std::str::from_utf8(bytes).map_or_else(|e| e.valid_up_to(), str::len);
        assert_eq!(snapshot.history[1].content.as_bytes(), &bytes[..valid]);
    }
}

// Deliver /reset only after the first answer delta. The writer waits until the
// worker has read past the command lines, so cancellation cannot depend on
// scheduler timing.
struct ResetInput {
    step: u8,
    first_command: &'static [u8],
    tail: &'static [u8],
    answer: std::sync::mpsc::Receiver<()>,
    queued: std::sync::mpsc::Sender<()>,
}
impl io::Read for ResetInput {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let line: &[u8] = match self.step {
            0 => b"hi\n",
            1 => {
                self.answer
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                if self.first_command.starts_with(b"/quit") {
                    self.queued.send(()).unwrap();
                }
                self.first_command
            }
            2 => {
                self.queued.send(()).unwrap();
                self.tail
            }
            _ => return Ok(0),
        };
        assert!(bytes.len() >= line.len());
        bytes[..line.len()].copy_from_slice(line);
        self.step += 1;
        Ok(line.len())
    }
}
struct ResetOutput {
    bytes: Vec<u8>,
    answer: Option<std::sync::mpsc::Sender<()>>,
    queued: std::sync::mpsc::Receiver<()>,
}
impl Write for ResetOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.len() >= 2
            && let Some(answer) = self.answer.take()
        {
            answer.send(()).unwrap();
            self.queued
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn cli_reset_cancels_an_active_answer_and_persists_cleared_history() {
    let cleared: &[&str] = &["Cancelled", "Reset"];
    for (first_command, tail, expected) in [
        (b"/reset\n".as_slice(), b"/quit\n".as_slice(), cleared),
        (b"\n/reset\n".as_slice(), b"/quit\n".as_slice(), cleared),
        (
            b"/status\n/status\n/reset\n".as_slice(),
            b"/quit\n".as_slice(),
            cleared,
        ),
        // A skipped non-UTF-8 line is not a prompt, so the /reset after it still cancels.
        (b"\xff\n/reset\n".as_slice(), b"/quit\n".as_slice(), cleared),
        (
            b"\xff\xfe\n/status\n/reset\n".as_slice(),
            b"/quit\n".as_slice(),
            cleared,
        ),
        // A queued prompt a /reset follows is never sent; the answer streaming now is not cancelled.
        (
            b"b\n/reset\nc\n/reset\n".as_slice(),
            b"/quit\n".as_slice(),
            &["Length", "Reset", "Reset"],
        ),
        (b"/quit\n/reset\n".as_slice(), b"".as_slice(), &["Length"]),
    ] {
        let dir = Temp::new();
        let options = larch::chat::Options {
            // One token guarantees a natural Done is generated even if reset sets
            // the engine cancel flag during its final step; facade delivery must cancel.
            // Without a reset, 128 tokens would show a wrongly applied one.
            max_tokens: Some(if expected == ["Length"] { 128 } else { 1 }),
            system: Some("system".into()),
            ..chat_options(&dir)
        };
        let (answer, heard_answer) = std::sync::mpsc::channel();
        let (queued, heard_queued) = std::sync::mpsc::channel();
        let input = io::BufReader::new(ResetInput {
            step: 0,
            first_command,
            tail,
            answer: heard_answer,
            queued,
        });
        let mut output = ResetOutput {
            bytes: vec![],
            answer: Some(answer),
            queued: heard_queued,
        };
        larch::chat::run(options, input, &mut output, &mut Vec::new()).unwrap();
        let path = fs::read_dir(dir.0.join("chat"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let records: Vec<serde_json::Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .skip(1)
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        // A natural end is Length or Eos; resume checks each record's exchange and metrics.
        let terminals: Vec<_> = records
            .iter()
            .map(|r| match r["terminal"].as_str().unwrap() {
                "Eos" => "Length",
                terminal => terminal,
            })
            .collect();
        assert_eq!(terminals, expected, "{first_command:?}");
        let (_, resumed) = Transcript::resume(&path).unwrap();
        if expected.ends_with(&["Reset"]) {
            assert_eq!(
                resumed.history,
                vec![ChatMessage {
                    role: Role::System,
                    content: "system".into(),
                    truncated: false,
                    ..ChatMessage::default()
                }]
            );
        }
    }
}

#[test]
fn identical_resume_id_creates_then_restores_policy_and_reprefills() {
    let dir = Temp::new();
    let path = larch::chat::transcript_path(&dir.0, "work").unwrap();
    // Whitespace-only lines are ignored, a non-UTF-8 line is skipped, and " /quit " is a command,
    // in the worker and in run.
    for prompt in [
        b"first\n   \n\xff\n /quit \n".as_slice(),
        b"second\n/quit\n".as_slice(),
    ] {
        let options = larch::chat::Options {
            resume: Some("work".into()),
            system: Some("system".into()),
            fail_overflow: true,
            ..chat_options(&dir)
        };
        larch::chat::run(
            options,
            Cursor::new(prompt),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();
    }
    let (transcript, snapshot) = Transcript::resume(&path).unwrap();
    assert_eq!(snapshot.history.len(), 5);
    assert_eq!(snapshot.history[1].content, "first");
    assert_eq!(snapshot.history[3].content, "second");
    assert_eq!(snapshot.options.on_overflow, Overflow::Fail);
    assert_eq!(snapshot.metrics.unwrap().cached_tokens, 0);
    drop(transcript);
    let options = larch::chat::Options {
        resume: Some("work".into()),
        system: Some("different".into()),
        ..chat_options(&dir)
    };
    assert!(
        larch::chat::run(options, Cursor::new(b""), &mut Vec::new(), &mut Vec::new())
            .unwrap_err()
            .to_string()
            .contains("persisted policy")
    );
}

/// Yields `first`, then panics if `panic`, else reads EOF; counts reads and reports its drop.
struct LineReader {
    first: &'static str,
    panic: bool,
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    dropped: std::sync::mpsc::Sender<()>,
}
impl io::Read for LineReader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        unreachable!()
    }
}
impl io::BufRead for LineReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        unreachable!()
    }
    fn consume(&mut self, _: usize) {
        unreachable!()
    }
    fn read_line(&mut self, line: &mut String) -> io::Result<usize> {
        if self.reads.fetch_add(1, Ordering::SeqCst) == 0 {
            line.push_str(self.first);
            return Ok(self.first.len());
        }
        assert!(!self.panic, "reader failure");
        Ok(0)
    }
}
impl Drop for LineReader {
    fn drop(&mut self) {
        let _ = self.dropped.send(());
    }
}

/// `/quit` stops the input worker before another read, and a reader that panics fails chat instead
/// of hanging it or exiting as if quit.
#[test]
fn quit_stops_reading_and_a_panicking_reader_ends_chat() {
    for (first, panic) in [("/quit\n", false), ("hi\n", true)] {
        let dir = Temp::new();
        let reads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (dropped, reader_dropped) = std::sync::mpsc::channel();
        let input = LineReader {
            first,
            panic,
            reads: reads.clone(),
            dropped,
        };
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result =
                larch::chat::run(chat_options(&dir), input, &mut Vec::new(), &mut Vec::new());
            done.send(result.map_err(|e| e.to_string())).unwrap();
        });
        let timeout = std::time::Duration::from_secs(60);
        let result = finished.recv_timeout(timeout).unwrap();
        if panic {
            assert_eq!(result.unwrap_err(), "input reader panicked");
        } else {
            result.unwrap();
        }
        reader_dropped.recv_timeout(timeout).unwrap();
        assert_eq!(reads.load(Ordering::SeqCst), if panic { 2 } else { 1 });
    }
}

#[test]
fn compact_journal_replays_overflow_reset_and_rejects_corrupt_earlier_records() {
    let dir = Temp::new();
    let path = dir.0.join("journal.jsonl");
    let (options, mut current) = conversation();
    let mut transcript = Transcript::create(&path, options, current.clone()).unwrap();
    for i in 0..10 {
        current.extend([
            ChatMessage {
                role: Role::User,
                content: format!("question {i}"),
                truncated: false,
                ..ChatMessage::default()
            },
            ChatMessage {
                role: Role::Assistant,
                content: "reply".into(),
                truncated: true,
                ..ChatMessage::default()
            },
        ]);
        transcript
            .append_turn(Terminal::Cancelled, None, 0, pair(&current))
            .unwrap();
    }
    current.drain(1..5);
    current.extend([
        ChatMessage {
            role: Role::User,
            content: "after overflow".into(),
            truncated: false,
            ..ChatMessage::default()
        },
        ChatMessage {
            role: Role::Assistant,
            content: "partial".into(),
            truncated: true,
            ..ChatMessage::default()
        },
    ]);
    transcript
        .append_turn(Terminal::Cancelled, None, 2, pair(&current))
        .unwrap();
    // A record that cannot replay is never written.
    assert!(
        transcript
            .append_turn(Terminal::Length, None, 0, pair(&current))
            .is_err()
    );
    drop(transcript);
    let bytes = fs::read_to_string(&path).unwrap();
    assert_eq!(bytes.matches("options").count(), 1);
    assert_eq!(bytes.matches("question 0").count(), 1);
    let (mut transcript, latest) = Transcript::resume(&path).unwrap();
    assert_eq!(latest.history, current);
    assert_eq!(latest.dropped_exchanges, 2);
    transcript.append_reset().unwrap();
    drop(transcript);
    assert_eq!(Transcript::resume(&path).unwrap().1.history, current[..1]);
    let bytes = fs::read_to_string(&path)
        .unwrap()
        .replacen("\"version\":2", "\"version\":99", 1);
    fs::write(&path, &bytes).unwrap();
    assert!(Transcript::resume(&path).is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), bytes);
}

#[test]
fn cancelled_overflow_turn_persists_removed_exchanges() {
    let dir = Temp::new();
    let path = larch::chat::transcript_path(&dir.0, "overflow").unwrap();
    let (options, mut history) = conversation();
    history[1].content = "x".repeat(450);
    drop(Transcript::create(&path, options, history).unwrap());
    let options = larch::chat::Options {
        resume: Some("overflow".into()),
        max_tokens: Some(128),
        ..chat_options(&dir)
    };
    let mut output = BrokenOutput {
        remaining: 2,
        accepted: vec![],
    };
    let error = larch::chat::run(
        options,
        Cursor::new(b"next\n"),
        &mut output,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(error.downcast_ref::<io::Error>().is_some());
    let (_, resumed) = Transcript::resume(&path).unwrap();
    assert_eq!(resumed.dropped_exchanges, 1);
    assert_eq!(resumed.terminal, Some(Terminal::Cancelled));
    assert_eq!(resumed.history[1].content, "next");
    assert!(resumed.history[2].content.is_empty() && resumed.history[2].truncated);
}

#[test]
fn text_token_totals_include_hidden_eos_and_empty_stop_deltas_with_or_without_logprobs() {
    let (v, _) = fixture(512);
    for logprobs in [0, 3] {
        for stop in [vec![], vec!["bcd".into()]] {
            let mut tokens = 0;
            let mut probabilities = 0;
            let has_stop = !stop.is_empty();
            for event in v
                .generate(
                    Prompt::Tokens(vec![256]),
                    GenParams {
                        logprobs,
                        stop,
                        ..params(10)
                    },
                )
                .unwrap()
            {
                if let TextEvent::Text {
                    tokens: count,
                    logprobs,
                    ..
                } = event
                {
                    tokens += count;
                    probabilities += logprobs.map_or(0, |values| values.len());
                }
            }
            let expected = if has_stop { 3 } else { 4 };
            assert_eq!(tokens, expected);
            assert_eq!(
                probabilities,
                if logprobs == 0 { 0 } else { expected as usize }
            );
        }
    }
}

#[test]
fn telemetry_off_loads_and_streams_through_the_public_facade() {
    let dir = Temp::new();
    let mut options = LoadOptions::new(tiny_file(&dir), CPU);
    options.telemetry = larch_engine::TelemetryTier::Off;
    let v = Larch::load(options).unwrap();
    let usage = usage(
        collect(v.generate(Prompt::Text("hi".into()), params(2)).unwrap()).1,
        Finish::Length,
    );
    assert_eq!(usage.completion_tokens, 2);
    assert!(!matches!(
        v.telemetry().reader(ReadFrom::Oldest).next(),
        Read::Record(_)
    ));
}

#[test]
fn resume_repairs_interrupted_initialization_but_preserves_complete_corruption() {
    for bytes in [
        b"".as_slice(),
        b"{\"version\":1".as_slice(),
        b"bad\npartial".as_slice(),
    ] {
        let dir = Temp::new();
        fs::create_dir(dir.0.join("chat")).unwrap();
        let path = larch::chat::transcript_path(&dir.0, "initial").unwrap();
        fs::write(&path, bytes).unwrap();
        let options = larch::chat::Options {
            resume: Some("initial".into()),
            system: Some("system".into()),
            ..chat_options(&dir)
        };
        let result = larch::chat::run(
            options,
            Cursor::new(b"hi\n/quit\n"),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        if bytes.contains(&b'\n') {
            assert!(result.is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        } else {
            result.unwrap();
            let (_, resumed) = Transcript::resume(&path).unwrap();
            assert_eq!(resumed.options.system.as_deref(), Some("system"));
            assert_eq!(resumed.history[1].content, "hi");
        }
    }
}

fn message(role: Role, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.into(),
        ..ChatMessage::default()
    }
}

fn call(id: &str) -> ToolCall {
    ToolCall {
        name: "f".into(),
        arguments: serde_json::json!({"path": "a"}),
        id: id.into(),
    }
}

/// An assistant message that calls `f` under `id`.
fn calling(id: &str) -> ChatMessage {
    ChatMessage {
        tool_calls: vec![call(id)],
        ..message(Role::Assistant, "")
    }
}

/// The tool message that answers the call `id`.
fn result(id: &str, content: &str) -> ChatMessage {
    ChatMessage {
        tool_call_id: id.into(),
        ..message(Role::Tool, content)
    }
}

#[test]
fn history_accepts_tool_exchanges_and_rejects_broken_ones() {
    let (v, _) = fixture(512);
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let user = || message(Role::User, "u");
    let answer = || message(Role::Assistant, "a");
    let cases = [
        (vec![user(), calling("1"), result("1", "r"), answer()], true),
        // Waiting for the tool's result.
        (vec![user(), calling("1")], true),
        (
            vec![
                user(),
                calling("1"),
                result("1", "r"),
                calling("2"),
                result("2", "r"),
                answer(),
            ],
            true,
        ),
        // Templates without IDs: both sides empty.
        (vec![user(), calling(""), result("", "r"), answer()], true),
        // A call left unanswered before the next prompt; a template may still refuse it.
        (vec![user(), calling("1"), user(), answer()], true),
        (vec![user(), answer(), result("", "r"), answer()], false),
        (
            vec![user(), calling("1"), result("2", "r"), answer()],
            false,
        ),
        (
            vec![
                user(),
                ChatMessage {
                    truncated: true,
                    ..calling("1")
                },
            ],
            false,
        ),
        (
            vec![
                user(),
                calling("1"),
                ChatMessage {
                    truncated: true,
                    ..result("1", "r")
                },
                answer(),
            ],
            false,
        ),
        (
            vec![
                ChatMessage {
                    tool_calls: vec![call("1")],
                    ..user()
                },
                answer(),
            ],
            false,
        ),
        (
            vec![
                user(),
                ChatMessage {
                    tool_call_id: "1".into(),
                    ..answer()
                },
            ],
            false,
        ),
        (
            vec![
                user(),
                calling("1"),
                result("1", "r"),
                result("1", "r"),
                answer(),
            ],
            false,
        ),
        (vec![user(), calling("1"), result("1", "r")], false),
        (vec![result("", "r"), answer()], false),
        (
            vec![user(), calling("1"), message(Role::System, "sys")],
            false,
        ),
    ];
    let mut s = v.session(options.clone());
    // The system message carries no tool fields either.
    let sys = ChatMessage {
        tool_call_id: "1".into(),
        ..message(Role::System, "sys")
    };
    assert_eq!(
        s.restore(options.clone(), vec![sys, user(), answer()])
            .err(),
        Some(EngineError::InvalidRequest(invalid::HISTORY))
    );
    for (rest, valid) in cases {
        let history: Vec<_> = std::iter::once(message(Role::System, "sys"))
            .chain(rest)
            .collect();
        let restored = s.restore(options.clone(), history.clone());
        if valid {
            restored.unwrap();
            assert_eq!(s.history(), history);
        } else {
            assert_eq!(
                restored.err(),
                Some(EngineError::InvalidRequest(invalid::HISTORY)),
                "{history:?}"
            );
        }
    }
}

#[test]
fn a_written_answer_to_a_call_keeps_the_parked_kv() {
    const HISTORY: Option<EngineError> = Some(EngineError::InvalidRequest(invalid::HISTORY));
    let (v, _e) = fixture(512);
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let mut s = v.session(options);
    // Only a call of the last answer can be answered.
    assert_eq!(s.answer_tool_call("1", "r", "a", false).err(), HISTORY);
    collect(s.send("question", params(1)).unwrap());
    assert_eq!(s.answer_tool_call("1", "r", "a", false).err(), HISTORY);
    s.set_tool_calls("", call("1")).unwrap();
    assert_eq!(s.answer_tool_call("2", "r", "a", false).err(), HISTORY);
    let (id, parked) = (s.id(), s.tokens_in_kv());
    s.answer_tool_call("1", "r", "the answer", false).unwrap();
    // The result and the answer are in the history; the session and its KV are the same.
    let history = s.history();
    let last: Vec<_> = history[history.len() - 2..]
        .iter()
        .map(|m| (m.role, m.content.as_str()))
        .collect();
    assert_eq!(last, [(Role::Tool, "r"), (Role::Assistant, "the answer")]);
    assert_eq!((s.id(), s.tokens_in_kv()), (id, parked));
    // Answered once: the call has its result now.
    assert_eq!(s.answer_tool_call("1", "r", "again", false).err(), HISTORY);
    // The next turn goes on from there.
    let (_, end) = collect(s.send("next", params(1)).unwrap());
    assert!(!matches!(end, TextEvent::Error(_)), "{end:?}");
}

#[test]
fn tool_result_turns_follow_every_terminal_row() {
    const HISTORY: Option<EngineError> = Some(EngineError::InvalidRequest(invalid::HISTORY));
    let (v, e) = fixture(512);
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let mut s = v.session(options.clone());
    // Only an answer carries calls, and only a call of the last answer can be answered.
    assert_eq!(s.set_tool_calls("", call("1")).err(), HISTORY);
    collect(s.send("question", params(1)).unwrap());
    assert_eq!(s.send_tool_result("1", "r", params(1)).err(), HISTORY);
    // Set once, with arguments the template can parse.
    let text = |arguments: &str| ToolCall {
        arguments: arguments.into(),
        ..call("1")
    };
    assert_eq!(
        s.set_tool_calls("", text("{not json")).err(),
        Some(EngineError::InvalidRequest(invalid::CHAT_TEMPLATE))
    );
    s.set_tool_calls("b", text(r#"{"path": "a"}"#)).unwrap();
    assert_eq!(s.set_tool_calls("b", call("1")).err(), HISTORY);
    let before = s.history();
    let parked = s.tokens_in_kv();
    assert_eq!(s.send_tool_result("2", "r", params(1)).err(), HISTORY);
    // Rejected at intake: the tool message is removed and the parked sequence stays.
    let p = GenParams {
        logprobs: 21,
        ..params(1)
    };
    assert_eq!(
        s.send_tool_result("1", "r", p).err(),
        Some(EngineError::InvalidRequest(invalid::LOGPROBS))
    );
    assert_eq!((s.history(), s.tokens_in_kv()), (before.clone(), parked));

    // Length: the answer is appended and the turn reuses all the KV the last one parked.
    let stream = s.send_tool_result("1", "r", params(2)).unwrap();
    for busy in [
        s.send("x", params(1)).err(),
        s.send_tool_result("1", "r", params(1)).err(),
        s.set_tool_calls("", call("1")).err(),
    ] {
        assert_eq!(
            busy,
            Some(EngineError::InvalidRequest(invalid::SESSION_BUSY))
        );
    }
    let (text, end) = collect(stream);
    assert_eq!(text, "bc");
    let u = usage(end, Finish::Length);
    assert_eq!(u.cached_tokens, parked);
    assert_eq!(
        s.history()[before.len()..],
        [result("1", "r"), message(Role::Assistant, "bc")]
    );
    assert_eq!(s.tokens_in_kv(), u.prompt_tokens + 1);

    // Eos.
    s.set_tool_calls("bc", call("2")).unwrap();
    let parked = s.tokens_in_kv();
    let (text, end) = collect(s.send_tool_result("2", "r", params(10)).unwrap());
    assert_eq!(text, "bcd");
    assert_eq!(usage(end, Finish::Eos).cached_tokens, parked);

    // StopSeq: the text before the stop string, parked under the same ID.
    s.set_tool_calls("bcd", call("3")).unwrap();
    let id = s.id();
    let p = GenParams {
        stop: vec!["c".into()],
        ..params(3)
    };
    let (text, end) = collect(s.send_tool_result("3", "r", p).unwrap());
    assert_eq!(text, "b");
    usage(end, Finish::StopSeq);
    assert_eq!(s.history().last(), Some(&message(Role::Assistant, "b")));
    assert!(s.tokens_in_kv() > 0);
    assert_eq!(s.id(), id);

    // Cancelled: the delivered part stays, truncated, and the KV is freed under a new ID.
    s.set_tool_calls("b", call("4")).unwrap();
    let mut stream = s.send_tool_result("4", "r", params(3)).unwrap();
    assert!(matches!(stream.next(), Some(TextEvent::Text { delta, .. }) if delta == "b"));
    drop(stream);
    let history = s.history();
    let truncated = ChatMessage {
        truncated: true,
        ..message(Role::Assistant, "b")
    };
    assert_eq!(history[history.len() - 2..], [result("4", "r"), truncated]);
    assert_eq!(e.parked_tokens(id), 0);
    assert_eq!(s.tokens_in_kv(), 0);
    assert_ne!(s.id(), id);
    assert_eq!(s.set_tool_calls("b", call("5")).err(), HISTORY);
    let mut resumed = v.session(SessionOptions::default());
    resumed.restore(options, history).unwrap();
}

#[test]
fn tool_result_fault_restores_the_waiting_exchange() {
    let (v, e) = fixture_model(
        &[("b", 1), ("c", 1), ("d", 1)],
        512,
        TEMPLATE,
        Box::new(FailOnThirdForward(FakeModel::new(261), 0)),
    );
    let mut s = v.session(SessionOptions::default());
    collect(s.send("question", params(1)).unwrap());
    s.set_tool_calls("b", call("1")).unwrap();
    let before = s.history();
    let id = s.id();
    let (partial, end) = collect(s.send_tool_result("1", "r", params(3)).unwrap());
    assert_eq!(partial, "b");
    assert_eq!(end, TextEvent::Error(EngineError::DeviceLost));
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), 0);
    assert_eq!(e.parked_tokens(id), 0);
}

/// FakeModel under the Qwen 3 architecture, the one family that takes several calls per turn.
struct Qwen3Fake(FakeModel, ModelConfig);
impl Qwen3Fake {
    fn new(vocab: u32) -> Self {
        let fake = FakeModel::new(vocab);
        let cfg = ModelConfig {
            arch: Arch::Qwen3,
            ..fake.config().clone()
        };
        Self(fake, cfg)
    }
}
impl Model for Qwen3Fake {
    fn config(&self) -> &ModelConfig {
        &self.1
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.0.forward(s, b, kv, logits)
    }
}

/// The qwen3 template with the generation header as one token, answered by `b` and EOS.
const QWEN3_TAILS: [(&str, i32); 2] = [("<|im_start|>assistant\n", 4), ("b", 1)];

fn qwen3_session() -> (Larch, larch::Session) {
    let (v, _) = fixture_model(&QWEN3_TAILS, 4096, QWEN3, Box::new(Qwen3Fake::new(260)));
    let mut s = v.session(SessionOptions::default());
    collect(s.send("q", params(4)).unwrap());
    s.set_several_tool_calls("b", vec![call("a"), call("b")])
        .unwrap();
    (v, s)
}

fn results(ids: &[&str]) -> Vec<(String, String)> {
    ids.iter()
        .map(|id| (id.to_string(), format!("r{id}")))
        .collect()
}

#[test]
fn two_calls_two_results_render_in_order_as_one_turn() {
    let (_v, mut s) = qwen3_session();
    let before = s.history();
    let parked = s.tokens_in_kv();
    // Given out of order, the results go in as the calls are ordered.
    let stream = s
        .send_tool_results(results(&["b", "a"]), params(4))
        .unwrap();
    let (text, end) = collect(stream);
    assert_eq!(text, "b");
    let u = usage(end, Finish::Eos);
    let mut sent = before.clone();
    sent.extend([result("a", "ra"), result("b", "rb")]);
    let mut history = sent.clone();
    history.push(message(Role::Assistant, "b"));
    assert_eq!(s.history(), history);
    assert_eq!(s.last_turn(), history[before.len()..]);
    // One turn: its request reused all the KV the question's turn parked, and its prompt holds
    // both results (below).
    assert_eq!(u.cached_tokens, parked);
    // The prompt is the template's render of the history with both calls and both results.
    let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let g = Gguf::parse(&write(&metadata(&QWEN3_TAILS, QWEN3), &tensors)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let messages: Vec<_> = sent
        .iter()
        .map(|m| larch_model::ToolMessage {
            role: m.role.as_str().into(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect();
    let text = ChatTemplate::from_gguf(&g, &tok)
        .unwrap()
        .render_tools(&messages, &[], true)
        .unwrap();
    assert_eq!(text.matches("<tool_call>").count(), 2, "{text}");
    let (a, b) = (
        text.find("<tool_response>\nra"),
        text.find("<tool_response>\nrb"),
    );
    assert!(a.is_some() && a < b, "{text}");
    assert_eq!(u.prompt_tokens as usize, tok.encode(&text, true).len());
}

#[test]
fn a_result_for_an_unknown_call_changes_nothing() {
    const HISTORY: Option<EngineError> = Some(EngineError::InvalidRequest(invalid::HISTORY));
    let (_v, mut s) = qwen3_session();
    let before = s.history();
    let parked = s.tokens_in_kv();
    for ids in [&["a", "x"][..], &["x"], &["a", "a"], &["a"], &["b"], &[]] {
        assert_eq!(
            s.send_tool_results(results(ids), params(4)).err(),
            HISTORY,
            "{ids:?}"
        );
        assert_eq!((s.history(), s.tokens_in_kv()), (before.clone(), parked));
    }
    // Calls the results could not be told apart by, and no calls, are refused too.
    collect(
        s.send_tool_results(results(&["a", "b"]), params(4))
            .unwrap(),
    );
    for calls in [vec![call("c"), call("c")], vec![]] {
        let error = s.set_several_tool_calls("", calls).unwrap_err();
        assert_eq!(error.downcast_ref::<EngineError>(), HISTORY.as_ref());
    }
    s.set_several_tool_calls("", vec![call("c")]).unwrap();
}

#[test]
fn two_calls_on_a_llama_model_are_refused_by_family() {
    let (v, _) = fixture(512);
    let mut s = v.session(SessionOptions::default());
    collect(s.send("question", params(1)).unwrap());
    let before = s.history();
    let error = s
        .set_several_tool_calls("", vec![call("1"), call("2")])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("llama") && error.contains("one tool call"),
        "{error}"
    );
    assert_eq!(s.history(), before);
    // One call is the one-call form's.
    s.set_several_tool_calls("b", vec![call("1")]).unwrap();
    assert_eq!(
        s.history().last(),
        Some(&ChatMessage {
            tool_calls: vec![call("1")],
            ..message(Role::Assistant, "b")
        })
    );
}

#[test]
fn restore_takes_one_result_per_call_in_the_calls_order() {
    let (qwen, _) = fixture_model(&QWEN3_TAILS, 4096, QWEN3, Box::new(Qwen3Fake::new(260)));
    let mut s = qwen.session(SessionOptions::default());
    let two = ChatMessage {
        tool_calls: vec![call("a"), call("b")],
        ..message(Role::Assistant, "")
    };
    let exchange = |rest: &[ChatMessage]| {
        let mut history = vec![message(Role::User, "q"), two.clone()];
        history.extend_from_slice(rest);
        history.push(message(Role::Assistant, "bc"));
        history
    };
    let (a, b) = (result("a", "ra"), result("b", "rb"));
    for (rest, valid) in [
        (vec![a.clone(), b.clone()], true),
        (vec![b.clone()], false),
        (vec![a.clone()], false),
        (vec![b.clone(), a.clone()], false),
        (vec![a.clone(), a.clone()], false),
        (vec![a.clone(), result("x", "r")], false),
    ] {
        let history = exchange(&rest);
        let restored = s.restore(SessionOptions::default(), history.clone());
        assert_eq!(restored.is_ok(), valid, "{history:?}");
    }
    // The last turn of a restored history is its results and their answer.
    assert_eq!(s.last_turn(), exchange(&[a.clone(), b.clone()])[2..]);
    // A llama model takes no answer with several calls, restored or set.
    let (llama, _) = fixture(512);
    assert_eq!(
        llama
            .session(SessionOptions::default())
            .restore(SessionOptions::default(), exchange(&[a, b]))
            .err(),
        Some(EngineError::InvalidRequest(invalid::HISTORY))
    );
}

/// The question, then an answer that calls `f` and `g` under the one ID `a`.
fn one_id_twice() -> [ChatMessage; 2] {
    let g = ToolCall {
        name: "g".into(),
        ..call("a")
    };
    let calling = ChatMessage {
        tool_calls: vec![call("a"), g],
        ..message(Role::Assistant, "")
    };
    [message(Role::User, "q"), calling]
}

#[test]
fn restore_refuses_two_calls_with_one_id() {
    let (_v, mut s) = qwen3_session();
    let before = (s.history(), s.id(), s.tokens_in_kv());
    assert!(before.2 > 0, "the question's turn parked its KV");
    let mut answered = one_id_twice().to_vec();
    answered.extend([
        result("a", "first"),
        result("a", "second"),
        message(Role::Assistant, "done"),
    ]);
    for history in [one_id_twice().to_vec(), answered] {
        assert_eq!(
            s.restore(SessionOptions::default(), history.clone()).err(),
            Some(EngineError::InvalidRequest(invalid::HISTORY)),
            "{history:?}"
        );
        assert_eq!((s.history(), s.id(), s.tokens_in_kv()), before);
    }
    // One call with no ID, for templates without call IDs, stays a history.
    let lone = ChatMessage {
        tool_calls: vec![call("")],
        ..message(Role::Assistant, "")
    };
    let history = vec![
        message(Role::User, "q"),
        lone,
        result("", "r"),
        message(Role::Assistant, "done"),
    ];
    s.restore(SessionOptions::default(), history).unwrap();
}

#[test]
fn a_journal_refuses_two_calls_with_one_id() {
    let dir = Temp::new();
    let options = SessionOptions::default();
    let metrics = Metrics {
        prompt_tokens: 2,
        completion_tokens: 1,
        cached_tokens: 0,
        ttft_ms: 1.0,
        tpot_ms: 1.0,
        tok_s: 1.0,
    };
    // A header: refused before its file exists.
    let path = dir.0.join("header.jsonl");
    assert!(Transcript::create(&path, options.clone(), one_id_twice().to_vec()).is_err());
    assert!(!path.exists());
    // An appended record, either way: refused unwritten.
    let path = dir.0.join("append.jsonl");
    // The file is read only while unlocked: Windows locks its bytes, not just the handle.
    drop(Transcript::create(&path, options.clone(), vec![]).unwrap());
    let bytes = fs::read(&path).unwrap();
    let (mut transcript, _) = Transcript::resume(&path).unwrap();
    let appended = [
        transcript.append_turn(
            Terminal::Eos,
            Some(metrics.clone()),
            0,
            Some(one_id_twice()),
        ),
        transcript.append_messages(
            Terminal::Eos,
            Some(metrics.clone()),
            0,
            one_id_twice().to_vec(),
        ),
    ];
    assert!(appended.iter().all(Result::is_err), "{appended:?}");
    drop(transcript);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    // Replay: a journal whose header or record holds such an answer does not resume, unchanged.
    let header = |history: Vec<ChatMessage>| {
        serde_json::json!({"version": 1, "options": options, "history": history}).to_string()
    };
    let record = serde_json::json!({
        "version": 2,
        "terminal": "Eos",
        "metrics": metrics,
        "dropped_exchanges": 0,
        "exchange": one_id_twice(),
    });
    for (name, journal) in [
        ("replay-header", header(one_id_twice().to_vec())),
        ("replay-record", format!("{}\n{record}", header(vec![]))),
    ] {
        let path = dir.0.join(format!("{name}.jsonl"));
        let journal = format!("{journal}\n");
        fs::write(&path, &journal).unwrap();
        let resumed = Transcript::resume(&path).map(|_| ());
        assert!(resumed.is_err(), "{name}");
        assert_eq!(fs::read(&path).unwrap(), journal.as_bytes(), "{name}");
    }
}

#[test]
fn overflow_drops_a_tool_exchange_as_one_unit() {
    // One token per character, one for each answer's header and one for the generation prompt.
    let (v, _) = fixture(30);
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let sys = message(Role::System, "sys");
    let tool_exchange = [
        message(Role::User, "1111"),
        calling("1"),
        result("1", "rrrr"),
        message(Role::Assistant, "bc"),
    ];
    let plain = [message(Role::User, "2222"), message(Role::Assistant, "bc")];
    let mut s = v.session(options.clone());
    let history: Vec<_> = [sys.clone()]
        .into_iter()
        .chain(tool_exchange.clone())
        .chain(plain.clone())
        .collect();
    s.restore(options.clone(), history).unwrap();
    // 3 + 12 + 7 + 11 prompt tokens and 2 to generate: 35; without the tool exchange, 23.
    let (_, end) = collect(s.send("ZZZZZZZZZZ", params(2)).unwrap());
    assert!(matches!(end, TextEvent::Done { timings, .. } if timings.dropped_exchanges == 1));
    let mut expected = vec![sys.clone()];
    expected.extend(plain.clone());
    expected.extend([
        message(Role::User, "ZZZZZZZZZZ"),
        message(Role::Assistant, "bc"),
    ]);
    assert_eq!(s.history(), expected);

    // A tool turn drops older exchanges, never its own.
    let history: Vec<_> = [sys.clone()]
        .into_iter()
        .chain(plain.clone())
        .chain(tool_exchange[..2].iter().cloned())
        .collect();
    s.restore(options.clone(), history).unwrap();
    let (_, end) = collect(s.send_tool_result("1", &"r".repeat(13), params(2)).unwrap());
    assert!(matches!(end, TextEvent::Done { timings, .. } if timings.dropped_exchanges == 1));
    let mut expected = vec![sys];
    expected.extend(tool_exchange[..2].iter().cloned());
    expected.extend([result("1", &"r".repeat(13)), message(Role::Assistant, "bc")]);
    assert_eq!(s.history(), expected);

    // Only the current tool exchange is left: the turn fails and changes nothing.
    let mut s = v.session(options);
    collect(s.send("1111", params(1)).unwrap());
    s.set_tool_calls("b", call("1")).unwrap();
    let before = s.history();
    let parked = s.tokens_in_kv();
    assert!(matches!(
        s.send_tool_result("1", &"r".repeat(30), params(2)).err(),
        Some(EngineError::PromptTooLong { limit: 30, .. })
    ));
    assert_eq!(s.history(), before);
    assert_eq!(s.tokens_in_kv(), parked);
}

#[test]
fn transcripts_roundtrip_tool_exchanges_and_their_drops() {
    let dir = Temp::new();
    let metrics = || {
        Some(Metrics {
            prompt_tokens: 20,
            completion_tokens: 1,
            cached_tokens: 0,
            ttft_ms: 1.0,
            tpot_ms: 0.0,
            tok_s: 0.0,
        })
    };
    // As a caller records a tool exchange: each turn after its calls are set.
    let path = dir.0.join("tools.jsonl");
    let (v, _) = fixture(512);
    let options = SessionOptions {
        system: Some("sys".into()),
        ..SessionOptions::default()
    };
    let mut s = v.session(options.clone());
    let mut transcript = Transcript::create(&path, options.clone(), s.history()).unwrap();
    collect(s.send("question", params(1)).unwrap());
    s.set_tool_calls("", call("1")).unwrap();
    transcript
        .append_turn(Terminal::Length, metrics(), 0, pair(&s.history()))
        .unwrap();
    collect(s.send_tool_result("1", "result \"é\"", params(1)).unwrap());
    transcript
        .append_turn(Terminal::Length, metrics(), 0, pair(&s.history()))
        .unwrap();
    drop(transcript);
    let (mut transcript, resumed) = Transcript::resume(&path).unwrap();
    assert_eq!(resumed.history, s.history());
    assert_eq!(resumed.history[2].tool_calls, [call("1")]);
    assert_eq!(resumed.history[3], result("1", "result \"é\""));

    // A turn that dropped the tool exchange replays as one drop of all four messages.
    let next = [
        message(Role::User, "next"),
        ChatMessage {
            truncated: true,
            ..message(Role::Assistant, "partial")
        },
    ];
    transcript
        .append_turn(Terminal::Cancelled, None, 1, Some(next.clone()))
        .unwrap();
    drop(transcript);
    let (_, resumed) = Transcript::resume(&path).unwrap();
    let mut expected = vec![message(Role::System, "sys")];
    expected.extend(next);
    assert_eq!(resumed.history, expected);

    // A record that cannot replay is never written: a tool result that itself calls a tool, one
    // for a call the answer before it did not make, a drop of more exchanges than there are.
    let path = dir.0.join("unanswered.jsonl");
    let history = vec![message(Role::User, "u"), calling("1")];
    let mut transcript = Transcript::create(&path, SessionOptions::default(), history).unwrap();
    let malformed = [
        ChatMessage {
            tool_calls: vec![call("2")],
            ..result("1", "r")
        },
        result("2", "r"),
    ];
    let orphan = [result("9", "r"), message(Role::Assistant, "a")];
    let next = [message(Role::User, "u"), message(Role::Assistant, "a")];
    for (dropped, exchange) in [(0, malformed), (0, orphan.clone()), (2, next)] {
        let appended = transcript.append_turn(Terminal::Length, metrics(), dropped, Some(exchange));
        assert!(appended.is_err());
    }
    drop(transcript);
    assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);
    // Nor does a file that holds one replay.
    let record = serde_json::json!({"version": 2, "terminal": "Length", "metrics": metrics(),
        "dropped_exchanges": 0, "exchange": orphan});
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(file, "{record}").unwrap();
    let error = format!("{:#}", Transcript::resume(&path).err().unwrap());
    assert!(error.contains("invalid transcript line 2"), "{error}");

    // A plain line has no tool fields, and a line written before they existed still reads.
    let path = dir.0.join("plain.jsonl");
    let old = r#"{"version":1,"options":{"system":null,"keep_kv":true,"on_overflow":"DropOldestExchanges"},"history":[{"role":"user","content":"u","truncated":false},{"role":"assistant","content":"a","truncated":false}]}"#;
    fs::write(&path, format!("{old}\n")).unwrap();
    let (transcript, resumed) = Transcript::resume(&path).unwrap();
    assert_eq!(
        resumed.history,
        [message(Role::User, "u"), message(Role::Assistant, "a")]
    );
    drop(transcript);
    fs::remove_file(&path).unwrap();
    drop(Transcript::create(&path, SessionOptions::default(), resumed.history).unwrap());
    assert_eq!(fs::read_to_string(&path).unwrap(), format!("{old}\n"));
}

#[test]
fn hide_thinking_withholds_think_blocks_and_their_stops() {
    // FakeModel answers `<think>`, `x`, `</think>`, `yx`, then EOS.
    let tails = [("<think>", 1), ("x", 1), ("</think>", 1), ("yx", 1)];
    let (v, _) = fixture_with(&tails, 512, TEMPLATE);
    let mut s = v.session(SessionOptions::default());
    let shown = GenParams {
        hide_thinking: false,
        ..params(4)
    };
    let stop = |p: GenParams| GenParams {
        stop: vec!["x".into()],
        ..p
    };
    // Text events count every token up to the end or the stop, hidden ones included.
    for (p, text, finish, counted) in [
        (params(4), "yx", Finish::Length, 4),
        (shown.clone(), "<think>x</think>yx", Finish::Length, 4),
        // Hidden, the stop matches only after the block; shown, inside it.
        (stop(params(4)), "y", Finish::StopSeq, 4),
        (stop(shown), "<think>", Finish::StopSeq, 2),
    ] {
        let mut tokens = 0;
        let (delivered, end) = collect(s.send("q", p).unwrap().inspect(|event| {
            if let TextEvent::Text { tokens: n, .. } = event {
                tokens += n;
            }
        }));
        assert_eq!(delivered, text);
        assert_eq!(s.history().last().unwrap().content, text);
        usage(end, finish);
        assert_eq!(tokens, counted, "{text}");
    }
    // Cut inside the block: nothing of it is shown, and the stream says it ended thinking.
    let mut stream = s.send("q", params(2)).unwrap();
    let (delivered, end) = collect(stream.by_ref());
    assert_eq!(delivered, "");
    usage(end, Finish::Length);
    assert!(stream.in_thinking());
    assert_eq!(s.history().last().unwrap().content, "");
}

#[test]
fn tool_exchange_roundtrips_the_tiny_tokenizer_template() {
    // The tiny tokenizer's own template. Its assistant header is one control token, so FakeModel
    // answers with the next tokens: the call exactly as the template renders `call("1")` (one
    // user-defined token), then end of turn.
    let g = Gguf::parse(&fs::read(tiny::TOKENIZER_GGUF).unwrap()).unwrap();
    let template = g
        .get("tokenizer.chat_template")
        .and_then(Value::as_str)
        .unwrap();
    let header = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    let call_text = r#"{"name": "f", "parameters": {"path": "a"}}"#;
    let tails = [(header, 3), (call_text, 4), ("<|eot_id|>", 3)];
    let (v, _) = fixture_with(&tails, 2048, template);
    let p = GenParams {
        tools: vec![serde_json::json!({"type": "function", "function": {
            "name": "f", "description": "Reads a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}})],
        ..params(4)
    };
    let mut s = v.session(SessionOptions::default());
    let (text, end) = collect(s.send("read a", p.clone()).unwrap());
    assert_eq!(text, call_text);
    // The tools reach the template, in a session as in a plain request.
    let prompt = |p: GenParams| {
        let user = Message {
            role: "user".into(),
            content: "read a".into(),
        };
        let end = collect(v.generate(Prompt::Messages(vec![user]), p).unwrap()).1;
        usage(end, Finish::Eos).prompt_tokens
    };
    let with_tools = prompt(p.clone());
    assert_eq!(usage(end, Finish::Eos).prompt_tokens, with_tools);
    assert!(prompt(params(4)) < with_tools);
    // encode_messages gives the ids a request with those messages and tools feeds the engine.
    let user = larch::ToolMessage {
        role: "user".into(),
        content: "read a".into(),
        tool_calls: vec![],
        tool_call_id: String::new(),
    };
    let ids = v.encode_messages(&[user], &p.tools, None).unwrap();
    assert_eq!(ids.len() as u32, with_tools);
    // The caller parses the call; the template renders it back to the tokens the engine fed.
    s.set_tool_calls("", call("1")).unwrap();
    let parked = s.tokens_in_kv();
    let (text, end) = collect(s.send_tool_result("1", "the text of a", p.clone()).unwrap());
    assert_eq!(usage(end, Finish::Eos).cached_tokens, parked);
    // This answer stays plain text, as a final answer does.
    assert_eq!(text, call_text);
    let parked = s.tokens_in_kv();
    let end = collect(s.send("thanks", p).unwrap()).1;
    assert_eq!(usage(end, Finish::Eos).cached_tokens, parked);
    assert_eq!(
        s.history()
            .iter()
            .map(|m| m.role.as_str())
            .collect::<Vec<_>>(),
        [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant"
        ]
    );
}

#[test]
fn hide_thinking_withholds_the_blank_lines_after_a_block() {
    // As the template drops them with the block; blank lines after the first text stay.
    let tails = [
        ("<think>", 1),
        ("x", 1),
        ("</think>", 1),
        ("\n\n", 1),
        ("y", 1),
        ("\n", 1),
    ];
    let (v, _) = fixture_with(&tails, 512, TEMPLATE);
    let mut s = v.session(SessionOptions::default());
    assert_eq!(collect(s.send("q", params(6)).unwrap()).0, "y\n");
    assert_eq!(s.history().last().unwrap().content, "y\n");
}

/// With `stream_thinking` the withheld block comes as it is written, before the answer, and adds
/// up to `Done::thinking`; without it no piece comes.
#[test]
fn stream_thinking_yields_the_withheld_block_as_it_comes() {
    let tails = [
        ("<think>", 1),
        ("x", 1),
        ("é", 1),
        ("</think>", 1),
        ("y", 1),
    ];
    let (v, _) = fixture_with(&tails, 512, TEMPLATE);
    let mut s = v.session(SessionOptions::default());
    let streamed = GenParams {
        stream_thinking: true,
        ..params(5)
    };
    let events: Vec<TextEvent> = s.send("q", streamed).unwrap().collect();
    let pieces: Vec<&str> = (events.iter())
        .filter_map(|e| match e {
            TextEvent::Thinking { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(pieces, ["x", "é"], "{events:?}");
    assert!(matches!(&events[2], TextEvent::Text { delta, .. } if delta == "y"));
    assert!(
        matches!(events.last(), Some(TextEvent::Done { thinking: Some(t), .. }) if t == "xé"),
        "{events:?}"
    );
    assert_eq!(s.history().last().unwrap().content, "y");
    let quiet = s.send("q", params(5)).unwrap();
    assert!(
        quiet
            .into_iter()
            .all(|e| !matches!(e, TextEvent::Thinking { .. }))
    );
}

#[test]
fn tuning_surface_identity_fault_flush_and_tuner_lifecycle() {
    let (v, e) = fixture(512);
    let Read::Record(first) = v.telemetry().reader(ReadFrom::Oldest).next() else {
        panic!("load writes the deployment record");
    };
    assert_eq!(v.engine_id(), first.header().engine_id);
    assert_eq!(v.deployment().ctx_max, e.deployment().ctx_max);

    let (tx, rx) = std::sync::mpsc::channel();
    let sent = tx.clone();
    v.attach_tuner(move || sent.send(None).unwrap());
    let sent = tx.clone();
    v.attach_tuner(move || sent.send(None).unwrap());
    assert_eq!(
        rx.try_recv(),
        Ok(None),
        "re-attaching stops the previous tuner"
    );
    assert!(v.stop_tuner());
    assert_eq!(rx.try_recv(), Ok(None));
    assert!(!v.stop_tuner());

    // The last owner's drop stops the tuner while the engine still answers a trial revert.
    let knobs = v.knobs();
    v.attach_tuner(move || {
        tx.send(Some(knobs.revert(RevertTarget::Trial(1), Authority::Gait)))
            .unwrap()
    });
    let s = v.session(SessionOptions::default());
    drop(v);
    assert!(rx.try_recv().is_err(), "a session still owns the engine");
    drop(s);
    assert_eq!(
        rx.try_recv(),
        Ok(Some(KnobReply::Rejected {
            reason: RejectReason::NothingToRevert
        }))
    );
    assert_ne!(e.health().state, HealthState::Ready);

    let (v, _e) = fixture_model(
        &[("b", 1), ("c", 1), ("d", 1)],
        512,
        TEMPLATE,
        Box::new(FailOnThirdForward(FakeModel::new(261), 0)),
    );
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    v.set_fault_flush(move |ring, _| {
        let mut reader = ring.reader(ReadFrom::Oldest);
        let mut failed = false;
        while let Read::Record(record) = reader.next() {
            failed |=
                matches!(record, Record::HealthEvent(_, h) if h.to == HealthState::Failed.raw());
        }
        tx.send(failed).unwrap();
    })
    .unwrap();
    let end = collect(v.generate(Prompt::Tokens(vec![256]), params(3)).unwrap()).1;
    assert_eq!(end, TextEvent::Error(EngineError::DeviceLost));
    assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok(true));
    assert!(v.set_fault_flush(|_, _| {}).is_err());
}

#[test]
fn tuner_stop_panic_still_shuts_the_engine_down() {
    let (v, e) = fixture(512);
    v.attach_tuner(|| panic!("stop failed"));
    let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(v)));
    assert!(dropped.is_err(), "the stop's panic propagates");
    assert_ne!(
        e.health().state,
        HealthState::Ready,
        "shutdown ran despite the panic"
    );
    assert!(e.submit(larch_engine::Request::new(vec![256], 1)).is_err());
}

#[test]
fn a_panicking_old_stop_leaves_the_new_tuner_attached() {
    let (v, _e) = fixture(512);
    v.attach_tuner(|| panic!("old stop failed"));
    let attached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.attach_tuner(|| {})));
    assert!(attached.is_err(), "the old stop's panic propagates");
    assert!(v.stop_tuner(), "the new tuner is still attached");
}

#[test]
fn overlapping_tuner_transitions_wait_for_the_previous_stop() {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let (v, _e) = fixture(512);
    let (entered, in_stop) = channel();
    let (release, gate) = channel::<()>();
    v.attach_tuner(move || {
        entered.send(()).unwrap();
        gate.recv().unwrap();
    });
    let (done, finished) = channel();
    std::thread::scope(|s| {
        let stopper = done.clone();
        let v = &v;
        s.spawn(move || stopper.send(("stop", v.stop_tuner())).unwrap());
        in_stop.recv().unwrap();
        s.spawn(move || {
            v.attach_tuner(|| {});
            done.send(("attach", true)).unwrap();
        });
        // Released before asserting, so a regression fails instead of hanging the scope.
        let early = finished.recv_timeout(Duration::from_millis(200));
        release.send(()).unwrap();
        assert!(
            early.is_err(),
            "neither call returns while the old tuner is still stopping"
        );
        // Both calls return once the gate opens, in either order.
        let mut done = [finished.recv().unwrap(), finished.recv().unwrap()];
        done.sort();
        assert_eq!(done, [("attach", true), ("stop", true)]);
    });
    assert!(
        v.stop_tuner(),
        "the new tuner is attached after the old one stopped"
    );
}

/// A score request reads the named ids back on the first text event, in the order given, with their
/// exact logprobs over the whole vocabulary, whether or not `logprobs` asks for the top ones too.
#[test]
fn score_ids_read_back_in_order() {
    let (v, _e) = fixture(512);
    // The fake gives the prompt's last token + 1 (here 257) logit 100 and the other 260 ids 0.
    let lse = 100.0 + (260.0 * (-100f64).exp()).ln_1p();
    let want = [(259, -lse), (3, -lse), (257, 100.0 - lse)];
    for logprobs in [0, 1, 20] {
        let p = GenParams {
            score_ids: want.iter().map(|w| w.0).collect(),
            logprobs,
            ..params(1)
        };
        let first = v.generate(Prompt::Tokens(vec![256]), p).unwrap().next();
        let Some(TextEvent::Text {
            logprobs: Some(probs),
            ..
        }) = &first
        else {
            panic!("logprobs {logprobs}: {first:?}");
        };
        let top = &probs[0].top;
        let ids: Vec<u32> = top.iter().map(|t| t.0).collect();
        assert_eq!(ids, [259, 3, 257], "logprobs {logprobs}");
        for (&(id, got), (_, want)) in top.iter().zip(want) {
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "id {id}: {got} vs {want}"
            );
        }
    }
}

/// A score request with a repetition penalty, or with an id past the vocabulary, is refused at submit;
/// one that sets no penalty is not, whatever the penalty knob's default.
#[test]
fn score_requests_are_checked_at_admission() {
    let (v, _e) = fixture(512);
    let score = |ids: Vec<u32>, repeat_penalty| {
        let p = GenParams {
            score_ids: ids,
            repeat_penalty,
            ..params(1)
        };
        v.generate(Prompt::Tokens(vec![256]), p).err()
    };
    assert_eq!(
        score(vec![257], Some(1.1)),
        Some(EngineError::InvalidRequest(invalid::SCORE_PENALTY))
    );
    assert_eq!(
        score(vec![261], None),
        Some(EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB))
    );
    assert_eq!(score(vec![257], None), None);
    let reply = v.knobs().set(KnobRequest::new(
        knob_id::REPEAT_PENALTY_DEFAULT,
        1.1,
        Authority::Human,
    ));
    assert!(matches!(reply, KnobReply::Applied { .. }), "{reply:?}");
    assert_eq!(score(vec![257], None), None);
}

/// `token_ids` is the vocabulary's own encoding of the text without the BOS it adds to a prompt; a
/// special token's text still encodes as that token.
#[test]
fn token_ids_add_no_special_tokens() {
    let g = Gguf::parse(&fs::read(tiny::TOKENIZER_GGUF).unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let prompt = tok.encode(" A", true);
    assert!(prompt.len() > 1 && prompt.first().copied() == tok.bos());
    let eos = tok.eos().unwrap();
    let eos_text = String::from_utf8(tok.decode_piece(eos).to_vec()).unwrap();
    let model: Box<dyn Model> = Box::new(FakeModel::new(512));
    let opts = EngineOptions::new("fake".into(), CPU);
    let e = start_with(opts, Box::new(move |_, _, _| Ok(model))).unwrap();
    let v = Larch::from_engine(e, tok, template);
    assert_eq!(v.token_ids(" A"), prompt[1..]);
    assert!(v.token_ids("").is_empty());
    assert_eq!(v.token_ids(&eos_text), [eos], "{eos_text:?}");
}

const QWEN3: &str = include_str!("../../larch-model/tests/data/qwen3.jinja");

/// The tokens `messages` render to with the template's `enable_thinking` set to `thinking`,
/// encoded as the facade encodes a rendered prompt.
fn rendered(tails: &[(&str, i32)], template: &str, user: &str, thinking: Option<bool>) -> usize {
    let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let g = Gguf::parse(&write(&metadata(tails, template), &tensors)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let message = larch_model::ToolMessage {
        role: "user".into(),
        content: user.into(),
        ..Default::default()
    };
    let text = ChatTemplate::from_gguf(&g, &tok)
        .unwrap()
        .render_tools_thinking(&[message], &[], true, thinking)
        .unwrap();
    tok.encode(&text, true).len()
}

#[test]
fn thinking_false_renders_the_empty_think_block() {
    // FakeModel answers the assistant header with `<think>`, `x`, `</think>`, `\n\n`, `y`, EOS; the
    // empty block `Some(false)` renders ends with `\n\n`, which it answers with `y`, EOS.
    let tails = [
        ("<|im_start|>assistant\n", 4),
        ("<think>", 4),
        ("x", 1),
        ("</think>", 4),
        ("\n\n", 4),
        ("y", 1),
    ];
    let (v, _) = fixture_with(&tails, 2048, QWEN3);
    let off = |p: GenParams| GenParams {
        thinking: Some(false),
        ..p
    };
    // Two tokens: thinking on ends inside the block; off, the answer and EOS.
    let watch = |stream: larch::TextStream| {
        let mut entered = false;
        let mut stream = stream;
        let mut end = None;
        while let Some(event) = stream.next() {
            entered |= stream.in_thinking();
            end = Some(event);
        }
        match end {
            Some(TextEvent::Done { usage, .. }) => (entered, usage.prompt_tokens),
            other => panic!("expected Done, got {other:?}"),
        }
    };
    let hi = || {
        vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }]
    };
    let (on, on_tokens) = watch(v.generate(Prompt::Messages(hi()), params(2)).unwrap());
    let (entered, off_tokens) = watch(v.generate(Prompt::Messages(hi()), off(params(2))).unwrap());
    assert!(on && !entered);
    assert!(off_tokens > on_tokens);
    assert_eq!(
        off_tokens as usize,
        rendered(&tails, QWEN3, "hi", Some(false))
    );
    assert_eq!(on_tokens as usize, rendered(&tails, QWEN3, "hi", None));
    // A session's turn renders the flag too.
    let mut s = v.session(SessionOptions::default());
    let (entered, tokens) = watch(s.send("hi", off(params(2))).unwrap());
    assert!(!entered);
    assert_eq!(tokens, off_tokens);
    assert_eq!(s.history().last().unwrap().content, "y");
}

/// Advances the test clock (`EngineOptions::clock`) by 7 ms per forward.
struct Ticking(FakeModel, std::sync::Arc<AtomicU64>);
impl Model for Ticking {
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
        self.1.fetch_add(7_000_000, Ordering::AcqRel);
        self.0.forward(s, b, kv, logits)
    }
}

/// FakeModel answers the generation prompt (BOS) with `<think>`, `<t0>`..`<t299>`, `</think>`,
/// `\n\n`, `<a0>`..`<a9>`, EOS: a 302-token block, then an answer of 11 tokens. The close the facade
/// feeds ends with `\n\n` too, so the same answer follows it.
fn thinker(context: u32) -> (Larch, EngineHandle) {
    thinker_with(context, &[])
}

/// [`thinker`] with block tokens renamed: `(i, name)` writes `name`, a control token, as the
/// block's `i`th.
fn thinker_with(context: u32, renamed: &[(usize, &str)]) -> (Larch, EngineHandle) {
    let names: Vec<String> = (0..300)
        .map(|i| format!("<t{i}>"))
        .chain((0..10).map(|i| format!("<a{i}>")))
        .collect();
    let mut tails = vec![("<think>", 4)];
    tails.extend(names[..300].iter().enumerate().map(|(i, n)| {
        match renamed.iter().find(|(at, _)| *at == i) {
            Some((_, name)) => (*name, 4),
            None => (n.as_str(), 1),
        }
    }));
    tails.extend([("</think>", 4), ("\n\n", 4)]);
    tails.extend(names[300..].iter().map(|n| (n.as_str(), 1)));
    let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let g = Gguf::parse(&write(&metadata(&tails, TEMPLATE), &tensors)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::from_gguf(&g, &tok).unwrap();
    let clock = std::sync::Arc::new(AtomicU64::new(0));
    let mut opts = EngineOptions::new("fake".into(), CPU);
    opts.max_context = context;
    // Room for a whole turn's step samples, so its request spans are still there to read.
    opts.ring_capacity = 4096;
    opts.clock = Some(clock.clone());
    let model = Ticking(FakeModel::new(tails.len() as u32 + 258), clock);
    let e = start_with(opts, Box::new(move |_, _, _| Ok(Box::new(model)))).unwrap();
    (Larch::from_engine(e.clone(), tok, template), e)
}

fn block(n: usize) -> String {
    (0..n).map(|i| format!("<t{i}>")).collect()
}

fn answer(n: usize) -> String {
    (0..n).map(|i| format!("<a{i}>")).collect()
}

type Reply = (String, Finish, larch::Usage, larch::Timings, Option<String>);

/// The text and the terminal event's parts; the text events count every generated token. This
/// fixture's SPM has no `▁` token, so a space comes back as its bytes; it is turned back here.
fn reply(stream: larch::TextStream) -> Reply {
    let mut counted = 0;
    let (text, end) = collect(stream.inspect(|event| {
        if let TextEvent::Text { tokens, .. } = event {
            counted += tokens;
        }
    }));
    match end {
        TextEvent::Done {
            finish,
            usage,
            timings,
            thinking,
            ..
        } => {
            // A stop's usage also counts what was sampled before the engine saw the stop.
            if finish != Finish::StopSeq {
                assert_eq!(counted, usage.completion_tokens);
            }
            assert_eq!(timings.cached_tokens, usage.cached_tokens);
            let space = |t: String| t.replace('▁', " ");
            (space(text), finish, usage, timings, thinking.map(space))
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

/// The cached tokens of session `id`'s request whose prompt had `prompt` tokens, from the
/// engine's request spans.
fn span_cached(v: &Larch, id: u64, prompt: u32) -> Option<u32> {
    let mut reader = v.telemetry().reader(ReadFrom::Oldest);
    let mut cached = None;
    loop {
        match reader.next() {
            Read::Record(Record::RequestSpan(_, r))
                if r.session_id == id && r.prompt_tokens == prompt =>
            {
                cached = Some(r.cached_tokens)
            }
            Read::Empty => return cached,
            _ => {}
        }
    }
}

fn budget(tokens: u32) -> GenParams {
    GenParams {
        thinking_budget: Some(tokens),
        ..params(0)
    }
}

#[test]
fn thinking_budget_cuts_the_block_and_continues_in_the_same_session() {
    let (v, _e) = thinker(4096);
    // Without a budget: the whole block is hidden and handed over on Done.
    let mut plain = v.session(SessionOptions::default());
    let (text, finish, whole, _, thinking) = reply(plain.send("q", params(0)).unwrap());
    assert_eq!((text.as_str(), finish), (answer(10).as_str(), Finish::Eos));
    assert_eq!(thinking.as_deref(), Some(block(300).as_str()));
    assert_eq!(whole.completion_tokens, 314);
    let prompt = whole.prompt_tokens;

    let close = v.token_ids(larch::text::THINKING_CLOSE);
    assert_eq!(close.last(), v.token_ids("\n\n").last());
    let inner = larch::text::THINKING_CLOSE.trim_end_matches("</think>\n\n");
    let mut s = v.session(SessionOptions::default());
    let id = s.id();
    let (text, finish, usage, timings, thinking) = reply(s.send("q", budget(50)).unwrap());
    // `<think>` and 49 block tokens, the close, then the same answer.
    assert_eq!((text.as_str(), finish), (answer(10).as_str(), Finish::Eos));
    assert_eq!(thinking, Some(block(49) + inner));
    assert!(timings.thinking_cut);
    assert_eq!(usage.completion_tokens, 50 + 11);
    // The turn's own prompt and reuse; the continuation re-sent the prompt, the 50 ids and the
    // close, and matched the parked KV to its last fed token (the 50th sampled token was never fed).
    let resent = prompt + 50 + close.len() as u32;
    assert_eq!((usage.prompt_tokens, usage.cached_tokens), (prompt, 0));
    assert_eq!(timings.continuation_prompt_tokens, resent);
    // One forward per step at 7 ms: the time to first token is the first request's, and every later
    // token took one step, the continuation's prefill included.
    assert!((timings.ttft_ms - 7.0).abs() < 1e-3, "{timings:?}");
    assert!((timings.tpot_ms - 7.0).abs() < 1e-9, "{timings:?}");
    // Not released: the session kept its id and parks the whole conversation.
    assert_eq!(s.id(), id);
    assert_eq!(s.tokens_in_kv(), resent + 11 - 1);
    assert_eq!(s.history().last().unwrap().content, answer(10));
    // The stored answer has no block, so the next turn matches up to the answer's start.
    let next = reply(s.send("q", params(0)).unwrap()).2;
    assert_eq!(next.cached_tokens, prompt);
    // The engine writes a request's span after the step that sent its Done, so it is read once a
    // later turn has run.
    assert_eq!(span_cached(&v, id, resent), Some(prompt + 49));
    // The caller's max_tokens holds for both requests together.
    let capped = GenParams {
        thinking_budget: Some(50),
        ..params(55)
    };
    let (text, finish, usage, timings, _) = reply(s.send("q", capped).unwrap());
    assert_eq!(
        (text.as_str(), finish),
        (answer(5).as_str(), Finish::Length)
    );
    assert_eq!(usage.completion_tokens, 55);
    assert!(timings.thinking_cut);
    // The continuation is the turn's request: a reset cancels it.
    let mut stream = s.send("q", budget(50)).unwrap();
    assert!(matches!(
        stream.next(),
        Some(TextEvent::Text { tokens: 51, .. })
    ));
    s.reset();
    assert_eq!(
        stream.next(),
        Some(TextEvent::Error(EngineError::Cancelled))
    );
}

/// A budget cut inside a call the block began drops the call's ids before the close: carried on,
/// the call ended after the close as the answer's first words, with its opener hidden in the
/// block (a thinking model's step showed `old_text>…</tool_call>` as its reply). A call the block closed
/// before the cut stays.
#[test]
fn a_budget_cut_inside_a_call_drops_the_unfinished_call() {
    let close = |v: &Larch| v.token_ids(larch::text::THINKING_CLOSE).len() as u32;
    // `<think>` is id 0 of the output: the block's 40th token is its 41st.
    let (v, _e) = thinker_with(4096, &[(40, "<tool_call>")]);
    let prompt = reply(
        v.session(SessionOptions::default())
            .send("q", params(0))
            .unwrap(),
    )
    .2
    .prompt_tokens;
    let mut s = v.session(SessionOptions::default());
    let (text, finish, _, timings, _) = reply(s.send("q", budget(50)).unwrap());
    assert_eq!((text.as_str(), finish), (answer(10).as_str(), Finish::Eos));
    assert!(timings.thinking_cut);
    assert_eq!(timings.continuation_prompt_tokens, prompt + 41 + close(&v));

    let (v, _e) = thinker_with(4096, &[(10, "<tool_call>"), (20, "</tool_call>")]);
    let mut s = v.session(SessionOptions::default());
    let (text, _, _, timings, _) = reply(s.send("q", budget(50)).unwrap());
    assert_eq!(text, answer(10));
    assert_eq!(timings.continuation_prompt_tokens, prompt + 50 + close(&v));
}

#[test]
fn thinking_budget_above_the_block_changes_nothing() {
    let (v, _e) = thinker(4096);
    let run = |p: GenParams| reply(v.session(SessionOptions::default()).send("q", p).unwrap());
    let plain = run(params(0));
    for over in [400, 314] {
        let (text, finish, usage, timings, thinking) = run(budget(over));
        assert_eq!(
            (&text, finish, usage, &thinking),
            (&plain.0, plain.1, plain.2, &plain.4)
        );
        assert!(!timings.thinking_cut);
        assert!(
            (timings.tpot_ms - plain.3.tpot_ms).abs() < 1e-9,
            "{timings:?}"
        );
    }
    // A budget that runs out after the block closes cuts nothing: the answer continues as it was.
    let (text, finish, usage, timings, thinking) = run(budget(305));
    assert_eq!((&text, finish, &thinking), (&plain.0, plain.1, &plain.4));
    assert_eq!(usage, plain.2);
    assert!(!timings.thinking_cut);
    assert_eq!(
        timings.continuation_prompt_tokens,
        plain.2.prompt_tokens + 305
    );
    // A caller's cap below the budget is the caller's: nothing continues.
    let (_, finish, usage, timings, _) = run(GenParams {
        thinking_budget: Some(50),
        ..params(40)
    });
    assert_eq!((finish, usage.completion_tokens), (Finish::Length, 40));
    assert!(!timings.thinking_cut);
}

#[test]
fn thinking_budget_stops_never_match_inside_the_block() {
    let (v, _e) = thinker(4096);
    let mut s = v.session(SessionOptions::default());
    let stops = |stop: &[&str]| GenParams {
        stop: stop.iter().map(|s| (*s).into()).collect(),
        ..budget(50)
    };
    let (text, finish, ..) = reply(s.send("q", stops(&["<t3>", "Considering"])).unwrap());
    assert_eq!((text, finish), (answer(10), Finish::Eos));
    // After the block a stop matches as always, in the continuation.
    let (text, finish, _, timings, _) = reply(s.send("q", stops(&["<a5>"])).unwrap());
    assert_eq!((text, finish), (answer(5), Finish::StopSeq));
    assert!(timings.thinking_cut);
    // Shown, the close is delivered with the block, so the text still parses as a closed block.
    let shown = GenParams {
        hide_thinking: false,
        ..budget(50)
    };
    let (text, _, _, timings, thinking) = reply(s.send("q", shown).unwrap());
    let close = larch::text::THINKING_CLOSE;
    assert_eq!(text, format!("<think>{}{close}{}", block(49), answer(10)));
    assert!(timings.thinking_cut && thinking.is_none());
    // Shown, a stop matches in the close as in the block, and the turn ends there.
    let shown = GenParams {
        hide_thinking: false,
        stop: vec!["Considering".into()],
        ..budget(50)
    };
    let (text, finish, usage, timings, _) = reply(s.send("q", shown).unwrap());
    assert_eq!(text, format!("<think>{}\n\n", block(49)));
    assert_eq!((finish, usage.completion_tokens), (Finish::StopSeq, 50));
    assert!(timings.thinking_cut);
    // A zero budget leaves no room for a first token.
    let zero = s.send("q", budget(0)).map(reply);
    assert!(
        matches!(zero, Err(EngineError::InvalidRequest(_))),
        "{zero:?}"
    );
}

#[test]
fn thinking_budget_ends_at_the_cut_when_the_close_cannot_fit() {
    let (v, _e) = thinker(4096);
    let prompt = reply(
        v.session(SessionOptions::default())
            .send("q", params(0))
            .unwrap(),
    )
    .2;
    let close = v.token_ids(larch::text::THINKING_CLOSE).len() as u32;
    // The continuation's prompt would fill the context: the turn ends at the cut, as at a cap.
    let (v, _e) = thinker(prompt.prompt_tokens + 50 + close);
    let mut s = v.session(SessionOptions::default());
    let mut stream = s.send("q", budget(50)).unwrap();
    let (text, end) = collect(stream.by_ref());
    assert_eq!(text, "");
    assert_eq!(usage(end, Finish::Length).completion_tokens, 50);
    assert!(stream.in_thinking());
    // `in_thinking` is about a withheld block; a shown one was delivered.
    let shown = GenParams {
        hide_thinking: false,
        ..budget(50)
    };
    let mut stream = s.send("q", shown).unwrap();
    assert!(collect(stream.by_ref()).0.starts_with("<think><t0>"));
    assert!(!stream.in_thinking());
}

#[test]
fn thinking_budget_reserves_the_close_so_the_answer_keeps_max_tokens() {
    let (v, _e) = thinker(4096);
    let close = v.token_ids(larch::text::THINKING_CLOSE).len() as u32;
    let mut s = v.session(SessionOptions::default());
    reply(s.send("q", params(1)).unwrap());
    // The second turn's prompt: both user lines and the first, empty, answer.
    let second = reply(s.send("q", params(1)).unwrap()).2.prompt_tokens;
    // A context the second turn's prompt and max_tokens fill exactly, with no room for the close:
    // the overflow policy drops the first exchange, so the continuation keeps all of max_tokens.
    let (v, _e) = thinker(second + 60 + close - 1);
    let mut s = v.session(SessionOptions::default());
    reply(s.send("q", params(1)).unwrap());
    let capped = GenParams {
        thinking_budget: Some(50),
        ..params(60)
    };
    let (_, finish, usage, timings, _) = reply(s.send("q", capped).unwrap());
    assert!(timings.thinking_cut);
    assert_eq!((finish, usage.completion_tokens), (Finish::Length, 60));
    assert_eq!(timings.dropped_exchanges, 1);
}

/// `activations` reads the residual stream of the prompt's last token back from the engine after
/// layer 0 and the last, in the order asked, for a token and a text prompt; a layer past the depth
/// is refused at submit.
#[test]
fn activations_read_back_layers_0_and_last() {
    let tails = [("b", 1), ("c", 1), ("d", 1)];
    let model = Box::new(FakeModel::with_layers(261, 4));
    let (v, _e) = fixture_model(&tails, 512, TEMPLATE, model);
    // FakeModel's row after layer l for input token t: [t, l, 2, 3, ..., 15].
    let row = |t: u32, l: u16| -> Vec<f32> {
        [t as f32, f32::from(l)]
            .into_iter()
            .chain((2..16).map(|i| i as f32))
            .collect()
    };
    let got = v.activations(Prompt::Tokens(vec![256, 7, 9]), &[3, 0]);
    assert_eq!(got, Ok(vec![row(9, 3), row(9, 0)]));
    let last = *v.token_ids("hi").last().unwrap();
    let got = v.activations(Prompt::Text("hi".into()), &[0, 3]);
    assert_eq!(got, Ok(vec![row(last, 0), row(last, 3)]));
    let past = v.activations(Prompt::Tokens(vec![256]), &[4]);
    assert_eq!(
        past,
        Err(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER))
    );
}

/// Nanoseconds per call of `f` over `n` calls, and whether its last call succeeded.
fn per_call(n: u32, mut f: impl FnMut() -> bool) -> (f64, bool) {
    let start = Instant::now();
    let mut ok = false;
    for _ in 0..n {
        ok = f();
    }
    (start.elapsed().as_nanos() as f64 / f64::from(n), ok)
}

/// A user question and `k` calls answered in one tool turn, then the answer.
fn tool_turn(turn: usize, k: usize) -> Vec<ChatMessage> {
    let ids: Vec<_> = (0..k).map(|i| format!("{turn}-{i}")).collect();
    let calling = ChatMessage {
        tool_calls: ids.iter().map(|id| call(id)).collect(),
        ..message(Role::Assistant, "")
    };
    let mut turn = vec![message(Role::User, "q"), calling];
    turn.extend(ids.iter().map(|id| result(id, "r")));
    turn.push(message(Role::Assistant, "done"));
    turn
}

/// The host-side cost of history validation and journal writes, for comparing a base build with
/// a head build: `cargo test --release -p larch --no-default-features --test facade --
/// --ignored --nocapture validation_costs`. A refused history prints `refused`.
#[test]
#[ignore = "a timing run, not a check"]
fn validation_costs() {
    let (v, _) = fixture_model(&QWEN3_TAILS, 4096, QWEN3, Box::new(Qwen3Fake::new(260)));
    let mut s = v.session(SessionOptions::default());
    let report = |name: &str, (ns, ok): (f64, bool)| {
        let ns = if ok {
            format!("{ns:.0} ns")
        } else {
            "refused".into()
        };
        println!("bench {name}: {ns}");
    };
    // An ordinary chat: one-call tool turns, validated whole by restore.
    for (turns, n) in [(1, 200_000), (20, 20_000), (200, 2_000)] {
        let history: Vec<_> = (0..turns).flat_map(|t| tool_turn(t, 1)).collect();
        let name = format!(
            "restore {turns} one-call turns ({} messages)",
            history.len()
        );
        report(
            &name,
            per_call(n, || {
                s.restore(SessionOptions::default(), history.clone())
                    .is_ok()
            }),
        );
    }
    // Several results per turn: a long history, and one turn of k results.
    let history: Vec<_> = (0..200).flat_map(|t| tool_turn(t, 3)).collect();
    report(
        "restore 200 three-call turns (1200 messages)",
        per_call(2_000, || {
            s.restore(SessionOptions::default(), history.clone())
                .is_ok()
        }),
    );
    for k in [3, 16, 64, 256] {
        let history = tool_turn(0, k);
        report(
            &format!("restore one turn of {k} results"),
            per_call(2_000, || {
                s.restore(SessionOptions::default(), history.clone())
                    .is_ok()
            }),
        );
    }
    // Journal writes (each synced) and their replay.
    let dir = Temp::new();
    let path = dir.0.join("bench.jsonl");
    let mut transcript = Transcript::create(&path, SessionOptions::default(), vec![]).unwrap();
    let pair = [message(Role::User, "q"), message(Role::Assistant, "done")];
    let metrics = Metrics {
        prompt_tokens: 2,
        completion_tokens: 1,
        cached_tokens: 0,
        ttft_ms: 1.0,
        tpot_ms: 1.0,
        tok_s: 1.0,
    };
    report(
        "append_turn, synced",
        per_call(2_000, || {
            transcript
                .append_turn(Terminal::Eos, Some(metrics.clone()), 0, Some(pair.clone()))
                .is_ok()
        }),
    );
    drop(transcript);
    report(
        "resume 2000 records",
        per_call(20, || Transcript::resume(&path).is_ok()),
    );
}

/// IR-07a: 100 seeded requests at temperature 1.0 on the tiny model, each held to a tool call in
/// the `Family::Qwen3` format (a two-tool name enum, arguments from `json_schema_regex`), all parse with
/// `toolcall::parse`, with arguments whose keys and value types match the schema, also across a
/// thinking-budget continuation. Free strings
/// are left out: under random weights a string closes about once in 200 bytes, past the tiny
/// context. The trigger path runs on the engine and grammar tests: a random model never writes
/// `<tool_call>` itself, so here the pattern spells it.
#[test]
fn grammar_tool_calls_always_parse() {
    use larch::api::{Grammar, grammar, json_schema_regex};
    use larch::toolcall::{Family, parse};
    use serde_json::{Value as Json, json};

    let dir = Temp::new();
    let v = Larch::load(LoadOptions::new(tiny_file(&dir), CPU)).unwrap();
    let weather = json!({"type": "object", "properties": {
        "city": {"enum": ["Paris", "Oslo"]}, "days": {"type": "integer", "minimum": 1},
        "metric": {"type": "boolean"}}, "required": ["city", "days"]});
    let search = json!({"type": "object", "properties": {
        "limit": {"type": ["integer", "null"]},
        "tags": {"type": "array", "items": {"type": "number"}}}, "required": ["tags"]});
    let call = |name: &str, schema: &Json| {
        let args = json_schema_regex(schema).unwrap();
        format!(r#"\{{"name": "{name}", "arguments": {args}\}}"#)
    };
    let pattern = format!(
        "<tool_call>\n(?:{}|{})\n</tool_call>",
        call("get_weather", &weather),
        call("search", &search)
    );
    let g = Grammar {
        trigger: None,
        pattern: pattern.clone(),
    };
    let streams: Vec<_> = (0..100)
        .map(|seed| {
            let p = GenParams {
                max_tokens: 400,
                temperature: Some(1.0),
                seed: Some(seed),
                // Odd seeds stop at the budget and continue: the grammar picks up where it was.
                thinking_budget: (seed % 2 == 1).then_some(9),
                grammar: Some(g.clone()),
                ..GenParams::default()
            };
            v.generate(Prompt::Text("call a tool".into()), p).unwrap()
        })
        .collect();
    let mut names = Vec::new();
    for (seed, stream) in streams.into_iter().enumerate() {
        let (text, end) = collect(stream);
        let TextEvent::Done {
            finish, timings, ..
        } = end
        else {
            panic!("{end:?}");
        };
        assert_eq!(
            (finish, timings.grammar_forced),
            (Finish::Eos, 0),
            "{text:?}"
        );
        assert_eq!(timings.continuation_prompt_tokens > 0, seed % 2 == 1);
        // The whole output is one call; the tiny vocabulary shows its EOS byte as U+FFFD.
        let call_text = text.strip_suffix('\u{fffd}').unwrap_or(&text);
        assert!(grammar::matches(&g, call_text).unwrap(), "{text:?}");
        let parsed = parse(Family::Qwen3, &text, false);
        assert_eq!(parsed.error, None, "{text:?}");
        let c = parsed.call.expect("a call");
        let args = c.args.as_object().unwrap();
        let keys: Vec<&str> = args.keys().map(String::as_str).collect();
        match c.name.as_str() {
            "get_weather" => {
                assert!(matches!(args["city"].as_str(), Some("Paris" | "Oslo")));
                assert!(args["days"].is_i64() || args["days"].is_u64(), "{text:?}");
                assert!(args.get("metric").is_none_or(Json::is_boolean));
                assert!(keys.iter().all(|k| ["city", "days", "metric"].contains(k)));
            }
            "search" => {
                let limit = args.get("limit");
                assert!(limit.is_none_or(|l| l.is_null() || l.is_i64() || l.is_u64()));
                assert!(args["tags"].as_array().unwrap().iter().all(Json::is_number));
                assert!(keys.iter().all(|k| ["limit", "tags"].contains(k)));
            }
            other => panic!("tool {other}"),
        }
        names.push(c.name);
    }
    assert!(names.iter().any(|n| n == "search") && names.iter().any(|n| n == "get_weather"));
}

#[test]
fn grammar_errors_are_typed() {
    let (v, _e) = fixture(512);
    let with = |pattern: &str, trigger: Option<&str>| GenParams {
        grammar: Some(larch::api::Grammar {
            trigger: trigger.map(Into::into),
            pattern: pattern.into(),
        }),
        ..params(4)
    };
    let err = |p| v.generate(Prompt::Text("hi".into()), p).err();
    let code = |c| Some(EngineError::InvalidRequest(c));
    assert_eq!(err(with("", None)), code(invalid::GRAMMAR_PATTERN));
    assert_eq!(err(with("(", None)), code(invalid::GRAMMAR_PATTERN));
    assert_eq!(err(with("a", Some(""))), code(invalid::GRAMMAR_PATTERN));
    assert_eq!(
        err(with("[01]*1[01]{20}", None)),
        code(invalid::GRAMMAR_SIZE)
    );
    // The fixture's model ends on its own; a constrained answer still holds to the pattern.
    let (text, _) = collect(
        v.generate(Prompt::Text("hi".into()), with("[bc]+", None))
            .unwrap(),
    );
    assert!(
        !text.is_empty() && text.bytes().all(|b| b == b'b' || b == b'c'),
        "{text:?}"
    );
}

/// Deltas on both layers of tiny's qwen3moe that reroute it.
fn router_deltas() -> RouterDeltas {
    RouterDeltas {
        layers: vec![0, 1],
        experts: tiny::EXPERTS,
        deltas: vec![vec![4.0, -4.0, 4.0, -4.0]; 2],
        ..Default::default()
    }
}

/// Rank-4 adapters on experts 0 and 2 of tiny's last layer, `b` times a ramp in B, A a ramp.
fn adapters(b: f32) -> RouterDeltas {
    let (d, ef, r) = (tiny::D as usize, tiny::EXPERT_FFN as usize, 4);
    let ramp = |n: usize, s: f32| (0..n).map(move |i| ((i * 7 % 13) as f32 - 6.0) * s);
    let one: Vec<f32> = ramp(d * r, 0.02 * b).chain(ramp(r * ef, 0.05)).collect();
    RouterDeltas {
        layers: vec![1],
        experts: tiny::EXPERTS,
        deltas: vec![[one.clone(), one].concat()],
        rank: r as u32,
        ids: vec![vec![0, 2]],
    }
}

/// A request's (id, logprob) tokens, usage and adaptation version.
fn ends(h: &RequestHandle) -> (Vec<(u32, f32)>, Usage, u64) {
    let mut tokens = Vec::new();
    for event in h.events.iter() {
        match event {
            EngineEvent::Prefilled { .. } => {}
            EngineEvent::Token { id, logprob, .. } => tokens.push((id, logprob)),
            EngineEvent::Done {
                usage,
                adapter_version,
                ..
            } => return (tokens, usage, adapter_version),
            EngineEvent::Error(e) => panic!("{e:?}"),
        }
    }
    panic!("no terminal")
}

#[test]
fn router_delta_versions_never_share_kv_and_none_restores_the_base() {
    let dir = Temp::new();
    let mut o = EngineOptions::new(tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE), CPU);
    o.prefix_cache_blocks = 64;
    let e = start(o).unwrap();
    let prompt: Vec<u32> = (1..=40).collect();
    let run = |session: u64| {
        let r = Request {
            session,
            keep_kv: session != 0,
            // The session reuses only its parked KV, the stateless prompt only the shared cache.
            prefix_cache: session == 0,
            ..Request::new(prompt.clone(), 2)
        };
        ends(&e.submit(r).unwrap())
    };
    // Under one version, a session's parked KV and the shared prefix both serve.
    run(7);
    assert!(run(7).1.cached_tokens > 0);
    let (base, ..) = run(0);
    let (warm, u, _) = run(0);
    assert!(u.cached_tokens > 0);
    e.set_router_deltas(Some(router_deltas()), 1).unwrap();
    assert_eq!(e.adapter_version(), 1);
    // Under another, neither does: the resumed session and the stateless prompt re-prefill.
    let (_, u, v) = run(7);
    assert_eq!((u.cached_tokens, v), (0, 1));
    let (adapted, u, v) = run(0);
    assert_eq!((u.cached_tokens, v), (0, 1));
    assert_ne!(adapted, base);
    e.set_router_deltas(None, 0).unwrap();
    // Version 0's cached prefix serves again, as before the swap, bit for bit.
    let (restored, u, v) = run(0);
    assert_eq!((restored, u.cached_tokens > 0, v), (warm, true, 0));
    // A version without deltas, or deltas without one, is refused.
    let bad = Err(EngineError::InvalidRequest(invalid::ROUTER_DELTAS));
    assert_eq!(e.set_router_deltas(None, 2), bad);
    assert_eq!(e.set_router_deltas(Some(router_deltas()), 0), bad);
    // A job runs between steps on the resident transformer.
    assert!(e.with_model(|m, _| m.is_some()).unwrap());
    e.shutdown();
}

#[test]
fn a_swap_during_generation_waits_for_that_answer() {
    let dir = Temp::new();
    let e = start(EngineOptions::new(
        tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE),
        CPU,
    ))
    .unwrap();
    let prompt: Vec<u32> = (1..=12).collect();
    let request = || Request::new(prompt.clone(), 24);
    let (base, ..) = ends(&e.submit(request()).unwrap());
    let h = e.submit(request()).unwrap();
    let Some(EngineEvent::Token { id, logprob, .. }) = h
        .events
        .iter()
        .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
    else {
        panic!("a first token")
    };
    let e2 = e.clone();
    let swap = std::thread::spawn(move || e2.set_router_deltas(Some(router_deltas()), 1));
    let (mut rest, _, v) = ends(&h);
    rest.insert(0, (id, logprob));
    assert_eq!(
        (rest, v),
        (base.clone(), 0),
        "the running answer stays on the base"
    );
    swap.join().unwrap().unwrap();
    let (next, _, v) = ends(&e.submit(request()).unwrap());
    assert_eq!(v, 1);
    assert_ne!(next, base);
    e.shutdown();
}

#[test]
fn token_positions_are_token_loss_and_one_text_scores_the_same_every_time() {
    let dir = Temp::new();
    let path = tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE);
    let v = Larch::load(LoadOptions::new(path, CPU)).unwrap();
    // Longer than one observed chunk, so it is scored in several.
    let tokens: Vec<u32> = (0..300u32).map(|i| 10 + (i * 7) % 40).collect();
    let mask: Vec<bool> = (0..tokens.len()).map(|i| i > 0).collect();
    let loss = v.token_loss(tokens.clone(), mask.clone()).unwrap();
    let positions = v.token_positions(tokens.clone(), mask.clone()).unwrap();
    assert_eq!(positions.len(), tokens.len() - 1);
    assert!(
        positions
            .iter()
            .zip(0..)
            .all(|(p, i)| p.pos == i && !p.top.is_empty())
    );
    let mean = -positions
        .iter()
        .map(|p| f64::from(p.next.unwrap()))
        .sum::<f64>()
        / positions.len() as f64;
    assert_eq!(loss, mean);
    // A reply in between moves what the logits budget sees; the score does not.
    let p = GenParams {
        max_tokens: 4,
        logprobs: 1,
        ..GenParams::default()
    };
    output(
        v.generate(Prompt::Tokens(tokens[..20].to_vec()), p)
            .unwrap(),
    );
    assert_eq!(v.token_loss(tokens, mask).unwrap(), loss);
}

#[test]
fn token_loss_hidden_at_and_the_moe_shape() {
    let dir = Temp::new();
    let path = tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE);
    let v = Larch::load(LoadOptions::new(path.clone(), CPU)).unwrap();
    assert_eq!(v.moe_shape(), Some((tiny::LAYERS as u16, tiny::EXPERTS)));
    assert_eq!(v.model_path(), path);
    assert_eq!(v.adapter_version(), 0);
    let tokens: Vec<u32> = (10..30).collect();
    let mask: Vec<bool> = (0..tokens.len()).map(|i| i >= 12).collect();
    // The host's cross-entropy from each prefix's exact logprob of the next token.
    let mut ce = 0.0;
    for i in 12..tokens.len() {
        let p = GenParams {
            max_tokens: 1,
            temperature: Some(0.0),
            score_ids: vec![tokens[i]],
            ..GenParams::default()
        };
        let stream = v.generate(Prompt::Tokens(tokens[..i].to_vec()), p).unwrap();
        let (_, lp, _) = output(stream);
        ce -= f64::from(lp[0].top[0].1);
    }
    ce /= (tokens.len() - 12) as f64;
    let loss = v.token_loss(tokens.clone(), mask.clone()).unwrap();
    assert!(
        (loss - ce).abs() <= 1e-3 * ce.abs().max(1.0),
        "{loss} vs {ce}"
    );
    let bad = Err(EngineError::InvalidRequest(invalid::LOSS_MASK));
    assert_eq!(v.token_loss(tokens.clone(), vec![true; tokens.len()]), bad);
    assert_eq!(v.token_loss(tokens.clone(), vec![false; tokens.len()]), bad);
    assert_eq!(v.token_loss(tokens.clone(), mask[1..].to_vec()), bad);
    // The residual entering layer 1 is layer 0's output, as `activations` taps it.
    let rows = v.hidden_at(tokens.clone(), 1, vec![3, 7]).unwrap();
    for (row, r) in rows.iter().zip([3, 7]) {
        let tapped = v
            .activations(Prompt::Tokens(tokens[..=r].to_vec()), &[0])
            .unwrap();
        // One prefill of all rows vs one per row: the same residual up to f16 batch-shape noise.
        let worst = row
            .iter()
            .zip(&tapped[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 5e-3, "row {r} differs by {worst}");
    }
    let swapped = v.hidden_at(tokens.clone(), 1, vec![7, 3, 7]).unwrap();
    assert_eq!(swapped, [rows[1].clone(), rows[0].clone(), rows[1].clone()]);
    assert_eq!(
        v.hidden_at(tokens.clone(), 0, vec![3]),
        Err(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER))
    );
    assert_eq!(
        v.encode_chat(Prompt::Tokens(tokens.clone()), None).unwrap(),
        tokens
    );
}

#[test]
fn expert_adapters_are_exact_at_zero_and_version_their_cache() {
    let dir = Temp::new();
    let mut o = EngineOptions::new(tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE), CPU);
    o.prefix_cache_blocks = 64;
    let e = start(o).unwrap();
    let prompt: Vec<u32> = (1..=40).collect();
    let run = || ends(&e.submit(Request::new(prompt.clone(), 2)).unwrap());
    // The uncached run is the one to compare: a cached prefix changes the batch shape.
    let (base, ..) = run();
    let (warm, u, _) = run();
    assert!(u.cached_tokens > 0);
    // B = 0 is the base model bit for bit, but a new version: nothing cached under 0 serves it.
    e.set_router_deltas(Some(adapters(0.0)), 1).unwrap();
    let (zero, u, v) = run();
    assert_eq!((zero, u.cached_tokens, v), (base.clone(), 0, 1));
    e.set_router_deltas(Some(adapters(1.0)), 2).unwrap();
    let (adapted, u, v) = run();
    assert_eq!((u.cached_tokens, v), (0, 2));
    assert_ne!(adapted, base);
    e.set_router_deltas(None, 0).unwrap();
    // Version 0's cached prefix serves again, as before the swaps.
    let (restored, u, _) = run();
    assert_eq!((restored, u.cached_tokens > 0), (warm, true));
    e.shutdown();
}

#[test]
fn token_loss_under_adapters_is_the_trainers_cross_entropy() {
    use larch_model::learn::{RouterCandidate, Suffix, Window};
    let dir = Temp::new();
    let path = tiny_arch_file(&dir, Arch::Qwen3Moe, TEMPLATE);
    let v = Larch::load(LoadOptions::new(path.clone(), CPU)).unwrap();
    let tokens: Vec<u32> = (10..30).collect();
    let mask: Vec<bool> = (0..tokens.len()).map(|i| i >= 12).collect();
    let rows: Vec<u32> = (0..tokens.len() as u32).collect();
    // Adapters on the last layer only, so the residual entering it is the base model's.
    let res: Vec<f32> = v.hidden_at(tokens.clone(), 1, rows).unwrap().concat();
    let d = adapters(1.0);
    let c = RouterCandidate {
        layers: d.layers.clone(),
        experts: d.experts,
        deltas: d.deltas.clone(),
        rank: d.rank,
        ids: d.ids.clone(),
        ..Default::default()
    };
    let pos: Vec<i32> = (0..tokens.len() as i32).collect();
    let w = Window {
        residual: &res,
        tokens: &tokens,
        positions: &pos,
        mask: &mask,
        weight: 1.0,
        row: larch_model::learn::Row::User,
    };
    let suffix = Suffix::load(&path, 1).unwrap();
    let (base, adapted) = (
        suffix.loss(&w, None).unwrap(),
        suffix.loss(&w, Some(&c)).unwrap(),
    );
    assert!((base - v.token_loss(tokens.clone(), mask.clone()).unwrap()).abs() < 2e-3);
    v.set_router_deltas(Some(d), 1).unwrap();
    let got = v.token_loss(tokens, mask).unwrap();
    assert!((adapted - base).abs() > 1e-3, "the adapters must matter");
    assert!((got - adapted).abs() < 2e-3, "{got} vs {adapted}");
}

/// The real tiny model with a generated projector as wide as it, on the CPU, with the shared prefix
/// cache the desktop app turns on; `mmproj: false` loads the model alone.
fn vision_fixture(dir: &Temp, mmproj: bool) -> Larch {
    let model = tiny_file(dir);
    let g = Gguf::parse(&fs::read(&model).unwrap()).unwrap();
    let d = match g.kv.get("llama.embedding_length") {
        Some(Value::U32(d)) => *d,
        v => panic!("embedding length: {v:?}"),
    };
    let mut opts = LoadOptions::new(model, CPU);
    if mmproj {
        let path = dir.0.join("mmproj.gguf");
        fs::write(&path, tiny::Projector::new(d, 5).gguf()).unwrap();
        opts.mmproj = Some(path);
    }
    opts.max_context = 512;
    opts.prefix_cache_blocks = 64;
    Larch::load(opts).unwrap()
}

fn picture(name: &str) -> std::sync::Arc<larch_model::vision::Image> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../larch-model/tests/data/vision")
        .join(name);
    std::sync::Arc::new(larch_model::vision::Image::new(fs::read(p).unwrap().into()).unwrap())
}

const PICTURE_SYSTEM: &str = "You answer questions about pictures, briefly and plainly, always.";

/// One chat's first turn: the system text, then `image` and the same question. The chat stays open
/// (its session is returned): a released session withdraws what it put in the prefix cache.
fn picture_turn(v: &Larch, image: &str) -> (larch::Session, larch::Usage) {
    let mut s = v.session(SessionOptions {
        system: Some(PICTURE_SYSTEM.into()),
        ..SessionOptions::default()
    });
    let stream = s.send_images("what is this?", &[picture(image)], params(1));
    let u = usage(collect(stream.unwrap()).1, Finish::Length);
    (s, u)
}

/// The regression the design names (D8): two chats with identical text and different images never
/// share KV past the first image row, while the same image does share its rows.
#[test]
fn two_chats_with_the_same_text_and_different_images_share_no_kv() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    // The system text, then `<img>`, then the image's rows.
    let image_at = v.token_ids(&format!("{PICTURE_SYSTEM}<img>")).len() as u32;
    let (_first, u) = picture_turn(&v, "palette.png");
    assert_eq!(u.cached_tokens, 0);
    let stats = v.prefix_cache_stats();
    let (_other, other) = picture_turn(&v, "rgba.png");
    assert!(
        other.cached_tokens > 0 && other.cached_tokens <= image_at,
        "another image reused {} tokens (its rows start at {image_at}; cache after the first {stats:?})",
        other.cached_tokens
    );
    let (_same, same) = picture_turn(&v, "palette.png");
    assert!(
        same.cached_tokens > image_at,
        "the same image reused {} tokens (its rows start at {image_at})",
        same.cached_tokens
    );
}

/// Encode on paste (S5): an image prepared as it is attached is encoded while the engine idles, so
/// the turn that sends it encodes nothing; an image nobody prepared is encoded by its turn.
#[test]
fn an_image_prepared_as_it_is_attached_is_not_encoded_again() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let before = v.image_tiles_encoded();
    // The fixtures are one tile each (4 x 2 pixels: the overview alone).
    v.prepare_image(&picture("palette.png")).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while v.image_tiles_encoded() == before {
        assert!(
            std::time::Instant::now() < deadline,
            "the prepared image was never encoded"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let prepared = v.image_tiles_encoded();
    assert_eq!(prepared, before + 1);
    let (s, _) = picture_turn(&v, "palette.png");
    assert_eq!(
        v.image_tiles_encoded(),
        prepared,
        "the turn encoded a prepared tile again"
    );
    let (t, _) = picture_turn(&v, "rgba.png");
    assert_eq!(v.image_tiles_encoded(), prepared + 1);
    // Another `Temp` while this one is held would wait on `cpu_alone`'s guard: the same folder.
    drop((s, t, v));
    let text = vision_fixture(&dir, false);
    assert!(text.prepare_image(&picture("palette.png")).is_err());
}

/// `restore` (how the agent prunes a chat) keeps the images its history still names: the next turn
/// renders the same prompt as a session that never restored.
#[test]
fn restore_keeps_the_images_its_history_names() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let mut plain = v.session(SessionOptions::default());
    let mut restored = v.session(SessionOptions::default());
    for s in [&mut plain, &mut restored] {
        let stream = s.send_images("what is this?", &[picture("palette.png")], params(1));
        collect(stream.unwrap());
    }
    let history = restored.history();
    restored
        .restore(SessionOptions::default(), history)
        .unwrap();
    let next = |s: &mut larch::Session| {
        usage(
            collect(s.send("and now?", params(1)).unwrap()).1,
            Finish::Length,
        )
        .prompt_tokens
    };
    assert_eq!(next(&mut restored), next(&mut plain));
}

/// A chat with images on a model without a projector (design D10): each image is its note, or
/// `[image]`, and the turn runs.
#[test]
fn a_text_only_model_sees_images_as_their_notes() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, false);
    let first = ChatMessage {
        role: Role::User,
        content: "what is this?".into(),
        images: vec![picture("palette.png").hex()],
        ..ChatMessage::default()
    };
    let answer = ChatMessage {
        role: Role::Assistant,
        content: "a square".into(),
        ..ChatMessage::default()
    };
    let history = vec![first, answer];
    let turn = |note: Option<&str>| {
        let mut s = v.session(SessionOptions::default());
        s.restore(SessionOptions::default(), history.clone())
            .unwrap();
        if let Some(n) = note {
            s.set_image_note(&history[0].images[0], n);
        }
        usage(
            collect(s.send("and now?", params(1)).unwrap()).1,
            Finish::Length,
        )
        .prompt_tokens
    };
    let (bare, noted) = (turn(None), turn(Some("a red square")));
    let len = |s: &str| v.token_ids(s).len() as u32;
    let noted_text = "[An earlier image, as the model described it: a red square]";
    assert_eq!(noted - bare, len(noted_text) - len("[image]"));
}

/// A reopened chat shows an image that had become its note as the note still (a note is stored only
/// once a prompt needed it), and a question about the picture brings its rows back.
#[test]
fn a_reopened_chat_keeps_its_notes_and_a_question_brings_the_picture_back() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let (old, new) = (picture("palette.png"), picture("rgba.png"));
    let msg = |role, content: &str, images: Vec<String>| ChatMessage {
        role,
        content: content.into(),
        images,
        ..ChatMessage::default()
    };
    let history = vec![
        msg(Role::User, "what is this?", vec![old.hex()]),
        msg(Role::Assistant, "a square", Vec::new()),
        msg(Role::User, "and this?", vec![new.hex()]),
        msg(Role::Assistant, "a circle", Vec::new()),
    ];
    let chat = |name: &str, note: bool| {
        let path = dir.0.join(name);
        larch::chat::store_image(&path, &old).unwrap();
        larch::chat::store_image(&path, &new).unwrap();
        if note {
            larch::chat::store_note(&path, &old.hex(), "a red square").unwrap();
        }
        path
    };
    let (noted, plain) = (chat("noted.jsonl", true), chat("plain.jsonl", false));
    let turn = |path: &Path, question: &str| {
        let mut s = v.session(SessionOptions::default());
        s.restore(SessionOptions::default(), history.clone())
            .unwrap();
        assert!(larch::chat::resume_images(&mut s, path, &history).is_empty());
        usage(
            collect(s.send(question, params(1)).unwrap()).1,
            Finish::Length,
        )
        .prompt_tokens
    };
    assert_ne!(
        turn(&noted, "and now?"),
        turn(&plain, "and now?"),
        "the older picture shows as its note"
    );
    assert_eq!(
        turn(&noted, "what is in the picture?"),
        turn(&plain, "what is in the picture?"),
        "a question about the picture shows its rows again"
    );
}

/// A ninth image is refused, and a send the session refuses (busy) keeps none of its images.
#[test]
fn a_refused_image_send_keeps_nothing() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let mut s = v.session(SessionOptions::default());
    let nine: Vec<_> = (0..9).map(|_| picture("palette.png")).collect();
    assert_eq!(
        s.send_images("x", &nine, params(1)).err(),
        Some(EngineError::InvalidRequest(invalid::IMAGE))
    );
    assert!(s.history().is_empty());
    let stream = s.send("hello", params(4)).unwrap();
    assert_eq!(
        s.send_images("look", &[picture("rgba.png")], params(1))
            .err(),
        Some(EngineError::InvalidRequest(invalid::SESSION_BUSY))
    );
    collect(stream);
    assert!(s.image(&picture("rgba.png").hex()).is_none());
}

/// An overflow that drops an image's exchange lets the image go too: the session keeps only the
/// images its history names, not every image it was ever sent (each holds its file's bytes).
#[test]
fn an_overflow_drops_the_images_of_the_exchanges_it_drops() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let mut s = v.session(SessionOptions::default());
    let first = picture("palette.png").hex();
    collect(
        s.send_images("what is this?", &[picture("palette.png")], params(1))
            .unwrap(),
    );
    assert!(s.image(&first).is_some());
    let mut dropped = 0;
    for _ in 0..20 {
        let end = collect(s.send(&"seven ".repeat(60), params(1)).unwrap()).1;
        if let TextEvent::Done { timings, .. } = end {
            dropped += timings.dropped_exchanges;
        }
        if !s.history().iter().any(|m| m.images.contains(&first)) {
            break;
        }
        assert!(s.image(&first).is_some(), "kept while its message is");
    }
    assert!(dropped > 0, "no exchange dropped");
    assert!(
        s.image(&first).is_none(),
        "the dropped exchange's image stayed"
    );
}

/// A stored image damaged on disk is a notice on resume, and its message shows `[image]` in its
/// place, as a model without vision would see it; sent again, the file is rewritten.
#[test]
fn a_damaged_image_file_shows_as_a_placeholder() {
    let dir = Temp::new();
    let v = vision_fixture(&dir, true);
    let path = dir.0.join("d.jsonl");
    let img = picture("palette.png");
    larch::chat::store_image(&path, &img).unwrap();
    let file = larch::chat::images_dir(&path).join(img.hex());
    let mut damaged = img.bytes().to_vec();
    let last = damaged.len() - 1;
    damaged[last] ^= 0xFF;
    fs::write(&file, &damaged).unwrap();
    let message = |images: Vec<String>, content: &str| ChatMessage {
        role: Role::User,
        content: content.into(),
        images,
        ..ChatMessage::default()
    };
    let answer = ChatMessage {
        role: Role::Assistant,
        content: "a square".into(),
        ..ChatMessage::default()
    };
    let history = vec![message(vec![img.hex()], "what is this?"), answer.clone()];
    let mut s = v.session(SessionOptions::default());
    s.restore(SessionOptions::default(), history.clone())
        .unwrap();
    let notices = larch::chat::resume_images(&mut s, &path, &history);
    assert_eq!(notices.len(), 1, "{notices:?}");
    let degraded = usage(
        collect(s.send("and now?", params(1)).unwrap()).1,
        Finish::Length,
    );
    // The same chat with the placeholder written as its text.
    let mut plain = v.session(SessionOptions::default());
    let typed = vec![message(Vec::new(), "[image]\nwhat is this?"), answer];
    plain.restore(SessionOptions::default(), typed).unwrap();
    let bare = usage(
        collect(plain.send("and now?", params(1)).unwrap()).1,
        Finish::Length,
    );
    assert_eq!(degraded.prompt_tokens, bare.prompt_tokens);
    larch::chat::store_image(&path, &img).unwrap();
    assert_eq!(fs::read(&file).unwrap(), **img.bytes());
}

/// A load straight into the documents plan (a restart that reopens a pinned chat) holds the
/// context and seats the pin's reload would set, so pinning the document reloads nothing.
#[test]
fn a_load_into_the_documents_plan_pins_without_a_reload() {
    let dir = Temp::new();
    let mut o = LoadOptions::new(pin_model(&dir, Arch::Llama), CPU);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    o.documents = Some(larch::api::DocumentsPlan {
        max_context: 32_768,
        seats: 2,
    });
    let v = Larch::load(o).unwrap();
    let knob = |name| v.knobs().effective().values[larch::knob::spec(name).unwrap().id as usize];
    assert_eq!(v.deployment().ctx_max, 32_768);
    assert_eq!(knob("max_batch_seqs"), 2.0);
    let doc = document(800);
    assert!(!v.pin_reloads(800));
    let (pin, read) = v.pin(doc.clone(), 0).unwrap();
    for event in read {
        if let TextEvent::Error(e) = event {
            panic!("{e:?}");
        }
    }
    assert_eq!(v.deployment().ctx_max, 32_768);
    assert_eq!(pin.tokens(), &doc[..]);
}

/// A load into the documents plan keeps the plan the options named to go back to, as the plan's
/// reload does: with no pin held, a request waiting for a seat the plan lacks brings it back.
#[test]
fn a_load_into_the_documents_plan_goes_back_to_the_options_plan_with_no_pin() {
    let dir = Temp::new();
    let mut o = LoadOptions::new(pin_model(&dir, Arch::Llama), CPU);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    o.documents = Some(larch::api::DocumentsPlan {
        max_context: 32_768,
        seats: 2,
    });
    let v = Larch::load(o).unwrap();
    assert_eq!(v.deployment().ctx_max, 32_768);
    let streams: Vec<_> = (0..3)
        .map(|i| {
            v.generate(Prompt::Tokens(vec![1, 2, 3 + i]), params(16))
                .unwrap()
        })
        .collect();
    for stream in streams {
        let (_, done) = collect(stream);
        assert!(matches!(done, TextEvent::Done { .. }), "{done:?}");
    }
    assert_eq!(v.deployment().ctx_max, 512);
}

/// A pinned document: the documents plan says what fits, the pin reloads the engine for the
/// document and its answer's room once, reads the document in with progress, and a question that
/// starts with it reuses all of it.
#[test]
fn a_pinned_document_reads_in_with_progress_and_its_question_reuses_it() {
    let dir = Temp::new();
    let mut o = LoadOptions::new(
        tiny_arch_file_ctx(&dir, Arch::Llama, TEMPLATE, Some(32_768)),
        CPU,
    );
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    let v = Larch::load(o).unwrap();
    let doc: Vec<u32> = (0..800).map(|i| i % 90 + 1).collect();
    let fit = v.document_fit(doc.len() as u32);
    assert!(fit.fits, "{fit:?}");
    // The context less a turn's room, a step at a time.
    assert_eq!(fit.max_tokens, 32_768 - larch::api::TURN_ROOM);
    assert!(!v.document_fit(fit.max_tokens + 1).fits);
    assert_eq!(v.deployment().ctx_max, 512);
    let (pin, read) = v.pin(doc.clone(), 0).unwrap();
    // One reload for the document and a turn: 17,184 tokens, to the next 16,384 step.
    assert_eq!(v.deployment().ctx_max, 32_768);
    assert_eq!(pin.tokens(), &doc[..]);
    let mut progress = Vec::new();
    for event in read {
        match event {
            TextEvent::Prefilled { done, total } => progress.push((done, total)),
            TextEvent::Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    let (done, total) = *progress.last().expect("progress");
    assert_eq!((done, total), (800, 800));
    assert!(progress.windows(2).all(|w| w[0].0 < w[1].0));
    let mut ask = doc.clone();
    ask.extend([7; 6]);
    let question = v.generate(Prompt::Tokens(ask), params(2)).unwrap();
    // A stream that did not opt in sees no progress.
    let events: Vec<_> = question.collect();
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, TextEvent::Prefilled { .. }))
    );
    let usage = usage(events.into_iter().last().unwrap(), Finish::Length);
    assert_eq!(usage.cached_tokens, 800);
    // A document past the plan is refused, and a pin that dropped no longer holds.
    let long = vec![1; fit.max_tokens as usize + 1];
    assert!(matches!(
        v.pin(long, 0),
        Err(EngineError::InvalidRequest(invalid::DOC_TOO_LONG))
    ));
    drop(pin);
}

/// A live pin's turn room is what the context and the pool leave beside the document, not under a
/// turn's room; the plan's two seats leave the pool at least the context.
#[test]
fn a_live_pins_turn_room_is_what_the_plan_leaves_beside_it() {
    let dir = Temp::new();
    let v = pin_fixture(&pin_model(&dir, Arch::Llama));
    let doc = document(800);
    let (pin, read) = v.pin(doc.clone(), 0).unwrap();
    read_in(read);
    assert!(v.pin_live(&pin));
    let d = v.deployment();
    let room = v.turn_room(&pin);
    assert!(room >= larch::api::TURN_ROOM, "{room}");
    assert!(room <= d.ctx_max - doc.len() as u32, "{room} of {d:?}");
}

/// Without the shared prefix cache nothing could keep a document, so a pin says so, before it
/// reloads anything.
#[test]
fn a_pin_without_a_prefix_cache_is_refused() {
    let dir = Temp::new();
    let mut o = LoadOptions::new(
        tiny_arch_file_ctx(&dir, Arch::Llama, TEMPLATE, Some(32_768)),
        CPU,
    );
    o.max_context = 512;
    let v = Larch::load(o).unwrap();
    assert!(matches!(
        v.pin(vec![1; 100], 0),
        Err(EngineError::InvalidRequest(invalid::NO_PREFIX_CACHE))
    ));
    assert_eq!(v.deployment().ctx_max, 512);
}

/// An engine that cannot plan (a model not from a file) holds no document.
#[test]
fn a_fixture_engine_holds_no_document() {
    let (v, _e) = fixture(512);
    let fit = v.document_fit(10);
    assert_eq!((fit.fits, fit.max_tokens), (false, 0));
}

/// The model file, trained for 32,768 tokens; written once, since its identity includes its write time.
fn pin_model(dir: &Temp, arch: Arch) -> PathBuf {
    tiny_arch_file_ctx(dir, arch, TEMPLATE, Some(32_768))
}

/// A model trained for 32,768 tokens, loaded with the shared prefix cache, as a pin needs.
fn pin_fixture(model: &Path) -> Larch {
    let mut o = LoadOptions::new(model.to_path_buf(), CPU);
    // A mark every 512 tokens, as 8,192 would take minutes of debug-build CPU to reach.
    o.pin_mark = Some(512);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    Larch::load(o).unwrap()
}

/// A document of `n` tokens.
fn document(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 7 + i / 5) % 90 + 1).collect()
}

/// The question's first token with its logprobs, over `doc` then a tail, and the tokens it reused.
fn ask(v: &Larch, doc: &[u32]) -> (Vec<TokenLogprob>, u32) {
    let mut prompt = doc.to_vec();
    prompt.extend([3, 14, 15, 9, 2, 6]);
    let p = GenParams {
        max_tokens: 1,
        temperature: Some(0.0),
        logprobs: 5,
        ..GenParams::default()
    };
    let mut logprobs = Vec::new();
    let mut cached = 0;
    for event in v.generate(Prompt::Tokens(prompt), p).unwrap() {
        match event {
            TextEvent::Text {
                logprobs: Some(l), ..
            } => logprobs.extend(l),
            TextEvent::Done { usage, .. } => cached = usage.cached_tokens,
            TextEvent::Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    (logprobs, cached)
}

/// Reads `pin`'s document in, and the progress the stream reported.
fn read_in(stream: larch::TextStream) {
    for event in stream {
        if let TextEvent::Error(e) = event {
            panic!("{e:?}");
        }
    }
}

/// A pin's two files as a caller keeps them: the pin file appended to, the sidecar replaced.
#[derive(Default, Clone)]
struct Files {
    pin: Vec<u8>,
    state: Vec<u8>,
    end: u32,
}

/// Brings `files` up to `pin`'s last checkpoint, as a caller does: append the segments, then
/// write the state, and again if it is ahead of the file.
fn export_all(v: &Larch, pin: &larch::Pin, files: &mut Files) {
    loop {
        files.end = v.export_pin(pin, &mut files.pin, files.end).unwrap();
        files.state.clear();
        let c = v.export_pin_state(pin, &mut files.state).unwrap();
        if c <= files.end {
            return;
        }
    }
}

/// Imports `files` of `doc` into `v`.
fn import(v: &Larch, doc: &[u32], files: &Files) -> Result<larch::Pin, PinError> {
    v.import_pin(doc.to_vec(), 0, &mut &files.pin[..], &mut &files.state[..])
}

/// A `.pin` of `doc` read on a fresh engine of `model`; the engine and the pin stay.
fn pin_file(model: &Path, doc: &[u32]) -> (Files, Larch, larch::Pin) {
    let v = pin_fixture(model);
    let (pin, read) = v.pin(doc.to_vec(), 0).unwrap();
    read_in(read);
    let mut files = Files::default();
    export_all(&v, &pin, &mut files);
    (files, v, pin)
}

/// A pin exported and imported on another engine of the same file gives the next token's logits
/// bit for bit as the source engine's, with the whole document reused; the tail past the file's
/// whole blocks is read by `pin` after the import, and the imported pin exports again to the same
/// bytes.
#[test]
fn an_exported_pin_imports_to_identical_logits() {
    for arch in [Arch::Llama, Arch::Qwen35] {
        let dir = Temp::new();
        let model = pin_model(&dir, arch);
        let doc = document(805);
        let (files, source, _held) = pin_file(&model, &doc);
        assert_eq!(files.end, 800, "{arch:?}");
        let (want, cached) = ask(&source, &doc);
        assert_eq!(cached, 800, "{arch:?}");
        let v = pin_fixture(&model);
        let pin = import(&v, &doc, &files).unwrap();
        assert_eq!(pin.tokens().len(), 800, "{arch:?}: whole blocks");
        assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 50);
        let (got, cached) = ask(&v, &doc);
        assert_eq!((got, cached), (want.clone(), 800), "{arch:?}");
        // The rest of the document read in after the import computes only it.
        let (all, read) = v.pin(doc.clone(), 0).unwrap();
        read_in(read);
        let (got, cached) = ask(&v, &doc);
        assert_eq!((got, cached), (want, 800), "{arch:?}");
        let mut again = Files::default();
        export_all(&v, &pin, &mut again);
        assert_eq!(
            (again.pin, again.state),
            (files.pin, files.state),
            "{arch:?}"
        );
        drop(all);
    }
}

/// A file appended to while the read goes (whatever the first export took, then the rest) is the
/// file one export of the whole makes. The segment is 512 tokens here, 8,192 in use.
#[test]
fn an_appended_pin_file_equals_one_export() {
    for arch in [Arch::Qwen35, Arch::Llama] {
        let dir = Temp::new();
        let model = pin_model(&dir, arch);
        let doc = document(2_600);
        let v = pin_fixture(&model);
        let (pin, read) = v.pin(doc.clone(), 0).unwrap();
        let mut files = Files::default();
        for event in read {
            match event {
                TextEvent::Prefilled { done, .. } if done >= 512 && files.end == 0 => {
                    files.end = v.export_pin(&pin, &mut files.pin, 0).unwrap();
                }
                TextEvent::Error(e) => panic!("{e:?}"),
                _ => {}
            }
        }
        // A mark while the read was under way, or the whole chain if it had finished.
        assert!(files.end % 512 == 0 || files.end == 2_592, "{arch:?}");
        let first = files.pin.len();
        export_all(&v, &pin, &mut files);
        assert_eq!(files.end, 2_592, "{arch:?}");
        assert!(first > 0 && first <= files.pin.len());
        let mut whole = Files::default();
        export_all(&v, &pin, &mut whole);
        assert!(
            files.pin == whole.pin && files.state == whole.state,
            "{arch:?}"
        );
        // Nothing new appends nothing.
        let end = v.export_pin(&pin, &mut files.pin, files.end).unwrap();
        assert_eq!((end, files.pin == whole.pin), (2_592, true), "{arch:?}");
    }
}

/// A pin of the first two segments resumes a longer document to the logits an uninterrupted read
/// gives: the read after the import computes only the rest.
#[test]
fn a_partial_pin_resumes_to_identical_logits() {
    for arch in [Arch::Qwen35, Arch::Llama] {
        let dir = Temp::new();
        let model = pin_model(&dir, arch);
        let doc = document(2_600);
        // Not ending on a block: a read that does saves a state one block earlier too (its own
        // repeat looks up one token short), which cuts its chunks where the full read's are not.
        let (head, ..) = pin_file(&model, &doc[..1_030]);
        assert_eq!(head.end, 1_024, "{arch:?}");
        let (_, source, _held) = pin_file(&model, &doc);
        let (want, cached) = ask(&source, &doc);
        assert_eq!(cached, 2_592, "{arch:?}");
        let w = pin_fixture(&model);
        let resumed = import(&w, &doc, &head).unwrap();
        assert_eq!(resumed.tokens().len(), 1_024);
        let (all, read) = w.pin(doc.clone(), 0).unwrap();
        read_in(read);
        let (got, cached) = ask(&w, &doc);
        assert_eq!((got, cached), (want, 2_592), "{arch:?}");
        drop((resumed, all));
    }
}

/// A short pin sharing a long read's checkpoint keeps it when the read moves on to its next mark:
/// A = B[..2049] ends on B's first mark, which B's retirement of it must leave, here with two seats
/// and a prefill chunk (192) that no mark (2,048) is a multiple of. A is pinned while B reads,
/// which lands it within the window between B's marks (a pin waits a few steps for the engine); it
/// relies on A landing there, and `retiring_a_mark_keeps_the_checkpoint_a_pin_ends_on` (sched.rs)
/// is the deterministic guard.
#[test]
fn a_long_reads_next_mark_keeps_a_shorter_pins_checkpoint() {
    let dir = Temp::new();
    let model = pin_model(&dir, Arch::Qwen35);
    let mut o = LoadOptions::new(model, CPU);
    o.pin_mark = Some(2_048);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    o.knob_overrides = vec![
        (knob_id::MAX_BATCH_SEQS, 2.0),
        (knob_id::PREFILL_CHUNK_TOKENS, 192.0),
    ];
    let v = Larch::load(o).unwrap();
    let b_doc = document(5_000);
    let a_doc = b_doc[..2_049].to_vec();
    let (b, b_read) = v.pin(b_doc.clone(), 0).unwrap();
    let mut a = None;
    // B reads past its first mark, then A is pinned.
    for event in b_read {
        match event {
            TextEvent::Prefilled { done, .. } if done > 2_048 && a.is_none() => {
                a = Some(v.pin(a_doc.clone(), 0).unwrap());
            }
            TextEvent::Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    let (a, a_read) = a.expect("B read past its mark");
    read_in(a_read);
    let mut files = Files::default();
    export_all(&v, &a, &mut files);
    assert_eq!(files.end, 2_048);
    assert!(!files.state.is_empty(), "A's checkpoint");
    assert_eq!(
        ask(&v, &a_doc).1,
        2_048,
        "a question over A reuses its prefix"
    );
    drop(b);
}

/// A pin file's reader that runs `tear` before its second read: after the header, so once the
/// import has taken its blocks.
struct Tear<'a, F: FnMut()> {
    bytes: &'a [u8],
    reads: u32,
    tear: F,
}

impl<F: FnMut()> io::Read for Tear<'_, F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reads += 1;
        if self.reads == 2 {
            (self.tear)();
        }
        self.bytes.read(buf)
    }
}

/// An import that another document's reload breaks off, once the import has taken its blocks, is
/// `IMPORT_TORN`, which the caller keeps the file on, not a `Miss`, which has it delete the file;
/// the same file then imports whole (#591 review).
#[test]
fn an_import_a_reload_tears_keeps_its_file_and_imports_again() {
    let dir = Temp::new();
    let model = pin_model(&dir, Arch::Llama);
    let doc = document(805);
    let (files, ..) = pin_file(&model, &doc);
    // A context that holds the document beside a turn: its import reloads for nothing.
    let mut o = LoadOptions::new(model, CPU);
    o.pin_mark = Some(512);
    o.max_context = 17_200;
    o.prefix_cache_blocks = 64;
    let v = Larch::load(o).unwrap();
    assert!(v.deployment().ctx_max < 900 + larch::api::TURN_ROOM);
    // Another chat's document, longer than this context holds: its pin reloads into a plan.
    let other = document(900);
    let mut held = None;
    let mut reader = Tear {
        bytes: &files.pin[..],
        reads: 0,
        tear: || held = Some(v.pin(other.clone(), 0).unwrap().0),
    };
    let torn = v.import_pin(doc.clone(), 0, &mut reader, &mut &files.state[..]);
    assert!(
        matches!(
            torn,
            Err(PinError::Engine(EngineError::InvalidRequest(
                invalid::IMPORT_TORN
            )))
        ),
        "{torn:?}"
    );
    assert!(held.is_some(), "the reload ran during the import");
    drop(held);
    let pin = import(&v, &doc, &files).unwrap();
    assert_eq!(pin.tokens().len(), 800);
    assert_eq!(ask(&v, &doc).1, 800);
}

/// A record torn off the end of the file is ignored: a dense model imports the segments before
/// it; a hybrid, whose state is of the whole count, does not import.
#[test]
fn a_torn_last_record_is_ignored() {
    let doc = document(2_100);
    for arch in [Arch::Llama, Arch::Qwen35] {
        let dir = Temp::new();
        let model = pin_model(&dir, arch);
        let (mut files, ..) = pin_file(&model, &doc);
        assert_eq!(files.end, 2_096);
        files.pin.truncate(files.pin.len() - 100);
        let v = pin_fixture(&model);
        match arch {
            Arch::Llama => {
                let pin = import(&v, &doc, &files).unwrap();
                assert_eq!(pin.tokens().len(), 2_048);
                assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 128);
            }
            _ => assert!(matches!(import(&v, &doc, &files), Err(PinError::Miss))),
        }
    }
}

/// The header's bytes (see `larch_engine`'s `pinfile`): where each field starts, and its checksum's place.
mod head {
    pub const VERSION: usize = 8;
    pub const PROJECTOR: usize = 20;
    pub const ABI: usize = 28;
    pub const BUILD: usize = 32;
    pub const LAYERS: usize = 48;
    pub const BLOCK: usize = 52;
    pub const DTYPE: usize = 60;
    pub const STATE: usize = 68;
    pub const SEGMENT: usize = 76;
    pub const LEN: usize = 80;
}

/// `bytes` with its first `len` bytes changed by `edit` and the checksum after them made good
/// again (a header, or a state sidecar whole), so only the field is what is wrong.
fn mended(bytes: &[u8], len: usize, edit: impl FnOnce(&mut [u8])) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut out = bytes.to_vec();
    edit(&mut out[..len]);
    let sum = Sha256::digest(&out[..len]);
    out[len..len + 32].copy_from_slice(&sum);
    out
}

/// A change to a few bytes of a file.
type Edit = Box<dyn FnOnce(&mut [u8])>;

/// An import is refused as a Miss for each way a file is not this engine's, and publishes nothing:
/// no pinned block, and the blocks it took for the layers it got through are back.
#[test]
fn an_import_refuses_what_is_not_its_own() {
    let doc = document(805);
    for arch in [Arch::Llama, Arch::Qwen35] {
        let dir = Temp::new();
        let model = pin_model(&dir, arch);
        let (files, ..) = pin_file(&model, &doc);
        let v = pin_fixture(&model);
        let used = v.prefix_cache_stats().unwrap().unique_blocks_used;
        let bump = |at: usize, by: u8| move |h: &mut [u8]| h[at] = h[at].wrapping_add(by);
        let head_of = |edit: Edit| Files {
            pin: mended(&files.pin, head::LEN, edit),
            // The sidecar names its pin file by the header's checksum.
            state: files.state.clone(),
            end: files.end,
        };
        let state_len = files.state.len() - 32;
        let state_of = |edit: Edit| Files {
            pin: files.pin.clone(),
            state: mended(&files.state, state_len, edit),
            end: files.end,
        };
        // The sidecar's fields: magic 0, version 8, header sum 12, tokens covered 44, state bytes 52.
        let cases: Vec<(&str, Files)> = vec![
            ("a version", head_of(Box::new(bump(head::VERSION, 1)))),
            ("a kernel ABI", head_of(Box::new(bump(head::ABI, 1)))),
            ("a build", head_of(Box::new(bump(head::BUILD, 1)))),
            ("a projector", head_of(Box::new(bump(head::PROJECTOR, 1)))),
            ("the layers", head_of(Box::new(bump(head::LAYERS, 1)))),
            ("the block bytes", head_of(Box::new(bump(head::BLOCK, 1)))),
            ("the dtype", head_of(Box::new(bump(head::DTYPE, 1)))),
            ("the state size", head_of(Box::new(bump(head::STATE, 1)))),
            ("the segment", head_of(Box::new(bump(head::SEGMENT + 1, 1)))),
            ("a header sum", {
                let mut f = files.clone();
                f.pin[head::LEN] ^= 1;
                f
            }),
            ("a record's count", {
                let mut f = files.clone();
                f.pin[head::LEN + 32] ^= 1;
                f
            }),
            ("a token", {
                let mut f = files.clone();
                f.pin[head::LEN + 32 + 4 + 8] ^= 1;
                f
            }),
            ("a layer", {
                let mut f = files.clone();
                f.pin[head::LEN + 32 + 4 + 4 * 512 + 10] ^= 1;
                f
            }),
            ("a record's sum", {
                let mut f = files.clone();
                // The first record's, which ends 32 bytes before the second begins.
                let first = (f.pin.len() - head::LEN - 32) / 2;
                f.pin[head::LEN + 32 + first - 1] ^= 1;
                f
            }),
            ("a sidecar magic", state_of(Box::new(bump(0, 1)))),
            ("a sidecar version", state_of(Box::new(bump(8, 1)))),
            ("another file's sidecar", state_of(Box::new(bump(12, 1)))),
            (
                "no tokens covered",
                state_of(Box::new(|s| s[44..52].fill(0))),
            ),
            (
                "an oversized cover",
                state_of(Box::new(|s| s[44..52].fill(0xff))),
            ),
            ("an odd cover", state_of(Box::new(bump(44, 1)))),
            ("a cover past the document", state_of(Box::new(bump(45, 4)))),
            ("a state size", state_of(Box::new(bump(52, 1)))),
            ("a sidecar sum", {
                let mut f = files.clone();
                *f.state.last_mut().unwrap() ^= 1;
                f
            }),
            ("a state", {
                let mut f = files.clone();
                let at = f.state.len() - 33;
                f.state[at] ^= 1;
                f
            }),
        ];
        for (what, f) in cases {
            // A dense model has no state bytes to damage.
            if what == "a state" && arch == Arch::Llama {
                continue;
            }
            let got = import(&v, &doc, &f);
            assert!(
                matches!(got, Err(PinError::Miss)),
                "{arch:?}: {what}: {got:?}"
            );
            let stats = v.prefix_cache_stats().unwrap();
            assert_eq!(stats.pinned_blocks, 0, "{arch:?}: {what}");
            assert_eq!(stats.unique_blocks_used, used, "{arch:?}: {what}");
        }
        // Truncated anywhere, and tokens that are not the file's prefix.
        for cut in [0, 7, head::LEN, head::LEN + 40, files.pin.len() / 3] {
            let f = Files {
                pin: files.pin[..cut].to_vec(),
                ..files.clone()
            };
            let got = import(&v, &doc, &f);
            assert!(
                matches!(got, Err(PinError::Miss)),
                "{arch:?}: cut {cut}: {got:?}"
            );
        }
        for cut in [0, 7, 44, files.state.len() - 1] {
            let f = Files {
                state: files.state[..cut].to_vec(),
                ..files.clone()
            };
            let got = import(&v, &doc, &f);
            assert!(
                matches!(got, Err(PinError::Miss)),
                "{arch:?}: sidecar cut {cut}: {got:?}"
            );
        }
        let mut other = doc.clone();
        other[100] += 1;
        let got = import(&v, &other, &files);
        assert!(matches!(got, Err(PinError::Miss)), "{arch:?}: not a prefix");
        let got = import(&v, &doc[..790], &files);
        assert!(
            matches!(got, Err(PinError::Miss)),
            "{arch:?}: shorter than the file"
        );
        assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 0);
        assert_eq!(v.prefix_cache_stats().unwrap().unique_blocks_used, used);
        // And the good files still load after all of that.
        assert!(import(&v, &doc, &files).is_ok(), "{arch:?}");
    }
}

/// Dropping a pin gives its blocks back to the cache.
#[test]
fn dropping_a_pin_releases_its_blocks() {
    let dir = Temp::new();
    let model = pin_model(&dir, Arch::Qwen35);
    let (_files, v, pin) = pin_file(&model, &document(805));
    assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 50);
    drop(pin);
    assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 0);
}

/// Files of another model do not load: the identity of the GGUF is in the key.
#[test]
fn an_import_refuses_another_models_file() {
    let (a, b) = (Temp::new(), Temp::new());
    let doc = document(805);
    let (files, ..) = pin_file(&pin_model(&a, Arch::Llama), &doc);
    let other = pin_fixture(&pin_model(&b, Arch::Qwen35));
    assert!(matches!(import(&other, &doc, &files), Err(PinError::Miss)));
}

/// The pins held and a turn's room must fit the pool: a pin past that is `PINS_FULL`, not a cold
/// one, and dropping one makes room.
#[test]
fn pins_past_the_pool_are_refused() {
    let dir = Temp::new();
    let v = pin_fixture(&pin_model(&dir, Arch::Llama));
    let longest = v.document_fit(1).max_tokens as usize;
    assert_eq!(longest, 32_768 - larch::api::TURN_ROOM as usize);
    // Only the registrations count here: a read is withdrawn with its stream.
    let pin = |n: u32| {
        let mut doc = document(longest);
        doc[0] = n;
        v.pin(doc, 0).map(|p| p.0)
    };
    let first = pin(1).unwrap();
    let pool = v.deployment().kv_pool_blocks as usize;
    let room = (pool - larch::api::TURN_ROOM as usize / 16) / (longest / 16);
    assert!(room >= 2, "the pool of {pool} blocks holds one longest pin");
    let mut held: Vec<_> = (2..=room as u32).map(|n| pin(n).unwrap()).collect();
    let full = pin(100);
    assert!(
        matches!(full, Err(EngineError::InvalidRequest(invalid::PINS_FULL))),
        "{full:?}"
    );
    drop(first);
    held.push(pin(100).unwrap());
}
