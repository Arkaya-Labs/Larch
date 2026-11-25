//! Folio's documents held end to end: small ones as plain text, medium ones pinned in a real
//! (tiny) engine on the CPU, exported when the chat loses focus and imported when it regains it,
//! and a full pool giving way.

mod common;

use std::{cell::RefCell, fs, path::Path, sync::atomic::AtomicBool};

use common::Fake;
use larch::{
    EngineError, GenParams, Larch, LoadOptions, Prompt, TextEvent,
    api::{TURN_ROOM, invalid},
};
use larch_agent::{
    AgentEvent,
    folio::{
        self, Doc, Documents, Rep, cache,
        held::{Env, Held, Kind, Released},
        source,
    },
};
use larch_kernels::CPU;
use larch_model::{
    Arch,
    gguf::{Gguf, Value, write},
    tiny,
};
use serde_json::json;

/// A tiny byte-token model trained for 32,768 tokens, whose template writes each message as it is.
fn model(dir: &Path, tools_first: bool) -> std::path::PathBuf {
    let bytes = tiny::gguf(Arch::Llama, 3);
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
        if k.ends_with(".context_length") {
            *v = Value::U32(32_768);
        }
    }
    let body = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";
    // A template that renders the tools before the messages, as Qwen 3.5's does.
    let head = "{% if tools %}{% for t in tools %}{{ t.function.name }}\n{% endfor %}{% endif %}";
    let template = if tools_first {
        format!("{head}{body}")
    } else {
        body.into()
    };
    let tokens: Vec<_> = (0..256)
        .map(|b| Value::Str(format!("<0x{b:02X}>")))
        .collect();
    kv.extend([
        ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Value::Arr(tokens)),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Arr(vec![Value::I32(6); 256]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(97)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(255)),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)),
        ("tokenizer.chat_template".into(), Value::Str(template)),
    ]);
    let path = dir.join("tiny.gguf");
    fs::write(&path, write(&kv, &tensors)).unwrap();
    path
}

/// The tiny model loaded with a prefix cache and a context too small for a document and a turn,
/// as a pin needs it.
fn engine(dir: &Path, tools_first: bool) -> Larch {
    let mut o = LoadOptions::new(model(dir, tools_first), CPU);
    o.pin_mark = Some(512);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    Larch::load(o).unwrap()
}

/// The prefix cache's namespace of the test's turns and pins.
const NAMESPACE: u64 = 7;

/// One document of about `n` bytes, which the byte tokenizer reads as that many tokens.
fn document(n: usize) -> Vec<Doc> {
    let line = "Rent is $1,200 a month. The lease ends 3 March 2027.\n";
    let text: String = line.repeat(n / line.len() + 1);
    vec![source::doc(0, "lease.md", "sha", Rep::Text, text)]
}

/// The tokens a question after the documents is prefilled with, and the ones it reused.
fn ask(v: &Larch, docs: &[Doc]) -> (Vec<u32>, u32) {
    let mut prompt = folio::held::prefix(v, docs, &[]).unwrap();
    prompt.extend([3, 14, 15, 9, 2, 6]);
    let p = GenParams {
        max_tokens: 1,
        temperature: Some(0.0),
        logprobs: 5,
        prefix_cache: true,
        cache_namespace: NAMESPACE,
        ..GenParams::default()
    };
    let (mut first, mut cached) = (Vec::new(), 0);
    for event in v.generate(Prompt::Tokens(prompt), p).unwrap() {
        match event {
            TextEvent::Text {
                logprobs: Some(l), ..
            } => first.extend(l.iter().map(|t| t.logprob.to_bits())),
            TextEvent::Done { usage, .. } => cached = usage.cached_tokens,
            TextEvent::Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    (first, cached)
}

struct Run<'a> {
    v: &'a Larch,
    data: &'a Path,
    stop: AtomicBool,
}

impl Run<'_> {
    fn env(&self, context: u32) -> Env<'_> {
        Env {
            larch: self.v,
            data: self.data,
            namespace: NAMESPACE,
            context,
            keep: true,
            budget: None,
            stop: &self.stop,
            tools: Vec::new(),
        }
    }
}

/// The chip states of a call, as it told them.
fn chips(events: &[AgentEvent]) -> Vec<Documents> {
    (events.iter())
        .filter_map(|e| match e {
            AgentEvent::Documents(d) => Some(d.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_medium_set_is_pinned_with_progress_and_its_question_reuses_it() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let docs = document(900);
    let mut told = Vec::new();
    let held = Held::open(
        &run.env(512),
        docs.clone(),
        &mut |e| told.push(e),
        &mut || {},
    )
    .unwrap();
    assert_eq!((held.kind, held.pinned(&v)), (Kind::Pinned, true));
    let chips = chips(&told);
    // The loaded context is 512 tokens: the reload is announced before the read, and the
    // load that was measured gives the seconds.
    assert!(
        matches!(chips.first(), Some(Documents::MakingRoom { secs }) if *secs > 0),
        "{chips:?}"
    );
    assert!(
        matches!(chips.get(1), Some(Documents::Reading { .. })),
        "{chips:?}"
    );
    assert_eq!(chips.last(), Some(&Documents::Ready));
    // The question reads only its own tokens: the whole block is the engine's.
    let block = folio::held::prefix(&v, &docs, &[]).unwrap().len() as u32;
    assert_eq!(ask(&v, &docs).1, block / 16 * 16);
    // The pin is on disk, with its state.
    let files: Vec<_> = fs::read_dir(cache::dir(dir.path()))
        .unwrap()
        .map(|e| e.unwrap().path().extension().unwrap().to_owned())
        .collect();
    assert!(
        files.iter().any(|e| e == "pin") && files.iter().any(|e| e == "state"),
        "{files:?}"
    );
}

/// A model that read its last set slowly sends a set it would read for over two minutes to
/// ask_docs, though the device holds it (#590 F8).
#[test]
fn a_set_too_slow_to_read_goes_to_ask_docs() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    // A hundred tokens in two minutes.
    cache::keep_read_rate(&cache::dir(dir.path()), v.model_path(), 100, 120.0).unwrap();
    let held = Held::open(&run.env(512), document(900), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!((held.kind, held.pinned(&v)), (Kind::Large, false));
}

#[test]
fn no_room_is_made_when_the_loaded_context_already_holds_the_document() {
    let dir = tempfile::tempdir().unwrap();
    let mut o = LoadOptions::new(model(dir.path(), false), CPU);
    o.pin_mark = Some(512);
    o.max_context = 20480;
    o.prefix_cache_blocks = 64;
    let v = Larch::load(o).unwrap();
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let mut told = Vec::new();
    // Routed as pinned by the 512-token context the chat reports, but the engine holds more.
    let held = Held::open(
        &run.env(512),
        document(900),
        &mut |e| told.push(e),
        &mut || {},
    )
    .unwrap();
    assert_eq!(held.kind, Kind::Pinned);
    let chips = chips(&told);
    assert!(
        !chips
            .iter()
            .any(|c| matches!(c, Documents::MakingRoom { .. })),
        "{chips:?}"
    );
}

#[test]
fn a_chat_that_loses_focus_exports_and_drops_its_pin_and_imports_it_back_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let docs = document(900);
    let mut held = Held::open(&run.env(512), docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    let (before, cached) = ask(&v, &docs);
    assert!(cached > 0);
    let pinned = v.prefix_cache_stats().unwrap().pinned_blocks;
    assert!(pinned > 0);
    // Focus moves: exported, then dropped.
    held.release(&run.env(512), &mut |_| {}, false);
    assert!(!held.pinned(&v));
    assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, 0);
    // Focus returns: imported from the file, and the answer is what it was.
    let mut told = Vec::new();
    held.hold(&run.env(512), &mut |e| told.push(e), &mut || {})
        .unwrap();
    assert!(held.pinned(&v));
    assert_eq!(v.prefix_cache_stats().unwrap().pinned_blocks, pinned);
    assert_eq!(chips(&told).last(), Some(&Documents::Ready));
    assert_eq!(ask(&v, &docs), (before, cached));
    // A file the engine refuses (another build, a torn write) is read again, not an error.
    held.release(&run.env(512), &mut |_| {}, false);
    let pin = fs::read_dir(cache::dir(dir.path()))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "pin"))
        .unwrap();
    let mut bytes = fs::read(&pin).unwrap();
    bytes[8] ^= 0xff;
    fs::write(&pin, bytes).unwrap();
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(ask(&v, &docs).1, cached);
}

/// A file holds whole blocks only, so the import of a set whose length is not a multiple of a
/// block is short by its tail, and the set is pinned again over the import for that tail. The file
/// already holds everything that pin can export: it is kept as it is, not written again from the
/// start (on the 128k Folio run that rewrite copied 2.75 GB off the device before the first answer).
#[test]
fn a_pin_imported_short_by_a_tail_under_a_block_keeps_its_file() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let docs = document(900);
    assert_ne!(folio::held::prefix(&v, &docs, &[]).unwrap().len() % 16, 0);
    let mut held = Held::open(&run.env(512), docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    let (before, cached) = ask(&v, &docs);
    held.release(&run.env(512), &mut |_| {}, false);
    let pin = fs::read_dir(cache::dir(dir.path()))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "pin"))
        .unwrap();
    // A torn trailing record, which the import ignores: still there after the import only if the
    // file was not written again.
    let mut bytes = fs::read(&pin).unwrap();
    bytes.extend([0xa5; 7]);
    fs::write(&pin, &bytes).unwrap();
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert!(held.pinned(&v));
    assert!(
        fs::read(&pin).unwrap() == bytes,
        "the pin file was written again"
    );
    assert_eq!(ask(&v, &docs), (before.clone(), cached));
    // The kept file imports again to the same answer.
    held.release(&run.env(512), &mut |_| {}, false);
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(ask(&v, &docs), (before, cached));
}

/// An import short by more than a block (a read stopped at its first checkpoint) is read on over
/// the import, and the file is written again from the start: a torn tail never sits between old
/// and new segments, and the new file imports to the same answer.
#[test]
fn a_pin_imported_short_by_more_than_a_block_writes_its_file_again() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let docs = document(1500);
    let stopped = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(true),
    };
    assert!(Held::open(&stopped.env(512), docs.clone(), &mut |_| {}, &mut || {}).is_err());
    let pin = fs::read_dir(cache::dir(dir.path()))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "pin"))
        .unwrap();
    let mut bytes = fs::read(&pin).unwrap();
    bytes.extend([0xa5; 7]);
    fs::write(&pin, &bytes).unwrap();
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let mut held = Held::open(&run.env(512), docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    assert!(held.pinned(&v));
    let now = fs::read(&pin).unwrap();
    assert!(
        now.len() > bytes.len() && !now.ends_with(&[0xa5; 7]),
        "not written again"
    );
    let (before, cached) = ask(&v, &docs);
    assert!(cached > 0);
    held.release(&run.env(512), &mut |_| {}, false);
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(ask(&v, &docs), (before, cached));
}

/// An import the engine refuses deletes the file, and the set is read again in full: the release
/// after writes the whole file again, from the start, not from where the earlier session's file
/// ended (which left an empty file that every later focus read the set past) (#590 Mac review).
#[test]
fn a_refused_import_writes_the_whole_file_again() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let docs = document(900);
    let mut held = Held::open(&run.env(512), docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    let (before, cached) = ask(&v, &docs);
    held.release(&run.env(512), &mut |_| {}, false);
    let pin = fs::read_dir(cache::dir(dir.path()))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "pin"))
        .unwrap();
    let whole = fs::read(&pin).unwrap().len();
    // A file of another build: refused, deleted and read again.
    let mut bytes = fs::read(&pin).unwrap();
    bytes[8] ^= 0xff;
    fs::write(&pin, bytes).unwrap();
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert!(held.pinned(&v));
    held.release(&run.env(512), &mut |_| {}, false);
    assert_eq!(fs::read(&pin).map(|b| b.len()).ok(), Some(whole));
    // The new file imports to the same answer.
    held.hold(&run.env(512), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(ask(&v, &docs), (before, cached));
}

/// The app's idle release (`covered_only`) lets a pin go only when its file holds it whole: with
/// "Keep documents ready" off no file holds it, and letting it go would cost the next question a
/// full read. A focus change still lets it go (#591 review).
#[test]
fn an_idle_release_keeps_a_pin_no_file_holds() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let mut off = run.env(512);
    off.keep = false;
    let pinned = || v.prefix_cache_stats().unwrap().pinned_blocks;
    let mut held = Held::open(&off, document(900), &mut |_| {}, &mut || {}).unwrap();
    let blocks = pinned();
    assert!(blocks > 0);
    assert_eq!(held.release(&off, &mut |_| {}, true), Released::Kept);
    assert!(held.pinned(&v));
    assert_eq!(pinned(), blocks);
    // Written to its file, it goes.
    assert_eq!(
        held.release(&run.env(512), &mut |_| {}, true),
        Released::Covered
    );
    assert_eq!(pinned(), 0);
    assert_eq!(held.release(&off, &mut |_| {}, true), Released::Nothing);
    // A focus change lets it go with no file.
    fs::remove_dir_all(cache::dir(dir.path())).unwrap();
    held.hold(&off, &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(pinned(), blocks);
    assert_eq!(held.release(&off, &mut |_| {}, false), Released::Uncovered);
    assert_eq!(pinned(), 0);
}

/// A restart that reopens a pinned chat loads straight into the plan its pin was held on
/// (`cache::plan`), so the import reloads nothing: one load, no MakingRoom, the same answer.
#[test]
fn a_held_pin_reopens_with_one_load() {
    let dir = tempfile::tempdir().unwrap();
    let docs = document(900);
    let cache_dir = cache::dir(dir.path());
    let making_room = |told: &[AgentEvent]| {
        chips(told)
            .iter()
            .any(|d| matches!(d, Documents::MakingRoom { .. }))
    };
    let (asked, plan) = {
        let v = engine(dir.path(), false);
        let run = Run {
            v: &v,
            data: dir.path(),
            stop: AtomicBool::new(false),
        };
        let mut told = Vec::new();
        let mut held = Held::open(
            &run.env(512),
            docs.clone(),
            &mut |e| told.push(e),
            &mut || {},
        )
        .unwrap();
        assert!(making_room(&told), "the first read reloads: {told:?}");
        let seats =
            v.knobs().effective().values[larch::knob::spec("max_batch_seqs").unwrap().id as usize];
        let plan = larch::api::DocumentsPlan {
            max_context: v.deployment().ctx_max,
            seats: seats as u32,
        };
        cache::keep_plan(&cache_dir, "chat", Some((plan, 512))).unwrap();
        let asked = ask(&v, &docs);
        held.release(&run.env(512), &mut |_| {}, false);
        (asked, plan)
    };
    assert_eq!(cache::plan(&cache_dir, "chat"), Some((plan, 512)));
    let mut o = LoadOptions::new(model(dir.path(), false), CPU);
    o.pin_mark = Some(512);
    o.max_context = 512;
    o.prefix_cache_blocks = 64;
    o.documents = cache::plan(&cache_dir, "chat").map(|(p, _)| p);
    let v = Larch::load(o).unwrap();
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let mut told = Vec::new();
    let held = Held::open(
        &run.env(512),
        docs.clone(),
        &mut |e| told.push(e),
        &mut || {},
    )
    .unwrap();
    assert!(held.pinned(&v));
    assert!(!making_room(&told), "{told:?}");
    assert_eq!(v.deployment().ctx_max, plan.max_context);
    assert_eq!(ask(&v, &docs), asked);
    cache::keep_plan(&cache_dir, "chat", None).unwrap();
    assert_eq!(cache::plan(&cache_dir, "chat"), None);
}

#[test]
fn a_full_pool_unpins_the_other_chats_and_the_pin_is_tried_again() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    // Other chats' pins, registered only (a read is withdrawn with its stream), fill the pool
    // but for a few blocks past a turn's room.
    let longest = v.document_fit(1).max_tokens as usize;
    let mut others = Vec::new();
    let pin = |n: u32, len: usize| {
        let mut doc = vec![5u32; len];
        (doc[0], doc[1]) = (n % 90 + 1, n / 90 + 1);
        v.pin(doc, NAMESPACE).map(|p| p.0).map_err(|e| {
            (
                matches!(e, EngineError::InvalidRequest(invalid::PINS_FULL)),
                format!("{e:?}"),
            )
        })
    };
    others.push(pin(0, 800).unwrap());
    let pool = v.deployment().kv_pool_blocks as usize;
    let mut used = 50;
    while pool - TURN_ROOM as usize / 16 - used > 8 {
        let blocks = (pool - TURN_ROOM as usize / 16 - used - 8).min(longest / 16);
        others.push(pin(others.len() as u32, blocks * 16).unwrap());
        used += blocks;
    }
    assert!(matches!(pin(99, 900), Err((true, _))), "the pool is full");
    let others = RefCell::new(Some(others));
    let mut released = 0;
    let held = Held::open(&run.env(512), document(900), &mut |_| {}, &mut || {
        released += 1;
        others.borrow_mut().take();
    })
    .unwrap();
    assert_eq!((held.pinned(&v), released), (true, 1));
}

/// A kept pin imported into a pool full of other chats' pins has them give way and imports
/// again, with no notice: `PINS_FULL` arrives from the import as `NoRoom` (#590 F4).
#[test]
fn an_import_into_a_full_pool_unpins_the_other_chats_and_imports_again() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let mut held = Held::open(&run.env(512), document(900), &mut |_| {}, &mut || {}).unwrap();
    held.release(&run.env(512), &mut |_| {}, false);
    let longest = v.document_fit(1).max_tokens as usize;
    let pin = |n: u32, len: usize| {
        let mut doc = vec![5u32; len];
        (doc[0], doc[1]) = (n % 90 + 1, n / 90 + 1);
        v.pin(doc, NAMESPACE).map(|p| p.0)
    };
    let mut others = vec![pin(0, 800).unwrap()];
    let pool = v.deployment().kv_pool_blocks as usize;
    let mut used = 50;
    while pool - TURN_ROOM as usize / 16 - used > 8 {
        let blocks = (pool - TURN_ROOM as usize / 16 - used - 8).min(longest / 16);
        others.push(pin(others.len() as u32, blocks * 16).unwrap());
        used += blocks;
    }
    let others = RefCell::new(Some(others));
    let (mut released, mut notices) = (0, Vec::new());
    held.hold(
        &run.env(512),
        &mut |e| {
            if let AgentEvent::Notice(n) = e {
                notices.push(n);
            }
        },
        &mut || {
            released += 1;
            others.borrow_mut().take();
        },
    )
    .unwrap();
    assert_eq!((held.pinned(&v), released), (true, 1));
    assert!(notices.is_empty(), "{notices:?}");
}

/// A held set is the same only under the context it was routed against: a context that shrank
/// (the documents plan went back) routes it again (#590 F7). A large set is kept too, so it is not
/// routed before every question (#590 F11).
#[test]
fn a_set_is_routed_again_when_the_context_changes_and_a_large_one_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), false);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let docs = document(900);
    let small = Held::open(&run.env(32_768), docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(small.kind, folio::held::Kind::Small);
    assert!(small.same(&docs, &run.env(32_768)));
    assert!(!small.same(&docs, &run.env(8_192)));
    let long = document(200_000);
    let large = Held::open(&run.env(512), long.clone(), &mut |_| {}, &mut || {}).unwrap();
    assert_eq!(large.kind, folio::held::Kind::Large);
    assert!(large.same(&long, &run.env(512)));
}

#[test]
fn a_set_that_fits_the_loaded_context_is_plain_text_and_nothing_is_pinned() {
    let fake = Fake::new(32768);
    let mut agent = fake.agent();
    let plain = agent.system_prompt().to_owned();
    fs::write(
        fake.files().join("lease.md"),
        "Rent is $1,200 a month.\nThe lease ends 3 March 2027.\n",
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    let mut told = Vec::new();
    agent
        .read_documents(&[], &stop, &mut |e| told.push(e), &mut || {})
        .unwrap();
    assert_eq!(chips(&told), [Documents::Ready]);
    assert!(
        agent
            .system_prompt()
            .starts_with("You answer questions about the user's documents")
    );
    assert!(agent.system_prompt().ends_with(&plain));
    assert_eq!(fake.engine.prefix_cache_stats().unwrap().pinned_blocks, 0);
    // Asked again, nothing is read again.
    let mut told = Vec::new();
    agent
        .read_documents(&[], &stop, &mut |e| told.push(e), &mut || {})
        .unwrap();
    assert!(told.is_empty(), "{told:?}");
}

#[test]
fn a_set_too_long_to_hold_is_named_in_the_prompt_for_ask_docs_and_unnamed_when_gone() {
    let fake = Fake::new(2048);
    let mut agent = fake.agent();
    let plain = agent.system_prompt().to_owned();
    let file = fake.files().join("manual.md");
    fs::write(&file, "The pump runs at 40 bar.\n".repeat(4000)).unwrap();
    let stop = AtomicBool::new(false);
    let mut told = Vec::new();
    agent
        .read_documents(&[], &stop, &mut |e| told.push(e), &mut || {})
        .unwrap();
    assert!(
        matches!(chips(&told)[..], [Documents::FitsUpTo { .. }]),
        "{told:?}"
    );
    let prompt = agent.system_prompt().to_owned();
    assert!(
        prompt.starts_with("These documents are attached to this chat"),
        "{prompt}"
    );
    assert!(prompt.contains("A. manual.md") && prompt.contains("ask_docs"));
    assert!(prompt.ends_with(&plain) && !prompt.contains("40 bar"));
    assert!(agent.pinned_documents().is_none());
    // Once the file is gone, the chat's own prompt is back.
    fs::remove_file(&file).unwrap();
    agent
        .read_documents(&[], &stop, &mut |_| {}, &mut || {})
        .unwrap();
    assert_eq!(agent.system_prompt(), plain);
}

#[test]
fn links_are_the_web_addresses_of_a_message() {
    let said = "Read https://a.example/x/Paper.PDF?dl=1, and (https://drive.google.com/file/d/1Ab/view), and https://a.example/post. Not ftp://x or example.com.";
    assert_eq!(
        folio::links(said),
        [
            "https://a.example/x/Paper.PDF?dl=1",
            "https://drive.google.com/file/d/1Ab/view",
            "https://a.example/post"
        ]
    );
    assert!(folio::is_document_link(
        "https://a.example/x/Paper.PDF?dl=1"
    ));
    assert!(folio::is_document_link(
        "https://drive.google.com/file/d/1Ab/view"
    ));
    assert!(!folio::is_document_link("https://a.example/post"));
}

/// An article of enough text, and an app page of a line.
const ARTICLE: &str = concat!(
    "<html><head><title>On Rivers</title></head><body><article><h1>On Rivers</h1>",
    "<p>A river is the slow work of water finding the lowest road across a country. It cuts the ",
    "valley it runs in, carries the soil of the hills to the plain, and bends wherever the ground ",
    "resists it. Towns grew along them for the water and the trade, and every flood since has ",
    "reminded those towns who the road belonged to first. A river has no plan, but it has a habit, ",
    "and the habit is downhill, patient and wide, across the whole of the years it is given.</p>",
    "<p>Engineers who straighten a river learn the habit too: the water returns to its bends, ",
    "undoes the channel, and takes back the land it was lent. The old maps of the valley show the ",
    "same loops as the new ones, because the shape belongs to the water and the ground together, ",
    "not to the people who live beside it and draw the maps and build the walls and wait.</p>",
    "<p>Those who know a river well do not fight its habit. They build above its reach, farm the ",
    "floodplain in the dry years, and leave the wet ones to the reeds, which is the oldest and ",
    "cheapest flood defence there is, and the only one that has never once failed.</p>",
    "</article></body></html>"
);
const APP_PAGE: &str = "<html><head><title>App</title></head><body><div id=root>Sign in to continue</div></body></html>";

#[test]
fn an_article_link_is_a_one_page_document_and_an_app_page_is_left_to_the_tools() {
    use common::serve_as;
    let dir = tempfile::tempdir().unwrap();
    let stop = AtomicBool::new(false);
    let article = serve_as("text/html", ARTICLE);
    let app = serve_as("text/html", APP_PAGE);
    let doc = folio::store_url(dir.path(), "c", &article, &stop)
        .unwrap()
        .unwrap();
    assert_eq!((doc.rep, doc.pages.len()), (Rep::Html, 1));
    assert!(doc.text.contains("habit is downhill"), "{}", doc.text);
    assert!(
        folio::store_url(dir.path(), "c", &app, &stop)
            .unwrap()
            .is_none()
    );
    // Only the article is in the chat's record.
    assert_eq!(source::load(dir.path(), "c").unwrap().len(), 1);
    // As a turn reads them: the app page and a dead link make no notice.
    let fake = Fake::new(32768);
    let mut agent = fake.agent();
    let dead = "http://127.0.0.1:1/gone".to_owned();
    let mut told = Vec::new();
    agent
        .read_documents(
            &[app, dead, article],
            &stop,
            &mut |e| told.push(e),
            &mut || {},
        )
        .unwrap();
    assert_eq!(chips(&told), [Documents::Ready]);
    assert!(
        told.iter().all(|e| !matches!(e, AgentEvent::Notice(_))),
        "{told:?}"
    );
    assert!(agent.system_prompt().contains("habit is downhill"));
}

#[test]
fn a_turn_with_tools_reuses_a_pin_rendered_with_them() {
    let dir = tempfile::tempdir().unwrap();
    let v = engine(dir.path(), true);
    let run = Run {
        v: &v,
        data: dir.path(),
        stop: AtomicBool::new(false),
    };
    let tools = vec![json!({"type": "function", "function": {
        "name": "web_search", "description": "Search",
        "parameters": {"type": "object", "properties": {}}}})];
    let docs = document(900);
    let mut env = run.env(512);
    env.tools = tools.clone();
    let held = Held::open(&env, docs.clone(), &mut |_| {}, &mut || {}).unwrap();
    assert!(held.pinned(&v));
    // The template puts the tool names before the system message: a pin without them starts
    // elsewhere and no real turn shares it.
    let pinned = folio::held::prefix(&v, &docs, &tools).unwrap().len() as u32;
    let message = |role: &str, content: String| larch_model::chat::Message {
        role: role.into(),
        content,
    };
    let messages = vec![
        message("system", folio::system(&docs, "plain text")),
        message("user", "When does the lease end?".into()),
    ];
    let p = GenParams {
        tools,
        prefix_cache: true,
        cache_namespace: NAMESPACE,
        max_tokens: 1,
        temperature: Some(0.0),
        ..GenParams::default()
    };
    let mut cached = 0;
    for event in v.generate(Prompt::Messages(messages), p).unwrap() {
        match event {
            TextEvent::Done { usage, .. } => cached = usage.cached_tokens,
            TextEvent::Error(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    assert_eq!(cached, pinned / 16 * 16);
}
