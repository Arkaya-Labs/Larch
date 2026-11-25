//! The prefix cache in a chat, on the CPU engine with the scripted model: the anchor at a chat's
//! start, one namespace per data folder, a system prompt that stays the same bytes, and history
//! that renders the same after a hidden think block.

mod common;

use std::fs;

use common::{CACHE_BLOCKS, Fake, call, chat_tools, events, spaced};
use larch::{GenParams, Session, SessionOptions, TextEvent, Usage};
use larch_agent::{Agent, AgentEvent, AgentOptions};
use serde_json::{Value, json};

/// Each model step's prompt and cached tokens in the audit log, in order.
/// The note after a user message whose turn has the chat's tools.
const ON: &str = "\n\n[agent] Tools are on for this message.";

fn steps(fake: &Fake) -> Vec<(u64, u64)> {
    let day = &larch::bench::utc_now()[..10];
    let log = fs::read_to_string(fake.data().join(format!("agent/log/{day}.jsonl"))).unwrap();
    log.lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|e| e["kind"] == "turn")
        .map(|e| {
            let n = |k: &str| e[k].as_u64().unwrap();
            (n("prompt_tokens"), n("cached_tokens"))
        })
        .collect()
}

/// The cached tokens of a turn's last step, from its `Done`.
fn cached(got: &[AgentEvent]) -> u32 {
    match got.last() {
        Some(AgentEvent::Done(t)) => t.cached_tokens,
        other => panic!("{other:?}"),
    }
}

fn notices(got: &[AgentEvent]) -> Vec<&str> {
    got.iter()
        .filter_map(|e| match e {
            AgentEvent::Notice(n) => Some(n.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_test_harness_turns_the_prefix_cache_on() {
    // A custom model's cache stays off, with no error, unless the load attests it may share KV.
    let fake = Fake::new(32768);
    let stats = fake.engine.prefix_cache_stats().unwrap();
    assert!(stats.capacity_blocks > 0, "{stats:?}");
    assert!(stats.capacity_blocks <= CACHE_BLOCKS, "{stats:?}");
}

#[test]
fn two_chats_on_one_data_folder_both_start_warm() {
    let fake = Fake::new(32768);
    for chat in 0..2 {
        let mut agent = fake.agent();
        let system = agent.history()[0].content.clone();
        let hits = fake.engine.prefix_cache_stats().unwrap().hits;
        fake.say("Hi.");
        let got = events(agent.turn("hello").unwrap());
        // The first turn reuses the anchor's whole blocks: the system prompt and the tool list.
        let anchored = fake.anchored(&system, &chat_tools(), "hello");
        assert_eq!(cached(&got), anchored, "chat {chat}");
        let system_tokens = fake.tokenizer.encode(&system, false).len() as u32;
        assert!(
            anchored >= system_tokens,
            "chat {chat}: {anchored} {system_tokens}"
        );
        assert!(fake.engine.prefix_cache_stats().unwrap().hits > hits);
        // The chat closes, and its session's own entries go with it; the anchor's stay.
    }
}

#[test]
fn two_data_folders_never_share_the_cache() {
    let fake = Fake::new(32768);
    let stats = || fake.engine.prefix_cache_stats().unwrap();
    // A chat's first turn: its two anchors, then its step, which reads them. Each returns what
    // the turn added to the cache's hits and reused tokens.
    let first_turn = |agent: &mut Agent| {
        let before = stats();
        fake.say("Hi.");
        let got = events(agent.turn("hello").unwrap());
        let after = stats();
        let system = agent.history()[0].content.clone();
        let delta = (
            after.hits - before.hits,
            after.reused_prompt_tokens - before.reused_prompt_tokens,
        );
        (
            delta,
            cached(&got),
            fake.anchored(&system, &chat_tools(), "hello"),
        )
    };
    let mut first = fake.agent();
    let (cold, cached, anchored) = first_turn(&mut first);
    assert_eq!(cached, anchored);
    let other = AgentOptions {
        data_dir: fake.dir.path().join("other"),
        ..fake.opts()
    };
    // The same system prompt, byte for byte: only the data folder differs. Its anchors reuse
    // nothing of the first folder's, so its first turn hits the cache as the first one did.
    let mut second = Agent::new(&fake.larch, other).unwrap();
    assert_eq!(second.history()[0], first.history()[0]);
    let (delta, cached, anchored) = first_turn(&mut second);
    assert_eq!(delta, cold, "the other folder's anchors reused the first's");
    assert_eq!(cached, anchored);
    // Another chat on the first folder does share them: the check above could see a share.
    let mut third = fake.agent();
    let (warm, ..) = first_turn(&mut third);
    assert!(warm.0 > cold.0 && warm.1 > cold.1, "{warm:?} {cold:?}");
}

#[test]
fn a_remember_mid_chat_keeps_the_prompt_and_says_so_once() {
    let fake = Fake::new(32768);
    let mut agent = fake.agent();
    let system = agent.history()[0].clone();
    let mut said = Vec::new();
    for fact in ["The user drinks tea", "The user walks to work"] {
        // It runs unasked, so the turn goes on to the reply.
        fake.say(&call("remember", json!({"text": fact, "type": "user"})));
        fake.say("Saved.");
        let got = events(agent.turn(&format!("remember: {fact}")).unwrap());
        said.extend(notices(&got).into_iter().map(String::from));
    }
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].contains("next chat"), "{said:?}");
    // The next turn reuses all the KV the last step left: the prompt was not rebuilt.
    fake.say("Fine.");
    events(agent.turn("thanks").unwrap());
    let steps = steps(&fake);
    let [.., (last_prompt, _), (_, next_cached)] = steps[..] else {
        panic!("{steps:?}")
    };
    assert!(next_cached >= last_prompt, "{steps:?}");
    assert_eq!(agent.history()[0], system);
    // The next chat's prompt has the facts.
    let next = fake.agent();
    assert!(next.history()[0].content.contains("The user drinks tea"));
}

#[test]
fn the_system_prompt_stays_the_same_bytes_for_ten_turns() {
    let fake = Fake::new(32768);
    fs::write(fake.files().join("a.txt"), "alpha\n").unwrap();
    let mut agent = fake.agent();
    let system = agent.history()[0].content.clone();
    for turn in 0..10 {
        match turn {
            3 => {
                fake.say(&call("read_file", json!({"path": "a.txt"})));
                fake.say("It says alpha.");
                events(agent.turn("read a.txt").unwrap());
            }
            6 => {
                let fact = json!({"text": "The user reads a.txt", "type": "user"});
                fake.say(&spaced("remember", fact));
                fake.say("Saved.");
                // A file was read in this chat, so the save asks.
                events(agent.turn("remember that").unwrap());
                events(agent.confirm(true).unwrap());
            }
            _ => {
                fake.say("Yes.");
                events(agent.turn(&format!("turn {turn}")).unwrap());
            }
        }
        assert_eq!(agent.history()[0].content, system, "turn {turn}");
    }
    // Each step reuses everything the step before it computed.
    let steps = steps(&fake);
    for pair in steps.windows(2) {
        let [(prompt, _), (_, cached)] = pair else {
            unreachable!()
        };
        assert!(cached >= prompt, "{steps:?}");
    }
}

#[test]
fn history_after_a_hidden_think_block_renders_the_same_on_chatml() {
    // The parked KV holds each answer's thinking and the history does not, so a turn reuses the
    // KV up to where the last answer starts, and prefills that answer and the new user line.
    let fake = Fake::chatml(32768);
    let opts = AgentOptions {
        arch: "qwen3".into(),
        ..fake.opts()
    };
    let mut agent = Agent::new(&fake.larch, opts).unwrap();
    for (i, user) in ["one", "two", "three"].into_iter().enumerate() {
        fake.say(&format!("<think>Let me count.</think>Answer {i}."));
        events(agent.turn(user).unwrap());
        if i > 0 {
            let steps = steps(&fake);
            let [.., (prev_prompt, _), (prompt, cached)] = steps[..] else {
                panic!("{steps:?}")
            };
            assert_eq!(cached, prev_prompt, "turn {}: {steps:?}", i + 1);
            let fed = format!(
                "Answer {}.<|im_end|>\n<|im_start|>user\n{user}{ON}<|im_end|>\n<|im_start|>assistant\n",
                i - 1
            );
            let fed = fake.tokenizer.encode(&fed, false).len() as u64;
            assert_eq!(prompt - cached, fed, "turn {}: {steps:?}", i + 1);
        }
        assert_eq!(agent.history()[2 * i + 2].content, format!("Answer {i}."));
    }
}

#[test]
fn a_chat_opens_when_its_anchor_fails_and_its_first_turn_says_why() {
    // The system prompt and the tool list are longer than this context, so the anchor fails.
    let fake = Fake::new(512);
    let mut agent = fake.agent();
    let got = events(agent.turn("hello").unwrap());
    let said = notices(&got);
    let why = "the system prompt did not reach the prefix cache: ";
    assert!(said.iter().any(|n| n.starts_with(why)), "{got:?}");
}

#[test]
fn a_chat_opens_with_no_request_and_its_first_turn_anchors_first() {
    // A caller may open a chat holding a lock (the desktop app opens one under its state's), so
    // opening sends the engine nothing; the anchor's prefills run in the first turn, which the
    // caller can cancel.
    let on = Fake::new(32768);
    let stats = || on.engine.prefix_cache_stats().unwrap();
    let id = on.agent().chat_id().to_owned();
    let mut chat = Agent::open(&on.larch, &id, on.opts()).unwrap();
    assert_eq!((on.fed_tokens(), stats().lookups), (0, 0));
    // The first turn sends the anchor of the whole tool list first, then the one of none, then
    // the step; the next turn sends no anchor.
    let system = chat.history()[0].content.clone();
    let anchor = on.tokens(&[("system", &system), ("user", "")], &chat_tools());
    on.say("Hi.");
    let got = events(chat.turn("hello").unwrap());
    assert_eq!(cached(&got), on.anchored(&system, &chat_tools(), "hello"));
    assert!(on.fed().starts_with(&on.text_of(&anchor)), "{}", on.fed());
    assert_eq!(stats().lookups, 3);
    on.say("Fine.");
    events(chat.turn("and you?").unwrap());
    assert_eq!(stats().lookups, 4);

    // A chat closed at its y/n question reopens with no request too, and its confirm, the
    // reopened chat's first turn, anchors first.
    let mut chat = on.agent();
    on.say(&call(
        "write_file",
        json!({"path": "a.txt", "text": "alpha"}),
    ));
    events(chat.turn("write a.txt").unwrap());
    let id = chat.chat_id().to_owned();
    drop(chat);
    let fed = on.fed_tokens();
    let lookups = stats().lookups;
    let mut chat = Agent::open(&on.larch, &id, on.opts()).unwrap();
    assert!(chat.parked().is_some());
    assert_eq!((on.fed_tokens(), stats().lookups), (fed, lookups));
    on.say("Written.");
    events(chat.confirm(true).unwrap());
    // Its anchors find the first chat's in the cache, so the model is fed the end of the first
    // one first, up to its generation prompt; then the other anchor, then the step.
    assert_eq!(stats().lookups, lookups + 3);
    // The cache keeps whole 16-token blocks and a lookup feeds at least one token, so the
    // uncached end starts at the last block edge before the anchor's last token (where the temp
    // folder's path puts it).
    let got = on.fed_from(fed);
    let end = on.text_of(&anchor[(anchor.len() - 1) / 16 * 16..]);
    assert!(got.starts_with(&end), "{end:?} then {got}");

    // With the cache off a chat anchors never.
    let off = Fake::uncached(32768);
    let stats = || off.engine.prefix_cache_stats().unwrap();
    assert_eq!(stats().capacity_blocks, 0);
    let mut chat = off.agent();
    assert_eq!(off.fed(), "");
    off.say("Hi.");
    events(chat.turn("hello").unwrap());
    let system = chat.history()[0].content.clone();
    let user = format!("hello{ON}");
    let turn = off.tokens(&[("system", &system), ("user", &user)], &chat_tools());
    assert!(off.fed().starts_with(&off.text_of(&turn)), "{}", off.fed());
}

/// Writes a probe for `fake`'s model under `sha` that gives every turn the probability
/// sigmoid(`bias`) of needing a tool: its weights are all 0.
fn constant_probe(fake: &Fake, sha: &str, bias: f32) {
    let layers = larch_agent::probe::layers(fake.larch.deployment().n_layers);
    let x =
        larch_agent::probe::features(&fake.larch, larch_agent::probe::prompt("", "hi"), &layers);
    write_probe(fake, sha, bias, layers, x.unwrap().len());
}

/// Writes a probe under `sha` that reads `layers` and takes `n` features, all weighted 0.
fn write_probe(fake: &Fake, sha: &str, bias: f32, layers: Vec<u16>, n: usize) {
    use larch_agent::probe::{self, Probe};
    let file = Probe {
        schema: probe::SCHEMA.into(),
        layers,
        mean: vec![0.0; n],
        scale: vec![1.0; n],
        weights: vec![0.0; n],
        bias,
        threshold: probe::THRESHOLD,
        rows: 2,
        auroc: None,
    };
    let path = probe::path(&fake.data(), sha);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::to_string(&file).unwrap()).unwrap();
}

#[test]
fn a_no_tool_verdict_renders_the_second_prefix() {
    let fake = Fake::new(32768);
    let sha = "cd".repeat(32);
    let opts = AgentOptions {
        router: larch_agent::router::Router::Probe,
        ..fake.opts()
    };
    let probed = |fake: &Fake| {
        let day = &larch::bench::utc_now()[..10];
        let log = fs::read_to_string(fake.data().join(format!("agent/log/{day}.jsonl"))).unwrap();
        log.lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|e| e["step"] == "router probe")
            .map(|e| e["detail"]["tool"].clone())
            .collect::<Vec<_>>()
    };
    for (bias, tool) in [(-5.0, false), (5.0, true)] {
        constant_probe(&fake, &sha, bias);
        let mut agent = Agent::new(&fake.larch, opts.clone()).unwrap();
        agent.set_model_sha256(Some(sha.clone()));
        let system = agent.history()[0].content.clone();
        let shown = if tool { chat_tools() } else { Vec::new() };
        fake.say("Hi.");
        let got = events(agent.turn_with(&fake.larch, "hello").unwrap());
        assert!(notices(&got).is_empty(), "{got:?}");
        assert_eq!(probed(&fake).last(), Some(&json!(tool)));
        // The step renders with the list the verdict gives: none, or all of it, and the note of
        // its tools only with them.
        let user = if tool {
            format!("hello{ON}")
        } else {
            "hello".into()
        };
        let prompt = fake.tokens(&[("system", system.as_str()), ("user", &user)], &shown);
        let (prompt_tokens, cached_tokens) = *steps(&fake).last().unwrap();
        assert_eq!(prompt_tokens, prompt.len() as u64, "tool {tool}");
        // And starts warm from that list's anchor: a chat has those two prefixes, no third.
        let anchored = u64::from(fake.anchored(&system, &shown, "hello"));
        assert!(
            cached_tokens >= anchored && anchored > 0,
            "tool {tool}: {cached_tokens} {anchored}"
        );
    }
    // `turn` without the facade (the app's `turn_for`) still reaches the probe through the
    // agent's own: no gate, no notice.
    let mut agent = Agent::new(&fake.larch, opts).unwrap();
    agent.set_model_sha256(Some(sha));
    let before = probed(&fake).len();
    fake.say("Hi.");
    let got = events(agent.turn("hello").unwrap());
    assert!(notices(&got).is_empty(), "{got:?}");
    assert_eq!(probed(&fake).len(), before + 1);
}

#[test]
fn a_probe_of_another_width_says_so_once_and_the_model_decides_from_then_on() {
    let fake = Fake::new(32768);
    let sha = "ef".repeat(32);
    let layers = larch_agent::probe::layers(fake.larch.deployment().n_layers);
    // One weight per layer: it loads, as the file alone cannot say the model's width.
    let n = layers.len();
    write_probe(&fake, &sha, 5.0, layers, n);
    let opts = AgentOptions {
        router: larch_agent::router::Router::Probe,
        ..fake.opts()
    };
    let mut agent = Agent::new(&fake.larch, opts).unwrap();
    agent.set_model_sha256(Some(sha));
    let lines = |fake: &Fake, step: &str| {
        let day = &larch::bench::utc_now()[..10];
        let log = fs::read_to_string(fake.data().join(format!("agent/log/{day}.jsonl"))).unwrap();
        log.lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|e| e["step"] == step)
            .count()
    };
    // The model decides alone: no gate token before the step.
    fake.say("Hi.");
    let got = events(agent.turn_with(&fake.larch, "hello").unwrap());
    let said = notices(&got);
    assert_eq!(said.len(), 1, "{got:?}");
    assert!(
        said[0].starts_with("router probe: no usable probe for this model (the model gave ")
            && said[0].contains(&format!("the probe takes {n}")),
        "{}",
        said[0]
    );
    assert_eq!(
        (lines(&fake, "router probe"), lines(&fake, "router logprob")),
        (1, 0)
    );
    // The next turn neither runs the probe's prefill again nor says it again.
    fake.say("Hi again.");
    let got = events(agent.turn_with(&fake.larch, "hello again").unwrap());
    assert!(notices(&got).is_empty(), "{got:?}");
    assert_eq!(
        (lines(&fake, "router probe"), lines(&fake, "router logprob")),
        (1, 0)
    );
    assert_eq!(fake.unsaid(), 0);
}

/// The logprob gate's lines in the audit log, in order.
fn gates(fake: &Fake) -> Vec<Value> {
    let day = &larch::bench::utc_now()[..10];
    let log = fs::read_to_string(fake.data().join(format!("agent/log/{day}.jsonl"))).unwrap();
    log.lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|e| e["step"] == "router logprob")
        .collect()
}

#[test]
fn a_session_that_keeps_no_kv_reads_nothing_from_the_prefix_cache() {
    // Why the logprob gate's session keeps KV: the engine looks a request up in the prefix cache
    // only when it is stateless or its session keeps KV, so a gate that kept none prefilled its
    // whole prompt.
    let fake = Fake::new(32768);
    let opts = |keep_kv| SessionOptions {
        system: Some("You answer in one word, and only ever one. ".repeat(8)),
        keep_kv,
        ..SessionOptions::default()
    };
    let usage = |session: &mut Session| -> Usage {
        let p = GenParams {
            max_tokens: 1,
            cache_namespace: 9,
            ..GenParams::default()
        };
        for event in session.send("hello", p).unwrap() {
            match event {
                TextEvent::Done { usage, .. } => return usage,
                TextEvent::Error(e) => panic!("{e:?}"),
                TextEvent::Text { .. }
                | TextEvent::Prefilled { .. }
                | TextEvent::Thinking { .. } => {}
            }
        }
        panic!("the stream ended without Done");
    };
    // The first session publishes its prompt's whole blocks; it stays open, so they stay.
    let mut first = fake.larch.session(opts(true));
    let published = usage(&mut first);
    assert_eq!(published.cached_tokens, 0, "{published:?}");
    let whole = published.prompt_tokens.saturating_sub(1) / 16 * 16;
    assert!(whole >= 16, "{published:?}");
    let mut without = fake.larch.session(opts(false));
    let read = usage(&mut without);
    assert_eq!(read.cached_tokens, 0, "{read:?}");
    let mut with = fake.larch.session(opts(true));
    let read = usage(&mut with);
    assert_eq!(read.cached_tokens, whole, "{read:?}");
    drop(first);
}

#[test]
fn the_gate_reads_what_the_prefix_cache_holds_on_every_turn() {
    let fake = Fake::new(32768);
    let opts = AgentOptions {
        router: larch_agent::router::Router::Logprob,
        ..fake.opts()
    };
    let mut agent = Agent::new(&fake.larch, opts).unwrap();
    let system = agent.history()[0].content.clone();
    // The gate's one token, then the step's reply.
    fake.say_tokens(&[fake.token("H")]);
    fake.say("Hi.");
    let got = events(agent.turn("hello").unwrap());
    let anchored = fake.anchored(&system, &chat_tools(), "hello");
    let gate = gates(&fake);
    assert_eq!(gate.len(), 1, "{gate:?}");
    // The first turn's gate reads the anchor, as the step does: its prompt is the step's.
    let (prompt, _) = steps(&fake)[0];
    let detail = &gate[0]["detail"];
    assert_eq!(detail["cached_tokens"], json!(anchored), "{detail}");
    assert_eq!(detail["prompt_tokens"], json!(prompt), "{detail}");
    assert!(detail["ms"].as_f64().is_some_and(|ms| ms > 0.0), "{detail}");
    assert_eq!(cached(&got), anchored);
    // The next turn's gate reads the first turn's prompt, which the chat's session published:
    // the first gate's own blocks went with its reset.
    fake.say_tokens(&[fake.token("H")]);
    fake.say("Hi again.");
    events(agent.turn("hello again").unwrap());
    let gate = gates(&fake);
    assert_eq!(gate.len(), 2, "{gate:?}");
    let read = gate[1]["detail"]["cached_tokens"].as_u64().unwrap();
    assert!(
        read >= prompt / 16 * 16 && read >= u64::from(anchored),
        "{read} {prompt} {anchored}"
    );
    assert_eq!(fake.unsaid(), 0);
}

/// Opens a chat and runs one turn.
fn opened(fake: &Fake) -> Agent {
    let mut agent = fake.agent();
    fake.say("Hi.");
    events(agent.turn("hello").unwrap());
    agent
}

/// One turn of `agent`, and its last step's cached tokens.
fn turn(fake: &Fake, agent: &mut Agent, user: &str) -> u32 {
    fake.say("Ok.");
    cached(&events(agent.turn(user).unwrap()))
}

#[test]
fn a_delete_reaches_an_open_chat_at_its_next_turn_with_its_history_whole() {
    let fake = Fake::new(32768);
    let data = fake.data();
    let tea =
        larch_agent::memory::remember(&data, "The user drinks green tea", "user", "t").unwrap();
    larch_agent::memory::remember(&data, "The user's sister Maya lives in Austin", "user", "t")
        .unwrap();
    // Two chats open at once.
    let mut a = opened(&fake);
    let mut b = opened(&fake);
    turn(&fake, &mut a, "one");
    let warm = turn(&fake, &mut a, "two");
    assert!(a.history()[0].content.contains("green tea"));
    let before = a.history();
    // The app deletes a fact while both chats are open: nothing is cancelled or cleared.
    larch_agent::memory::forget(&data, &tea.slug).unwrap();
    assert_eq!(a.history(), before);
    let cold = turn(&fake, &mut a, "three");
    let history = a.history();
    assert!(
        !history[0].content.contains("green tea"),
        "{}",
        history[0].content
    );
    assert!(history[0].content.contains("Maya lives in Austin"));
    // Every message after the system prompt is still there, then the new turn's two.
    assert_eq!(history.len(), before.len() + 2);
    assert_eq!(history[1..before.len()], before[1..]);
    // One prefill: the new prompt's anchor is all the turn read from the cache.
    assert!(cold < warm, "{cold} {warm}");
    // And then warm again.
    assert!(turn(&fake, &mut a, "four") > warm);
    // The other chat sees it at its next turn too.
    turn(&fake, &mut b, "again");
    assert!(!b.history()[0].content.contains("green tea"));
}

#[test]
fn a_chat_whose_memory_block_did_not_change_keeps_its_cached_prefix() {
    let fake = Fake::new(32768);
    let data = fake.data();
    larch_agent::memory::remember(&data, "The user drinks green tea", "user", "t").unwrap();
    let mut a = opened(&fake);
    let mut warm = turn(&fake, &mut a, "one");
    let system = a.history()[0].content.clone();
    // Off and on again moves rev twice, but the block and the tools come out the same.
    larch_agent::memory::set_enabled(&data, false).unwrap();
    larch_agent::memory::set_enabled(&data, true).unwrap();
    let cached = turn(&fake, &mut a, "two");
    assert!(cached > warm, "{cached} {warm}");
    warm = cached;
    // An add (in another chat or a capture) moves no rev: the prompt stays, the prefix too.
    larch_agent::memory::remember(&data, "The user walks to work", "user", "t").unwrap();
    let cached = turn(&fake, &mut a, "three");
    assert!(cached > warm, "{cached} {warm}");
    assert_eq!(a.history()[0].content, system);
}

#[test]
fn memory_off_takes_the_block_and_the_memory_tools_out_of_an_open_chat() {
    let fake = Fake::new(32768);
    let data = fake.data();
    larch_agent::memory::remember(&data, "The user drinks green tea", "user", "t").unwrap();
    let mut a = opened(&fake);
    let len = a.history().len();
    larch_agent::memory::set_enabled(&data, false).unwrap();
    let from = fake.fed_tokens();
    turn(&fake, &mut a, "one");
    let history = a.history();
    assert!(
        !history[0].content.contains("Memory index"),
        "{}",
        history[0].content
    );
    assert_eq!(history.len(), len + 2);
    // The re-prefilled prompt lists no memory tool.
    let fed = fake.fed_from(from);
    assert!(fed.contains("read_file"), "{fed}");
    for tool in ["\"remember\"", "\"recall\"", "\"forget\""] {
        assert!(!fed.contains(tool), "{tool}: {fed}");
    }
    // A new chat opens without them too, and a recall that still comes is refused.
    let b = fake.agent();
    assert!(!b.history()[0].content.contains("green tea"));
    let ctx = larch_agent::tool::ToolCtx {
        allowed: Vec::new(),
        data_dir: data.clone(),
        source: "t".into(),
    };
    let registry = larch_agent::tool::Registry::builtin();
    let e = (registry.get("recall").unwrap())
        .call(&json!({"query": "tea"}), &ctx)
        .unwrap_err();
    assert!(e.to_string().contains("memory is off"), "{e}");
    // On again: back at the next turn.
    larch_agent::memory::set_enabled(&data, true).unwrap();
    turn(&fake, &mut a, "two");
    assert!(a.history()[0].content.contains("green tea"));
}

#[test]
fn off_then_a_turn_then_on_brings_the_block_back_with_one_reprefill_each_way() {
    let fake = Fake::new(32768);
    let data = fake.data();
    larch_agent::memory::remember(&data, "The user drinks green tea", "user", "t").unwrap();
    let mut a = opened(&fake);
    turn(&fake, &mut a, "one");
    let warm = turn(&fake, &mut a, "two");
    let on = a.history()[0].content.clone();
    assert!(on.contains("file and memory tools"), "{on}");
    larch_agent::memory::set_enabled(&data, false).unwrap();
    let cold = turn(&fake, &mut a, "three");
    assert!(cold < warm, "{cold} {warm}");
    let off = a.history()[0].content.clone();
    // Off, the prompt names no memory tool and shows no memory, and says memory is off, so a
    // request to remember is answered with where to turn it on.
    assert!(
        !off.contains("memory tools") && !off.contains("green tea"),
        "{off}"
    );
    assert!(
        off.contains("Memory is off") && off.contains("Memory page"),
        "{off}"
    );
    let warm_off = turn(&fake, &mut a, "four");
    assert!(warm_off > cold, "{warm_off} {cold}");
    larch_agent::memory::set_enabled(&data, true).unwrap();
    let cold = turn(&fake, &mut a, "five");
    // On again, the prompt is the same bytes as before, and the cache warms again once: less
    // than its warm turn before off, though the history has grown since.
    assert_eq!(a.history()[0].content, on);
    assert!(cold < warm, "{cold} {warm}");
    let rewarm = turn(&fake, &mut a, "six");
    assert!(rewarm > cold, "{rewarm} {cold}");
    let said: Vec<String> = (a.history().iter().skip(1).step_by(2))
        .map(|m| m.content.split("\n\n").next().unwrap().to_owned())
        .collect();
    assert_eq!(
        said,
        ["hello", "one", "two", "three", "four", "five", "six"]
    );
}

#[test]
fn a_correction_in_one_chat_reaches_another_open_chat() {
    let fake = Fake::new(32768);
    let data = fake.data();
    let maya =
        larch_agent::memory::remember(&data, "The user's sister Maya lives in Austin", "user", "t")
            .unwrap();
    let mut a = opened(&fake);
    turn(&fake, &mut a, "one");
    let rev = larch_agent::memory::rev(&data);
    // Chat B's capture corrects the fact, as `capture::save` does for a `replaces`.
    let said = ["Maya moved to Denver last month.".to_string()];
    let item = format!(
        "{{\"facts\":[{{\"text\":\"The user's sister Maya lives in Denver.\",\"quote\":\"Maya moved to Denver last month\",\"type\":\"user\",\"replaces\":\"{}\"}}]}}",
        maya.slug
    );
    let out = larch_agent::memory::capture::save(
        &data,
        &item,
        false,
        &said,
        std::slice::from_ref(&maya.slug),
        "chat B",
    )
    .unwrap();
    assert_eq!(out.saved.len(), 1, "{out:?}");
    assert_eq!(larch_agent::memory::rev(&data), rev + 1);
    turn(&fake, &mut a, "two");
    let system = &a.history()[0].content;
    assert!(
        system.contains("Denver") && !system.contains("Austin"),
        "{system}"
    );
    // A `remember` that updates a near-duplicate moves rev too; a new fact does not.
    let rev = larch_agent::memory::rev(&data);
    larch_agent::memory::remember(&data, "The user walks to work", "user", "t").unwrap();
    assert_eq!(larch_agent::memory::rev(&data), rev);
    let r = larch_agent::memory::remember(&data, "the user walks to work!", "user", "t").unwrap();
    assert!(r.old.is_some());
    assert_eq!(larch_agent::memory::rev(&data), rev + 1);
}

/// A kept "no tools" stands when the probe reads a later message as needing a tool past
/// `probe_high`: the probe routes within the user's limit and never lifts it.
#[test]
fn a_probe_hold_does_not_lift_a_kept_no_tools() {
    const KEPT: &str = "\n\n[agent] No tools until the user turns them back on: the user asked.";
    let fake = Fake::new(32768);
    let sha = "ef".repeat(32);
    fs::create_dir_all(fake.data().join("agent")).unwrap();
    fs::write(
        fake.data().join("agent/config.toml"),
        "router = \"probe\"\nprobe_low = 0.02\nprobe_high = 0.5\n",
    )
    .unwrap();
    // Every message reads as needing a tool: p = sigmoid(5) ≈ 0.99.
    constant_probe(&fake, &sha, 5.0);
    let opts = AgentOptions {
        router: larch_agent::router::Router::Probe,
        ..fake.opts()
    };
    let mut agent = Agent::new(&fake.larch, opts).unwrap();
    agent.set_model_sha256(Some(sha));
    fake.say("Fine.");
    events(agent.turn("from now on, no tools").unwrap());
    fake.say("Paris.");
    let got = events(agent.turn("what is the capital of France?").unwrap());
    assert!(
        !got.iter().any(|e| matches!(e, AgentEvent::ToolCall { .. })),
        "{got:?}"
    );
    let users: Vec<String> = (agent.history().into_iter())
        .filter(|m| m.role == larch::Role::User)
        .map(|m| m.content)
        .collect();
    assert!(users.iter().all(|u| u.ends_with(KEPT)), "{users:?}");
    assert_eq!(fake.unsaid(), 0);
}
