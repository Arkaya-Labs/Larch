//! A1B2 CPU evidence for `larch bench`: the real engine on its fake model with the tiny tokenizer
//! fixture and its chat template, and a fake Ollama over a local socket.

use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use larch_engine::{EngineEvent, EngineHandle, Request, start_with};
use larch_kernels::{KError, KvPool, Stream, Tensor};
use larch_model::{
    Arch, FakeModel, Model, ModelConfig, StepBatch,
    gguf::{Gguf, Value as Kv, write},
    tiny,
};
use larch_records::{EngineSample, ReadFrom, Record};
use serde_json::{Value, json};

use larch::bench::{self, EngineKind, Mode, Options, Replay, Telemetry, Workload};

static NEXT: AtomicU64 = AtomicU64::new(1);

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-bench-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn file(&self, name: &str, body: &str) -> PathBuf {
        let p = self.0.join(name);
        fs::write(&p, body).unwrap();
        p
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn vocab() -> Gguf {
    let bytes = fs::read(tiny::TOKENIZER_GGUF).unwrap();
    Gguf::parse(&bytes).unwrap()
}

/// The fixture's vocabulary under a template whose generation prompt ends in the byte `a`: the fake
/// model then answers `b`, `c`, `d`, ..., which re-tokenise to the same ids. The fixture's own
/// template ends in `\n\n`, after which the fake's next ids are merges that re-tokenise differently, and
/// generated replay rightly reports `cached_tokens < tokens_in_kv()` (10.1.1).
fn roundtrip_vocab() -> Gguf {
    const T: &str = "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>a{{ m.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>a{% endif %}";
    let mut kv: Vec<_> = vocab().kv.into_iter().collect();
    for (k, v) in &mut kv {
        if k == "tokenizer.chat_template" {
            *v = Kv::Str(T.into());
        }
    }
    let no_tensors: &[(&str, Vec<u64>, u32, Vec<u8>)] = &[];
    Gguf::parse(&write(&kv, no_tensors)).unwrap()
}

/// The real engine and scheduler on the fake model, sized to the tiny tokenizer's vocabulary.
fn engine(opts: &Options) -> EngineHandle {
    let mut e = bench::engine_options(opts).unwrap();
    e.ring_capacity = 4096;
    start_with(e, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(331))))).unwrap()
}

/// The fake model at 1 ms a step. The engine checks cancels before every step, so a cancel sent on
/// a request's first token always lands while a long answer still has hundreds of steps to go.
struct Paced(FakeModel);

impl Model for Paced {
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
        thread::sleep(Duration::from_millis(1));
        self.0.forward(s, b, kv, logits)
    }
}

/// Two conversations of three timed turns, a system message, scripted answers, then one single-turn line.
const CONVS: &str = r#"{"id":"a","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello there"},{"role":"assistant","content":"Hi.","max_tokens":4},{"role":"user","content":"What is two plus two?"},{"role":"assistant","content":"Four.","max_tokens":5},{"role":"user","content":"Thanks"},{"role":"assistant","max_tokens":3}]}
{"id":"b","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Name a colour"},{"role":"assistant","content":"Blue.","max_tokens":4},{"role":"user","content":"Another"},{"role":"assistant","content":"Red.","max_tokens":4},{"role":"user","content":"One more"},{"role":"assistant","max_tokens":2}]}
"#;

fn options(dir: &Temp, workload: Workload, prompts: &str) -> Options {
    Options {
        model: "tiny-tokenizer.gguf".into(),
        engine: EngineKind::Larch,
        url: None,
        cache_prompt: None,
        workload: Some(workload),
        mode: None,
        replay: Replay::Scripted,
        concurrency: vec![2],
        prompts: Some(dir.file("w.jsonl", prompts)),
        pp: 16,
        tg: 4,
        npl: vec![1, 2],
        duration: None,
        slice: None,
        cancel_rate: 0.0,
        natural_eos: false,
        telemetry: Telemetry::T1,
        set: vec!["max_context=512".into()],
        cpu: true,
        device: None,
        machine: "cpu".into(),
        json: Some(dir.0.join("out.json")),
    }
}

fn run(opts: &Options) -> (anyhow::Result<()>, Value, String) {
    run_on(opts, &vocab())
}

fn run_on(opts: &Options, vocab: &Gguf) -> (anyhow::Result<()>, Value, String) {
    let mut out = Vec::new();
    let r = bench::run_with(opts, vocab, Some(engine(opts)), &mut out);
    let json = fs::read_to_string(opts.json.as_ref().unwrap()).unwrap();
    (
        r,
        serde_json::from_str(&json).unwrap(),
        String::from_utf8(out).unwrap(),
    )
}

fn requests(v: &Value) -> &Vec<Value> {
    v["requests"].as_array().unwrap()
}

#[test]
fn sha256_matches_the_standard_vectors() {
    let hex = |b: [u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    assert_eq!(
        hex(bench::sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(bench::sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // Two blocks: the padding crosses the 56-byte boundary.
    assert_eq!(
        hex(bench::sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // Token IDs hash as consecutive little-endian int32, as the llama.cpp driver hashes them.
    assert_eq!(
        bench::prompt_sha256(&[1, 256]),
        hex(bench::sha256(&[1, 0, 0, 0, 0, 1, 0, 0]))
    );
}

#[test]
fn workloads_parse_as_the_driver_reads_them_and_bad_lines_are_refused() {
    let dir = Temp::new();
    let w = bench::load_workload(&dir.file("w.jsonl", CONVS)).unwrap();
    assert_eq!(w.name, "w.jsonl");
    assert_eq!(w.sha256, {
        let h = bench::sha256(CONVS.as_bytes());
        h.iter().map(|x| format!("{x:02x}")).collect::<String>()
    });
    assert_eq!(w.convs.len(), 2);
    let turns: Vec<_> = w.convs[0]
        .turns
        .iter()
        .map(|t| (t.n_msgs, t.max_tokens))
        .collect();
    assert_eq!(turns, [(2, 4), (4, 5), (6, 3)]);

    for bad in [
        // an assistant turn without max_tokens
        r#"{"id":"x","messages":[{"role":"user","content":"a"},{"role":"assistant"}]}"#,
        // a continued assistant message without its scripted content
        r#"{"id":"x","messages":[{"role":"user","content":"a"},{"role":"assistant","max_tokens":2},{"role":"user","content":"b"},{"role":"assistant","max_tokens":2}]}"#,
        // ends on a user message
        r#"{"id":"x","messages":[{"role":"user","content":"a"}]}"#,
        // unknown role
        r#"{"id":"x","messages":[{"role":"tool","content":"a"},{"role":"assistant","max_tokens":2}]}"#,
        // assistant first
        r#"{"id":"x","messages":[{"role":"assistant","max_tokens":2}]}"#,
        "",
    ] {
        assert!(
            bench::load_workload(&dir.file("bad.jsonl", bad)).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn the_frozen_workloads_parse() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    for (file, convs, turns) in [
        ("w1-chat.jsonl", 1, 12),
        ("w2-chats.jsonl", 8, 64),
        ("w3-agent.jsonl", 3, 60),
        ("w5-longprompt.jsonl", 2, 2),
        ("chat.jsonl", 200, 200),
    ] {
        let w = bench::load_workload(&root.join(file)).unwrap();
        assert_eq!(w.convs.len(), convs, "{file}");
        assert_eq!(
            w.convs.iter().map(|c| c.turns.len()).sum::<usize>(),
            turns,
            "{file}"
        );
    }
}

#[test]
fn summary_uses_nearest_rank_and_counts_failures_apart() {
    let req = |t_submit: f64, t_first: f64, t_done: f64, n: u32, error: Option<&str>| {
        bench::Req {
            gen_tokens: n,
            t_submit_ms: t_submit,
            t_first_ms: Some(t_first),
            t_done_ms: t_done,
            error: error.map(Into::into),
            ..bench::Req::default()
        }
        .finish()
    };
    let reqs = vec![
        req(0.0, 10.0, 40.0, 4, None),   // ttft 10, tpot 10
        req(0.0, 20.0, 50.0, 4, None),   // ttft 20, tpot 10
        req(10.0, 40.0, 100.0, 4, None), // ttft 30, tpot 20
        req(10.0, 50.0, 60.0, 0, Some("Overloaded")),
    ];
    let s = bench::summary(&reqs);
    assert_eq!(s["requests"], 4);
    assert_eq!(s["failures"], 1);
    assert_eq!(s["cancelled"], 0);
    assert_eq!(s["gen_tokens"], 12);
    assert_eq!(s["wall_s"], 0.1);
    assert!((s["agg_tok_s"].as_f64().unwrap() - 120.0).abs() < 1e-9);
    assert_eq!(s["ttft_ms"]["p50"], 20.0);
    assert_eq!(s["ttft_ms"]["p99"], 30.0);
    assert_eq!(s["ttft_ms"]["mean"], 20.0);
    assert_eq!(s["tpot_ms"]["p50"], 10.0);
    assert_eq!(s["tpot_ms"]["max"], 20.0);
    // A failed request carries no latencies, as in the driver.
    let failed = serde_json::to_value(&reqs[3]).unwrap();
    assert!(failed["ttft_ms"].is_null() && failed["evaluated_tokens"].is_null());
    assert!(bench::summary(&[])["ttft_ms"].is_null());
}

#[test]
fn scripted_replay_reuses_each_conversation_and_hashes_every_turn() {
    let dir = Temp::new();
    let opts = options(&dir, Workload::Chat, CONVS);
    let (r, v, table) = run(&opts);
    r.unwrap();
    // The driver's schema, key for key, so baseline.py check and compare read it.
    for k in [
        "schema",
        "engine",
        "driver",
        "mode",
        "llama_cpp",
        "system_info",
        "machine",
        "started_at",
        "model",
        "date_string",
        "params",
        "routes",
        "workload",
        "concurrency",
        "cache_prompt",
        "summary",
        "requests",
    ] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
    // The opt-in route switches this process runs with, so a row shows which route ran.
    let r = larch_kernels::routes();
    assert_eq!(
        v["routes"],
        serde_json::json!({"quant_i8": r.quant_i8, "prefill_tc": r.prefill_tc, "mixed_graphs": r.mixed_graphs})
    );
    assert_eq!(v["schema"], "larch-bench/1");
    assert_eq!(v["engine"], "larch");
    assert_eq!(v["mode"], "run");
    assert_eq!(v["replay"], "scripted");
    assert_eq!(v.get("duration"), Some(&Value::Null));
    assert_eq!(v["model"], "tiny-tokenizer.gguf");
    assert_eq!(v["date_string"], "26 Jul 2024");
    assert_eq!(v["concurrency"], 2);
    assert_eq!(v["cache_prompt"], true);
    assert_eq!(v["params"]["n_ctx_seq"], 512);
    assert_eq!(v["workload"]["conversations"], 2);
    assert!(
        table.contains("w.jsonl c=2: 6 requests, 0 failed"),
        "{table}"
    );

    // Every prompt is the scripted history rendered with the fixed date, so its hash is reproducible.
    let w = bench::load_workload(opts.prompts.as_ref().unwrap()).unwrap();
    let (tok, tmpl) = bench::vocab_of(&vocab()).unwrap();
    let reqs = requests(&v);
    assert_eq!(reqs.len(), 6);
    for r in reqs {
        let conv = w.convs.iter().find(|c| c.id == r["conv"]).unwrap();
        let turn = &conv.turns[r["turn"].as_u64().unwrap() as usize];
        let (_, prompt) = bench::render(&tok, &tmpl, &conv.messages[..turn.n_msgs]).unwrap();
        assert_eq!(r["prompt_sha256"], bench::prompt_sha256(&prompt));
        assert_eq!(r["prompt_tokens"], prompt.len());
        // ignore_eos: every answer is exactly its target length.
        assert_eq!(r["gen_tokens"], turn.max_tokens);
        assert_eq!(r["max_tokens"], turn.max_tokens);
        let gaps = r["token_gaps_ms"]
            .as_array()
            .expect("per-token receive gaps");
        assert_eq!(gaps.len(), turn.max_tokens.saturating_sub(1) as usize);
        assert!(gaps.iter().all(|gap| gap.as_f64().unwrap() >= 0.0));
        assert!(r["error"].is_null());
        let cached = r["cached_tokens"].as_u64().unwrap();
        assert_eq!(
            r["evaluated_tokens"].as_u64().unwrap(),
            prompt.len() as u64 - cached
        );
        // Turn 0 is fresh; later turns reuse the parked history up to the scripted answer.
        if r["turn"] == 0 {
            assert_eq!(cached, 0);
        } else {
            assert!(cached > 0 && cached < prompt.len() as u64, "{r}");
        }
        assert!(r["ttft_ms"].as_f64().unwrap() >= 0.0);
    }
    assert!(v["summary"]["token_gap_ms"]["p99"].as_f64().is_some());
    // Closed loop, one conversation per client: each client's turns run back to back.
    for c in [0, 1] {
        let mine: Vec<_> = reqs.iter().filter(|r| r["client"] == c).collect();
        assert_eq!(mine.len(), 3);
        assert!(mine.windows(2).all(|p| {
            p[0]["conv"] == p[1]["conv"]
                && p[0]["t_done_ms"].as_f64() <= p[1]["t_submit_ms"].as_f64()
        }));
    }
}

#[test]
fn batch_reuses_nothing_and_runs_every_line() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.workload = None;
    opts.mode = Some(Mode::Serve);
    let (r, v, _) = run(&opts);
    r.unwrap();
    assert_eq!(v["cache_prompt"], false);
    assert!(
        requests(&v)
            .iter()
            .all(|r| r["cached_tokens"] == 0 && r["error"].is_null())
    );
    assert_eq!(v["summary"]["requests"], 6);
}

/// A run under the 20-minute GPU hold cap: W4 in contiguous slices of the file, each its own result.
#[test]
fn slices_split_the_workload_in_file_order() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.workload = None;
    opts.mode = Some(Mode::Serve);
    let key = |v: &Value| -> Vec<(String, u64)> {
        let mut k: Vec<_> = requests(v)
            .iter()
            .map(|r| {
                (
                    r["conv"].as_str().unwrap().to_string(),
                    r["turn"].as_u64().unwrap(),
                )
            })
            .collect();
        k.sort();
        k
    };
    let (r, whole, _) = run(&opts);
    r.unwrap();
    assert!(whole.get("slice").is_none());
    let mut seen = Vec::new();
    for k in 0..2 {
        opts.slice = Some((k, 2));
        let (r, v, _) = run(&opts);
        r.unwrap();
        assert_eq!(
            v["slice"],
            serde_json::json!({"k": k, "n": 2, "first": k, "end": k + 1})
        );
        assert_eq!(v["workload"]["conversations"], 1);
        // The whole file's hash: a joined run is checked against the manifest like any other.
        assert_eq!(v["workload"]["sha256"], whole["workload"]["sha256"]);
        // The binary that ran it: bench-join joins only parts of one build.
        assert_eq!(v["build"], whole["build"]);
        assert_eq!(
            v["build"].as_str().map(str::len),
            Some(64),
            "{}",
            v["build"]
        );
        assert_eq!(key(&v)[0].0, ["a", "b"][k]);
        seen.extend(key(&v));
    }
    seen.sort();
    assert_eq!(seen, key(&whole));
}

#[test]
fn generated_replay_runs_clean_through_sessions() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.replay = Replay::Generated;
    let (r, v, _) = run_on(&opts, &roundtrip_vocab());
    r.unwrap();
    assert_eq!(v["replay"], "generated");
    let reqs = requests(&v);
    assert_eq!(reqs.len(), 6);
    for r in reqs {
        assert!(r["error"].is_null(), "{r}");
        // The engine's own answers continue the conversation: the KV it parked is exactly reused.
        assert_eq!(r["cached_tokens"], r["kv_tokens"], "{r}");
        if r["turn"] != 0 {
            assert!(r["cached_tokens"].as_u64().unwrap() > 0, "{r}");
        }
    }
}

#[test]
fn generated_replay_is_refused_for_ollama_and_tune_is_not_benched_here() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.engine = EngineKind::Ollama;
    opts.url = Some("http://127.0.0.1:1".into());
    opts.set.clear();
    opts.replay = Replay::Generated;
    assert!(bench::run_with(&opts, &vocab(), None, &mut Vec::new()).is_err());
}

#[test]
fn cancellation_ends_requests_mid_stream_and_the_engine_keeps_serving() {
    let dir = Temp::new();
    // 400-token answers at 1 ms a step: a cancel sent on the first token has 399 steps to land, so
    // the fake can no longer finish an answer before its client reads the first token.
    let long: String = (0..6)
        .map(|i| format!(r#"{{"id":"l{i}","messages":[{{"role":"user","content":"Go {i}"}},{{"role":"assistant","max_tokens":400}}]}}"#) + "\n")
        .collect();
    let mut opts = options(&dir, Workload::Batch, &long);
    opts.cancel_rate = 0.5;
    let mut e = bench::engine_options(&opts).unwrap();
    e.ring_capacity = 4096;
    let paced = start_with(
        e,
        Box::new(|_, _, _| Ok(Box::new(Paced(FakeModel::new(331))))),
    )
    .unwrap();
    let r = bench::run_with(&opts, &vocab(), Some(paced), &mut Vec::new());
    r.unwrap();
    let v = result(&opts);
    let reqs = requests(&v);
    let cancelled = reqs.iter().filter(|r| r["error"] == "Cancelled").count();
    assert_eq!(cancelled, 3);
    assert_eq!(v["summary"]["cancelled"], 3);
    assert_eq!(v["summary"]["failures"], 0);
    // Cancelled after the first token: a time to first token exists, the answer is short.
    for r in reqs.iter().filter(|r| r["error"] == "Cancelled") {
        assert!(r["t_first_ms"].is_number());
        assert!(r["gen_tokens"].as_u64() < r["max_tokens"].as_u64());
    }
    assert!(
        reqs.iter()
            .filter(|r| r["error"].is_null())
            .all(|r| r["gen_tokens"] == r["max_tokens"])
    );
}

#[test]
fn live_set_is_applied_on_time_and_lands_in_the_timeline() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.set.push("max_batch_seqs=2@0".into());
    opts.set.push("schedule_policy=balanced".into());
    let (r, v, _) = run(&opts);
    r.unwrap();
    let t = v["knob_timeline"].as_array().unwrap();
    let startup: Vec<_> = t.iter().filter(|e| e["at_s"].is_null()).collect();
    assert_eq!(startup.len(), 2, "{t:?}");
    let live = t.iter().find(|e| e["knob"] == "max_batch_seqs").unwrap();
    assert_eq!(live["at_s"], 0.0);
    assert_eq!(live["applied"], true);
    // The header is the run's start, after the changes due at 0 s.
    assert_eq!(v["params"]["n_seq_max"], 2.0);
    // Applied before any client submitted.
    let first = requests(&v)[0]["t_submit_ms"].as_f64().unwrap();
    assert!(live["t_ms"].as_f64().unwrap() <= first, "{live}");
    assert!(
        live["reply"].as_str().unwrap().starts_with("Applied"),
        "{live}"
    );
    // The engine's own record of the change, from the ring.
    let revs = v["telemetry"]["revisions"].as_array().unwrap();
    assert!(
        revs.iter().any(|r| r["knob"] == "max_batch_seqs"
            && r["effective"] == 2.0
            && r["cause"] == "Human"),
        "{revs:?}"
    );
    // In force before the first request, so the guardrail has no baseline to trip on.
    assert_eq!(v["telemetry"]["guardrail_trips"], 0);
    assert!(v["telemetry"]["engine_samples"].as_u64().unwrap() > 0);

    // A live change is never undone, so a second concurrency would start from it: refused.
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.concurrency = vec![1, 2];
    opts.set.push("max_batch_seqs=2@0".into());
    let refused = bench::run_with(&opts, &vocab(), Some(engine(&opts)), &mut Vec::new());
    assert!(
        refused
            .unwrap_err()
            .to_string()
            .contains("one --concurrency")
    );

    // A change the run never reaches is recorded as such and fails the invocation, after the result.
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.set.push("max_batch_seqs=2@3600".into());
    let (r, v, _) = run(&opts);
    let err = r.unwrap_err().to_string();
    assert!(err.contains("1 --set k=v@S not applied"), "{err}");
    let live = &v["knob_timeline"].as_array().unwrap()[1];
    assert_eq!(
        (
            &live["knob"],
            &live["at_s"],
            &live["reply"],
            &live["applied"]
        ),
        (
            &"max_batch_seqs".into(),
            &3600.0.into(),
            &"NotReached".into(),
            &false.into()
        )
    );
    assert_eq!(v["summary"]["failures"], 0);

    // So does a live change the engine refuses: the row never ran with it.
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.set.push("max_batch_seqs=0@0".into());
    let (r, v, _) = run(&opts);
    let err = r.unwrap_err().to_string();
    assert!(err.contains("1 --set k=v@S not applied"), "{err}");
    let live = &v["knob_timeline"].as_array().unwrap()[1];
    assert!(
        live["reply"].as_str().unwrap().starts_with("Rejected"),
        "{live}"
    );
    assert_eq!(live["applied"], false);

    // A startup value the engine rejects fails the run by name, as chat does.
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.set.push("max_batch_seqs=0".into());
    let err = bench::run_with(&opts, &vocab(), Some(engine(&opts)), &mut Vec::new());
    assert!(err.is_err());
}

#[test]
fn telemetry_off_writes_no_engine_samples() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.telemetry = Telemetry::Off;
    let (r, v, _) = run(&opts);
    r.unwrap();
    assert_eq!(v["params"]["telemetry"], "off");
    assert_eq!(v["telemetry"]["engine_samples"], 0);
    assert!(v["telemetry"]["achieved_gbps"].is_null());
    assert!(v["telemetry"]["step_profile"]["graph_captures"].is_null());
    assert!(v["telemetry"]["step_profile"]["capture_step_dur_ns"].is_null());
}

#[test]
fn several_concurrencies_write_one_result_each() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.concurrency = vec![1, 3];
    let mut out = Vec::new();
    bench::run_with(&opts, &vocab(), Some(engine(&opts)), &mut out).unwrap();
    for n in [1, 3] {
        let p = dir.0.join(format!("out-c{n}.json"));
        let v: Value = serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap();
        assert_eq!(v["concurrency"], n);
        assert_eq!(v["summary"]["requests"], 6);
    }
}

#[test]
fn every_concurrency_of_one_invocation_reuses_its_conversations() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chats, CONVS);
    opts.concurrency = vec![1, 2];
    bench::run_with(&opts, &vocab(), Some(engine(&opts)), &mut Vec::new()).unwrap();
    for n in [1, 2] {
        let p = dir.0.join(format!("out-c{n}.json"));
        let v: Value = serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap();
        // Released session ids never park again, so a later run must not reuse an earlier run's.
        assert!(
            requests(&v)
                .iter()
                .filter(|r| r["turn"] != 0)
                .all(|r| r["cached_tokens"].as_u64() > Some(0)),
            "c={n}: {v}"
        );
    }
}

#[test]
fn duration_wraps_the_workload_until_time_is_up() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.duration = Some(1.0);
    let (r, v, _) = run(&opts);
    r.unwrap();
    // Six requests take a few ms on the fake: a second of closed loop runs many more.
    assert!(v["summary"]["requests"].as_u64().unwrap() > 6);
    assert!(v["summary"]["wall_s"].as_f64().unwrap() >= 0.9);
    // A soak is not the workload's row: the scoreboard tells them apart by this.
    assert_eq!(v["duration"], 1.0);
}

#[test]
fn fixed_runs_the_drivers_loop_and_refuses_any_other_shape() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Fixed, CONVS);
    opts.workload = None;
    opts.mode = Some(Mode::Batched);
    // Each phase is one engine group (#64): the parked continuations decode together from the first
    // step, so run_with's decode_shape check passes and every pl writes its row, each prompt's
    // parked KV reused whole.
    let mut out = Vec::new();
    bench::run_with(&opts, &vocab(), Some(engine(&opts)), &mut out).unwrap();
    let rows: Vec<_> = result(&opts)["fixed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            // bench-merge reads speed (agg at pl=8) and speed_tg (single stream at pl=1).
            assert!(r["speed"].as_f64().unwrap() > 0.0 && r["speed_tg"].as_f64().unwrap() > 0.0);
            // Each row's decode steps that replayed a graph or ran eager on a refused capture
            // (bench-merge reads both); none on the CPU, which has no graphs.
            let counts = (&r["graph_replays"], &r["eager_decode_steps"]);
            assert_eq!(counts, (&json!(0), &json!(0)), "{r}");
            ["pl", "n_kv", "cached_tokens"].map(|k| r[k].as_u64().unwrap())
        })
        .collect();
    assert_eq!(rows, [[1, 20, 16], [2, 40, 32]]);
    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("fixed pp=16 tg=4 pl=2: T "), "{out}");
    fs::remove_file(opts.json.as_ref().unwrap()).unwrap();

    // Fewer seats than pl: the group cannot decode together, so the shape check refuses the row
    // and a refused run writes no result.
    let mut o = opts.clone();
    o.set.push("max_batch_seqs=1".into());
    o.npl = vec![2];
    let err = bench::run_with(&o, &vocab(), Some(engine(&o)), &mut Vec::new()).unwrap_err();
    let err = format!("{err:#}");
    assert!(
        err.contains("fixed pp=16 tg=4 pl=2: decode step 0 of 4 was Decode with 1 decode"),
        "{err}"
    );
    assert!(!opts.json.as_ref().unwrap().exists());

    // Under KV pressure later prompts reclaim earlier parked ones: refused before decode would
    // recompute them as prefill and count it as decode time.
    let mut o = opts.clone();
    o.set = vec!["max_context=512".into(), "max_batch_seqs=1".into()];
    (o.pp, o.tg, o.npl) = (256, 128, vec![4]);
    let err = bench::run_with(&o, &vocab(), Some(engine(&o)), &mut Vec::new()).unwrap_err();
    let err = format!("{err:#}");
    assert!(
        err.contains("pl=4: 0 of a prompt's 256 tokens parked after the prefill phase"),
        "{err}"
    );
    assert!(!opts.json.as_ref().unwrap().exists());

    // fixed() has no knob thread: a live change would be recorded nowhere and applied never.
    let mut o = opts.clone();
    o.set.push("max_batch_seqs=2@0".into());
    assert!(bench::run_with(&o, &vocab(), Some(engine(&o)), &mut Vec::new()).is_err());

    // Its step check needs every step's sample.
    for (telemetry, set) in [
        (Telemetry::T0, None),
        (Telemetry::T1, Some("telemetry_tier=t0")),
    ] {
        let mut o = opts.clone();
        o.telemetry = telemetry;
        o.set.extend(set.map(Into::into));
        let err = bench::run_with(&o, &vocab(), None, &mut Vec::new()).unwrap_err();
        assert!(err.to_string().contains("--telemetry t1"), "{err}");
    }

    // The driver's bound: pp + tg must stay under the per-sequence context; npl 0 is no shape.
    for (pp, npl) in [(508, vec![1]), (16, vec![0])] {
        let mut o = opts.clone();
        o.pp = pp;
        o.npl = npl;
        assert!(bench::run_with(&o, &vocab(), Some(engine(&o)), &mut Vec::new()).is_err());
    }
}

#[test]
fn fixed_warms_up_at_the_measured_prompt_length_clamped_to_the_context() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Fixed, CONVS);
    (opts.workload, opts.mode, opts.npl) = (None, Some(Mode::Batched), vec![1]);
    // max_context=512: a 600-token prompt is warmed at the 510 a request may hold (+2 sampled, ids
    // past the 331-token vocabulary wrapped); fixed() then refuses the shape, not the warm-up.
    for (pp, warmed) in [(256, 256), (600, 510)] {
        opts.pp = pp;
        let e = engine(&opts);
        let mut ring = e.ring().reader(ReadFrom::Oldest);
        let r = bench::run_with(&opts, &vocab(), Some(e), &mut Vec::new());
        match r {
            Ok(_) => assert_eq!(pp, 256),
            Err(err) => assert!(
                err.to_string()
                    .contains("pp=600 tg=4: context 512 is too small"),
                "{err}"
            ),
        }
        let mut records = Vec::new();
        ring.drain(&mut records, usize::MAX);
        let first = records.iter().find_map(|r| match r {
            Record::EngineSample(_, s) => Some(s.n_prefill_tokens),
            _ => None,
        });
        assert_eq!(first, Some(warmed), "pp={pp}");
    }
}

#[test]
fn decode_shape_takes_only_tg_steps_of_pl_decode_rows() {
    // Real t1 samples: a request on its own prefills in one step, then decodes one row a step.
    let dir = Temp::new();
    let e = engine(&options(&dir, Workload::Fixed, CONVS));
    let mut ring = e.ring().reader(ReadFrom::Latest);
    let h = e.submit(Request::new((0..16).collect(), 5)).unwrap();
    assert!(
        h.events
            .iter()
            .any(|ev| matches!(ev, EngineEvent::Done { .. }))
    );
    e.parked_tokens(0); // a round trip through the engine thread: the last step's sample is written
    let mut records = Vec::new();
    ring.drain(&mut records, usize::MAX);
    let s: Vec<EngineSample> = records
        .into_iter()
        .filter_map(|r| match r {
            Record::EngineSample(_, s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(s.len(), 5);
    bench::decode_shape(&s[1..], 1, 4).unwrap();
    for (steps, pl, tg, why) in [
        (&s[..4], 1, 4, "decode step 0 of 4 was Prefill"),
        (&s[1..], 2, 4, "decode step 0 of 4 was Decode with 1 decode"),
        (&s[1..4], 1, 4, "took 3 engine steps"),
        (&[s[1], s[3]][..], 1, 2, "skip steps"),
        (&[][..], 1, 4, "skip steps"),
    ] {
        let err = bench::decode_shape(steps, pl, tg).unwrap_err().to_string();
        assert!(err.contains(why), "{err}");
    }
}

#[test]
fn bad_arguments_fail_before_the_model_loads() {
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Batch, CONVS);
    opts.model = dir.0.join("absent.gguf");
    for d in [-1.0, 0.0, f64::NAN, f64::INFINITY, 1e300] {
        opts.duration = Some(d);
        let err = format!(
            "{:#}",
            bench::run(opts.clone(), &mut Vec::new()).unwrap_err()
        );
        assert!(err.contains("--duration"), "{d}: {err}");
    }
    opts.duration = None;
    // A slice of a workload without reuse only: a conversation's turns never straddle two runs.
    for (workload, slice, duration) in [
        (Workload::Batch, (2, 2), None),
        (Workload::Batch, (0, 0), None),
        // More parts than conversations: a part would run nothing.
        (Workload::Batch, (0, 3), None),
        (Workload::Batch, (0, 2), Some(60.0)),
        (Workload::Chat, (0, 2), None),
    ] {
        let mut o = opts.clone();
        (o.workload, o.slice, o.duration) = (Some(workload), Some(slice), duration);
        let err = format!("{:#}", bench::run(o, &mut Vec::new()).unwrap_err());
        assert!(err.contains("--slice"), "{slice:?}: {err}");
    }
    // A live change would restart in every part, so a sliced run takes startup values only.
    let mut o = opts.clone();
    (o.workload, o.slice, o.set) = (
        Some(Workload::Batch),
        Some((0, 2)),
        vec!["max_batch_seqs=2@1".into()],
    );
    let err = format!("{:#}", bench::run(o, &mut Vec::new()).unwrap_err());
    assert!(err.contains("--slice"), "{err}");
    for at in ["-1", "NaN", "inf", "1e300"] {
        opts.set = vec![format!("max_batch_seqs=2@{at}")];
        let err = format!(
            "{:#}",
            bench::run(opts.clone(), &mut Vec::new()).unwrap_err()
        );
        assert!(err.contains("--set max_batch_seqs=2@"), "{at}: {err}");
    }
    // The workload file too: a missing one, or one generated replay cannot continue.
    opts.set.clear();
    opts.prompts = Some(dir.0.join("absent.jsonl"));
    let err = format!(
        "{:#}",
        bench::run(opts.clone(), &mut Vec::new()).unwrap_err()
    );
    assert!(err.contains("absent.jsonl"), "{err}");
    opts.workload = Some(Workload::Chat);
    opts.replay = Replay::Generated;
    opts.prompts = Some(dir.file(
        "two-users.jsonl",
        r#"{"id":"x","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"},{"role":"assistant","max_tokens":2}]}"#,
    ));
    let err = format!(
        "{:#}",
        bench::run(opts.clone(), &mut Vec::new()).unwrap_err()
    );
    assert!(err.contains("user/assistant pairs"), "{err}");
}

/// What the fake Ollama reports for a prompt of n tokens (Ollama's `prompt_eval_count` counts
/// cached tokens too, and `prompt_eval_cached_count` may be absent).
#[derive(Clone, Copy)]
enum Counts {
    /// n, and 3 of them evaluated.
    Cached,
    /// n, no cached count.
    NoCachedCount,
    /// n + 1: Ollama tokenised something other than what the hash names.
    Mismatch,
    /// n, and 2^32 + 1 of them cached: more than any prompt, and 1 if narrowed by wrapping.
    TooManyCached,
    TooManyOutputs,
}

/// The next request to a fake server as (connection, request line, body). A request that never
/// comes fails the test after 30 s instead of hanging it.
fn next_request(l: &TcpListener) -> (TcpStream, String, Vec<u8>) {
    l.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let s = loop {
        match l.accept() {
            Ok((s, _)) => break s,
            Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            Err(e) => panic!("no request came: {e}"),
        }
    };
    s.set_nonblocking(false).unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    let (mut line, mut header, mut len) = (String::new(), String::new(), 0);
    r.read_line(&mut line).unwrap();
    loop {
        header.clear();
        r.read_line(&mut header).unwrap();
        if header == "\r\n" {
            break;
        }
        if let Some(v) = header.to_ascii_lowercase().strip_prefix("content-length: ") {
            len = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body).unwrap();
    (s, line, body)
}

/// A fake Ollama: answers `answers` /api/generate requests (the warm-up first) with streamed NDJSON
/// over HTTP/1.0, counting prompt tokens with the fixture's tokenizer; returns the bodies.
fn fake_ollama(answers: usize, counts: Counts) -> (String, thread::JoinHandle<Vec<Value>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let h = thread::spawn(move || {
        let (tok, _) = bench::vocab_of(&vocab()).unwrap();
        let mut bodies = Vec::new();
        for _ in 0..answers {
            let (mut w, line, body) = next_request(&l);
            assert_eq!(line, "POST /api/generate HTTP/1.0\r\n");
            let body: Value = serde_json::from_slice(&body).unwrap();
            let n = body["options"]["num_predict"].as_u64().unwrap();
            let prompt = tok.encode(body["prompt"].as_str().unwrap(), true).len();
            write!(
                w,
                "HTTP/1.0 200 OK\r\nContent-Type: application/x-ndjson\r\n\r\n"
            )
            .unwrap();
            for _ in 0..n {
                // The client may hang up after the first token: a cancelled request.
                if writeln!(w, r#"{{"response":"x","done":false}}"#).is_err() {
                    break;
                }
            }
            // The warm-up gets plain counts: the cases under test are the timed turns'.
            let generated = if !bodies.is_empty() && matches!(counts, Counts::TooManyOutputs) {
                n + 1
            } else {
                n
            };
            let counts = match if bodies.is_empty() {
                Counts::Cached
            } else {
                counts
            } {
                Counts::Cached => format!(
                    r#""prompt_eval_count":{prompt},"prompt_eval_cached_count":{}"#,
                    prompt - 3
                ),
                Counts::NoCachedCount => format!(r#""prompt_eval_count":{prompt}"#),
                Counts::Mismatch => format!(r#""prompt_eval_count":{}"#, prompt + 1),
                Counts::TooManyOutputs => format!(r#""prompt_eval_count":{prompt}"#),
                Counts::TooManyCached => format!(
                    r#""prompt_eval_count":{prompt},"prompt_eval_cached_count":{}"#,
                    (1u64 << 32) + 1
                ),
            };
            let _ = writeln!(
                w,
                r#"{{"response":"","done":true,{counts},"eval_count":{generated},"total_duration":1000000,"conversation":{},"turn":{}}}"#,
                body["larch_session"]["conversation"], body["larch_session"]["turn"]
            );
            bodies.push(body);
        }
        bodies
    });
    (url, h)
}

fn result(opts: &Options) -> Value {
    serde_json::from_str(&fs::read_to_string(opts.json.as_ref().unwrap()).unwrap()).unwrap()
}

fn ollama(dir: &Temp, w: Workload) -> Options {
    let mut opts = options(dir, w, CONVS);
    opts.engine = EngineKind::Ollama;
    opts.set.clear();
    opts.concurrency = vec![1];
    opts
}

#[test]
fn ollama_streams_raw_prompts_and_reports_its_http_overhead() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Chat);
    let (url, server) = fake_ollama(4, Counts::Cached);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    let v = result(&opts);
    let bodies = server.join().unwrap();
    assert_eq!(v["engine"], "ollama");
    assert_eq!(v["params"]["n_ctx_seq"], 8192);
    assert_eq!(v["routes"], Value::Null);
    // Loaded and warmed before the clock, at the run's context, as larch and the driver are.
    assert_eq!(bodies[0]["model"], "larch-tiny-tokenizer-8192");
    assert_eq!(bodies[0]["options"]["num_ctx"], 8192);
    assert_eq!(bodies[0]["keep_alive"], -1);
    let w = bench::load_workload(opts.prompts.as_ref().unwrap()).unwrap();
    let (tok, tmpl) = bench::vocab_of(&vocab()).unwrap();
    assert_eq!(requests(&v).len(), 3);
    for (r, body) in requests(&v).iter().zip(&bodies[1..]) {
        let conv = &w.convs[0];
        let turn = &conv.turns[r["turn"].as_u64().unwrap() as usize];
        // The same rendered prompt as larch and llama.cpp, sent raw, so the hash check applies.
        let (text, prompt) = bench::render(&tok, &tmpl, &conv.messages[..turn.n_msgs]).unwrap();
        assert_eq!(body["prompt"], text);
        assert_eq!(body["raw"], true);
        assert_eq!(body["model"], "larch-tiny-tokenizer-8192");
        // Pure greedy, as larch and llama.cpp: no default repeat penalty moves the argmax.
        assert_eq!(body["options"]["temperature"], 0.0);
        assert_eq!(body["options"]["repeat_penalty"], 1.0);
        assert_eq!(body["options"]["num_ctx"], 8192);
        assert_eq!(r["prompt_sha256"], bench::prompt_sha256(&prompt));
        assert_eq!(r["gen_tokens"], turn.max_tokens);
        // Its own cached count; the rest it evaluated.
        assert_eq!(r["cached_tokens"], prompt.len() as u64 - 3);
        assert_eq!(r["evaluated_tokens"], 3);
        // Its own column: client e2e minus the server's total_duration (1 ms).
        let overhead = r["http_overhead_ms"].as_f64().unwrap();
        assert!((overhead - (r["e2e_ms"].as_f64().unwrap() - 1.0)).abs() < 1e-9);
    }
    assert!(v["summary"]["http_overhead_ms"]["p50"].is_number());
}

#[test]
fn ollama_declares_its_prompt_cache_and_never_infers_cache_counts() {
    let dir = Temp::new();
    // W4: llama.cpp reuses nothing, Ollama always caches prompts, so compare must refuse the pair.
    let mut opts = ollama(&dir, Workload::Batch);
    let (url, server) = fake_ollama(7, Counts::NoCachedCount);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["cache_prompt"], true);
    // No cached count from Ollama: unavailable, not prompt - evaluated.
    assert!(
        requests(&v)
            .iter()
            .all(|r| r["cached_tokens"].is_null() && r["evaluated_tokens"].is_null()),
        "{v}"
    );
    assert!(v["summary"]["cached_tokens"].is_null() && v["summary"]["evaluated_tokens"].is_null());
    assert_eq!(v["summary"]["failures"], 0);
}

#[test]
fn ollama_prompt_counts_must_match_the_hashed_tokens() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    let (url, server) = fake_ollama(7, Counts::Mismatch);
    opts.url = Some(url);
    assert!(bench::run_with(&opts, &vocab(), None, &mut Vec::new()).is_err());
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["summary"]["failures"], 6);
    let err = requests(&v)[0]["error"].as_str().unwrap();
    assert!(err.contains("tokenised"), "{err}");
}

#[test]
fn ollama_cached_counts_above_the_prompt_fail_the_turn() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    let (url, server) = fake_ollama(7, Counts::TooManyCached);
    opts.url = Some(url);
    assert!(bench::run_with(&opts, &vocab(), None, &mut Vec::new()).is_err());
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["summary"]["failures"], 6);
    assert!(v["summary"]["cached_tokens"].is_null() || v["summary"]["cached_tokens"] == 0);
    let err = requests(&v)[0]["error"].as_str().unwrap();
    assert!(err.contains("cached of"), "{err}");
}

#[test]
fn ollama_takes_the_rows_context_and_no_other_knob() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    opts.set = vec!["max_context=2560".into()];
    let (url, server) = fake_ollama(7, Counts::Cached);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    assert!(
        server
            .join()
            .unwrap()
            .iter()
            .all(|b| b["options"]["num_ctx"] == 2560)
    );
    let v = result(&opts);
    assert_eq!(v["params"]["n_ctx_seq"], 2560);
    // The Modelfile model baseline.py creates (8192 or 16384); num_ctx sets the row's context.
    assert_eq!(v["params"]["ollama_model"], "larch-tiny-tokenizer-8192");

    opts.set.push("max_batch_seqs=2".into());
    assert!(bench::run_with(&opts, &vocab(), None, &mut Vec::new()).is_err());
}

#[test]
fn ollama_cancel_hangs_up_after_first_chunk_without_inventing_token_counts() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    opts.cancel_rate = 1.0;
    let (url, server) = fake_ollama(7, Counts::Cached);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["summary"]["cancelled"], 6);
    assert!(
        requests(&v)
            .iter()
            .all(|r| r["error"] == "Cancelled" && r["gen_tokens"] == 0 && r["http_chunks"] == 1)
    );
}

#[test]
fn an_ollama_that_does_not_answer_fails_at_warm_up_before_the_clock() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    // A port of our own that hangs up: a freed port could be taken by another test's fake.
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    opts.url = Some(format!("http://{}", l.local_addr().unwrap()));
    let server = thread::spawn(move || drop(l.accept().unwrap()));
    let err = bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap_err();
    server.join().unwrap();
    assert!(format!("{err:#}").contains("warm-up"), "{err:#}");
    assert!(!opts.json.as_ref().unwrap().exists());
}

#[test]
fn ollama_takes_one_concurrency_per_server() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Chats);
    opts.concurrency = vec![1, 4];
    // Refused before any request: no server is listening here.
    opts.url = Some("http://127.0.0.1:1".into());
    let err = bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap_err();
    assert!(err.to_string().contains("one --concurrency"), "{err}");
}

/// How the fake OpenAI-compatible server answers a timed completion (the warm-up's is `Clean`).
#[derive(Clone, Copy, PartialEq)]
enum Sse {
    /// The answer and its counts; the first event's text is empty and the rest come 100 ms later.
    Clean,
    /// `usage.prompt_tokens` one more than the prompt sent.
    PromptOff,
    /// One token short of `max_tokens`, as if EOS ended it.
    Short,
    /// 3 cached prompt tokens.
    Cached,
    /// An error event where the usage would be.
    ErrorEvent,
    /// llama-server's `error:` line where the usage would be.
    ErrorLine,
    /// `[DONE]` with no usage before it.
    NoUsage,
    /// The usage, then the end of the body without `[DONE]`.
    NoDone,
    /// `prompt_tokens_details` null, as vLLM sends by default.
    NoCachedCount,
    /// Only the usage event and `[DONE]`: no `choices[0]` event at all.
    NoChoices,
}

/// A fake OpenAI-compatible server: `/v1/models`, `/version` and `/metrics` saying
/// `enable_prefix_caching` (404 when `metrics` is None, as any other GET), and `answers` streamed
/// `/v1/completions` (the warm-up first), alternately close-delimited and chunked in 8-byte chunks,
/// which split the `data:` lines; returns the completion bodies.
fn fake_openai(
    answers: usize,
    sse: Sse,
    metrics: Option<bool>,
) -> (String, thread::JoinHandle<Vec<Value>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let h = thread::spawn(move || {
        let mut bodies = Vec::new();
        while bodies.len() < answers {
            let (mut s, line, body) = next_request(&l);
            if line != "POST /v1/completions HTTP/1.1\r\n" {
                let got = match line.as_str() {
                    "GET /v1/models HTTP/1.1\r\n" => Some(
                        json!({"object": "list", "data": [{"id": "fake", "object": "model",
                            "root": "/w", "max_model_len": 8192, "owned_by": "fake"}]})
                        .to_string(),
                    ),
                    "GET /version HTTP/1.1\r\n" => Some(r#"{"version":"9.9"}"#.into()),
                    "GET /metrics HTTP/1.1\r\n" => metrics.map(|m| {
                        format!(
                            "# TYPE vllm:cache_config_info gauge\nvllm:cache_config_info{{block_size=\"16\",enable_prefix_caching=\"{}\",gpu_memory_utilization=\"0.9\"}} 1.0\n",
                            if m { "True" } else { "False" }
                        )
                    }),
                    _ => None,
                };
                let (status, got) = got.map_or(("404 Not Found", String::new()), |g| ("200 OK", g));
                write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{got}",
                    got.len()
                )
                .unwrap();
                continue;
            }
            let body: Value = serde_json::from_slice(&body).unwrap();
            let (p, n) = (
                body["prompt"].as_array().unwrap().len(),
                body["max_tokens"].as_u64().unwrap(),
            );
            // The warm-up is answered clean: the cases under test are the timed turns'.
            let sse = if bodies.is_empty() { Sse::Clean } else { sse };
            let data = |v: Value| format!("data: {v}\n\n");
            let mut first =
                data(json!({"choices": [{"index": 0, "text": "", "finish_reason": null}]}));
            let mut rest: String = (1..=n)
                .map(|i| {
                    let finish = (i == n).then_some("length");
                    data(json!({"choices": [{"index": 0, "text": "x", "finish_reason": finish}]}))
                })
                .collect();
            if sse == Sse::NoChoices {
                (first, rest) = (String::new(), String::new());
            }
            let mut usage = json!({"prompt_tokens": p, "completion_tokens": n,
                "prompt_tokens_details": {"cached_tokens": 0}});
            match sse {
                Sse::PromptOff => usage["prompt_tokens"] = json!(p + 1),
                Sse::Short => usage["completion_tokens"] = json!(n - 1),
                Sse::Cached => usage["prompt_tokens_details"]["cached_tokens"] = json!(3),
                Sse::NoCachedCount => usage["prompt_tokens_details"] = Value::Null,
                _ => {}
            }
            match sse {
                Sse::ErrorEvent => {
                    rest += &data(json!({"error": {"message": "boom", "code": 500}}))
                }
                Sse::ErrorLine => rest += "error: {\"message\":\"slot gone\"}\n\n",
                Sse::NoUsage => {}
                _ => rest += &data(json!({"choices": [], "usage": usage})),
            }
            if sse != Sse::NoDone {
                rest += "data: [DONE]\n\n";
            }
            let chunked = bodies.len() % 2 == 1;
            let te = if chunked {
                "Transfer-Encoding: chunked\r\n"
            } else {
                ""
            };
            write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{te}Connection: close\r\n\r\n"
            )
            .unwrap();
            // Each chunk's CRLF goes out with the next size line, so the first event's last bytes
            // wait the 100 ms for theirs: a decoder that waited too would see the first event late.
            let mut crlf = "";
            for (k, piece) in [first, rest].iter().enumerate() {
                if k == 1 && sse == Sse::Clean {
                    // 40 ms over the 60 asserted: on a loaded machine the client wakes for the
                    // first event a few ms after it is written.
                    thread::sleep(Duration::from_millis(100));
                }
                let mut out = Vec::new();
                for c in piece.as_bytes().chunks(8) {
                    if chunked {
                        let ext = if k == 1 { ";x=y" } else { "" };
                        write!(out, "{crlf}{:x}{ext}\r\n", c.len()).unwrap();
                        crlf = "\r\n";
                    }
                    out.extend_from_slice(c);
                }
                // The client hangs up on a failed turn.
                let _ = s.write_all(&out);
            }
            if chunked {
                let _ = write!(s, "{crlf}0\r\n\r\n");
            }
            bodies.push(body);
        }
        bodies
    });
    (url, h)
}

#[test]
fn openai_streams_token_ids_and_refuses_what_the_hash_does_not_name() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    opts.engine = EngineKind::OpenAi;
    let run = |o: &Options| {
        bench::run_with(o, &vocab(), None, &mut Vec::new()).map_err(|e| format!("{e:#}"))
    };
    // A server's knobs are its startup flags; it serves any N, so several reach it.
    opts.url = Some("http://127.0.0.1:1".into());
    opts.set = vec!["max_context=512".into()];
    let err = run(&opts).unwrap_err();
    assert!(err.contains("startup flags"), "{err}");
    opts.set.clear();
    opts.concurrency = vec![1, 2];
    let err = run(&opts).unwrap_err();
    assert!(err.contains("connecting to 127.0.0.1:1"), "{err}");
    opts.concurrency = vec![1];

    let (url, server) = fake_openai(7, Sse::Clean, Some(false));
    opts.url = Some(url);
    // The operator's word never overrides the server's, and is refused before any completion.
    opts.cache_prompt = Some(true);
    let err = run(&opts).unwrap_err();
    assert!(err.contains("--cache-prompt"), "{err}");
    assert!(!opts.json.as_ref().unwrap().exists());
    opts.cache_prompt = None;
    run(&opts).unwrap();
    let bodies = server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["engine"], "openai");
    assert_eq!(
        v["server"],
        json!({"model": "fake", "root": "/w", "max_model_len": 8192, "owned_by": "fake", "version": "9.9"})
    );
    assert_eq!(v["params"]["n_ctx_seq"], 8192);
    assert_eq!(v["cache_prompt"], false);
    assert_eq!(v["ignore_eos"], true);
    assert_eq!(v["routes"], Value::Null);
    let (tok, tmpl) = bench::vocab_of(&vocab()).unwrap();
    assert_eq!(bodies[0]["prompt"], json!(tok.encode("warm up", true)));
    assert_eq!(v["warmup"]["output_tokens"], 2);
    let w = bench::load_workload(opts.prompts.as_ref().unwrap()).unwrap();
    let turns = w
        .convs
        .iter()
        .flat_map(|c| c.turns.iter().map(move |t| (c, t)));
    assert_eq!(requests(&v).len(), 6);
    for ((r, body), (conv, turn)) in requests(&v).iter().zip(&bodies[1..]).zip(turns) {
        let (_, prompt) = bench::render(&tok, &tmpl, &conv.messages[..turn.n_msgs]).unwrap();
        // The hashed ids themselves: no server template or tokenizer touches the prompt.
        assert_eq!(body["prompt"], json!(prompt));
        assert_eq!(body["model"], "fake");
        assert_eq!(body["max_tokens"], turn.max_tokens);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(r["prompt_sha256"], bench::prompt_sha256(&prompt));
        assert_eq!(r["gen_tokens"], turn.max_tokens);
        assert_eq!(r["cached_tokens"], 0);
        assert_eq!(r["http_chunks"], turn.max_tokens);
        // t_first is the first event, empty text and all, 100 ms before the rest.
        let (e2e, ttft) = (
            r["e2e_ms"].as_f64().unwrap(),
            r["ttft_ms"].as_f64().unwrap(),
        );
        assert!(e2e - ttft >= 60.0, "{r}");
    }

    for (sse, why) in [
        (Sse::PromptOff, "prompt tokens"),
        (Sse::Short, "ignore_eos"),
        (Sse::Cached, "cache_prompt false"),
        (Sse::ErrorEvent, "boom"),
        (Sse::ErrorLine, "slot gone"),
        (Sse::NoChoices, "no choices[0] event"),
        (Sse::NoUsage, "without usage"),
        (Sse::NoDone, "[DONE]"),
    ] {
        let (url, server) = fake_openai(7, sse, Some(false));
        opts.url = Some(url);
        assert!(run(&opts).is_err(), "{why}");
        server.join().unwrap();
        let v = result(&opts);
        assert_eq!(v["summary"]["failures"], 6, "{why}");
        assert!(
            requests(&v)
                .iter()
                .all(|r| r["error"].as_str().unwrap().contains(why)),
            "{v}"
        );
    }

    // A server that caches prefixes keeps them from one run to the next: one run per start.
    let (url, server) = fake_openai(7, Sse::Cached, Some(true));
    opts.url = Some(url);
    opts.concurrency = vec![1, 2];
    let err = run(&opts).unwrap_err();
    assert!(err.contains("prefix cache"), "{err}");
    opts.concurrency = vec![1];
    run(&opts).unwrap();
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["cache_prompt"], true);
    assert!(requests(&v).iter().all(|r| r["cached_tokens"] == 3), "{v}");

    // Where /metrics does not say, nothing runs until the operator states the cache.
    let (url, server) = fake_openai(7, Sse::NoCachedCount, None);
    opts.url = Some(url);
    let err = run(&opts).unwrap_err();
    assert!(err.contains("--cache-prompt"), "{err}");
    opts.cache_prompt = Some(true);
    run(&opts).unwrap();
    server.join().unwrap();
    let v = result(&opts);
    assert_eq!(v["cache_prompt"], true);
    // No cached count from the server: unknown, never inferred.
    assert!(
        requests(&v).iter().all(|r| r["cached_tokens"].is_null()),
        "{v}"
    );
}

#[test]
fn serving_profile_reports_actual_sequences_and_measured_step_time() {
    let dir = Temp::new();
    let opts = options(&dir, Workload::Batch, CONVS);
    let (result, value, _) = run(&opts);
    result.unwrap();
    let profile = &value["telemetry"]["step_profile"];
    let shapes = profile["shapes"].as_array().expect("step shape profile");
    assert!(!shapes.is_empty());
    assert!(shapes.iter().all(|s| s["gpu_ns"].as_u64().is_some()));
    assert!(
        shapes
            .iter()
            .any(|s| s["prefill_tokens"].as_u64().unwrap() > 2)
    );
    let active = profile["active_sequences"].as_array().unwrap();
    assert!(
        active
            .iter()
            .all(|s| (1..=2).contains(&s["sequences"].as_u64().unwrap()))
    );
    assert!(active.iter().all(|s| s["dur_ns"].as_u64().unwrap() > 0));
    assert!(profile["peak_kv_tokens_used"].as_u64().unwrap() > 0);
    assert_eq!(profile["steps"], value["telemetry"]["engine_samples"]);
    assert!(profile["gpu_ns"].as_u64().unwrap() > 0);
    assert_eq!(profile["graph_replays"], 0);
    assert_eq!(profile["graph_captures"], 0);
    assert_eq!(profile["capture_step_dur_ns"], 0);
    assert_eq!(profile["capture_counts_complete"], true);
    assert!(shapes.iter().all(|s| s["graph_captures"] == 0));
}

#[test]
fn decimated_telemetry_does_not_claim_complete_capture_counts() {
    let dir = Temp::new();
    let body = r#"{"id":"long","messages":[{"role":"user","content":"Go"},{"role":"assistant","max_tokens":200}]}"#;
    let mut opts = options(&dir, Workload::Batch, body);
    opts.concurrency = vec![1];
    opts.telemetry = Telemetry::T0;
    let (result, value, _) = run(&opts);
    result.unwrap();
    assert!(value["telemetry"]["engine_samples"].as_u64().unwrap() > 0);
    let profile = &value["telemetry"]["step_profile"];
    assert_eq!(profile["capture_counts_complete"], false);
    assert!(profile["graph_captures"].is_null());
    assert!(profile["capture_step_dur_ns"].is_null());
}

#[test]
fn natural_eos_counts_actual_short_completion_without_changing_historical_mode() {
    let dir = Temp::new();
    let body = r#"{"id":"eos","messages":[{"role":"user","content":"Go"},{"role":"assistant","max_tokens":400}]}"#;
    let mut opts = options(&dir, Workload::Batch, body);
    opts.concurrency = vec![1];
    let (res, historical, _) = run(&opts);
    res.unwrap();
    assert_eq!(historical["summary"]["gen_tokens"], 400);
    opts.natural_eos = true;
    let (res, natural, _) = run(&opts);
    res.unwrap();
    let generated = natural["summary"]["gen_tokens"].as_u64().unwrap();
    assert!(generated > 0 && generated < 400, "{natural}");
    assert_eq!(natural["ignore_eos"], false);
    assert_eq!(natural["requests"][0]["finish_reason"], "Eos");
    assert_eq!(natural["summary"]["failures"], 0);
}

#[test]
fn larch_http_uses_common_stream_driver_with_explicit_boundary_and_cache_policy() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    opts.engine = EngineKind::LarchHttp;
    opts.natural_eos = true;
    let (url, server) = fake_ollama(7, Counts::NoCachedCount);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    let v = result(&opts);
    assert_eq!(v["engine"], "larch-http");
    assert_eq!(v["cache_prompt"], true);
    assert_eq!(v["stream_boundary"], "http-text-chunks");
    assert!(v["summary"]["http_chunk_gap_ms"].is_object());
    assert!(v["summary"]["token_gap_ms"].is_null());
    let bodies = server.join().unwrap();
    assert_eq!(bodies.len(), 7);
    assert!(bodies[0]["larch_session"].is_null());
    assert!(bodies[1..].iter().all(|b| b["larch_session"].is_object()));
    assert!(
        v["cache_policy"]
            .as_str()
            .unwrap()
            .contains("conversation identity")
    );
}

#[test]
fn larch_http_keeps_original_scripted_prompts_and_conversation_turns() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Chat);
    opts.engine = EngineKind::LarchHttp;
    opts.natural_eos = true;
    let (url, server) = fake_ollama(4, Counts::Cached);
    opts.url = Some(url);
    bench::run_with(&opts, &vocab(), None, &mut Vec::new()).unwrap();
    let bodies = server.join().unwrap();
    let workload = bench::load_workload(opts.prompts.as_ref().unwrap()).unwrap();
    let conv = &workload.convs[0];
    let (tok, template) = bench::vocab_of(&vocab()).unwrap();
    for (k, body) in bodies[1..].iter().enumerate() {
        let turn = &conv.turns[k];
        let (text, _) = bench::render(&tok, &template, &conv.messages[..turn.n_msgs]).unwrap();
        assert_eq!(body["prompt"], text);
        assert_eq!(body["options"]["num_predict"], turn.max_tokens);
        assert_eq!(body["larch_session"]["conversation"], conv.id);
        assert_eq!(body["larch_session"]["turn"], k);
        assert_eq!(body["larch_session"]["last"], k + 1 == conv.turns.len());
    }
}

#[test]
fn http_terminal_output_count_above_cap_fails_without_completed_tokens() {
    let dir = Temp::new();
    let mut opts = ollama(&dir, Workload::Batch);
    let (url, server) = fake_ollama(7, Counts::TooManyOutputs);
    opts.url = Some(url);
    assert!(bench::run_with(&opts, &vocab(), None, &mut Vec::new()).is_err());
    assert_eq!(server.join().unwrap().len(), 7);
    let v = result(&opts);
    assert_eq!(v["summary"]["failures"], 6);
    assert_eq!(v["summary"]["gen_tokens"], 0);
    assert!(requests(&v).iter().all(|r| {
        r["error"]
            .as_str()
            .unwrap()
            .contains("exceeds requested cap")
    }));
}

/// The tiny hybrid (Qwen 3.5 shape) GGUF with a byte vocabulary and `template`, at `dir`.
fn tiny_hybrid_file(dir: &Temp, template: &str) -> PathBuf {
    let bytes = tiny::gguf(Arch::Qwen35, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = (g.tensors.iter())
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv: Vec<_> = (g.kv.into_iter())
        .filter(|(k, _)| !k.starts_with("tokenizer."))
        .collect();
    let byte = |b: u32| Kv::Str(format!("<0x{b:02X}>"));
    kv.extend([
        ("tokenizer.ggml.model".into(), Kv::Str("llama".into())),
        (
            "tokenizer.ggml.tokens".into(),
            Kv::Arr((0..256).map(byte).collect()),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            Kv::Arr(vec![Kv::I32(6); 256]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), Kv::U32(97)),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(255)),
        ("tokenizer.ggml.add_bos_token".into(), Kv::Bool(false)),
        ("tokenizer.ggml.add_space_prefix".into(), Kv::Bool(false)),
        ("tokenizer.chat_template".into(), Kv::Str(template.into())),
    ]);
    let path = dir.0.join("tiny-hybrid.gguf");
    fs::write(&path, write(&kv, &tensors)).unwrap();
    path
}

/// Scripted replay on a hybrid model reuses each turn's whole history: the turn parks its state
/// at its history end, as the app's sessions do, so the next prompt, which carries the scripted
/// answer, not larch's, runs on from there instead of reading the last turn again.
#[test]
fn scripted_replay_on_a_hybrid_reuses_the_whole_history() {
    // A past reply renders unlike its generation prompt, as Qwen 3.5's template does.
    let template = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}>{% endif %}";
    let dir = Temp::new();
    let mut opts = options(&dir, Workload::Chat, CONVS);
    opts.model = tiny_hybrid_file(&dir, template);
    let g = Gguf::parse(&fs::read(&opts.model).unwrap()).unwrap();
    bench::run(opts.clone(), &mut Vec::new()).unwrap();
    let v: Value =
        serde_json::from_str(&fs::read_to_string(opts.json.as_ref().unwrap()).unwrap()).unwrap();
    let w = bench::load_workload(opts.prompts.as_ref().unwrap()).unwrap();
    let (tok, tmpl) = bench::vocab_of(&g).unwrap();
    let reqs = requests(&v);
    assert_eq!(reqs.len(), 6);
    for r in reqs {
        let conv = w.convs.iter().find(|c| c.id == r["conv"]).unwrap();
        let k = r["turn"].as_u64().unwrap() as usize;
        let cached = r["cached_tokens"].as_u64().unwrap();
        if k == 0 {
            assert_eq!(cached, 0, "{r}");
            continue;
        }
        let last = &conv.messages[..conv.turns[k - 1].n_msgs];
        let (_, prompt) = bench::render(&tok, &tmpl, last).unwrap();
        let end = bench::history_end(&tok, &tmpl, last, &prompt).unwrap();
        assert!(end > 0 && (end as usize) < prompt.len(), "{end}");
        assert_eq!(cached, u64::from(end), "{r}");
    }
}
