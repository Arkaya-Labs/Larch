//! `--demo`: the design's sample data, so the window can be seen and reviewed with no engine.

use larch_app::contract::*;
use serde_json::json;

use crate::logic::{self, GIB, Task};
use crate::ui::{Ask, Dl, Ecl, Item, Stats, Step, StepState, Thread, now_ms};

const GB: f64 = GIB;

fn model(name: &str, quant: &str, gb: f64, lib: &str, fit: Fit, supported: bool) -> ModelEntry {
    let file = format!("{lib}/{}-{quant}.gguf", name.replace(' ', "-"));
    ModelEntry {
        path: file,
        name: name.into(),
        arch: "qwen3".into(),
        quant: quant.into(),
        size_bytes: (gb * GB) as u64,
        supported,
        fit,
        source: "Larch".into(),
        library: lib.into(),
        vision: false,
        faster: None,
    }
}

fn step(now: &str, st: StepState) -> Step {
    Step {
        name: "read_papers".into(),
        args: json!({}),
        now: now.into(),
        done: now.into(),
        sum: String::new(),
        st,
        count: 1,
    }
}

/// Fills `s` with the mockup's chats, models, engine numbers and memory.
pub fn seed(s: &mut Ecl) {
    tasks(s);
    let now = now_ms();
    let min = 60_000;
    let gpu = (16. * GB) as u64;
    s.prefs.name = "Alex".into();
    s.prefs.advanced = true;
    s.st.health = Health::Ready;
    s.st.device = "RTX 5070 Ti".into();
    s.st.gpu_total_bytes = Some(gpu);
    s.st.data_dir = "~/.local/share/larch".into();
    s.st.libraries = vec!["/home/you/models".into(), "/mnt/d/gguf".into()];
    s.st.code_mode = true;
    let fits = |max_seqs| Fit::Fits {
        kv_bytes: 0,
        max_seqs,
    };
    s.models = vec![
        model(
            "Qwen3.5 35B A3B",
            "Q3_K_M",
            15.9,
            "/home/you/models",
            fits(3),
            true,
        ),
        model(
            "Llama 3.1 8B Instruct",
            "Q4_K_M",
            4.9,
            "/home/you/models",
            fits(16),
            true,
        ),
        model(
            "Qwen2.5 Coder 14B",
            "Q4_K_M",
            9.0,
            "/home/you/models",
            fits(6),
            true,
        ),
        model(
            "Qwen2.5 7B Instruct",
            "Q8_0",
            8.1,
            "/mnt/d/gguf",
            fits(8),
            true,
        ),
        model(
            "Gemma 3 27B",
            "Q4_0",
            15.6,
            "/mnt/d/gguf",
            Fit::Unknown,
            false,
        ),
    ];
    s.st.model = Some(s.models[0].path.clone());
    // The 35B card's faster version: offered, being made (`LARCH_DEMO_FASTER=making`) or in use
    // (`on`), to see each state without making one.
    let big = s.models[0].path.clone();
    let copy = big.replace(".gguf", ".faster.gguf");
    let state = std::env::var("LARCH_DEMO_FASTER").unwrap_or_default();
    s.models[0].faster = Some(Faster {
        copy: (state == "on").then(|| copy.clone()),
        building: state == "making",
    });
    match state.as_str() {
        "making" => {
            s.fasters.insert(big, (0.62, 150.));
        }
        "on" => s.st.model = Some(copy),
        _ => {}
    }
    for (m, t) in s.models.iter().zip([47., 168., 96., 121.]) {
        s.prefs.tok_s.insert(m.path.clone(), t);
    }
    s.dls.insert(
        "d1".into(),
        Dl {
            repo: "Qwen/Qwen3-30B-A3B-Instruct-GGUF".into(),
            file: "Qwen3-30B-A3B-Instruct-Q4_K_M.gguf".into(),
            got: (11.4 * GB) as u64,
            total: (18.4 * GB) as u64,
            bps: 42e6,
        },
    );
    let chats = [
        ("moe", "Compare the four MoE papers", 2, 1),
        ("cancel", "Fix the flaky cancel test", 5, 40),
        ("invite", "Draft the beta invite email", 3, 60 * 26),
        ("cuda", "What changed in CUDA 13.4", 2, 60 * 72),
    ];
    s.chats = chats
        .iter()
        .map(|(id, t, turns, ago)| ChatEntry {
            id: (*id).into(),
            title: (*t).into(),
            updated_at: now - ago * min,
            turns: *turns,
        })
        .collect();
    let papers: Vec<String> = [
        "switch-routing.pdf",
        "expert-choice.pdf",
        "dropless.pdf",
        "st-moe.pdf",
    ]
    .iter()
    .map(|p| format!("/home/you/papers/{p}"))
    .collect();
    let mut t = Thread::new("Compare the four MoE papers");
    t.loaded = true;
    t.live = true;
    t.task = Some(Task::Research);
    t.files = papers.clone();
    t.stats = Some(Stats {
        tok_s: 46.,
        ttft_ms: 131.,
        reused: 0.71,
    });
    t.items = vec![
        Item::User { text: "Which routing loss actually holds up at 35B? Use the four papers I attached.".into(), files: papers, images: vec![] },
        Item::Crew(research_crew()),
        Item::Steps(vec![step("Read 4 papers · 61 pages", StepState::Ok), step("Picked 23 relevant passages", StepState::Ok), step("Checked 6 numbers against the tables", StepState::Ok), Step { name: "write_file".into(), now: "Saving the comparison".into(), ..step("", StepState::Wait) }]),
        Item::Text {
            text: "## Only one of the four is measured at 35B.\n\nThe **expert-choice** routing loss is the only one with results at the 30B+ scale [2]; the other three report 7B or smaller [1, 4]. The dropless variant claims it should transfer, but its 13B run is the largest it shows [3], so treat that as a hypothesis, not a result.".into(),
            cut: false,
        },
    ];
    t.ask = Some(Ask::Code {
        name: "write_file".into(),
        args: json!({"path": "~/papers/moe-routing-comparison.md", "content": "x\n".repeat(84)}),
        path: Some("~/papers/moe-routing-comparison.md".into()),
        folder: Some("~/papers".into()),
    });
    t.updated = now;
    s.threads.insert("moe".into(), t);
    let mut c = Thread::new("Fix the flaky cancel test");
    c.loaded = true;
    c.items = vec![
        Item::User { text: "The cancel test fails one run in ten. Find out why and fix it.".into(), files: vec![], images: vec![] },
        Item::Text { text: "The test cancels before the first token is queued, so sometimes there is nothing to cancel yet. I made it wait for the first `Delta` before calling `cancel`.".into(), cut: false },
    ];
    c.ask = Some(Ask::Confirm {
        name: "bash".into(),
        args: json!({"command": "cargo test -p larch-app cancel -- --test-threads 1"}),
    });
    c.updated = now - 40 * min;
    s.threads.insert("cancel".into(), c);
    let mut d = pinned_doc();
    d.updated = now - 2 * min;
    s.threads.insert("report".into(), d);
    s.chats.insert(
        0,
        ChatEntry {
            id: "report".into(),
            title: "Why did net income fall?".into(),
            updated_at: now - 2 * min,
            turns: 1,
        },
    );
    s.facts = [
        ("prefers-35b", "Prefers results measured at 35B."),
        ("cite-pages", "Cite page numbers when quoting papers."),
        ("uses-wsl", "Builds in WSL on Windows 11."),
        ("gpu", "Has an RTX 5070 Ti with 16 GB."),
    ]
    .iter()
    .enumerate()
    .map(|(i, (slug, text))| Fact {
        slug: (*slug).into(),
        text: (*text).into(),
        source: "a chat".into(),
        updated_at: now - (i as u64 + 1) * 3 * 60 * min,
    })
    .collect();
    let knob = |id, name: &str, value, default| Knob {
        id,
        name: name.into(),
        value,
        min: 0.,
        max: 4096.,
        default,
        activation: "live".into(),
    };
    s.knobs = vec![
        knob(1, "max_batch_seqs", 20., 16.),
        knob(2, "prefill_chunk_tokens", 1024., 512.),
        knob(3, "kv_pool_fraction", 0.85, 0.85),
        knob(4, "temperature_default", 0.6, 0.7),
    ];
    s.ledger.changes = vec![
        ConfigChange {
            engine: 1,
            rev: 9,
            knob: "max_batch_seqs".into(),
            old: 16.,
            new: 20.,
            cause: "gait".into(),
            at: now - 3 * min,
        },
        ConfigChange {
            engine: 1,
            rev: 7,
            knob: "prefill_chunk_tokens".into(),
            old: 512.,
            new: 1024.,
            cause: "gait".into(),
            at: now - 26 * 60 * min,
        },
        ConfigChange {
            engine: 1,
            rev: 4,
            knob: "temperature_default".into(),
            old: 0.7,
            new: 0.6,
            cause: "human".into(),
            at: now - 50 * 60 * min,
        },
    ];
    let mut tok = 38.;
    for i in 0..60u64 {
        tok += ((i * 7919) % 11) as f64 * 0.35 - 1.6;
        tok = tok.clamp(36., 49.);
        s.samples
            .push_back((now - (60 - i) * 10_000, if i > 34 { tok + 3. } else { tok }));
    }
    s.last = Some(EngineSample {
        tok_s: 47.,
        running: 3,
        waiting: 0,
        queue_wait_ms: 0.,
        weights_bytes: (9.3 * GB) as u64,
        kv_used_bytes: (2.4 * GB) as u64,
        kv_total_bytes: (3.8 * GB) as u64,
        gpu_used_bytes: Some((13.1 * GB) as u64),
        gpu_total_bytes: Some(gpu),
        power_cap: None,
    });
    // The MCP tools page: a server on, one waiting for a sign-in, one that failed; the registry's
    // results start with the servers that need no sign-in.
    let tool = |name: &str, description: &str, read_only: bool| McpToolInfo {
        name: name.into(),
        description: description.into(),
        read_only,
        left_out: None,
    };
    let http = |url: &str| McpTransport::Http {
        url: url.into(),
        headers: Default::default(),
    };
    s.mcp.searched = true;
    s.mcp.servers = vec![
        McpServerStatus {
            id: "deepwiki".into(),
            name: "DeepWiki".into(),
            source: "com.deepwiki/deepwiki".into(),
            transport: http("https://mcp.deepwiki.com/mcp"),
            enabled: true,
            trusted: true,
            status: "connected".into(),
            tools: vec![
                tool(
                    "read_wiki_structure",
                    "The wiki pages of a GitHub repository.",
                    false,
                ),
                tool("read_wiki_contents", "A repository's whole wiki.", false),
                tool(
                    "ask_wiki_question",
                    "Answers a question about a repository.",
                    false,
                ),
            ],
            error: None,
            signed_in: false,
            can_sign_in: true,
            signing: false,
        },
        McpServerStatus {
            id: "time".into(),
            name: "Time".into(),
            source: "ai.b77/time".into(),
            transport: http("https://time.b77.ai/mcp"),
            enabled: true,
            trusted: false,
            status: "needs-sign-in".into(),
            tools: Vec::new(),
            error: Some("this server needs a sign-in: press Sign in on its card".into()),
            signed_in: false,
            can_sign_in: true,
            signing: false,
        },
        McpServerStatus {
            id: "notes".into(),
            name: "Notes".into(),
            source: "custom".into(),
            transport: McpTransport::Stdio {
                command: "nope-mcp".into(),
                args: Vec::new(),
                env: Default::default(),
                secret_args: Vec::new(),
            },
            enabled: true,
            trusted: false,
            status: "error".into(),
            tools: Vec::new(),
            error: Some("the server would not start: nope-mcp: No such file or directory".into()),
            signed_in: false,
            can_sign_in: false,
            signing: false,
        },
    ];
    s.mcp.found = crate::mcp::suggested()
        .into_iter()
        .chain([
            crate::mcp::card(&json!({"server": {"name": "ai.b77/time", "title": "Time", "version": "1.0.0",
                "description": "A live clock for your assistant.", "websiteUrl": "https://b77.ai/apps/time",
                "remotes": [{"type": "streamable-http", "url": "https://time.b77.ai/mcp"}]}})),
            crate::mcp::card(&json!({"server": {"name": "com.acme/legacy", "description": "An old server that only speaks SSE.",
                "remotes": [{"type": "sse", "url": "https://acme.example/sse"}]}})),
        ])
        .collect();
    s.hf.searched = true;
    let gguf = |repo: &str, q: &str, gb: f64, dl: bool| HfFile {
        file: format!(
            "{}-{q}.gguf",
            repo.rsplit('/')
                .next()
                .unwrap_or(repo)
                .trim_end_matches("-GGUF")
        ),
        parts: vec!["1".into()],
        size_bytes: (gb * GB) as u64,
        quant: q.into(),
        downloaded: dl,
        mmproj: None,
    };
    for (repo, q, gb, dls, done, kv) in [
        (
            "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            "Q4_K_M",
            9.0,
            2_100_000,
            true,
            196_608,
        ),
        (
            "Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "Q4_K_M",
            18.6,
            1_400_000,
            false,
            98_304,
        ),
        (
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "Q8_0",
            8.1,
            1_900_000,
            false,
            57_344,
        ),
        (
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "Q4_K_M",
            10.4,
            640_000,
            false,
            163_840,
        ),
        (
            "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
            "Q3_K_M",
            15.9,
            910_000,
            false,
            262_144,
        ),
        (
            "mistralai/Codestral-22B-GGUF",
            "Q4_K_M",
            13.3,
            410_000,
            false,
            229_376,
        ),
        (
            "bigcode/StarCoder2-15B-Instruct-GGUF",
            "Q4_K_M",
            9.9,
            210_000,
            false,
            131_072,
        ),
    ] {
        s.hf.results.push(HfModel {
            repo: repo.into(),
            downloads: dls,
            tags: vec![],
            arch: String::new(),
            params: None,
            context_length: None,
            arch_runs: false,
        });
        let f = gguf(repo, q, gb, done);
        s.hf.info.insert(
            (repo.into(), f.file.clone()),
            RemoteModel {
                arch: "qwen2".into(),
                supported: true,
                quant: q.into(),
                weights_bytes: f.size_bytes,
                params: 0,
                kv_bytes_per_token: Some(kv),
                context_length: Some(131_072),
                expert_bytes: 0,
                unloadable: vec![],
            },
        );
        s.hf.files.insert(
            repo.into(),
            Ok(vec![f, gguf(repo, "Q8_0", gb * 1.8, false)]),
        );
    }
}

/// A chat on a pinned document: its chip and an answer whose cites are checked against the
/// document, as the app checks them (the demo has no core to read it from).
pub fn pinned_doc() -> Thread {
    use larch_agent::folio::{Rep, source};
    let path = "/home/you/docs/q3-report.pdf".to_owned();
    let text = "Q3 Report\nHighlights\nRevenue grew 12% to $4.1M.\nGross margin held at 61%.\n\
                Churn fell to 2.3% a month.\n\u{c}Costs\nHosting cost rose 18% as the new region opened.\n\
                Headcount stayed at 14.\nNet income fell to $310k.\n\u{c}Outlook\n\
                We expect Q4 revenue of $4.4M.\nHiring resumes in January.\n";
    let answer = "Hosting cost rose when the new region opened: \"Hosting cost rose 18% as the new region opened\" [p. 2]. Net income fell: \"Net income fell to $310k\" [p. 2]. Revenue still grew: \"Revenue grew 12% to $4.1M\" [p. 1].";
    let docs = [source::doc(0, &path, "", Rep::Pdf, text.to_owned())];
    let mut t = Thread::new("Why did net income fall?");
    t.loaded = true;
    t.files = vec![path];
    t.docs.chip = Some("Document · Ready".into());
    t.docs.first_word = Some("First word in 0.9 s".into());
    t.cites.insert(1, logic::cite_links(answer, &docs));
    t.items = vec![
        Item::User {
            text: "Why did net income fall?".into(),
            files: vec![],
            images: vec![],
        },
        Item::Text {
            text: answer.into(),
            cut: false,
        },
    ];
    t
}

/// The Tasks mockups' runs and schedules, laid out around this week so the calendar shows them.
fn tasks(s: &mut Ecl) {
    use crate::tasks_model::{
        CapabilitySet, Category, Decision, Record, ScheduledRecord, StepNode,
    };
    use chrono::{Datelike, Duration, Local, NaiveTime, TimeZone, Utc};
    use larch_task::schedule::{Policy, Schedule};
    let node = |id: &str, deps: &[&str]| StepNode {
        id: id.into(),
        depends_on: deps.iter().map(|d| (*d).into()).collect(),
        kind: "step",
    };
    let strings = |items: &[&str]| items.iter().map(|i| (*i).to_owned()).collect::<Vec<_>>();
    let run = |id: &str, status: &str, goal: &str, nodes: Vec<StepNode>| Record {
        id: id.into(),
        revision: 6,
        status: status.into(),
        goal: goal.into(),
        nodes,
        has_initial_graph: true,
        startable_steps: true,
        planning: "adaptive".into(),
        scope: vec![],
        acceptance: vec![],
        result: None,
        category: match status {
            "waiting" => Category::NeedsYou,
            "succeeded" | "failed" | "cancelled" => Category::History,
            _ => Category::Running,
        },
        recovery: None,
        decision: None,
        skipped: vec![],
        attempt_failure: None,
        unresolved_effects: false,
        started: vec![],
        completed: vec![],
        results: Default::default(),
        origin: None,
    };
    let moe = [
        "Find the four papers",
        "Read Switch + top-k papers",
        "Read soft-MoE paper",
        "Read expert-choice paper",
        "Compare and recommend",
        "Save to Papers/MoE",
    ];
    let mut compare = run(
        "demo-moe",
        "running",
        "Compare the four MoE papers and tell me which routing holds up at 30B.",
        vec![
            node(moe[0], &[]),
            node(moe[1], &[moe[0]]),
            node(moe[2], &[moe[1]]),
            node(moe[3], &[moe[1]]),
            node(moe[4], &[moe[2], moe[3]]),
            node(moe[5], &[moe[4]]),
        ],
    );
    compare.started = strings(&moe[..4]);
    compare.completed = strings(&moe[..3]);
    compare.results = [
        (moe[0], "Switch Transformer, top-2 GShard, soft MoE and expert-choice routing: all four have results past 7B."),
        (moe[1], "Top-k drops tokens once an expert fills; Switch's capacity factor 1.25 loses about 1 % of tokens at 30B."),
        (moe[2], "Soft MoE mixes tokens into slots, so nothing drops, but its decoder variant trails top-2 by 0.6 % at 13B."),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let mut flaky = run(
        "demo-flaky",
        "running",
        "Fix the flaky cancel test",
        vec![
            node("Reproduce it 50 times", &[]),
            node("Find the race", &["Reproduce it 50 times"]),
            node("Patch and rerun", &["Find the race"]),
            node("Open a pull request", &["Patch and rerun"]),
        ],
    );
    flaky.started = strings(&["Reproduce it 50 times", "Find the race"]);
    flaky.completed = strings(&["Reproduce it 50 times"]);
    let mut tidy = run(
        "demo-tidy",
        "waiting",
        "Tidy the downloads folder",
        vec![
            node("List what is there", &[]),
            node("Move 214 files into dated folders", &["List what is there"]),
        ],
    );
    tidy.started = strings(&["List what is there", "Move 214 files into dated folders"]);
    tidy.completed = strings(&["List what is there"]);
    tidy.decision = Some(Decision {
        request_id: "move:1".into(),
        step: "Move 214 files into dated folders".into(),
        action_digest: "sha256:5f1c…9a2e".into(),
        scope: "move files in ~/Downloads".into(),
        expires_at: now_ms() / 1000 + 25 * 60,
    });
    let mut rename = run(
        "demo-rename",
        "succeeded",
        "Rename screenshots by date",
        vec![node("Rename them", &[])],
    );
    rename.started = strings(&["Rename them"]);
    rename.completed = strings(&["Rename them"]);
    rename.result = Some("Renamed 38 screenshots to YYYY-MM-DD hh.mm.png.".into());
    let cuda = run(
        "demo-cuda",
        "cancelled",
        "Check CUDA 13.4 notes",
        vec![node("Read the notes", &[])],
    );
    let today = Local::now().date_naive();
    let monday = today - Duration::days(i64::from(today.weekday().num_days_from_monday()));
    let at = |day: i64, h: u32, m: u32| {
        Local
            .from_local_datetime(
                &(monday + Duration::days(day))
                    .and_hms_opt(h, m, 0)
                    .unwrap_or_default(),
            )
            .earliest()
            .unwrap_or_default()
            .with_timezone(&Utc)
    };
    let mut records = vec![compare, flaky, tidy, rename, cuda];
    // The digest's runs this week, before now: Monday ran, Tuesday stopped, then it ran again.
    for (day, status) in [
        (0, "succeeded"),
        (1, "failed"),
        (2, "succeeded"),
        (3, "succeeded"),
        (4, "succeeded"),
    ] {
        let when = at(day, 7, 45);
        if when > Utc::now() {
            break;
        }
        let mut digest = run(
            &format!("demo-digest-{day}"),
            status,
            "Morning paper digest",
            vec![node("Read new MoE papers", &[])],
        );
        digest.origin = Some(("digest".into(), when.timestamp()));
        records.push(digest);
    }
    s.tasks.capabilities = CapabilitySet {
        authoring: true,
        explicit_start: true,
        controls: true,
        retry_child: true,
        execution: true,
        planning: true,
        schedules: true,
        decisions: true,
        autonomous_application: false,
        deletes: true,
    };
    s.tasks.replace(records);
    for (run, kinds) in [
        (
            "demo-moe",
            &[
                "attempt_started",
                "attempt_finished",
                "attempt_started",
                "attempt_finished",
                "attempt_started",
                "attempt_started",
            ][..],
        ),
        (
            "demo-tidy",
            &[
                "attempt_started",
                "attempt_finished",
                "attempt_started",
                "decision_required",
            ][..],
        ),
    ] {
        let events: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| (i as u64 + 1, (*k).to_owned()))
            .collect();
        let _ = s.tasks.apply_events(run, &events);
    }
    // The MoE run's finished steps, as the service records them: a minute or so each on a 4 GB card.
    let start = now_ms() as i64 - 9 * 60_000;
    let attempts = [
        (moe[0], 0, 41_000, 3_120, 286),
        (moe[1], 41_000, 155_000, 9_840, 912),
        (moe[2], 155_000, 236_000, 7_410, 655),
    ];
    s.tasks.set_attempts(
        "demo-moe",
        attempts
            .iter()
            .map(
                |(step, from, to, prompt, completion)| crate::tasks_model::Attempt {
                    step: (*step).into(),
                    started_ms: start + from,
                    finished_ms: start + to,
                    tokens: Some((*prompt, *completion)),
                },
            )
            .collect(),
    );
    let local = |h, m| NaiveTime::from_hms_opt(h, m, 0).unwrap_or_default();
    let tz: chrono_tz::Tz = iana_time_zone::get_timezone()
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(chrono_tz::UTC);
    let daily = |h, m| Schedule::Calendar {
        start: monday,
        time: local(h, m),
        timezone: tz,
        rule_revision: chrono_tz::IANA_TZDB_VERSION.into(),
        days: larch_task::schedule::EVERY_DAY,
    };
    let weekdays = |h, m| match daily(h, m) {
        Schedule::Calendar {
            start,
            time,
            timezone,
            rule_revision,
            ..
        } => Schedule::Calendar {
            start,
            time,
            timezone,
            rule_revision,
            days: 0b001_1111,
        },
        other => other,
    };
    let rules = [
        (
            "digest",
            "Morning paper digest: read new MoE papers and file the good ones",
            daily(7, 45),
            true,
        ),
        (
            "repo",
            "Repo check: new CI failures on dev",
            weekdays(18, 0),
            true,
        ),
        (
            "cleanup",
            "Weekly disk cleanup",
            Schedule::Interval {
                anchor: at(6, 10, 0) - Duration::weeks(1),
                seconds: 7 * 86_400,
            },
            false,
        ),
    ];
    let (catch_up, overlap) = crate::tasks::policy_labels(&Policy::default());
    s.tasks.replace_schedules(
        rules
            .iter()
            .map(|(id, goal, rule, enabled)| ScheduledRecord {
                id: (*id).into(),
                revision: 1,
                goal: (*goal).into(),
                steps: vec!["Read new MoE papers".into()],
                planning: "initial".into(),
                scope: vec![],
                acceptance: vec![],
                schedule: crate::tasks::rule_label(rule),
                next_due: rule
                    .next_at_or_after(Utc::now())
                    .ok()
                    .flatten()
                    .map(|t| t.to_rfc3339()),
                enabled: *enabled,
                catch_up: catch_up.clone(),
                overlap: overlap.clone(),
            })
            .collect(),
    );
    s.task_rules = rules
        .into_iter()
        .map(|(id, _, rule, _)| (id.to_owned(), (rule, Policy::default())))
        .collect();
}

fn helper(task: &str, now: &str, st: StepState, frac: Option<f32>) -> crate::crew::Helper {
    crate::crew::Helper {
        task: task.into(),
        now: now.into(),
        st,
        frac,
    }
}

/// A research request: one reader per paper, the last waiting for a seat.
fn research_crew() -> crate::crew::Crew {
    crate::crew::Crew::new(vec![
        helper(
            "switch-routing.pdf",
            "Load-balancing loss tested up to 7B (p. 9)",
            StepState::Ok,
            None,
        ),
        helper(
            "expert-choice.pdf",
            "Reading section 4 · 14 of 22 pages",
            StepState::Run,
            Some(0.64),
        ),
        helper(
            "dropless.pdf",
            "Checking the 13B table against the text",
            StepState::Run,
            None,
        ),
        helper("st-moe.pdf", "Waiting for a free seat", StepState::No, None),
    ])
}

#[cfg(test)]
mod tests {
    /// The demo has a pinned document: its chip, and an answer whose cites open its pages.
    #[test]
    fn the_demo_shows_a_pinned_document() {
        let t = super::pinned_doc();
        assert_eq!(t.docs.chip.as_deref(), Some("Document · Ready"));
        assert!(t.docs.first_word.is_some());
        let cites = &t.cites[&1];
        assert_eq!(cites.len(), 3);
        for c in cites {
            let hit = c.hit.clone().expect("each demo quote is on its page");
            assert!(!c.missing && c.page[hit].len() > 10, "{}", c.label);
        }
    }
}
