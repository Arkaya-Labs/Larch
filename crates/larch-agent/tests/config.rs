//! `<data_dir>/agent/config.toml`: the defaults without it, every key, and the errors that name the
//! key and the line.

use std::{fs, path::Path, path::PathBuf};

use larch_agent::{
    code::task::Verify,
    config::{Config, Thinking, load},
    router::{self, Router},
    skills::web_answer::Reader,
    web::search::Provider,
};

/// Writes `text` as the data folder's config.toml.
fn write(data_dir: &Path, text: &str) {
    fs::create_dir_all(data_dir.join("agent")).unwrap();
    fs::write(data_dir.join("agent/config.toml"), text).unwrap();
}

/// The home folder `~` stands for.
fn home() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    PathBuf::from(std::env::var_os(var).expect(var))
}

#[test]
fn a_missing_file_is_the_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let want = Config {
        tool_format: None,
        git_tools: false,
        allowed: Vec::new(),
        mcp_allow: Vec::new(),
        max_workers: 8,
        docs_workers: 7,
        docs_window: 0,
        docs_keep_ready: true,
        docs_cache_bytes: None,
        web_pages: 5,
        web_reader: Reader::Plain,
        paper_tokens: 1500,
        doc_followups: false,
        followup_tokens: 6000,
        research_keep: 3,
        thinking: Thinking {
            budget: 100,
            answer_max_tokens: 0,
        },
        recall_hits: 5,
        memory_keys: false,
        memory_capture: true,
        router: Router::Off,
        router_margin: router::MARGIN,
        search_keys: Vec::new(),
        tool_grammar: true,
        code_experts: false,
        code_verify: Verify::Full,
        outside_pick: false,
        probe_low: None,
        probe_high: None,
    };
    assert_eq!(load(dir.path()).unwrap(), want);
    assert_eq!(Config::default(), want);
}

#[test]
fn an_unknown_key_fails_with_the_key_in_the_message() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "git_tools = true\ncolour = \"blue\"\n");
    let err = load(dir.path()).unwrap_err().to_string();
    let path = dir.path().join("agent").join("config.toml");
    assert!(
        err.starts_with(&format!("{}:2: ", path.display())) && err.contains("'colour'"),
        "{err}"
    );
}

#[test]
fn every_key_reads_with_comments_crlf_a_bom_and_a_home_folder() {
    let dir = tempfile::tempdir().unwrap();
    let elsewhere = dir.path().join("elsewhere");
    let text = format!(
        "\u{feff}# The agent's settings.\r\n\
         tool_format = \"mistral\" # the GGUF says llama\r\n\
         git_tools = true\r\n\
         \r\n\
         allowed = [\"~/notes\", {}]\r\n\
         mcp_allow = [\"~/shared\"]\r\n\
         max_workers = 3\r\n\
         docs_workers = 14\r\n\
         docs_window = 128\r\n\
         docs_keep_ready = false\r\n\
         docs_cache_bytes = 1073741824\r\n\
         web_pages = 3\r\n\
         web_reader = \"plain\"\r\n\
         paper_tokens = 6000\r\n\
         doc_followups = \"on\"\r\n\
         followup_tokens = 4000\r\n\
         research_keep = 0\r\n\
         thinking_budget = 0 # off\r\n\
         answer_max_tokens = 2000\r\n\
         recall_hits = 10\r\n\
         memory_keys = true\r\n\
         memory_capture = false\r\n\
         router = \"logprob\"\r\n\
         router_margin = 3.5\r\n\
         tool_grammar = false\r\n\
         code_experts = true\r\n\
         code_verify = \"simple\"\r\n",
        serde_json::to_string(&elsewhere).unwrap()
    );
    write(dir.path(), &text);
    let want = Config {
        tool_format: Some("mistral".into()),
        git_tools: true,
        allowed: vec![home().join("notes"), elsewhere],
        mcp_allow: vec![home().join("shared")],
        max_workers: 3,
        docs_workers: 14,
        docs_window: 128,
        docs_keep_ready: false,
        docs_cache_bytes: Some(1 << 30),
        web_pages: 3,
        web_reader: Reader::Plain,
        paper_tokens: 6000,
        doc_followups: true,
        followup_tokens: 4000,
        research_keep: 0,
        thinking: Thinking {
            budget: 0,
            answer_max_tokens: 2000,
        },
        recall_hits: 10,
        memory_keys: true,
        memory_capture: false,
        router: Router::Logprob,
        router_margin: 3.5,
        // Read by search_keys_read_an_empty_one_is_unset_and_debug_hides_them.
        search_keys: Vec::new(),
        tool_grammar: false,
        code_experts: true,
        code_verify: Verify::Simple,
        outside_pick: false,
        probe_low: None,
        probe_high: None,
    };
    assert_eq!(load(dir.path()).unwrap(), want);
}

#[test]
fn web_pages_takes_1_to_10() {
    let dir = tempfile::tempdir().unwrap();
    for pages in [1, 10] {
        write(dir.path(), &format!("web_pages = {pages}\n"));
        assert_eq!(load(dir.path()).unwrap().web_pages, pages);
    }
    for text in ["web_pages = 0", "web_pages = 11", "web_pages = 2.5"] {
        write(dir.path(), &format!("{text}\n"));
        let err = load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains(":1: web_pages takes a whole number from 1 to 10"),
            "{text}: {err}"
        );
    }
}

#[test]
fn web_reader_takes_plain_or_chunks() {
    let dir = tempfile::tempdir().unwrap();
    for (name, reader) in [("plain", Reader::Plain), ("chunks", Reader::Chunks)] {
        write(dir.path(), &format!("web_reader = \"{name}\"\n"));
        assert_eq!(load(dir.path()).unwrap().web_reader, reader);
    }
    for (text, says) in [
        (
            "web_reader = \"fast\"",
            "web_reader takes plain or chunks, not 'fast'",
        ),
        ("web_reader = true", "web_reader takes plain or chunks"),
    ] {
        write(dir.path(), &format!("{text}\n"));
        let err = load(dir.path()).unwrap_err().to_string();
        assert!(err.contains(&format!(":1: {says}")), "{text}: {err}");
    }
}

#[test]
fn research_keep_takes_a_whole_number() {
    let dir = tempfile::tempdir().unwrap();
    for text in ["research_keep = -1", "research_keep = 2.5"] {
        write(dir.path(), &format!("{text}\n"));
        let err = load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains(":1: research_keep takes a whole number from 0"),
            "{text}: {err}"
        );
    }
}

#[test]
fn recall_hits_takes_1_to_10() {
    let dir = tempfile::tempdir().unwrap();
    for hits in [1, 10] {
        write(dir.path(), &format!("recall_hits = {hits}\n"));
        assert_eq!(load(dir.path()).unwrap().recall_hits, hits);
    }
    for text in [
        "recall_hits = 0",
        "recall_hits = 11",
        "recall_hits = -1",
        "recall_hits = 2.5",
    ] {
        write(dir.path(), &format!("{text}\n"));
        let err = load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains(":1: recall_hits takes a whole number from 1 to 10"),
            "{text}: {err}"
        );
    }
}

#[test]
fn a_value_the_key_does_not_take_fails_with_the_key_and_its_line() {
    let dir = tempfile::tempdir().unwrap();
    for (text, says) in [
        ("git_tools = \"yes\"", "git_tools"),
        ("memory_keys = 1", "memory_keys takes true or false"),
        ("tool_grammar = \"off\"", "tool_grammar takes true or false"),
        (
            "router = \"gate\"",
            "router takes off, logprob, probe or score, not 'gate'",
        ),
        (
            "router = \"Logprob\"",
            "router takes off, logprob, probe or score, not 'Logprob'",
        ),
        ("router = true", "router takes off, logprob, probe or score"),
        (
            "router_margin = 0",
            "router_margin takes a number of nats above 0",
        ),
        (
            "router_margin = -1.5",
            "router_margin takes a number of nats above 0",
        ),
        ("router_margin = 101", "at most 100"),
        (
            "router_margin = \"2\"",
            "router_margin takes a number of nats",
        ),
        ("probe_low = 2", "probe_low takes a number from 0 to 1"),
        ("probe_low = -1", "probe_low takes a number from 0 to 1"),
        ("probe_high = 1.5", "probe_high takes a number from 0 to 1"),
        ("probe_high = -0.5", "probe_high takes a number from 0 to 1"),
        (
            "probe_high = \"0.5\"",
            "probe_high takes a number from 0 to 1",
        ),
        // Reversed, on the line of the second.
        (
            "probe_low = 0.8\nprobe_high = 0.2",
            "probe_low (0.8) is above probe_high (0.2)",
        ),
        (
            "probe_high = 0.2\nprobe_low = 0.8",
            "probe_low (0.8) is above probe_high (0.2)",
        ),
        ("tool_format = 3", "tool_format"),
        (
            "tool_format = \"chatml\"",
            "tool_format takes qwen3, llama3 or mistral, not 'chatml'",
        ),
        ("max_workers = 0", "max_workers"),
        ("max_workers = 2.5", "max_workers"),
        (
            "docs_workers = 0",
            "docs_workers takes a whole number from 1 to 9223372036854775807",
        ),
        ("docs_workers = 7.5", "docs_workers"),
        (
            "docs_window = -1",
            "docs_window takes a whole number from 0 (the chunk alone) to 4294967295",
        ),
        ("docs_window = 4294967296", "docs_window"),
        ("allowed = \"~/notes\"", "allowed"),
        // A relative folder would depend on where the agent was started.
        ("allowed = [\"notes\"]", "'notes'"),
        ("allowed = [\"~bob/notes\"]", "another user's home folder"),
        ("git_tools = true false", "git_tools"),
        ("git_tools = tru", "git_tools"),
        ("git_tools", "'git_tools'"),
        ("git_tools = true\ngit_tools = false", "git_tools"),
        // Only the three names, not the architectures that map to them.
        ("tool_format = \"llama\"", "not 'llama'"),
        ("tool_format = \"qwen2\"", "not 'qwen2'"),
        ("tool_format = \"QWEN3\"", "not 'QWEN3'"),
        // JSON reads these and TOML does not, so a TOML crate would refuse them later.
        (
            r#"allowed = ["\/tmp\/x"]"#,
            r"allowed: TOML has no escape \/",
        ),
        (
            r#"allowed = ["/tmp/\ud83d\ude00"]"#,
            r"allowed: TOML has no escape \ud83d",
        ),
        (
            "max_workers = 9223372036854775808",
            "max_workers takes a whole number from 1 to 9223372036854775807",
        ),
        (
            "thinking_budget = 4097",
            "thinking_budget takes a whole number from 0 (thinking off) to 4096",
        ),
        ("thinking_budget = -1", "thinking_budget"),
        ("thinking_budget = 1.5", "thinking_budget"),
        ("thinking_budget = \"100\"", "thinking_budget"),
        (
            "answer_max_tokens = 4294967296",
            "answer_max_tokens takes a whole number from 0 (no cap) to 4294967295",
        ),
        // serde_json's own position is left out: the line is the file's.
        (
            "max_workers = 1_000",
            "max_workers: the value does not read: trailing characters",
        ),
    ] {
        write(dir.path(), &format!("# settings\n{text}\n"));
        let err = load(dir.path()).unwrap_err().to_string();
        let line = if text.contains('\n') { ":3: " } else { ":2: " };
        assert!(err.contains(line) && err.contains(says), "{text}: {err}");
        assert!(!err.contains(" at line "), "{text}: {err}");
    }
    // The largest whole number TOML has still reads, and the escapes TOML shares do too: `\u00e9`
    // and `\\`, which JSON writes for a backslash.
    let cafe = serde_json::to_string(&dir.path().join("caf")).unwrap();
    let cafe = format!("{}\\u00e9\"", cafe.strip_suffix('"').unwrap());
    let slash = serde_json::to_string(&dir.path().join(r"a\b")).unwrap();
    assert!(slash.contains(r"\\"), "{slash}");
    write(
        dir.path(),
        &format!("max_workers = 9223372036854775807\nallowed = [{cafe}, {slash}]\n"),
    );
    let config = load(dir.path()).unwrap();
    assert_eq!(config.max_workers as u64, i64::MAX as u64);
    assert_eq!(
        config.allowed,
        [dir.path().join("café"), dir.path().join(r"a\b")]
    );
    // The largest budget reads.
    write(dir.path(), "thinking_budget = 4096\n");
    assert_eq!(load(dir.path()).unwrap().thinking.budget, 4096);
    // One byte-order mark starts the file; a second is not part of TOML.
    write(dir.path(), "\u{feff}\u{feff}git_tools = true\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(err.contains(":1: ") && err.contains("unknown key"), "{err}");
}

#[test]
fn config_toml_sets_the_router_and_its_margin() {
    let dir = tempfile::tempdir().unwrap();
    for (text, router, margin) in [
        ("router = \"off\"\n", Router::Off, router::MARGIN),
        ("router = \"probe\"\n", Router::Probe, router::MARGIN),
        // score parses: the chat refuses it by name (Router::check), not the file.
        ("router = \"score\"\n", Router::Score, router::MARGIN),
        ("router_margin = 1\n", Router::Off, 1.0),
        (
            "router = \"logprob\"\nrouter_margin = 100\n",
            Router::Logprob,
            100.0,
        ),
    ] {
        write(dir.path(), text);
        let config = load(dir.path()).unwrap();
        assert_eq!(
            (config.router, config.router_margin),
            (router, margin),
            "{text}"
        );
    }
    // The unknown-key message names both new keys.
    write(dir.path(), "routers = \"off\"\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("router, router_margin"), "{err}");
}

#[test]
fn the_probe_thresholds_take_0_to_1_low_at_most_high() {
    let dir = tempfile::tempdir().unwrap();
    for (text, low, high) in [
        ("probe_low = 0\nprobe_high = 1\n", Some(0.0), Some(1.0)),
        ("probe_low = 0.0\nprobe_high = 1.0\n", Some(0.0), Some(1.0)),
        ("probe_low = 0.5\nprobe_high = 0.5\n", Some(0.5), Some(0.5)),
        (
            "probe_low = 0.02\nprobe_high = 0.5\n",
            Some(0.02),
            Some(0.5),
        ),
        // One alone: the other stays unset (no mask, or no forcing).
        ("probe_low = 1\n", Some(1.0), None),
        ("probe_high = 0\n", None, Some(0.0)),
    ] {
        write(dir.path(), text);
        let config = load(dir.path()).unwrap();
        assert_eq!((config.probe_low, config.probe_high), (low, high), "{text}");
    }
}

#[test]
fn search_keys_read_an_empty_one_is_unset_and_debug_hides_them() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "tavily_key = \"tvly-secret-1\"\nbrave_key = \"\"\nexa_key = \"exa-secret-2\"\n",
    );
    let config = load(dir.path()).unwrap();
    let providers: Vec<_> = config.search_keys.iter().map(|(p, _)| *p).collect();
    assert_eq!(providers, [Provider::Tavily, Provider::Exa]);
    let shown = format!("{config:?}");
    assert!(!shown.contains("secret"), "{shown}");
    write(dir.path(), "serper_key = 12345\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(err.ends_with(":1: serper_key takes a string"), "{err}");
    assert!(!err.contains("12345"), "{err}");
    write(dir.path(), "bing_key = \"x\"\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(
        err.ends_with("brave_key, serper_key, tavily_key and exa_key"),
        "{err}"
    );
}

#[test]
fn doc_followups_takes_on_or_off_and_followup_tokens_500_and_up() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "doc_followups = true\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(
        err.contains("doc_followups takes \"on\" or \"off\""),
        "{err}"
    );
    write(dir.path(), "followup_tokens = 499\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(
        err.contains("followup_tokens") && err.contains("500"),
        "{err}"
    );
    write(
        dir.path(),
        "doc_followups = \"off\"\nfollowup_tokens = 500\n",
    );
    let config = load(dir.path()).unwrap();
    assert_eq!((config.doc_followups, config.followup_tokens), (false, 500));
}

#[test]
fn paper_tokens_takes_500_and_up() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "paper_tokens = 499\n");
    let err = load(dir.path()).unwrap_err().to_string();
    assert!(err.contains("paper_tokens") && err.contains("500"), "{err}");
    write(dir.path(), "paper_tokens = 500\n");
    assert_eq!(load(dir.path()).unwrap().paper_tokens, 500);
}
