//! IR-05b through the real loop: once untrusted content is in the chat, a fetch opens only a URL
//! from the user's messages, a search result or a fetched page's links; any other asks, and its
//! decline tells the model why. The search and the fetch ride built-in slots, as inject.rs's do.

mod common;

use std::fs;

use common::{Fake, call, events, spaced};
use larch::Role;
use larch_agent::{
    Agent, AgentEvent,
    tool::{ARGS_TOKENS, Permission, Registry, Tool, ToolCtx, ToolError, ToolSpec},
};
use serde_json::{Value, json};

/// The slot of the faked `web_search`.
const SEARCH: &str = "list_dir";
/// The slot of the faked `fetch_page`.
const FETCH: &str = "read_file";
/// On this machine: a web host's link would hold the first step to read_papers, which opens it.
const USER_URL: &str = "http://localhost:8080/start";
const HIT: &str = "https://hit.example/result?id=7";
/// The link every fetched page holds.
const LINK: &str = "https://site.example/next";

/// `web_search` by class and result shape: one hit, `HIT`.
struct Search;

impl Tool for Search {
    fn spec(&self) -> ToolSpec {
        spec(SEARCH, "query")
    }

    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        Ok(json!([{"title": "A hit", "url": HIT, "snippet": "about it"}]).to_string())
    }
}

/// `fetch_page` by class, check and result shape: each page links to `LINK`, and each URL it
/// opens is a line of the data folder's `fetched`.
struct Fetch;

impl Tool for Fetch {
    fn spec(&self) -> ToolSpec {
        spec(FETCH, "url")
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let url = args["url"].as_str().unwrap_or_default();
        let log = ctx.data_dir.join("fetched");
        let before = fs::read_to_string(&log).unwrap_or_default();
        fs::write(&log, format!("{before}{url}\n")).unwrap();
        let text = format!("A page with a [link]({LINK}).");
        Ok(json!({"url": url, "final_url": url, "title": "", "text": text}).to_string())
    }

    fn opens<'a>(&self, args: &'a Value) -> Option<&'a str> {
        args["url"].as_str()
    }
}

fn spec(name: &str, arg: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "A test tool.".into(),
        schema: json!({
            "type": "object",
            "properties": {arg: {"type": "string"}},
            "required": [arg],
            "additionalProperties": false
        }),
        permission: Permission::Read,
        untrusted_results: true,
        max_result_tokens: 4000,
        max_args_tokens: ARGS_TOKENS,
    }
}

fn agent(fake: &Fake) -> Agent {
    let mut agent = fake.agent();
    agent.replace_tool(Box::new(Search));
    agent.replace_tool(Box::new(Fetch));
    agent
}

fn fetched(fake: &Fake) -> Vec<String> {
    let log = fs::read_to_string(fake.data().join("fetched")).unwrap_or_default();
    log.lines().map(String::from).collect()
}

fn fetch(url: &str) -> String {
    call(FETCH, json!({ "url": url }))
}

#[test]
fn before_untrusted_content_any_url_fetches() {
    let fake = Fake::new(32768);
    let mut agent = agent(&fake);
    let made_up = "https://made-up.example/?d=anything";
    fake.say(&fetch(made_up));
    fake.say("Done.");
    let got = events(agent.turn("fetch something").unwrap());
    assert!(agent.parked().is_none(), "{got:?}");
    assert_eq!(fetched(&fake), [made_up]);
}

#[test]
fn after_untrusted_content_only_seen_urls_fetch_and_the_rest_ask_with_the_reason() {
    let fake = Fake::new(32768);
    let mut agent = agent(&fake);
    let added_query = format!("{HIT}&d=secret");
    let made_up = "https://made-up.example/c?d=secret";
    // The user's URL holds the first step to a call, as the grammar writes it.
    fake.say(&spaced(SEARCH, json!({"query": "q"})));
    fake.say(&fetch(USER_URL));
    fake.say(&fetch(HIT));
    fake.say(&fetch(LINK));
    fake.say(&fetch(&added_query));
    let got = events(
        agent
            .turn(&format!("start at {USER_URL}, my code is 7731"))
            .unwrap(),
    );
    assert!(agent.untrusted());
    assert_eq!(fetched(&fake), [USER_URL, HIT, LINK]);
    let asked = |got: &[AgentEvent], url: &str| {
        let notice = got
            .iter()
            .any(|e| matches!(e, AgentEvent::Notice(n) if n.contains(url)));
        let confirm = got
            .iter()
            .any(|e| matches!(e, AgentEvent::NeedsConfirm { call } if call.args["url"] == url));
        notice && confirm
    };
    assert!(asked(&got, &added_query), "{got:?}");
    // The decline quotes the URL to the model; that does not make it seen.
    fake.say(&fetch(&added_query));
    let got = events(agent.confirm(false).unwrap());
    assert!(asked(&got, &added_query), "{got:?}");
    fake.say(&fetch(made_up));
    let got = events(agent.confirm(false).unwrap());
    assert!(asked(&got, made_up), "{got:?}");
    fake.say("I fetched what I could.");
    events(agent.confirm(false).unwrap());
    assert_eq!(fetched(&fake), [USER_URL, HIT, LINK]);
    // Each decline tells the model why, not only that the user said no.
    let history = agent.history();
    let declines: Vec<&str> = (history.iter())
        .filter(|m| m.role == Role::Tool && m.content.starts_with("the user declined"))
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(declines.len(), 3, "{history:?}");
    assert!(
        declines.iter().all(|d| d.contains("carry data out")),
        "{declines:?}"
    );
}

#[test]
fn a_fetch_the_user_allows_runs_and_its_links_become_seen() {
    let fake = Fake::new(32768);
    let mut agent = agent(&fake);
    let other = "https://other.example/";
    fake.say(&call(SEARCH, json!({"query": "q"})));
    fake.say(&fetch(other));
    events(agent.turn("search").unwrap());
    assert!(agent.parked().is_some());
    fake.say(&fetch(LINK));
    fake.say("Done.");
    events(agent.confirm(true).unwrap());
    assert!(agent.parked().is_none());
    assert_eq!(fetched(&fake), [other, LINK]);
}

/// The rule reads the results of the tools whose results are untrusted, when they are JSON: a new
/// one is a new source of URLs, which this test makes someone decide on. web_answer's, read_papers'
/// and research's answers are model-written Markdown, so they make a chat untrusted and give no
/// URLs; their own fetches go through the same check (tests/web_answer.rs, tests/read_papers.rs).
#[test]
fn the_untrusted_built_ins_are_the_known_ones() {
    let registry = Registry::builtin();
    let all: Vec<_> = ["web_search", "fetch_page"]
        .into_iter()
        .chain(larch_agent::CHAT_TOOLS)
        .chain(larch_agent::GIT_TOOLS)
        .collect();
    let untrusted: Vec<String> = (registry.specs(&all).into_iter())
        .filter(|s| s.untrusted_results)
        .map(|s| s.name)
        .collect();
    assert_eq!(
        untrusted,
        [
            "web_search",
            "fetch_page",
            "web_answer",
            "read_papers",
            "research"
        ]
    );
    let fetch = registry.get("fetch_page").unwrap();
    assert_eq!(fetch.opens(&json!({"url": HIT})), Some(HIT));
}
