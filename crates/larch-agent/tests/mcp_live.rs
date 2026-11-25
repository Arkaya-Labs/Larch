//! The MCP client against real outside servers: public ones over Streamable HTTP (DeepWiki, and
//! GitMCP on the older 2025-03-26 revision) and the reference time server over stdio
//! (`uvx mcp-server-time`, which needs uv and downloads it once). They need the network, so they
//! are ignored by default: `cargo test -p larch-agent --test mcp_live -- --ignored`.

use std::collections::BTreeMap;

use larch_agent::mcp::client::{Server, Transport};
use larch_agent::tool::{Tool, ToolCtx};
use serde_json::json;

fn ctx() -> ToolCtx {
    ToolCtx {
        allowed: vec![],
        data_dir: std::env::temp_dir(),
        source: "mcp_live".into(),
    }
}

/// Lists the server's tools, then calls the one named `tool` as a chat would.
fn time_server(id: &str, transport: Transport, tool: &str, args: serde_json::Value) {
    let server = Server::new(id, transport, None);
    let listed = server.connect().expect("connects and lists tools");
    let names: Vec<_> = listed.iter().map(|t| t.name.clone()).collect();
    eprintln!("{id}: {names:?}");
    let tools = server.chat_tools();
    let t = (tools
        .iter()
        .find(|t| t.spec().name == format!("{id}__{tool}")))
    .unwrap_or_else(|| panic!("{id}__{tool} among {names:?}"));
    let out = t.call(&args, &ctx()).expect("the call returns a result");
    eprintln!("{id}__{tool} -> {out}");
    assert!(!out.trim().is_empty());
}

#[test]
#[ignore = "needs uv and the network"]
fn the_reference_time_server_over_stdio() {
    let transport = Transport::Stdio {
        command: "uvx".into(),
        args: vec!["mcp-server-time".into()],
        env: BTreeMap::new(),
        secret_args: Vec::new(),
    };
    time_server(
        "time",
        transport,
        "get_current_time",
        json!({"timezone": "Asia/Tokyo"}),
    );
}

/// Connects over Streamable HTTP, lists the tools, and calls `tool` with `args` as a chat would.
fn remote(id: &str, url: &str, tool: &str, args: serde_json::Value) {
    let transport = Transport::Http {
        url: url.into(),
        headers: BTreeMap::new(),
    };
    let server = Server::new(id, transport, None);
    let listed = server.connect().expect("connects and lists tools");
    let names: Vec<_> = listed.iter().map(|t| t.name.clone()).collect();
    eprintln!("{id}: {names:?}");
    let tools = server.chat_tools();
    let t = (tools
        .iter()
        .find(|t| t.spec().name == format!("{id}__{tool}")))
    .unwrap_or_else(|| panic!("{id}__{tool} among {names:?}"));
    let out = t.call(&args, &ctx()).expect("the call returns a result");
    eprintln!(
        "{id}__{tool} -> {}",
        out.chars().take(300).collect::<String>()
    );
    assert!(!out.trim().is_empty());
}

/// DeepWiki's public server (no sign-in; replies as an event stream).
#[test]
#[ignore = "needs the network"]
fn deepwiki_over_streamable_http() {
    remote(
        "deepwiki",
        "https://mcp.deepwiki.com/mcp",
        "read_wiki_structure",
        json!({"repoName": "modelcontextprotocol/servers"}),
    );
}

/// GitMCP speaks revision 2025-03-26: the client accepts the older revision it answers with.
#[test]
#[ignore = "needs the network"]
fn gitmcp_on_an_older_revision() {
    remote(
        "gitmcp",
        "https://gitmcp.io/modelcontextprotocol/servers",
        "fetch_servers_documentation",
        json!({}),
    );
}

/// A server behind OAuth answers 401: the error says it needs a sign-in, not just "unauthorized".
#[test]
#[ignore = "needs the network"]
fn a_server_behind_a_sign_in_says_so() {
    let transport = Transport::Http {
        url: "https://time.b77.ai/mcp".into(),
        headers: BTreeMap::new(),
    };
    let err = Server::new("b77_time", transport, None)
        .connect()
        .unwrap_err()
        .to_string();
    assert!(err.contains("needs a sign-in"), "{err}");
}

/// The registry's Time server signs in through OAuth: its sign-in service is found the way the app
/// finds it (its 401, its resource metadata, its issuer's metadata at the appended path). Reads
/// public metadata only; nothing is registered.
#[test]
#[ignore = "needs the network"]
fn the_time_servers_sign_in_service_is_found() {
    use larch_agent::mcp::oauth;
    let url = "https://time.b77.ai/mcp";
    let challenge = oauth::probe(url);
    assert!(challenge.resource_metadata.is_some(), "{challenge:?}");
    let (resource, server) = oauth::discover(url, &challenge).expect("discovers");
    eprintln!("{resource:?}\n{server:?}");
    assert_eq!(resource.resource, url);
    assert!(server.registration_endpoint.is_some());
    assert!(server.authorization_endpoint.starts_with("https://"));
}
