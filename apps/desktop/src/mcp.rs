//! The MCP tools page's logic, with no UI: MCP Registry entries as cards, a card plus the user's
//! answers as the server to install, "add your own" input, what an install will run (secrets
//! hidden), and the tool-count warning. Ported from the web UI's `mcp-core.js`, with its tests.

use std::collections::BTreeMap;

use larch_app::contract::{McpServer, McpTransport};
use serde_json::{Value, json};

/// Past this many tools from enabled servers the tab warns: in the phase 0 bench each extra tool
/// made wrong picks more likely and the first token later.
pub const WARN_TOOLS: usize = 10;
pub const MANY_TOOLS: usize = 30;

/// Where an install answer goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputKind {
    Header,
    Env,
    /// A package argument: positional, or named (`--flag value`).
    Arg {
        named: bool,
    },
}

/// One value an install asks for (an API key, a header, a package argument).
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub kind: InputKind,
    pub name: String,
    pub description: String,
    pub required: bool,
    pub secret: bool,
    /// Its default.
    pub value: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OptionKind {
    /// A Streamable HTTP endpoint: nothing runs on this computer. `fixed`: the headers the
    /// registry gives a value, sent as they are.
    Remote {
        url: String,
        fixed: Vec<(String, String)>,
    },
    /// A package run with npx, uvx or docker.
    Local {
        registry_type: String,
        command: String,
        needs: String,
        package: Value,
    },
}

/// One way to install a card's server.
#[derive(Clone, Debug, PartialEq)]
pub struct InstallOption {
    pub kind: OptionKind,
    pub supported: bool,
    /// Why it is not supported.
    pub why: String,
    pub inputs: Vec<Input>,
}

/// A registry entry as the tab shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub id: String,
    pub title: String,
    pub publisher: String,
    pub description: String,
    pub version: String,
    pub website: String,
    pub options: Vec<InstallOption>,
    pub installable: bool,
    /// One of [`suggested`]: works without a sign-in.
    pub suggested: bool,
}

impl Card {
    /// The first option this app can install.
    pub fn option(&self) -> Option<&InstallOption> {
        self.options.iter().find(|o| o.supported)
    }
}

struct Runner {
    command: &'static str,
    needs: &'static str,
    pre: &'static [&'static str],
}

fn runner(registry_type: &str) -> Option<Runner> {
    match registry_type {
        "npm" => Some(Runner {
            command: "npx",
            needs: "Node.js (npx)",
            pre: &["-y"],
        }),
        "pypi" => Some(Runner {
            command: "uvx",
            needs: "uv (uvx)",
            pre: &[],
        }),
        "oci" => Some(Runner {
            command: "docker",
            needs: "Docker",
            pre: &["run", "-i", "--rm"],
        }),
        _ => None,
    }
}

fn spec(registry_type: &str, p: &Value) -> String {
    let id = p["identifier"].as_str().unwrap_or_default();
    let version = p["version"].as_str().unwrap_or_default();
    match (registry_type, version.is_empty()) {
        (_, true) => id.to_owned(),
        ("npm", false) => format!("{id}@{version}"),
        ("pypi", false) => format!("{id}=={version}"),
        // An image that names a tag or digest already keeps it.
        ("oci", false)
            if id
                .rsplit('/')
                .next()
                .is_some_and(|last| last.contains([':', '@'])) =>
        {
            id.to_owned()
        }
        ("oci", false) => format!("{id}:{version}"),
        _ => id.to_owned(),
    }
}

/// `io.github.someone/server` → `someone`; `com.acme/x` → `acme`.
pub fn publisher(name: &str) -> String {
    let ns: Vec<&str> = name
        .split('/')
        .next()
        .unwrap_or_default()
        .split('.')
        .collect();
    match ns.as_slice() {
        ["io", "github", who, ..] if !who.is_empty() => (*who).to_owned(),
        [_, second, ..] => (*second).to_owned(),
        [first] => (*first).to_owned(),
        [] => String::new(),
    }
}

fn s(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_owned()
}

/// The ways an entry can be installed: remote endpoints first (nothing runs here), then packages
/// this app knows how to run; SSE-only remotes and other package types are listed with the reason.
fn options(server: &Value) -> Vec<InstallOption> {
    let mut out = Vec::new();
    for r in server["remotes"].as_array().into_iter().flatten() {
        let http = r["type"] == "streamable-http";
        let fixed = (r["headers"].as_array().into_iter().flatten())
            .filter_map(|h| Some((s(&h["name"]), h["value"].as_str()?.to_owned())))
            .filter(|(_, v)| !v.contains('{'))
            .collect();
        let inputs = (r["headers"].as_array().into_iter().flatten())
            .filter(|h| {
                h["value"].is_null() || h["value"].as_str().is_some_and(|v| v.contains('{'))
            })
            .map(|h| Input {
                kind: InputKind::Header,
                name: s(&h["name"]),
                description: s(&h["description"]),
                required: h["isRequired"] == true,
                secret: h["isSecret"] == true,
                value: String::new(),
            })
            .collect();
        out.push(InstallOption {
            kind: OptionKind::Remote {
                url: s(&r["url"]),
                fixed,
            },
            supported: http,
            why: if http {
                String::new()
            } else {
                format!("Uses {}, which this app does not speak yet.", s(&r["type"]))
            },
            inputs,
        });
    }
    for p in server["packages"].as_array().into_iter().flatten() {
        let ty = s(&p["registryType"]);
        let run = runner(&ty);
        let stdio = p["transport"].is_null() || p["transport"]["type"] == "stdio";
        // An argument whose value is a template (`{name}` filled from variables) is a form this
        // app cannot fill yet: refused, not run without it.
        let templated = (p["packageArguments"].as_array().into_iter().flatten())
            .any(|a| a["value"].as_str().is_some_and(|v| v.contains('{')));
        let why = match (&run, stdio) {
            (None, _) => format!("{ty} packages are not supported yet."),
            (Some(_), false) => format!(
                "Its package serves over {}, not stdio.",
                s(&p["transport"]["type"])
            ),
            _ if templated => {
                "Its package arguments use templates this app cannot fill yet.".into()
            }
            _ => String::new(),
        };
        let mut inputs: Vec<Input> = (p["environmentVariables"].as_array().into_iter().flatten())
            .map(|v| Input {
                kind: InputKind::Env,
                name: s(&v["name"]),
                description: s(&v["description"]),
                required: v["isRequired"] == true,
                secret: v["isSecret"] == true,
                value: match &v["default"] {
                    Value::Null => String::new(),
                    Value::String(d) => d.clone(),
                    d => d.to_string(),
                },
            })
            .collect();
        inputs.extend(
            (p["packageArguments"].as_array().into_iter().flatten())
                .filter(|a| a["isRequired"] == true && a["value"].is_null())
                .map(|a| Input {
                    kind: InputKind::Arg {
                        named: a["type"] == "named",
                    },
                    name: a["name"]
                        .as_str()
                        .or(a["valueHint"].as_str())
                        .unwrap_or("argument")
                        .to_owned(),
                    description: s(&a["description"]),
                    required: true,
                    secret: a["isSecret"] == true,
                    value: String::new(),
                }),
        );
        out.push(InstallOption {
            supported: run.is_some() && stdio && !templated,
            why,
            kind: OptionKind::Local {
                command: run.as_ref().map_or("", |r| r.command).to_owned(),
                needs: run.as_ref().map_or("", |r| r.needs).to_owned(),
                registry_type: ty,
                package: p.clone(),
            },
            inputs,
        });
    }
    out
}

/// A registry entry (`{server, _meta}` or the bare server) as a card.
pub fn card(entry: &Value) -> Card {
    let server = if entry["server"].is_object() {
        &entry["server"]
    } else {
        entry
    };
    let name = s(&server["name"]);
    let options = options(server);
    Card {
        title: server["title"].as_str().map_or_else(
            || name.rsplit('/').next().unwrap_or_default().to_owned(),
            str::to_owned,
        ),
        publisher: publisher(&name),
        description: s(&server["description"]),
        version: s(&server["version"]),
        website: server["websiteUrl"]
            .as_str()
            .or(server["repository"]["url"].as_str())
            .unwrap_or_default()
            .to_owned(),
        installable: options.iter().any(|o| o.supported),
        suggested: entry["_meta"]["larch/suggested"] == true,
        id: name,
        options,
    }
}

/// Shown first while the search box is empty: servers that work without a sign-in, some of which
/// the MCP Registry does not list. DeepWiki was checked end to end (larch-agent's
/// tests/mcp_live.rs); Time is the MCP reference server.
pub fn suggested() -> Vec<Card> {
    [
        json!({"server": {"name": "com.deepwiki/deepwiki", "title": "DeepWiki", "websiteUrl": "https://deepwiki.com",
            "description": "Answers about any public GitHub repository: its wiki pages, and questions about its code. Ask with owner/repo.",
            "remotes": [{"type": "streamable-http", "url": "https://mcp.deepwiki.com/mcp"}]}, "_meta": {"larch/suggested": true}}),
        json!({"server": {"name": "io.github.modelcontextprotocol/time", "title": "Time", "websiteUrl": "https://github.com/modelcontextprotocol/servers",
            "description": "The current time in any time zone, and conversions between them.",
            "packages": [{"registryType": "pypi", "identifier": "mcp-server-time", "transport": {"type": "stdio"}}]}, "_meta": {"larch/suggested": true}}),
    ]
    .iter()
    .map(card)
    .collect()
}

/// A short, unique id for a new server: lower-case letters, digits and `-`.
pub fn server_id(name: &str, taken: &[String]) -> String {
    let last = name
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut base = String::new();
    for c in last.chars() {
        if c.is_ascii_alphanumeric() {
            base.push(c);
        } else if !base.ends_with('-') {
            base.push('-');
        }
    }
    let mut base: String = base.trim_matches('-').chars().take(24).collect();
    base = base.trim_end_matches('-').to_owned();
    if base.is_empty() {
        base = "server".into();
    }
    let (mut id, mut n) = (base.clone(), 2);
    while taken.contains(&id) {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

/// The server to install from a card's option and the user's answers (by input name), or what is
/// missing.
pub fn plan(
    card: &Card,
    option: &InstallOption,
    answers: &BTreeMap<String, String>,
    taken: &[String],
) -> Result<McpServer, Vec<String>> {
    if !option.supported {
        return Err(vec![option.why.clone()]);
    }
    let value = |i: &Input| {
        answers
            .get(&i.name)
            .cloned()
            .unwrap_or_else(|| i.value.clone())
            .trim()
            .to_owned()
    };
    let missing: Vec<String> = (option.inputs.iter())
        .filter(|i| i.required && value(i).is_empty())
        .map(|i| format!("{} is required.", i.name))
        .collect();
    if !missing.is_empty() {
        return Err(missing);
    }
    let transport = match &option.kind {
        OptionKind::Remote { url, fixed } => McpTransport::Http {
            url: url.clone(),
            headers: (fixed.iter().cloned())
                .chain(
                    (option.inputs.iter())
                        .filter(|i| !value(i).is_empty())
                        .map(|i| (i.name.clone(), value(i))),
                )
                .collect(),
        },
        OptionKind::Local {
            registry_type,
            package,
            ..
        } => {
            let run = runner(registry_type).expect("a supported option has a runner");
            let mut args: Vec<String> = run.pre.iter().map(|a| (*a).to_owned()).collect();
            for a in package["runtimeArguments"].as_array().into_iter().flatten() {
                let Some(v) = a["value"].as_str().filter(|v| *v != "-y") else {
                    continue;
                };
                if a["type"] == "named" {
                    args.push(s(&a["name"]));
                }
                args.push(v.to_owned());
            }
            let mut env = BTreeMap::new();
            for i in &option.inputs {
                let v = value(i);
                if v.is_empty() {
                    continue;
                }
                if i.kind == InputKind::Env {
                    // docker passes each variable into the container by name.
                    if registry_type == "oci" {
                        args.extend(["-e".to_owned(), i.name.clone()]);
                    }
                    env.insert(i.name.clone(), v);
                }
            }
            args.push(spec(registry_type, package));
            // The package's arguments in the registry's order: a fixed value as it is, an asked one
            // as answered (its position kept as secret when the registry marks it so).
            let mut secret_args = Vec::new();
            let asked = |name: &str| {
                (option.inputs.iter())
                    .find(|i| matches!(i.kind, InputKind::Arg { .. }) && i.name == name)
            };
            for a in package["packageArguments"].as_array().into_iter().flatten() {
                let named = a["type"] == "named";
                let name = (a["name"].as_str().or(a["valueHint"].as_str()))
                    .unwrap_or("argument")
                    .to_owned();
                let (v, secret) = match a["value"].as_str() {
                    Some(v) => (v.to_owned(), a["isSecret"] == true),
                    None => match asked(&name) {
                        Some(i) => (value(i), i.secret),
                        None => continue,
                    },
                };
                if v.is_empty() {
                    continue;
                }
                if named {
                    args.push(s(&a["name"]));
                }
                if secret {
                    secret_args.push(args.len());
                }
                args.push(v);
            }
            McpTransport::Stdio {
                command: run.command.to_owned(),
                args,
                env,
                secret_args,
            }
        }
    };
    Ok(McpServer {
        id: server_id(
            if card.title.is_empty() {
                &card.id
            } else {
                &card.title
            },
            taken,
        ),
        name: if card.title.is_empty() {
            card.id.clone()
        } else {
            card.title.clone()
        },
        source: card.id.clone(),
        transport,
        enabled: true,
        trusted: false,
        install: String::new(),
    })
}

/// "Add your own": a command line (quotes group words) or an http(s) URL.
pub fn custom(name: &str, target: &str, taken: &[String]) -> Result<McpServer, Vec<String>> {
    let (name, target) = (name.trim(), target.trim());
    if name.is_empty() {
        return Err(vec!["Give the server a name.".into()]);
    }
    if target.is_empty() {
        return Err(vec!["Type the command that starts it, or its URL.".into()]);
    }
    let lower = target.to_ascii_lowercase();
    let transport = if lower.starts_with("http://") || lower.starts_with("https://") {
        if url::Url::parse(target).is_err() || target.contains(char::is_whitespace) {
            return Err(vec!["That URL is not valid.".into()]);
        }
        McpTransport::Http {
            url: target.to_owned(),
            headers: BTreeMap::new(),
        }
    } else {
        let words = split_line(target);
        // Leading NAME=value words set the server's environment, as a shell line's do; an `env`
        // before them goes, so a line written for Linux or macOS runs on Windows, which has no
        // `env` program.
        let mut rest = match words.first().map(String::as_str) {
            Some("env") => &words[1..],
            _ => &words[..],
        };
        let mut env = BTreeMap::new();
        while let Some((name, value)) = rest.first().and_then(|w| assignment(w)) {
            env.insert(name, value);
            rest = &rest[1..];
        }
        let Some((command, args)) = rest.split_first() else {
            return Err(vec!["Type the command after the variables.".into()]);
        };
        McpTransport::Stdio {
            command: command.clone(),
            args: args.to_vec(),
            env,
            secret_args: Vec::new(),
        }
    };
    Ok(McpServer {
        id: server_id(name, taken),
        name: name.to_owned(),
        source: "custom".into(),
        transport,
        enabled: true,
        trusted: false,
        install: String::new(),
    })
}

/// `line`'s words: spaces split them, and double or single quotes group spaces into a word, also
/// inside one (`KEY="C:\\My Files\\a.json"`). A backslash is kept as it is, as Windows paths
/// need.
fn split_line(line: &str) -> Vec<String> {
    let (mut words, mut word, mut quote, mut started) = (Vec::new(), String::new(), None, false);
    for c in line.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => word.push(c),
            (None, '"' | '\'') => (quote, started) = (Some(c), true),
            (None, c) if c.is_whitespace() => {
                if started {
                    words.push(std::mem::take(&mut word));
                }
                started = false;
            }
            (None, c) => {
                word.push(c);
                started = true;
            }
        }
    }
    if started {
        words.push(word);
    }
    words
}

/// `word` as an environment assignment, `NAME=value` with a variable's name.
fn assignment(word: &str) -> Option<(String, String)> {
    let (name, value) = word.split_once('=')?;
    let mut chars = name.chars();
    let first = chars.next()?;
    let ok = (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    ok.then(|| (name.to_owned(), value.to_owned()))
}

/// What an install will do, with the values of `secrets` (input names) hidden.
pub fn describe(transport: &McpTransport, secrets: &[String]) -> String {
    match transport {
        McpTransport::Http { url, headers } if headers.is_empty() => format!("Connects to {url}"),
        McpTransport::Http { url, headers } => {
            format!(
                "Connects to {url} with headers {}",
                headers.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        }
        McpTransport::Stdio {
            command,
            args,
            env,
            secret_args,
        } => {
            let quote = |w: &str| {
                if w.contains([' ', '"', '\'']) {
                    format!("{w:?}")
                } else {
                    w.to_owned()
                }
            };
            let env: Vec<String> = (env.iter())
                .map(|(k, v)| {
                    format!(
                        "{k}={}",
                        if secrets.contains(k) {
                            "••••".into()
                        } else {
                            quote(v)
                        }
                    )
                })
                .collect();
            let cmd: Vec<String> = std::iter::once(quote(command))
                .chain(
                    args.iter()
                        .enumerate()
                        .map(|(at, a)| match secret_args.contains(&at) {
                            true => "••••".to_owned(),
                            false => quote(a),
                        }),
                )
                .collect();
            let env = if env.is_empty() {
                String::new()
            } else {
                format!("{} ", env.join(" "))
            };
            format!("Runs on this computer: {env}{}", cmd.join(" "))
        }
    }
}

/// The tab's warning for this many tools from enabled servers.
pub fn tool_warning(count: usize) -> Option<String> {
    if count > MANY_TOOLS {
        Some(format!(
            "{count} outside tools are on. Models this size pick the wrong tool far more often with this many, and replies start later. Turn off servers you do not need."
        ))
    } else if count > WARN_TOOLS {
        Some(format!(
            "{count} outside tools are on. Each one makes a wrong pick a little more likely and replies a little slower."
        ))
    } else {
        None
    }
}

/// The parts of an MCP tool's name a confirm shows: its server and its tool, from
/// `<server>__<tool>`; `None` for a built-in tool.
pub fn outside_tool(name: &str) -> Option<(&str, &str)> {
    name.split_once("__")
        .filter(|(server, tool)| !server.is_empty() && !tool.is_empty())
}

/// Every argument an outside tool's call sends, whole, for its y/n: `key: value` (a string as
/// written, any other value as JSON), a value's further lines after it. Allow sends all of it, so
/// none is clipped or left out. Empty for any other tool.
pub fn call_lines(name: &str, args: &Value) -> Vec<String> {
    if outside_tool(name).is_none() {
        return Vec::new();
    }
    let Some(map) = args.as_object() else {
        return vec![args.to_string()];
    };
    let mut lines = Vec::new();
    for (key, value) in map {
        let text = match value {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let mut parts = text.split('\n');
        lines.push(format!("{key}: {}", parts.next().unwrap_or_default()));
        lines.extend(parts.map(|p| format!("  {p}")));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The y/n for an outside call shows every argument whole: a recipient after a long message,
    /// and the rest of a call that has a path.
    #[test]
    fn an_outside_calls_y_n_shows_every_argument_whole() {
        let message = "m".repeat(240);
        let lines = call_lines(
            "mail__send",
            &serde_json::json!({"message": message, "recipient": "someone@example.com"}),
        );
        assert!(lines.contains(&format!("message: {message}")), "{lines:?}");
        assert!(
            lines.contains(&"recipient: someone@example.com".to_owned()),
            "{lines:?}"
        );
        let lines = call_lines(
            "files__move",
            &serde_json::json!({"path": "/a", "destination": "/b", "overwrite": true}),
        );
        let mut sorted = lines.clone();
        sorted.sort();
        assert_eq!(sorted, ["destination: /b", "overwrite: true", "path: /a"]);
        // Further lines of a value follow it.
        assert_eq!(
            call_lines("notes__add", &serde_json::json!({"text": "one\ntwo"})),
            ["text: one", "  two"]
        );
        // A built-in tool shows its own way.
        assert!(call_lines("write_file", &serde_json::json!({"path": "/a"})).is_empty());
    }

    // Shaped as https://registry.modelcontextprotocol.io/v0/servers returns them.
    fn remote() -> Value {
        json!({"server": {"name": "ai.b77/time", "title": "Time", "description": "A live clock.", "version": "1.0.0",
            "websiteUrl": "https://b77.ai/apps/time", "remotes": [{"type": "streamable-http", "url": "https://time.b77.ai/mcp"}]}})
    }
    fn npm() -> Value {
        json!({"server": {"name": "com.pulsemcp/remote-filesystem", "description": "Cloud files.", "version": "0.1.2",
            "repository": {"url": "https://github.com/pulsemcp/mcp-servers"},
            "packages": [{"registryType": "npm", "identifier": "remote-filesystem-mcp-server", "version": "0.1.2", "runtimeHint": "npx",
                "transport": {"type": "stdio"}, "runtimeArguments": [{"type": "positional", "value": "-y"}],
                "environmentVariables": [{"name": "GCS_BUCKET", "isRequired": true, "description": "Bucket."},
                    {"name": "GCS_PRIVATE_KEY", "isSecret": true}, {"name": "GCS_MAKE_PUBLIC", "default": "false"}]}]}})
    }
    fn no_answers() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    /// A package with a fixed `--transport stdio`, a secret `--api-key` the user types and a fixed
    /// trailing flag: every argument runs in the registry's order, and the key is masked in the
    /// preview and wherever the server is shown, though the program gets it.
    #[test]
    fn fixed_package_arguments_run_and_a_secret_one_is_masked() {
        let entry = serde_json::json!({"server": {"name": "io.github.acme/notes", "packages": [{
        "registryType": "npm", "identifier": "@acme/notes", "version": "1.0.0",
        "transport": {"type": "stdio"},
        "packageArguments": [
            {"type": "named", "name": "--transport", "value": "stdio", "isRequired": true},
            {"type": "named", "name": "--api-key", "isRequired": true, "isSecret": true},
            {"type": "positional", "value": "serve"}
        ]}]}});
        let c = card(&entry);
        let o = c.option().unwrap();
        let answers = BTreeMap::from([("--api-key".to_string(), "sk-synthetic".to_string())]);
        let s = plan(&c, o, &answers, &[]).unwrap();
        let McpTransport::Stdio {
            args, secret_args, ..
        } = &s.transport
        else {
            panic!("stdio");
        };
        let tail: Vec<&str> = args
            .iter()
            .rev()
            .take(5)
            .rev()
            .map(String::as_str)
            .collect();
        assert_eq!(
            tail,
            ["--transport", "stdio", "--api-key", "sk-synthetic", "serve"]
        );
        assert_eq!(secret_args.len(), 1);
        assert_eq!(args[secret_args[0]], "sk-synthetic");
        let shown = describe(&s.transport, &[]);
        assert!(!shown.contains("sk-synthetic"), "{shown}");
        assert!(
            shown.contains("--api-key ••••") && shown.contains("--transport stdio"),
            "{shown}"
        );
    }

    /// A remote's header with a fixed value is sent as it is; a templated package argument is
    /// refused, not dropped.
    #[test]
    fn fixed_headers_are_sent_and_templated_arguments_are_refused() {
        let entry = serde_json::json!({"server": {"name": "io.example/api",
            "remotes": [{"type": "streamable-http", "url": "https://api.example/mcp",
                "headers": [{"name": "X-Protocol", "value": "v1"},
                            {"name": "Authorization", "isRequired": true, "isSecret": true}]}],
            "packages": [{"registryType": "npm", "identifier": "x", "version": "1",
                "packageArguments": [{"type": "named", "name": "--dir", "value": "{home}/x"}]}]}});
        let c = card(&entry);
        let remote = &c.options[0];
        let answers = BTreeMap::from([("Authorization".to_string(), "Bearer t".to_string())]);
        let s = plan(&c, remote, &answers, &[]).unwrap();
        let McpTransport::Http { headers, .. } = &s.transport else {
            panic!("http");
        };
        assert_eq!(headers.get("X-Protocol").map(String::as_str), Some("v1"));
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer t")
        );
        let local = &c.options[1];
        assert!(
            !local.supported && local.why.contains("templates"),
            "{}",
            local.why
        );
    }

    #[test]
    fn registry_entries_become_cards_with_publisher_website_and_options() {
        let c = card(&remote());
        assert_eq!((c.title.as_str(), c.publisher.as_str()), ("Time", "b77"));
        assert_eq!(c.website, "https://b77.ai/apps/time");
        assert!(c.installable && !c.suggested);
        let f = card(&npm());
        assert_eq!(
            (f.title.as_str(), f.publisher.as_str()),
            ("remote-filesystem", "pulsemcp")
        );
        assert_eq!(f.website, "https://github.com/pulsemcp/mcp-servers");
        assert!(
            matches!(&f.options[0].kind, OptionKind::Local { needs, .. } if needs == "Node.js (npx)")
        );
        assert_eq!(publisher("io.github.someone/x"), "someone");
    }

    #[test]
    fn unsupported_transports_say_why_and_are_not_installable() {
        let c = card(
            &json!({"server": {"name": "com.acme/old", "remotes": [{"type": "sse", "url": "https://acme/sse"}]}}),
        );
        assert!(!c.installable);
        assert!(c.options[0].why.contains("sse"));
        assert_eq!(
            plan(&c, &c.options[0], &no_answers(), &[]).unwrap_err(),
            vec![c.options[0].why.clone()]
        );
    }

    #[test]
    fn an_npm_package_plans_npx_with_its_version_and_the_env_given() {
        let c = card(&npm());
        let o = c.option().unwrap();
        assert_eq!(
            plan(&c, o, &no_answers(), &[]).unwrap_err(),
            vec!["GCS_BUCKET is required."]
        );
        let answers = BTreeMap::from([
            ("GCS_BUCKET".into(), "b1".into()),
            ("GCS_PRIVATE_KEY".into(), "k".into()),
        ]);
        let s = plan(&c, o, &answers, &[]).unwrap();
        let McpTransport::Stdio {
            command, args, env, ..
        } = &s.transport
        else {
            panic!()
        };
        assert_eq!(
            (command.as_str(), args.clone()),
            (
                "npx",
                vec![
                    "-y".to_string(),
                    "remote-filesystem-mcp-server@0.1.2".into()
                ]
            )
        );
        assert_eq!(env.len(), 3);
        assert_eq!(s.source, "com.pulsemcp/remote-filesystem");
        let d = describe(&s.transport, &["GCS_PRIVATE_KEY".into()]);
        assert!(
            d.contains("GCS_PRIVATE_KEY=••••")
                && d.ends_with("npx -y remote-filesystem-mcp-server@0.1.2"),
            "{d}"
        );
        assert!(!d.contains("=k "));
    }

    #[test]
    fn a_pypi_package_plans_uvx_and_a_remote_plans_http() {
        let t = card(
            &json!({"server": {"name": "io.github.someone/mcp-time", "packages": [{"registryType": "pypi", "identifier": "mcp-server-time", "version": "0.6.2", "transport": {"type": "stdio"}}]}}),
        );
        let s = plan(&t, t.option().unwrap(), &no_answers(), &[]).unwrap();
        assert_eq!(
            s.transport,
            McpTransport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp-server-time==0.6.2".into()],
                env: BTreeMap::new(),
                secret_args: Vec::new(),
            }
        );
        let r = card(&remote());
        let s = plan(&r, r.option().unwrap(), &no_answers(), &[]).unwrap();
        assert_eq!(
            describe(&s.transport, &[]),
            "Connects to https://time.b77.ai/mcp"
        );
    }

    #[test]
    fn server_ids_are_short_safe_and_unique() {
        assert_eq!(server_id("Time", &[]), "time");
        assert_eq!(
            server_id("ai.b77/My Time!", &["my-time".into()]),
            "my-time-2"
        );
        assert_eq!(server_id("", &[]), "server");
    }

    #[test]
    fn custom_servers_take_a_quoted_command_line_or_a_url() {
        let s = custom("Notes", r#"uvx "mcp server" --root '/my notes'"#, &[]).unwrap();
        assert_eq!(
            s.transport,
            McpTransport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp server".into(), "--root".into(), "/my notes".into()],
                env: BTreeMap::new(),
                secret_args: Vec::new(),
            }
        );
        assert_eq!(
            custom("Web", "https://x.example/mcp", &["web".into()])
                .unwrap()
                .id,
            "web-2"
        );
        assert_eq!(
            custom("", "x", &[]).unwrap_err(),
            vec!["Give the server a name."]
        );
        assert_eq!(
            custom("A", "http://bad url", &[]).unwrap_err(),
            vec!["That URL is not valid."]
        );
    }

    #[test]
    fn a_custom_command_line_sets_variables_on_every_os() {
        // The Calendar guide's line for Linux, as typed on any OS: `env` goes, the variable is
        // the server's.
        let line = "env GOOGLE_OAUTH_CREDENTIALS=/home/me/gcp.json npx @cocal/google-calendar-mcp";
        let s = custom("Calendar", line, &[]).unwrap();
        assert_eq!(
            s.transport,
            McpTransport::Stdio {
                command: "npx".into(),
                args: vec!["@cocal/google-calendar-mcp".into()],
                env: BTreeMap::from([(
                    "GOOGLE_OAUTH_CREDENTIALS".into(),
                    "/home/me/gcp.json".into()
                )]),
                secret_args: Vec::new(),
            }
        );
        // A Windows path with spaces, quoted inside the word, and no `env`.
        let line = r#"KEY="C:\Users\Me Too\a.json" B=1 npx.cmd -y pkg"#;
        let McpTransport::Stdio {
            command, args, env, ..
        } = custom("W", line, &[]).unwrap().transport
        else {
            panic!("stdio");
        };
        assert_eq!(
            (command.as_str(), args),
            ("npx.cmd", vec!["-y".to_string(), "pkg".into()])
        );
        assert_eq!(env["KEY"], r"C:\Users\Me Too\a.json");
        assert_eq!(env["B"], "1");
        // Variables only, or an `=` in an argument: not a variable after the command.
        assert_eq!(
            custom("X", "A=1", &[]).unwrap_err(),
            vec!["Type the command after the variables."]
        );
        let s = custom("Y", "uvx srv --opt=a=b", &[]).unwrap();
        let McpTransport::Stdio { args, env, .. } = s.transport else {
            panic!("stdio");
        };
        assert_eq!(
            (args, env.len()),
            (vec!["srv".to_string(), "--opt=a=b".into()], 0)
        );
        assert_eq!(split_line(r#"a "" b"#), vec!["a", "", "b"]);
    }

    #[test]
    fn the_tool_count_warning_starts_past_10_and_grows_past_30() {
        assert_eq!(tool_warning(10), None);
        assert!(tool_warning(11).unwrap().contains("11 outside tools"));
        assert!(tool_warning(31).unwrap().contains("far more often"));
    }

    #[test]
    fn the_suggested_servers_install_without_a_sign_in() {
        let s = suggested();
        assert!(s.iter().all(|c| c.suggested && c.installable));
        let deepwiki = plan(&s[0], s[0].option().unwrap(), &no_answers(), &[]).unwrap();
        assert_eq!(
            deepwiki.transport,
            McpTransport::Http {
                url: "https://mcp.deepwiki.com/mcp".into(),
                headers: BTreeMap::new()
            }
        );
        let time = plan(&s[1], s[1].option().unwrap(), &no_answers(), &[]).unwrap();
        assert_eq!(
            time.transport,
            McpTransport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp-server-time".into()],
                env: BTreeMap::new(),
                secret_args: Vec::new(),
            }
        );
    }

    #[test]
    fn an_mcp_tool_name_splits_into_server_and_tool() {
        assert_eq!(
            outside_tool("time__get_current_time"),
            Some(("time", "get_current_time"))
        );
        assert_eq!(outside_tool("read_file"), None);
        // The agent names tools the same way.
        assert_eq!(
            larch_app::contract::mcp_tool_name("Time", "Get Time!"),
            "time__get_time"
        );
    }
}
