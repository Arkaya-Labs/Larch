//! A GitHub repository an outside tool wants as `owner/repo`, from the bare name a small model
//! wrote. DeepWiki's tools take `repoName` "in owner/repo format (e.g. \"facebook/react\")"; Qwen3
//! 1.7B wrote "ollama", and on the server's error "ollama/react". Asked alone under a grammar it
//! named 7 of 20 projects right ("qwen/qwen", "rust/rust", "facebook/vllm"); GitHub's own search,
//! the most-starred repository whose name is the one asked, named them (10 Nov). Only the name
//! leaves the computer, to api.github.com, and only for an argument the server describes as
//! owner/repo; a search that fails leaves the argument as it was, for the server to refuse.

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex, PoisonError},
    time::Duration,
};

use serde_json::Value;

/// The repositories a search reads, most stars first.
const RESULTS: usize = 10;

/// Each name asked this process's life and its repository, or none: GitHub allows ten searches a
/// minute without a sign-in.
static FOUND: LazyLock<Mutex<HashMap<String, Option<String>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// `args` with each string (or string in a list) of a property `schema` describes as owner/repo
/// that has no `/`, as `find` names it; the rest as it was. None when nothing changed.
pub(crate) fn resolve(
    schema: &Value,
    args: &Value,
    find: impl Fn(&str) -> Option<String>,
) -> Option<Value> {
    let props = schema["properties"].as_object()?;
    let mut out = args.clone();
    let mut changed = false;
    for (key, prop) in props {
        let wants =
            (prop["description"].as_str()).is_some_and(|d| d.to_lowercase().contains("owner/repo"));
        if !wants {
            continue;
        }
        let mut fix = |v: &mut Value| {
            if let Some(name) = v.as_str().map(str::trim).filter(|n| !n.contains('/'))
                && let Some(full) = find(name)
            {
                *v = full.into();
                changed = true;
            }
        };
        match out.get_mut(key) {
            Some(Value::Array(items)) => items.iter_mut().for_each(&mut fix),
            Some(v) => fix(v),
            None => {}
        }
    }
    changed.then_some(out)
}

/// The repository GitHub's search finds for `name`: the most-starred whose name is `name` (case,
/// spaces, dots, dashes and underscores aside), else the most-starred that has it in its name.
/// Asked once a process per name.
pub(crate) fn find(name: &str) -> Option<String> {
    let key = name.to_lowercase();
    if let Some(found) = lock().get(&key) {
        return found.clone();
    }
    let found = search(name);
    // A failed search (offline, rate-limited) is tried again next time.
    if let Ok(found) = &found {
        lock().insert(key, found.clone());
    }
    found.ok().flatten()
}

/// The repository `find` named for `name` earlier in this process, with no search: nothing leaves
/// the computer.
pub(crate) fn cached(name: &str) -> Option<String> {
    lock().get(&name.to_lowercase()).cloned().flatten()
}

fn lock() -> std::sync::MutexGuard<'static, HashMap<String, Option<String>>> {
    FOUND.lock().unwrap_or_else(PoisonError::into_inner)
}

fn search(name: &str) -> Result<Option<String>, ()> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(10))
        .build();
    let body: Value = agent
        .get("https://api.github.com/search/repositories")
        .set("Accept", "application/vnd.github+json")
        .set("User-Agent", "larch-agent")
        .query("q", &format!("{name} in:name"))
        .query("sort", "stars")
        .query("order", "desc")
        .query("per_page", &RESULTS.to_string())
        .call()
        .map_err(|_| ())?
        .into_string()
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or(())?;
    let items: Vec<(String, String)> = (body["items"].as_array().into_iter().flatten())
        .filter_map(|i| Some((i["full_name"].as_str()?.into(), i["name"].as_str()?.into())))
        .collect();
    Ok(pick(name, &items))
}

/// Of `items` (full name, name), most stars first: the first whose name is `name`, else the first.
fn pick(name: &str, items: &[(String, String)]) -> Option<String> {
    let plain = |s: &str| -> String {
        (s.chars())
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let want = plain(name);
    (items.iter())
        .find(|(_, n)| plain(n) == want)
        .or(items.first())
        .map(|(full, _)| full.clone())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn items(names: &[&str]) -> Vec<(String, String)> {
        (names.iter())
            .map(|f| (f.to_string(), f.rsplit('/').next().unwrap().to_string()))
            .collect()
    }

    #[test]
    fn the_repository_named_as_asked_wins_over_more_stars() {
        // GitHub's top three for "rust" and "stable diffusion" in:name by stars (10 Nov).
        let rust = items(&["rustdesk/rustdesk", "rust-lang/rust", "rust-lang/rustlings"]);
        assert_eq!(pick("rust", &rust).as_deref(), Some("rust-lang/rust"));
        let sd = items(&[
            "AUTOMATIC1111/stable-diffusion-webui",
            "CompVis/stable-diffusion",
        ]);
        assert_eq!(
            pick("stable diffusion", &sd).as_deref(),
            Some("CompVis/stable-diffusion")
        );
        let qwen = items(&["agentscope-ai/QwenPaw", "QwenLM/qwen-code"]);
        assert_eq!(
            pick("qwen", &qwen).as_deref(),
            Some("agentscope-ai/QwenPaw")
        );
        assert_eq!(pick("x", &[]), None);
    }

    #[test]
    fn only_bare_names_of_owner_repo_arguments_are_resolved() {
        // DeepWiki's schemas (10 Nov).
        let ask = json!({"type": "object", "properties": {
            "repoName": {"anyOf": [{"type": "string"}, {"items": {"type": "string"}, "type": "array"}],
                "description": "GitHub repository or list of repositories (max 10) in owner/repo format."},
            "question": {"type": "string", "description": "The question to ask about the repository."}}});
        let find = |n: &str| (n == "ollama").then(|| "ollama/ollama".to_string());
        let got = resolve(
            &ask,
            &json!({"repoName": "ollama", "question": "ollama?"}),
            find,
        );
        assert_eq!(
            got,
            Some(json!({"repoName": "ollama/ollama", "question": "ollama?"}))
        );
        let list = json!({"repoName": ["ollama", "facebook/react"], "question": "q"});
        assert_eq!(
            resolve(&ask, &list, find).unwrap()["repoName"],
            json!(["ollama/ollama", "facebook/react"])
        );
        // Already owner/repo, not found, or no owner/repo argument: nothing changes.
        let full = json!({"repoName": "ollama/ollama", "question": "q"});
        assert_eq!(resolve(&ask, &full, |_| panic!("no search")), None);
        assert_eq!(resolve(&ask, &json!({"repoName": "zzz"}), find), None);
        let other = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        assert_eq!(resolve(&other, &json!({"city": "ollama"}), find), None);
    }
}
