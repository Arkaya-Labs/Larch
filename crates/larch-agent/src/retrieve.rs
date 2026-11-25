//! Item 6 (toolcall-opt): which installed MCP servers a message is for, by BM25 over their tools'
//! names, descriptions and parameter names, so a turn shows the built-ins plus only those servers'
//! tools instead of every outside tool. A server's score is its best tool's. No model step.

use std::collections::HashMap;

/// Words too common to tell tools apart.
const STOP: [&str; 48] = [
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are", "be",
    "it", "this", "that", "my", "me", "i", "you", "your", "can", "what", "how", "do", "does",
    "from", "at", "as", "about", "use", "using", "get", "please", "pls", "tell", "show", "give",
    "some", "any", "all", "if", "when", "which", "will", "would",
];

/// `text` as lower-case words of letters and digits, stop words out, each `stem`med; `snake_case`
/// and `kebab-case` split.
pub(crate) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(str::to_lowercase)
        .filter(|w| !STOP.contains(&w.as_str()))
        .map(stem)
        .collect()
}

/// `w` without a plural's or a verb's `s`, so "create a new page" meets `create-pages`
/// ("Creates … pages"): without it a model was shown create-database ("Creates a new …") and
/// not the tool it was asked for. Rough, but the same for the message and every tool.
fn stem(w: String) -> String {
    let n = w.len();
    if n <= 3 || !w.ends_with('s') || ["ss", "us", "is"].iter().any(|e| w.ends_with(e)) {
        return w;
    }
    if w.ends_with("ies") && n > 4 {
        return format!("{}y", &w[..n - 3]);
    }
    if ["ches", "shes", "xes", "zes", "sses"]
        .iter()
        .any(|e| w.ends_with(e))
    {
        return w[..n - 2].to_owned();
    }
    w[..n - 1].to_owned()
}

/// A BM25 index of tool documents, each with the server it belongs to and the tool's name.
pub(crate) struct Index {
    docs: Vec<(String, String, HashMap<String, f32>, f32)>,
    df: HashMap<String, f32>,
    avg_len: f32,
}

const K1: f32 = 1.2;
const B: f32 = 0.75;

impl Index {
    /// `tools`: (server, tool name, the tool's text).
    pub(crate) fn new(tools: &[(String, String, String)]) -> Index {
        let mut docs = Vec::new();
        let mut df: HashMap<String, f32> = HashMap::new();
        for (server, name, text) in tools {
            let mut tf: HashMap<String, f32> = HashMap::new();
            let ws = words(text);
            for w in &ws {
                *tf.entry(w.clone()).or_default() += 1.0;
            }
            for w in tf.keys() {
                *df.entry(w.clone()).or_default() += 1.0;
            }
            docs.push((server.clone(), name.clone(), tf, ws.len() as f32));
        }
        let avg_len = (docs.iter().map(|d| d.3).sum::<f32>() / docs.len().max(1) as f32).max(1.0);
        Index { docs, df, avg_len }
    }

    /// Each tool's BM25 for `query`, by name, in the index's order.
    pub(crate) fn tools(&self, query: &str) -> Vec<(&str, f32)> {
        let n = self.docs.len() as f32;
        let q = words(query);
        (self.docs.iter())
            .map(|(_, name, tf, len)| {
                let mut s = 0.0;
                for w in &q {
                    let Some(f) = tf.get(w) else { continue };
                    let df = self.df.get(w).copied().unwrap_or(0.0);
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    s += idf * f * (K1 + 1.0) / (f + K1 * (1.0 - B + B * len / self.avg_len));
                }
                (name.as_str(), s)
            })
            .collect()
    }

    /// Each server's score for `query`: its best tool's BM25, highest first.
    pub(crate) fn servers(&self, query: &str) -> Vec<(String, f32)> {
        let mut best: HashMap<String, f32> = HashMap::new();
        for ((server, ..), (_, s)) in self.docs.iter().zip(self.tools(query)) {
            let e = best.entry(server.clone()).or_default();
            *e = e.max(s);
        }
        let mut out: Vec<_> = best.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_server_whose_tools_share_the_message_words_scores_highest() {
        let tools = vec![
            (
                "deepwiki".into(),
                "deepwiki__ask_wiki_question".into(),
                "ask_wiki_question Ask any question about a GitHub repository repoName question"
                    .into(),
            ),
            (
                "deepwiki".into(),
                "deepwiki__read_wiki_contents".into(),
                "read_wiki_contents View documentation about a GitHub repository repoName".into(),
            ),
            (
                "calendar".into(),
                "calendar__create_event".into(),
                "create_event Create an event on the calendar title start end".into(),
            ),
            (
                "weather".into(),
                "weather__forecast".into(),
                "forecast Weather forecast for a city city units days".into(),
            ),
        ];
        let index = Index::new(&tools);
        let got = index.servers("how does the ollama github repository load models");
        assert_eq!(got[0].0, "deepwiki");
        assert!(got[0].1 > 0.0);
        let got = index.servers("what's the weather in austin");
        assert_eq!(got[0].0, "weather");
        let got = index.servers("write me a poem");
        assert!(got.iter().all(|(_, s)| *s == 0.0), "{got:?}");
        // Within a server, the tool that matches scores highest.
        let got = index.tools("read the documentation contents of a repository");
        let best = got.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(best.0, "deepwiki__read_wiki_contents");
    }

    /// A singular in the message meets a plural or a verb's `s` in a tool: "add a page" reaches
    /// create_pages ("Creates … pages"), which no word of it matched before.
    #[test]
    fn a_page_in_the_message_meets_a_tool_that_creates_pages() {
        assert_eq!(
            words("Creates pages, queries entries; searches boxes; status analysis class"),
            [
                "create", "page", "query", "entry", "search", "box", "status", "analysis", "class"
            ]
        );
        let tools = vec![
            (
                "notes".into(),
                "notes__create_pages".into(),
                "create pages Creates one or more pages with properties and content".into(),
            ),
            (
                "notes".into(),
                "notes__create_database".into(),
                "create database Creates a new database with a title and properties".into(),
            ),
            (
                "notes".into(),
                "notes__search".into(),
                "search Searches pages and databases by keywords".into(),
            ),
        ];
        let index = Index::new(&tools);
        let got = index.tools("add a page to my notes");
        let score = |n: &str| got.iter().find(|t| t.0 == n).unwrap().1;
        assert!(
            score("notes__create_pages") > 0.0 && score("notes__create_database") == 0.0,
            "{got:?}"
        );
    }
}
