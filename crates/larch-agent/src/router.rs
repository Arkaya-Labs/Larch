//! Who decides between a tool call and a plain answer before a chat step: the model alone, a gate
//! on the first token's logprobs, or a probe on the prompt's hidden state (`probe`). And the mode
//! the user picked for a message (`Intent`): a change to a project, code, the web or chat.
//!
//! No model step reads a message's intent before its turn: the model sees every tool and decides.
//! A classifier did until 16 Nov (`router::intent`, 30 Oct); it read a message alone, so "yes" to
//! the assistant's "Would you like me to look up current prices?" was chat, whose turn had no web
//! tool, and the model's search call was refused (a user's Cayman S chat).

use std::{path::PathBuf, sync::atomic::AtomicBool};

use larch::{EngineError, GenParams, Larch, SessionOptions, api::Grammar};
use serde::{Deserialize, Serialize};

use crate::runtime::inference::SeatAdmission;
use crate::{parse::Family, run, skills::ask_docs, workers};

/// The decider in front of a step's tool choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Router {
    /// The model decides alone.
    #[default]
    Off,
    /// A gate on the first token's logprobs.
    Logprob,
    /// A probe on the prompt's hidden state.
    Probe,
    /// A decider that scores the answers' labels.
    Score,
}

impl Router {
    /// Fails for a router this build does not have, naming the PR that adds it.
    pub fn check(self) -> Result<(), String> {
        match self {
            Router::Score => {
                Err("router score is not built yet: v2 PR 13 (agent/score-router) adds it".into())
            }
            _ => Ok(()),
        }
    }
}

/// The first git checkout a request names, anywhere in it; `intent` says whether it asks for a
/// change to it (code mode) or about it.
pub fn named_repo(request: &str) -> Option<PathBuf> {
    // A sentence's end trims, never a path's start (`./toy`).
    let trim = |w: &str| {
        let w = w.trim_start_matches(['"', '\'', '`', '(']);
        w.trim_end_matches(['"', '\'', '`', ')', ',', ';', ':', '!', '?', '.'])
            .to_owned()
    };
    let words = request
        .split_whitespace()
        .map(trim)
        .filter(|w| !w.is_empty());
    words
        .filter_map(|w| crate::tool::expand_home(&w).ok())
        .find(|p| p.join(".git").exists())
}

/// The mode the user picked for a message (the desktop app's chips, the Iris panel): whether it
/// goes to code mode, and which tools its turn may call. A message with none is a turn with every
/// tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intent {
    /// Edit the files of the chat's project: code mode.
    Change,
    /// Run, build, test, time or measure the project's code without a change: code mode with no
    /// tool that writes (`CodeTask::read_only`).
    Run,
    /// Write, explain or review code: an answer with no tool.
    Code,
    /// Needs the internet: every tool.
    Web,
    /// Anything else: every tool but the web's.
    Chat,
}

/// One worker step that reads `request` as one of `labels`, as `system` says, and its label. It
/// thinks not, and with `prefix_cache` shares its system prompt's KV across calls in `namespace`,
/// as a worker's group does (`Workers::run_until`).
fn pick(
    larch: &Larch,
    family: Option<Family>,
    namespace: u64,
    prefix_cache: bool,
    system: &str,
    labels: &[&str],
    request: &str,
) -> Result<String, EngineError> {
    let p = GenParams {
        // The grammar ends the reply at its label; the cap only bounds a label a tokenizer
        // spells in many tokens ("change" is 6 in a byte-level one) and its end of turn.
        max_tokens: 8,
        grammar: Some(Grammar {
            trigger: None,
            pattern: format!("({})", labels.join("|")),
        }),
        ..ask_docs::params(family, namespace)
    };
    if prefix_cache {
        // A session's own prefix goes when it is dropped; the anchor's stays. An engine that
        // fails it fails the session too, which says so.
        let _ = run::anchor(larch, system, &p);
    }
    let mut session = larch.session(SessionOptions {
        system: Some(system.into()),
        keep_kv: prefix_cache,
        ..SessionOptions::default()
    });
    // Foreground, before the turn on the caller's thread: seats never hold it back, so it needs
    // no stop.
    let (never, class) = (AtomicBool::default(), p.class);
    let stream = SeatAdmission::of(larch).seat(class, &never, || {
        session.send(&format!("Message: {request}"), p)
    })?;
    let a = workers::answer(session.id(), stream, &never)?;
    Ok(a.text.trim().to_owned())
}

/// Whether the model reads `request` as asking for what `server`'s tools (`tools`, each name and
/// description) look up: one step held to yes or no, its prompt the server's own descriptions.
/// With the tools shown, Qwen3 1.7B answered "Teach me about ollama" from memory with DeepWiki
/// installed; asked this alone, it says whether the tools fit (10 Nov).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fits(
    larch: &Larch,
    family: Option<Family>,
    namespace: u64,
    prefix_cache: bool,
    server: &str,
    tools: &[(String, String)],
    request: &str,
) -> Result<bool, EngineError> {
    let list: String = (tools.iter().take(FITS_TOOLS))
        .map(|(name, about)| format!("- {name}: {about}\n"))
        .collect();
    let system = format!(
        "An assistant can answer from its own knowledge, or call the tools of {server} to look \
         things up. {server}'s tools:\n{list}Say whether the user's message asks about something \
         these tools look up. Reply with only yes or no."
    );
    let label = pick(
        larch,
        family,
        namespace,
        prefix_cache,
        &system,
        &["yes", "no"],
        request,
    )?;
    Ok(label == "yes")
}

/// The tools of a server `fits` lists: a prompt that stays short for the small model.
const FITS_TOOLS: usize = 8;

/// config.toml's `router_margin` by default, in nats.
// ponytail: 2.0 is a first guess; the Mac's multi-step rows at several margins pick the default
pub const MARGIN: f32 = 2.0;

/// The largest margin a reading reports, and `router_margin` takes: past it the call is certain.
pub const CERTAIN: f32 = 100.0;

/// The top logprobs the gate asks for on the first token.
pub const TOP: u8 = 2;

/// The text a call of `family` can start with: its marker, and for Llama 3 and Mistral the bare
/// JSON a call usually starts with instead (`parse.rs`), alone or after a space or a newline.
fn spellings(family: Family) -> &'static [&'static str] {
    match family {
        Family::Qwen3 => &["<tool_call>"],
        Family::Llama3 => &["<|python_tag|>", "{", "{\"", " {", " {\"", "{\n"],
        Family::Mistral => &["[TOOL_CALLS]", "[", "[{", " [", " [{", "[{\""],
        // Every Inkling message, an answer's too, opens with `<|message_model|>`: a call shows
        // only after the tool's name, so no first token tells it.
        Family::Inkling => &[],
    }
}

/// The token ids a call of `family` can start with: each spelling `encode` reads as one token.
pub fn starts(family: Family, encode: impl Fn(&str) -> Vec<u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = (spellings(family).iter())
        .filter_map(|s| match encode(s).as_slice() {
            &[id] => Some(id),
            _ => None,
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The gate's reading of a step's first token.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reading {
    /// The call starts outweigh the best other token.
    pub call: bool,
    /// At least how far, in nats: the log of the winning side's probability less the other's, at
    /// most `CERTAIN`.
    pub margin: f32,
}

/// Reads the first token's `top` logprobs: every call start in `starts` summed, against the best
/// other token. A side `top` does not show is bounded by the probability `top` leaves over, so the
/// margin never exceeds the true one: a call start below the top could be the rest, and the best
/// other token below the top is at most the rest and at most the last shown. `None` for no `top`.
pub fn read(starts: &[u32], top: &[(u32, f32)]) -> Option<Reading> {
    if top.is_empty() {
        return None;
    }
    let p = |lp: f32| f64::from(lp).exp();
    let shown: f64 = top.iter().map(|&(_, lp)| p(lp)).sum();
    let rest = (1.0 - shown).max(0.0);
    let call: f64 = (top.iter().filter(|(id, _)| starts.contains(id)))
        .map(|&(_, lp)| p(lp))
        .sum();
    let other = (top.iter().filter(|(id, _)| !starts.contains(id)))
        .map(|&(_, lp)| p(lp))
        .reduce(f64::max);
    let lowest = top.iter().map(|&(_, lp)| p(lp)).reduce(f64::min)?;
    let other_most = other.unwrap_or(lowest.min(rest));
    let (is_call, margin) = if call > other_most {
        (true, call.ln() - other_most.ln())
    } else {
        // The best other token is shown here: with none shown, every shown token starts a call.
        (false, other.unwrap_or(0.0).ln() - (call + rest).ln())
    };
    // Two sides of probability 0 (a -inf logprob shown) are no margin at all.
    let margin = if margin.is_nan() { 0.0 } else { margin as f32 };
    Some(Reading {
        call: is_call,
        margin: margin.clamp(-CERTAIN, CERTAIN),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkout_named_anywhere_in_a_request_is_found() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("toy");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let shown = repo.display();
        for request in [
            format!("Fix the failing test in '{shown}'."),
            format!("please fix the test in {shown}"),
            format!("can you add a README to ({shown})?"),
        ] {
            assert_eq!(named_repo(&request), Some(repo.clone()), "{request}");
        }
        // A folder that is not a checkout, and no folder.
        assert_eq!(named_repo(&format!("fix {}", tmp.path().display())), None);
        assert_eq!(named_repo("code a linked list in cpp"), None);
    }

    #[test]
    fn only_score_is_not_built() {
        for router in [Router::Off, Router::Logprob, Router::Probe] {
            assert_eq!(router.check(), Ok(()), "{router:?}");
        }
        let e = Router::Score.check().unwrap_err();
        assert!(
            e.contains("router score") && e.contains("v2 PR 13 (agent/score-router)"),
            "{e}"
        );
    }

    #[test]
    fn every_single_token_spelling_of_a_call_start_counts() {
        // A tokenizer that reads `{` and `{"` as one token each, and the rest as two.
        let encode = |s: &str| match s {
            "{" => vec![7],
            "{\"" => vec![8],
            "<|python_tag|>" => vec![9],
            "<tool_call>" => vec![5],
            "[TOOL_CALLS]" => vec![6],
            "[" => vec![4],
            _ => vec![1, 2],
        };
        assert_eq!(starts(Family::Llama3, encode), [7, 8, 9]);
        assert_eq!(starts(Family::Qwen3, encode), [5]);
        assert_eq!(starts(Family::Mistral, encode), [4, 6]);
    }

    #[test]
    fn the_margin_weighs_every_call_start_against_the_best_other_token() {
        let ln = |p: f64| p.ln() as f32;
        let near = |got: Option<Reading>, call, margin: f64| {
            let got = got.unwrap();
            assert_eq!(got.call, call, "{got:?}");
            assert!(
                (f64::from(got.margin) - margin).abs() < 1e-4,
                "{got:?} {margin}"
            );
        };
        // Words at 0.6 against a call start at 0.3: the rest, 0.1, may start calls too.
        near(
            read(&[7], &[(1, ln(0.6)), (7, ln(0.3))]),
            false,
            (0.6f64 / 0.4).ln(),
        );
        // A call start at 0.6 against words at 0.3.
        near(read(&[7], &[(7, ln(0.6)), (1, ln(0.3))]), true, 2f64.ln());
        // Two starts (`{` and `<|python_tag|>`) count together, against the best other token
        // below them: at most the rest, 0.1.
        near(
            read(&[7, 9], &[(7, ln(0.5)), (9, ln(0.4))]),
            true,
            9f64.ln(),
        );
        // Two words and no start shown: a call is at most the rest.
        near(
            read(&[7], &[(1, ln(0.5)), (2, ln(0.45))]),
            false,
            (0.5f64 / 0.05).ln(),
        );
        // A certain token is capped, not infinite.
        let sure = read(&[7], &[(1, 0.0), (2, -200.0)]).unwrap();
        assert_eq!((sure.call, sure.margin), (false, CERTAIN));
        assert_eq!(read(&[7], &[]), None);
    }
}
