//! What the user's plain words ask of the turn's tools, read by Rust so the decision does not
//! swing with the model (v3 design D2, D3 step 1 and D6).

use std::path::{Path, PathBuf};

use crate::{
    skills::{research::Effort, web_answer},
    tool::{self, ToolCtx, ToolError},
};

/// What the user's "no tools" turns off, and how long it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limit {
    /// Only the web: the file and memory tools stay on.
    pub web: bool,
    /// Until the user asks for tools again, not this message only.
    pub kept: bool,
}

/// Whether `text` says one of `phrases`, as whole words in any case; apostrophes, curly or not,
/// split words, so "what's my" reads "what s my" and a phrase is written that way.
pub(crate) fn said(text: &str, phrases: &[&str]) -> bool {
    let words = spoken(text);
    phrases.iter().any(|p| words.contains(&format!(" {p} ")))
}

/// `text`'s words as `said` reads them, in lower case, with a space before and after each.
fn spoken(text: &str) -> String {
    let words: Vec<String> = (text.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    format!(" {} ", words.join(" "))
}

/// Words that keep the turn's tools off.
const NO_TOOLS: [&str; 8] = [
    "without tools",
    "without any tools",
    "without using tools",
    "without using any tools",
    "no tools",
    "don t use tools",
    "do not use tools",
    "don t use any tools",
];

/// Words that keep the web off: "don't search the web; read ~/notes/a.md" still reads.
const NO_WEB: [&str; 17] = [
    "don t search",
    "dont search",
    "do not search",
    "without searching",
    "no web search",
    "don t browse",
    "no internet",
    "don t use the internet",
    "do not use the internet",
    "without the internet",
    "don t look it up",
    "do not look it up",
    "dont look it up",
    "no need to look it up",
    "no need to search",
    "don t need to search",
    "don t google",
];

/// Words that make a "no tools" last past its message.
const KEEP: [&str; 5] = [
    "from now on",
    "for the rest of this chat",
    "for the rest of the chat",
    "for the rest of this conversation",
    "until i say",
];

/// Words that ask for the tools again.
const LIFT: [&str; 6] = [
    "search again",
    "tools again",
    "you can search",
    "you can use tools",
    "tools are ok",
    "tools back on",
];

/// Words that ask for a web search; "look up" and "web search" alone name a lookup or the tool.
const WEB: [&str; 5] = [
    "search the web",
    "search online",
    "search the internet",
    "look it up",
    "google it",
];

/// Words that point at the user's own things or at the chat, which no search finds.
const OWN: [&str; 16] = [
    "my",
    "mine",
    "i said",
    "you said",
    "this",
    "these",
    "above",
    "below",
    "file",
    "files",
    "notes",
    "folder",
    "document",
    "essay",
    "paragraph",
    "sentence",
];

/// Words of a question about the clock or the calendar, which needs no search.
const CLOCK: [&str; 7] = [
    "current time",
    "what time",
    "time is it",
    "today s date",
    "the date today",
    "what is the date",
    "what day is it",
];

/// The most words of a message that is a question of its own.
const SHORT_WORDS: usize = 30;

/// Whether `user` reads as a question on its own: one line, not pasted, at most `SHORT_WORDS`
/// words.
pub(crate) fn short(user: &str) -> bool {
    let user = user.trim();
    !user.contains('\n') && !user.contains("```") && user.split_whitespace().count() <= SHORT_WORDS
}

/// Whether `user`, a chat's first message, asks the web for the latest of something: a short
/// question about the latest (`web_answer::asks_latest`) that names no path, none of the user's
/// own things or the chat (`OWN`), and not the clock (`CLOCK`). "My current notes" stays the
/// files', "the last paragraph" the chat's.
fn latest_on_web(user: &str) -> bool {
    short(user)
        && web_answer::asks_latest(user)
        && user_paths(user).is_empty()
        && !said(user, &OWN)
        && !said(user, &CLOCK)
}

/// Whether `user`'s own words ask for the web ("look it up", "search the web").
pub(crate) fn asks_web(user: &str) -> bool {
    said(user, &WEB)
}

/// `user` without the words that ask for a search ("look it up"), which a search engine would
/// read as part of the subject; empty when they are all there is.
pub(crate) fn web_question(user: &str) -> String {
    let mut text = user.to_owned();
    for phrase in WEB {
        // ASCII lower case keeps each byte in its place.
        while let Some(at) = text.to_ascii_lowercase().find(phrase) {
            text.replace_range(at..at + phrase.len(), " ");
        }
    }
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = text.trim_matches(|c: char| !c.is_alphanumeric() && c != '?');
    let text = ["for ", "about "]
        .iter()
        .find_map(|w| text.strip_prefix(w))
        .unwrap_or(text);
    text.trim().to_owned()
}

/// The user's "no tools" or "no web search" in `user`, and how long it holds.
pub(crate) fn limit(user: &str) -> Option<Limit> {
    let web = match (said(user, &NO_TOOLS), said(user, &NO_WEB)) {
        (true, _) => false,
        (false, true) => true,
        (false, false) => return None,
    };
    Some(Limit {
        web,
        kept: said(user, &KEEP),
    })
}

/// Whether `user` asks for tools again after a kept limit: for them, or with a tool `signal`
/// (`force`), and not for no tools in the same breath.
pub(crate) fn lifts(user: &str, signal: bool) -> bool {
    limit(user).is_none() && (signal || said(user, &LIFT))
}

/// Words that tell the model it is wrong about something that changes: only a search settles it.
const DISPUTE: [&str; 14] = [
    "it was released",
    "it was announced",
    "it s out now",
    "it is out now",
    "already came out",
    "already released",
    "already been released",
    "already announced",
    "that s out of date",
    "that is out of date",
    "you re out of date",
    "you are out of date",
    "that s outdated",
    "that is outdated",
];

/// Words of a question about how to do something, whose paths are not files to read.
const HOW_TO: [&str; 4] = ["how do i", "how to", "how can i", "how should i"];

/// Hosts on the user's own machine, which web_answer cannot open.
const LOCAL: [&str; 4] = ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"];

/// Words that point at a paper the chat has read.
const PAPER: [&str; 6] = [
    "the paper",
    "this paper",
    "that paper",
    "the papers",
    "these papers",
    "those papers",
];

/// The things of the user's that live in files, after "my" or "our": "what do my notes say".
/// "Doc" (a doctor too), "downloads" and "folder" (tech help), and "my paper" ("help me write my
/// paper") are left out.
const MY_FILES: [&str; 13] = [
    "file",
    "files",
    "note",
    "notes",
    "document",
    "documents",
    "pdf",
    "pdfs",
    "papers",
    // The M5's everyday misses (31 Oct): "my downloads folder", "my journal", "my thesis".
    "downloads",
    "folder",
    "journal",
    "thesis",
];

/// Words that point at what was attached: "the paper" alone is anyone's paper.
const THIS_FILE: [&str; 12] = [
    // "summarize the contract pdf i attached" (the M5's random-person afternoon, 31 Oct).
    "i attached",
    "i sent you",
    "i uploaded",
    "this pdf",
    "this document",
    "these pdfs",
    "these documents",
    "these files",
    "these notes",
    "the attached",
    "attached file",
    "attached files",
];

/// Words that ask what the files hold, one of which `about_files` needs.
const READS: [&str; 18] = [
    "say",
    "says",
    "read",
    "summarize",
    "summarise",
    "summary",
    "mention",
    "mentions",
    "in my",
    "from my",
    "through my",
    "of my",
    "which of",
    "check",
    "search",
    "find",
    "look",
    "quote",
];

/// Words of a question about an app, a device or a lost file, not about what files hold.
const NOT_READS: [&str; 12] = [
    "app", "apps", "won t", "wont", "deleted", "delete", "recover", "format", "crash", "crashing",
    "back up", "sync",
];

/// Whether `user` plainly asks what the user's files hold: "what do my notes say", "summarize my
/// files", "which of my pdfs", "what does this document say"; not "my notes app keeps crashing".
pub(crate) fn about_files(user: &str) -> bool {
    let words = spoken(user);
    let mine = ["my", "our"].iter().any(|o| {
        MY_FILES
            .iter()
            .any(|t| words.contains(&format!(" {o} {t} ")))
    });
    (mine || said(user, &THIS_FILE)) && said(user, &READS) && !said(user, &NOT_READS)
}

/// Words too common to name a paper by its file: "help me write a paper" is not paper.pdf.
const COMMON: [&str; 8] = [
    "papers",
    "document",
    "documents",
    "download",
    "downloads",
    "readme",
    "untitled",
    "scan",
];

/// Whether a file's stem as `spoken` words names one paper: two words or more, or one of six
/// letters or more that is not `COMMON` ("resnet", not "paper" or "notes").
fn names_a_paper(stem: &str) -> bool {
    stem.contains(' ') || (stem.len() >= 6 && !COMMON.contains(&stem))
}

/// The answer to a question about the user's files in a chat with nothing attached, which no
/// model step could answer.
pub const NO_FOLDER: &str =
    "I can't read your files in this chat yet. Attach a folder or the files, then ask again.";

/// What the user's words hold the turn's first step to (`force`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Force {
    /// A call of this tool.
    Tool(&'static str),
    /// A call of any tool.
    Any,
    /// No model step: this text is the answer.
    Fixed(String),
}

/// `/research [standard|deep] <request>` (Research mode, v3 R4): the effort, standard when the
/// next word names none, and the request, which may be empty.
pub(crate) fn research(user: &str) -> Option<(Effort, &str)> {
    let user = user.trim_start();
    let rest = (user.get(..9))
        .filter(|c| c.eq_ignore_ascii_case("/research"))
        .map(|_| &user[9..])
        .filter(|r| r.is_empty() || r.starts_with(char::is_whitespace))?
        .trim();
    let (first, after) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    Some(match clap::ValueEnum::from_str(first, true) {
        Ok(effort) => (effort, after.trim()),
        Err(_) => (Effort::Standard, rest),
    })
}

/// The words of `user` as written, less the quotes and punctuation around them.
pub(crate) fn words(user: &str) -> impl Iterator<Item = &str> {
    user.split_whitespace().map(|w| {
        w.trim_start_matches(['(', '[', '"', '\'', '`'])
            .trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']', '"', '\'', '`'])
    })
}

/// The paths in `user`: words that start `~/`, or absolute paths with a folder in them (so
/// `/research` and `and/or` are not).
// ponytail: a path with a space in it is cut there; read quoted paths when a chat shows one
pub(crate) fn user_paths(user: &str) -> Vec<&str> {
    words(user)
        .filter(|w| {
            w.starts_with("~/")
                || (Path::new(w).is_absolute() && Path::new(w).components().count() > 2)
        })
        .collect()
}

/// What `user`'s plain words hold the turn's first step to, in this order: a folder or file in
/// the home folder that is there and outside the allowed folders gets the fixed text, unless the
/// message is pasted text or a how-to question, and so does a question about the user's files
/// (`about_files`) in a chat with nothing attached (`NO_FOLDER`); a web search (not "why did you search"), or a
/// dispute of a fact that changes, web_answer; a sentence that asks to "remember that",
/// remember; a PDF path, or a URL of a host on the web, read_papers, which opens the link
/// (web_answer would search instead); another path in the allowed folders, any call that reads;
/// a chat's `first` message that asks the web for the latest (`latest_on_web`), web_answer;
/// "the paper", or the name of a paper read before (`papers`, the files' stems), read_papers;
/// a question about the user's files (`about_files`), any call.
/// Words such as "my", "research" or "report" force nothing. A path outside the home folder
/// (`/usr/local/bin`) or hidden in it (`~/.zshrc`) is about the system, not the user's files, so
/// it forces nothing either; nor does a word that names nothing, such as `~/My` cut from
/// `~/My Notes` or `app.py:3:5` from a compiler error.
pub(crate) fn force(user: &str, ctx: &ToolCtx, papers: &[String], first: bool) -> Option<Force> {
    let paths = user_paths(user);
    let inside: Vec<&str> = (paths.iter().copied())
        .filter(|p| !matches!(ctx.resolve(p, false), Err(ToolError::Denied(_))))
        .collect();
    let home = tool::expand_home("~").ok();
    let personal = |p: &&str| {
        let hidden = p
            .split(['/', '\\'])
            .any(|s| s.starts_with('.') && s != "." && s != "..");
        let expanded = tool::expand_home(p).ok();
        let mine = |e: PathBuf| home.as_ref().is_some_and(|h| e.starts_with(h)) && e.exists();
        !hidden && expanded.is_some_and(mine)
    };
    let pasted = user.trim().contains('\n') || user.contains("```");
    let url = words(user).any(|w| {
        let w = w.to_lowercase();
        let host = ["http://", "https://", "www."]
            .iter()
            .find_map(|s| w.strip_prefix(s));
        host.is_some_and(|h| !h.is_empty() && !LOCAL.iter().any(|l| h.starts_with(l)))
    });
    let plain = !pasted && !said(user, &HOW_TO);
    let outside = (paths.iter().copied())
        .filter(|p| !inside.contains(p))
        .find(personal)
        .filter(|_| plain);
    let read_before = !papers.is_empty();
    let paper = (read_before && said(user, &PAPER))
        || (papers.iter().map(|p| spoken(p)))
            .any(|p| names_a_paper(p.trim()) && said(user, &[p.trim()]));
    if ctx.allowed.is_empty() && (outside.is_some() || (about_files(user) && plain)) {
        Some(Force::Fixed(NO_FOLDER.into()))
    } else if let Some(p) = outside {
        Some(Force::Fixed(format!(
            "{p} is not attached to this chat, so I can't read it. Attach it, then ask again."
        )))
    } else if (said(user, &WEB) && !said(user, &["why did you"])) || said(user, &DISPUTE) {
        Some(Force::Tool("web_answer"))
    } else if asks_to_remember(user) {
        Some(Force::Tool("remember"))
    } else if url || inside.iter().any(|p| p.to_lowercase().ends_with(".pdf")) {
        Some(Force::Tool("read_papers"))
    } else if !inside.is_empty() {
        Some(Force::Any)
    } else if first && latest_on_web(user) {
        Some(Force::Tool("web_answer"))
    } else if paper {
        Some(Force::Tool("read_papers"))
    } else if about_files(user) && plain {
        Some(Force::Any)
    } else {
        None
    }
}

/// What a plain reply says that a tool it may call would have settled (D5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hit {
    /// It says it cannot reach the web.
    Denied,
    /// It says it cannot read the user's files.
    DeniedFiles,
    /// It says something has not happened, with no search behind it.
    UnsearchedNo,
    /// It gives an event's result dated in or after the model's cutoff year.
    AfterCutoff,
}

impl Hit {
    /// Why the reply was withdrawn, for the audit log.
    pub(crate) fn why(self) -> &'static str {
        match self {
            Hit::Denied | Hit::DeniedFiles => "the reply denied a tool it has",
            Hit::UnsearchedNo => "the reply said no without a search",
            Hit::AfterCutoff => "the reply gave a result after the model's cutoff without a search",
        }
    }
}

/// A reply that says it cannot reach the web, after "I", "I'm" or "I am" (`says_i`).
const DENIES_WEB: [&str; 26] = [
    "can t access the internet",
    "cannot access the internet",
    "unable to access the internet",
    "not able to access the internet",
    "don t have access to the internet",
    "do not have access to the internet",
    "have no internet access",
    "don t have internet access",
    "do not have internet access",
    "can t browse",
    "cannot browse",
    "unable to browse",
    "don t have browsing",
    "do not have browsing",
    "have no real time access",
    "have no real time data",
    "have no real time information",
    "don t have real time",
    "do not have real time",
    "can t search the web",
    "cannot search the web",
    "can t search the internet",
    "cannot search the internet",
    "don t have access to online",
    "do not have access to online",
    "can t look it up",
];

/// A reply that says it cannot read the user's files, after "I" as `DENIES_WEB` is.
const DENIES_FILES: [&str; 14] = [
    "can t read pdf",
    "can t read pdfs",
    "cannot read pdf",
    "cannot read pdfs",
    "unable to read pdf",
    "unable to read pdfs",
    "can t read files",
    "cannot read files",
    "can t open files",
    "cannot open files",
    "can t access your files",
    "cannot access your files",
    "don t have access to your files",
    "do not have access to your files",
];

/// A reply that says something has not happened, or that its knowledge stops: a search settles
/// both. "Does not exist" and "has not been released" are not here: coding answers say them of
/// files and locks.
const NOT_YET: [&str; 15] = [
    "has not announced",
    "hasn t announced",
    "has not been announced",
    "hasn t been announced",
    "have not announced",
    "haven t announced",
    "has not been officially",
    "as of my knowledge",
    "as of my last",
    "my knowledge cutoff",
    "i have no information about",
    "i don t have information about",
    "i don t have any information about",
    "i do not have information about",
    "i do not have any information about",
];

/// An event's result, which next to a year the model cannot know needs a search; "wins" and
/// "beat" are forecasts and deadlines as often.
const RESULT: [&str; 5] = ["won", "released", "announced", "launched", "was elected"];

/// What the model says about the user, which no search finds.
const ABOUT_YOU: [&str; 2] = ["information about you", "information about your"];

/// The first check a plain reply's `text` trips, for a step that may call `tools` on a model whose
/// knowledge ends in `cutoff`, a year (D5): the web checks need web_answer, the file check
/// read_papers or read_file. A year alone ("plan my 2026 budget") trips nothing, nor does one
/// after this year, which is a plan or a forecast.
pub(crate) fn check_reply(text: &str, cutoff: Option<i32>, tools: &[&str]) -> Option<Hit> {
    let web = tools.contains(&"web_answer");
    let files = tools
        .iter()
        .any(|t| ["read_papers", "read_file"].contains(t));
    let now: Option<i32> = larch::bench::utc_now()[..4].parse().ok();
    let known = cutoff.zip(now).map(|(c, n)| c..=n);
    let unknowable = |sentence: &str| {
        let year = |w: &str| {
            w.len() == 4
                && w.parse()
                    .is_ok_and(|y: i32| known.as_ref().is_some_and(|k| k.contains(&y)))
        };
        // "won't" reads "won t", which is no result.
        let words = spoken(sentence).replace(" won t ", " ");
        let result = RESULT.iter().any(|p| words.contains(&format!(" {p} ")));
        sentence.split(|c: char| !c.is_ascii_digit()).any(year) && result
    };
    if web && says_i(text, &DENIES_WEB) {
        Some(Hit::Denied)
    } else if files && says_i(text, &DENIES_FILES) {
        Some(Hit::DeniedFiles)
    } else if web && said(text, &NOT_YET) && !said(text, &ABOUT_YOU) {
        Some(Hit::UnsearchedNo)
    } else if web && text.split(['.', '!', '?', '\n']).any(unknowable) {
        Some(Hit::AfterCutoff)
    } else {
        None
    }
}

/// Whether `text` says one of `phrases` of the model itself: after "I", "I'm" or "I am".
fn says_i(text: &str, phrases: &[&str]) -> bool {
    let words = spoken(text);
    let i = ["i", "i m", "i am"];
    (i.iter()).any(|i| (phrases.iter()).any(|p| words.contains(&format!(" {i} {p} "))))
}

/// Whether a sentence of `user` asks to save a fact: it opens "(please) remember that" and is no
/// question, or opens "can you (please) remember that".
fn asks_to_remember(user: &str) -> bool {
    user.split_inclusive(['.', '!', '?', '\n']).any(|s| {
        let words = spoken(s);
        let asked = ["can you", "could you", "will you", "would you"]
            .iter()
            .find_map(|a| words.strip_prefix(&format!(" {a}")));
        let rest = asked.unwrap_or(&words);
        let rest = rest.strip_prefix(" please").unwrap_or(rest);
        rest.starts_with(" remember that ") && (asked.is_some() || !s.trim_end().ends_with('?'))
    })
}

/// The outside (MCP) server a message names, as `outside` reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Named {
    /// Its tools' `<server>__` prefix; empty for every outside tool ("use the mcp" in a chat
    /// with several servers).
    pub prefix: String,
    /// The message asks for a call: "use deepwiki", "ping deepwiki", "ask my calendar", "the
    /// deepwiki mcp".
    pub call: bool,
}

/// Words up to two before a server's name that ask to call it ("use teh mcp", "check my
/// calendar").
const CALL_BEFORE: [&str; 15] = [
    "use", "using", "ask", "call", "query", "try", "ping", "check", "via", "with", "search",
    "open", "from", "through", "test",
];

/// Words right after a server's name that ask to call it: "the deepwiki mcp".
const CALL_AFTER: [&str; 5] = ["mcp", "tool", "tools", "server", "connector"];

/// Words up to four before a server's name, in its clause, that refuse it: "do not use
/// deepwiki", "don't ask my calendar", "answer without the mcp".
const REFUSE_BEFORE: [&str; 8] = [
    "not", "don", "dont", "never", "without", "avoid", "skip", "stop",
];

/// The outside server of `servers` (the `<server>` part of its tools' names, `mcp::tool_name`)
/// that `user` names as whole words, a `_` in the name read as a space, or for "mcp" with no
/// server named: the one server, or all of them. A small model's intent label and its own
/// choice both missed "Use deepwiki and teach me about ollama" (Qwen3 1.7B, 10 Nov): its label
/// was code, which shows no tool, and with the tools shown it asked "Would you like me to look
/// it up?". Naming a server is the user's plain word for its tools. A clause that refuses it
/// ("Do not use deepwiki; answer from memory") does not name it, and a question of how to use
/// it ("How do I use deepwiki?") names it without asking for a call.
pub(crate) fn outside(user: &str, servers: &[&str]) -> Option<Named> {
    let how = said(user, &HOW_TO);
    let clauses: Vec<Vec<String>> = (user.split(['.', ';', '!', '?', ',', '\n']))
        .map(|c| spoken(c).split_whitespace().map(str::to_owned).collect())
        .filter(|w: &Vec<String>| !w.is_empty())
        .collect();
    // The first clause that names `name`, and where.
    let find = |name: &[&str]| {
        clauses.iter().find_map(|words| {
            let at = (words.windows(name.len()))
                .position(|w| w.iter().map(String::as_str).eq(name.iter().copied()))?;
            Some((words, at))
        })
    };
    let calls = |words: &[String], at: usize, len: usize| {
        !how && ((words[at.saturating_sub(2)..at].iter())
            .any(|w| CALL_BEFORE.contains(&w.as_str()))
            || (words.get(at + len)).is_some_and(|w| CALL_AFTER.contains(&w.as_str())))
    };
    let refused = |words: &[String], at: usize| {
        (words[at.saturating_sub(4)..at].iter()).any(|w| REFUSE_BEFORE.contains(&w.as_str()))
    };
    for server in servers {
        let name: Vec<&str> = server.split('_').filter(|w| !w.is_empty()).collect();
        if name.is_empty() {
            continue;
        }
        if let Some((words, at)) = find(&name) {
            if refused(words, at) {
                continue;
            }
            return Some(Named {
                prefix: format!("{server}__"),
                call: calls(words, at, name.len()),
            });
        }
    }
    if servers.is_empty() {
        return None;
    }
    let (words, at) = find(&["mcp"]).or_else(|| find(&["mcps"]))?;
    if refused(words, at) {
        return None;
    }
    Some(Named {
        prefix: match servers {
            [one] => format!("{one}__"),
            _ => String::new(),
        },
        call: calls(words, at, 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_named_outside_server_is_read_with_whether_to_call_it() {
        let one = ["deepwiki"];
        let named = |user, servers: &[&str]| outside(user, servers);
        let call = |prefix: &str| {
            Some(Named {
                prefix: prefix.into(),
                call: true,
            })
        };
        // The user's own messages to the app (10 Nov).
        assert_eq!(
            named("Use deepwiki and teach me about ollama", &one),
            call("deepwiki__")
        );
        assert_eq!(named("Please ping deepwiki", &one), call("deepwiki__"));
        assert_eq!(named("use teh mcp", &one), call("deepwiki__"));
        assert_eq!(
            named("Use deepwiki, and give me a list of repos", &one),
            call("deepwiki__")
        );
        // Named, not called: the turn keeps its tools, and the model decides.
        assert_eq!(
            named("What is DeepWiki?", &one),
            Some(Named {
                prefix: "deepwiki__".into(),
                call: false
            })
        );
        // A name of several words, and "mcp" after it.
        let two = ["deepwiki", "google_calendar"];
        assert_eq!(
            named("what's on my Google Calendar mcp today", &two),
            call("google_calendar__")
        );
        assert_eq!(
            named("check my google calendar", &two),
            call("google_calendar__")
        );
        // "mcp" alone with several servers is every outside tool.
        assert_eq!(named("use the mcp", &two), call(""));
        // No server named, or a name inside another word, or no servers at all.
        assert_eq!(named("How does qwen train their models", &one), None);
        assert_eq!(named("deepwikipedia is not it", &one), None);
        assert_eq!(named("use the mcp", &[]), None);
        // Refused in its clause: not named, so no call is held to it.
        assert_eq!(
            named("Do not use deepwiki; answer from memory.", &one),
            None
        );
        assert_eq!(named("Don't ask deepwiki, just answer", &one), None);
        assert_eq!(named("Please never use the mcp for this", &one), None);
        assert_eq!(named("answer without deepwiki", &one), None);
        // A refusal in another clause leaves a request standing.
        assert_eq!(named("No, use deepwiki", &one), call("deepwiki__"));
        // How to use it: named, the model decides.
        assert_eq!(
            named("How do I use deepwiki?", &one),
            Some(Named {
                prefix: "deepwiki__".into(),
                call: false
            })
        );
    }

    #[test]
    fn a_research_command_names_its_effort_and_its_request() {
        use crate::skills::research::Effort::{Deep, Standard};
        let cases = [
            (
                "/research deep the history of RISC-V",
                Some((Deep, "the history of RISC-V")),
            ),
            (
                "  /RESEARCH   Standard   ferries\n\nand piers ",
                Some((Standard, "ferries\n\nand piers")),
            ),
            (
                "/research deeply about ferries",
                Some((Standard, "deeply about ferries")),
            ),
            ("/research Deep", Some((Deep, ""))),
            ("/research", Some((Standard, ""))),
            ("/researcher ferries", None),
            ("please /research ferries", None),
            ("research ferries", None),
        ];
        for (user, want) in cases {
            assert_eq!(research(user), want, "{user:?}");
        }
    }

    #[test]
    fn said_matches_whole_words_in_any_case() {
        assert!(said("What's MY name?", &["my"]));
        assert!(said("don\u{2019}t search", &["don t search"]));
        assert!(!said("mystery", &["my"]));
        assert!(!said("", &["my"]));
    }

    #[test]
    fn no_tools_holds_for_the_message_or_until_lifted() {
        let off = |web, kept| Some(Limit { web, kept });
        for turn in [
            "without using any tools, what is 2+2?",
            "Without tools: who wrote Dune?",
            "answer with no tools",
            "no tools, don't search",
        ] {
            assert_eq!(limit(turn), off(false, false), "{turn}");
        }
        for web in [
            "don't search, just tell me",
            "do not search the web for this",
            "without searching, what's the capital of Peru?",
            "no internet please",
            "don't look it up",
            "Don't search the web; read ~/notes/groceries.md",
            "no need to look it up, what is 2+2?",
            "don't google it",
            "dont look it up",
            "I don't need to search, just tell me",
        ] {
            assert_eq!(limit(web), off(true, false), "{web}");
        }
        assert_eq!(
            limit("for the rest of this chat, no tools"),
            off(false, true)
        );
        for kept in [
            "from now on, don't search the web",
            "don't use the internet for the rest of the chat",
            "From now on don't search the web; you may still read my local files",
        ] {
            assert_eq!(limit(kept), off(true, true), "{kept}");
        }
        for free in [
            "look it up",
            "search the web for the latest F1 result",
            "what tools do you have?",
            "the search was slow",
        ] {
            assert_eq!(limit(free), None, "{free}");
        }
    }

    #[test]
    fn asking_for_tools_or_a_search_lifts_a_kept_limit() {
        for lift in ["ok, you can search again", "use tools again"] {
            assert!(lifts(lift, false), "{lift}");
        }
        assert!(lifts("search the web for it", true));
        assert!(!lifts("and of Italy?", false));
        assert!(!lifts("don't look it up", true));
    }

    #[test]
    fn a_first_short_question_about_the_latest_is_the_webs_and_one_about_the_chat_is_not() {
        let home = tool::expand_home("~").unwrap();
        let ctx = ToolCtx {
            allowed: vec![home],
            data_dir: tempfile::tempdir().unwrap().path().into(),
            source: String::new(),
        };
        let web = Some(Force::Tool("web_answer"));
        assert_eq!(
            force("who won the latest UT Austin game", &ctx, &[], true),
            web
        );
        for own in [
            "rewrite the last paragraph: we met at noon and talked for an hour",
            "what was the last thing I said?",
            "what's the current time?",
            "the latest version of my essay is below",
            "summarize the latest entries in ~/notes/log.md",
            "who won the latest game?\nand the one before?",
        ] {
            assert_ne!(force(own, &ctx, &[], true), web, "{own}");
        }
        // A follow-up may be about the chat's own answer.
        assert_eq!(force("explain the last point", &ctx, &[], false), None);
        assert_eq!(
            force("who won the latest UT Austin game", &ctx, &[], false),
            None
        );
    }

    #[test]
    fn only_the_users_words_ask_for_the_web() {
        assert!(asks_web("what is this? look it up"));
        assert!(asks_web("Search the web for this error"));
        assert!(!asks_web("what is this?"));
        assert!(!asks_web("what does the website in the picture say?"));
    }

    #[test]
    fn a_web_question_drops_the_words_that_ask_for_a_search() {
        for (user, want) in [
            (
                "who won the last F1 race? look it up",
                "who won the last F1 race?",
            ),
            (
                "Search the web for the latest Zest release",
                "the latest Zest release",
            ),
            (
                "look it up: the price of a Pixel 11",
                "the price of a Pixel 11",
            ),
            ("google it", ""),
        ] {
            assert_eq!(web_question(user), want, "{user}");
        }
    }

    #[test]
    fn user_paths_are_home_or_absolute_paths_with_a_folder() {
        let dir = tempfile::tempdir().unwrap();
        let notes = dir.path().join("notes").join("a.md");
        let notes = notes.to_str().unwrap();
        let user = format!("read (~/papers-math/a.pdf), \"~/x\" and {notes}.");
        assert_eq!(user_paths(&user), ["~/papers-math/a.pdf", "~/x", notes]);
        assert!(user_paths("/research and/or ~ //").is_empty());
    }

    #[test]
    fn force_reads_only_clear_words() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx {
            allowed: vec![dir.path().into()],
            data_dir: dir.path().into(),
            source: String::new(),
        };
        let inside = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();
        // A folder in the home folder that is there: a word that names nothing gets no fixed text.
        let home = tool::expand_home("~").unwrap();
        let there = (tempfile::Builder::new().prefix("larch-test-"))
            .tempdir_in(home)
            .unwrap();
        let name = format!("~/{}", there.path().file_name().unwrap().to_str().unwrap());
        let app = there.path().join("app.py");
        std::fs::write(&app, "").unwrap();
        let app = app.to_str().unwrap();
        let web = Some(Force::Tool("web_answer"));
        let outside = format!(
            "{name} is not attached to this chat, so I can't read it. Attach it, then ask again."
        );
        let remember = Some(Force::Tool("remember"));
        let rows = [
            // No fixed text for a cut path; its words still ask about the user's notes.
            ("summarize ~/My Notes/a.md", false, Some(Force::Any)),
            (
                &format!("summarize the papers in {name}"),
                false,
                Some(Force::Fixed(outside.clone())),
            ),
            (
                &format!("look it up in {name}"),
                false,
                Some(Force::Fixed(outside.clone())),
            ),
            (
                &format!("Traceback (most recent call last):\n  File \"{app}\", line 3"),
                false,
                None,
            ),
            (&format!("error at {app}:3:5"), false, None),
            (&format!("how do I add {name} to my PATH"), false, None),
            ("why did you search the web for that?", false, None),
            ("how do I look up a key in a HashMap", false, None),
            ("is web search on?", false, None),
            ("what does the http:// prefix mean?", false, None),
            ("is http://localhost:3000 up?", false, None),
            ("my server at https://127.0.0.1:8080 is slow", false, None),
            ("why should I remember that the API changed?", false, None),
            ("Can you remember that I like tea?", false, remember.clone()),
            (
                "Please remember that my cat is Tom.",
                false,
                remember.clone(),
            ),
            (
                "what is on https://example.com/a?",
                false,
                Some(Force::Tool("read_papers")),
            ),
            (
                "summarize https://www.paulgraham.com/startupideas.html",
                false,
                Some(Force::Tool("read_papers")),
            ),
            (
                "is www.example.com down",
                false,
                Some(Force::Tool("read_papers")),
            ),
            (
                "search the web for the latest F1 result",
                false,
                web.clone(),
            ),
            ("look it up", false, web.clone()),
            ("Google it", false, web.clone()),
            ("it was released last week", false, web.clone()),
            ("that's out of date", false, web.clone()),
            ("remember that I like tea", false, remember.clone()),
            (
                &format!("what is in {}", inside("a.pdf")),
                false,
                Some(Force::Tool("read_papers")),
            ),
            (
                &format!("list {}", inside("notes/b")),
                false,
                Some(Force::Any),
            ),
            (
                "what does the paper say about the proof?",
                true,
                Some(Force::Tool("read_papers")),
            ),
            ("what does the paper say about the proof?", false, None),
            // With a folder attached, a question about the user's files holds the step to a call.
            ("what do my notes say about stew?", false, Some(Force::Any)),
            ("my notes app keeps crashing on my iphone", false, None),
            ("help me write my paper on the war", false, None),
            ("what are the notes of a C major chord?", false, None),
            ("how do I report a bug", false, None),
            ("what is my best laptop option", false, None),
            ("I am doing research on X", false, None),
            ("do you remember that song?", false, None),
            ("how do I fix outdated npm packages", false, None),
            ("how do I add /usr/local/bin to my PATH", false, None),
            ("what does ~/.zshrc do", false, None),
            ("cats and/or dogs, see /research", true, None),
        ];
        for (user, read, want) in rows {
            let papers = if read {
                vec!["kepler".into()]
            } else {
                Vec::new()
            };
            assert_eq!(force(user, &ctx, &papers, false), want, "{user}");
        }
        // A paper read before is named by its file's stem.
        let papers = ["resnet".to_owned(), "attention-is-all-you-need".to_owned()];
        let paper = Some(Force::Tool("read_papers"));
        assert_eq!(
            force("why did resnet pick 152 layers", &ctx, &papers, false),
            paper
        );
        let asked = "what does attention is all you need say about heads";
        assert_eq!(force(asked, &ctx, &papers, false), paper);
        assert_eq!(
            force("what is a residual network", &ctx, &papers, false),
            None
        );
        // A common or short stem names no paper.
        let papers = [
            "paper".to_owned(),
            "Documents".to_owned(),
            "notes".to_owned(),
        ];
        for user in [
            "help me write a paper on climate change",
            "what documents do I need for a visa?",
            "take notes on this",
        ] {
            assert_eq!(force(user, &ctx, &papers, false), None, "{user}");
        }
    }

    /// bench/agent/no-folders.jsonl: with nothing attached, a question about the user's files gets
    /// the fixed answer, and no other row does.
    #[test]
    fn with_nothing_attached_a_question_about_files_gets_the_fixed_answer() {
        let home = tool::expand_home("~").unwrap();
        let ctx = ToolCtx {
            allowed: Vec::new(),
            data_dir: home.clone(),
            source: String::new(),
        };
        let rows = include_str!("../../../bench/agent/no-folders.jsonl");
        let mut fixed = 0;
        for line in rows.lines() {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            let (id, user) = (row["id"].as_str().unwrap(), row["user"].as_str().unwrap());
            let want = (row["expect"]["answer_must_contain"].as_array())
                .is_some_and(|a| a.iter().any(|w| w == "attach a folder"));
            let got = force(user, &ctx, &[], false);
            assert_eq!(
                got == Some(Force::Fixed(NO_FOLDER.into())),
                want,
                "{id}: {got:?}"
            );
            fixed += usize::from(want);
        }
        assert_eq!(fixed, 10);
        // A path the user pastes gets it too.
        let there = (tempfile::Builder::new().prefix("larch-test-"))
            .tempdir_in(&home)
            .unwrap();
        let name = format!("~/{}", there.path().file_name().unwrap().to_str().unwrap());
        let user = format!("summarize the papers in {name}");
        assert_eq!(
            force(&user, &ctx, &[], false),
            Some(Force::Fixed(NO_FOLDER.into()))
        );
        // A pasted paper URL is a web job: read_papers opens it (H3).
        assert_eq!(
            force(
                "summarize https://arxiv.org/abs/1706.03762",
                &ctx,
                &[],
                false
            ),
            Some(Force::Tool("read_papers"))
        );
    }

    /// bench/agent/everyday.jsonl with nothing attached: the my-files rows the model sent to
    /// recall or read_papers on the M5 (31 Oct) get the fixed answer, and no row of another
    /// category does.
    #[test]
    fn everyday_rows_get_the_fixed_answer_only_about_the_users_files() {
        let ctx = ToolCtx {
            allowed: Vec::new(),
            data_dir: tool::expand_home("~").unwrap(),
            source: String::new(),
        };
        let rows = include_str!("../../../bench/agent/everyday.jsonl");
        let mut fixed = Vec::new();
        for line in rows.lines() {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            let (id, user) = (row["id"].as_str().unwrap(), row["user"].as_str().unwrap());
            if force(user, &ctx, &[], false) == Some(Force::Fixed(NO_FOLDER.into())) {
                assert_eq!(row["category"], "myfiles", "{id}: {user}");
                fixed.push(id.to_owned());
            }
        }
        for id in ["008", "010", "012"] {
            let id = format!("everyday-myfiles-{id}");
            assert!(fixed.contains(&id), "{id}: {fixed:?}");
        }
        // The random-person afternoon's (31 Oct), with nothing attached.
        for user in [
            "check my notes for the wifi password",
            "look thru my downloads and find my resume",
            "summarize the contract pdf i attached",
        ] {
            assert_eq!(
                force(user, &ctx, &[], false),
                Some(Force::Fixed(NO_FOLDER.into())),
                "{user}"
            );
        }
    }

    /// The bench's routing rows (`bench/agent/routing/dev.jsonl`, the user's last message): a
    /// `none` row is never forced to a tool, a row that takes a tool is never limited or forced to
    /// another one, and the rows whose words are plain are held as labelled.
    #[test]
    fn the_routing_rows_are_never_forced_against_their_label() {
        // With folders present, a general question of source-routing.jsonl is never held to a
        // file tool either.
        let rows = [
            include_str!("../../../bench/agent/routing/dev.jsonl"),
            include_str!("../../../bench/agent/source-routing.jsonl"),
        ]
        .concat();
        let home = tool::expand_home("~").unwrap();
        let ctx = ToolCtx {
            allowed: vec![home.clone()],
            data_dir: home,
            source: String::new(),
        };
        let held = [
            ("routing-notools-web-001", Some(Force::Tool("web_answer"))),
            ("routing-nosearch-web-001", Some(Force::Tool("web_answer"))),
            ("routing-disputed-web-002", Some(Force::Tool("web_answer"))),
            ("routing-paper-read-001", Some(Force::Tool("read_papers"))),
            ("routing-paper-read-002", Some(Force::Tool("read_papers"))),
            ("routing-paper-none-001", None),
        ];
        let limited = [
            "routing-notools-none-001",
            "routing-nosearch-none-001",
            "routing-nosearch-none-002",
        ];
        for line in rows.lines() {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            let (id, user) = (row["id"].as_str().unwrap(), row["user"].as_str().unwrap());
            let calls = (row["history"].as_array().into_iter().flatten())
                .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten());
            let read = calls.clone().any(|c| c["name"] == "read_papers");
            let papers = if read {
                vec!["kepler".into()]
            } else {
                Vec::new()
            };
            let (force, limit) = (force(user, &ctx, &papers, false), limit(user));
            match row["expect"]["tool"].as_str() {
                None => assert!(limit.is_some() || force.is_none(), "{id}: {force:?}"),
                Some(tool) => {
                    assert_eq!(limit, None, "{id}");
                    let ok = match &force {
                        None | Some(Force::Any) => true,
                        Some(Force::Tool(t)) => *t == tool,
                        Some(Force::Fixed(_)) => false,
                    };
                    assert!(ok, "{id}: {force:?}");
                }
            }
            if let Some((_, want)) = held.iter().find(|(h, _)| *h == id) {
                assert_eq!(&force, want, "{id}");
            }
            assert_eq!(limited.contains(&id), limit.is_some(), "{id}");
        }
    }

    #[test]
    fn a_reply_check_fires_on_clear_words_only() {
        let all = ["web_answer", "read_papers", "read_file"];
        let rows = [
            (
                "I can't access the internet, but here is what I know.",
                Some(Hit::Denied),
            ),
            ("I don\u{2019}t have real-time data.", Some(Hit::Denied)),
            ("I cannot browse websites.", Some(Hit::Denied)),
            ("Sorry, I can't read PDFs.", Some(Hit::DeniedFiles)),
            (
                "Apple has not announced an M6 chip.",
                Some(Hit::UnsearchedNo),
            ),
            (
                "As of my knowledge cutoff, it is not out.",
                Some(Hit::UnsearchedNo),
            ),
            (
                "I have no information about that race.",
                Some(Hit::UnsearchedNo),
            ),
            (
                "Max Verstappen won the 2026 United States Grand Prix.",
                Some(Hit::AfterCutoff),
            ),
            ("It was released in March 2025.", Some(Hit::AfterCutoff)),
            ("The file does not exist.", None),
            ("Python 3.13 was released in 2024.", None),
            ("Here is a plan for your 2026 budget.", None),
            ("Today is 26 Oct 2026.", None),
            ("In 2026 you can save more. The team won.", None),
            ("There is no real time to lose.", None),
            ("Max won't race in 2026.", None),
            ("Ferrari wins the 2026 title in my forecast.", None),
            ("Ship it before you beat the 2026 deadline.", None),
            ("The 2099 final was won by Max.", None),
            (
                "Max won the 2026 race but won't race again.",
                Some(Hit::AfterCutoff),
            ),
            ("You can't browse while the VPN is down.", None),
            ("The laptop has no internet access offline.", None),
            ("Your script cannot read PDFs without poppler.", None),
            ("I don't have information about your schedule.", None),
            (
                "I'm unable to access the internet right now.",
                Some(Hit::Denied),
            ),
            ("I have no real-time data.", Some(Hit::Denied)),
        ];
        for (reply, want) in rows {
            assert_eq!(check_reply(reply, Some(2025), &all), want, "{reply}");
        }
        assert_eq!(
            check_reply("I can't access the internet.", Some(2025), &["read_file"]),
            None
        );
        assert_eq!(
            check_reply("I can't read PDFs.", Some(2025), &["web_answer"]),
            None
        );
        assert_eq!(check_reply("Max won the 2026 race.", None, &all), None);
    }
}
