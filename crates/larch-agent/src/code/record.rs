//! A chat's code task across its runs: what it set out to do, what the user asked since, and
//! what the last run changed and left. The next run's request is built from it (`request`), so
//! "continue" three turns later still knows the task.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::task::{Packet, Verdict};

/// The later asks a record keeps, newest last: the task's recent course is enough.
const ASKS: usize = 5;
/// The longest ask kept, in bytes: its gist is enough.
const ASK_BYTES: usize = 600;
/// The longest objective kept, in bytes: a detailed spec fits, a pasted file does not.
const OBJECTIVE_BYTES: usize = 2000;
/// The files and failing tests named, each list: the rest of a big change adds little.
const NAMED: usize = 20;
/// The longest name or verdict kept, in bytes.
const LINE_BYTES: usize = 300;

/// A chat's code task, saved as JSON under `<data>/agent/code/tasks/<chat_id>.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRecord {
    /// The task's first request.
    pub objective: String,
    /// The user's later requests in it, at most `ASKS`, newest last.
    pub asks: Vec<String>,
    /// The files the last run changed.
    pub changed: Vec<String>,
    /// What the last run left: the tests still failing after it.
    pub left: Vec<String>,
    /// How the last run ended, in words: why it stopped and what its tests said.
    pub verdict: String,
    /// The folder the task works in; None in a record from before it was kept, which is no
    /// folder's (`load_in`).
    #[serde(default)]
    pub folder: Option<PathBuf>,
}

impl TaskRecord {
    /// Where chat `chat_id`'s record is kept.
    pub fn path(data: &Path, chat_id: &str) -> PathBuf {
        data.join("agent/code/tasks")
            .join(format!("{chat_id}.json"))
    }

    /// Chat `chat_id`'s record; None when it has none yet.
    pub fn load(data: &Path, chat_id: &str) -> io::Result<Option<Self>> {
        match fs::read(Self::path(data, chat_id)) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Chat `chat_id`'s record when its task works in `folder`; None when it has none there, so
    /// a chat that moves to another folder starts a new task.
    pub fn load_in(data: &Path, chat_id: &str, folder: &Path) -> io::Result<Option<Self>> {
        Ok(Self::load(data, chat_id)?.filter(|r| r.folder.as_deref() == Some(folder)))
    }

    /// Saves the record as chat `chat_id`'s.
    pub fn save(&self, data: &Path, chat_id: &str) -> io::Result<()> {
        let path = Self::path(data, chat_id);
        fs::create_dir_all(path.parent().expect("under tasks"))?;
        fs::write(path, serde_json::to_vec(self).map_err(io::Error::other)?)
    }

    /// The task in `folder` after a run the user asked with `words` ended with `packet`: a new
    /// task when there was none, else `words` is one more ask.
    pub fn ran(record: Option<Self>, words: &str, packet: &Packet, folder: &Path) -> Self {
        let mut r = match record {
            Some(mut r) => {
                r.asks.push(cut(words, ASK_BYTES));
                let over = r.asks.len().saturating_sub(ASKS);
                r.asks.drain(..over);
                r
            }
            None => Self {
                objective: cut(words, OBJECTIVE_BYTES),
                ..Self::default()
            },
        };
        r.folder = Some(folder.to_owned());
        let named = |names: &mut dyn Iterator<Item = &String>| {
            names.take(NAMED).map(|n| cut(n, LINE_BYTES)).collect()
        };
        r.changed = named(&mut packet.scope.iter().map(|s| &s.path));
        // Partial names the tests still failing, or, untested, what the review still finds.
        r.left = match &packet.verdict {
            Verdict::Partial(left) | Verdict::AsBefore(left) => named(&mut left.iter()),
            _ => named(&mut packet.tests_after.iter().flat_map(|t| &t.failing)),
        };
        let tests = (packet.tests_after.as_ref())
            .map(|t| format!("{} tests passed, {} failed", t.passed, t.failed));
        // Its label alone: what it left is in `left` already.
        let sure = match packet.verdict {
            Verdict::Verified => "verified",
            Verdict::Partial(_) => "partial",
            Verdict::AsBefore(_) => "passes except failures that were there before the change",
            Verdict::Unverified => "unverified",
        };
        let verdict = [packet.stopped.as_deref(), tests.as_deref(), Some(sure)];
        r.verdict = cut(
            &verdict.into_iter().flatten().collect::<Vec<_>>().join("; "),
            LINE_BYTES,
        );
        r
    }

    /// The last run's change was undone or discarded: the folder has none of it.
    pub fn undone(&mut self) {
        self.changed.clear();
        self.verdict = "the user undid the last run's change".into();
    }
}

/// What a code run is asked for `now` in a chat whose task is `record`: the objective, the asks
/// since, what the last run changed and left, then "Now: ". A chat's first request is as it is.
/// The objective leads, so the run's commit and branch are named after the task.
pub fn request(record: Option<&TaskRecord>, now: &str) -> String {
    let Some(r) = record else {
        return now.to_owned();
    };
    let mut out = r.objective.clone();
    if !r.asks.is_empty() {
        out += "\n\nAsked since:";
        for a in &r.asks {
            out += &format!("\n- {a}");
        }
    }
    let lists = [("Changed so far", &r.changed), ("Still failing", &r.left)];
    let mut said = false;
    for (name, list) in lists.into_iter().filter(|(_, l)| !l.is_empty()) {
        out += &format!(
            "{}{name}: {}",
            if said { "\n" } else { "\n\n" },
            list.join(", ")
        );
        said = true;
    }
    if !r.verdict.is_empty() {
        out += &format!(
            "{}The last run: {}",
            if said { "\n" } else { "\n\n" },
            r.verdict
        );
    }
    format!("{out}\n\nNow: {now}")
}

/// The ask a code run's `request` (`request`) was made for now: its "Now: " words, or all of it.
pub fn now(request: &str) -> &str {
    request
        .rsplit_once("\n\nNow: ")
        .map_or(request, |(_, now)| now)
}

/// The files an answer drafted: each fenced block whose fence line ends with a file name
/// (```html index.html or ```index.html) and that is closed, as (name, contents). A name is a
/// bare file name with an extension, never a path, as it is written into the user's folder.
pub fn drafts(answer: &str) -> Vec<(String, String)> {
    let fence = |l: &str| l.trim_start().starts_with("```");
    let mut out = Vec::new();
    let mut lines = answer.lines();
    while let Some(line) = lines.next() {
        let Some(info) = line.trim_start().strip_prefix("```") else {
            continue;
        };
        let name = info.split_whitespace().last().filter(|n| file_name(n));
        let mut body = String::new();
        let closed = loop {
            match lines.next() {
                Some(l) if fence(l) => break true,
                Some(l) => body += &format!("{l}\n"),
                None => break false,
            }
        };
        if let Some(name) = name.filter(|_| closed) {
            out.push((name.to_owned(), body));
        }
    }
    out
}

/// Whether `n` is a bare file name with an extension that starts with a letter (`index.html`,
/// not `python3.11`, `../x.js` or `.env`).
fn file_name(n: &str) -> bool {
    let ok = |c: char| c.is_ascii_alphanumeric() || "._-".contains(c);
    n.chars().all(ok)
        && !n.starts_with('.')
        && n.rsplit_once('.')
            .is_some_and(|(_, ext)| ext.starts_with(|c: char| c.is_ascii_alphabetic()))
}

/// `s` cut to `bytes` at a character boundary.
fn cut(s: &str, bytes: usize) -> String {
    s[..s.floor_char_boundary(bytes)].to_owned()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{TaskRecord, drafts, request};
    use crate::code::task::{Packet, Scope, TestRun, Verdict};

    fn packet(changed: &[&str], failing: &[&str], stopped: &str) -> Packet {
        Packet {
            diff: String::new(),
            verdict: Verdict::Partial(failing.iter().map(|f| f.to_string()).collect()),
            tests_before: None,
            tests_after: Some(TestRun {
                passed: 3,
                failed: failing.len() as u32,
                failing: failing.iter().map(|f| f.to_string()).collect(),
                shared: Vec::new(),
                tail: String::new(),
            }),
            scope: (changed.iter())
                .map(|p| Scope {
                    path: p.to_string(),
                    inside: true,
                })
                .collect(),
            commands_run: vec![],
            pr_title: String::new(),
            pr_body: String::new(),
            stopped: Some(stopped.into()),
        }
    }

    #[test]
    fn a_first_request_is_as_it_is() {
        assert_eq!(request(None, "fix the test"), "fix the test");
    }

    #[test]
    fn a_follow_up_run_gets_the_objective_and_what_is_left() {
        let first = packet(&["index.html"], &["test_menu"], "the budget ran out");
        let r = TaskRecord::ran(
            None,
            "code a coffee shop website",
            &first,
            Path::new("/cafe"),
        );
        let r = TaskRecord::ran(Some(r), "add a menu page", &first, Path::new("/cafe"));
        let got = request(Some(&r), "continue");
        assert_eq!(
            got,
            "code a coffee shop website\n\nAsked since:\n- add a menu page\n\nChanged so far: \
             index.html\nStill failing: test_menu\nThe last run: the budget ran out; 3 tests \
             passed, 1 failed; partial\n\nNow: continue"
        );
    }

    #[test]
    fn a_record_stays_bounded() {
        let p = packet(&[], &[], "done");
        let mut r = TaskRecord::ran(None, &"x".repeat(5000), &p, Path::new("/x"));
        for i in 0..8 {
            r = TaskRecord::ran(
                Some(r),
                &format!("ask {i} {}", "é".repeat(400)),
                &p,
                Path::new("/x"),
            );
        }
        assert_eq!(r.objective.len(), 2000);
        assert_eq!(r.asks.len(), 5);
        assert!(r.asks[0].starts_with("ask 3 ") && r.asks[0].len() <= 600);
        r.undone();
        assert!(r.changed.is_empty() && r.verdict.contains("undid"));
    }

    /// A record is its folder's task only: another folder, or a record from before folders were
    /// kept, starts a new one.
    #[test]
    fn a_record_is_for_its_folder_only() {
        let tmp = tempfile::tempdir().unwrap();
        let (data, cafe) = (tmp.path(), Path::new("/cafe"));
        let r = TaskRecord::ran(None, "fix it", &packet(&[], &[], "done"), cafe);
        r.save(data, "c").unwrap();
        assert_eq!(TaskRecord::load_in(data, "c", cafe).unwrap(), Some(r));
        assert_eq!(
            TaskRecord::load_in(data, "c", Path::new("/shop")).unwrap(),
            None
        );
        let old = r#"{"objective":"x","asks":[],"changed":[],"left":[],"verdict":""}"#;
        fs::write(TaskRecord::path(data, "c"), old).unwrap();
        assert!(TaskRecord::load(data, "c").unwrap().is_some());
        assert_eq!(TaskRecord::load_in(data, "c", cafe).unwrap(), None);
    }

    #[test]
    fn only_closed_blocks_named_by_a_file_are_drafts() {
        let answer = "Here:\n```html index.html\n<h1>Hi</h1>\n```\n```css\nh1{}\n```\n\
                      ```style.css\nh1 { color: red }\n```\n```js ../evil.js\nx\n```\n\
                      ```python3.11\nprint()\n```\n```app.js\nunclosed";
        assert_eq!(
            drafts(answer),
            [
                ("index.html".into(), "<h1>Hi</h1>\n".into()),
                ("style.css".into(), "h1 { color: red }\n".into())
            ]
        );
    }
}
