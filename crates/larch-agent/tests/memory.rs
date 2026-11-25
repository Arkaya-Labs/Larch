//! File memory: caps, recall through the FTS5 index over facts, daily notes and transcripts,
//! near-duplicates, rebuilds and the memory tools.

use std::{
    fs::{self, File},
    path::Path,
};

use larch::{ChatMessage, Role, SessionOptions, chat::Transcript};
use larch_agent::memory::{
    ALL, AUTO, MEMORY_BYTES, MemoryError, Recall, Source, day, forget, list, note, notes,
    parse_keys, rebuild, recall, recall_with, remember, save_keys, system_prefix, update,
};
use larch_agent::tool::{Registry, ToolCtx};
use serde_json::{Value, json};

fn memory(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("agent/memory")
}

fn index_lines(data_dir: &Path) -> Vec<String> {
    fs::read_to_string(memory(data_dir).join("MEMORY.md"))
        .unwrap_or_default()
        .lines()
        .map(String::from)
        .collect()
}

fn fact_count(data_dir: &Path) -> usize {
    fs::read_dir(memory(data_dir).join("facts")).map_or(0, |d| d.count())
}

fn paths(hits: &[larch_agent::memory::Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.path.as_str()).collect()
}

/// Facts and daily notes a few weeks of use leave behind.
fn lived_in(data_dir: &Path) {
    for (text, kind) in [
        (
            "The user prefers tabs over spaces in Rust code",
            "preference",
        ),
        ("The user works on the larch inference engine", "project"),
        ("Reviews should list findings by severity", "preference"),
        ("The user lives in Berlin and works in the mornings", "user"),
        ("Release notes go in the CHANGELOG file", "project"),
        ("The bench machine has a 3050 GPU with 8 GB", "reference"),
    ] {
        remember(data_dir, text, kind, "agent").unwrap();
    }
    for line in [
        "Reviewed f4/tool-messages: 3 findings, 1 high",
        "Listed changes on a2/mmvq",
        "Read crates/larch/src/session.rs for the overflow rule",
        "Wrote notes.txt with the release checklist",
    ] {
        note(data_dir, line).unwrap();
    }
}

#[test]
fn a_fact_file_has_the_fixed_front_matter_and_an_index_line() {
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(
        dir.path(),
        "The user deploys the staging server with Ansible.\nBecause the old script broke twice.",
        "project",
        "chat-1 turn 4",
    )
    .unwrap()
    .slug;
    assert_eq!(slug, "the-user-deploys-the-staging-server");
    let file = fs::read_to_string(memory(dir.path()).join(format!("facts/{slug}.md"))).unwrap();
    let lines: Vec<&str> = file.lines().collect();
    assert_eq!(lines[0], "---");
    let keys: Vec<&str> = lines[1..5]
        .iter()
        .map(|l| l.split_once(": ").unwrap().0)
        .collect();
    assert_eq!(keys, ["type", "created", "source", "confidence"]);
    assert_eq!(lines[1], "type: project");
    assert_eq!(lines[3], "source: chat-1 turn 4");
    assert_eq!(lines[4], "confidence: medium");
    assert_eq!(lines[5], "---");
    assert_eq!(
        &lines[6..],
        [
            "The user deploys the staging server with Ansible.",
            "Because the old script broke twice."
        ]
    );
    assert_eq!(
        index_lines(dir.path()),
        [
            "- [the-user-deploys-the-staging-server] The user deploys the staging server with Ansible. Because the old script broke twice."
        ]
    );
}

/// Sixty facts, the old cap: `fact-0` to `fact-59`.
fn sixty_facts(data_dir: &Path) {
    let facts = memory(data_dir).join("facts");
    fs::create_dir_all(&facts).unwrap();
    let mut index = String::new();
    for i in 0..60 {
        fs::write(
            facts.join(format!("fact-{i}.md")),
            format!("---\ntype: user\ncreated: 2026-10-01T00:00:00Z\nsource: test\nconfidence: medium\n---\nfact number {i}\n"),
        )
        .unwrap();
        index.push_str(&format!("- [fact-{i}] fact number {i}\n"));
    }
    fs::write(memory(data_dir).join("MEMORY.md"), &index).unwrap();
}

#[test]
fn memory_has_no_cap() {
    let dir = tempfile::tempdir().unwrap();
    sixty_facts(dir.path());
    let text = "The user keeps a zebra named Otto";
    let slug = remember(dir.path(), text, "user", "agent").unwrap().slug;
    assert_eq!(fact_count(dir.path()), 61);
    assert_eq!(
        index_lines(dir.path()).last().unwrap(),
        &format!("- [{slug}] {text}")
    );
    // A fact file without its MEMORY.md line (a crash between the two writes, or a hand edit)
    // gets its line back on an update.
    fs::write(
        memory(dir.path()).join("facts/llama.md"),
        "---\ntype: user\ncreated: 2026-10-01T01:00:00Z\nsource: test\nconfidence: medium\n---\nThe user keeps a llama named Pablo\n",
    )
    .unwrap();
    let again = remember(
        dir.path(),
        "The user keeps a llama named Pablo",
        "user",
        "agent",
    );
    assert_eq!(again.unwrap().slug, "llama");
    assert_eq!(index_lines(dir.path()).len(), 62);
}

#[test]
fn the_system_prefix_is_user_md_then_the_newest_memory_md_lines_within_their_caps() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(system_prefix(dir.path()), "");
    let mem = memory(dir.path());
    fs::create_dir_all(&mem).unwrap();
    let user: Vec<String> = (0..40).map(|i| format!("user line {i}")).collect();
    // 40 lines of 200 bytes are more than the budget: the newest that fit go in, in file order.
    let facts: Vec<String> = (0..40)
        .map(|i| format!("- [f{i:02}] fact {i:02} {}", "x".repeat(184)))
        .collect();
    assert_eq!(facts[0].len(), 200);
    fs::write(mem.join("USER.md"), user.join("\n\n")).unwrap();
    fs::write(mem.join("MEMORY.md"), facts.join("\n")).unwrap();
    let fit = MEMORY_BYTES / 201;
    let newest = &facts[facts.len() - fit..];
    assert_eq!(
        system_prefix(dir.path()),
        format!(
            "## About the user\n{}\n\n## Memory index\n{}\n{FORGET}",
            user[..30].join("\n"),
            newest.join("\n")
        )
    );
    // A line too long for the budget is left out; older lines that fit still go in.
    let mut lines = facts[..2].to_vec();
    lines.push(format!("- [huge] {}", "y".repeat(MEMORY_BYTES)));
    fs::write(mem.join("MEMORY.md"), lines.join("\n")).unwrap();
    assert_eq!(
        system_prefix(dir.path()),
        format!(
            "## About the user\n{}\n\n## Memory index\n{}\n{FORGET}",
            user[..30].join("\n"),
            facts[..2].join("\n")
        )
    );
}

#[test]
fn a_long_fact_shows_whole_in_the_system_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let text = "My sister Maya lives in Austin, Texas, and works as a product designer at Figma; \
she moved there in March 2024 with her husband Daniel and their two kids, Leo and Ana, ages 6 and 9, near Zilker Park.";
    assert!(text.len() >= 190 && text.len() <= 240, "{}", text.len());
    let slug = remember(dir.path(), text, "user", "agent").unwrap().slug;
    assert!(
        system_prefix(dir.path()).contains(&format!("- [{slug}] {text}\n")),
        "{}",
        system_prefix(dir.path())
    );
}

#[test]
fn update_keeps_the_slug_and_front_matter_and_returns_the_old_text() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let slug = "the-user-lives-in-berlin-and";
    let path = memory(dir.path()).join(format!("facts/{slug}.md"));
    let before = fs::read_to_string(&path).unwrap();
    let new = "The user lives in Munich and works in the evenings";
    assert_eq!(
        update(dir.path(), slug, new).unwrap(),
        "The user lives in Berlin and works in the mornings"
    );
    let after = fs::read_to_string(&path).unwrap();
    let head = |file: &str| file.split("\n---\n").next().unwrap().to_owned();
    assert_eq!(head(&after), head(&before));
    assert!(after.ends_with(&format!("---\n{new}\n")), "{after}");
    assert!(index_lines(dir.path()).contains(&format!("- [{slug}] {new}")));
    assert_eq!(fact_count(dir.path()), 6);
    assert_eq!(
        list(dir.path())
            .unwrap()
            .iter()
            .find(|f| f.slug == slug)
            .map(|f| (f.text.as_str(), f.kind.as_str())),
        Some((new, "user"))
    );
    for missing in ["no-such-fact", "../USER", "THE-USER-LIVES-IN-BERLIN-AND"] {
        let err = update(dir.path(), missing, new).unwrap_err();
        assert!(matches!(err, MemoryError::NoFact(_)), "{missing}: {err:?}");
    }
    assert!(matches!(
        update(dir.path(), slug, " ?! "),
        Err(MemoryError::Empty)
    ));
}

#[test]
fn list_gives_the_facts_in_memory_md_order() {
    let dir = tempfile::tempdir().unwrap();
    assert!(list(dir.path()).unwrap().is_empty());
    lived_in(dir.path());
    // A hand-written line and a line whose file is gone are not facts.
    let mem = memory(dir.path());
    let mut lines = index_lines(dir.path());
    lines.insert(1, "Note by hand".into());
    lines.push("- [gone] A fact whose file is gone".into());
    fs::write(mem.join("MEMORY.md"), lines.join("\n")).unwrap();
    let facts = list(dir.path()).unwrap();
    let slugs: Vec<&str> = facts.iter().map(|f| f.slug.as_str()).collect();
    assert_eq!(slugs.len(), 6);
    assert_eq!(slugs[0], "the-user-prefers-tabs-over-spaces");
    assert_eq!(
        facts[0].text,
        "The user prefers tabs over spaces in Rust code"
    );
    assert_eq!(facts[0].kind, "preference");
    assert_eq!(facts[0].source, "agent");
    assert!(facts[0].created.starts_with("20"), "{:?}", facts[0]);
}

#[test]
fn remember_merges_two_people_whose_facts_share_most_words() {
    // Documents what `remember`'s near-duplicate update does today, in a memory with a few facts
    // in it: the second sister replaces the first. Capture never merges this way; it updates
    // only the slug the model names.
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let maya = "My sister Maya lives in Austin and works as a designer at Figma";
    let mira = "My sister Mira lives in Austin and works as a designer at Figma";
    let first = remember(dir.path(), maya, "user", "agent").unwrap();
    assert_eq!(first.old, None);
    let second = remember(dir.path(), mira, "user", "agent").unwrap();
    assert_eq!(second.slug, first.slug);
    assert_eq!(second.old.as_deref(), Some(maya));
    assert_eq!(fact_count(dir.path()), 7);
}

/// The line under the memory index.
const FORGET: &str = "`forget` takes the slug in brackets, not a path.";

#[test]
fn each_memory_file_has_its_heading_only_when_it_holds_lines() {
    let dir = tempfile::tempdir().unwrap();
    let mem = memory(dir.path());
    fs::create_dir_all(&mem).unwrap();
    fs::write(mem.join("MEMORY.md"), "- [tabs] The user prefers tabs\n").unwrap();
    fs::write(mem.join("USER.md"), "\n\n").unwrap();
    assert_eq!(
        system_prefix(dir.path()),
        format!("## Memory index\n- [tabs] The user prefers tabs\n{FORGET}")
    );
    fs::write(mem.join("USER.md"), "The user is called Sam.\n").unwrap();
    fs::write(mem.join("MEMORY.md"), "").unwrap();
    assert_eq!(
        system_prefix(dir.path()),
        "## About the user\nThe user is called Sam."
    );
}

#[test]
fn recall_finds_a_fact_and_a_daily_note_written_by_earlier_calls() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let slug = remember(
        dir.path(),
        "The staging server is deployed with Ansible",
        "project",
        "agent",
    )
    .unwrap()
    .slug;
    note(
        dir.path(),
        "Restarted the staging server after a failed deploy",
    )
    .unwrap();
    let hits = recall(dir.path(), "staging server").unwrap();
    let found = paths(&hits);
    let fact = format!("agent/memory/facts/{slug}.md");
    assert!(found.contains(&fact.as_str()), "{found:?}");
    let daily = hits
        .iter()
        .find(|h| h.path.starts_with("agent/memory/daily/") && h.snippet.contains("Restarted"))
        .unwrap_or_else(|| panic!("{hits:?}"));
    assert_eq!(daily.line, 5);
    let fact_hit = hits.iter().find(|h| h.path == fact).unwrap();
    assert_eq!(fact_hit.line, 7);
    assert!(hits.len() <= 5);
    assert!(
        hits.windows(2).all(|w| w[0].score >= w[1].score),
        "{hits:?}"
    );
    assert!(hits.iter().all(|h| h.score > 0.0));
}

#[test]
fn a_transcript_line_is_found_through_the_index() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    assert!(recall(dir.path(), "billing").unwrap().is_empty());
    // Written after the index was built, so recall must notice the newer file.
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        truncated: false,
        ..ChatMessage::default()
    };
    drop(
        Transcript::create(
            &dir.path().join("chat/abc.jsonl"),
            SessionOptions::default(),
            vec![
                message(Role::User, "Which port does the billing service use?"),
                message(Role::Assistant, "It listens on port 8443."),
            ],
        )
        .unwrap(),
    );
    let hits = recall(dir.path(), "billing port").unwrap();
    assert_eq!(hits[0].path, "chat/abc.jsonl", "{hits:?}");
    assert_eq!(hits[0].line, 1);
    assert!(hits[0].snippet.contains("8443"), "{hits:?}");
    assert!(!hits[0].snippet.contains("\"role\""), "{hits:?}");
    // A deleted file leaves no newer file behind; only its folder's time shows the change.
    fs::remove_file(dir.path().join("chat/abc.jsonl")).unwrap();
    assert!(recall(dir.path(), "billing").unwrap().is_empty());
}

#[test]
fn a_folder_that_goes_and_comes_back_goes_and_comes_back_in_recall() {
    // A deleted folder leaves no time at all: the user who clears the chat history, say. One put
    // back from the trash keeps its old time.
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        truncated: false,
        ..ChatMessage::default()
    };
    for (query, folder) in [
        ("zebracorn", "agent/memory/facts"),
        ("billing port", "chat"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let fact = "The zebracorn lives in the basement";
        remember(dir.path(), fact, "user", "agent").unwrap();
        let history = vec![
            message(Role::User, "Which port does the billing service use?"),
            message(Role::Assistant, "It listens on port 8443."),
        ];
        let chat = dir.path().join("chat/abc.jsonl");
        drop(Transcript::create(&chat, SessionOptions::default(), history).unwrap());
        // File times tick coarsely, and a time equal to the index's counts as a change: the index
        // must be newer than every file for the move alone to show.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(recall(dir.path(), query).unwrap().len(), 1, "{query}");
        let (here, away) = (
            dir.path().join(folder),
            dir.path().join(format!("{folder}-trash")),
        );
        fs::rename(&here, &away).unwrap();
        assert!(recall(dir.path(), query).unwrap().is_empty(), "{query}");
        fs::rename(&away, &here).unwrap();
        assert_eq!(recall(dir.path(), query).unwrap().len(), 1, "{query}");
    }
}

#[test]
fn a_file_changed_at_the_index_s_exact_time_is_read_again() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    assert!(recall(dir.path(), "zebra").unwrap().is_empty());
    let built = fs::metadata(memory(dir.path()).join("index.sqlite"))
        .unwrap()
        .modified()
        .unwrap();
    // A coarse clock gives a note written right after the index the index's own time. Changing
    // a file that exists leaves its folder's time alone.
    let daily = fs::read_dir(memory(dir.path()).join("daily"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut file = fs::OpenOptions::new().append(true).open(&daily).unwrap();
    std::io::Write::write_all(&mut file, b"- 12:00Z Fed the zebra\n").unwrap();
    file.set_modified(built).unwrap();
    drop(file);
    assert_eq!(paths(&recall(dir.path(), "zebra").unwrap()).len(), 1);
}

#[test]
fn a_line_written_while_the_index_builds_is_found_after_it() {
    // The index must date from the start of its build: dated at the end, a line added to a file
    // the build had already read looks older than the index, and no recall reads it again.
    let dir = tempfile::tempdir().unwrap();
    let daily = memory(dir.path()).join("daily");
    fs::create_dir_all(&daily).unwrap();
    let day: String = (0..200)
        .map(|i| format!("- 09:00Z line {i} of a busy day\n"))
        .collect();
    for n in 0..100 {
        fs::write(daily.join(format!("2020-{n:04}.md")), &day).unwrap();
    }
    let tmp = memory(dir.path()).join("index.sqlite.tmp");
    std::thread::scope(|s| {
        let build = s.spawn(|| rebuild(dir.path()));
        while !tmp.exists() && !build.is_finished() {
            std::thread::yield_now();
        }
        // By now the build has read the first file, and it reads 99 more.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut first = fs::OpenOptions::new()
            .append(true)
            .open(daily.join("2020-0000.md"))
            .unwrap();
        std::io::Write::write_all(&mut first, b"- 09:01Z the zebracorn moved\n").unwrap();
        build.join().unwrap().unwrap();
    });
    assert_eq!(recall(dir.path(), "zebracorn").unwrap().len(), 1);
}

// Windows has no FIFO in a folder; the test above covers it, though only for a build that is
// still running after its sleep.
#[cfg(unix)]
#[test]
fn a_line_written_while_the_index_builds_is_newer_than_the_index() {
    // The same rule, with a FIFO that holds the build at a known point: after the first file.
    let dir = tempfile::tempdir().unwrap();
    let daily = memory(dir.path()).join("daily");
    fs::create_dir_all(&daily).unwrap();
    let first = daily.join("2020-01-01.md");
    fs::write(&first, "- 09:00Z the day began\n").unwrap();
    let fifo = daily.join("2020-01-02.md");
    let made = std::process::Command::new("mkfifo").arg(&fifo).status();
    assert!(made.unwrap().success());
    std::thread::scope(|s| {
        let build = s.spawn(|| {
            let built = rebuild(dir.path());
            // A build that failed before it opened the FIFO would leave the open below waiting
            // for ever; an open for reading and writing never waits, and lets it go on.
            drop(fs::OpenOptions::new().read(true).write(true).open(&fifo));
            built
        });
        // Opening the FIFO returns once the build opens it, after it read the first file.
        let mut pipe = fs::OpenOptions::new().write(true).open(&fifo).unwrap();
        let mut file = fs::OpenOptions::new().append(true).open(&first).unwrap();
        std::io::Write::write_all(&mut file, b"- 09:01Z the zebracorn moved\n").unwrap();
        // File times tick coarsely: the build ends ticks after the line.
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::io::Write::write_all(&mut pipe, b"- 09:02Z a note through the pipe\n").unwrap();
        drop(pipe);
        build.join().unwrap().unwrap();
    });
    // A file at the index's time or later is read again (stale).
    let time = |p: &Path| fs::metadata(p).unwrap().modified().unwrap();
    assert!(time(&first) >= time(&memory(dir.path()).join("index.sqlite")));
}

#[test]
fn a_near_duplicate_updates_the_fact_instead_of_adding_one() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let slug = "the-user-prefers-tabs-over-spaces";
    let fact = memory(dir.path()).join(format!("facts/{slug}.md"));
    let created = fs::read_to_string(&fact)
        .unwrap()
        .lines()
        .nth(2)
        .unwrap()
        .to_owned();
    let before = fact_count(dir.path());
    let lines = index_lines(dir.path());
    // A contradiction of the fact has another slug, but the index finds it: an update.
    let flipped = "The user now prefers spaces over tabs in Rust code";
    assert_eq!(
        remember(dir.path(), flipped, "preference", "agent")
            .unwrap()
            .slug,
        slug
    );
    assert_eq!(fact_count(dir.path()), before);
    // Its line moves to the end, as the newest.
    let mut want = lines[1..].to_vec();
    want.push(format!("- [{slug}] {flipped}"));
    assert_eq!(index_lines(dir.path()), want, "updated, last");
    let file = fs::read_to_string(&fact).unwrap();
    assert!(file.ends_with(&format!("---\n{flipped}\n")), "{file}");
    assert_eq!(file.lines().nth(2).unwrap(), created);
    // The same words again, in another case and punctuation: the same fact.
    let again = remember(
        dir.path(),
        "the user now prefers spaces over tabs in Rust code!",
        "preference",
        "agent",
    );
    assert_eq!(again.unwrap().slug, slug);
    assert_eq!(fact_count(dir.path()), before);
    // Different facts that share words are added, not merged: merging would lose the old fact.
    for (text, kind) in [
        ("The user prefers Go over Rust for scripts", "preference"),
        ("The user's cat is called Miso", "user"),
    ] {
        assert_ne!(
            remember(dir.path(), text, kind, "agent").unwrap().slug,
            slug,
            "{text}"
        );
    }
    assert_eq!(fact_count(dir.path()), before + 2);
}

#[test]
fn a_rephrased_fact_is_updated_in_an_index_of_real_size() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    // Words like "the", "user" and "code" fill a real index, as here; a raw BM25 score triples
    // from 10 to 310 rows, and the merge decision must not move with it.
    for i in 0..300 {
        note(
            dir.path(),
            &format!("The user asked about item {i} in the code and the docs"),
        )
        .unwrap();
    }
    let slug = "the-user-prefers-tabs-over-spaces";
    let different = remember(
        dir.path(),
        "The user prefers Go over Rust for scripts",
        "preference",
        "agent",
    );
    assert_ne!(different.unwrap().slug, slug);
    let before = fact_count(dir.path());
    let rephrased = "In Rust code the user prefers tabs, not spaces";
    assert_eq!(
        remember(dir.path(), rephrased, "preference", "agent")
            .unwrap()
            .slug,
        slug
    );
    assert_eq!(fact_count(dir.path()), before);
    assert!(index_lines(dir.path()).contains(&format!("- [{slug}] {rephrased}")));
}

#[test]
fn the_index_rebuilds_from_the_files_with_the_same_results() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let query = "user rust staging release";
    let first = recall(dir.path(), query).unwrap();
    assert!(!first.is_empty());
    fs::remove_file(memory(dir.path()).join("index.sqlite")).unwrap();
    assert_eq!(recall(dir.path(), query).unwrap(), first);
    rebuild(dir.path()).unwrap();
    assert_eq!(recall(dir.path(), query).unwrap(), first);
    // An index built before it listed its folders is built again.
    let index = rusqlite::Connection::open(memory(dir.path()).join("index.sqlite")).unwrap();
    index.execute_batch("DROP TABLE folders").unwrap();
    drop(index);
    assert_eq!(recall(dir.path(), query).unwrap(), first);
    // The same hits come from an index that was kept or built again: its list shows which.
    let index = rusqlite::Connection::open(memory(dir.path()).join("index.sqlite")).unwrap();
    let listed: i64 = index
        .query_row("SELECT count(*) FROM folders", [], |r| r.get(0))
        .unwrap();
    assert_eq!(listed, 2, "facts and daily");
}

#[test]
fn forget_removes_the_file_and_the_line_and_nothing_else() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    fs::write(memory(dir.path()).join("USER.md"), "Name: Sam\n").unwrap();
    let slug = "release-notes-go-in-the-changelog";
    assert!(!recall(dir.path(), "changelog").unwrap().is_empty());
    forget(dir.path(), slug).unwrap();
    assert!(!memory(dir.path()).join(format!("facts/{slug}.md")).exists());
    assert!(!index_lines(dir.path()).iter().any(|l| l.contains(slug)));
    assert_eq!(index_lines(dir.path()).len(), 5);
    assert!(recall(dir.path(), "changelog").unwrap().is_empty());
    let err = forget(dir.path(), slug).unwrap_err();
    assert_eq!(err.to_string(), format!("no fact named '{slug}'"));
    // A slug is a name, never a path, and never another case of a name: on a case-insensitive
    // file system that would delete the file and keep its MEMORY.md line.
    let tabs = "THE-USER-PREFERS-TABS-OVER-SPACES";
    for bad in [
        "../USER",
        "../../agent/memory/USER",
        "facts/x",
        "",
        "a.md",
        tabs,
    ] {
        assert!(
            matches!(forget(dir.path(), bad), Err(MemoryError::NoFact(_))),
            "{bad}"
        );
    }
    assert!(memory(dir.path()).join("USER.md").exists());
    let kept = format!("facts/{}.md", tabs.to_lowercase());
    assert!(memory(dir.path()).join(kept).exists());
    assert_eq!(index_lines(dir.path()).len(), 5);
}

#[test]
fn a_line_whose_fact_file_is_gone_is_forgotten() {
    // forget deletes the file before the line, so a crash between the two leaves such a line.
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(dir.path(), "The user drinks green tea", "user", "agent")
        .unwrap()
        .slug;
    fs::remove_file(memory(dir.path()).join(format!("facts/{slug}.md"))).unwrap();
    forget(dir.path(), &slug).unwrap();
    assert!(index_lines(dir.path()).is_empty());
}

#[test]
fn a_byte_order_mark_does_not_hide_the_first_line() {
    // Some Windows editors put one in front of MEMORY.md.
    let dir = tempfile::tempdir().unwrap();
    let (tea, zebra) = (
        "The user drinks green tea",
        "The zebracorn lives in the basement",
    );
    let slug = remember(dir.path(), tea, "user", "agent").unwrap().slug;
    remember(dir.path(), zebra, "user", "agent").unwrap();
    let mark = || {
        let path = memory(dir.path()).join("MEMORY.md");
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, format!("\u{feff}{text}")).unwrap();
    };
    mark();
    assert_eq!(
        remember(dir.path(), tea, "user", "agent").unwrap().slug,
        slug
    );
    assert_eq!(index_lines(dir.path()).len(), 2);
    mark();
    forget(dir.path(), &slug).unwrap();
    let line = format!("- [the-zebracorn-lives-in-the-basement] {zebra}");
    assert_eq!(index_lines(dir.path()), [line]);
}

#[test]
fn forget_deletes_only_the_slug_it_was_given_byte_for_byte() {
    // APFS and NTFS give one file to names that differ in bytes: 'ſ' is 's', 'ς' is 'σ', and
    // decomposed Hangul is the composed form. Deleting through such a name kept the line.
    let dir = tempfile::tempdir().unwrap();
    let decomposed = "\u{1109}\u{1161}\u{110b}\u{116d}\u{11bc}\u{110c}\u{1161}\u{1102}\u{1173}\u{11ab}-\u{110e}\u{1161}\u{1105}\u{1173}\u{11af}-\u{1106}\u{1161}\u{1109}\u{1175}\u{11ab}\u{1103}\u{1161}";
    for (text, other) in [
        ("The user drinks green tea", "the-uſer-drinks-green-tea"),
        ("Σήμερα πίνει τσάι", "ςήμερα-πίνει-τσάι"),
        ("사용자는 차를 마신다", decomposed),
    ] {
        let slug = remember(dir.path(), text, "user", "agent").unwrap().slug;
        assert_ne!(slug, other);
        let err = forget(dir.path(), other).unwrap_err();
        assert!(matches!(err, MemoryError::NoFact(_)), "{other}: {err:?}");
        let file = memory(dir.path()).join(format!("facts/{slug}.md"));
        assert!(file.exists(), "{other}");
    }
    assert_eq!(index_lines(dir.path()).len(), 3);
}

// NTFS lists a name as it was written, so only a Unix file system holds such a pair.
#[cfg(unix)]
#[test]
fn forget_deletes_nothing_through_another_name_for_a_lines_fact() {
    // HFS+ lists the decomposed form of a composed slug, which a recall hit's path shows: a
    // name with no line of its own that opens a fact that has one. A hard link makes such a
    // pair on any Unix file system.
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(dir.path(), "The user drinks green tea", "user", "agent")
        .unwrap()
        .slug;
    let facts = memory(dir.path()).join("facts");
    fs::hard_link(facts.join(format!("{slug}.md")), facts.join("green-tea.md")).unwrap();
    let err = forget(dir.path(), "green-tea").unwrap_err();
    assert!(matches!(err, MemoryError::NoFact(_)), "{err:?}");
    assert!(facts.join("green-tea.md").exists());
    assert_eq!(index_lines(dir.path()).len(), 1);
    // A fact file that no line names is still forgotten.
    fs::write(facts.join("orphan.md"), "x\n").unwrap();
    forget(dir.path(), "orphan").unwrap();
    assert!(!facts.join("orphan.md").exists());
}

#[test]
fn forget_deletes_a_fact_whose_name_the_disk_lists_in_another_form() {
    // HFS+ stores names decomposed: 'é' in the slug is listed as 'e' and a combining accent, while
    // a lookup by the slug still opens the file. Checked on an HFS+ disk image on macOS.
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(
        dir.path(),
        "Café au lait every morning for the user",
        "preference",
        "agent",
    )
    .unwrap()
    .slug;
    assert_eq!(slug, "caf\u{e9}-au-lait-every-morning-for");
    forget(dir.path(), &slug).unwrap();
    assert_eq!(fact_count(dir.path()), 0);
    assert!(index_lines(dir.path()).is_empty());
    assert!(recall(dir.path(), "café au lait").unwrap().is_empty());
}

#[test]
fn queries_are_plain_words_whatever_their_punctuation() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    assert!(recall(dir.path(), "").unwrap().is_empty());
    assert!(recall(dir.path(), "*** -- :: \"").unwrap().is_empty());
    let hits = recall(dir.path(), "\"tabs\" OR NEAR(spaces* -rust: ^code").unwrap();
    assert!(paths(&hits).contains(&"agent/memory/facts/the-user-prefers-tabs-over-spaces.md"));
}

#[test]
fn remember_checks_the_type_and_the_text() {
    let dir = tempfile::tempdir().unwrap();
    let shown = |text, kind| {
        remember(dir.path(), text, kind, "agent")
            .unwrap_err()
            .to_string()
    };
    assert_eq!(
        shown("The user likes tea", "habit"),
        "type 'habit' is not one of user, project, preference, reference"
    );
    assert_eq!(shown(" ?! ", "user"), "the fact has no words");
    assert_eq!(fact_count(dir.path()), 0);
}

#[test]
fn the_memory_tools_write_through_the_same_functions() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    // `remember`'s result ends in its receipt, which the turn takes off.
    let said = |r: String| r.split('\u{1e}').next().unwrap().to_owned();
    let call = |name: &str, args: Value| registry.get(name).unwrap().call(&args, &ctx).map(said);
    let slug = call(
        "remember",
        json!({"text": "The user drinks green tea", "type": "preference"}),
    )
    .unwrap();
    assert_eq!(slug, "the-user-drinks-green-tea");
    // An update says what it replaced.
    assert_eq!(
        call(
            "remember",
            json!({"text": "the user drinks green tea!", "type": "preference"}),
        )
        .unwrap(),
        format!("{slug}; it replaced: The user drinks green tea")
    );
    // The fact says where it came from: the loop puts the chat ID and turn in the context.
    let fact = std::fs::read_to_string(dir.path().join(format!("agent/memory/facts/{slug}.md")));
    assert!(fact.unwrap().contains("\nsource: test\n"));
    let found: Value =
        serde_json::from_str(&call("recall", json!({"query": "tea"})).unwrap()).unwrap();
    assert_eq!(found[0]["path"], format!("agent/memory/facts/{slug}.md"));
    assert_eq!(found[0]["line"], 7);
    assert!(found[0]["snippet"].as_str().unwrap().contains("tea"));
    assert!(found[0]["score"].as_f64().unwrap() > 0.0);
    assert_eq!(
        call("forget", json!({"slug": slug})).unwrap(),
        format!("forgot '{slug}'")
    );
    assert_eq!(
        call("forget", json!({"slug": slug}))
            .unwrap_err()
            .to_string(),
        format!("no fact named '{slug}'")
    );
}

#[test]
fn a_longer_fact_containing_an_old_one_is_added_not_merged() {
    let dir = tempfile::tempdir().unwrap();
    let old = remember(dir.path(), "The user lives in Berlin", "user", "agent")
        .unwrap()
        .slug;
    let new = remember(
        dir.path(),
        "The user's sister lives in Berlin and visits every summer",
        "user",
        "agent",
    )
    .unwrap()
    .slug;
    assert_ne!(new, old);
    let file = fs::read_to_string(memory(dir.path()).join(format!("facts/{old}.md"))).unwrap();
    assert!(file.ends_with("---\nThe user lives in Berlin\n"), "{file}");
    assert_eq!(index_lines(dir.path()).len(), 2);
}

#[test]
fn the_second_fact_of_a_new_memory_is_added_not_merged() {
    // Scored on a table of the two texts alone, every word weighs the same, and facts that share
    // most of their words read as one.
    for (first, second) in [
        (
            "The user has a cat named Miso",
            "The user has a dog named Rex",
        ),
        (
            "Deploys go to the staging server first",
            "Backups go to the staging server first",
        ),
        (
            "The user prefers tabs in Rust code",
            "The user prefers spaces in Python code",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let slug = remember(dir.path(), first, "user", "agent").unwrap().slug;
        assert_ne!(
            remember(dir.path(), second, "user", "agent").unwrap().slug,
            slug,
            "{second}"
        );
        assert_eq!(fact_count(dir.path()), 2, "{second}");
        let file = fs::read_to_string(memory(dir.path()).join(format!("facts/{slug}.md")));
        assert!(
            file.unwrap().ends_with(&format!("---\n{first}\n")),
            "{first}"
        );
        // The same text again is still the same fact.
        assert_eq!(
            remember(dir.path(), first, "user", "agent").unwrap().slug,
            slug
        );
        assert_eq!(fact_count(dir.path()), 2, "{first}");
    }
}

#[test]
fn a_one_line_text_does_not_replace_a_fact_of_several_lines() {
    let dir = tempfile::tempdir().unwrap();
    let full =
        "The user deploys staging with Ansible.\nWhy: the old shell script broke twice in March.";
    let old = remember(dir.path(), full, "project", "agent").unwrap().slug;
    let short = "The user deploys staging with Ansible.";
    assert_ne!(
        remember(dir.path(), short, "project", "agent")
            .unwrap()
            .slug,
        old
    );
    let file = fs::read_to_string(memory(dir.path()).join(format!("facts/{old}.md"))).unwrap();
    assert!(file.ends_with(&format!("---\n{full}\n")), "{file}");
}

#[test]
fn the_same_text_finds_its_fact_when_another_fact_has_the_best_line() {
    // Word lists, not prose: the second fact's first line ranks above the first fact's lines.
    let dir = tempfile::tempdir().unwrap();
    let first = "answers release editor\nWhy: team coffee";
    let slug = remember(dir.path(), first, "project", "agent")
        .unwrap()
        .slug;
    for text in [
        "answers editor the release coffee\nWhy: dark rust tea they mode",
        "tea team short deploy",
    ] {
        assert_ne!(
            remember(dir.path(), text, "project", "agent").unwrap().slug,
            slug
        );
    }
    assert_eq!(
        remember(dir.path(), first, "project", "agent")
            .unwrap()
            .slug,
        slug
    );
    assert_eq!(fact_count(dir.path()), 3);
}

#[test]
fn a_file_in_facts_that_is_not_a_fact_does_not_hide_one() {
    // A note the user keeps there, ranked first: no slug names it, so the fact behind it is the
    // one to update.
    let dir = tempfile::tempdir().unwrap();
    let text = "The user drinks green tea every morning";
    let slug = remember(dir.path(), text, "user", "agent").unwrap().slug;
    fs::write(
        memory(dir.path()).join("facts/Notes.md"),
        format!("{text}\n"),
    )
    .unwrap();
    assert_eq!(
        remember(dir.path(), text, "user", "agent").unwrap().slug,
        slug
    );
    assert_eq!(index_lines(dir.path()).len(), 1);
}

#[test]
fn facts_that_share_their_first_six_words_are_both_kept() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let text = "The user works on the larch desktop app too";
    let slug = remember(dir.path(), text, "project", "agent").unwrap().slug;
    assert_eq!(slug, "the-user-works-on-the-larch-2");
    let first = memory(dir.path()).join("facts/the-user-works-on-the-larch.md");
    let first = fs::read_to_string(first).unwrap();
    assert!(
        first.ends_with("---\nThe user works on the larch inference engine\n"),
        "{first}"
    );
    assert!(index_lines(dir.path()).contains(&format!("- [{slug}] {text}")));
}

#[test]
fn a_fact_without_spaces_gets_a_short_slug() {
    let dir = tempfile::tempdir().unwrap();
    // One "word" of 300 bytes: no file system takes it as a name.
    let text = "柏林的用户喜欢写代码".repeat(10);
    let slug = remember(dir.path(), &text, "user", "agent").unwrap().slug;
    assert!(slug.len() <= 48, "{slug}");
    let file = fs::read_to_string(memory(dir.path()).join(format!("facts/{slug}.md"))).unwrap();
    assert!(file.ends_with(&format!("---\n{text}\n")), "{file}");
}

#[test]
fn rebuild_deletes_the_old_index_with_its_journal() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    // A crash mid-update leaves a hot journal; next to a new index, SQLite would replay it.
    let journal = memory(dir.path()).join("index.sqlite-journal");
    fs::write(&journal, b"left by a crash").unwrap();
    rebuild(dir.path()).unwrap();
    assert!(!journal.exists());
    assert_eq!(
        paths(&recall(dir.path(), "tabs").unwrap())[0],
        "agent/memory/facts/the-user-prefers-tabs-over-spaces.md"
    );
}

#[test]
fn recall_leaves_out_tool_results_and_calls_but_finds_the_users_line() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        ..ChatMessage::default()
    };
    // A file the model read and a call whose arguments hold a `content`: what a tool saw, not what
    // was said, so neither is a line recall finds.
    let call = ChatMessage {
        tool_calls: vec![larch::session::ToolCall {
            name: "write_file".into(),
            arguments: json!({"path": "billing.toml", "content": "quokka"}),
            id: "000000001".into(),
        }],
        ..message(Role::Assistant, "")
    };
    let path = dir.path().join("chat/abc.jsonl");
    let mut transcript = Transcript::create(&path, SessionOptions::default(), vec![]).unwrap();
    let metrics = larch::chat::Metrics {
        prompt_tokens: 10,
        completion_tokens: 5,
        cached_tokens: 0,
        ttft_ms: 1.0,
        tpot_ms: 1.0,
        tok_s: 1.0,
    };
    for exchange in [
        [
            message(Role::User, "Which port does the billing service use?"),
            call,
        ],
        [
            ChatMessage {
                tool_call_id: "000000001".into(),
                ..message(Role::Tool, "port = 8443 # zanzibar")
            },
            message(Role::Assistant, "It listens on port 8443."),
        ],
    ] {
        transcript
            .append_turn(
                larch::chat::Terminal::Eos,
                Some(metrics.clone()),
                0,
                Some(exchange),
            )
            .unwrap();
    }
    drop(transcript);
    for word in ["zanzibar", "quokka"] {
        let hits = recall(dir.path(), word).unwrap();
        assert!(hits.is_empty(), "{word}: {hits:?}");
    }
    let hits = recall(dir.path(), "billing").unwrap();
    let user = hits
        .iter()
        .find(|h| h.path == "chat/abc.jsonl")
        .unwrap_or_else(|| panic!("{hits:?}"));
    assert!(user.snippet.contains("billing service"), "{user:?}");
}

#[test]
fn a_transcripts_system_prompt_is_not_indexed() {
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    // An agent chat's system prompt carries MEMORY.md: indexed, every chat would hold a copy of
    // every fact, ranked above the fact itself.
    let system = system_prefix(dir.path());
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        truncated: false,
        ..ChatMessage::default()
    };
    let options = SessionOptions {
        system: Some(system.clone()),
        ..SessionOptions::default()
    };
    let history = vec![
        message(Role::System, &system),
        message(Role::User, "Which port does the billing service use?"),
        message(Role::Assistant, "It listens on port 8443."),
    ];
    drop(Transcript::create(&dir.path().join("chat/abc.jsonl"), options, history).unwrap());
    let hits = recall(dir.path(), "tabs").unwrap();
    assert!(
        hits.iter().all(|h| !h.path.starts_with("chat/")),
        "{hits:?}"
    );
    assert_eq!(
        paths(&recall(dir.path(), "billing").unwrap())[0],
        "chat/abc.jsonl"
    );
}

#[test]
fn a_transcript_that_cannot_be_read_fails_recall_but_not_a_write() {
    // Not a lock, the one read error a running chat causes: a folder where a transcript belongs,
    // or a file where the transcripts' folder belongs.
    let broken: [fn(&Path); 2] = [
        |data| fs::create_dir_all(data.join("chat/broken.jsonl")).unwrap(),
        |data| fs::write(data.join("chat"), "").unwrap(),
    ];
    for (i, break_chat) in broken.into_iter().enumerate() {
        let dir = tempfile::tempdir().unwrap();
        lived_in(dir.path());
        // An index exists, as after any recall.
        assert!(!recall(dir.path(), "tabs").unwrap().is_empty());
        break_chat(dir.path());
        let err = recall(dir.path(), "tabs").unwrap_err();
        assert!(matches!(err, MemoryError::Io(_)), "{i}: {err:?}");
        // Writes read only facts and daily notes, so a transcript never blocks them.
        let slug = remember(
            dir.path(),
            "The user keeps a zebra named Otto",
            "user",
            "agent",
        )
        .unwrap()
        .slug;
        forget(dir.path(), &slug).unwrap();
        forget(dir.path(), "release-notes-go-in-the-changelog").unwrap();
        assert_eq!(index_lines(dir.path()).len(), 5, "{i}");
    }
}

#[test]
fn a_memory_md_that_is_not_utf8_is_left_as_it_is() {
    // Read as text, its other bytes would come back as U+FFFD in every write.
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(dir.path(), "The user drinks green tea", "user", "agent")
        .unwrap()
        .slug;
    let path = memory(dir.path()).join("MEMORY.md");
    let mut bytes = fs::read(&path).unwrap();
    bytes.extend_from_slice(b"Caf\xe9 every morning (written by hand)\n");
    fs::write(&path, &bytes).unwrap();
    for err in [
        remember(
            dir.path(),
            "The user keeps a zebra named Otto",
            "user",
            "agent",
        )
        .unwrap_err(),
        forget(dir.path(), &slug).unwrap_err(),
    ] {
        assert_eq!(
            err.to_string(),
            "memory: MEMORY.md is not UTF-8 text; ask the user to save it as UTF-8"
        );
    }
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(fact_count(dir.path()), 1);
}

#[test]
fn a_linked_memory_file_stays_a_link() {
    // MEMORY.md kept with the user's dotfiles: a write goes to the file the link names.
    // Also when the file it names does not exist yet, as `fs::write` follows it. The link is
    // relative, so it names a file from the link's folder, not the working folder.
    for lines in [1, 0] {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("dotfiles")).unwrap();
        let kept = dir.path().join("dotfiles/MEMORY.md");
        // With this system's separator: Windows reads no '/' in a link's target.
        let to: std::path::PathBuf = ["..", "..", "dotfiles", "MEMORY.md"].iter().collect();
        if lines == 1 {
            fs::write(
                &kept,
                "- [the-user-drinks-green-tea] The user drinks green tea\n",
            )
            .unwrap();
        }
        fs::create_dir_all(memory(dir.path())).unwrap();
        let link = memory(dir.path()).join("MEMORY.md");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&to, &link).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(&to, &link).unwrap();
        remember(
            dir.path(),
            "The user keeps a zebra named Otto",
            "user",
            "agent",
        )
        .unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().is_symlink());
        let text = fs::read_to_string(&kept).unwrap();
        assert_eq!(text.lines().count(), lines + 1, "{text}");
    }
}

#[test]
fn memory_used_at_the_same_time_keeps_every_line_and_reads_a_whole_index() {
    // Two chats that share a data folder write to one MEMORY.md, and recall and rebuild the
    // index the writes leave stale.
    let dir = tempfile::tempdir().unwrap();
    lived_in(dir.path());
    let new = [
        "The user keeps a zebra named Otto",
        "Builds run on a machine in the basement",
        "Invoices are due on the fifth of each month",
        "The staging database restarts every Sunday night",
        "Meetings with the design team happen on Thursdays",
        "Backups go to an external drive called Vault",
        "Her laptop is a ThinkPad from 2019",
        "Passwords live in a manager, never in notes",
    ];
    let gone = [
        "release-notes-go-in-the-changelog",
        "reviews-should-list-findings-by-severity",
        "the-bench-machine-has-a-3050",
    ];
    let (readers, rounds) = (3, 30);
    let start = std::sync::Barrier::new(new.len() + gone.len() + 2 * readers);
    std::thread::scope(|s| {
        for text in new {
            s.spawn(|| {
                start.wait();
                remember(dir.path(), text, "project", "agent").unwrap();
            });
        }
        for slug in gone {
            s.spawn(|| {
                start.wait();
                forget(dir.path(), slug).unwrap();
            });
        }
        for _ in 0..readers {
            s.spawn(|| {
                start.wait();
                for _ in 0..rounds {
                    recall(dir.path(), "user staging").unwrap();
                }
            });
            s.spawn(|| {
                start.wait();
                for _ in 0..rounds {
                    rebuild(dir.path()).unwrap();
                }
            });
        }
    });
    assert_eq!(index_lines(dir.path()).len(), 6 + new.len() - gone.len());
    assert_eq!(fact_count(dir.path()), 6 + new.len() - gone.len());
}

/// A fact, a daily note and a transcript line, each about the billing port; the fact and the
/// note are from 2026-10-10.
fn one_of_each(data_dir: &Path) {
    let facts = memory(data_dir).join("facts");
    fs::create_dir_all(&facts).unwrap();
    fs::write(
        facts.join("billing-runs-on-port-8443.md"),
        "---\ntype: project\ncreated: 2026-10-10T09:00:00Z\nsource: test\nconfidence: medium\n---\nBilling runs on port 8443\n",
    )
    .unwrap();
    fs::write(
        memory(data_dir).join("MEMORY.md"),
        "- [billing-runs-on-port-8443] Billing runs on port 8443\n",
    )
    .unwrap();
    fs::create_dir_all(memory(data_dir).join("daily")).unwrap();
    fs::write(
        memory(data_dir).join("daily/2026-10-10.md"),
        "- 10:00Z chat 1: read_file billing.toml: billing port 8443\n",
    )
    .unwrap();
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        ..ChatMessage::default()
    };
    let history = vec![
        message(Role::User, "Which port does billing use?"),
        message(Role::Assistant, "I do not know."),
    ];
    let path = data_dir.join("chat/abc.jsonl");
    drop(Transcript::create(&path, SessionOptions::default(), history).unwrap());
}

fn sources(hits: &[larch_agent::memory::Hit]) -> Vec<Source> {
    hits.iter().map(|h| h.source).collect()
}

/// Every source or `sources`, 5 hits, dated from `today`.
fn how<'a>(sources: &'a [Source], today: &str) -> Recall<'a> {
    Recall {
        sources,
        hits: 5,
        today: day(today).unwrap(),
    }
}

#[test]
fn every_row_has_its_source_and_automatic_recall_reads_no_transcript() {
    let dir = tempfile::tempdir().unwrap();
    one_of_each(dir.path());
    let all = recall_with(dir.path(), "billing port", how(&ALL, "2026-10-17")).unwrap();
    let mut got = sources(&all);
    got.sort_by_key(|s| s.as_str());
    let want = [Source::Daily, Source::Fact, Source::Transcript];
    assert_eq!(got, want, "{all:?}");
    let transcript = all.iter().find(|h| h.source == Source::Transcript);
    assert_eq!(transcript.unwrap().path, "chat/abc.jsonl");
    let auto = recall_with(dir.path(), "billing port", how(&AUTO, "2026-10-17")).unwrap();
    assert_eq!(auto.len(), 2, "{auto:?}");
    assert!(
        auto.iter().all(|h| !h.path.starts_with("chat/")),
        "{auto:?}"
    );
    // The plain `recall` searches everything.
    assert_eq!(recall(dir.path(), "billing port").unwrap().len(), 3);
}

#[test]
fn the_recall_tool_reads_past_chats_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    one_of_each(dir.path());
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    let paths_for = |args: Value| -> Vec<String> {
        let out = registry.get("recall").unwrap().call(&args, &ctx).unwrap();
        let hits: Vec<Value> = serde_json::from_str(&out).unwrap();
        (hits.iter())
            .map(|h| h["path"].as_str().unwrap().to_owned())
            .collect()
    };
    // Without a source, facts and daily notes: a forgotten fact never comes back from a chat.
    let auto = paths_for(json!({"query": "billing port"}));
    assert_eq!(auto.len(), 2, "{auto:?}");
    assert!(!auto.contains(&"chat/abc.jsonl".to_owned()), "{auto:?}");
    for (source, path) in [
        ("fact", "agent/memory/facts/billing-runs-on-port-8443.md"),
        ("daily", "agent/memory/daily/2026-10-10.md"),
        ("transcript", "chat/abc.jsonl"),
    ] {
        let got = paths_for(json!({"query": "billing port", "source": source}));
        assert_eq!(got, [path], "{source}");
    }
}

#[test]
fn the_recall_tool_returns_at_most_recall_hits_lines() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..12 {
        note(dir.path(), &format!("zebra sighting number {i}")).unwrap();
    }
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    let recall = || {
        let tool = registry.get("recall").unwrap();
        tool.call(&json!({"query": "zebra"}), &ctx)
    };
    let count = || {
        serde_json::from_str::<Vec<Value>>(&recall().unwrap())
            .unwrap()
            .len()
    };
    assert_eq!(count(), 5);
    for hits in [1, 10] {
        let text = format!("recall_hits = {hits}\n");
        fs::write(dir.path().join("agent/config.toml"), text).unwrap();
        assert_eq!(count(), hits);
    }
    // A config.toml the chat could not open fails the tool with its reason, not a default.
    fs::write(dir.path().join("agent/config.toml"), "recall_hits = 11\n").unwrap();
    let err = recall().unwrap_err().to_string();
    assert!(
        err.contains("recall_hits takes a whole number from 1 to 10"),
        "{err}"
    );
}

#[test]
fn an_index_built_before_the_source_column_is_built_again() {
    let dir = tempfile::tempdir().unwrap();
    one_of_each(dir.path());
    // The index v1 wrote, newer than every file, with the folders it was built from.
    let path = memory(dir.path()).join("index.sqlite");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE lines USING fts5(text, path UNINDEXED, line UNINDEXED);
         INSERT INTO lines VALUES ('billing port', 'agent/memory/daily/2026-10-10.md', 1);
         CREATE TABLE folders (name TEXT);
         INSERT INTO folders VALUES ('agent/memory/facts'), ('agent/memory/daily'), ('chat');",
    )
    .unwrap();
    drop(conn);
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    let file = File::options().write(true).open(&path).unwrap();
    file.set_modified(later).unwrap();
    let hits = recall(dir.path(), "billing port").unwrap();
    assert_eq!(hits.len(), 3, "{hits:?}");
}

#[test]
fn a_date_phrase_keeps_the_rows_of_its_days() {
    let dir = tempfile::tempdir().unwrap();
    let daily = memory(dir.path()).join("daily");
    fs::create_dir_all(&daily).unwrap();
    for (day, text) in [
        (
            "2026-06-10",
            "- 09:00Z chat 1: boiler quote from Tomasz: 640 EUR",
        ),
        (
            "2026-07-10",
            "- 09:00Z chat 2: boiler fixed by Tomasz: 180 EUR",
        ),
        ("2026-10-16", "- 09:00Z chat 3: boiler serviced, all fine"),
    ] {
        fs::write(daily.join(format!("{day}.md")), format!("{text}\n")).unwrap();
    }
    let facts = memory(dir.path()).join("facts");
    fs::create_dir_all(&facts).unwrap();
    // Saved in May about a trip in July: a day without its year is the one nearest the fact.
    fs::write(
        facts.join("the-lisbon-trip-starts-on-3-july.md"),
        "---\ntype: user\ncreated: 2026-05-20T09:00:00Z\nsource: test\nconfidence: medium\n---\nThe Lisbon trip starts on 3 July\n",
    )
    .unwrap();
    let found = |query: &str| -> Vec<String> {
        let hits = recall_with(dir.path(), query, how(&AUTO, "2026-10-17")).unwrap();
        (hits.iter())
            .filter(|h| h.source == Source::Daily)
            .map(|h| h.path.trim_start_matches("agent/memory/daily/").to_owned())
            .collect()
    };
    let sorted = |query: &str| {
        let mut days = found(query);
        days.sort();
        days
    };
    let (june, july, october) = ("2026-06-10.md", "2026-07-10.md", "2026-10-16.md");
    assert_eq!(sorted("boiler Tomasz"), [june, july, october]);
    assert_eq!(found("boiler quote in June"), [june]);
    assert_eq!(
        found("In July, what did Tomasz say about the boiler"),
        [july]
    );
    assert_eq!(found("boiler yesterday?"), [october]);
    assert_eq!(found("boiler on 2026-07-10"), [july]);
    assert_eq!(sorted("boiler between June and July"), [june, july]);
    assert_eq!(found("the boiler before the Lisbon trip"), [june]);
    assert_eq!(sorted("the boiler after the Lisbon trip"), [july, october]);
    assert!(found("boiler in September").is_empty());
    // A phrase alone lists its days' lines, newest first.
    let hits = recall_with(dir.path(), "this week", how(&AUTO, "2026-10-17")).unwrap();
    assert_eq!(paths(&hits), ["agent/memory/daily/2026-10-16.md"]);
    // Words that name nothing in those days leave them their lines.
    assert_eq!(found("What did I do yesterday?"), [october]);
    // An event no fact dates is words like any other.
    assert_eq!(found("the boiler after the Porto trip").len(), 3);
}

/// One `- 09:00Z chat <n>: <text>` line per day, in daily notes.
fn daily_lines(data_dir: &Path, lines: &[(&str, &str)]) {
    let daily = memory(data_dir).join("daily");
    fs::create_dir_all(&daily).unwrap();
    for (n, (day, text)) in lines.iter().enumerate() {
        let line = format!("- 09:00Z chat {n}: {text}\n");
        fs::write(daily.join(format!("{day}.md")), line).unwrap();
    }
}

#[test]
fn last_week_keeps_its_monday_and_sunday_and_neither_day_beside_them() {
    let dir = tempfile::tempdir().unwrap();
    // 2026-10-17 is a Saturday: last week ran from Monday 5 to Sunday 11 October.
    let days = [
        "2026-10-04",
        "2026-10-05",
        "2026-10-11",
        "2026-10-12",
        "2026-10-17",
    ];
    let lines: Vec<(&str, &str)> = days.iter().map(|d| (*d, "gym session")).collect();
    daily_lines(dir.path(), &lines);
    let hits = recall_with(dir.path(), "gym last week", how(&AUTO, "2026-10-17")).unwrap();
    let mut got = paths(&hits);
    got.sort();
    assert_eq!(
        got,
        [
            "agent/memory/daily/2026-10-05.md",
            "agent/memory/daily/2026-10-11.md"
        ]
    );
}

#[test]
fn an_event_is_dated_from_its_whole_fact_when_the_date_is_on_another_line() {
    let dir = tempfile::tempdir().unwrap();
    daily_lines(
        dir.path(),
        &[
            ("2026-09-30", "packing list: sunscreen and adapters"),
            ("2026-10-05", "packing the souvenirs for home"),
        ],
    );
    let facts = memory(dir.path()).join("facts");
    fs::create_dir_all(&facts).unwrap();
    // The event's name on one line, its date on the next; the front matter's own date is no
    // body text.
    fs::write(
        facts.join("the-lisbon-trip.md"),
        "---\ntype: user\ncreated: 2026-08-01T09:00:00Z\nsource: test\nconfidence: medium\n---\nThe Lisbon trip\nDeparture is on 2026-10-03\n",
    )
    .unwrap();
    let daily = |query: &str| -> Vec<String> {
        let hits = recall_with(dir.path(), query, how(&AUTO, "2026-10-17")).unwrap();
        (hits.iter())
            .filter(|h| h.source == Source::Daily)
            .map(|h| h.path.trim_start_matches("agent/memory/daily/").to_owned())
            .collect()
    };
    assert_eq!(daily("packing before the Lisbon trip"), ["2026-09-30.md"]);
    assert_eq!(daily("packing after the Lisbon trip"), ["2026-10-05.md"]);
}

#[test]
fn a_key_points_at_its_note_line_and_leaves_the_note_as_it_is() {
    let dir = tempfile::tempdir().unwrap();
    note(dir.path(), "chat 1: read_file notes.txt: 12 lines").unwrap();
    let text = "chat 1: read_file travel/booking.txt: Cantina 32 at 20:00";
    let noted = note(dir.path(), text).unwrap().unwrap();
    assert_eq!(noted.line, 2);
    let file = memory(dir.path()).join(format!("daily/{}.md", noted.day));
    let before = fs::read_to_string(&file).unwrap();
    assert_eq!(before.lines().nth(1), Some(noted.text.as_str()));
    let keys = ["dinner reservation".to_owned(), "Ines".to_owned()];
    save_keys(dir.path(), &noted, &keys).unwrap();
    assert_eq!(fs::read_to_string(&file).unwrap(), before);
    let today = noted.day.as_str();
    let hit = recall_with(dir.path(), "the dinner reservation", how(&AUTO, today)).unwrap();
    assert_eq!(hit.len(), 1, "{hit:?}");
    let note_path = format!("agent/memory/daily/{today}.md");
    assert_eq!(
        (&hit[0].path, hit[0].line, hit[0].source),
        (&note_path, 2, Source::Key)
    );
    // The note's text is the value.
    assert_eq!(hit[0].snippet, noted.text);
    // Found by its keys and by its own words, the line counts once.
    let both = recall_with(dir.path(), "dinner Cantina", how(&AUTO, today)).unwrap();
    assert_eq!(both.len(), 1, "{both:?}");
    // New keys for the line replace its old ones.
    save_keys(dir.path(), &noted, &["table for two".to_owned()]).unwrap();
    let gone = recall_with(dir.path(), "reservation", how(&AUTO, today)).unwrap();
    assert!(gone.is_empty(), "{gone:?}");
    let keys_file = memory(dir.path()).join(format!("keys/{today}.txt"));
    assert_eq!(fs::read_to_string(keys_file).unwrap(), "2 table for two\n");
}

#[test]
fn key_replies_parse_to_at_most_three_short_phrases() {
    let ok = |reply: &str, cut: bool| parse_keys(reply, cut).unwrap();
    assert_eq!(
        ok("boiler quote\nTomasz Nowak\n640 EUR", false),
        ["boiler quote", "Tomasz Nowak", "640 EUR"]
    );
    assert_eq!(
        ok(
            "Key phrases:\n1. boiler quote\n2) Tomasz Nowak\n- \"pump\"\n* `640 EUR`.",
            false
        ),
        ["boiler quote", "Tomasz Nowak", "pump"]
    );
    // A comma and a space part phrases; a number's comma does not.
    assert_eq!(
        ok("rent, 1,850 USD; Harbour Lettings", false),
        ["rent", "1,850 USD", "Harbour Lettings"]
    );
    // A bare closing think tag, a Markdown task, a fenced call, a repeat and a summary.
    let messy = "planning</think>\n- [ ] dentist\n```json\n{\"name\": \"remember\"}\n```\nDentist\n\
                 The user booked a dentist appointment for Tuesday morning at ten";
    assert_eq!(ok(messy, false), ["dentist"]);
    assert_eq!(ok("a\nb\nc\nd", false), ["a", "b", "c"]);
    // Cut at the token limit, the last line is unfinished.
    assert_eq!(ok("boiler quote\nTomasz Now", true), ["boiler quote"]);
    for reply in [
        "",
        "none",
        "Key phrases:",
        "The user asked about the boiler and the plumber came back.",
    ] {
        let err = parse_keys(reply, false).unwrap_err();
        assert!(matches!(err, MemoryError::NoKeys(_)), "{reply}: {err:?}");
    }
}

#[test]
fn eight_writers_of_notes_and_keys_lose_no_line() {
    let dir = tempfile::tempdir().unwrap();
    let rounds = 20;
    let start = std::sync::Barrier::new(8);
    std::thread::scope(|s| {
        for w in 0..8 {
            let (start, dir) = (&start, dir.path());
            s.spawn(move || {
                start.wait();
                for r in 0..rounds {
                    let noted = note(dir, &format!("writer {w} round {r}"))
                        .unwrap()
                        .unwrap();
                    save_keys(dir, &noted, &[format!("w{w}r{r}")]).unwrap();
                }
            });
        }
    });
    let all = notes(dir.path()).unwrap();
    assert_eq!(all.len(), 8 * rounds);
    let mut keys = 0;
    for entry in fs::read_dir(memory(dir.path()).join("keys")).unwrap() {
        let path = entry.unwrap().path();
        let day = path.file_stem().unwrap().to_str().unwrap().to_owned();
        for line in fs::read_to_string(&path).unwrap().lines() {
            keys += 1;
            // Each key names the line its writer noted.
            let (at, key) = line.split_once(' ').unwrap();
            let (w, r) = key.trim_start_matches('w').split_once('r').unwrap();
            let noted = (all.iter())
                .find(|n| n.day == day && n.line.to_string() == at)
                .unwrap();
            let wrote = format!("writer {w} round {r}");
            assert!(noted.text.ends_with(&wrote), "{line}: {noted:?}");
        }
    }
    assert_eq!(keys, 8 * rounds);
}

#[test]
fn remembering_a_fact_memory_has_word_for_word_is_refused_before_the_user_is_asked() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    remember(dir.path(), "I run the Mac lanes on an M5", "user", "test").unwrap();
    let registry = Registry::builtin();
    let tool = registry.get("remember").unwrap();
    let again = json!({"text": " I run the Mac lanes on an M5 ", "type": "user"});
    assert_eq!(
        tool.prepare(&again, &ctx).unwrap_err().to_string(),
        "already in memory as [i-run-the-mac-lanes-on]; nothing to save"
    );
    // A changed fact is still an update the user is asked about.
    let changed = json!({"text": "I run the Mac lanes on an M5 Max", "type": "user"});
    assert!(tool.prepare(&changed, &ctx).is_ok());
}

#[test]
fn a_known_fact_with_another_type_or_no_memory_md_line_can_be_saved_again() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    remember(dir.path(), "I run the Mac lanes on an M5", "user", "test").unwrap();
    let registry = Registry::builtin();
    let tool = registry.get("remember").unwrap();
    let same = json!({"text": "I run the Mac lanes on an M5", "type": "user"});
    assert!(tool.prepare(&same, &ctx).is_err());
    // remember would update the type.
    let retyped = json!({"text": "I run the Mac lanes on an M5", "type": "project"});
    assert!(tool.prepare(&retyped, &ctx).is_ok());
    // A crash between the two writes, or a hand edit, lost the line: remember would add it back.
    fs::write(dir.path().join("agent/memory/MEMORY.md"), "").unwrap();
    assert!(tool.prepare(&same, &ctx).is_ok());
}

/// 40 facts of about 200 bytes after one old fact, so the old one is outside `MEMORY_BYTES`;
/// the old fact's slug.
fn old_fact_then_forty(data_dir: &Path) -> String {
    let old = remember(
        data_dir,
        "The user lives in Berlin and is allergic to peanuts",
        "user",
        "test",
    )
    .unwrap()
    .slug;
    for i in 0..40 {
        let text = format!(
            "Fact {i}: {} the end",
            format!("filler{i} ").repeat(180 / (8 + i.to_string().len()))
        );
        remember(data_dir, &text, "project", "test").unwrap();
    }
    assert!(!system_prefix(data_dir).contains(&format!("[{old}]")));
    old
}

#[test]
fn an_updated_fact_moves_to_the_end_and_into_the_system_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let old = old_fact_then_forty(dir.path());
    let long = format!(
        "The user moved to Munich and is allergic to peanuts and shellfish; {}",
        "they carry an epinephrine pen at all times. ".repeat(4)
    );
    update(dir.path(), &old, &long).unwrap();
    assert!(system_prefix(dir.path()).contains("Munich"));
    let last = index_lines(dir.path()).pop().unwrap();
    assert!(last.starts_with(&format!("- [{old}] ")), "{last}");
    assert!(system_prefix(dir.path()).len() <= MEMORY_BYTES + 200);
}

#[test]
fn a_fact_outside_the_budget_can_be_saved_again_into_the_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let old = old_fact_then_forty(dir.path());
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.path().to_path_buf(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    let tool = registry.get("remember").unwrap();
    let again =
        json!({"text": "The user lives in Berlin and is allergic to peanuts", "type": "user"});
    let args = tool.prepare(&again, &ctx).unwrap();
    let result = tool.call(&args, &ctx).unwrap();
    assert_eq!(
        result.split('\u{1e}').next().unwrap(),
        format!("{old}; it replaced: The user lives in Berlin and is allergic to peanuts")
    );
    assert!(system_prefix(dir.path()).contains(&format!("[{old}]")));
    // Now it shows, so the same text is known again.
    assert!(tool.prepare(&again, &ctx).is_err());
}

#[test]
fn a_crlf_fact_file_gives_its_text_without_the_front_matter() {
    let dir = tempfile::tempdir().unwrap();
    let slug = remember(dir.path(), "The user drinks green tea", "user", "test")
        .unwrap()
        .slug;
    let path = memory(dir.path()).join(format!("facts/{slug}.md"));
    let crlf = fs::read_to_string(&path).unwrap().replace('\n', "\r\n");
    fs::write(&path, crlf).unwrap();
    let fact = list(dir.path()).unwrap().remove(0);
    assert_eq!(
        (fact.text.as_str(), fact.kind.as_str()),
        ("The user drinks green tea", "user")
    );
    assert_eq!(
        update(dir.path(), &slug, "The user drinks black tea").unwrap(),
        "The user drinks green tea"
    );
}

#[test]
fn list_skips_a_fact_whose_file_is_gone_or_empty() {
    let dir = tempfile::tempdir().unwrap();
    let gone = remember(dir.path(), "The user drinks green tea", "user", "test")
        .unwrap()
        .slug;
    let empty = remember(dir.path(), "The user cycles to work", "user", "test")
        .unwrap()
        .slug;
    let kept = remember(dir.path(), "The user plays the cello", "user", "test")
        .unwrap()
        .slug;
    // As a `forget` in another chat leaves it between the index read and the file read.
    fs::remove_file(memory(dir.path()).join(format!("facts/{gone}.md"))).unwrap();
    fs::write(memory(dir.path()).join(format!("facts/{empty}.md")), "").unwrap();
    let slugs: Vec<String> = list(dir.path())
        .unwrap()
        .into_iter()
        .map(|f| f.slug)
        .collect();
    assert_eq!(slugs, [kept]);
}

#[test]
fn updating_a_hand_written_fact_gives_it_remembers_front_matter() {
    let dir = tempfile::tempdir().unwrap();
    let mem = memory(dir.path());
    fs::create_dir_all(mem.join("facts")).unwrap();
    fs::write(mem.join("facts/tabs.md"), "The user prefers tabs\n").unwrap();
    fs::write(mem.join("MEMORY.md"), "- [tabs] The user prefers tabs\n").unwrap();
    assert_eq!(
        update(dir.path(), "tabs", "The user prefers tabs, width 4").unwrap(),
        "The user prefers tabs"
    );
    let file = fs::read_to_string(mem.join("facts/tabs.md")).unwrap();
    assert!(file.starts_with("---\ntype: user\ncreated: 20"), "{file}");
    assert!(
        file.contains("\nsource: the desktop app\nconfidence: medium\n---\n"),
        "{file}"
    );
    let fact = list(dir.path()).unwrap().remove(0);
    assert_eq!(
        (fact.kind.as_str(), fact.text.as_str()),
        ("user", "The user prefers tabs, width 4")
    );
}

#[test]
fn list_skips_a_fact_file_it_cannot_read_and_keeps_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let kept = remember(dir.path(), "The user plays the cello", "user", "test")
        .unwrap()
        .slug;
    // A folder where a fact's file would be: reading it fails with something other than
    // NotFound.
    let mem = memory(dir.path());
    fs::create_dir_all(mem.join("facts/odd.md")).unwrap();
    let index = fs::read_to_string(mem.join("MEMORY.md")).unwrap();
    fs::write(
        mem.join("MEMORY.md"),
        format!("- [odd] An odd one\n{index}"),
    )
    .unwrap();
    let slugs: Vec<String> = list(dir.path())
        .unwrap()
        .into_iter()
        .map(|f| f.slug)
        .collect();
    assert_eq!(slugs, [kept]);
}

#[test]
fn undo_refuses_a_fact_changed_since_its_save() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path();
    let maya = "The user's sister Maya lives in Austin";
    let saved = remember(data, maya, "user", "chat 1 turn 1").unwrap();
    let receipt = saved.clone().receipt(maya);
    // The user edits it since: the Undo leaves the edit.
    update(data, &saved.slug, "Maya lives in Denver, since 2025").unwrap();
    let e = larch_agent::memory::undo(data, &receipt).unwrap_err();
    assert!(matches!(e, MemoryError::Changed(_)), "{e}");
    assert!(e.to_string().contains("changed since"), "{e}");
    assert_eq!(
        list(data).unwrap()[0].text,
        "Maya lives in Denver, since 2025"
    );

    // Forgotten, then saved again under the same slug: the old receipt is not the new fact's.
    forget(data, &saved.slug).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let again = remember(data, maya, "user", "chat 2 turn 1").unwrap();
    assert_eq!(again.slug, saved.slug);
    let e = larch_agent::memory::undo(data, &receipt).unwrap_err();
    assert!(matches!(e, MemoryError::Changed(_)), "{e}");
    assert_eq!(list(data).unwrap().len(), 1);
    // Its own receipt undoes it.
    larch_agent::memory::undo(data, &again.receipt(maya)).unwrap();
    assert!(list(data).unwrap().is_empty());
}

#[test]
fn undo_of_an_update_restores_its_text_type_and_source() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path();
    let first = remember(
        data,
        "The user drinks green tea",
        "preference",
        "the desktop app",
    )
    .unwrap();
    let text = "the user drinks green tea!";
    let second = remember(data, text, "reference", "chat 1 turn 1").unwrap();
    assert_eq!(second.slug, first.slug);
    let fact = &list(data).unwrap()[0];
    assert_eq!(
        (fact.kind.as_str(), fact.source.as_str()),
        ("reference", "chat 1 turn 1")
    );
    larch_agent::memory::undo(data, &second.receipt(text)).unwrap();
    let fact = &list(data).unwrap()[0];
    assert_eq!(fact.text, "The user drinks green tea");
    assert_eq!(
        (
            fact.kind.as_str(),
            fact.source.as_str(),
            fact.created.as_str()
        ),
        ("preference", "the desktop app", first.created.as_str())
    );
}

#[test]
fn two_updates_undo_newest_first_and_the_oldest_waits() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path();
    let new = remember(data, "The desk is 4 feet wide", "user", "test").unwrap();
    let slug = new.slug.clone();
    let r0 = new.receipt("The desk is 4 feet wide");
    let mut receipts = Vec::new();
    for width in ["2", "8"] {
        let text = format!("The desk is {width} feet wide");
        let old = update(data, &slug, &text).unwrap();
        let fact = &list(data).unwrap()[0];
        receipts.push(larch_agent::memory::Receipt {
            slug: slug.clone(),
            text,
            old: Some(old),
            was: Some(larch_agent::memory::Was {
                kind: fact.kind.clone(),
                source: fact.source.clone(),
            }),
            created: fact.created.clone(),
        });
    }
    let text = || list(data).unwrap()[0].text.clone();
    // Oldest first: refused, as the fact has changed since.
    assert!(matches!(
        larch_agent::memory::undo(data, &receipts[0]),
        Err(MemoryError::Changed(_))
    ));
    assert!(matches!(
        larch_agent::memory::undo(data, &r0),
        Err(MemoryError::Changed(_))
    ));
    assert_eq!(text(), "The desk is 8 feet wide");
    // Newest first, each in turn, back to the start.
    larch_agent::memory::undo(data, &receipts[1]).unwrap();
    assert_eq!(text(), "The desk is 2 feet wide");
    larch_agent::memory::undo(data, &receipts[0]).unwrap();
    assert_eq!(text(), "The desk is 4 feet wide");
    larch_agent::memory::undo(data, &r0).unwrap();
    assert!(list(data).unwrap().is_empty());
}
