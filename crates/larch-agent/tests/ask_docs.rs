//! The ask-docs skill on the CPU: gather and rank on the document set, the folder check, the
//! chunks, their headers and windows, the workers' replies, the line check, the citations, the
//! conflict rule, scripted runs and the bench validators.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

use larch::{EngineError, Finish, Larch};
use larch_agent::{
    audit::{AuditEvent, EventKind},
    bench::{self, CitedQuote, Docs, Question, docs_metrics},
    parse::Family,
    skill::{Skill, SkillError, SkillInput, SkillRun},
    skills::ask_docs::{
        AnswerRequest, AskDocs, CHUNK_BYTES, Chunk, Cited, DOCS_WORKERS, Excerpt, Fact, Facts,
        FactsError, Gathered, HEADER_BYTES, HEADING_BYTES, MAX_FILE_BYTES, MAX_PDFS, NOT_FOUND,
        WORKER_SYSTEM, answer_request, cite, conflicts, cover_names, disagreeing, excerpt,
        facts_grammar, file_words, gather, gather_in, head, missing_names, page_lines, parse_facts,
        pointers, question_names, rank, rank_pages, with_pages, without_conflict_talk,
    },
    tool::{ToolCtx, ToolError},
    workers::Workers,
};
use larch_engine::{EngineOptions, start_with};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{
    ChatTemplate, FakeModel, Model, ModelConfig, StepBatch, Tokenizer,
    gguf::{Gguf, Value, write},
};
use serde_json::json;

/// The committed document set (bench/agent/docs, written by make_docs.py).
const DOCS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/agent/docs");

/// Only `allowed` may be read; the data folder is a fresh one inside `tmp`.
fn ctx(tmp: &Path, allowed: &Path) -> ToolCtx {
    ToolCtx {
        allowed: vec![allowed.to_path_buf()],
        data_dir: tmp.join("data"),
        source: String::new(),
    }
}

fn questions() -> Vec<Question> {
    bench::doc_set(Path::new(DOCS)).unwrap().questions
}

fn symlink(target: &Path, link: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(target, link).unwrap();
}

/// The file's text from the chunk's first line on.
fn from_line(chunk: &Chunk, root: &Path) -> String {
    let text = fs::read_to_string(root.join(&chunk.file)).unwrap();
    text.split_inclusive('\n')
        .skip(chunk.line as usize - 1)
        .collect()
}

#[test]
fn gather_reads_the_document_set_and_skips_hidden_files_and_pdfs() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    assert_eq!((g.files, g.skipped, g.texts.len()), (24, 0, 24));
    // Chunks of about 256 tokens: 116 of them in the 24 documents.
    assert_eq!(g.chunks.len(), 116);
    for chunk in &g.chunks {
        assert!(chunk.text.len() <= CHUNK_BYTES, "{chunk:?}");
        // A chunk is the file's own text, starting at the line and the byte it names.
        assert!(
            from_line(chunk, Path::new(DOCS)).starts_with(&chunk.text),
            "{}:{}",
            chunk.file.display(),
            chunk.line
        );
        assert_eq!(
            g.texts[&chunk.file].get(chunk.start..chunk.start + chunk.text.len()),
            Some(chunk.text.as_str())
        );
    }
    // The hidden note and the PDF hold answers that no document gives: neither was read.
    assert!(
        !g.chunks
            .iter()
            .any(|c| c.text.contains("pinecone-42") || c.text.contains("X1234567"))
    );
}

/// The lease as a text PDF (bench/agent/docs-pdf, written by make_pdf.py) and the passport scan.
const DOCS_PDF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/agent/docs-pdf");

#[test]
fn a_pdf_is_read_with_its_lines_and_a_scan_is_named_as_without_text() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = ctx(tmp.path(), Path::new(DOCS_PDF));
    let g = gather(&ctx, DOCS_PDF).unwrap();
    assert_eq!(
        head(&g, DOCS_PDF),
        "Read 1 file (1 pdf, 0 md, 0 txt); no text in scan-passport.pdf"
    );
    // The planted facts are on the lines the questions name, as the workers' numbered lines
    // show them, and the scan's number is never read.
    for q in bench::doc_set(Path::new(DOCS_PDF)).unwrap().questions {
        for f in q.facts {
            let line = g.texts[&f.file].lines().nth(f.line as usize - 1).unwrap();
            assert!(line.starts_with(&f.quote), "{}: {line}", q.id);
        }
    }
    assert!(!g.texts.values().any(|t| t.contains("X1234567")));
    // The text is kept, so a follow-up reads it from the cache; so is the scan's none, so it is
    // tried once.
    let cache = tmp.path().join("data/agent/extract");
    let mut cached: Vec<_> = fs::read_dir(&cache)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    cached.sort_by_key(|c| fs::metadata(c).unwrap().len());
    assert_eq!(cached.len(), 2);
    assert_eq!(fs::read_to_string(&cached[0]).unwrap(), "");
    fs::write(&cached[1], "Monthly rent: from the cache.\n").unwrap();
    let again = gather(&ctx, DOCS_PDF).unwrap();
    assert_eq!(
        again.texts[Path::new("lease-alder-street.pdf")],
        "Monthly rent: from the cache.\n"
    );
}

#[test]
fn a_changed_pdf_misses_the_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("docs");
    fs::create_dir(&folder).unwrap();
    let pdf = folder.join("lease.pdf");
    fs::copy(Path::new(DOCS_PDF).join("lease-alder-street.pdf"), &pdf).unwrap();
    let ctx = ctx(tmp.path(), &folder);
    gather(&ctx, folder.to_str().unwrap()).unwrap();
    let file = fs::File::options().append(true).open(&pdf).unwrap();
    file.set_modified(std::time::SystemTime::UNIX_EPOCH)
        .unwrap();
    gather(&ctx, folder.to_str().unwrap()).unwrap();
    assert_eq!(
        fs::read_dir(tmp.path().join("data/agent/extract"))
            .unwrap()
            .count(),
        2
    );
}

#[test]
fn the_first_line_says_why_nothing_was_read() {
    let tmp = tempfile::tempdir().unwrap();
    let docx = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/agent/docs-docx");
    let g = gather(&ctx(tmp.path(), Path::new(docx)), docx).unwrap();
    assert_eq!(
        head(&g, docx),
        format!("No readable files in {docx} (saw: .docx, .json)")
    );
    let empty = tmp.path().join("empty");
    fs::create_dir(&empty).unwrap();
    let empty = empty.to_str().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(empty)), empty).unwrap();
    assert_eq!(head(&g, empty), format!("No files match {empty}"));
    let scan = Path::new(DOCS_PDF).join("scan-passport.pdf");
    let scan = scan.to_str().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(scan)), scan).unwrap();
    assert_eq!(head(&g, scan), "No text in scan-passport.pdf");
}

#[test]
fn an_attached_file_is_read_alone() {
    // Attaching a file allows that file, not the folder it is in.
    let tmp = tempfile::tempdir().unwrap();
    let file = Path::new(DOCS).join("lease-alder-street.md");
    let ctx = ctx(tmp.path(), &file);
    let g = gather(&ctx, file.to_str().unwrap()).unwrap();
    assert_eq!(head(&g, ""), "Read 1 file (0 pdf, 1 md, 0 txt)");
    assert!(matches!(gather(&ctx, DOCS), Err(ToolError::Denied(_))));
}

#[test]
fn a_folder_outside_the_allowed_folders_is_denied_before_any_read() {
    let tmp = tempfile::tempdir().unwrap();
    let (allowed, outside) = (tmp.path().join("allowed"), tmp.path().join("outside"));
    fs::create_dir(&allowed).unwrap();
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("secret.md"), "The safe code is 1234.\n").unwrap();
    let ctx = ctx(tmp.path(), &allowed);
    for folder in [outside.to_str().unwrap(), "../outside", "../missing"] {
        assert!(
            matches!(gather(&ctx, folder), Err(ToolError::Denied(_))),
            "{folder}"
        );
    }
}

#[test]
fn a_paragraph_over_the_cap_is_cut_at_a_character_boundary() {
    let tmp = tempfile::tempdir().unwrap();
    // One line of 2,001 bytes: "a" puts every two-byte "é" at an odd offset, so byte 1,024
    // falls inside one.
    let long = format!("a{}", "é".repeat(1000));
    fs::write(tmp.path().join("long.md"), format!("{long}\n\nnext\n")).unwrap();
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    let texts: Vec<_> = g.chunks.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            &long[..1023],
            format!("{}\n\nnext\n", &long[1023..]).as_str()
        ]
    );
    assert!(g.chunks.iter().all(|c| c.line == 1));
    assert_eq!(g.chunks[1].start, 1023);
}

/// Paragraph `i` of a file of many: four lines of 67 bytes, so three pack into a chunk.
fn paragraph(i: usize) -> String {
    (0..4)
        .map(|l| {
            format!("Paragraph {i:02}, line {l}: the boiler was serviced and the gas checked.\n")
        })
        .collect()
}

#[test]
fn chunks_of_1024_bytes_keep_paragraphs_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let paragraphs: Vec<_> = (0..10).map(paragraph).collect();
    let folder = folder(tmp.path(), &[("log.md", &paragraphs.join("\n"))]);
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    // 268 bytes a paragraph and a blank line between: three to a chunk, then the last one.
    assert_eq!(g.chunks.len(), 4, "{g:?}");
    let mut lines = 1;
    for (i, chunk) in g.chunks.iter().enumerate() {
        assert!(chunk.text.len() <= CHUNK_BYTES, "{chunk:?}");
        // Each chunk is whole paragraphs, the next ones in the file.
        let n = if i < 3 { 3 } else { 1 };
        let want = paragraphs[i * 3..i * 3 + n].join("\n");
        assert_eq!(chunk.text, want);
        assert_eq!(chunk.line, lines);
        lines += 15;
    }
    // A paragraph over the cap is cut at its last line break below it, so each chunk is whole
    // lines: 15 lines of 67 bytes, then the other 5.
    let long: String = (0..5).map(paragraph).collect::<String>();
    fs::write(folder.join("long.md"), &long).unwrap();
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    let cut: Vec<_> = (g.chunks.iter())
        .filter(|c| c.file == Path::new("long.md"))
        .map(|c| (c.line, c.text.len()))
        .collect();
    assert_eq!(cut, [(1, 15 * 67), (16, 5 * 67)]);
}

#[test]
fn a_link_out_of_the_allowed_folders_is_skipped_and_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("docs");
    fs::create_dir(&folder).unwrap();
    fs::write(tmp.path().join("secret.md"), "The safe code is 1234.\n").unwrap();
    fs::write(folder.join("notes.md"), "Buy milk.\n").unwrap();
    symlink(&tmp.path().join("secret.md"), &folder.join("link.md"));
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (1, 1));
    assert_eq!(g.chunks.len(), 1);
    assert_eq!(g.chunks[0].file, PathBuf::from("notes.md"));
}

#[test]
fn a_link_to_a_hidden_file_or_another_type_is_skipped_and_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(
        tmp.path(),
        &[
            ("notes.md", "Buy milk.\n"),
            (".secret.md", "The safe code is 1234.\n"),
            ("key.json", "{\"key\": \"abc\"}\n"),
        ],
    );
    symlink(&folder.join(".secret.md"), &folder.join("a.md"));
    symlink(&folder.join("key.json"), &folder.join("b.md"));
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (1, 2));
    let files: Vec<_> = g.chunks.iter().map(|c| c.file.clone()).collect();
    assert_eq!(files, [PathBuf::from("notes.md")]);
}

#[test]
fn the_agents_own_data_folder_is_not_gathered() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("notes.md", "Buy milk.\n")]);
    // The data folder inside the folder asked about, holding a daily note, and a link to it.
    let ctx = ToolCtx {
        data_dir: folder.join("data"),
        ..ctx(tmp.path(), &folder)
    };
    let daily = folder.join("data/agent/memory/daily");
    fs::create_dir_all(&daily).unwrap();
    fs::write(daily.join("2026-10-15.md"), "- 09:00Z ask-docs on docs\n").unwrap();
    symlink(&daily.join("2026-10-15.md"), &folder.join("today.md"));
    let g = gather(&ctx, folder.to_str().unwrap()).unwrap();
    // The note is not a file to read; the link to it is one, not read.
    assert_eq!((g.files, g.skipped), (1, 1));
    assert_eq!(g.chunks[0].file, PathBuf::from("notes.md"));
}

#[test]
fn documents_inside_the_data_folder_are_read_outside_its_agent_folder() {
    // The data folder above the documents, as `--data-dir <tmp> --folder <tmp>/docs` gives.
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", "Rent: $1,450.\n")]);
    let notes = tmp.path().join("agent/memory/daily");
    fs::create_dir_all(&notes).unwrap();
    fs::write(notes.join("2026-10-15.md"), "- 09:00Z ask-docs on docs\n").unwrap();
    let ctx = ToolCtx {
        allowed: vec![tmp.path().into()],
        data_dir: tmp.path().into(),
        source: String::new(),
    };
    let g = gather(&ctx, folder.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (1, 0));
    // Asked about the whole data folder, the agent's notes stay out.
    let g = gather(&ctx, tmp.path().to_str().unwrap()).unwrap();
    let files: Vec<_> = g.chunks.iter().map(|c| c.file.clone()).collect();
    assert_eq!(files, [PathBuf::from("docs/lease.md")]);
}

#[test]
fn a_link_into_a_hidden_folder_is_skipped_and_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("notes.md", "Buy milk.\n")]);
    for sub in [".private", "plans"] {
        fs::create_dir(folder.join(sub)).unwrap();
    }
    fs::write(folder.join(".private/diary.md"), "The safe code is 1234.\n").unwrap();
    fs::write(folder.join("plans/hall.md"), "Paint the hall.\n").unwrap();
    symlink(&folder.join(".private/diary.md"), &folder.join("a.md"));
    // A link into a folder that is not hidden is read, and so is one into an allowed folder below
    // a hidden one, as iCloud Drive is below ~/Library.
    symlink(&folder.join("plans/hall.md"), &folder.join("b.md"));
    let cloud = tmp.path().join(".cloud/notes");
    fs::create_dir_all(&cloud).unwrap();
    fs::write(cloud.join("trip.md"), "Train at 9:40.\n").unwrap();
    symlink(&cloud.join("trip.md"), &folder.join("c.md"));
    let ctx = ToolCtx {
        allowed: vec![folder.clone(), cloud],
        ..ctx(tmp.path(), &folder)
    };
    let g = gather(&ctx, folder.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (4, 1));
    assert!(!g.chunks.iter().any(|c| c.text.contains("1234")), "{g:?}");
    // A folder the user names below a hidden one is read, but a link out of it into the hidden
    // folder is not: the user named the folder, not the hidden one above it.
    let named = tmp.path().join(".vault/notes");
    fs::create_dir_all(&named).unwrap();
    fs::write(named.join("rent.md"), "Rent is due on the 1st.\n").unwrap();
    fs::write(
        tmp.path().join(".vault/keys.md"),
        "The key is under the mat.\n",
    )
    .unwrap();
    symlink(&tmp.path().join(".vault/keys.md"), &named.join("keys.md"));
    let whole = ToolCtx {
        allowed: vec![tmp.path().to_path_buf()],
        ..ctx
    };
    let g = gather(&whole, named.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (1, 1));
}

#[test]
fn files_without_text_say_so() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("a.md", ""), ("b.txt", " \n\n")]);
    let run = ask_rent(tmp.path(), &folder);
    let note = "_The 2 files read hold no text._";
    assert_eq!(run.markdown, format!("{NOT_FOUND}\n\n{note}"));
    assert_eq!(run.answer, NOT_FOUND);
}

#[test]
fn a_link_to_nothing_or_a_folder_that_cannot_be_listed_is_skipped_and_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("notes.md", "Buy milk.\n")]);
    symlink(&folder.join("gone.md"), &folder.join("old.md"));
    // Two links that lead to each other.
    symlink(&folder.join("b.md"), &folder.join("a.md"));
    symlink(&folder.join("a.md"), &folder.join("b.md"));
    let locked = folder.join("locked");
    fs::create_dir(&locked).unwrap();
    fs::write(locked.join("plan.md"), "Paint the hall.\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    }
    // An administrator (root, as in a container) still lists it, and so does Windows.
    let listed = fs::read_dir(&locked).is_ok();
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap());
    // The answer names a folder it could not list.
    let run = (!listed).then(|| ask_rent(tmp.path(), &folder));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let g = g.unwrap();
    // The two links and the loop are files not read; the folder is no file.
    assert_eq!((g.files, g.skipped), (1 + usize::from(listed), 3));
    let unlisted: Vec<PathBuf> = if listed {
        vec![]
    } else {
        vec!["locked".into()]
    };
    assert_eq!(g.unlisted, unlisted);
    if let Some(run) = run {
        assert!(
            run.markdown.contains("_3 files were not read")
                && run
                    .markdown
                    .contains("_This folder could not be listed: locked._"),
            "{}",
            run.markdown
        );
    }
}

#[test]
fn past_200_files_the_rest_are_counted_not_read() {
    let tmp = tempfile::tempdir().unwrap();
    for i in 0..201 {
        fs::write(tmp.path().join(format!("{i:03}.md")), "Note.\n").unwrap();
    }
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.skipped), (200, 1));
    assert_eq!(g.chunks.last().unwrap().file, PathBuf::from("199.md"));
}

#[test]
fn files_the_system_hides_are_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(
        tmp.path(),
        &[
            ("notes.md", "Buy milk.\n"),
            ("pin.md", "The pin is 1234.\n"),
        ],
    );
    // macOS and Windows hide a file by a flag, as they hide ~/Library and AppData.
    let pin = folder.join("pin.md");
    let flag = if cfg!(target_os = "macos") {
        Some(
            std::process::Command::new("chflags")
                .arg("hidden")
                .arg(&pin)
                .status(),
        )
    } else if cfg!(windows) {
        Some(
            std::process::Command::new("attrib")
                .arg("+h")
                .arg(&pin)
                .status(),
        )
    } else {
        None
    };
    let flagged = flag.is_some();
    assert!(flag.is_none_or(|s| s.unwrap().success()));
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    let files: Vec<_> = g.chunks.iter().map(|c| c.file.clone()).collect();
    let mut want = vec![PathBuf::from("notes.md")];
    if !flagged {
        want.push("pin.md".into());
    }
    assert_eq!(files, want);
}

#[test]
fn every_planted_quote_occurs_exactly_once_in_its_file() {
    let qs = questions();
    let mix = [0, 1, 2].map(|n| qs.iter().filter(|q| q.facts.len() == n).count());
    // Four that no document answers, 14 from one document, 6 from two.
    assert_eq!(mix, [4, 14, 6]);
    for q in &qs {
        for p in &q.facts {
            let text = fs::read_to_string(Path::new(DOCS).join(&p.file)).unwrap();
            assert_eq!(text.matches(&p.quote).count(), 1, "{}: {}", q.id, p.quote);
            let line = text[..text.find(&p.quote).unwrap()].matches('\n').count() + 1;
            assert_eq!(p.line as usize, line, "{}: {}", q.id, p.quote);
        }
    }
}

#[test]
fn ranking_puts_every_planted_chunk_in_the_chunks_the_defaults_send() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    // The production defaults: config.toml's and the skill's, the chunk alone (window 0).
    let top = larch_agent::config::Config::default().docs_workers;
    assert_eq!((top, AskDocs::default().docs_workers), (7, 7));
    assert_eq!(larch_agent::config::Config::default().docs_window, 0);
    assert!(g.chunks.len() > top);
    let mut missed = Vec::new();
    for q in questions() {
        // At most `top`: only the chunks that share a word with the question, stop words aside.
        let chosen = rank(&g.chunks, &q.question, top).unwrap();
        assert!(!chosen.is_empty() && chosen.len() <= top, "{}", q.id);
        // Best first, each with its score.
        let scores: Vec<f64> = chosen.iter().map(|&(_, s)| s.unwrap()).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
        for fact in &q.facts {
            let found = chosen.iter().any(|&(i, _)| {
                g.chunks[i].file == fact.file && g.chunks[i].text.contains(&fact.quote)
            });
            if !found {
                missed.push((q.id.clone(), fact.file.clone()));
            }
        }
    }
    // q18, "Can my cat Miso live with me at Alder Street, and how much does Miso weigh?", is the
    // case: with "how much" in the query the washer manual's prices ranked the lease's pet clause
    // eighth.
    assert!(missed.is_empty(), "{missed:?}");
}

#[test]
fn a_small_folder_sends_every_chunk() {
    let tmp = tempfile::tempdir().unwrap();
    for (name, text) in [
        ("a.md", "Apples.\n"),
        ("b.txt", "Bread.\n"),
        ("c.md", "Cheese.\n"),
    ] {
        fs::write(tmp.path().join(name), text).unwrap();
    }
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    // No word of the question is in any chunk; all three still go, with no score.
    assert_eq!(
        rank(&g.chunks, "When does the train leave?", 3).unwrap(),
        [(0, None), (1, None), (2, None)]
    );
}

#[test]
fn docs_workers_7_and_14_pick_the_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    let q = "How much do I pay each month for the Alder Street rent and the renters insurance?";
    for top in [7, 14] {
        assert_eq!(rank(&g.chunks, q, top).unwrap().len(), top);
    }
    // In a run, one worker each, the default 7 when config.toml names none.
    let files: Vec<_> = (1..=20)
        .map(|i| {
            (
                format!("n{i:02}.md"),
                format!("Note {i}: the rent was paid.\n"),
            )
        })
        .collect();
    let files: Vec<_> = files
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    let folder = folder(tmp.path(), &files);
    for (workers, want) in [(None, 7), (Some(7), 7), (Some(14), 14)] {
        let skill = AskDocs {
            docs_workers: workers.unwrap_or(AskDocs::default().docs_workers),
            ..AskDocs::default()
        };
        let run = ask_with(skill, tmp.path(), &folder, "How much is the rent? Nothing.");
        let turns = run
            .steps
            .iter()
            .filter(|e| e.step.as_deref() == Some("worker"));
        assert_eq!(turns.count(), want, "{workers:?}");
        assert_eq!(run.answer, NOT_FOUND);
    }
}

/// The chunks of a Markdown file with headings two deep, each heading on a chunk of its own.
fn headed() -> Gathered {
    let tmp = tempfile::tempdir().unwrap();
    let pad = |name: &str| format!("{name} {}\n", "word ".repeat(200));
    let text = format!(
        "# Lease\n\n{}\n## Rent\n\n{}\n### Late fee\n\n{}\n## Pets\n\n{}",
        pad("Intro"),
        pad("Rent"),
        pad("Fee"),
        pad("Pets")
    );
    let folder = folder(
        tmp.path(),
        &[("lease.md", &text), ("router.txt", "Wi-Fi password: abc\n")],
    );
    gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap()
}

#[test]
fn a_1_mib_one_line_heading_keeps_every_header_short() {
    // The largest file gather reads, one heading line of it: 1,024 chunks, each under it.
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    fs::create_dir(&docs).unwrap();
    let text = format!("# {}\n", "A".repeat((1 << 20) - 3));
    assert_eq!(text.len() as u64, MAX_FILE_BYTES);
    fs::write(docs.join("big.md"), &text).unwrap();
    let g = gather(&ctx(tmp.path(), &docs), docs.to_str().unwrap()).unwrap();
    assert_eq!(g.chunks.len(), 1024);
    let titles: usize = g
        .chunks
        .iter()
        .map(|c| c.heading.iter().map(String::len).sum::<usize>())
        .sum();
    assert_eq!(titles, 1024 * HEADING_BYTES);
    // One list, shared by every chunk.
    assert!(
        g.chunks
            .iter()
            .all(|c| std::sync::Arc::ptr_eq(&c.heading, &g.chunks[0].heading))
    );
    let headers: Vec<String> = g.chunks.iter().map(Chunk::header).collect();
    assert!(
        headers.iter().all(|h| h.len() <= HEADER_BYTES),
        "{}",
        headers[0].len()
    );
    assert_eq!(
        headers[0],
        format!("# big.md > {}", "A".repeat(HEADING_BYTES))
    );
    // A cut never splits a character: a title of three-byte characters keeps 40 of them.
    let wide = format!("# {}\n\nRent.\n", "€".repeat(100));
    fs::write(docs.join("big.md"), &wide).unwrap();
    let g = gather(&ctx(tmp.path(), &docs), docs.to_str().unwrap()).unwrap();
    assert_eq!(*g.chunks[0].heading, ["€".repeat(HEADING_BYTES / 3)]);
    // Six levels of long headings: the header stops at HEADER_BYTES, on a character boundary.
    let deep: String = (1..=6)
        .map(|l| format!("{} {}\n\n", "#".repeat(l), "é".repeat(100)))
        .collect();
    fs::write(docs.join("big.md"), deep + "Rent.\n").unwrap();
    let g = gather(&ctx(tmp.path(), &docs), docs.to_str().unwrap()).unwrap();
    let last = g.chunks.last().unwrap();
    // Three titles of 120 bytes are past it already.
    assert!(last.heading.len() >= 3, "{}", last.heading.len());
    let header = last.header();
    assert!(
        header.len() <= HEADER_BYTES && header.len() > HEADER_BYTES - 2,
        "{}",
        header.len()
    );
    assert!(header.starts_with("# big.md > éé"), "{header}");
}

/// A name the question gives that only an unranked chunk holds is read anyway: the chunk naming
/// the most of them takes the worst slot that names none, and the better half stays.
#[test]
fn every_name_the_question_gives_is_read_once() {
    let tmp = tempfile::tempdir().unwrap();
    let crystal = "The crystal pins break when the crystal pins move.\n";
    for n in 0..4 {
        fs::write(tmp.path().join(format!("crystal{n}.md")), crystal).unwrap();
    }
    fs::write(tmp.path().join("pinout.md"), "VQFN pins.\n").unwrap();
    fs::write(
        tmp.path().join("packages.md"),
        "WQFN is RUK; VQFN is RGE.\n",
    )
    .unwrap();
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    let at = |name: &str| {
        (g.chunks.iter())
            .position(|c| c.file.ends_with(name))
            .unwrap()
    };
    let q = "Which crystal pins break from the WQFN (RUK) to the VQFN?";
    let ranked: Vec<_> = (0..4)
        .map(|n| (at(&format!("crystal{n}.md")), Some(1.0)))
        .collect();
    let chosen: Vec<_> = (cover_names(&g.chunks, ranked.clone(), q, 4).iter())
        .map(|c| c.0)
        .collect();
    assert_eq!(
        chosen,
        [ranked[0].0, ranked[1].0, ranked[2].0, at("packages.md")]
    );
    // With room, nothing goes; with every name read, nothing changes.
    assert_eq!(cover_names(&g.chunks, ranked.clone(), q, 5).len(), 5);
    assert_eq!(
        cover_names(&g.chunks, ranked.clone(), "Which pins break?", 4),
        ranked
    );
    // Every pick names WQFN: the worst of the lower half still makes way for the VQFN chunk, as
    // the others keep WQFN read.
    let wqfn = "WQFN crystal pins.\n";
    for n in 0..4 {
        fs::write(tmp.path().join(format!("crystal{n}.md")), wqfn).unwrap();
    }
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    let at = |name: &str| {
        (g.chunks.iter())
            .position(|c| c.file.ends_with(name))
            .unwrap()
    };
    let ranked: Vec<_> = (0..4)
        .map(|n| (at(&format!("crystal{n}.md")), Some(1.0)))
        .collect();
    let chosen: Vec<_> = (cover_names(&g.chunks, ranked.clone(), "WQFN or VQFN?", 4).iter())
        .map(|c| c.0)
        .collect();
    assert_eq!(chosen[..3], [ranked[0].0, ranked[1].0, ranked[2].0]);
    assert!(g.chunks[chosen[3]].text.contains("VQFN"), "{chosen:?}");
}

#[test]
fn a_question_of_only_stop_words_still_ranks_by_them() {
    let tmp = tempfile::tempdir().unwrap();
    for (name, text) in [
        ("a.md", "Is it ready?\n"),
        ("b.md", "Bread.\n"),
        ("c.md", "Cheese.\n"),
        ("d.md", "Dates.\n"),
    ] {
        fs::write(tmp.path().join(name), text).unwrap();
    }
    let g = gather(&ctx(tmp.path(), tmp.path()), tmp.path().to_str().unwrap()).unwrap();
    let chosen = rank(&g.chunks, "Is it?", 3).unwrap();
    assert_eq!(chosen.iter().map(|&(i, _)| i).collect::<Vec<_>>(), [0]);
}

#[test]
fn the_header_holds_the_file_and_the_heading_path() {
    let g = headed();
    let headers: Vec<_> = g.chunks.iter().map(|c| (c.line, c.header())).collect();
    assert_eq!(
        headers,
        [
            (1, "# lease.md > Lease".to_string()),
            (5, "# lease.md > Lease > Rent".into()),
            (9, "# lease.md > Lease > Rent > Late fee".into()),
            // A heading of the same level closes the one before and those under it.
            (13, "# lease.md > Lease > Pets".into()),
            // A file without headings is its name alone.
            (1, "# router.txt".into()),
        ]
    );
    // The worker reads the header first.
    let rendered = excerpt(&g, &g.chunks[2], 0).render();
    assert!(
        rendered.starts_with(
            "# lease.md > Lease > Rent > Late fee\n9: ### Late fee\n10: \n11: Fee word"
        ),
        "{rendered}"
    );
}

#[test]
fn the_header_decides_the_cabin_against_the_home_router() {
    // The same words in the cabin's notes and the router's, a chunk below the title, which holds
    // the section's heading: only the header says which one is the cabin's. The router's file
    // comes first, so a tie picks it.
    let tmp = tempfile::tempdir().unwrap();
    let intro = format!(
        "{}\n",
        "Plans for the weekend away with friends. ".repeat(24)
    );
    let cabin =
        format!("# Pine Hollow cabin\n\n{intro}\n## Getting in\n\nWi-Fi password: pine-77\n");
    let home =
        format!("# Home internet\n\n{intro}\n## Getting online\n\nWi-Fi password: maple-58\n");
    let folder = folder(
        tmp.path(),
        &[("home-router.md", &home), ("pine-cabin.md", &cabin)],
    );
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    assert_eq!(g.chunks.len(), 4, "{g:?}");
    let best = |q: &str| {
        let chosen = rank(&g.chunks, q, 1).unwrap();
        let c = &g.chunks[chosen[0].0];
        (c.header(), c.text.trim().to_string())
    };
    assert_eq!(
        best("What is the Wi-Fi password at the Pine Hollow cabin?"),
        (
            "# pine-cabin.md > Pine Hollow cabin > Getting in".into(),
            "Wi-Fi password: pine-77".into()
        )
    );
    assert_eq!(
        best("What is the Wi-Fi password of the home router?"),
        (
            "# home-router.md > Home internet > Getting online".into(),
            "Wi-Fi password: maple-58".into()
        )
    );
    // On the document set, the cabin's chunk ranks first for the cabin question and the router's
    // for the router question.
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    for (q, file) in [
        (
            "What is the Wi-Fi password at the Pine Hollow cabin?",
            "cabin-weekend-plan.md",
        ),
        (
            "What is the Wi-Fi password of the home router?",
            "home-router-setup.txt",
        ),
    ] {
        let chosen = rank(&g.chunks, q, DOCS_WORKERS).unwrap();
        assert_eq!(g.chunks[chosen[0].0].file, PathBuf::from(file), "{q}");
    }
}

#[test]
fn rank_pages_gives_each_page_its_best_two_chunks_before_the_rest() {
    let chunk = |file: &str, line: u32, text: &str| Chunk {
        file: file.into(),
        line,
        heading: Arc::from([]),
        start: 0,
        text: text.into(),
    };
    // The newsroom page names the chip once in a long chunk; the review pages name it over and over, so
    // their third chunks outscore the newsroom's best.
    let lots = "M5 chip M5 chip M5 chip announce";
    let long = "the M5 has a faster GPU and a lot of other parts that are named here at length";
    let mut chunks = vec![
        chunk("a.com/review", 1, lots),
        chunk("a.com/review", 2, lots),
        chunk("a.com/review", 3, lots),
        chunk("a.com/review", 4, lots),
        chunk("b.com/review", 1, lots),
        chunk("b.com/review", 2, lots),
        chunk("b.com/review", 3, lots),
        chunk("newsroom.example/chips", 1, long),
        chunk("newsroom.example/chips", 2, long),
        chunk("newsroom.example/chips", 3, "About us."),
    ];
    // Shipping text that names nothing asked, so the chip's words are rare enough to score.
    chunks.extend((5..25).map(|l| chunk("a.com/review", l, "Shipping and returns.")));
    let q = "What did Apple announce about the M5 chip?";
    let by_score = rank(&chunks, q, 7).unwrap();
    assert!(by_score.iter().all(|(i, _)| *i < 7), "{by_score:?}");

    let chosen = rank_pages(&chunks, q, 7).unwrap();
    assert_eq!(chosen.len(), 7);
    let of = |file: &str| {
        (chosen.iter())
            .filter(|(i, _)| chunks[*i].file == Path::new(file))
            .count()
    };
    assert_eq!(of("newsroom.example/chips"), 2, "{chosen:?}");
    assert!(
        of("a.com/review") >= 2 && of("b.com/review") >= 2,
        "{chosen:?}"
    );
    // Still best first.
    assert!(chosen.windows(2).all(|w| w[0].1 >= w[1].1), "{chosen:?}");
    // Three pages and three slots: each page's best chunk.
    let chosen = rank_pages(&chunks, q, 3).unwrap();
    let files: BTreeSet<_> = chosen.iter().map(|(i, _)| &chunks[*i].file).collect();
    assert_eq!(files.len(), 3, "{chosen:?}");
    // A folder that fits sends every chunk, as `rank` does.
    assert_eq!(
        rank_pages(&chunks, q, 30).unwrap(),
        rank(&chunks, q, 30).unwrap()
    );
}

#[test]
fn a_page_whose_lines_all_score_about_zero_gives_its_longest_list() {
    let chunk = |line: u32, text: &str| Chunk {
        file: "sally.example/cookies".into(),
        line,
        heading: Arc::from([]),
        start: 0,
        text: text.into(),
    };
    // a user's cookie pages: every chunk says "cookies", so the word scores about 0, and the
    // ingredient list says none of the other words asked.
    let story = "These cookies are the best cookies, and my family loves these cookies.";
    let mut chunks: Vec<Chunk> = (1..=8).map(|l| chunk(l, story)).collect();
    chunks.push(chunk(
        9,
        "- 2 cups flour\n- 1 cup butter\n- 2 eggs\n- 1 cup chips",
    ));
    chunks.extend((10..14).map(|l| chunk(l, "I made these cookies twice and they were great.")));
    chunks.push(chunk(14, "- Share\n- Print"));
    let chosen = rank_pages(&chunks, "chocolate chip cookies recipe", 3).unwrap();
    assert!(
        chosen.iter().any(|(i, _)| chunks[*i].line == 9),
        "{chosen:?}"
    );
}

#[test]
fn a_page_whose_intro_says_the_question_words_still_sends_its_list() {
    // A user (28 Oct): an intro and comments say the question's words; the list or table it asks for
    // (ingredients, prices, results) says few of them. Five pages and seven readers, as a web
    // question has; most of a real page's chunks (photos, ads, sidebars) say none of the
    // question's words, so those words score above 0.
    let cases = [
        (
            "give me an online recipe for chocolate chip cookies",
            "This chocolate chip cookie recipe is the best recipe online. Jump to recipe.",
            "I made this recipe online and the chocolate chip cookies were great.",
            "- 1 cup butter\n- 1 cup sugar\n- 2 eggs\n- 2 cups chocolate chips",
        ),
        (
            "how much does a new BMW M3 cost",
            "How much does the new BMW M3 cost? Our M3 price guide shows what a new M3 costs.",
            "The new M3 costs too much, but the BMW is worth it.",
            "Trim | MSRP\nM3 | $76,000\nM3 Competition | $80,000\nM3 CS | $119,000",
        ),
        (
            "who won the last Texas football game",
            "Texas football won again. Who won the last Texas football game? Read on.",
            "Texas football fans: who won the last game was never in doubt.",
            "Date | Opponent | Result\n10/10 | Florida | W 31-20\n10/17 | Oklahoma | W 27-3\n\
             10/24 | Mississippi | W 34-27",
        ),
    ];
    for (question, intro, comment, list) in cases {
        let mut chunks = Vec::new();
        let mut add = |file: &str, texts: &[&str]| {
            for (line, text) in texts.iter().enumerate() {
                chunks.push(Chunk {
                    file: file.into(),
                    line: line as u32 + 1,
                    heading: Arc::from([]),
                    start: 0,
                    text: (*text).into(),
                });
            }
        };
        for page in ["a", "b", "c"] {
            add(
                &format!("{page}.example/page"),
                &[
                    intro,
                    intro,
                    "Subscribe to the newsletter.",
                    "Photo: a gallery.",
                    "Photo: another one.",
                    "Advertisement.",
                    "More from the site.",
                    list,
                    comment,
                    comment,
                    "Share your photo.",
                    "Follow us.",
                ],
            );
        }
        add("other.example/page", &[intro, comment, "About us."]);
        add("news.example/page", &[comment, "Contact us."]);
        let chosen = rank_pages(&chunks, question, 7).unwrap();
        assert!(
            chosen.iter().any(|(i, _)| chunks[*i].text == list),
            "{question}: {chosen:?}"
        );
    }
}

#[test]
fn a_page_whose_lines_all_score_about_zero_keeps_its_lead_beside_a_share_list() {
    // Three game recaps: every chunk says "Texas" and "game", so both score about 0, and no chunk
    // says "won". A share list is no answer (second review, Important 5).
    let mut chunks = Vec::new();
    for page in 1..=3 {
        let file = format!("recap{page}.example/texas");
        let share = match page {
            3 => {
                "- [Share on Facebook](https://f.example/s)\n- [Share on X](https://x.example/s)\n- [Print](https://p.example/s)"
            }
            _ => "- Share on Facebook\n- Share on X",
        };
        for (line, text) in [
            "Texas beat Mississippi 34-27 in the game on Saturday.",
            "The Texas game drew a big crowd.",
            "Texas fans stayed long after the game.",
            share,
        ]
        .into_iter()
        .enumerate()
        {
            chunks.push(Chunk {
                file: file.clone().into(),
                line: line as u32 + 1,
                heading: Arc::from([]),
                start: 0,
                text: text.into(),
            });
        }
    }
    let chosen = rank_pages(&chunks, "who won the Texas game", 7).unwrap();
    for page in 1..=3 {
        let lead = |i: &usize| {
            chunks[*i].line == 1
                && chunks[*i].file.to_str() == Some(&format!("recap{page}.example/texas"))
        };
        assert!(
            chosen.iter().any(|(i, _)| lead(i)),
            "page {page}: {chosen:?}"
        );
    }
}

#[test]
fn docs_window_0_hands_the_chunk_alone_and_128_adds_up_to_128_tokens_a_side() {
    let tmp = tempfile::tempdir().unwrap();
    // Three chunks of 1,000 bytes and more, and lines of 20 bytes.
    let text: String = (1..=150)
        .map(|n| {
            let blank = if n % 10 == 0 && n < 150 { "\n" } else { "" };
            format!("line {n:03} of a file{blank}\n")
        })
        .collect();
    let folder = folder(tmp.path(), &[("a.md", "Other file.\n"), ("b.md", &text)]);
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    let chunks: Vec<_> = g
        .chunks
        .iter()
        .filter(|c| c.file == Path::new("b.md"))
        .collect();
    assert!(chunks.len() >= 3, "{chunks:?}");
    let file: Vec<&str> = text.lines().collect();
    let shown = |e: &Excerpt| {
        // The lines are the file's, numbered as in the file.
        for (n, line) in (e.first..).zip(&e.lines) {
            assert_eq!(file[n as usize - 1], line, "{n}");
        }
        e.lines.iter().map(|l| l.len() + 1).sum::<usize>()
    };
    for (i, chunk) in chunks.iter().enumerate() {
        let alone = excerpt(&g, chunk, 0);
        assert_eq!((alone.first, alone.header.as_str()), (chunk.line, "# b.md"));
        assert_eq!(alone.lines.concat(), chunk.text.replace('\n', ""));
        let wide = excerpt(&g, chunk, 128);
        shown(&wide);
        let before = (chunk.line - wide.first) as usize;
        let after = wide.lines.len() - alone.lines.len() - before;
        let bytes = |lines: &[String]| lines.iter().map(|l| l.len() + 1).sum::<usize>();
        let (b, a) = (
            bytes(&wide.lines[..before]),
            bytes(&wide.lines[before + alone.lines.len()..]),
        );
        // Up to 512 bytes a side, and the next line would not fit, unless the file ends there.
        assert!(b <= 512 && a <= 512, "{b} {a}");
        assert!(
            wide.first == 1 || b + file[wide.first as usize - 2].len() + 1 > 512,
            "{b}"
        );
        let last = wide.first as usize + wide.lines.len() - 1;
        assert!(last == file.len() || a + file[last].len() + 1 > 512, "{a}");
        // The window stops at the file's edge.
        if i == 0 {
            assert_eq!((wide.first, before), (1, 0));
        }
        if i == chunks.len() - 1 {
            assert_eq!((last, after), (file.len(), 0));
        } else {
            assert!(after > 0 && before + after > 20, "{before} {after}");
        }
        assert!(!wide.render().contains("Other file"));
    }
}

fn fact(line: u64, claim: &str) -> Fact {
    Fact {
        line,
        claim: claim.into(),
    }
}

#[test]
fn the_worker_system_prompt_is_under_200_tokens_and_asks_for_lines() {
    // Four bytes a token, as everywhere in the agent.
    assert!(
        WORKER_SYSTEM.len() / 4 < 200,
        "{} bytes",
        WORKER_SYSTEM.len()
    );
    for word in [
        "\"line\"",
        "\"claim\"",
        "reply with only []",
        "number of the line",
        "naming what it is about",
        // a user's cookies (28 Oct): a reader given the whole ingredient list said [], since no
        // line of it names the cookies.
        "counts line by line",
    ] {
        assert!(WORKER_SYSTEM.contains(word), "{word}");
    }
    // The worker writes no quote: Rust copies it from the line.
    assert!(!WORKER_SYSTEM.contains("\"quote\""), "{WORKER_SYSTEM}");
    // A fact about any one thing the question names helps, so a question about two things does
    // not leave every worker at []; a near miss (the home router's password for a question about
    // the cabin's) still does not.
    assert!(
        WORKER_SYSTEM.contains("about one of the things the question names")
            && !WORKER_SYSTEM.contains(SAME_THING),
        "{WORKER_SYSTEM}"
    );
    assert!(
        WORKER_SYSTEM.contains(
            "A fact about something similar, such as another home, account or year, does not help"
        ),
        "{WORKER_SYSTEM}"
    );
}

/// The orchestrator's rule, which `scripted` checks: a fact must be about what the question names.
const SAME_THING: &str = "exactly what the question names";
/// The orchestrator's line for a question the facts answer in part, which `scripted` checks too.
const PARTIAL: &str = "If the question asks about two or more things and the facts answer only \
                       some of them, answer those and say which the documents do not answer.";

#[test]
fn facts_parse_from_a_list_with_text_around_it() {
    let reply = "Here they are:\n```json\n[{\"line\": 14, \"claim\": \"Rent is $1,850\"}]\n```\n";
    assert_eq!(
        parse_facts(reply, Finish::Eos).unwrap(),
        [fact(14, "Rent is $1,850")]
    );
    // Brackets in the words around the list, and fields the worker adds, which nothing reads.
    let reply = "Facts [see below]:\n[{\"line\": 3, \"claim\": \"Rent\", \"quote\": \"x\", \
                 \"confidence\": \"high\"}]\n(see [1])";
    assert_eq!(parse_facts(reply, Finish::Eos).unwrap(), [fact(3, "Rent")]);
}

#[test]
fn only_the_reply_brackets_alone_is_nothing_found() {
    // `[]`, blanks around it aside, is the one way to say the excerpt does not help.
    for nothing in ["[]", " [] ", "\n[]\n"] {
        assert_eq!(
            parse_facts(nothing, Finish::Eos).unwrap(),
            [],
            "{nothing:?}"
        );
    }
    // A whole list cut right after it is kept.
    assert_eq!(parse_facts("[]", Finish::Length).unwrap(), []);
    // Any other reply without a fact is an error, so the worker gets its retry: a cut list, an
    // empty list with a blank inside, in a code fence, with words after it, beside a broken list,
    // or as a checklist's box.
    for broken in [
        "[",
        "[ ]",
        "[\n]",
        "```json\n[]\n```",
        "[]\n\nNothing in the excerpt is about the cabin.",
        "[{\"line\": 3, \"claim\": \"x\",}]\n[]",
        "[]\n{\"line\": 3, \"claim\": \"Rent\"}",
        "- [ ] Wage statement from Lattice Point Software",
        "1. [ ] Wage statement from Lattice Point Software",
        "The rent is $1,850.",
        "",
    ] {
        let e = parse_facts(broken, Finish::Eos).unwrap_err();
        assert!(matches!(e, FactsError::Json(_)), "{broken:?}: {e}");
        let e = e.to_string();
        assert!(e.starts_with("[agent] ") && e.contains("JSON list"), "{e}");
    }
}

#[test]
fn a_reply_that_is_no_list_of_facts_gets_a_message_for_the_model() {
    // The parser's own words say what is wrong with a list of the wrong shape.
    for (reply, words) in [
        (r#"[{"claim": "Rent is $1,850"}]"#, "missing field `line`"),
        (r#"[{"line": "14", "claim": "Rent"}]"#, "expected u64"),
        (r#"[{"line": -1, "claim": "Rent"}]"#, "expected u64"),
        (
            r#"[{"line": 18446744073709551616, "claim": "Rent"}]"#,
            "expected u64",
        ),
        // So do they after a bracket in the words before the list: the error is the list's.
        (
            "Facts [see below]:\n[{\"claim\": \"Rent\"}]",
            "missing field `line`",
        ),
        ("[ ]", "the reply [] alone"),
        ("```json\n[]\n```", "the reply [] alone, with no fence"),
        ("[]\nNothing here.", "the reply [] alone"),
    ] {
        let e = parse_facts(reply, Finish::Eos).unwrap_err().to_string();
        assert!(e.contains(words), "{reply}: {e}");
    }
    // The message names the shape to reply with.
    let e = parse_facts("The rent is $1,850.", Finish::Eos)
        .unwrap_err()
        .to_string();
    assert!(e.contains(r#"[{"line": ..., "claim": ...}]"#), "{e}");
}

#[test]
fn a_bracket_inside_a_claim_is_no_list_of_its_own() {
    // The tax checklist has "- [ ]" lines. A list broken after such a claim is an error with the
    // list's own message, and the facts after such a line in the words before the list are kept.
    let broken = r#"[{"line": 4, "claim": "- [ ] Wage statement from Lattice Point Software"},]"#;
    let e = parse_facts(broken, Finish::Eos).unwrap_err().to_string();
    assert!(e.contains("trailing comma"), "{e}");
    let wage = fact(4, "A wage statement is needed");
    let prose = format!(
        "The checklist has - [ ] Wage statement.\n[{}]",
        serde_json::to_string(&wage).unwrap()
    );
    assert_eq!(parse_facts(&prose, Finish::Eos).unwrap(), [wage]);
}

#[test]
fn a_list_broken_lines_below_a_bracket_reports_where_it_broke() {
    // The "[x]" inside the first claim is text the failed parse read, so it starts no list, and
    // the message is the list's own, on line 3.
    let broken = "[\n  {\"line\": 1, \"claim\": \"- [x] Wage statement\"},\n  {\"line\": 2 \"claim\": \"c\"}\n]";
    let e = parse_facts(broken, Finish::Eos).unwrap_err().to_string();
    assert!(e.contains("line 3"), "{e}");
}

#[test]
fn a_cut_reply_asks_for_the_list_without_reasoning() {
    // A thinking model cut at max_tokens shows no text at all; a "[]" it goes on to take back is
    // cut too.
    for cut in [
        "",
        "[]\n\nOn reflection the excerpt does help:\n[",
        "[{\"line\": 3",
    ] {
        let e = parse_facts(cut, Finish::Length).unwrap_err();
        assert!(matches!(e, FactsError::Cut), "{cut:?}");
        let e = e.to_string();
        assert!(
            e.starts_with("[agent] ") && e.contains("token limit"),
            "{e}"
        );
    }
}

/// Lines 10 to 14 of a lease, as a worker reads them.
fn lease_excerpt() -> Excerpt {
    Excerpt {
        file: "lease.md".into(),
        header: "# lease.md > Rent".into(),
        first: 10,
        lines: [
            "# Rent",
            "",
            "Tenant: Sam Lee. The monthly",
            "  rent is   $1,450\tper  ",
            "month.",
        ]
        .map(String::from)
        .to_vec(),
    }
}

#[test]
fn the_quote_copied_from_a_line_is_its_text_with_whitespace_runs_as_one_space() {
    let e = lease_excerpt();
    assert_eq!(
        e.render(),
        "# lease.md > Rent\n10: # Rent\n11: \n12: Tenant: Sam Lee. The monthly\n13:   rent is   \
         $1,450\tper  \n14: month.\n"
    );
    let want = Cited {
        claim: "Rent is $1,450".into(),
        quote: "rent is $1,450 per".into(),
        file: "lease.md".into(),
        line: 13,
    };
    assert_eq!(e.cite(&fact(13, "Rent is $1,450")), Some(want));
}

#[test]
fn a_worker_line_number_out_of_range_is_dropped() {
    let e = lease_excerpt();
    // Before the excerpt, after it, at the ends of u64, and a blank line, which shows nothing.
    for line in [0, 9, 15, u64::from(u32::MAX) + 10, u64::MAX, 11] {
        assert_eq!(e.cite(&fact(line, "Rent")), None, "{line}");
    }
    for line in [10, 14] {
        assert_eq!(e.cite(&fact(line, "Rent")).unwrap().line as u64, line);
    }
}

#[test]
fn a_claim_whose_amount_or_date_its_line_lacks_cites_the_nearest_line_that_holds_it() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    let status = excerpt_of(&g, "project-status-2026-08.md", 11);
    let at = |e: &Excerpt, line: u64, claim: &str| e.cite(&fact(line, claim)).unwrap();
    // The m1 worker's claim in the audit at 97dc246, one line below its date.
    let date = at(
        &status,
        12,
        "Countertop installation is scheduled for 14 October 2026, two weeks after the cabinets.",
    );
    assert_eq!((date.line, &date.claim[..12]), (11, "Countertop i"));
    assert!(
        date.quote
            .starts_with("Countertop installation is scheduled for 14 October 2026")
    );
    // A claim with no amount or date, or with one no line holds, keeps its line.
    assert_eq!(at(&status, 12, "Plumbing follows the countertop").line, 12);
    assert_eq!(
        at(&status, 12, "The countertop is in on 2 March 2027").line,
        12
    );
    // The budget is on lines 3 and 17: the nearer one, or the earlier of two as near. Written
    // without the comma, it is the same amount.
    assert_eq!(at(&status, 15, "The budget is $25000").line, 17);
    assert_eq!(at(&status, 10, "The budget is $25,000").line, 3);
    assert_eq!(at(&status, 17, "The budget is $25,000").line, 17);
    // The amount on the line after the one the worker named, as a wrapped sentence goes on.
    let bank = excerpt_of(&g, "bank-letter-savings-rate.txt", 26);
    let insured = "the balance is protected by the national deposit insurance scheme up to \
                   $250,000 per person";
    assert_eq!(at(&bank, 26, insured).line, 27);
    // A line the excerpt does not show still cites nothing.
    assert_eq!(status.cite(&fact(99, "The budget is $25,000")), None);
}

/// The answer turn's system prompt before `NOT_FOUND`, byte for byte as the Mac bench measured it
/// on m2 at 97dc246.
const ANSWER_SYSTEM_97DC246: &str = "Answer the user's question from the numbered facts below \
and nothing else. Each fact says the file and section it comes from. Use only facts about exactly \
what the question names: the same place, person, thing or year. A fact about something similar, \
such as another home, account or year, does not answer the question. After each fact you use, \
give its number in brackets, such as [2]. Be brief. If the question asks about two or more things \
and the facts answer only some of them, answer those and say which the documents do not answer. \
If they answer no part of it, reply with only this sentence: ";

#[test]
fn m2s_answer_request_is_97dc246s_and_llama_3_gets_each_line_once_without_claims() {
    let fact = |claim: &str, quote: &str, file: &str, line| Cited {
        claim: claim.into(),
        quote: quote.into(),
        file: PathBuf::from(file),
        line,
    };
    let rent = fact(
        "Rent is $1,450 a month",
        "rent is $1,450 per",
        "lease.md",
        8,
    );
    let again = fact("the rent", "rent is $1,450 per", "lease.md", 8);
    let mail = fact(
        "Rent is now $1,500",
        "Rent: $1,500 from May.",
        "mail.txt",
        2,
    );
    let facts = Facts {
        cited: vec![rent.clone(), again.clone(), mail.clone()],
        conflicts: vec![(0, 2), (1, 2)],
        ..Facts::default()
    };
    let texts = BTreeMap::from([(PathBuf::from("lease.md"), LEASE.to_string())]);
    let q = "How much is the rent?";
    // Every family but Llama 3: the 97dc246 request, every fact with its claim and line.
    for family in [None, Some(Family::Qwen3), Some(Family::Mistral)] {
        assert_eq!(
            answer_request(family, q, &facts, &texts),
            AnswerRequest {
                system: format!("{ANSWER_SYSTEM_97DC246}{NOT_FOUND}"),
                prompt: "Question: How much is the rent?\n\nFacts:\n\
                         [1] Rent is $1,450 a month (from lease.md > Rent: \"rent is $1,450 per\")\n\
                         [2] the rent (from lease.md > Rent: \"rent is $1,450 per\")\n\
                         [3] Rent is now $1,500 (from mail.txt: \"Rent: $1,500 from May.\")\n\
                         Facts [1] and [3] give different values for the same thing: give both, \
                         with their numbers.\n\
                         Facts [2] and [3] give different values for the same thing: give both, \
                         with their numbers.\n"
                    .into(),
                facts: facts.cited.clone(),
            },
            "{family:?}"
        );
    }
    // Llama 3: each line once and alone, the pairs on the lines it shows.
    let llama = answer_request(Some(Family::Llama3), q, &facts, &texts);
    assert_eq!(
        llama.prompt,
        "Question: How much is the rent?\n\nFacts:\n\
         [1] \"rent is $1,450 per\" (from lease.md > Rent)\n\
         [2] \"Rent: $1,500 from May.\" (from mail.txt)\n\
         Facts [1] and [2] give different values for the same thing: give both, with their \
         numbers.\n"
    );
    assert_eq!(llama.facts, [rent, mail]);
    assert!(llama.system.ends_with(NOT_FOUND), "{}", llama.system);
    assert!(
        llama
            .system
            .contains("Each fact is a line of the user's documents")
    );
    assert!(!llama.system.contains(PARTIAL), "{}", llama.system);
    assert!(llama.system.contains(SAME_THING), "{}", llama.system);
}

/// `n` facts, the `i`th from `f{i}.md` line `i`.
fn facts(n: u32) -> Vec<Cited> {
    (1..=n)
        .map(|i| Cited {
            claim: format!("claim {i}"),
            quote: format!("quote {i}"),
            file: format!("f{i}.md").into(),
            line: i,
        })
        .collect()
}

#[test]
fn an_orchestrator_citation_7_with_six_facts_is_dropped_and_counted() {
    let six = facts(6);
    assert_eq!(
        cite("Rent [2]. Fee [7]. Both [1, 6]. Some [3, 7].", &six),
        (
            "Rent [f2.md:2]. Fee. Both [f1.md:1, f6.md:6]. Some [f3.md:3].".into(),
            2
        )
    );
    // Brackets that hold no fact number stay as they are, and so does a citation of [0] go.
    assert_eq!(
        cite(
            "See [the lease] and [a] [0] [99999999999999999999999].",
            &six
        ),
        ("See [the lease] and [a].".into(), 2)
    );
    assert_eq!(cite("No [ end", &six), ("No [ end".into(), 0));
}

#[test]
fn the_conflict_rule_flags_1_against_1850_and_not_two_dates_on_different_things() {
    let fact = |file: &str, claim: &str| Cited {
        claim: claim.into(),
        quote: claim.into(),
        file: file.into(),
        line: 1,
    };
    let rent = fact("lease.md", "The monthly rent for Alder Street is $1,850.");
    let cut = fact("email.txt", "The monthly rent for Alder Street is now $1.");
    let same = fact(
        "bank.txt",
        "The monthly rent for Alder Street, $1,850.00, went out.",
    );
    let mine = fact("lease.md", "The monthly rent for Alder Street is $1,000.");
    assert_eq!(conflicts(&[rent.clone(), cut.clone()]), [(0, 1)]);
    // The same amount, or two amounts in the same file, is no conflict.
    assert_eq!(conflicts(&[rent.clone(), same]), []);
    assert_eq!(conflicts(&[rent.clone(), mine]), []);
    // Cents written two ways are one amount.
    let cents = fact(
        "bank.txt",
        "The monthly rent for Alder Street is $1,850.50.",
    );
    let again = fact("card.txt", "The monthly rent for Alder Street is $1850.5.");
    assert_eq!(conflicts(&[cents, again]), []);
    // The same number in another currency is another amount.
    let pounds = fact("uk.txt", "The monthly rent for Alder Street is £1,850.");
    assert_eq!(conflicts(&[rent.clone(), pounds]), [(0, 1)]);
    // Two dates on different things share fewer than three words.
    let flight = fact(
        "trip.md",
        "The return flight from Lisbon leaves on Thursday 29 October.",
    );
    let dentist = fact(
        "dentist.txt",
        "The next dentist appointment is on Tuesday 3 November 2026.",
    );
    assert_eq!(conflicts(&[flight, dentist]), []);
    // Two dates of the same thing differ; one without its year is the same day.
    let due = fact(
        "vet.md",
        "Miso's rabies booster is due on 14 February 2027.",
    );
    let later = fact("notes.md", "Miso's rabies booster is due on March 3rd.");
    let again = fact("card.txt", "Miso's rabies booster is due on February 14.");
    assert_eq!(conflicts(&[due, later, again]), [(0, 1), (1, 2)]);
}

/// The bench's line `line` of `file` as Rust cites it, with `claim` as the worker's words.
fn bench_fact(file: &str, line: u32, claim: &str) -> Cited {
    let text = fs::read_to_string(Path::new(DOCS).join(file)).unwrap();
    let quote = text.lines().nth(line as usize - 1).unwrap();
    Cited {
        claim: claim.into(),
        quote: quote.split_whitespace().collect::<Vec<_>>().join(" "),
        file: file.into(),
        line,
    }
}

#[test]
fn a_rent_and_a_premium_or_an_appointment_and_a_meeting_are_no_conflict_but_two_dates_are() {
    // m2's q01 at 362df68: "$1,850 [lease-alder-street.md:14] and $18.50
    // [insurance-renters-renewal.txt:21]", the worker that read the premium calling it the rent.
    let rent = bench_fact(
        "lease-alder-street.md",
        14,
        "The monthly rent for the apartment at 42 Alder Street is $1,850",
    );
    let premium = bench_fact(
        "insurance-renters-renewal.txt",
        21,
        "The monthly rent for the apartment at 42 Alder Street is $18.50",
    );
    assert_eq!(premium.quote, "Your premium: $18.50 per month.");
    assert_eq!(conflicts(&[rent.clone(), premium]), []);
    // q11: the dentist appointment against the residents' meeting of 12 August.
    let dentist = bench_fact(
        "dentist-appointment.txt",
        8,
        "The next dentist appointment is on Tuesday 3 November 2026",
    );
    let meeting = bench_fact(
        "residents-meeting-notes-2026.md",
        360,
        "The next dentist appointment is on 12 August 2026",
    );
    assert_eq!(conflicts(&[dentist, meeting]), []);
    // Two lines that name the rent still disagree.
    let old = Cited {
        claim: "The monthly rent for the apartment at 42 Alder Street is $1,450".into(),
        quote: "Monthly rent: $1,450.".into(),
        file: "lease-2024.md".into(),
        ..rent.clone()
    };
    assert_eq!(conflicts(&[rent, old]), [(0, 1)]);
    // q03 and q17: August's status report has the countertop on 14 October, October's on 3
    // November, and both lines name the countertop installation.
    let august = bench_fact(
        "project-status-2026-08.md",
        11,
        "The countertop installation is scheduled for 14 October 2026",
    );
    let october = bench_fact(
        "project-status-2026-10.md",
        11,
        "The countertop installation is now scheduled for 3 November 2026",
    );
    assert_eq!(conflicts(&[august, october]), [(0, 1)]);
}

#[test]
fn a_sentence_saying_the_facts_disagree_is_dropped() {
    let q01 = "The monthly rent for the apartment at 42 Alder Street is $1,850 \
               [lease-alder-street.md:14] and $18.50 [insurance-renters-renewal.txt:21]. The \
               documents provide conflicting values for the same thing.";
    assert_eq!(
        without_conflict_talk(q01),
        (
            "The monthly rent for the apartment at 42 Alder Street is $1,850 \
             [lease-alder-street.md:14] and $18.50 [insurance-renters-renewal.txt:21]."
                .into(),
            1
        )
    );
    // A full stop inside a number or a file name ends no sentence.
    let q18 = "Miso weighs 4.6 kg [vet-records-miso.md:17]. The lease and the vet record \
               disagree! One cat is allowed [lease-alder-street.md:34].";
    assert_eq!(
        without_conflict_talk(q18),
        (
            "Miso weighs 4.6 kg [vet-records-miso.md:17]. One cat is allowed \
             [lease-alder-street.md:34]."
                .into(),
            1
        )
    );
    let both = "Miso weighs 4.6 kg [vet-records-miso.md:17]. The values conflict at 4.6 kg \
                [vet-records-miso.md:17].";
    assert_eq!(
        without_conflict_talk(both),
        ("Miso weighs 4.6 kg [vet-records-miso.md:17].".into(), 1)
    );
    assert_eq!(without_conflict_talk(CITED), (CITED.into(), 0));
}

#[test]
fn the_names_of_a_question_are_its_capitalized_words_past_the_first() {
    let names = |q: &str| question_names(q);
    assert_eq!(
        names("What is the Wi-Fi password at the Pine Hollow cabin?"),
        ["Wi", "Fi", "Pine", "Hollow"]
    );
    // Not "I", a month or a weekday, nor the first word of a sentence.
    assert_eq!(
        names("Can I see Miso's vet on Friday in January? Is Miso well?"),
        ["Miso"]
    );
    assert_eq!(names("When is my next dentist appointment?"), [""; 0]);
    // A year is a name; another number is not.
    assert_eq!(
        names("What is the monthly rent for 17 Maple Avenue in 2027?"),
        ["Maple", "Avenue", "2027"]
    );
    assert_eq!(names("Is 1850 or 2100 the fee?"), [""; 0]);
}

/// The excerpt the defaults (window 0) send with line `line` of `file`, as its worker reads it.
fn excerpt_at(g: &Gathered, file: &str, line: u32) -> String {
    excerpt_of(g, file, line).render()
}

/// The excerpt at the defaults that shows `line` of `file`.
fn excerpt_of(g: &Gathered, file: &str, line: u32) -> Excerpt {
    g.chunks
        .iter()
        .map(|c| excerpt(g, c, 0))
        .find(|e| {
            e.file == Path::new(file) && (e.first..e.first + e.lines.len() as u32).contains(&line)
        })
        .unwrap()
}

#[test]
fn an_answer_whose_excerpts_never_name_the_cabin_is_withheld() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    // Every q21 answer the Mac gave at 362df68 (m2: "The Wi-Fi password at the Pine Hollow cabin
    // is maple-otter-58 [home-router-setup.txt:7]."; m1: "maple-otter-58
    // [home-router-setup.txt:7]") cites only the router, whose excerpt names no Pine Hollow.
    let files = files_of(&g);
    let q21 = "What is the Wi-Fi password at the Pine Hollow cabin?";
    let router = [excerpt_at(&g, "home-router-setup.txt", 7)];
    assert_eq!(
        missing_names(q21, &files, &router),
        ["Pine", "Hollow", "cabin"]
    );
    // The cabin's own excerpt does.
    let cabin = [excerpt_at(&g, "cabin-weekend-plan.md", 11)];
    assert_eq!(missing_names(q21, &files, &cabin), [""; 0]);
    // q23: the 2024 lease's rent is no answer for 2027.
    let q23 = "What is the monthly rent for 17 Maple Avenue in 2027?";
    let lease = [excerpt_at(&g, "lease-maple-avenue-2024.md", 12)];
    assert_eq!(missing_names(q23, &files, &lease), ["2027"]);
    // An answer to half of q18 from the vet record alone says nothing of Alder Street.
    let q18 = "Can my cat Miso live with me at Alder Street, and how much does Miso weigh?";
    let vet = [excerpt_at(&g, "vet-records-miso.md", 17)];
    assert_eq!(missing_names(q18, &files, &vet), ["Alder", "Street"]);
    // A name is held as the start of a word, case aside.
    assert_eq!(
        missing_names("Is it on Alder St?", &[], &["42 ALDER STREET".into()]),
        [""; 0]
    );
}

/// A name written in capitals and digits alone is an acronym or a model, held only as a whole
/// word or its plural: "UTSA" is another school than "UT", an "M30" another part than an "M3".
#[test]
fn a_capitals_name_is_held_only_as_a_whole_word() {
    let held = |q: &str, text: &str| missing_names(q, &[], &[text.into()]).is_empty();
    assert!(!held("Who won the last UT game?", "Texas beat UTSA 30-6."));
    assert!(held("Who won the last UT game?", "UT beat Kentucky 16-13."));
    assert!(held(
        "When does the NFL draft start?",
        "The NFL's draft opens April 23."
    ));
    assert!(held(
        "How many cores does the CPU have?",
        "Both CPUs have 8 cores."
    ));
    assert!(!held("What does the M3 cost?", "The M30 costs $40."));
    assert!(!held(
        "What did IBM report?",
        "IBMers reported strong sales."
    ));
    // Other names still match the start of a word, years whole.
    assert!(held("Is it on Alder St?", "42 ALDER STREET"));
    assert!(!held(
        "What is the rent in 2027?",
        "Rent is $900 (ref 20271)."
    ));
}

/// The paths `gather` read, as `missing_names` takes them.
fn files_of(g: &Gathered) -> Vec<&Path> {
    g.texts.keys().map(PathBuf::as_path).collect()
}

#[test]
fn a_question_typed_in_lower_case_still_names_the_cabin_through_its_file() {
    // a user's words on the Mac (#221's CA-WIFI): m1's ask_docs answered "what's the wifi password at
    // the cabin?" with the router's password, cited at home-router-setup.txt:7. The question has
    // no capitalized name, but "cabin" names cabin-weekend-plan.md, and the router's excerpt
    // never says it, so that answer is withheld.
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    let files = files_of(&g);
    let router = [excerpt_at(&g, "home-router-setup.txt", 7)];
    for q in [
        "what's the wifi password at the cabin?",
        "whats the wifi password at pine hollow cabin",
    ] {
        assert_eq!(question_names(q), [""; 0], "{q}");
        assert_eq!(file_words(q, &files), ["cabin"], "{q}");
        assert_eq!(missing_names(q, &files, &router), ["cabin"], "{q}");
    }
    // A word file names use for their kind, or that names no file, asks for nothing.
    assert_eq!(
        file_words("what do my documents and notes say about the wifi?", &files),
        [""; 0]
    );
    assert_eq!(
        file_words("what's the wifi password at home?", &files),
        [""; 0]
    );
    // A plural is the file's word.
    assert_eq!(
        file_words("when are my dentist appointments?", &files),
        ["dentist", "appointments"]
    );
}

#[test]
fn every_planted_fact_excerpt_names_what_its_question_names() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS)), DOCS).unwrap();
    let files = files_of(&g);
    let (mut named, mut by_file) = (0, 0);
    for q in questions() {
        if q.facts.is_empty() {
            continue;
        }
        let held: Vec<String> = (q.facts.iter())
            .map(|f| excerpt_at(&g, f.file.to_str().unwrap(), f.line))
            .collect();
        named += question_names(&q.question).len();
        by_file += file_words(&q.question, &files).len();
        // A complete answer is never withheld.
        assert_eq!(
            missing_names(&q.question, &files, &held),
            [""; 0],
            "{}",
            q.id
        );
    }
    assert!(named >= 10, "{named}");
    assert!(by_file >= 20, "{by_file}");
    // m2's q16 at 362df68 cites the trip's last line, not the return flight's: it stands.
    let q16 = "Is my dentist appointment before or after my return flight from Lisbon?";
    let held = [
        excerpt_at(&g, "dentist-appointment.txt", 8),
        excerpt_at(&g, "trip-plan-lisbon.md", 22),
    ];
    assert_eq!(missing_names(q16, &files, &held), [""; 0]);
}

// The scripted model's replies. A worker's first reply is never a list, and its retry is `FACTS`
// when its prompt asks about the rent, except on the pancake chunk, which never gets a list. A
// question with "Raise" fails each retry; with "Forget", a retry does not fit the context with its
// excerpt, and the worker shown only the feedback answers `[]`. With "Nothing", a worker answers
// `[]` at once; with "Bracket", "[" and then `[]`; with "Which rent", `RENT_HIGH` on an excerpt
// that names Alder Street and `RENT_LOW` on any other. The orchestrator answers `ANSWER` when its
// system prompt has the same-thing rule and its prompt holds the fact that passed the line check,
// numbered, and no fact that failed it, or, told its facts are lines, when its prompt holds the
// line of that fact (`LINES_LISTED`); else `LEAK`. Asked to answer at length, it goes on past
// the reply limit. Only a worker with "word word" text, in its excerpt or its question, gets that
// text whole, one token a byte; every other prompt is a single cue token. The question is in a
// worker's system prompt, the excerpt in its user message.
const BAD: &str = "The rent is $1,450.";
const FACTS: &str =
    r#"[{"line": 8, "claim": "Rent is $1,450 a month"}, {"line": 99, "claim": "MISSING"}]"#;
/// Cites the one fact and a seventh that is not there, and says the facts disagree.
const ANSWER: &str = "The rent is $1,450 a month [1] [7]. The facts conflict.";
/// `ANSWER` as Rust writes it when it found no conflict: the fact's source for its number, the
/// unknown number gone, and so is the sentence that says the facts disagree.
const CITED: &str = "The rent is $1,450 a month [lease.md:8].";
/// The notes under `CITED`.
const DROPPED: &str = "_1 citation named no fact and was dropped._\n\n_1 sentence that said the \
                       facts disagree was dropped: the facts' values are for different things._";
const LEAK: &str = "LEAK";
const RENT_HIGH: &str =
    r#"[{"line": 1, "claim": "The monthly rent for the flat at Alder Street is $1,850"}]"#;
const RENT_LOW: &str =
    r#"[{"line": 1, "claim": "The monthly rent for the flat at Alder Street is $1"}]"#;

/// The orchestrator's line for the rent fact, as `scripted` looks for it.
const LISTED: &str = r#"[1] Rent is $1,450 a month (from lease.md > Rent: "rent is $1,450 per")"#;

/// The orchestrator's line for the rent fact on Llama 3, as `scripted` looks for it.
const LINES_LISTED: &str = r#"[1] "rent is $1,450 per" (from lease.md > Rent)"#;

/// The rent is on line 8.
const LEASE: &str = "# Lease\n\nLandlord: A. Park.\n\n# Rent\n\nTenant: Sam Lee. The monthly\nrent is $1,450 per\nmonth, due on the 1st.\n";

/// The fake model with a close second: the token after its pick scores just below it, so a
/// sampled reply strays from the script and only greedy decoding keeps to it.
struct CloseSecond(FakeModel);

impl Model for CloseSecond {
    fn config(&self) -> &ModelConfig {
        self.0.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.0.forward(s, b, kv, logits)?;
        let mut l = vec![0f32; logits.shape().iter().product::<i64>() as usize];
        s.download(logits.buffer(), logits.offset(), &mut l)?;
        // A prefill chunk that ends no prompt has no rows of logits.
        let vocab = self.0.config().vocab as usize;
        for row in l.chunks_mut(vocab) {
            let pick = row.iter().position(|&x| x > 0.0).unwrap_or_default();
            row[(pick + 1) % vocab] = 99.0;
        }
        s.upload(logits.buffer(), logits.offset(), &l)
    }
}

/// The fake model with a cue token before each reply: the template picks a cue, and the model,
/// which continues with the next token (the one after it a close second), gives the reply (one
/// user-defined token) and then an end of turn.
fn scripted() -> Larch {
    let ends = ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>"];
    let mut tokens: Vec<_> = (0..256)
        .map(|b| Value::Str(format!("<0x{b:02X}>")))
        .collect();
    let mut types = vec![Value::I32(6); 256];
    for (i, (reply, end)) in [BAD, FACTS, ANSWER, LEAK].into_iter().zip(ends).enumerate() {
        tokens.extend([format!("<cue{i}>"), reply.into(), end.into()].map(Value::Str));
        types.extend([3, 4, 3].map(Value::I32));
    }
    // More words than a reply may have, with no end of turn among them.
    tokens.push(Value::Str("<cue4>".into()));
    types.push(Value::I32(3));
    tokens.extend((0..1100).map(|i| Value::Str(format!("w{i} "))));
    types.extend((0..1100).map(|_| Value::I32(4)));
    // An answer of spaces only, and an empty list.
    for (cue, reply, end) in [
        ("<cue5>", "  ", "<|endoftext|>"),
        ("<cue6>", "[]", "<|fim_pad|>"),
        ("<cue7>", "[", "<|repo_name|>"),
        ("<cue8>", RENT_HIGH, "<|file_sep|>"),
        ("<cue9>", RENT_LOW, "</s>"),
    ] {
        tokens.extend([cue, reply, end].map(|t| Value::Str(t.into())));
        types.extend([3, 4, 3].map(Value::I32));
    }
    let vocab = tokens.len() as u32;
    let template = format!(
        "{{%- if messages[0].content.startswith({worker}) -%}}\
         {{%- if '[agent]' in messages[1].content %}}<cue6>\
         {{%- elif 'word word' in messages[1].content %}}{{{{ messages[1].content }}}}\
         {{%- elif 'word word' in messages[0].content %}}{{{{ messages[0].content }}}}\
         {{%- elif 'Which rent' in messages[0].content %}}\
         {{% if 'Alder' in messages[1].content %}}<cue8>{{% else %}}<cue9>{{% endif %}}\
         {{%- elif 'Bracket' in messages[0].content %}}\
         {{% if messages|length < 4 %}}<cue7>{{% else %}}<cue6>{{% endif %}}\
         {{%- elif 'Nothing' in messages[0].content %}}<cue6>\
         {{%- elif 'Pancakes' in messages[1].content or messages|length < 4 %}}<cue0>\
         {{%- elif 'Raise' in messages[0].content %}}{{{{ raise_exception('the retry fails') }}}}\
         {{%- elif 'Forget' in messages[0].content %}}{{% for i in range(5000) %}}x{{% endfor %}}\
         {{%- elif 'Question: How much is the rent?' in messages[0].content %}}<cue1>\
         {{%- else %}}<cue0>{{% endif -%}}\
         {{%- elif 'at length' in messages[1].content %}}<cue4>\
         {{%- elif 'Say nothing' in messages[1].content %}}<cue5>\
         {{%- elif {disagree} in messages[1].content %}}<cue2>\
         {{%- elif 'Each fact is a line of' in messages[0].content and {lines} in \
         messages[1].content %}}<cue2>\
         {{%- elif {same} in messages[0].content and {conflict} in messages[0].content and \
         {listed} in messages[1].content and not ('MISSING' in messages[1].content) %}}<cue2>\
         {{%- else %}}<cue3>{{% endif -%}}",
        worker = serde_json::to_string(WORKER_SYSTEM).unwrap(),
        same = serde_json::to_string(SAME_THING).unwrap(),
        conflict = serde_json::to_string(PARTIAL).unwrap(),
        listed = serde_json::to_string(LISTED).unwrap(),
        disagree = serde_json::to_string(&disagreeing(1, 2)).unwrap(),
        lines = serde_json::to_string(LINES_LISTED).unwrap()
    );
    let metadata = [
        ("tokenizer.ggml.model", Value::Str("llama".into())),
        ("tokenizer.ggml.tokens", Value::Arr(tokens)),
        ("tokenizer.ggml.token_type", Value::Arr(types)),
        ("tokenizer.ggml.eos_token_id", Value::U32(258)),
        ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ("tokenizer.ggml.add_space_prefix", Value::Bool(false)),
    ];
    let no_tensors: [(&str, Vec<u64>, u32, Vec<u8>); 0] = [];
    let g = Gguf::parse(&write(&metadata, &no_tensors)).unwrap();
    let tokenizer = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::new(&template, &tokenizer).unwrap();
    let engine = start_with(
        EngineOptions::new("fake".into(), CPU),
        Box::new(move |_, _, _| Ok(Box::new(CloseSecond(FakeModel::new(vocab))))),
    )
    .unwrap();
    Larch::from_engine(engine, tokenizer, template)
}

/// A folder `docs` in `tmp` with `files` in it.
fn folder(tmp: &Path, files: &[(&str, &str)]) -> PathBuf {
    let folder = tmp.join("docs");
    fs::create_dir(&folder).unwrap();
    for (name, text) in files {
        fs::write(folder.join(name), text).unwrap();
    }
    folder
}

/// Asks the scripted model about the rent in `folder`, three workers at a time.
fn ask_rent(tmp: &Path, folder: &Path) -> SkillRun {
    ask(tmp, folder, "How much is the rent?")
}

/// Asks the scripted model `question` about `folder`, three workers at a time.
fn ask(tmp: &Path, folder: &Path, question: &str) -> SkillRun {
    ask_with(AskDocs::default(), tmp, folder, question)
}

/// `skill` on the scripted model, whose replies are whole tokens that the facts grammar would
/// bend (`the_facts_grammar_allows_only_lists_parse_facts_reads` is its own test).
fn unheld(skill: AskDocs) -> AskDocs {
    AskDocs {
        facts_grammar: false,
        ..skill
    }
}

/// Asks the scripted model `question` about `folder` with `skill`, three workers at a time.
fn ask_with(skill: AskDocs, tmp: &Path, folder: &Path, question: &str) -> SkillRun {
    let input = SkillInput {
        folder: folder.to_str().unwrap().into(),
        question: question.into(),
        search: String::new(),
    };
    unheld(skill)
        .run(
            &scripted(),
            &ctx(tmp, folder),
            Workers {
                max_workers: 3,
                prefix_cache: false,
            },
            &input,
        )
        .unwrap()
}

/// Each audit event of the data folder, in order.
fn logged(tmp: &Path) -> Vec<AuditEvent> {
    let mut files: Vec<_> = fs::read_dir(tmp.join("data/agent/log"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|f| {
            fs::read_to_string(f)
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// What two lists of events must share: the times are left out, which the log may round.
fn shape(events: &[AuditEvent]) -> Vec<(EventKind, Option<String>, u64, Option<String>)> {
    events
        .iter()
        .map(|e| (e.kind, e.step.clone(), e.session, e.error.clone()))
        .collect()
}

#[test]
fn a_llama_3_run_answers_from_the_lines_and_cites_them() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    let skill = AskDocs {
        family: Some(Family::Llama3),
        ..AskDocs::default()
    };
    let run = ask_with(skill, tmp.path(), &folder, "How much is the rent?");
    assert_eq!(run.answer, CITED, "{}", run.markdown);
}

#[test]
fn a_run_cites_the_verified_fact_and_counts_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(
        tmp.path(),
        &[
            ("lease.md", LEASE),
            ("notes.md", "Buy milk.\n"),
            ("pancakes.md", "Pancakes: flour, eggs, milk.\n"),
            ("trip.txt", "Train at 9:40 from platform 2.\n"),
        ],
    );
    let run = ask_rent(tmp.path(), &folder);
    // The orchestrator saw the fact that passed, numbered, and neither fact that failed; Rust
    // wrote its citation as the fact's source and dropped the one of no fact. Under the answer,
    // the citation dropped and the excerpt that got no list are counted.
    let notes: Vec<_> = run.markdown.split("\n\n").collect();
    assert_eq!(
        notes[..3].join("\n\n"),
        format!("{CITED}\n\n{DROPPED}"),
        "{}",
        run.markdown
    );
    assert!(notes[3].contains("1 of 4 excerpts"), "{}", run.markdown);
    assert_eq!(run.answer, CITED);
    let facts: Facts = serde_json::from_value(run.facts.clone()).unwrap();
    let want = Facts {
        // The quote is line 8 as Rust copied it.
        cited: vec![Cited {
            claim: "Rent is $1,450 a month".into(),
            quote: "rent is $1,450 per".into(),
            file: "lease.md".into(),
            line: 8,
        }],
        // Two facts from each of the three chunks that gave a list, less the one that passed:
        // line 99 is in no excerpt, and the one-line files have no line 8 either.
        dropped: 5,
        moved: 0,
        // The pancake worker gave no list, before the retry or after it.
        unparsed: 1,
        conflicts: Vec::new(),
    };
    assert_eq!(facts, want);

    // A line per step and per model turn, the same in the log as in the run.
    assert_eq!(shape(&logged(tmp.path())), shape(&run.steps));
    let steps: Vec<_> = run
        .steps
        .iter()
        .map(|e| (e.kind, e.step.as_deref().unwrap()))
        .collect();
    let mut order = vec![(EventKind::Step, "gather"), (EventKind::Step, "rank")];
    for _ in 0..4 {
        order.extend([(EventKind::Turn, "worker"), (EventKind::Turn, "retry")]);
    }
    order.extend([
        (EventKind::Step, "workers"),
        (EventKind::Step, "verify"),
        (EventKind::Turn, "answer"),
    ]);
    assert_eq!(steps, order);
    for e in &run.steps {
        assert_eq!(e.skill.as_deref(), Some("ask-docs"));
        match e.kind {
            EventKind::Step => assert!(e.wall_ms.is_some(), "{e:?}"),
            _ => assert!(
                e.prompt_tokens > Some(0) && e.generated_tokens > Some(0) && e.ttft_ms.is_some(),
                "{e:?}"
            ),
        }
    }
    let detail = |step| {
        let event = run.steps.iter().find(|e| e.step.as_deref() == Some(step));
        event.unwrap().detail.clone()
    };
    assert_eq!(
        detail("gather"),
        Some(json!({"files": 4, "chunks": 4, "skipped": 0, "unlisted": 0,
                    "head": "Read 4 files (0 pdf, 3 md, 1 txt)"}))
    );
    // A folder this small sends every chunk, ranked by nothing: no score.
    let names = ["lease.md", "notes.md", "pancakes.md", "trip.txt"];
    let chosen: Vec<_> = names
        .iter()
        .map(|f| json!({"file": f, "line": 1, "score": null}))
        .collect();
    // The words it ranked by: the question, and the model's search words when it gave some.
    assert_eq!(
        detail("rank"),
        Some(json!({ "chosen": chosen, "ranked_by": "How much is the rent?" }))
    );
    // Every worker's first reply was refused, so each was retried.
    assert_eq!(
        detail("workers"),
        Some(json!({"workers": 4, "at_once": 2, "retried": 4}))
    );
    assert_eq!(detail("verify"), Some(run.facts.clone()));
    // The answer's citations of no fact are counted in its turn, and its thinking.
    assert_eq!(
        detail("answer"),
        Some(
            json!({"unknown_citations": 1, "conflict_sentences_dropped": 1,
                    "names_not_cited": [], "thinking_tokens": 0})
        )
    );
    // Each worker's turns carry its chunk, the lines it read and its reply's length and hash;
    // the pancake one never gave a list.
    let replies: Vec<_> = run.steps[2..10].iter().map(|e| e.detail.clone()).collect();
    let want: Vec<_> = names
        .iter()
        .flat_map(|&f| {
            let retry = if f == "pancakes.md" { BAD } else { FACTS };
            let last = if f == "lease.md" { 9 } else { 1 };
            [BAD, retry].map(|reply| {
                let hash = larch::bench::hex(&larch::bench::sha256(reply.as_bytes()));
                Some(
                    json!({"file": f, "line": 1, "lines": [1, last], "reply_bytes": reply.len(),
                            "reply_sha256": hash}),
                )
            })
        })
        .collect();
    assert_eq!(replies, want);
    // Each worker's two turns share its session, which no other turn has; the error the worker
    // was sent back is on the turn that caused it.
    let turns = &run.steps[2..10];
    for (i, pair) in turns.chunks(2).enumerate() {
        assert_eq!(pair[0].session, pair[1].session);
        assert!(pair[0].error.as_deref().unwrap().starts_with("[agent] "));
        assert_eq!(pair[1].error.is_some(), i == 2, "{pair:?}");
    }
    let mut sessions: Vec<_> = turns.iter().step_by(2).map(|e| e.session).collect();
    sessions.push(run.steps[12].session);
    sessions.sort_unstable();
    sessions.dedup();
    assert_eq!(sessions.len(), 5);

    // One line in today's notes says what the skill did.
    let daily = tmp.path().join("data/agent/memory/daily");
    let notes: Vec<_> = fs::read_dir(daily).unwrap().collect();
    assert_eq!(notes.len(), 1);
    let note = fs::read_to_string(notes[0].as_ref().unwrap().path()).unwrap();
    assert_eq!(note.lines().count(), 1, "{note}");
    assert!(
        note.contains("ask-docs") && note.contains("How much is the rent?"),
        "{note}"
    );
}

#[test]
fn the_audit_log_holds_the_chosen_chunks_and_each_workers_reply_hash() {
    let tmp = tempfile::tempdir().unwrap();
    // One file more than the workers read, so the rank chooses and scores them.
    let mut files = vec![("lease.md".to_string(), LEASE.to_string())];
    files.extend((1..=DOCS_WORKERS).map(|i| {
        (
            format!("n{i:02}.md"),
            format!("Note {i}: the rent was paid.\n"),
        )
    }));
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    let folder = folder(tmp.path(), &files);
    ask_rent(tmp.path(), &folder);
    let log = logged(tmp.path());
    let rank = log
        .iter()
        .find(|e| e.step.as_deref() == Some("rank"))
        .unwrap();
    let chosen = rank.detail.as_ref().unwrap()["chosen"].as_array().unwrap();
    assert_eq!(chosen.len(), DOCS_WORKERS, "{chosen:?}");
    let scores: Vec<f64> = chosen
        .iter()
        .map(|c| c["score"].as_f64().unwrap())
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
    // One event per worker, in the order of the chunks it read, with the reply's length and hash:
    // its text, which quotes the documents, is logged only when LARCH_AGENT_LOG_REPLIES asks.
    let workers: Vec<_> = log
        .iter()
        .filter(|e| e.step.as_deref() == Some("worker"))
        .map(|e| e.detail.clone().unwrap())
        .collect();
    assert_eq!(workers.len(), DOCS_WORKERS);
    for (worker, chunk) in workers.iter().zip(chosen) {
        let hash = larch::bench::hex(&larch::bench::sha256(BAD.as_bytes()));
        let last = if chunk["file"] == "lease.md" { 9 } else { 1 };
        assert_eq!(
            *worker,
            json!({"file": chunk["file"], "line": chunk["line"], "lines": [1, last],
                   "reply_bytes": BAD.len(), "reply_sha256": hash})
        );
    }
}

#[test]
fn no_survivors_gives_the_not_in_the_documents_answer() {
    let tmp = tempfile::tempdir().unwrap();
    let big = "x".repeat(1 << 20) + "\n";
    let folder = folder(
        tmp.path(),
        &[
            ("big.md", &big),
            ("notes.md", "Buy milk.\n"),
            ("pancakes.md", "Pancakes: flour, eggs, milk.\n"),
            ("trip.txt", "Train at 9:40 from platform 2.\n"),
        ],
    );
    let run = ask_rent(tmp.path(), &folder);
    // No model was asked: Rust says so, that one file was not read, and that one excerpt got no
    // list of facts, so a failed worker does not pass for a document that says nothing.
    assert!(run.markdown.starts_with(NOT_FOUND), "{}", run.markdown);
    // The big file is named, with why: not left to a count.
    for note in [
        "1 file was not read",
        "1 of 3 excerpts",
        "big.md was over 1 MB",
    ] {
        assert!(run.markdown.contains(note), "{}", run.markdown);
    }
    assert!(
        !run.steps
            .iter()
            .any(|e| e.step.as_deref() == Some("answer"))
    );
    let facts: Facts = serde_json::from_value(run.facts).unwrap();
    let want = Facts {
        cited: Vec::new(),
        dropped: 4,
        moved: 0,
        unparsed: 1,
        conflicts: Vec::new(),
    };
    assert_eq!(facts, want);
}

#[test]
fn brackets_alone_are_nothing_found_and_a_lone_bracket_gets_one_retry() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    // "[]" at once: no retry, no fact, and Rust's answer.
    let run = ask(tmp.path(), &folder, "How much is the rent? Nothing.");
    assert_eq!(run.markdown, NOT_FOUND);
    let turns: Vec<_> = (run.steps.iter())
        .filter(|e| e.kind == EventKind::Turn)
        .map(|e| (e.step.clone().unwrap(), e.error.is_some()))
        .collect();
    assert_eq!(turns, [("worker".to_string(), false)]);
    // "[" is an error the worker is told of, once; its "[]" then counts.
    let run = ask(tmp.path(), &folder, "How much is the rent? Bracket.");
    assert_eq!(run.markdown, NOT_FOUND);
    let turns: Vec<_> = (run.steps.iter())
        .filter(|e| e.kind == EventKind::Turn)
        .map(|e| (e.step.clone().unwrap(), e.error.clone()))
        .collect();
    assert_eq!(turns.len(), 2, "{turns:?}");
    assert_eq!(turns[0].0, "worker");
    assert!(
        turns[0].1.as_deref().unwrap().contains("JSON list"),
        "{turns:?}"
    );
    assert_eq!(turns[1], ("retry".to_string(), None));
    let facts: Facts = serde_json::from_value(run.facts).unwrap();
    assert_eq!((facts.cited.len(), facts.unparsed), (0, 0));
}

#[test]
fn two_rents_from_two_files_are_a_conflict_line_under_the_answer() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(
        tmp.path(),
        &[
            ("email.txt", "The landlord wrote that the rent is now $1.\n"),
            ("lease.md", "Monthly rent at Alder Street: $1,850.\n"),
        ],
    );
    let run = ask(tmp.path(), &folder, "Which rent do I pay?");
    let facts: Facts = serde_json::from_value(run.facts.clone()).unwrap();
    assert_eq!(facts.conflicts, [(0, 1)]);
    // The orchestrator is told the two disagree (the scripted one then answers `ANSWER`), and
    // the line is under the answer, with both sources.
    // Rust found the conflict, so the answer keeps its sentence that says so.
    let answer = "The rent is $1,450 a month [email.txt:1]. The facts conflict.";
    assert_eq!(
        run.markdown,
        format!(
            "{answer}\n\n_1 citation named no fact and was dropped._\n\n_Conflict: \"The landlord wrote that the rent is now $1.\" \
             [email.txt:1] against \"Monthly rent at Alder Street: $1,850.\" [lease.md:1]._"
        )
    );
    assert_eq!(run.answer, answer);
}

#[test]
fn a_fact_cited_at_the_line_with_its_amount_is_counted_as_moved() {
    let tmp = tempfile::tempdir().unwrap();
    // The worker names line 1 for the rent on line 2.
    let lease = ("lease.md", "Alder Street lease.\nMonthly rent: $1,850.\n");
    let folder = folder(tmp.path(), &[lease]);
    let run = ask(tmp.path(), &folder, "Which rent do I pay?");
    let facts: Facts = serde_json::from_value(run.facts.clone()).unwrap();
    let cited: Vec<_> = facts.cited.iter().map(|c| (c.line, &c.quote[..])).collect();
    assert_eq!(cited, [(2, "Monthly rent: $1,850.")]);
    assert_eq!((facts.moved, facts.dropped), (1, 0));
}

#[test]
fn an_answer_citing_no_excerpt_that_names_the_questions_place_is_withheld() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    // The lease names no Birch Lane: the cited rent is another home's.
    let run = ask(
        tmp.path(),
        &folder,
        "How much is the rent? It is for Birch Lane.",
    );
    assert_eq!(run.answer, NOT_FOUND, "{}", run.markdown);
    assert!(
        run.markdown.contains(
            "_An answer was withheld: no fact it cited names Birch, Lane, which the question \
             names._"
        ),
        "{}",
        run.markdown
    );
    let answer = (run.steps.iter())
        .find(|e| e.step.as_deref() == Some("answer"))
        .unwrap();
    assert_eq!(
        answer.detail,
        Some(
            json!({"unknown_citations": 1, "conflict_sentences_dropped": 1,
                    "names_not_cited": ["Birch", "Lane"], "thinking_tokens": 0})
        )
    );
    // A file that names Birch, not Lane: the note sends the person to Birch alone.
    let tmp = tempfile::tempdir().unwrap();
    let trees = ("trees.md", "Birch trees line the street.\n");
    let folder = self::folder(tmp.path(), &[("lease.md", LEASE), trees]);
    let run = ask(
        tmp.path(),
        &folder,
        "How much is the rent? It is for Birch Lane.",
    );
    assert!(
        run.markdown
            .contains("The documents do name Birch, in parts the facts did not come from"),
        "{}",
        run.markdown
    );
    // The lease names its tenant, so the same answer stands for Sam Lee.
    let tmp = tempfile::tempdir().unwrap();
    let lease = self::folder(tmp.path(), &[("lease.md", LEASE)]);
    let run = ask(
        tmp.path(),
        &lease,
        "How much is the rent? It is for Sam Lee.",
    );
    assert_eq!(run.answer, CITED, "{}", run.markdown);
}

#[test]
fn the_workers_read_the_window_docs_window_names() {
    let tmp = tempfile::tempdir().unwrap();
    let paragraphs: Vec<_> = (0..10).map(paragraph).collect();
    let folder = folder(tmp.path(), &[("log.md", &paragraphs.join("\n"))]);
    // Four chunks at lines 1, 16, 31 and 46, the file's last line 49.
    let read = |window| {
        let skill = AskDocs {
            docs_window: window,
            ..AskDocs::default()
        };
        let run = ask_with(
            skill,
            tmp.path(),
            &folder,
            "When was the boiler serviced? Nothing.",
        );
        let lines: Vec<_> = (run.steps.iter())
            .filter(|e| e.step.as_deref() == Some("worker"))
            .map(|e| e.detail.as_ref().unwrap()["lines"].clone())
            .collect();
        lines
    };
    assert_eq!(
        read(0),
        [
            json!([1, 14]),
            json!([16, 29]),
            json!([31, 44]),
            json!([46, 49])
        ]
    );
    // 128 tokens are 512 bytes: seven lines of 67 bytes and two blank ones a side, 471 bytes, as
    // an eighth would not fit; none before the first chunk or after the last.
    assert_eq!(
        read(128),
        [
            json!([1, 23]),
            json!([7, 38]),
            json!([22, 49]),
            json!([37, 49])
        ]
    );
}

#[test]
fn an_empty_answer_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    // An answer that cites nothing is not withheld for the names it does not give.
    let run = ask(
        tmp.path(),
        &folder,
        "How much is the rent? Say nothing of Birch Lane.",
    );
    assert_eq!(run.markdown, "_The model gave an empty answer._");
    let answer = run.steps.last().unwrap();
    assert_eq!(answer.step.as_deref(), Some("answer"));
    assert_eq!(answer.error.as_deref(), Some("empty answer"));
}

#[test]
fn a_folder_without_documents_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("scan.pdf", "%PDF-1.4\n")]);
    let run = ask_rent(tmp.path(), &folder);
    let note = format!(
        "_No .pdf, .md or .txt file was read from {}._",
        folder.display()
    );
    assert_eq!(run.markdown, format!("{NOT_FOUND}\n\n{note}"));
}

#[test]
fn the_spec_names_the_skill_as_its_audit_lines_do() {
    // The fields a SKILL.md front matter has: the audit log's name, and no tool.
    let spec = AskDocs::spec();
    assert!(spec.tools.is_empty());
    assert!(spec.description.contains("file:line"), "{spec:?}");
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("scan.pdf", "%PDF-1.4\n")]);
    for e in ask_rent(tmp.path(), &folder).steps {
        assert_eq!(e.skill.as_ref(), Some(&spec.name));
    }
}

#[test]
fn no_excerpt_sharing_a_word_with_the_question_says_so() {
    // Text without spaces is one FTS5 token a sentence, so "家賃" matches none of 17 files.
    let tmp = tempfile::tempdir().unwrap();
    let files: Vec<_> = (1..=17)
        .map(|n| (format!("{n:02}.md"), format!("家賃は月額{n}万円です。\n")))
        .collect();
    let files: Vec<_> = files
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    let folder = folder(tmp.path(), &files);
    let run = ask(tmp.path(), &folder, "家賃");
    let note = "_None of the 17 excerpts shares a word with the question, so none was read._";
    assert_eq!(run.markdown, format!("{NOT_FOUND}\n\n{note}"));
}

#[test]
fn a_worker_engine_error_fails_the_run_after_the_turns_are_logged() {
    let tmp = tempfile::tempdir().unwrap();
    // A paragraph of 5,000 bytes in five chunks, each with a window of the whole file: a prompt
    // past the fake engine's 4,096 tokens.
    let long = "word word\n".repeat(500);
    let folder = folder(tmp.path(), &[("lease.md", LEASE), ("long.md", &long)]);
    let input = SkillInput {
        folder: folder.to_str().unwrap().into(),
        question: "How much is the rent?".into(),
        search: String::new(),
    };
    let skill = AskDocs {
        docs_window: 2000,
        ..AskDocs::default()
    };
    let got = unheld(skill).run(
        &scripted(),
        &ctx(tmp.path(), &folder),
        Workers {
            max_workers: 3,
            prefix_cache: false,
        },
        &input,
    );
    assert!(
        matches!(
            got,
            Err(SkillError::Engine(EngineError::PromptTooLong { .. }))
        ),
        "{got:?}"
    );
    let log = logged(tmp.path());
    let workers = log.iter().find(|e| e.step.as_deref() == Some("workers"));
    let error = workers.and_then(|e| e.error.as_deref()).unwrap_or_default();
    assert!(error.contains("4096"), "{log:?}");
    // The lease worker's two turns are in the log; nothing came after the workers step.
    let turns: Vec<_> = log.iter().filter_map(|e| e.step.as_deref()).collect();
    assert_eq!(turns, ["gather", "rank", "worker", "retry", "workers"]);
}

#[test]
fn a_worker_whose_retry_fails_has_its_first_turn_logged() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    let input = SkillInput {
        folder: folder.to_str().unwrap().into(),
        question: "How much is the rent? Raise.".into(),
        search: String::new(),
    };
    let got = unheld(AskDocs::default()).run(
        &scripted(),
        &ctx(tmp.path(), &folder),
        Workers {
            max_workers: 3,
            prefix_cache: false,
        },
        &input,
    );
    assert!(
        matches!(got, Err(SkillError::Engine(EngineError::InvalidRequest(_)))),
        "{got:?}"
    );
    let steps: Vec<_> = logged(tmp.path())
        .into_iter()
        .filter_map(|e| e.step)
        .collect();
    assert_eq!(steps, ["gather", "rank", "worker", "workers"]);
}

#[test]
fn a_retry_that_lost_its_excerpt_counts_as_no_list() {
    // The session drops the excerpt and the first reply to fit the retry, so the "[]" the worker
    // gives is about nothing: a worker without a list, not one that found nothing.
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    let run = ask(tmp.path(), &folder, "How much is the rent? Forget.");
    let note = "_No list of facts came back for 1 of 1 excerpts, so the answer may miss what they \
                say; the audit log says why._";
    assert_eq!(run.markdown, format!("{NOT_FOUND}\n\n{note}"));
    let retry = run
        .steps
        .iter()
        .find(|e| e.step.as_deref() == Some("retry"));
    let error = retry.and_then(|e| e.error.as_deref()).unwrap_or_default();
    assert!(error.contains("excerpt"), "{retry:?}");
}

#[test]
fn an_answer_cut_at_the_reply_limit_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    let run = ask(
        tmp.path(),
        &folder,
        "How much is the rent? Answer at length.",
    );
    assert!(
        run.markdown
            .ends_with("_The answer stops at the limit of 1024 tokens._"),
        "{}",
        run.markdown
    );
    let answer = run.steps.last().unwrap();
    assert_eq!(answer.step.as_deref(), Some("answer"));
    assert_eq!(answer.generated_tokens, Some(1024));
    let error = answer.error.as_deref().unwrap_or_default();
    assert!(error.contains("1024 tokens"), "{answer:?}");
}

#[test]
fn the_validators_score_hand_made_runs_on_the_document_set() {
    // A fact the bench scores: `quote`, from `file`, at the line where it first starts. The skill
    // cites whole lines, but the scoring takes any quote.
    let cited = |file: &Path, quote: &str| {
        let text = fs::read_to_string(Path::new(DOCS).join(file)).unwrap();
        let line = text[..text.find(quote).unwrap()].matches('\n').count() as u32 + 1;
        Cited {
            claim: "claim".into(),
            quote: quote.into(),
            file: file.into(),
            line,
        }
    };
    let cite = |c: &Cited| format!("{}:{}", c.file.display(), c.line);
    let run = |answer: String, cited: Vec<Cited>, dropped| -> Result<SkillRun, SkillError> {
        Ok(SkillRun {
            markdown: answer.clone(),
            answer,
            facts: serde_json::to_value(Facts {
                cited,
                dropped,
                moved: 0,
                unparsed: 0,
                conflicts: Vec::new(),
            })
            .unwrap(),
            steps: Vec::new(),
        })
    };
    let qs = questions();
    let one = qs.iter().find(|q| q.facts.len() == 1).unwrap();
    let two = qs.iter().find(|q| q.facts.len() == 2).unwrap();
    let none: Vec<_> = qs.iter().filter(|q| q.facts.is_empty()).collect();
    assert_eq!(none.len(), 4);

    // Recalled: the planted quote, cited.
    let p = &one.facts[0];
    let a = cited(&p.file, &p.quote);
    let r1 = run(format!("Yes ({}).", cite(&a)), vec![a], 1);

    // The first planted fact is recalled through a quote that shares its last 20 characters.
    // The second is not: its own quote is cited only inside a longer name and with a digit after
    // the line, and another sentence of its file, cited, does not overlap it.
    let (p1, p2) = (&two.facts[0], &two.facts[1]);
    let text = fs::read_to_string(Path::new(DOCS).join(&p1.file)).unwrap();
    let end = text.find(&p1.quote).unwrap() + p1.quote.len();
    let b = cited(&p1.file, &text[end - 20..end + 30]);
    let c = cited(&p2.file, &p2.quote);
    let p2_text = fs::read_to_string(Path::new(DOCS).join(&p2.file)).unwrap();
    let d = cited(&p2.file, p2_text.lines().next().unwrap());
    let (b_at, c_at, d_at) = (cite(&b), cite(&c), cite(&d));
    let r2 = run(
        format!("{b_at} (old-{c_at}) ({c_at}0), ({d_at})."),
        vec![b, c, d],
        0,
    );

    // Abstained only when the answer says so and cites nothing, not even a fact that failed.
    let r3 = run(NOT_FOUND.into(), Vec::new(), 3);
    let e = cited(&p1.file, &p1.quote);
    let r4 = run(format!("{NOT_FOUND} But see ({}).", cite(&e)), vec![e], 0);
    let r5 = run(format!("{NOT_FOUND} But see notes.md:3."), Vec::new(), 0);
    // A model may leave off the full stop.
    let r6 = run(NOT_FOUND.trim_end_matches('.').into(), Vec::new(), 0);

    let asked: Vec<Question> = [one, two].into_iter().chain(none).cloned().collect();
    let m = docs_metrics(&[r1, r2, r3, r4, r5, r6], &asked).unwrap();
    let want = Docs {
        // 5 cited, of 5 cited and 4 dropped.
        precision: Some(5.0 / 9.0),
        recall: Some(2.0 / 3.0),
        abstain_rate: Some(2.0 / 4.0),
        errors: Default::default(),
        questions: Vec::new(),
    };
    assert_eq!(
        Docs {
            questions: Vec::new(),
            ..m.clone()
        },
        want
    );
    // Each question: facts given and passed, the planted facts recalled (by index), whether the
    // answer abstains, and the facts the answer cites.
    let each: Vec<_> = m
        .questions
        .iter()
        .map(|q| {
            let cited = q.cited.len();
            (
                q.id.as_str(),
                q.facts_emitted,
                q.facts_verified,
                q.recalled.clone(),
                q.abstained,
                cited,
            )
        })
        .collect();
    let id = |i: usize| asked[i].id.as_str();
    assert_eq!(
        each,
        [
            (id(0), 2, 1, vec![0], false, 1),
            (id(1), 3, 3, vec![0], false, 2),
            (id(2), 3, 0, vec![], true, 0),
            (id(3), 1, 1, vec![], false, 1),
            (id(4), 0, 0, vec![], false, 0),
            (id(5), 0, 0, vec![], true, 0),
        ]
    );
    let first = &m.questions[0];
    assert!(first.answer.starts_with("Yes ("), "{first:?}");
    let planted = CitedQuote {
        file: p.file.clone(),
        line: p.line,
        quote: p.quote.clone(),
    };
    assert_eq!(first.cited, [planted]);
    assert_eq!(docs_metrics(&[], &[]).unwrap(), Docs::default());

    // Only the answer alone abstains: q21's near miss told after the sentence leaks, though it
    // cites nothing. The notes Rust writes under the answer do not count.
    let q21 = qs.iter().find(|q| q.id == "q21").unwrap();
    let leak = format!("{NOT_FOUND} The Wi-Fi password is maple-otter-58.");
    let noted = run(NOT_FOUND.into(), Vec::new(), 0).map(|r| SkillRun {
        markdown: format!("{NOT_FOUND}\n\n_1 file was not read._"),
        ..r
    });
    let m = docs_metrics(
        &[run(leak, Vec::new(), 0), noted],
        &[q21.clone(), q21.clone()],
    )
    .unwrap();
    assert_eq!(m.abstain_rate, Some(0.5));

    // Not recalled: a short quote of the planted one, from another line of its file ("month" of
    // "twelve months" on the lease's line 6, where the rent is on line 14), and the planted quote
    // itself said to be from another file.
    let short = cited(&p.file, "month");
    assert!(
        p.quote.contains("month") && short.line != p.line,
        "{short:?}"
    );
    let elsewhere = Cited {
        file: "lease-maple-avenue-2024.md".into(),
        ..cited(&p.file, &p.quote)
    };
    let (short_at, elsewhere_at) = (cite(&short), cite(&elsewhere));
    let r = run(
        format!("Twelve months ({short_at}), ({elsewhere_at})."),
        vec![short, elsewhere],
        0,
    );
    let m = docs_metrics(&[r], std::slice::from_ref(one)).unwrap();
    assert_eq!(m.recall, Some(0.0));

    // Recalled: a part of a planted quote that spans two lines, from its second line.
    let q09 = qs.iter().find(|q| q.id == "q09").unwrap();
    let p = &q09.facts[0];
    let part = cited(&p.file, "to 2.75% APY");
    assert_eq!((p.quote.lines().count(), part.line), (2, p.line + 1));
    let r = run(format!("2.75% ({}).", cite(&part)), vec![part], 0);
    let m = docs_metrics(&[r], std::slice::from_ref(q09)).unwrap();
    assert_eq!(m.recall, Some(1.0));
}

#[test]
fn bench_docs_asks_every_question_of_the_folder_and_scores_the_answers() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    // The second and fourth questions are too long for the context, so their workers fail; the
    // others still run.
    let rent = json!([{"file": "lease.md", "quote": "rent is $1,450 per", "line": 8}]);
    let long = "word ".repeat(1000);
    let questions = json!([
        {"id": "q01", "question": "How much is the rent?", "facts": rent},
        {"id": "q02", "question": long, "facts": rent},
        {"id": "q03", "question": "When does the train leave?", "facts": []},
        {"id": "q04", "question": long, "facts": []},
    ]);
    fs::write(folder.join("questions.json"), questions.to_string()).unwrap();
    let set = bench::doc_set(&folder).unwrap();
    let mut asked = Vec::new();
    let docs = bench::run_docs(
        &scripted(),
        None,
        Default::default(),
        false,
        &set,
        Workers::default(),
        |q, run| {
            let run = run.as_ref().map(|r| r.markdown.clone());
            asked.push((q.id.clone(), run.map_err(|e| e.to_string())));
        },
    )
    .unwrap();
    assert_eq!(
        asked[0],
        ("q01".into(), Ok(format!("{CITED}\n\n{DROPPED}")))
    );
    let failed = asked[1].1.clone().unwrap_err();
    assert!(failed.contains("4096"), "{failed}");
    let train = asked[2].1.as_ref().unwrap();
    assert!(train.starts_with(NOT_FOUND), "{train}");
    assert_eq!(asked[3].1, Err(failed.clone()));
    // The rent: one fact passed of two, and the planted fact is cited. The long questions failed,
    // so the second recalls nothing of its planted fact and the fourth did not abstain, and the
    // report keeps their errors. The train: no fact, and the answer says so.
    let want = Docs {
        precision: Some(0.5),
        recall: Some(0.5),
        abstain_rate: Some(0.5),
        errors: [("q02".into(), failed.clone()), ("q04".into(), failed)].into(),
        questions: Vec::new(),
    };
    assert_eq!(
        Docs {
            questions: Vec::new(),
            ..docs.clone()
        },
        want
    );
    // The questions that ran, each with its time.
    let ran: Vec<_> = docs
        .questions
        .iter()
        .map(|q| {
            (
                q.id.as_str(),
                q.recalled.clone(),
                q.abstained,
                q.wall_ms > 0.0,
            )
        })
        .collect();
    assert_eq!(
        ran,
        [("q01", vec![0], false, true), ("q03", vec![], true, true)]
    );
    // A run per question, or no score.
    assert!(docs_metrics(&[], &set.questions).is_err());
    // An ID twice would leave one error under it, so the set fails before its run.
    let twice = json!([questions[0], questions[0]]);
    fs::write(folder.join("questions.json"), twice.to_string()).unwrap();
    let e = bench::doc_set(&folder).unwrap_err().to_string();
    assert!(e.contains("q01 twice"), "{e}");
}

#[test]
fn the_cli_prints_the_error_and_exits_1() {
    let tmp = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .args([
            "run",
            "ask-docs",
            "--question",
            "When?",
            "--folder",
            DOCS,
            "--allow",
            DOCS,
            "-m",
            "missing",
            "--cpu",
        ])
        .arg("--data-dir")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("missing.gguf"), "{stderr}");
    assert!(out.stdout.is_empty());
}

#[test]
fn gather_stops_walking_at_20000_entries_and_the_answer_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[]);
    // a/ fills the walk, so b/ after it is never listed.
    fs::create_dir(folder.join("a")).unwrap();
    for i in 0..20_000 {
        fs::File::create(folder.join("a").join(i.to_string())).unwrap();
    }
    fs::create_dir(folder.join("b")).unwrap();
    fs::write(folder.join("b/lease.md"), "Rent: 1,850 USD a month.\n").unwrap();
    let g = gather(&ctx(tmp.path(), &folder), folder.to_str().unwrap()).unwrap();
    assert_eq!((g.files, g.stopped), (0, true));
    let run = ask_rent(tmp.path(), &folder);
    assert!(
        run.markdown.ends_with(
            "\n\n_Stopped after 20000 files and folders; name a smaller folder to search the \
             rest._"
        ),
        "{}",
        run.markdown
    );
}

#[test]
fn a_question_extracts_at_most_max_pdfs_and_the_next_one_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    fs::create_dir(&docs).unwrap();
    let lease = Path::new(DOCS_PDF).join("lease-alder-street.pdf");
    for i in 0..=MAX_PDFS {
        fs::copy(&lease, docs.join(format!("lease-{i:02}.pdf"))).unwrap();
    }
    let ctx = ctx(tmp.path(), &docs);
    let folder = docs.to_str().unwrap();
    let g = gather(&ctx, folder).unwrap();
    assert_eq!((g.files, g.skipped), (MAX_PDFS, 1));
    // The 20 come from the cache, so the one left is extracted.
    let g = gather(&ctx, folder).unwrap();
    assert_eq!((g.files, g.skipped), (MAX_PDFS + 1, 0));
}

#[test]
fn a_stopped_question_extracts_no_pdf_and_caches_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let ctx = ctx(tmp.path(), Path::new(DOCS_PDF));
    let g = gather_in(&ctx, Some(DOCS_PDF), &AtomicBool::new(true)).unwrap();
    assert_eq!((g.files, g.skipped, g.blank.len()), (0, 2, 0));
    assert!(!tmp.path().join("data/agent/extract").exists());
    assert_eq!(head(&g, "docs"), "No readable files in docs (2 skipped)");
}

#[test]
fn no_folder_named_reads_every_attachment_each_folder_under_its_name() {
    let tmp = tempfile::tempdir().unwrap();
    let notes = tmp.path().join("notes");
    fs::create_dir(&notes).unwrap();
    fs::write(notes.join("a.md"), "The alarm code is 4417.\n").unwrap();
    let lease = tmp.path().join("lease.txt");
    fs::write(&lease, "Monthly rent: $1,850.\n").unwrap();
    let ctx = ToolCtx {
        allowed: vec![notes, lease, tmp.path().join("gone")],
        data_dir: tmp.path().join("data"),
        source: String::new(),
    };
    let g = gather_in(&ctx, None, &AtomicBool::default()).unwrap();
    let files: Vec<_> = g.texts.keys().cloned().collect();
    assert_eq!(
        files,
        [PathBuf::from("lease.txt"), PathBuf::from("notes/a.md")]
    );
    assert_eq!(g.files, 2);
    assert_eq!(g.unlisted, [tmp.path().join("gone")]);
    assert!(g.chunks.iter().any(|c| c.file == Path::new("notes/a.md")));
}

#[test]
fn the_first_line_names_skipped_files_and_at_most_five_types() {
    let g = Gathered {
        others: [".a", ".b", ".c", ".d", ".e", ".f"]
            .map(String::from)
            .into(),
        ..Gathered::default()
    };
    assert_eq!(
        head(&g, "~"),
        "No readable files in ~ (saw: .a, .b, .c, .d, .e, …)"
    );
    let g = Gathered { skipped: 3, ..g };
    assert_eq!(head(&g, "~"), "No readable files in ~ (3 skipped)");
}

#[test]
fn a_docs_question_runs_its_before_calls_first_and_one_that_fails_ends_it() {
    let tmp = tempfile::tempdir().unwrap();
    let folder = folder(tmp.path(), &[("lease.md", LEASE)]);
    let listed = json!([{"tool": "list_dir", "args": {"path": folder}}]);
    let unknown = json!([{"tool": "no_such_tool", "args": {}}]);
    let questions = json!([
        {"id": "q01", "question": "When does the train leave?", "facts": [], "before": listed},
        {"id": "q02", "question": "When does the train leave?", "facts": [], "before": unknown},
    ]);
    fs::write(folder.join("questions.json"), questions.to_string()).unwrap();
    let set = bench::doc_set(&folder).unwrap();
    let docs = bench::run_docs(
        &scripted(),
        None,
        Default::default(),
        false,
        &set,
        Workers::default(),
        |_, _| {},
    )
    .unwrap();
    let q = &docs.questions[0];
    assert_eq!(q.id, "q01");
    assert!(q.before[0].contains("lease.md"), "{:?}", q.before);
    assert!(
        docs.errors["q02"].contains("no_such_tool"),
        "{:?}",
        docs.errors
    );
}

/// A docs question may say the outcome a right run comes to (#400's docs-pdf and docs-docx
/// questions): it is scored on `bench::outcome` of ask_docs's first line, from the gather step,
/// then the answer. A question without one scores as before.
#[test]
fn a_docs_questions_outcome_is_scored_on_ask_docs_first_line() {
    let asked: Vec<Question> = serde_json::from_value(json!([
        {"id": "pdf-q22", "question": "What is my passport number?", "facts": [], "outcome": "unreadable"},
        {"id": "docx-q01", "question": "What is the monthly rent?", "facts": [], "outcome": "no_files"},
        {"id": "pdf-q23", "question": "What is my passport number?", "facts": [], "outcome": "unreadable"},
        {"id": "q24", "question": "How long is the dishwasher warranty?", "facts": []},
    ]))
    .unwrap();
    assert_eq!(asked[0].outcome, Some(bench::Outcome::Unreadable));
    assert_eq!(asked[3].outcome, None);
    let run = |head: Option<&str>| -> Result<SkillRun, SkillError> {
        let gather = AuditEvent {
            kind: EventKind::Step,
            step: Some("gather".into()),
            detail: head.map(|h| json!({"files": 0, "head": h})),
            ..AuditEvent::default()
        };
        Ok(SkillRun {
            markdown: NOT_FOUND.into(),
            answer: NOT_FOUND.into(),
            facts: serde_json::to_value(Facts::default()).unwrap(),
            steps: vec![gather],
        })
    };
    let runs = [
        run(Some("No text in scan-passport.pdf")),
        run(Some("No readable files in /docs (saw: .docx)")),
        // A folder it read: the refusal is no unreadable file.
        run(Some("Read 1 file (1 pdf, 0 md, 0 txt)")),
        run(None),
    ];
    let m = docs_metrics(&runs, &asked).unwrap();
    let got: Vec<_> = (m.questions.iter())
        .map(|q| (q.id.as_str(), q.outcome, q.right))
        .collect();
    use bench::Outcome::{NoFiles, Refused, Unreadable};
    assert_eq!(
        got,
        [
            ("pdf-q22", Some(Unreadable), true),
            ("docx-q01", Some(NoFiles), true),
            ("pdf-q23", Some(Refused), false),
            ("q24", Some(Refused), true),
        ]
    );
}

/// The workers' grammar holds a reply to what `parse_facts` reads: a small model's lone object, a
/// blank inside an empty list and words around a list are not its output.
#[test]
fn the_facts_grammar_allows_only_lists_parse_facts_reads() {
    let g = facts_grammar();
    let held = |text: &str| larch::api::grammar::matches(&g, text).unwrap();
    for list in [
        "[]",
        r#"[{"line": 12, "claim": "the rent is $1,850"}]"#,
        // A list over lines, with an escaped quote in a claim.
        r#"[
  {"line": 3, "claim": "a"},
  {"line": 4, "claim": "b \"c\""}
]"#,
    ] {
        assert!(held(list), "{list}");
        parse_facts(list, Finish::Eos).unwrap();
    }
    for other in [
        r#"{"line": 12, "claim": "x"}"#,
        "[ ]",
        r#"[{"claim": "x"}]"#,
        r#"[{"line": 1, "claim": "x", "quote": "y"}]"#,
        r#"Here: [{"line": 1, "claim": "x"}]"#,
    ] {
        assert!(!held(other), "{other}");
    }
}

/// A PDF's pages come from its form feeds, which leave its lines as they were, and a citation of
/// a line of a PDF names its page: "pdf:4595" named nothing the person could find (24 Nov).
#[test]
fn a_pdf_citation_names_its_page_not_a_line_of_its_text() {
    assert_eq!(page_lines("a\nb\n\u{c}c\n\u{c}d\ne\n\u{c}"), [1, 3, 4]);
    assert_eq!(page_lines("one page\n"), [1]);
    let file = PathBuf::from("guide.pdf");
    let mut g = Gathered::default();
    g.pages.insert(file.clone(), vec![1, 40, 95]);
    let at = |line| Cited {
        claim: "a limit".into(),
        quote: "a limit".into(),
        file: file.clone(),
        line,
    };
    let note = Cited {
        file: "notes.md".into(),
        ..at(7)
    };
    let facts = Facts {
        cited: vec![at(12), at(41), at(60), note],
        ..Facts::default()
    };
    assert_eq!((g.page(&file, 1), g.page(&file, 39)), (Some(1), Some(1)));
    assert_eq!((g.page(&file, 40), g.page(&file, 200)), (Some(2), Some(3)));
    let answer = "The limit is $7,000 [guide.pdf:12]. It is phased out [guide.pdf:41, \
                  guide.pdf:60, notes.md:7]. See [1].";
    assert_eq!(
        with_pages(answer, &facts, &g),
        "The limit is $7,000 [guide.pdf, p. 1]. It is phased out [guide.pdf, p. 2; notes.md:7]. \
         See [1]."
    );
    // A citation of a line that is the start of another's is its own.
    let facts = Facts {
        cited: vec![at(4)],
        ..Facts::default()
    };
    assert_eq!(with_pages("[guide.pdf:41]", &facts, &g), "[guide.pdf:41]");
}

/// A PDF read by gather knows its pages.
#[test]
fn gather_keeps_a_pdfs_pages_and_none_of_its_form_feeds() {
    let tmp = tempfile::tempdir().unwrap();
    let g = gather(&ctx(tmp.path(), Path::new(DOCS_PDF)), DOCS_PDF).unwrap();
    let lease = Path::new("lease-alder-street.pdf");
    assert_eq!(g.pages[lease].first(), Some(&1));
    assert!(!g.texts[lease].contains('\u{c}'));
    assert!(g.chunks.iter().all(|c| !c.text.contains('\u{c}')));
}

/// A PDF's contents and index pages name every subject of the book, so they ranked first for a
/// tax guide's questions and their workers found nothing (24 Nov): they are left out of the
/// ranking. The same lines in a note are its content and rank.
#[test]
fn a_pdfs_contents_and_index_pages_rank_after_the_rest() {
    // An index's entries under its letter heading, wrapped as a PDF's two columns wrap them.
    let index = "Roth IRAs 38\nAge limit 39\nContribution limit reduced 40\nDetermining reduced \
                 limit\n(Worksheet 2-2) 43\nContributions 39-44\nR\nConversion 29, 44\n\
                 Defined 39\nExcess contributions 43\nIncome limits 41\nRecharacterization 29\n";
    let contents = "Contents\nChapter 2. Roth IRAs . . . . . . 38\nWhat Is a Roth IRA? . . . . . 39\n\
                    Income limits . . . . . . 40\nContribution limits . . . . . 41\n\
                    Can You Contribute to a Roth IRA? . . . . 39\nRoth contributions . . . . 42\n\
                    Roth IRA rules . . . . . 43\nRoth income . . . . . 44\n";
    // A pin table's rows end in numbers too, and are facts.
    let pins = "Pin table\nPA0 14\nPA1 15\nPA2 16\nPA3 17\nPA4 20\nPA5 21\nPA6 22\nPA7 23\n\
                Roth IRA income limits contribution limit 24\n";
    let rule = "For 2025, your Roth IRA contribution limit is reduced if your modified AGI is at \
                least $150,000. Income limits apply.\n";
    let other = |n: usize| format!("Chapter {n} is about something else: rollovers.\n");
    let chunks = |file: &str| -> Vec<Chunk> {
        [
            index.to_owned(),
            rule.to_owned(),
            contents.to_owned(),
            pins.to_owned(),
        ]
        .into_iter()
        .chain((0..8).map(other))
        .enumerate()
        .map(|(i, text)| Chunk {
            file: file.into(),
            line: i as u32 * 20 + 1,
            heading: Arc::from(Vec::new()),
            start: i * 1000,
            text,
        })
        .collect()
    };
    let question = "What are the Roth IRA income limits and contribution limit?";
    let pdf = rank(&chunks("guide.pdf"), question, 3).unwrap();
    let mut first: Vec<usize> = pdf[..2].iter().map(|c| c.0).collect();
    first.sort();
    assert_eq!(first, [1, 3], "the facts rank first: {pdf:?}");
    assert!(
        [0, 2].contains(&pdf[2].0),
        "a pointer page ranks after them: {pdf:?}"
    );
    let md = rank(&chunks("guide.md"), question, 3).unwrap();
    assert!(md.iter().any(|c| c.0 == 0 || c.0 == 2), "{md:?}");
}

/// A limits table with a column letter is no index: its amounts ("8,000") are not page numbers,
/// so it ranks as the facts it holds, and a page `pointers` mistakes for one ranks after the
/// rest instead of never (review of the contents and index filter, 24 Nov).
#[test]
fn a_pdfs_table_of_amounts_with_a_letter_column_is_no_index() {
    let table = "Table 1-1. Contribution limits\nA\nAge under 50 7,000\nAge 50 or older 8,000\n\
                 Spousal IRA 7,000\nCatch-up amount 1,000\nSIMPLE IRA 16,500\n\
                 SIMPLE IRA age 50 or older 20,000\nSEP limit 70,000\nSEP compensation 350,000\n\
                 Roth IRA limit 7,000\n";
    let chunk = |text: &str| Chunk {
        file: "guide.pdf".into(),
        line: 1,
        heading: Arc::from(Vec::new()),
        start: 0,
        text: text.into(),
    };
    assert!(!pointers(&chunk(table)), "a table of amounts is kept");
    // A pointer page alone still ranks: the filter orders, it never hides.
    let index = "R\nRoth IRAs 38\nAge limit 39\nContribution limit 40\nContributions 39-44\n\
                 Conversion 29, 44\nDefined 39\nExcess contributions 43\nIncome limits 41\n";
    assert!(pointers(&chunk(index)));
    let ranked = rank(&[chunk(index)], "Roth IRA contribution limit", 1).unwrap();
    assert_eq!(ranked.iter().map(|c| c.0).collect::<Vec<_>>(), [0]);
}

/// The words the model gives in `search` rank with the question's: a question in the user's words
/// ("income limits for opening a Roth IRA") finds the part in the document's ("modified AGI ...
/// phased out"), which it shares no word with (a tax guide, 24 Nov).
#[test]
fn the_models_search_words_rank_with_the_question() {
    let tmp = tempfile::tempdir().unwrap();
    let mut files: Vec<(String, String)> = (0..9)
        .map(|n| {
            (
                format!("note-{n}.md"),
                format!("Opening hours for someone: {n} to 5.\n"),
            )
        })
        .collect();
    let phase = "Your contribution is phased out when your modified AGI is $150,000.\n";
    files.push(("rules.md".into(), phase.into()));
    let named: Vec<(&str, &str)> = files
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    let folder = folder(tmp.path(), &named);
    let chosen = |search: &str| {
        let input = SkillInput {
            folder: folder.to_str().unwrap().into(),
            question: "What are the income limits for opening one for someone?".into(),
            search: search.into(),
        };
        let run = unheld(AskDocs::default())
            .run(
                &scripted(),
                &ctx(tmp.path(), &folder),
                Workers {
                    max_workers: 3,
                    prefix_cache: false,
                },
                &input,
            )
            .unwrap();
        let rank = run.steps.iter().find(|e| e.step.as_deref() == Some("rank"));
        let detail = rank.and_then(|e| e.detail.clone()).unwrap();
        (detail["chosen"].as_array().unwrap().iter())
            .map(|c| c["file"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert!(!chosen("").contains(&"rules.md".to_owned()));
    assert_eq!(chosen("modified AGI phased out")[0], "rules.md");
}
