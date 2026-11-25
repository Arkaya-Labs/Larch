//! Aider's Polyglot benchmark (Aider-AI/polyglot-benchmark: 225 Exercism exercises in C++, Go,
//! Java, JavaScript, Python and Rust) on code mode, by Aider's protocol (benchmark/benchmark.py):
//! the model gets the exercise's instructions and its solution files, never its tests; the tests
//! then run outside code mode, and when they fail a second try gets their output. pass1 and pass2
//! are the share passing after each try; pass2 is the headline.
//!
//!   LARCH_MODEL=<gguf> POLYGLOT_DIR=<checkout> POLYGLOT_OUT=<file.jsonl> \
//!     cargo test --release -p larch-agent --test polyglot -- --ignored --nocapture polyglot
//!
//! POLYGLOT_LANGS=python,rust runs only those; POLYGLOT_LIMIT=n the first n of each language, by
//! name; POLYGLOT_NPM=<dir> holds the node_modules the JavaScript tests run with (Aider's
//! /npm-install). One JSON line per exercise is appended to POLYGLOT_OUT, and an exercise already
//! there is skipped, so a run that stopped goes on where it was; the summary covers the whole file.
//! Each exercise's code mode data folder, with the run's trace, and each try's diff and test
//! output are kept under `<POLYGLOT_OUT>.traces/<lang>/<name>/`.
//! `grader_passes_examples_and_fails_stubs` checks the toolchains: every exercise's test command
//! fails on its stub and passes on its example solution.
#![cfg(unix)]

mod common;

use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, OpenOptions},
    io::{self, Read, Seek, Write},
    os::unix::{fs::symlink, process::CommandExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use larch::Larch;
use larch_agent::{
    AgentEvent,
    code::{
        allow::Decision,
        r#loop::{BUDGET, Code, decide},
        task::{CodeTask, Overrides},
    },
};
use regex::Regex;
use serde_json::{Value, json};

/// The languages, as the checkout names their folders.
const LANGS: [&str; 6] = ["cpp", "go", "java", "javascript", "python", "rust"];

/// Aider's limit on one test run (`run_unit_tests`).
const TEST_TIMEOUT: Duration = Duration::from_secs(180);

/// The bytes of failing test output the second try sees, its end.
// ponytail: Aider sends the whole output; a cap keeps a gradle or cmake log from crowding a
// local model's context. Raise it when a model's context has the room.
const TAIL_BYTES: usize = 8192;

/// What makes code mode find a test command itself (task.rs `RUNNERS`): hidden from the model
/// with the tests, so it runs none, as Aider's model runs none; the grader puts them back.
const MANIFESTS: [&str; 4] = ["Cargo.toml", "pyproject.toml", "pytest.ini", "package.json"];

/// What a solution file list is in: Aider's `instructions_addendum`.
fn addendum(files: &str) -> String {
    format!(
        "\n####\n\nUse the above instructions to modify the supplied files: {files}\nDon't change \
         the names of existing functions or classes, as they may be referenced from other code \
         like unit tests, etc.\nOnly use standard libraries, don't suggest installing any \
         packages.\n"
    )
}

/// Aider's `test_failures`, after the test output.
fn failures(files: &str) -> String {
    format!(
        "\n####\n\nSee the testing errors above.\nThe tests are correct, don't try and change \
         them.\nFix the code in {files} to resolve the errors.\n"
    )
}

/// Each language's test command, Aider's (`TEST_COMMANDS`, cpp-test.sh, npm-test.sh) run in
/// the exercise's folder.
fn test_cmd(lang: &str) -> &'static str {
    match lang {
        // Aider's image has pytest on PATH; Homebrew's Python has it as a module.
        "python" => "python3 -m pytest",
        "rust" => "cargo test -- --include-ignored",
        "go" => "go test ./...",
        "javascript" => "npm run test",
        "cpp" => {
            "mkdir -p build && cd build && cmake -DEXERCISM_RUN_ALL_TESTS=1 -G 'Unix Makefiles' .. \
             && make"
        }
        "java" => "./gradlew test",
        _ => panic!("no test command for {lang}"),
    }
}

/// An exercise's files, from its `.meta/config.json`.
#[derive(Debug)]
struct Files {
    /// The files the model is told to change: Aider's solution set, less what it ignores.
    solution: Vec<String>,
    /// What the model's folder lacks besides the dot folders, and the grader puts back: the
    /// tests, a helper of theirs (Go's cases_test.go), and `MANIFESTS`.
    hidden: Vec<String>,
}

/// The files `config` (`.meta/config.json`'s text) names.
fn files(config: &str) -> Files {
    let config: Value = serde_json::from_str(config).expect("a .meta/config.json");
    let list = |key: &str| -> Vec<String> {
        (config["files"][key].as_array().into_iter().flatten())
            .filter_map(|f| f.as_str().map(String::from))
            .collect()
    };
    let (test, example) = (list("test"), list("example"));
    let ignored = |f: &String| {
        ["Cargo.toml", "CMakeLists.txt"].contains(&f.as_str())
            || f.starts_with(".meta/")
            || f.starts_with(".docs/")
            || test.contains(f)
            || example.contains(f)
    };
    let solution = list("solution")
        .into_iter()
        .filter(|f| !ignored(f))
        .collect();
    // An editor file is shown to the student; one of the tests' (Go's cases) would show the cases.
    let tests_helper = |f: &String| f.to_lowercase().contains("test");
    let editor = list("editor").into_iter().filter(tests_helper);
    let hidden = (test.iter().cloned().chain(editor))
        .chain(MANIFESTS.map(String::from))
        .collect();
    Files { solution, hidden }
}

/// The last whole lines of `text` within `cap` bytes, after a line that says how many were cut.
fn tail(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.into();
    }
    let mut start = text.ceil_char_boundary(text.len() - cap);
    if let Some(nl) = text[start..]
        .find('\n')
        .filter(|nl| start + nl + 1 < text.len())
    {
        start += nl + 1;
    }
    let cut = text[..start].lines().count();
    format!("... {cut} lines cut\n{}", &text[start..])
}

/// Test output as Aider shows it (`cleanup_test_output`): no timings, which would vary the
/// prompt, and the folder `dir` as the exercise's `name`, by its real path too: macOS's tests
/// print /private/var for the /var a temporary folder is named by.
fn cleanup(out: &str, dir: &Path, name: &str) -> String {
    let timing = Regex::new(r"\bin \d+\.\d+s\b").expect("a valid regex");
    let mut out = timing.replace_all(out, "").into_owned();
    // The real path first: it ends with the path given, which would leave its prefix behind.
    for d in [dir.canonicalize().unwrap_or_default().as_path(), dir] {
        if !d.as_os_str().is_empty() {
            out = out.replace(d.to_str().expect("a UTF-8 path"), name);
        }
    }
    out
}

/// The exercises POLYGLOT_OUT's `text` holds, by language and name; a line torn by a crash is
/// not one.
fn done(text: &str) -> HashSet<(String, String)> {
    (text.lines())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| Some((v["lang"].as_str()?.into(), v["name"].as_str()?.into())))
        .collect()
}

/// Per language and in total: exercises, pass rate after each try, and median seconds.
fn summary(lines: &[Value]) -> String {
    let mut by: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for l in lines {
        let lang = l["lang"].as_str().unwrap_or("?");
        by.entry(lang.into()).or_default().push(l);
    }
    by.insert("total".into(), lines.iter().collect());
    let mut out = format!(
        "{:<10} {:>5} {:>9} {:>9} {:>9}\n",
        "lang", "n", "pass1", "pass2", "median_s"
    );
    for (lang, ls) in by {
        let rate = |key: &str| {
            let n = ls.iter().filter(|l| l[key] == true).count();
            100.0 * n as f64 / ls.len() as f64
        };
        let mut secs: Vec<f64> = ls.iter().filter_map(|l| l["wall_s"].as_f64()).collect();
        secs.sort_by(f64::total_cmp);
        let m = secs.len() / 2;
        let median = match secs.len() {
            0 => 0.0,
            n if n % 2 == 0 => (secs[m - 1] + secs[m]) / 2.0,
            _ => secs[m],
        };
        out += &format!(
            "{lang:<10} {:>5} {:>8.1}% {:>8.1}% {median:>9.0}\n",
            ls.len(),
            rate("pass1"),
            rate("pass2")
        );
    }
    out
}

/// Copies the files under `from` to `to` that `keep` takes, by their path relative to `from`.
fn copy(from: &Path, to: &Path, keep: &dyn Fn(&Path) -> bool) -> io::Result<()> {
    fn walk(root: &Path, dir: &Path, to: &Path, keep: &dyn Fn(&Path) -> bool) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let rel = path.strip_prefix(root).expect("under the root");
            if !keep(rel) {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, to, keep)?;
            } else {
                fs::create_dir_all(to.join(rel).parent().expect("a file's folder"))?;
                fs::copy(&path, to.join(rel))?;
            }
        }
        Ok(())
    }
    fs::create_dir_all(to)?;
    walk(from, from, to, keep)
}

/// The folder the model works in: the exercise at `dir` without its dot folders (instructions,
/// examples, approaches) and its hidden files.
fn model_folder(dir: &Path, files: &Files, to: &Path) -> io::Result<()> {
    let keep = |rel: &Path| {
        let dot = rel.components().count() == 1
            && rel.to_string_lossy().starts_with('.')
            && dir.join(rel).is_dir();
        !dot && !files.hidden.iter().any(|h| rel == Path::new(h))
    };
    copy(dir, to, &keep)
}

/// Runs `cmd` in `dir` through `sh`, its process group killed past `timeout`: whether it exited
/// 0, None when it timed out, and its stdout and stderr together.
fn run(cmd: &str, dir: &Path, timeout: Duration) -> io::Result<(Option<bool>, String)> {
    let mut log = tempfile::tempfile()?;
    let mut child = Command::new("sh")
        .args(["-c", cmd])
        .current_dir(dir)
        // The cargo running this holds its target folder's lock.
        .env_remove("CARGO_TARGET_DIR")
        // As in Aider's image: a backtrace would fill the second try's output.
        .env_remove("RUST_BACKTRACE")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log.try_clone()?)
        .process_group(0)
        .spawn()?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status.success());
        }
        if Instant::now() >= deadline {
            break None;
        }
        thread::sleep(Duration::from_millis(100));
    };
    if status.is_none() {
        // std has no killpg; kill(1) reaches the group, as the sandbox's `kill_group` does.
        Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stderr(Stdio::null())
            .status()?;
        child.wait()?;
    }
    let mut out = Vec::new();
    log.rewind()?;
    log.read_to_end(&mut out)?;
    Ok((status, String::from_utf8_lossy(&out).into_owned()))
}

/// Grades the exercise at `ex` (of `lang`) as solved in `solved`: a copy of it in a folder named
/// as the exercise, which C++'s CMakeLists.txt takes its name from, its hidden files back from
/// `ex`, Aider's per-language fixes, then its test command. Whether it passed, and the output
/// the second try would see.
fn grade(
    lang: &str,
    ex: &Path,
    files: &Files,
    solved: &Path,
    npm: Option<&Path>,
) -> io::Result<(bool, String)> {
    let name = ex.file_name().expect("a name").to_string_lossy();
    let tmp = tempfile::tempdir()?;
    let dir = &tmp.path().join(&*name);
    copy(solved, dir, &|rel| rel != Path::new(".git"))?;
    let disabled = Regex::new(r"@Disabled\([^)]*\)\s*\n").expect("a valid regex");
    let xtest = Regex::new(r"\bxtest\(").expect("a valid regex");
    for f in files.hidden.iter().filter(|f| ex.join(f).is_file()) {
        let to = dir.join(f);
        fs::create_dir_all(to.parent().expect("a file's folder"))?;
        let text = fs::read_to_string(ex.join(f))?;
        // Aider runs Java's disabled tests too, and JavaScript's skipped ones (npm-test.sh).
        let text = match f {
            f if f.ends_with(".java") => disabled.replace_all(&text, ""),
            f if f.ends_with(".spec.js") => xtest.replace_all(&text, "test("),
            _ => text.as_str().into(),
        };
        fs::write(to, text.as_bytes())?;
    }
    if lang == "javascript" {
        let npm = npm.expect("POLYGLOT_NPM=<dir> for javascript");
        for f in ["node_modules", "package-lock.json"] {
            if !dir.join(f).exists() {
                symlink(npm.join(f), dir.join(f))?;
            }
        }
    }
    Ok(match run(test_cmd(lang), dir, TEST_TIMEOUT)? {
        (None, _) => (false, "Tests timed out!".into()),
        (Some(passed), out) => (passed, cleanup(&out, dir, &name)),
    })
}

/// Prints a call the model makes, as code_eval does, so a night's log shows what each try did.
fn print_call(e: AgentEvent) {
    if let AgentEvent::ToolCall { name, args } = e {
        println!("  -> {name} {args}");
    }
}

/// One exercise's two tries, and its JSON line; its trace goes to `traces`, made afresh.
fn exercise(
    larch: &Larch,
    code: &Code,
    lang: &str,
    ex: &Path,
    npm: Option<&Path>,
    traces: &Path,
) -> Value {
    let started = Instant::now();
    let name = ex
        .file_name()
        .expect("a name")
        .to_string_lossy()
        .into_owned();
    let (mut passed, mut tokens, mut stopped) = (Vec::new(), 0, String::new());
    let tmp = tempfile::tempdir().unwrap();
    // A run that stopped mid-exercise left its trace and worktree: this run starts it again.
    let _ = fs::remove_dir_all(traces);
    fs::create_dir_all(traces).unwrap();
    let mut tries = || -> Result<(), String> {
        let config = fs::read_to_string(ex.join(".meta/config.json")).map_err(|e| e.to_string())?;
        let files = files(&config);
        let repo = tmp.path().join(&name);
        model_folder(ex, &files, &repo).map_err(|e| format!("copy: {e}"))?;
        common::commit(&repo);
        // Aider's file_list is the base names; the paths say the same and find the files.
        let list = files.solution.join(" ");
        let mut instructions = String::new();
        for doc in [
            "introduction.md",
            "instructions.md",
            "instructions.append.md",
        ] {
            instructions += &fs::read_to_string(ex.join(".docs").join(doc)).unwrap_or_default();
        }
        let first = instructions + &addendum(&list);
        let mut request = first.clone();
        loop {
            let overrides = Overrides {
                test_cmd: None,
                allowed_paths: None,
                check_cmds: vec![],
                budget: BUDGET,
            };
            let task = CodeTask::new(&request, &repo, overrides).map_err(|e| e.to_string())?;
            let data = traces.join("data");
            let mut outcome = code
                .run_with(larch, &task, &data, &mut print_call, &mut |_| Decision::No)
                .map_err(|e| format!("code mode: {e}"))?;
            tokens += outcome.tokens;
            stopped = outcome.packet.stopped.take().unwrap_or_default();
            // The change goes into the repository, so a second try starts from it; the grade is
            // of what it merged, as a user gets it, not of what else the worktree holds.
            let merged = decide(&mut outcome, &mut "approve\n".as_bytes(), &mut io::sink());
            let graded = grade(lang, ex, &files, &repo, npm).map_err(|e| format!("grade: {e}"));
            let (pass, out) = graded?;
            passed.push(pass);
            let n = passed.len();
            let kept = fs::write(traces.join(format!("try-{n}.diff")), &outcome.packet.diff)
                .and_then(|_| fs::write(traces.join(format!("try-{n}.tests.txt")), &out));
            kept.map_err(|e| format!("trace: {e}"))?;
            if pass || passed.len() == 2 {
                break;
            }
            if !merged.map_err(|e| e.to_string())? {
                return Err("the try's change could not be merged".into());
            }
            // Aider's second message follows the first in one chat; code mode starts afresh,
            // so its request carries the first one too.
            request = format!("{first}\n{}{}", tail(&out, TAIL_BYTES), failures(&list));
        }
        Ok(())
    };
    let error = tries().err();
    json!({
        "lang": lang,
        "name": name,
        "pass1": passed.first() == Some(&true),
        "pass2": passed.contains(&true),
        "wall_s": started.elapsed().as_secs_f64().round(),
        "tokens": tokens,
        "stopped": stopped,
        "error": error,
    })
}

/// The exercise folders of each language POLYGLOT_LANGS names (all by default), the first
/// POLYGLOT_LIMIT of each by name.
fn exercises() -> Vec<(String, PathBuf)> {
    let root = PathBuf::from(std::env::var("POLYGLOT_DIR").expect("POLYGLOT_DIR=<checkout>"));
    let langs = std::env::var("POLYGLOT_LANGS").unwrap_or_else(|_| LANGS.join(","));
    let limit = std::env::var("POLYGLOT_LIMIT")
        .map_or(usize::MAX, |n| n.parse().expect("POLYGLOT_LIMIT=<n>"));
    let mut all = Vec::new();
    for lang in langs.split(',') {
        let practice = root.join(lang).join("exercises/practice");
        let mut dirs: Vec<PathBuf> = (fs::read_dir(&practice)
            .unwrap_or_else(|e| panic!("{practice:?}: {e}")))
        .map(|e| e.unwrap().path())
        .collect();
        dirs.sort();
        all.extend(dirs.into_iter().take(limit).map(|d| (lang.to_string(), d)));
    }
    all
}

#[test]
#[ignore = "needs a model and the benchmark: LARCH_MODEL=<gguf> POLYGLOT_DIR=<checkout> POLYGLOT_OUT=<file.jsonl>"]
fn polyglot() {
    let model = std::env::var("LARCH_MODEL").expect("LARCH_MODEL=<gguf>");
    let out = PathBuf::from(std::env::var("POLYGLOT_OUT").expect("POLYGLOT_OUT=<file.jsonl>"));
    let npm = std::env::var_os("POLYGLOT_NPM").map(PathBuf::from);
    let text = fs::read_to_string(&out).unwrap_or_default();
    let seen = done(&text);
    let todo: Vec<_> = (exercises().into_iter())
        .filter(|(lang, d)| {
            let name = d.file_name().unwrap().to_string_lossy().into_owned();
            !seen.contains(&(lang.clone(), name))
        })
        .collect();
    let js = todo.iter().any(|(lang, _)| lang == "javascript");
    assert!(
        !js || npm.is_some(),
        "POLYGLOT_NPM=<dir with node_modules> for javascript"
    );
    println!(
        "{} exercises to run, {} done before",
        todo.len(),
        seen.len()
    );
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .unwrap();
    if !text.is_empty() && !text.ends_with('\n') {
        file.write_all(b"\n").unwrap();
    }
    if !todo.is_empty() {
        let traces = PathBuf::from(format!("{}.traces", out.display()));
        let (larch, code) = common::real(&model, 0);
        for (lang, dir) in &todo {
            println!("{lang} {}", dir.display());
            let name = dir.file_name().unwrap();
            let traces = traces.join(lang).join(name);
            let line = exercise(&larch, &code, lang, dir, npm.as_deref(), &traces);
            println!("{line}");
            file.write_all(format!("{line}\n").as_bytes()).unwrap();
        }
    }
    let text = fs::read_to_string(&out).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    println!("{}", summary(&lines));
}

/// Where an example file goes to stand as the solution: Java's reference tree under src/main,
/// any other the solution file of its extension (example.rs is src/lib.rs).
fn example_to(example: &str, solution: &[String]) -> String {
    if let Some(rest) = example.strip_prefix(".meta/src/reference/") {
        return format!("src/main/{rest}");
    }
    let ext = Path::new(example).extension();
    (solution.iter().find(|s| Path::new(s).extension() == ext))
        .unwrap_or_else(|| panic!("no solution file for {example}"))
        .clone()
}

#[test]
#[ignore = "needs the benchmark and its toolchains: POLYGLOT_DIR=<checkout>"]
fn grader_passes_examples_and_fails_stubs() {
    let npm = std::env::var_os("POLYGLOT_NPM").map(PathBuf::from);
    let mut wrong = Vec::new();
    for (lang, ex) in exercises() {
        let files = files(&fs::read_to_string(ex.join(".meta/config.json")).unwrap());
        let tmp = tempfile::tempdir().unwrap();
        let (stub, solved) = (tmp.path().join("stub"), tmp.path().join("solved"));
        model_folder(&ex, &files, &stub).unwrap();
        model_folder(&ex, &files, &solved).unwrap();
        let config: Value =
            serde_json::from_str(&fs::read_to_string(ex.join(".meta/config.json")).unwrap())
                .unwrap();
        for e in config["files"]["example"].as_array().unwrap() {
            let e = e.as_str().unwrap();
            fs::copy(ex.join(e), solved.join(example_to(e, &files.solution))).unwrap();
        }
        let (on_stub, _) = grade(&lang, &ex, &files, &stub, npm.as_deref()).unwrap();
        let (on_example, out) = grade(&lang, &ex, &files, &solved, npm.as_deref()).unwrap();
        let name = ex.file_name().unwrap().to_string_lossy().into_owned();
        println!("{lang} {name}: stub {on_stub}, example {on_example}");
        if on_stub || !on_example {
            wrong.push(format!(
                "{lang}/{name}: stub {on_stub}, example {on_example}\n{}",
                tail(&out, 2000)
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}

/// The whole adapter on the fake model and a made-up Rust exercise: the model's folder lacks the
/// tests and Cargo.toml, its first try's wrong change fails the tests, the second gets their
/// output in Aider's words, fixes it and passes.
#[test]
fn a_failed_first_try_gets_the_test_output_and_the_second_passes() {
    let fake = common::Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let ex = tmp.path().join("rust/exercises/practice/adder");
    for (path, text) in [
        (
            ".meta/config.json",
            r#"{"files": {"solution": ["src/lib.rs", "Cargo.toml"], "test": ["tests/adder.rs"],
                "example": [".meta/example.rs"]}}"#,
        ),
        (
            ".meta/example.rs",
            "pub fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n",
        ),
        (".docs/instructions.md", "Add two numbers.\n"),
        (
            "Cargo.toml",
            "[package]\nname = \"adder\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        ),
        (
            "src/lib.rs",
            "pub fn add(a: u32, b: u32) -> u32 {\n    todo!()\n}\n",
        ),
        (
            "tests/adder.rs",
            "#[test]\nfn hidden_case() {\n    assert_eq!(adder::add(3, 4), 7);\n}\n",
        ),
    ] {
        fs::create_dir_all(ex.join(path).parent().unwrap()).unwrap();
        fs::write(ex.join(path), text).unwrap();
    }
    // A done is reviewed once in a folder without tests, then the reviewer reads the next, which
    // ends the try.
    for (old, new) in [("todo!()", "a - b"), ("a - b", "a + b")] {
        let edit = json!({"path": "src/lib.rs", "old_text": old, "new_text": new});
        common::qwen3_call(&fake, "edit", edit, false);
        fake.say("Done.");
        fake.say("Done.");
        fake.say("ok");
    }
    let traces = tmp.path().join("traces");
    let line = exercise(
        &fake.larch,
        &common::qwen3_code(),
        "rust",
        &ex,
        None,
        &traces,
    );
    assert_eq!(line["error"], Value::Null, "{line}");
    assert_eq!(
        (&line["pass1"], &line["pass2"]),
        (&json!(false), &json!(true)),
        "{line}"
    );
    let fed = fake.fed();
    assert!(fed.contains("Add two numbers."), "{fed}");
    assert!(
        fed.contains("modify the supplied files: src/lib.rs"),
        "{fed}"
    );
    assert!(fed.contains("attempt to subtract with overflow"), "{fed}");
    assert!(
        fed.contains("Fix the code in src/lib.rs to resolve the errors."),
        "{fed}"
    );
    assert!(!fed.contains("adder::add(3, 4)"), "{fed}");
    assert!(!fed.contains("Cargo.toml"), "{fed}");
    let first = fs::read_to_string(traces.join("try-1.diff")).unwrap();
    assert!(first.contains("+    a - b"), "{first}");
    assert!(traces.join("data/agent/trace").is_dir());
}

/// A C++ exercise replayed as the real one went: the model fixes the solution, builds it with the
/// CMakeLists that takes the target's name from the folder, and says done. Its build finds the
/// exercise's name (the worktree is named as the exercise), the passing build check makes the done final (no review
/// sends a right answer back), and the build tree it made stays out of what is graded (a cache
/// naming the worktree breaks cmake anywhere else).
#[test]
fn a_cpp_try_that_builds_in_its_folder_is_done_and_graded_on_its_sources() {
    let fake = common::Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let ex = tmp.path().join("cpp/exercises/practice/adder");
    let cmake = "get_filename_component(exercise ${CMAKE_CURRENT_SOURCE_DIR} NAME)\n\
                 cmake_minimum_required(VERSION 3.10)\nproject(${exercise} CXX)\n\
                 string(REPLACE \"-\" \"_\" file ${exercise})\n\
                 if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${file}_test.cpp)\n\
                 add_executable(${exercise} ${file}.cpp ${file}_test.cpp)\n\
                 add_custom_command(TARGET ${exercise} POST_BUILD COMMAND ${exercise})\n\
                 else()\nadd_library(${exercise} ${file}.cpp)\nendif()\n";
    for (path, text) in [
        (
            ".meta/config.json",
            r#"{"files": {"solution": ["adder.cpp", "adder.h"], "test": ["adder_test.cpp"],
                "example": [".meta/example.cpp"]}}"#,
        ),
        (".docs/instructions.md", "Add two numbers.\n"),
        ("CMakeLists.txt", cmake),
        ("adder.h", "int add(int a, int b);\n"),
        (
            "adder.cpp",
            "#include \"adder.h\"\nint add(int a, int b) { return 0; }\n",
        ),
        (
            "adder_test.cpp",
            "#include \"adder.h\"\nint main() { return add(3, 4) == 7 ? 0 : 1; }\n",
        ),
    ] {
        fs::create_dir_all(ex.join(path).parent().unwrap()).unwrap();
        fs::write(ex.join(path), text).unwrap();
    }
    let edit = json!({"path": "adder.cpp", "old_text": "return 0;", "new_text": "return a + b;"});
    common::qwen3_call(&fake, "edit", edit, false);
    // The configure step writes the cache wherever the build then fails (a sandbox may refuse make).
    let build = json!({"command": "basename \"$PWD\"; cmake -B build -S . >/dev/null 2>&1; ls build/CMakeCache.txt"});
    common::qwen3_call(&fake, "bash", build, false);
    fake.say("Done.");
    let traces = tmp.path().join("traces");
    let line = exercise(
        &fake.larch,
        &common::qwen3_code(),
        "cpp",
        &ex,
        None,
        &traces,
    );
    let fed = fake.fed();
    if fed.contains("the sandbox is unavailable") {
        println!("skipped: no sandbox");
        return;
    }
    assert_eq!(line["error"], Value::Null, "{line}");
    assert!(
        fed.contains("exit code 0\nadder\nbuild/CMakeCache.txt"),
        "{fed}"
    );
    assert!(
        !fed.contains("check your work against the request"),
        "{fed}"
    );
    assert_eq!(line["pass1"], json!(true), "{line}");
    let diff = fs::read_to_string(traces.join("try-1.diff")).unwrap();
    assert!(!diff.contains("CMakeCache"), "{diff}");
}

#[test]
fn the_config_names_the_files_aider_shows_and_the_ones_the_model_never_sees() {
    let rust = r#"{"files": {"solution": ["src/lib.rs", "Cargo.toml"], "test": ["tests/accumulate.rs"],
        "example": [".meta/example.rs"]}}"#;
    let f = files(rust);
    assert_eq!(f.solution, ["src/lib.rs"]);
    assert_eq!(f.hidden[..2], ["tests/accumulate.rs", "Cargo.toml"]);
    assert!(f.hidden.contains(&"package.json".into()));
    let go = r#"{"files": {"solution": ["forth.go"], "test": ["forth_test.go"],
        "example": [".meta/example.go"], "editor": ["cases_test.go"], "invalidator": ["go.mod"]}}"#;
    let f = files(go);
    assert_eq!(f.solution, ["forth.go"]);
    assert_eq!(f.hidden[..2], ["forth_test.go", "cases_test.go"]);
    let java = r#"{"files": {"solution": ["src/main/java/Hangman.java"],
        "test": ["src/test/java/HangmanTest.java"], "example": [],
        "editor": ["src/main/java/Part.java"]}}"#;
    let f = files(java);
    assert!(!f.hidden.iter().any(|h| h.ends_with("Part.java")));
}

#[test]
fn tail_keeps_the_last_whole_lines_under_the_cap() {
    assert_eq!(tail("a\nb\n", 100), "a\nb\n");
    let text: String = (0..100).map(|i| format!("line {i}\n")).collect();
    let cut = tail(&text, 40);
    assert!(cut.starts_with("... "), "{cut}");
    assert!(cut.ends_with("line 99\n"), "{cut}");
    let kept = cut.split_once('\n').unwrap().1;
    assert!(kept.len() <= 40 && kept.starts_with("line "), "{cut}");
    // A cut inside a multi-byte character moves to the next boundary.
    assert!(tail(&"é".repeat(50), 9).ends_with("éééé"));
}

#[test]
fn cleanup_drops_timings_and_the_folder_path_as_aider_does() {
    let dir = Path::new("/tmp/x/wt-1");
    let out = "FAILED /tmp/x/wt-1/a_test.py::t - 1 failed in 0.12s\nok";
    assert_eq!(
        cleanup(out, dir, "accumulate"),
        "FAILED accumulate/a_test.py::t - 1 failed \nok"
    );
    // A folder through a link, as macOS's /var is, printed by its real path or by the link's.
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    fs::create_dir_all(real.join("wt-2")).unwrap();
    symlink(&real, tmp.path().join("link")).unwrap();
    let dir = tmp.path().join("link/wt-2");
    let out = format!(
        "{}/a.go:3 and {}/b.go:4",
        real.join("wt-2").canonicalize().unwrap().display(),
        dir.display()
    );
    assert_eq!(
        cleanup(&out, &dir, "ledger"),
        "ledger/a.go:3 and ledger/b.go:4"
    );
}

#[test]
fn resume_reads_the_finished_exercises_and_skips_a_torn_line() {
    let text = format!(
        "{}\n{}\n{{\"lang\":\"go\",\"na",
        json!({"lang": "rust", "name": "accumulate", "pass1": true}),
        json!({"lang": "go", "name": "forth", "pass1": false})
    );
    let seen = done(&text);
    assert_eq!(seen.len(), 2);
    assert!(seen.contains(&("go".into(), "forth".into())));
}

#[test]
fn summary_counts_passes_per_language_and_the_median_time() {
    let lines = [
        json!({"lang": "go", "pass1": true, "pass2": true, "wall_s": 10.0}),
        json!({"lang": "go", "pass1": false, "pass2": true, "wall_s": 30.0}),
        json!({"lang": "go", "pass1": false, "pass2": false, "wall_s": 20.0}),
        json!({"lang": "rust", "pass1": false, "pass2": false, "wall_s": 5.0}),
    ];
    let s = summary(&lines);
    assert!(
        s.contains("go             3     33.3%     66.7%        20"),
        "{s}"
    );
    assert!(
        s.contains("total          4     25.0%     50.0%        15"),
        "{s}"
    );
}
