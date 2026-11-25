//! One coding task: the user's repository and its base, the iris branch and worktree the agent
//! writes in, the commands that judge it, and the packet a human approves before the merge. Or,
//! in place (`CodeTask::in_folder`), any folder of the user's, which a checkpoint commit saves
//! before the agent edits it and `undo` puts back.

use std::{
    env,
    fmt::Write as _,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    tool::{Tool, ToolError, plain},
    tools::{code, git},
};

/// The agent's commits carry its own name, never the user's, and are never signed: the
/// repository's signing program is not the agent's to run.
pub(crate) const IRIS: [&str; 6] = [
    "-c",
    "user.name=Iris",
    "-c",
    "user.email=iris@localhost",
    "-c",
    "commit.gpgsign=false",
];

/// A diff as the packet shows it: renames as a delete and an add, so the scope sees both paths;
/// every file as text, so an attribute the agent wrote (`-diff`) cannot hide a change from the
/// human approving it; and a repository the agent made in the worktree as the commit it adds,
/// which `NO_NESTED` would hide while the merge still brings it in. Between two commits git
/// starts nothing inside it.
const DIFF: [&str; 6] = [
    "diff",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--text",
    "--ignore-submodules=none",
];

/// Before every git run on a worktree: git never starts in a repository the agent made inside it,
/// as `status` would to see whether it changed, and so never runs what its config names
/// (`core.fsmonitor`). The packet's diffs show it all the same (`DIFF`).
const NO_NESTED: [&str; 2] = ["-c", "diff.ignoreSubmodules=all"];

/// The most files git does not track yet, and their bytes, a checkpoint takes: each is read and
/// stored, so a folder past them (a home folder, a download pile) is refused before any is.
// ponytail: guesses; raise when a real project hits them
const CHECKPOINT_FILES: usize = 20_000;
const CHECKPOINT_BYTES: u64 = 1 << 30;

/// The first file found, holding the text when one is given, names the test command: a manifest,
/// Django's runner, pytest's config, a Makefile's test target. `{tests}` takes the test files near
/// the changed files, `{tests app}` the folders under tests/ they are in (`near_tests`).
const RUNNERS: [(&str, &str, &str); 9] = [
    ("Cargo.toml", "", "cargo test"),
    (
        "tests/runtests.py",
        "",
        "python3 tests/runtests.py --parallel 1 {tests app}",
    ),
    ("pyproject.toml", "", "pytest {tests}"),
    ("pytest.ini", "", "pytest {tests}"),
    ("setup.cfg", "[tool:pytest]", "pytest {tests}"),
    ("tox.ini", "[pytest]", "pytest {tests}"),
    ("conftest.py", "", "pytest {tests}"),
    ("package.json", "", "npm test"),
    ("Makefile", "\ntest:", "make test"),
];

/// The most test files a suite has that runs whole when no test is near the change
/// (`near_tests`): past it, a whole run outlasts the command timeout.
// ponytail: 30 by judgment (Django has ~1700, Sphinx ~120); time a whole run when a mid-size
// suite shows it fits
const SMALL_SUITE: usize = 30;

/// Python's build check of the changed modules the base has: an import of each from the
/// folder's top, which runs what a module does as it loads (an enum member that breaks its
/// class), not only its syntax, and which the base's import of it tells from what the change
/// did. Each module is imported on its own and a failing one named (`FAILED_IMPORT`), so one
/// the base fails too hides no other, a script's exit at its top (argparse) too, which reads no
/// file list as its arguments. Bytecode goes to the private tmp, or nowhere where Python
/// predates that.
pub(crate) const IMPORT: &str = "PYTHONPYCACHEPREFIX=\"$TMPDIR\" python3 -B -c \"import importlib, \
sys, traceback\nbad = 0\nfiles = sys.argv[1:]\nfor f in files:\n sys.argv[1:] = []\n \
m = f[:-3].replace('/', '.')\n try: importlib.import_module(m[:-9] if m.endswith('.__init__') \
else m)\n except KeyboardInterrupt: raise\n except BaseException: bad = 1; traceback.print_exc(); \
print('import failed: ' + f, file=sys.stderr)\nsys.exit(bad)\" {*.py old}";

/// The line `IMPORT` prints for each module whose import failed, before its path.
pub(crate) const FAILED_IMPORT: &str = "import failed: ";

/// Python's build check of the new files: a compile, as a new script's import would run it
/// (`input()`, `sys.argv[1]`) with no base to tell that from a fault.
// ponytail: Python 3.7 and older lack PYTHONPYCACHEPREFIX, so py_compile writes __pycache__
// beside the new files; compile() them in memory when a user on such a Python shows up
pub(crate) const COMPILE: &str = "PYTHONPYCACHEPREFIX=\"$TMPDIR\" python3 -m py_compile {*.py new}";

/// The build checks a folder without tests gets (`checks`): a file that says the language,
/// matched by its path's end, and the command. `{*.ext}` takes the changed files of that
/// extension, `{*.ext dir}` their folders (`expand`), so a check reads only what the run
/// changed, cheap in a big repository too. Nothing they write lands in the folder: Python's
/// bytecode and javac's classes go to the private tmp, and javac runs no annotation processor.
const CHECKS: [(&str, &str); 6] = [
    (".py", IMPORT),
    (".py", COMPILE),
    (
        ".js",
        "for f in {*.js}; do node --check \"$f\" || exit 1; done",
    ),
    ("go.mod", "go vet {*.go dir}"),
    (
        ".java",
        "javac -d \"$TMPDIR\" -proc:none -implicit:none -sourcepath {root} {*.java}",
    ),
    // ponytail: .cpp alone, with the compiler's own include paths; add .cc and the build's flags
    // when a project with them shows the check dropped
    (".cpp", "c++ -std=c++17 -fsyntax-only {*.cpp}"),
];

/// The most Java files compiled without a build file that names how, and the most C++ files
/// compiled at all (`checks`): past it, one file's dependencies are a guess.
const LOOSE: usize = 20;

/// The checks `CHECKS` gives a folder of `files`, relative to it: Java only with a build file or
/// up to `LOOSE` sources, C++ only up to `LOOSE` sources, its `{root}` the Maven or Gradle source folder when there is one.
// ponytail: none on Windows: cmd.exe neither takes these quotes nor expands $TMPDIR; add when a
// Windows run needs build checks
pub fn checks(files: &[String]) -> Vec<String> {
    if cfg!(windows) {
        return Vec::new();
    }
    let has =
        |end: &str| (files.iter()).any(|f| f == end || (end.starts_with('.') && f.ends_with(end)));
    let built = ["build.gradle", "build.gradle.kts", "pom.xml"]
        .iter()
        .any(|b| has(b));
    let count = |ext: &str| files.iter().filter(|f| f.ends_with(ext)).count();
    let root = match files.iter().any(|f| f.starts_with("src/main/java/")) {
        true => "src/main/java",
        false => ".",
    };
    (CHECKS.iter())
        .filter(|(end, _)| has(end))
        .filter(|(end, _)| *end != ".java" || built || count(".java") <= LOOSE)
        .filter(|(end, _)| *end != ".cpp" || count(".cpp") <= LOOSE)
        .map(|(_, cmd)| cmd.replace("{root}", root))
        .collect()
}

/// `cmd` (`CHECKS`) for the `changed` files, relative to the folder, in one run; None when no
/// changed file is of its kind. `{*.ext old}` takes those of `old`, the ones the base has,
/// `{*.ext new}` the rest. A command without a `{*.` placeholder is as it is.
pub(crate) fn expand(cmd: &str, changed: &[String], old: &[String]) -> Option<String> {
    let Some((head, rest)) = cmd.split_once("{*") else {
        return Some(cmd.into());
    };
    let (spec, tail) = rest.split_once('}')?;
    let (ext, which) = spec.split_once(' ').unwrap_or((spec, ""));
    let dir = which == "dir";
    let kept = |f: &&String| match which {
        "old" => old.contains(f),
        "new" => !old.contains(f),
        _ => true,
    };
    let mut args: Vec<String> = (changed.iter())
        .filter(|f| f.ends_with(ext))
        .filter(kept)
        .map(|f| match dir {
            true => match Path::new(f).parent().and_then(Path::to_str) {
                Some("") | None => ".".into(),
                Some(d) => format!("./{d}"),
            },
            false => f.clone(),
        })
        .collect();
    args.sort();
    args.dedup();
    let quoted: Vec<String> = (args.iter())
        .map(|a| format!("'{}'", a.replace('\'', r"'\''")))
        .collect();
    (!quoted.is_empty()).then(|| format!("{head}{}{tail}", quoted.join(" ")))
}

/// The name a base checkout's folder in the tmp starts with (`CodeTask::at_base`).
const BASE_CHECKOUT: &str = "larch-base-";

/// How old a base checkout another run left is before `at_base` removes it: well past any
/// base run, which the command timeout ends.
const STALE: std::time::Duration = std::time::Duration::from_secs(3600);

/// A base checkout in the tmp, removed when dropped, a panic too, with any folder its run made
/// read-only, which would keep `TempDir` from removing it.
struct Checkout(tempfile::TempDir);

impl Drop for Checkout {
    fn drop(&mut self) {
        // What still fails is left for the next run's sweep; a drop has no one to tell.
        if fs::remove_dir_all(self.0.path()).is_err() {
            writable(self.0.path());
            let _ = fs::remove_dir_all(self.0.path());
        }
    }
}

/// Makes `dir` and every folder in it writable, so it can be removed.
// ponytail: unix only; Windows' read-only files keep a checkout until its sweep fails too, add
// it when a Windows run shows leftovers
#[cfg_attr(not(unix), allow(unused_variables))]
fn writable(dir: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).is_ok() {
            for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    writable(&entry.path());
                }
            }
        }
    }
}

/// Removes the base checkouts in the tmp older than `STALE`, which a run killed mid-way left.
/// One it cannot remove (another user's) is left: it is not this run's.
fn sweep() {
    for entry in fs::read_dir(env::temp_dir())
        .into_iter()
        .flatten()
        .flatten()
    {
        let ours = entry
            .file_name()
            .to_string_lossy()
            .starts_with(BASE_CHECKOUT);
        let age = (entry.metadata().and_then(|m| m.modified()).ok()).and_then(|t| t.elapsed().ok());
        if ours && age.is_some_and(|age| age > STALE) {
            writable(&entry.path());
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

/// A copy of the index at `from` in `dir`, with its time: git takes an entry no older than
/// its index as maybe changed (racy git), and a copy's later time would let a file rewritten
/// at the same size in the index's second pass as unchanged. The time is read first: one read
/// after the copy could be a rewrite's, newer than the bytes copied.
fn copy_index(from: &Path, dir: &Path) -> std::io::Result<()> {
    let to = dir.join("index");
    let time = fs::metadata(from)?.modified()?;
    fs::copy(from, &to)?;
    fs::File::options()
        .write(true)
        .open(&to)?
        .set_modified(time)
}

/// A task's limits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Wall-clock seconds.
    pub seconds: u64,
    /// Tool calls.
    pub tool_calls: u32,
    /// Generated tokens.
    pub tokens: u64,
    /// Attempts, each in its own worktree.
    pub k: u32,
}

/// What the user set for a task; the rest is detected.
#[derive(Clone, Debug)]
pub struct Overrides {
    /// The test command, instead of the manifest's.
    pub test_cmd: Option<String>,
    /// The paths the change may touch, instead of the whole repository.
    pub allowed_paths: Option<Vec<String>>,
    /// Commands that must pass besides the tests.
    pub check_cmds: Vec<String>,
    /// The task's limits.
    pub budget: Budget,
}

/// A coding task on the user's repository.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeTask {
    /// What the user asked for.
    pub request: String,
    /// The user's checkout, canonical; written only by `apply`.
    pub repo: PathBuf,
    /// The checkout's HEAD when the task began; every diff is against it.
    pub base_sha: String,
    /// The current attempt's iris branch, while its worktree exists.
    pub branch: Option<String>,
    /// The current attempt's worktree.
    pub worktree: Option<PathBuf>,
    /// Repository paths the change may touch; "." is the whole repository, never `.git`.
    pub allowed_paths: Vec<String>,
    /// The command that runs the tests, when one was given or a manifest names it.
    pub test_cmd: Option<String>,
    /// Commands that must pass besides the tests.
    pub check_cmds: Vec<String>,
    /// The task's limits.
    pub budget: Budget,
    /// What sets this task's worktree and branch apart from another with the same request (a
    /// chat's id, when chats run tasks side by side); None for the CLI's one task at a time.
    #[serde(default)]
    pub tag: Option<String>,
    /// Files a chat drafted, by bare name, that a run in place writes after its checkpoint where
    /// the folder lacks them and its allowances cover them (`Code::run_until`).
    #[serde(skip)]
    pub drafts: Vec<(String, String)>,
    /// The git that keeps the checkpoints of a task the agent does in place, in `repo` itself;
    /// None for one in worktrees.
    #[serde(default)]
    pub folder: Option<Folder>,
    /// The tests on the base, which a whole-suite command's (no `{tests}`) failures are compared
    /// with (`test_until`); None until the run has them.
    #[serde(skip)]
    pub base_tests: Option<TestRun>,
    /// How the tests check the work (config.toml's `code_verify`).
    #[serde(skip)]
    pub verify: Verify,
    /// The run reads and runs but asks to change nothing (`router::Intent::Run`): it offers no
    /// tool that writes, and its first done ends it with the answer. Its commands, bash's and
    /// the tests', write only the folder's build output (`code::BUILD_DIRS`).
    #[serde(skip)]
    pub read_only: bool,
}

/// How code mode checks the work with the tests, for an A/B run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Verify {
    /// Failures are compared with the base's, and a done runs the tests and holds until they
    /// pass, with the build checks and the reviewer where they do not say.
    #[default]
    Full,
    /// run_tests reports the run as it is, and a done ends the attempt with the last run the
    /// model saw since its last change as the verdict.
    Simple,
}

/// The git around a folder the agent edits in place: the repository the folder is in, or a
/// private one under the data folder for a folder git does not track, so none is made in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Folder {
    /// The repository's git directory.
    pub git_dir: PathBuf,
    /// Where the repository's paths start: its top, or the folder itself for a private one.
    pub top: PathBuf,
    /// The user's own index, which snapshots start from and never write; None for a private
    /// repository, whose index is the agent's.
    pub index: Option<PathBuf>,
    /// The repository's HEAD when the task began, which an index starts from when the user has
    /// none; None without one.
    pub head: Option<String>,
    /// The branch that keeps the checkpoint, the folder as it was before the run (`base_sha`).
    pub branch: Option<String>,
    /// The folder as the run left it, committed on the checkpoint (`CodeTask::packet`).
    pub after: Option<String>,
    /// A bare repository under the data folder holding a copy of the checkpoint and of `after`,
    /// which every step falls back to once a command broke the folder's own (`rm -rf .git`), so
    /// undo still works; None for a private repository, already outside the folder.
    #[serde(default)]
    pub backup: Option<PathBuf>,
}

/// One run of the tests; `CodeTask::test` fills it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestRun {
    /// Tests that passed.
    pub passed: u32,
    /// Tests that failed.
    pub failed: u32,
    /// The failing tests' names.
    pub failing: Vec<String>,
    /// Failing tests the base fails too, by name (`code::against`): not in `failed`, not the
    /// change's, and never a pass.
    #[serde(default)]
    pub shared: Vec<String>,
    /// The end of the output.
    pub tail: String,
}

impl TestRun {
    /// Whether no test ran (none near the change): it says nothing, neither pass nor failure.
    pub fn none_ran(&self) -> bool {
        self.passed == 0 && self.failed == 0 && self.shared.is_empty()
    }
}

/// A changed path, and whether it lies inside the allowed paths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    /// The path, relative to the repository.
    pub path: String,
    /// Whether an allowed path holds it.
    pub inside: bool,
}

/// What a human reviews before an attempt is merged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Packet {
    /// The iris branch's diff against `base_sha`.
    pub diff: String,
    /// The tests on the base, once run.
    pub tests_before: Option<TestRun>,
    /// The tests on the branch, once run.
    pub tests_after: Option<TestRun>,
    /// Every changed path, in git's order.
    pub scope: Vec<Scope>,
    /// The commands the attempt ran, in order.
    pub commands_run: Vec<String>,
    /// The pull request's title, also the commit's message.
    pub pr_title: String,
    /// The pull request's body.
    pub pr_body: String,
    /// Why the attempt stopped, when code mode's loop made it: the model said it was done and
    /// the tests agreed, or the budget that ran out.
    pub stopped: Option<String>,
    /// How sure the run is of the change, and what is left.
    #[serde(default)]
    pub verdict: Verdict,
}

/// How sure a packet is of its change: what checked it, and what that left.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// The tests ran after the change and none failed.
    Verified,
    /// Checked, with this left: the tests still failing, or what the review still finds.
    Partial(Vec<String>),
    /// No failure of the change's own, but these tests fail as they did on the base: never a
    /// pass, as an unfixed test a request names fails on the base too.
    AsBefore(Vec<String>),
    /// Nothing could say whether it works: no tests ran after it, or their failures could not
    /// be compared with the base's, and the review found nothing.
    #[default]
    Unverified,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Verified => f.write_str("verified: the tests pass"),
            Verdict::Partial(left) => write!(f, "partial, still left: {}", left.join("; ")),
            Verdict::AsBefore(old) => write!(
                f,
                "passes except {} failures that were there before the change: {}",
                old.len(),
                old.join(", ")
            ),
            Verdict::Unverified => {
                f.write_str("unverified: no check could say whether the change works")
            }
        }
    }
}

/// Proof that a human approved one packet's diff: only `Packet::approve` makes one, and
/// `CodeTask::apply` takes nothing else.
#[derive(Debug)]
pub struct Approval {
    diff: String,
}

impl CodeTask {
    /// The task `request` on the checkout at `repo`, at its HEAD, with `overrides` over what
    /// the repository's manifests say.
    pub fn new(request: &str, repo: &Path, overrides: Overrides) -> Result<Self, ToolError> {
        let real =
            fs::canonicalize(repo).map_err(|e| ToolError::Io(repo.display().to_string(), e))?;
        let repo = plain(real);
        let base_sha = git_in(&repo, &["rev-parse", "--verify", "HEAD"])?
            .trim_end()
            .to_string();
        Ok(Self {
            request: request.into(),
            base_sha,
            branch: None,
            worktree: None,
            allowed_paths: overrides.allowed_paths.unwrap_or_else(|| vec![".".into()]),
            test_cmd: overrides.test_cmd.or_else(|| detected(&repo)),
            check_cmds: overrides.check_cmds,
            budget: overrides.budget,
            tag: None,
            drafts: Vec::new(),
            folder: None,
            base_tests: None,
            verify: Verify::Full,
            read_only: false,
            repo,
        })
    }

    /// The task `request` done in place in `folder`, any folder of the user's but a drive's top,
    /// the home folder or one above it, and one that holds or is in `data_dir`. Its git is the
    /// repository the folder is in, or a private one under `data_dir`; `checkpoint` saves the
    /// folder before the agent writes anything.
    pub fn in_folder(
        request: &str,
        folder: &Path,
        data_dir: &Path,
        overrides: Overrides,
    ) -> Result<Self, ToolError> {
        let shown = |e| ToolError::Io(folder.display().to_string(), e);
        let repo = plain(fs::canonicalize(folder).map_err(shown)?);
        let data = std::path::absolute(data_dir).map_err(shown)?;
        let data = plain(fs::canonicalize(&data).unwrap_or(data));
        let home = env::home_dir()
            .and_then(|h| fs::canonicalize(h).ok())
            .map(plain);
        let refused = |why: &str| {
            Err(ToolError::Invalid(format!(
                "'{}' is {why}: name the project's own folder",
                repo.display()
            )))
        };
        if repo.parent().is_none() {
            return refused("a drive's top");
        }
        if home.is_some_and(|home| home.starts_with(&repo)) {
            return refused("the home folder or one above it");
        }
        if data.starts_with(&repo) || repo.starts_with(&data) {
            return refused("Larch's data folder, or holds it");
        }
        // git searches from the folder once, as `git::own` does, so it refuses another user's
        // repository; a folder in none gets a private one, named after the folder's real path.
        let found = git::run(
            &repo,
            None,
            &["rev-parse", "--show-toplevel", "--absolute-git-dir"],
        );
        let private = data
            .join("agent")
            .join("folders")
            .join(format!("{:016x}.git", crate::namespace(&repo)));
        let folder = match found {
            Ok(out) => {
                let mut lines = out.lines();
                let (Some(top), Some(git_dir)) = (lines.next(), lines.next()) else {
                    return Err(ToolError::Git(format!("rev-parse said: {out}")));
                };
                let git_dir = PathBuf::from(git_dir);
                let head = git::run(&repo, Some(&git_dir), &["rev-parse", "--verify", "HEAD"]);
                Folder {
                    index: Some(git_dir.join("index")),
                    head: head.ok().map(|h| h.trim_end().to_string()),
                    top: plain(fs::canonicalize(top).map_err(shown)?),
                    git_dir,
                    branch: None,
                    after: None,
                    backup: Some(private),
                }
            }
            Err(ToolError::Git(said)) if said.contains("not a git repository") => {
                let git_dir = private;
                if !git_dir.join("HEAD").is_file() {
                    fs::create_dir_all(&git_dir).map_err(shown)?;
                    git::run(&git_dir, Some(&git_dir), &["init", "--quiet", "--bare"])?;
                }
                Folder {
                    git_dir,
                    top: repo.clone(),
                    index: None,
                    head: None,
                    branch: None,
                    after: None,
                    backup: None,
                }
            }
            Err(e) => return Err(e),
        };
        Ok(Self {
            request: request.into(),
            base_sha: String::new(),
            branch: None,
            worktree: None,
            allowed_paths: overrides.allowed_paths.unwrap_or_else(|| vec![".".into()]),
            test_cmd: overrides.test_cmd.or_else(|| detected(&repo)),
            check_cmds: overrides.check_cmds,
            budget: overrides.budget,
            tag: None,
            drafts: Vec::new(),
            folder: Some(folder),
            base_tests: None,
            verify: Verify::Full,
            read_only: false,
            repo,
        })
    }

    /// Sets `check_cmds` from the attempt's files (`checks`) when the task has neither tests nor
    /// checks: what a done is checked against instead.
    /// Each runs once on all the files of its kind as they are now, the base: one that does not
    /// pass there (a library the sandbox cannot fetch, a warning already in the code, a time
    /// limit, no sandbox) could not tell this run's change from the base, and is dropped. But
    /// Python's two, which only ever run on the changed files: `IMPORT`, which would run every
    /// script's top level and is told from the base at the check (`Attempt::checked`), and
    /// `COMPILE`, which a file the repository keeps broken on purpose (a test fixture) would
    /// drop, though it only compiles the new files.
    // ponytail: a base the size of a big repository takes up to the command timeout per check;
    // check only the files the request names when a large run shows the wait
    pub(crate) fn detect_checks(&mut self) -> Result<(), ToolError> {
        // Tests near the change may be none: the checks stand in for them then.
        let near = (self.test_cmd.as_deref()).is_some_and(|c| c.contains("{tests"));
        if (self.test_cmd.is_some() && !near) || !self.check_cmds.is_empty() {
            return Ok(());
        }
        let files = self.files()?;
        let dir = self.dir()?;
        let passes = |run: &str| {
            matches!(
                code::build_check(run, dir, None, self.read_only),
                Ok((Some(true), ..))
            )
        };
        self.check_cmds = (checks(&files).into_iter())
            .filter(|c| {
                c == IMPORT || c == COMPILE || expand(c, &files, &[]).is_some_and(|r| passes(&r))
            })
            .collect();
        Ok(())
    }

    /// Those of `files`, relative to the folder, the base has.
    pub(crate) fn in_base(&self, files: &[String]) -> Result<Vec<String>, ToolError> {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let index = self.state_index()?;
        let args = [
            &["ls-tree", "--name-only", &self.base_sha, "--"][..],
            &(files.iter()).map(String::as_str).collect::<Vec<_>>(),
        ]
        .concat();
        let listed = self.indexed(&index, &args)?;
        Ok(listed.lines().map(String::from).collect())
    }

    /// The paths that differ between the base and `tree`, a state of the attempt's folder
    /// (`state`), relative to it: those still there, then every one, removed and renamed too.
    pub(crate) fn changed_files(
        &self,
        tree: &str,
    ) -> Result<(Vec<String>, Vec<String>), ToolError> {
        let names = ["diff", "--name-only", "-z", "--no-renames", "--relative"];
        let base = self.base_sha.as_str();
        let out = match &self.folder {
            Some(_) => {
                let args = [&names[..], &[base, tree, "--", "."]].concat();
                self.folder_git(&self.index()?, &args)?
            }
            None => self.git(&[&names[..], &[base, tree]].concat())?,
        };
        let dir = self.dir()?;
        let all: Vec<String> = out.split_terminator('\0').map(String::from).collect();
        let there = (all.iter()).filter(|p| dir.join(p).is_file()).cloned();
        Ok((there.collect(), all))
    }

    /// Whether a done's tests would run none: the command takes the tests near the change
    /// (`near`), and none is near it in a suite too big to run whole.
    pub(crate) fn none_near(&self) -> Result<bool, ToolError> {
        let Some(cmd) = (self.test_cmd.as_deref()).filter(|c| c.contains("{tests")) else {
            return Ok(false);
        };
        let changed = match self.state(&self.state_index()?)? {
            Some(tree) => self.changed_files(&tree)?.0,
            None => Vec::new(),
        };
        Ok(near(cmd, &self.files()?, &changed).is_none())
    }

    /// Sets `test_cmd` from the manifests now in the attempt's folder, when it has none: a run
    /// that writes one makes its task tested. Whether it did.
    pub(crate) fn detect_tests(&mut self) -> Result<bool, ToolError> {
        if self.test_cmd.is_some() {
            return Ok(false);
        }
        self.test_cmd = detected(self.dir()?);
        Ok(self.test_cmd.is_some())
    }

    /// Saves the folder of a task done in place as it is now, files git does not track yet too
    /// but none it ignores, as a commit on the new branch `larch/checkpoint-<task>-<ms>`
    /// with no parent, so its copy in `Folder::backup` is the folder alone, never the
    /// repository's history: the base of every diff and of `undo`. The user's index, HEAD and
    /// files stay as they are. Refused past `CHECKPOINT_FILES` or `CHECKPOINT_BYTES` new files.
    /// A `read_only` run gets the commit, its diff base, but no branch: it has nothing to undo.
    pub fn checkpoint(&mut self) -> Result<(), ToolError> {
        self.in_place()?;
        let index = self.index()?;
        let new = self.folder_git(
            &index,
            &[
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
                "--",
                ".",
            ],
        )?;
        let new: Vec<&str> = new.split_terminator('\0').collect();
        let bytes: u64 = (new.iter())
            .filter_map(|p| fs::symlink_metadata(self.repo.join(p)).ok())
            .map(|m| m.len())
            .sum();
        if new.len() > CHECKPOINT_FILES || bytes > CHECKPOINT_BYTES {
            return Err(ToolError::Invalid(format!(
                "'{}' holds {} files git would save that it does not track yet, {} MB; a \
                 checkpoint takes up to {CHECKPOINT_FILES} and {} MB: name the project's own \
                 folder, or ignore the rest in .gitignore",
                self.repo.display(),
                new.len(),
                bytes >> 20,
                CHECKPOINT_BYTES >> 20
            )));
        }
        let request = self.request.lines().next().unwrap_or_default();
        let message = format!("larch checkpoint before: {request}");
        let commit = self.commit(&index, &message, None)?;
        if self.read_only {
            self.base_sha = commit;
            return Ok(());
        }
        let ms = (SystemTime::now().duration_since(UNIX_EPOCH)).map_or(0, |d| d.as_millis());
        // The time alone keeps it apart; the tag, which tells worktrees apart, would only
        // lengthen a name the user reads.
        let branch = format!("larch/checkpoint-{}-{ms}", slug(&self.request));
        // The empty old value: git refuses a branch that already exists.
        let created = format!("refs/heads/{branch}");
        self.folder_git(&index, &["update-ref", &created, &commit, ""])?;
        self.keep(&commit, true)?;
        self.base_sha = commit;
        self.folder.as_mut().expect("in place").branch = Some(branch);
        Ok(())
    }

    /// An index for `state`, a copy of the attempt's or the user's: one per attempt, so each
    /// `state` rehashes only the files that changed since the last.
    pub fn state_index(&self) -> Result<tempfile::TempDir, ToolError> {
        if self.folder.is_some() {
            return self.index();
        }
        let (worktree, _) = self.attempt()?;
        let io = |e| ToolError::Io("an index of the attempt's own".into(), e);
        let dir = tempfile::tempdir().map_err(io)?;
        let admin = self.admin(worktree)?;
        copy_index(&admin.join("index"), dir.path()).map_err(io)?;
        Ok(dir)
    }

    /// The attempt's folder as a git tree, every file in it git does not ignore, written through
    /// `index` (`state_index`), or None when it is its base's: one id for one diff against the
    /// base. No index the user or the attempt reads changes, so a `git diff` still shows it.
    // ponytail: the first call hashes every file git does not ignore (a node_modules no
    // .gitignore names, in place up to CHECKPOINT_FILES); skip untracked folders when an eval
    // shows a step slow on it
    pub fn state(&self, index: &tempfile::TempDir) -> Result<Option<String>, ToolError> {
        let base = format!("{}^{{tree}}", self.base_sha);
        let (now, base) = match &self.folder {
            Some(_) => (
                self.tree(index)?,
                self.folder_git(index, &["rev-parse", &base])?,
            ),
            None => {
                let git = |args: &[&str]| self.indexed(index, args);
                git(&["add", "--all"])?;
                (git(&["write-tree"])?, git(&["rev-parse", &base])?)
            }
        };
        let now = now.trim_end();
        Ok((now != base.trim_end()).then(|| now.into()))
    }

    /// Puts back, as the checkpoint has them, the files the run changed, made or removed in the
    /// folder of a task done in place: the paths that differ between the checkpoint and the
    /// folder the run left (`Folder::after`), and only those. Refused when one of them changed
    /// again since, which would lose the user's own change. The paths it put back.
    pub fn undo(&self) -> Result<Vec<String>, ToolError> {
        let folder = self.in_place()?;
        let after = (folder.after.as_deref())
            .ok_or_else(|| ToolError::Invalid("the run has not ended: nothing to undo".into()))?;
        let index = self.index()?;
        let names = |from: &str, to: &str| -> Result<Vec<String>, ToolError> {
            let diff = [
                "diff",
                "--name-only",
                "-z",
                "--no-renames",
                "--relative",
                from,
                to,
            ];
            let out = self.folder_git(&index, &[&diff[..], &["--", "."]].concat())?;
            Ok(out.split_terminator('\0').map(String::from).collect())
        };
        let changed = names(&self.base_sha, after)?;
        if changed.is_empty() {
            return Ok(changed);
        }
        let now = self.commit(&index, "larch: the folder before undo", Some(after))?;
        let again: Vec<String> = (names(after, &now)?.into_iter())
            .filter(|p| changed.contains(p))
            .collect();
        if !again.is_empty() {
            return Err(ToolError::Invalid(format!(
                "changed again since the run, so undo would lose that: {}; put them back by hand \
                 from the branch {}",
                again.join(", "),
                folder.branch.as_deref().unwrap_or(&self.base_sha)
            )));
        }
        // The index holds the folder as it is now, so restore removes the files the run made,
        // which the checkpoint lacks; the paths go through a file, as there may be many.
        let list = index.path().join("paths");
        fs::write(&list, changed.join("\0")).map_err(|e| ToolError::Io("undo".into(), e))?;
        let list = utf8(&list)?;
        let restore = [
            "restore",
            "--worktree",
            "--source",
            &self.base_sha,
            "--pathspec-file-nul",
            "--pathspec-from-file",
            list,
        ];
        self.folder_git(&index, &restore)?;
        Ok(changed)
    }

    /// The folder's git of a task done in place, or why there is none.
    fn in_place(&self) -> Result<&Folder, ToolError> {
        self.folder
            .as_ref()
            .ok_or_else(|| ToolError::Invalid("the task is not done in place".into()))
    }

    /// An index of this task's own in a temporary folder: a copy of the user's, so git reads
    /// again only the files that changed, and the user's stays untouched; a private
    /// repository's own index is copied the same way.
    fn index(&self) -> Result<tempfile::TempDir, ToolError> {
        let folder = self.in_place()?;
        let io = |e| ToolError::Io("an index of the task's own".into(), e);
        let dir = tempfile::tempdir().map_err(io)?;
        let theirs = folder.index.clone().unwrap_or(folder.git_dir.join("index"));
        // The backup has none of the objects the user's index names, and the checkpoint has no
        // parent to keep HEAD's alive, so either starts from the base once there is one.
        let own = self.live(folder) == folder.git_dir;
        if !self.base_sha.is_empty() && (!own || !theirs.is_file()) {
            self.folder_git(&dir, &["read-tree", &self.base_sha])?;
        } else if theirs.is_file() {
            copy_index(&theirs, dir.path()).map_err(io)?;
        } else if let Some(head) = &folder.head {
            self.folder_git(&dir, &["read-tree", head])?;
        }
        Ok(dir)
    }

    /// The folder as it is now, written as a tree through `index`: the files git does not
    /// ignore, under the folder only; the rest of the repository as `index` had it.
    fn tree(&self, index: &tempfile::TempDir) -> Result<String, ToolError> {
        self.folder_git(index, &["add", "--all", "--", "."])?;
        Ok(self
            .folder_git(index, &["write-tree"])?
            .trim_end()
            .to_string())
    }

    /// The folder as it is now, committed as Iris on `parent` with `message`, on no branch.
    fn commit(
        &self,
        index: &tempfile::TempDir,
        message: &str,
        parent: Option<&str>,
    ) -> Result<String, ToolError> {
        let tree = self.tree(index)?;
        let mut commit = vec!["commit-tree", &tree, "-m", message];
        commit.extend(parent.iter().flat_map(|p| ["-p", *p]));
        let made = self.folder_git(index, &[&IRIS[..], &commit].concat())?;
        Ok(made.trim_end().to_string())
    }

    /// git on the folder of a task done in place, from the folder (so `.` is it), through its
    /// repository and `index`, never the user's.
    fn folder_git(&self, index: &tempfile::TempDir, args: &[&str]) -> Result<String, ToolError> {
        let folder = self.in_place()?;
        let file = index.path().join("index");
        let env = [
            ("GIT_WORK_TREE", folder.top.as_path()),
            ("GIT_INDEX_FILE", &file),
        ];
        git::run_env(&self.repo, Some(self.live(folder)), &env, args)
    }

    /// The git that holds the checkpoint and the run's end state (`Folder::after`): the
    /// folder's own, or `Folder::backup` once a command left the folder's without either
    /// (removed, made again empty, or collected by a gc).
    fn live<'f>(&self, folder: &'f Folder) -> &'f Path {
        let Some(backup) = &folder.backup else {
            return &folder.git_dir;
        };
        let has = |sha: &str| {
            let object = format!("{sha}^{{commit}}");
            git::run(
                &self.repo,
                Some(&folder.git_dir),
                &["cat-file", "-e", &object],
            )
            .is_ok()
        };
        let needed = [Some(self.base_sha.as_str()), folder.after.as_deref()];
        match (needed.into_iter().flatten()).all(|sha| sha.is_empty() || has(sha)) {
            true => &folder.git_dir,
            false => backup,
        }
    }

    /// Copies `commit`, made in the folder's own git, to `Folder::backup`; nothing when there is
    /// none or the backup is the git in use already. With `settings`, the backup also takes the
    /// folder's ignore and attribute files and its line-ending settings, so a fallback
    /// snapshot sees the folder as the checkpoint did.
    // ponytail: one ref per run, never pruned; drop old ones when a backup's size shows
    fn keep(&self, commit: &str, settings: bool) -> Result<(), ToolError> {
        let folder = self.in_place()?;
        let Some(backup) = &folder.backup else {
            return Ok(());
        };
        if self.live(folder) == backup.as_path() {
            return Ok(());
        }
        let io = |e| ToolError::Io(backup.display().to_string(), e);
        if !backup.join("HEAD").is_file() {
            fs::create_dir_all(backup).map_err(io)?;
            git::run(backup, Some(backup), &["init", "--quiet", "--bare"])?;
        }
        if settings {
            fs::create_dir_all(backup.join("info")).map_err(io)?;
            for name in ["exclude", "attributes"] {
                let theirs = fs::read(folder.git_dir.join("info").join(name)).unwrap_or_default();
                fs::write(backup.join("info").join(name), theirs).map_err(io)?;
            }
            let own = |args: &[&str]| git::run(&self.repo, Some(&folder.git_dir), args);
            let kept = |args: &[&str]| git::run(backup, Some(backup), args);
            let set = |key: &str, value: Option<&str>| -> Result<(), ToolError> {
                match value {
                    Some(value) => drop(kept(&["config", key, value])?),
                    None if kept(&["config", "--get", key]).is_ok() => {
                        drop(kept(&["config", "--unset-all", key])?)
                    }
                    None => {}
                }
                Ok(())
            };
            // A configured file is copied whole beside the backup and named the same way, so its
            // rules keep their own precedence below info/'s.
            for (key, copy) in [
                ("core.excludesFile", "larch-excludes"),
                ("core.attributesFile", "larch-attributes"),
            ] {
                match own(&["config", "--path", "--get", key]) {
                    Ok(file) => {
                        let theirs = fs::read(folder.top.join(file.trim_end())).unwrap_or_default();
                        fs::write(backup.join(copy), theirs).map_err(io)?;
                        set(key, Some(utf8(&backup.join(copy))?))?;
                    }
                    Err(_) => set(key, None)?,
                }
            }
            // Plain values git reads the folder's bytes and names by, never a program to run.
            for key in [
                "core.autocrlf",
                "core.eol",
                "core.safecrlf",
                "core.filemode",
                "core.symlinks",
                "core.ignoreCase",
            ] {
                let value = own(&["config", "--get", key]).ok();
                set(key, value.as_deref().map(str::trim_end))?;
            }
        }
        let refspec = format!("{commit}:refs/larch/{commit}");
        // The backup fetches, so only its own config picks where the objects go: the folder's
        // (a url rewrite, a receiving hook) never runs. A local path and an unnamed commit are
        // allowed for this fetch alone.
        let fetch = [
            "-c",
            "protocol.file.allow=always",
            "-c",
            "uploadpack.allowAnySHA1InWant=true",
            "fetch",
            "--quiet",
            "--no-tags",
            utf8(&folder.git_dir)?,
            &refspec,
        ];
        git::run(backup, Some(backup), &fetch).map(drop)
    }

    /// The folder the attempt works in: its worktree, or the user's folder done in place.
    pub fn dir(&self) -> Result<&Path, ToolError> {
        match &self.folder {
            Some(_) => Ok(&self.repo),
            None => self.attempt().map(|(worktree, _)| worktree),
        }
    }

    /// Whether the attempt changed anything: its worktree's status, or the folder against its
    /// checkpoint.
    pub fn changed(&self) -> Result<bool, ToolError> {
        match &self.folder {
            Some(_) => Ok(self.state(&self.index()?)?.is_some()),
            None => Ok(!self.git(&["status", "--porcelain"])?.is_empty()),
        }
    }

    /// The files of the attempt's folder git tracks or would, relative to it.
    pub fn files(&self) -> Result<Vec<String>, ToolError> {
        let listed = match &self.folder {
            Some(_) => {
                let index = self.index()?;
                let all = [
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ];
                self.folder_git(&index, &[&all[..], &["--", "."]].concat())?
            }
            None => self.git(&["ls-files", "-z"])?,
        };
        Ok(listed.split_terminator('\0').map(String::from).collect())
    }

    /// Attempt `n`'s worktree, `<data_dir>/agent/code/<task>/<n>/<project>` on the new branch
    /// `iris/<task>-<n>` at `base_sha`; `<task>` is the request's first words, then `tag`, and
    /// `<project>` the checkout's folder name, which a build may take its name from (CMake's
    /// `get_filename_component(... NAME)`).
    /// Two tasks with the same name collide: git refuses the second.
    pub fn add_worktree(&mut self, data_dir: &Path, n: u32) -> Result<(), ToolError> {
        if self.folder.is_some() {
            return Err(ToolError::Invalid(
                "the task is done in place: it has no worktree".into(),
            ));
        }
        if self.worktree.is_some() || self.branch.is_some() {
            return Err(ToolError::Invalid(
                "the task's attempt is still open; discard it first".into(),
            ));
        }
        let name = match &self.tag {
            Some(tag) => format!("{}-{}", slug(&self.request), slug(tag)),
            None => slug(&self.request),
        };
        let io = |e| ToolError::Io(data_dir.display().to_string(), e);
        // git runs in the checkout, so a relative data folder would land inside it.
        let path = std::path::absolute(data_dir)
            .map_err(io)?
            .join("agent")
            .join("code")
            .join(&name)
            .join(n.to_string())
            .join(self.repo.file_name().unwrap_or("repo".as_ref()));
        let branch = format!("iris/{name}-{n}");
        let shown = utf8(&path)?;
        git_in(
            &self.repo,
            &[
                "worktree",
                "add",
                "--quiet",
                "-b",
                &branch,
                shown,
                &self.base_sha,
            ],
        )?;
        self.worktree = Some(path);
        self.branch = Some(branch);
        Ok(())
    }

    /// Removes the current attempt's worktree and its branch, each forgotten once it is gone,
    /// so a discard that failed half-way can be run again. A task done in place has neither:
    /// its folder is the user's, and its checkpoint branch stays.
    pub fn discard(&mut self) -> Result<(), ToolError> {
        if self.folder.is_some() {
            return Ok(());
        }
        if let Some(path) = &self.worktree {
            match self.admin(path) {
                // git checks the worktree's `.git` before it removes it, and the sandboxed
                // commands may have rewritten that file: the one git wrote goes back first.
                Ok(admin) => {
                    let dot = path.join(".git");
                    restore(&dot, utf8(&admin)?)
                        .map_err(|e| ToolError::Io(dot.display().to_string(), e))?;
                    let removed =
                        git_in(&self.repo, &["worktree", "remove", "--force", utf8(path)?]);
                    // The attempt's folder, empty once its worktree is gone; one left holds
                    // nothing and the next attempt of that number reuses it.
                    if removed.is_ok()
                        && let Some(attempt) = path.parent()
                    {
                        let _ = fs::remove_dir(attempt);
                    }
                    if let Err(e) = removed {
                        // A removal that failed half-way (a file git could not delete) leaves
                        // files, not a worktree: git forgot it, so its branch can go now and the
                        // same request can run again once the files are gone.
                        if self.admin(path).is_err()
                            && let Some(branch) = &self.branch
                        {
                            git_in(&self.repo, &["branch", "--quiet", "-D", branch])?;
                            self.branch = None;
                        }
                        return Err(e);
                    }
                }
                // git keeps no folder for it: a removal that failed half-way dropped the folder
                // and left the files, which only the file system can remove now; prune forgets
                // an entry whose files are gone.
                Err(_) => {
                    if path.exists() {
                        fs::remove_dir_all(path)
                            .map_err(|e| ToolError::Io(path.display().to_string(), e))?;
                    }
                    git_in(&self.repo, &["worktree", "prune"])?;
                }
            }
            self.worktree = None;
        }
        if let Some(branch) = &self.branch {
            git_in(&self.repo, &["branch", "--quiet", "-D", branch])?;
            self.branch = None;
        }
        Ok(())
    }

    /// The current attempt's packet: everything in its worktree committed on its branch, the
    /// diff against `base_sha` and the scope of every changed path. In place, the folder as the
    /// run left it is committed on the checkpoint, on no branch (`Folder::after`), and the diff
    /// is the folder's alone. The tests and commands are the caller's to fill. In a worktree,
    /// every new file goes in, however it was made (a rename is a command's `mv`), but build
    /// output (`build_output`): it would land in the user's checkout naming a worktree that is
    /// gone. The body names what was left out.
    pub fn packet(&mut self, pr_title: &str, pr_body: &str) -> Result<Packet, ToolError> {
        let base = self.base_sha.clone();
        let mut body = pr_body.to_string();
        let (names, diff) = match &self.folder {
            Some(_) => {
                let index = self.index()?;
                let after = self.commit(&index, pr_title, Some(&base))?;
                let diff = |more: &[&str]| {
                    let args = [
                        &DIFF[..],
                        &["--relative"],
                        more,
                        &[base.as_str(), after.as_str(), "--", "."],
                    ];
                    self.folder_git(&index, &args.concat())
                };
                let shown = (diff(&["--name-only", "-z"])?, diff(&[])?);
                self.keep(&after, false)?;
                self.folder.as_mut().expect("in place").after = Some(after);
                shown
            }
            None => {
                self.git(&["add", "--update"])?;
                let new = ["ls-files", "-z", "--others", "--exclude-standard"];
                // One pathspec per dump or build tree, never per file, so a big tree stays short.
                let out: Vec<String> = (self.git(&new)?.split_terminator('\0'))
                    .filter_map(build_output)
                    .map(|p| format!(":(exclude,literal){p}"))
                    .collect();
                let mut add = vec!["add", "--all", "--", "."];
                add.extend(out.iter().map(String::as_str));
                self.git(&add)?;
                let left = self.git(&new)?;
                let left: Vec<&str> = left.split_terminator('\0').collect();
                if !left.is_empty() {
                    let shown = left[..left.len().min(5)].join(", ");
                    let more = match left.len().saturating_sub(5) {
                        0 => String::new(),
                        n => format!(" and {n} more"),
                    };
                    let _ = write!(body, "\n\nLeft out, build output: {shown}{more}");
                }
                // What is staged, not what the folder holds: a file left out stays untracked.
                let staged = [
                    "diff",
                    "--cached",
                    "--name-only",
                    "--ignore-submodules=none",
                ];
                if !self.git(&staged)?.is_empty() {
                    let commit = ["commit", "--quiet", "--allow-empty-message", "-m", pr_title];
                    self.git(&[&IRIS[..], &commit].concat())?;
                }
                let diff = |more: &[&str]| {
                    self.git(&[&DIFF[..], more, &[base.as_str(), "HEAD", "--"]].concat())
                };
                (diff(&["--name-only", "-z"])?, diff(&[])?)
            }
        };
        Ok(Packet {
            diff,
            tests_before: None,
            tests_after: None,
            scope: names
                .split_terminator('\0')
                .map(|path| Scope {
                    path: path.into(),
                    inside: inside(&self.allowed_paths, path),
                })
                .collect(),
            commands_run: vec![],
            pr_title: pr_title.into(),
            pr_body: body,
            stopped: None,
            verdict: Verdict::Unverified,
        })
    }

    /// Merges the current attempt's branch into the user's checkout, once a human approved
    /// `packet` and while the branch still holds the diff they approved.
    pub fn apply(&self, packet: &Packet, approval: Approval) -> Result<(), ToolError> {
        let (_, branch) = self.attempt()?;
        // The commit is read once and merged by its ID, so one made after the check never is;
        // its diff comes from the worktree, as the packet's did, under the same attributes.
        let head = self.git(&["rev-parse", "--verify", branch])?;
        let head = head.trim_end();
        let now = self.git(&[&DIFF[..], &[&self.base_sha, head, "--"]].concat())?;
        if approval.diff != packet.diff || now != packet.diff {
            return Err(ToolError::Invalid(format!(
                "'{branch}' holds not the diff that was approved; render its packet again and approve that"
            )));
        }
        // The approval covers base_sha..head alone: a checkout off base_sha's line would take in
        // commits the packet never showed, and the abort below must only undo this call's merge.
        let merging = ["rev-parse", "--quiet", "--verify", "MERGE_HEAD"];
        if holds(&self.repo, &merging)? {
            return Err(ToolError::Invalid(
                "the checkout is in the middle of a merge; finish or abort it first".into(),
            ));
        }
        let ancestor = ["merge-base", "--is-ancestor", &self.base_sha, "HEAD"];
        if !holds(&self.repo, &ancestor)? {
            return Err(ToolError::Invalid(format!(
                "the checkout's HEAD does not descend from {}, so the merge would bring in commits the packet never showed; check out the branch the task started on",
                self.base_sha
            )));
        }
        let message = format!("Merge branch '{branch}'");
        let merge = ["merge", "--quiet", "--no-edit", "-m", &message, head];
        let merged = git_in(&self.repo, &[&IRIS[..], &merge].concat());
        // A merge stopped by a conflict is undone, leaving the checkout as the user had it.
        if merged.is_err() && holds(&self.repo, &merging)? {
            git_in(&self.repo, &["merge", "--abort"])?;
        }
        merged.map(drop)
    }

    /// Runs `test_cmd` in the current attempt's worktree, through the sandbox: the run, for the
    /// packet's `tests_before` and `tests_after`, and the text `run_tests` returns, whose first
    /// line says how the command ended.
    pub fn test(&self) -> Result<(TestRun, String), ToolError> {
        self.test_until(None, true)
    }

    /// `test`, its command killed once `stop` is set: the user's Stop of the run. A command on
    /// the tests near the change (`near`) runs those; with `base`, failures it shows are run
    /// again with the changed sources as the base has them, and those that fail there too are
    /// not the change's (`code::against`). Without `base`, as beside the model's reads, the
    /// folder is never touched. A whole-suite command's failures are compared with `base_tests`,
    /// the same command's run on the base; with `base` and none, they could not be compared.
    pub fn test_until(
        &self,
        stop: Option<&Arc<AtomicBool>>,
        base: bool,
    ) -> Result<(TestRun, String), ToolError> {
        self.test_these(stop, base, &[])
    }

    /// `test_until` on `tests`, labels the model named (`code::labels`), in the command's
    /// `{tests}` or `{tests app}` or after it, instead of the tests near the change; their
    /// failures are compared with the same labels' run on the base, as near tests' are.
    pub fn test_these(
        &self,
        stop: Option<&Arc<AtomicBool>>,
        base: bool,
        tests: &[String],
    ) -> Result<(TestRun, String), ToolError> {
        let worktree = self.dir()?;
        let cmd = self.test_cmd.as_deref().ok_or_else(|| {
            ToolError::Invalid(
                "the task has no test command: none was found in the repository, and the user \
                 named none"
                    .into(),
            )
        })?;
        // In a worktree, an editable install's import path names the user's checkout instead.
        let user = self.folder.is_none().then_some(self.repo.as_path());
        // Simple reports the run as it is, compared with nothing: the one place it differs.
        let full = self.verify == Verify::Full;
        let base = base && full;
        if !cmd.contains("{tests") && tests.is_empty() {
            let (after, shown) = code::run_tests(cmd, worktree, user, stop, self.read_only)?;
            let before = match &self.base_tests {
                _ if after.failed == 0 || !full || code::uncompared(&shown) => {
                    return Ok((after, shown));
                }
                // No test ran on the base, so none failed there: these failures are the change's.
                Some(before) if before.none_ran() => return Ok((after, shown)),
                Some(before) => Ok(before.clone()),
                None if base => Err("the suite has no run on the base to compare with".into()),
                None => return Ok((after, shown)),
            };
            return Ok(code::against(before, after, &shown));
        }
        let (changed, all) = match self.state(&self.state_index()?)? {
            Some(tree) => self.changed_files(&tree)?,
            None => Default::default(),
        };
        let run = match tests.is_empty() {
            true => near(cmd, &self.files()?, &changed),
            false => Some(with_tests(cmd, tests)),
        };
        let Some(run) = run else {
            let none = TestRun {
                passed: 0,
                failed: 0,
                failing: Vec::new(),
                shared: Vec::new(),
                tail: String::new(),
            };
            let why = "no tests ran: none is near the changed files, and the suite is too big to \
                       run whole; name the tests to run: run_tests with tests=[...] (e.g. a test \
                       file or the runner's label)";
            return Ok((none, why.into()));
        };
        let (after, shown) = code::run_tests(&run, worktree, user, stop, self.read_only)?;
        if !base || after.failed == 0 || all.is_empty() || code::uncompared(&shown) {
            return Ok((after, shown));
        }
        let folders = [worktree, self.repo.as_path()];
        // A checkout that fails (a full tmp, a path too long there) leaves only no comparison.
        // Every changed path as the base has it, a removed or renamed one too, and a test the
        // change wrote: one the base lacks never fails there.
        let before = (self.at_base(&all, |base, _| code::base_tests(&run, base, &folders, stop)))
            .unwrap_or_else(|e| Err(format!("the base could not be checked out: {e}")));
        Ok(code::against(before, after, &shown))
    }

    /// `run` in a checkout of its own of the folder as it is now (every file git does not
    /// ignore) but for `files`, relative to the folder, as the base has them, one it lacks left
    /// out; given that checkout's folder and those of `files` the base has. The folder itself is
    /// never written, so a crash or a save of the user's at any moment loses nothing. What git
    /// ignores (a build's output, a virtualenv) is not in the checkout: a base run that needs it
    /// fails there, which only leaves the comparison undone.
    pub fn at_base<T>(
        &self,
        files: &[String],
        run: impl FnOnce(&Path, &[String]) -> T,
    ) -> Result<T, ToolError> {
        let index = self.state_index()?;
        // The index now holds the folder as it is.
        self.state(&index)?;
        let git = |args: &[&str]| self.indexed(&index, args);
        let mut kept = Vec::new();
        for file in files {
            let path = format!("./{file}");
            // --cacheinfo takes the path from the top, which --full-name gives.
            let listed = git(&["ls-tree", "--full-name", &self.base_sha, "--", &path])?;
            let entry = (listed.trim_end().split_once('\t')).and_then(|(head, top)| {
                let mut parts = head.split(' ');
                Some((parts.next()?, parts.nth(1)?, top))
            });
            // A folder in the base (the change made a file of it) comes back through its
            // files, which are paths of their own; --add takes back a removed one.
            match entry.filter(|(mode, ..)| *mode != "040000") {
                Some((mode, blob, top)) => {
                    let info = format!("{mode},{blob},{top}");
                    let add = ["update-index", "--add", "--replace", "--cacheinfo", &info];
                    git(&add)?;
                    kept.push(file.clone());
                }
                None => drop(git(&["update-index", "--force-remove", "--", &path])?),
            }
        }
        let io = |e| ToolError::Io("a checkout of the base".into(), e);
        sweep();
        let tmp = (tempfile::Builder::new().prefix(BASE_CHECKOUT).tempdir()).map_err(io)?;
        let tmp = Checkout(tmp);
        let to = format!("--prefix={}/", utf8(tmp.0.path())?);
        git(&["checkout-index", "--all", "--force", &to])?;
        let prefix = match &self.folder {
            Some(_) => git(&["rev-parse", "--show-prefix"])?,
            None => String::new(),
        };
        Ok(run(&tmp.0.path().join(prefix.trim_end()), &kept))
    }

    /// git on the attempt's folder through `index` (`state_index`), never the folder's own.
    fn indexed(&self, index: &tempfile::TempDir, args: &[&str]) -> Result<String, ToolError> {
        if self.folder.is_some() {
            return self.folder_git(index, args);
        }
        let (worktree, _) = self.attempt()?;
        let file = index.path().join("index");
        let env = [("GIT_INDEX_FILE", file.as_path())];
        let admin = self.admin(worktree)?;
        git::run_env(
            worktree,
            Some(&admin),
            &env,
            &[&NO_NESTED[..], args].concat(),
        )
    }

    /// The `run_tests` tool of the current attempt, for code mode's registry
    /// (`Registry::replace`), whose command a Stop of the run (`stop`) ends; the next attempt
    /// makes its own.
    pub fn run_tests(&self, stop: Option<Arc<AtomicBool>>) -> Box<dyn Tool> {
        Box::new(code::RunTests(self.clone(), stop))
    }

    /// git on the current attempt's worktree, through the folder git keeps for it in the
    /// checkout's repository, never through the worktree's `.git`: the sandboxed commands may
    /// rewrite that file, and git would then run what the repository it names says.
    pub(crate) fn git(&self, args: &[&str]) -> Result<String, ToolError> {
        let (worktree, _) = self.attempt()?;
        git::run(
            worktree,
            Some(&self.admin(worktree)?),
            &[&NO_NESTED[..], args].concat(),
        )
    }

    /// The folder git keeps for the worktree at `path`: the one in the checkout's `worktrees/`
    /// whose `gitdir`, which only git writes, names `path`'s `.git`.
    fn admin(&self, path: &Path) -> Result<PathBuf, ToolError> {
        let common = git_in(&self.repo, &["rev-parse", "--git-common-dir"])?;
        let worktrees = self.repo.join(common.trim_end()).join("worktrees");
        let io = |e| ToolError::Io(worktrees.display().to_string(), e);
        let real = fs::canonicalize(path).map_err(io)?;
        for entry in fs::read_dir(&worktrees).map_err(io)? {
            let admin = entry.map_err(io)?.path();
            // An entry without one is another worktree's, half made or half removed.
            let Ok(named) = fs::read_to_string(admin.join("gitdir")) else {
                continue;
            };
            let dot = Path::new(named.trim_end());
            let top = dot.parent().and_then(|p| fs::canonicalize(p).ok());
            if dot.ends_with(".git") && top.is_some_and(|top| top == real) {
                return Ok(admin);
            }
        }
        Err(ToolError::Invalid(format!(
            "git keeps no folder for the worktree '{}'",
            path.display()
        )))
    }

    /// The current attempt's worktree and branch.
    fn attempt(&self) -> Result<(&Path, &str), ToolError> {
        match (&self.worktree, &self.branch) {
            (Some(path), Some(branch)) => Ok((path, branch)),
            _ => Err(ToolError::Invalid(
                "the task has no worktree; add one first".into(),
            )),
        }
    }
}

impl Packet {
    /// The approval `apply` needs, when the human said yes to this packet as rendered.
    pub fn approve(&self, yes: bool) -> Option<Approval> {
        yes.then(|| Approval {
            diff: self.diff.clone(),
        })
    }

    /// The packet as the CLI shows it for approval.
    pub fn render(&self) -> String {
        let tests = |run: &Option<TestRun>| match run {
            None => "not run".to_string(),
            Some(t) if t.none_ran() => "not run".to_string(),
            Some(t) => {
                let mut shown = format!("{} passed, {} failed", t.passed, t.failed);
                if !t.failing.is_empty() {
                    let _ = write!(shown, ": {}", t.failing.join(", "));
                }
                if !t.shared.is_empty() {
                    let _ = write!(shown, "; {} {}", code::AS_BEFORE, t.shared.join(", "));
                }
                shown
            }
        };
        let mut out = format!("{}\n\n{}\n\n", self.pr_title, self.pr_body);
        let _ = writeln!(out, "tests before: {}", tests(&self.tests_before));
        let _ = writeln!(out, "tests after: {}", tests(&self.tests_after));
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "stopped: {why}");
        }
        let _ = writeln!(out, "verdict: {}", self.verdict);
        for s in &self.scope {
            let place = if s.inside {
                "inside"
            } else {
                "outside the allowed paths"
            };
            let _ = writeln!(out, "{place}: {}", s.path);
        }
        for c in &self.commands_run {
            let _ = writeln!(out, "$ {c}");
        }
        out + "\n" + &self.diff
    }
}

/// The test command the first of `RUNNERS` found in `dir` names.
fn detected(dir: &Path) -> Option<String> {
    let holds = |file: &str, text: &str| {
        fs::read_to_string(dir.join(file)).is_ok_and(|t| format!("\n{t}").contains(text))
    };
    (RUNNERS.iter())
        .find(|(file, text, _)| holds(file, text))
        .map(|(_, _, cmd)| cmd.to_string())
}

/// Whether `path` is a Python test file by its name, as pytest and Django find them.
fn is_test(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".py")
        && (name.starts_with("test_") || name.ends_with("_test.py") || name == "tests.py")
}

/// The test files of `files` near the `changed` ones: those changed, and each whose path has the
/// name of a changed module (its file's, or its package's for `__init__.py`) as whole words
/// (`tests/model_enums/tests.py` for `enums.py`, `tests/test_ext_autodoc.py` for
/// `autodoc/__init__.py`).
fn near_tests(files: &[String], changed: &[String]) -> Vec<String> {
    let words = |path: &str| -> Vec<String> {
        let path = path.strip_suffix(".py").unwrap_or(path);
        (path.split(['/', '_', '-'])).map(String::from).collect()
    };
    let mut near: Vec<String> = changed.iter().filter(|f| is_test(f)).cloned().collect();
    for f in changed.iter().filter(|f| f.ends_with(".py") && !is_test(f)) {
        let mut names = f.strip_suffix(".py").unwrap_or(f).rsplit('/');
        let module = match names.next() {
            Some("__init__") => names.next(),
            name => name,
        };
        let Some(module) = module else { continue };
        // A name of several words (date_utils) matches them in a row.
        let module = words(module);
        near.extend(
            (files.iter())
                .filter(|t| is_test(t) && words(t).windows(module.len()).any(|w| w == module))
                .cloned(),
        );
    }
    near.sort();
    near.dedup();
    near
}

/// `cmd`, a test command, run on the tests of `files` near the `changed` ones (`near_tests`):
/// `{tests}` their paths, `{tests app}` the folders under tests/ they are in. With none near,
/// the whole suite when it has up to `SMALL_SUITE` test files, else None. A command without
/// either is as it is.
// ponytail: the whole suite on Windows, whose cmd.exe keeps the quotes `{tests}` puts around
// each path, and a failing one runs whole again on the base: two full runs per run_tests on a
// Django-sized suite; quote for cmd.exe when a Windows run needs near tests
pub(crate) fn near(cmd: &str, files: &[String], changed: &[String]) -> Option<String> {
    let app = match (cmd.contains("{tests}"), cmd.contains("{tests app}")) {
        (true, _) => false,
        (_, true) => true,
        _ => return Some(cmd.into()),
    };
    if cfg!(windows) {
        return Some(with_tests(cmd, &[]));
    }
    let mut args: Vec<String> = (near_tests(files, changed).into_iter())
        .filter_map(|t| match app {
            true => (t.strip_prefix("tests/")?.split_once('/')).map(|(app, _)| app.to_string()),
            false => Some(t),
        })
        .collect();
    args.dedup();
    if args.is_empty() && files.iter().filter(|f| is_test(f)).count() > SMALL_SUITE {
        return None;
    }
    Some(with_tests(cmd, &args))
}

/// `cmd` with `tests`, each shell-quoted, in its `{tests}` or `{tests app}`, or after it when it
/// has neither. On Windows they go bare: cmd.exe would keep the quotes, and a named test is one
/// word (`code::labels`).
fn with_tests(cmd: &str, tests: &[String]) -> String {
    let quote = |t: &String| match cfg!(windows) {
        true => t.clone(),
        false => format!("'{}'", t.replace('\'', r"'\''")),
    };
    let args = tests.iter().map(quote).collect::<Vec<_>>().join(" ");
    match ["{tests}", "{tests app}"]
        .into_iter()
        .find(|k| cmd.contains(k))
    {
        Some(key) => cmd.replace(key, &args).trim_end().into(),
        None => format!("{cmd} {args}").trim_end().into(),
    }
}

/// git on the repository whose top is `dir`, never one around it nor another user's.
fn git_in(dir: &Path, args: &[&str]) -> Result<String, ToolError> {
    let real = fs::canonicalize(dir).map_err(|e| ToolError::Io(dir.display().to_string(), e))?;
    git::own(&real, args)?.ok_or_else(|| {
        ToolError::Invalid(format!(
            "'{}' is not the top of a git repository",
            dir.display()
        ))
    })
}

/// What of the untracked `path` is build output, never source, as one path: a core dump
/// (`core`, `core.<pid>`), or the CMake build tree its `CMakeCache.txt` names.
fn build_output(path: &str) -> Option<&str> {
    let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
    let pid = |p: &str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
    match name {
        "CMakeCache.txt" if !dir.is_empty() => Some(dir),
        "CMakeCache.txt" | "core" => Some(path),
        _ => name.strip_prefix("core.").filter(|p| pid(p)).map(|_| path),
    }
}

/// Puts back the gitfile `dot` naming `admin`, as `git worktree add` wrote it, over whatever is
/// there now.
fn restore(dot: &Path, admin: &str) -> io::Result<()> {
    match fs::symlink_metadata(dot) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(dot)?,
        Ok(_) => fs::remove_file(dot)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    fs::write(dot, format!("gitdir: {admin}\n"))
}

/// Whether git's yes-or-no check `args` holds: exit 0 yes, exit 1 with nothing said no, and
/// anything else an error.
fn holds(dir: &Path, args: &[&str]) -> Result<bool, ToolError> {
    match git_in(dir, args) {
        Ok(_) => Ok(true),
        Err(ToolError::Git(said)) if said.is_empty() => Ok(false),
        Err(e) => Err(e),
    }
}

/// `path` as git takes it; a path that is not Unicode, read lossily, would name another.
fn utf8(path: &Path) -> Result<&str, ToolError> {
    path.to_str()
        .ok_or_else(|| ToolError::Invalid(format!("'{}' is not Unicode", path.display())))
}

/// The request's first words, lower case, as a branch and folder name.
fn slug(request: &str) -> String {
    let words: Vec<String> = request
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(6)
        .map(str::to_ascii_lowercase)
        .collect();
    if words.is_empty() {
        "task".into()
    } else {
        words.join("-")
    }
}

/// Whether one of `allowed` holds `path`; `.git` is never inside.
fn inside(allowed: &[String], path: &str) -> bool {
    let under = |dir: &str| path.strip_prefix(dir).is_some_and(|r| r.starts_with('/'));
    path != ".git"
        && !under(".git")
        && allowed
            .iter()
            .map(|a| a.trim_end_matches('/'))
            .any(|a| a == "." || a == path || under(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(files: &[&str]) -> Vec<String> {
        files.iter().map(|f| f.to_string()).collect()
    }

    /// Python sources get one import of the changed modules the base has and one compile of the
    /// new ones, and nothing when none changed.
    #[test]
    fn python_is_imported_in_one_run() {
        let checks = checks(&names(&["pkg/a.py", "README.md"]));
        assert_eq!(checks, [IMPORT, COMPILE]);
        let changed = names(&["pkg/a.py", "it's.py", "new.py", "notes.txt"]);
        let old = names(&["pkg/a.py", "it's.py"]);
        let run = expand(IMPORT, &changed, &old).unwrap();
        assert!(run.contains("importlib.import_module("), "{run}");
        assert!(
            run.ends_with(r#"sys.exit(bad)" 'it'\''s.py' 'pkg/a.py'"#),
            "{run}"
        );
        assert!(run.starts_with("PYTHONPYCACHEPREFIX=\"$TMPDIR\" "), "{run}");
        let run = expand(COMPILE, &changed, &old).unwrap();
        assert!(run.ends_with("py_compile 'new.py'"), "{run}");
        assert_eq!(expand(COMPILE, &old, &old), None);
        assert_eq!(expand(IMPORT, &names(&["notes.txt"]), &[]), None);
    }

    /// JavaScript gets node --check on each changed .js file.
    #[test]
    fn javascript_is_checked_file_by_file() {
        let checks = checks(&names(&["web/app.js", "index.html"]));
        let run = expand(&checks[0], &names(&["web/app.js", "b.js"]), &[]).unwrap();
        assert!(
            run.starts_with("for f in 'b.js' 'web/app.js'; do node --check"),
            "{run}"
        );
    }

    /// A Go module gets go vet on each folder of a changed .go file, once; no go.mod, no check.
    #[test]
    fn go_is_vetted_by_changed_package() {
        let checks = checks(&names(&["go.mod", "ledger.go", "sub/x.go"]));
        let changed = names(&["a.go", "sub/x.go", "ledger.go"]);
        assert_eq!(
            expand(&checks[0], &changed, &[]).unwrap(),
            "go vet '.' './sub'"
        );
        assert!(super::checks(&names(&["ledger.go"])).is_empty());
    }

    /// Java compiles from the Maven or Gradle source folder with a build file, without one only
    /// a few loose files, never running an annotation processor.
    #[test]
    fn java_needs_a_build_file_or_a_few_files() {
        let built = checks(&names(&["build.gradle", "src/main/java/a/A.java"]));
        assert!(built[0].contains("-proc:none"), "{built:?}");
        assert!(built[0].contains("-sourcepath src/main/java "), "{built:?}");
        let loose: Vec<String> = (0..=LOOSE).map(|n| format!("A{n}.java")).collect();
        assert!(checks(&loose).is_empty());
        assert!(
            checks(&loose[1..])
                .iter()
                .any(|c| c.contains("-sourcepath . "))
        );
    }

    /// C++ gets a syntax-only compile of the changed .cpp files, in a folder of a few sources.
    #[test]
    fn cpp_is_compiled_syntax_only_in_a_small_folder() {
        let small = checks(&names(&["adder.cpp", "adder.h", "CMakeLists.txt"]));
        let run = expand(&small[0], &names(&["adder.cpp", "adder.h"]), &[]).unwrap();
        assert!(run.contains("-fsyntax-only"), "{run}");
        assert!(run.ends_with(" 'adder.cpp'"), "{run}");
        let many: Vec<String> = (0..=LOOSE).map(|n| format!("a{n}.cpp")).collect();
        assert!(checks(&many).is_empty());
    }

    /// Django runs the apps of the tests near the change, pytest the test files, a small suite
    /// whole when none is near, a big one not at all.
    #[cfg(not(windows))]
    #[test]
    fn tests_near_the_changed_files_run() {
        let django = "python3 tests/runtests.py --parallel 1 {tests app}";
        let files = names(&[
            "django/db/models/enums.py",
            "tests/model_enums/tests.py",
            "tests/model_fields/test_charfield.py",
        ]);
        let changed = names(&["django/db/models/enums.py"]);
        let run = near(django, &files, &changed).unwrap();
        assert!(run.ends_with("--parallel 1 'model_enums'"), "{run}");
        let files = names(&[
            "sphinx/ext/autodoc/__init__.py",
            "tests/test_ext_autodoc.py",
            "tests/test_ext_autodoc_configs.py",
            "tests/test_util.py",
        ]);
        let changed = names(&["sphinx/ext/autodoc/__init__.py", "tests/test_new.py"]);
        let run = near("pytest {tests}", &files, &changed).unwrap();
        assert_eq!(
            run,
            "pytest 'tests/test_ext_autodoc.py' 'tests/test_ext_autodoc_configs.py' \
             'tests/test_new.py'"
        );
        let small = names(&["calc.py", "test_calc.py"]);
        assert_eq!(
            near("pytest {tests}", &small, &[]).as_deref(),
            Some("pytest")
        );
        let big: Vec<String> = (0..=SMALL_SUITE)
            .map(|n| format!("tests/test_{n}.py"))
            .collect();
        assert_eq!(near("pytest {tests}", &big, &names(&["README.md"])), None);
        assert_eq!(near("cargo test", &big, &[]).as_deref(), Some("cargo test"));
        // A module name of several words matches them in a row.
        let mut files = big.clone();
        files.extend(names(&[
            "pkg/date_utils.py",
            "tests/test_date_utils.py",
            "tests/test_utils_date.py",
        ]));
        let run = near("pytest {tests}", &files, &names(&["pkg/date_utils.py"]));
        assert_eq!(run.as_deref(), Some("pytest 'tests/test_date_utils.py'"));
    }

    /// Windows runs the whole suite: cmd.exe would keep the paths' quotes.
    #[cfg(windows)]
    #[test]
    fn windows_runs_the_whole_suite() {
        let files = names(&["calc.py", "test_calc.py"]);
        let run = near("pytest {tests}", &files, &names(&["calc.py"]));
        assert_eq!(run.as_deref(), Some("pytest"));
    }

    /// A user's command with braces of its own runs as it is.
    #[test]
    fn a_command_without_a_placeholder_is_as_given() {
        assert_eq!(
            expand("awk '{print}' x", &[], &[]).as_deref(),
            Some("awk '{print}' x")
        );
    }
}
