//! A code task on a toy repository: its worktree, its test command, its packet and the approval
//! that merges it.

use std::{fs, path::Path};

use larch_agent::{
    code::task::{Budget, CodeTask, Overrides, Scope, TestRun},
    skill::TypedResult,
};

/// Runs git for the test's own setup, isolated from the host's configuration.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .current_dir(dir)
        .env_clear()
        .envs(std::env::var_os("PATH").map(|p| ("PATH", p)))
        .env("HOME", dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    let shown = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "git {args:?}: {shown}");
    String::from_utf8(out.stdout).unwrap()
}

/// `root` as a repository whose one commit holds `files`.
fn repo(root: &Path, files: &[&str]) {
    git(root, &["init", "-q"]);
    for f in files {
        fs::write(root.join(f), "x\n").unwrap();
    }
    fs::write(root.join("README.md"), "toy\n").unwrap();
    git(root, &["add", "."]);
    git(root, &["commit", "-q", "-m", "start"]);
}

fn overrides() -> Overrides {
    Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: Budget {
            seconds: 60,
            tool_calls: 10,
            tokens: 1000,
            k: 1,
        },
    }
}

/// A task on a fresh toy repository and data folder, its attempt 0 worktree made.
fn started(allowed: Option<Vec<String>>) -> (tempfile::TempDir, CodeTask) {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir_all(checkout.join("src")).unwrap();
    repo(&checkout, &["src/lib.rs"]);
    let mut task = CodeTask::new(
        "Fix the parser!",
        &checkout,
        Overrides {
            allowed_paths: allowed,
            ..overrides()
        },
    )
    .unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    (tmp, task)
}

#[test]
fn the_worktree_starts_at_base_and_discard_removes_it_and_its_branch() {
    let (tmp, mut task) = started(None);
    let checkout = tmp.path().join("checkout");
    assert_eq!(
        task.base_sha,
        git(&checkout, &["rev-parse", "HEAD"]).trim_end()
    );
    assert_eq!(task.allowed_paths, ["."]);
    let wt = task.worktree.clone().unwrap();
    // Named as the project, as a build that takes its name from its folder needs.
    assert_eq!(
        wt,
        tmp.path().join("data/agent/code/fix-the-parser/0/checkout")
    );
    assert_eq!(task.branch.as_deref(), Some("iris/fix-the-parser-0"));
    assert_eq!(git(&wt, &["rev-parse", "HEAD"]).trim_end(), task.base_sha);
    assert_eq!(
        git(&wt, &["branch", "--show-current"]).trim_end(),
        "iris/fix-the-parser-0"
    );
    // The user's checkout stays on its branch, untouched.
    assert_eq!(
        git(&checkout, &["branch", "--show-current"]).trim_end(),
        "main"
    );
    assert_eq!(git(&checkout, &["status", "--porcelain"]), "");
    // One attempt at a time: a second would leave the first behind, untracked.
    let second = task.add_worktree(&tmp.path().join("data"), 1).unwrap_err();
    assert!(second.to_string().contains("still open"), "{second}");

    task.discard().unwrap();
    assert!(!wt.parent().unwrap().exists());
    assert_eq!(git(&checkout, &["branch", "--list", "iris/*"]), "");
    assert_eq!(
        git(&checkout, &["worktree", "list", "--porcelain"])
            .matches("worktree ")
            .count(),
        1
    );
    assert_eq!((&task.worktree, &task.branch), (&None, &None));
    task.discard().unwrap();
    // A folder inside a repository is not one: git never climbs to the checkout around it.
    let inner = CodeTask::new("x", &checkout.join("src"), overrides()).unwrap_err();
    assert!(
        inner
            .to_string()
            .contains("not the top of a git repository"),
        "{inner}"
    );
}

/// The state is the worktree's diff against the base, a new file in it too, through an index of
/// its own: `git diff` in the worktree still shows the change.
#[test]
fn the_state_is_the_diff_against_the_base_and_leaves_git_diff_alone() {
    let (_tmp, task) = started(None);
    let wt = task.worktree.clone().unwrap();
    let index = task.state_index().unwrap();
    assert_eq!(task.state(&index).unwrap(), None);
    fs::write(wt.join("src/lib.rs"), "y\n").unwrap();
    let changed = task.state(&index).unwrap();
    assert!(changed.is_some());
    assert!(
        git(&wt, &["diff"]).contains("+y"),
        "{}",
        git(&wt, &["diff"])
    );
    fs::write(wt.join("new.txt"), "n\n").unwrap();
    assert_ne!(task.state(&index).unwrap(), changed);
    fs::remove_file(wt.join("new.txt")).unwrap();
    assert_eq!(task.state(&index).unwrap(), changed);
    fs::write(wt.join("src/lib.rs"), "x\n").unwrap();
    assert_eq!(task.state(&index).unwrap(), None);
}

#[test]
fn the_test_command_comes_from_the_manifest_unless_given() {
    let cases = [
        ("Cargo.toml", Some("cargo test")),
        ("pyproject.toml", Some("pytest {tests}")),
        ("pytest.ini", Some("pytest {tests}")),
        ("package.json", Some("npm test")),
        ("Makefile", None),
    ];
    for (manifest, expected) in cases {
        let tmp = tempfile::tempdir().unwrap();
        repo(tmp.path(), &[manifest]);
        let task = CodeTask::new("x", tmp.path(), overrides()).unwrap();
        assert_eq!(task.test_cmd.as_deref(), expected, "{manifest}");
    }
    let tmp = tempfile::tempdir().unwrap();
    repo(tmp.path(), &["Cargo.toml"]);
    let given = Overrides {
        test_cmd: Some("make check".into()),
        ..overrides()
    };
    let task = CodeTask::new("x", tmp.path(), given).unwrap();
    assert_eq!(task.test_cmd.as_deref(), Some("make check"));
}

/// Repositories without a manifest name their tests by convention: Django's runner, pytest's
/// section in setup.cfg, a Makefile's test target.
#[test]
fn the_test_command_comes_from_the_repository_conventions() {
    let cases = [
        (
            "tests/runtests.py",
            "",
            "python3 tests/runtests.py --parallel 1 {tests app}",
        ),
        ("setup.cfg", "[flake8]\n[tool:pytest]\n", "pytest {tests}"),
        ("Makefile", "all:\n\ntest:\n\tpytest\n", "make test"),
    ];
    for (file, text, expected) in cases {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("tests")).unwrap();
        fs::write(tmp.path().join(file), text).unwrap();
        repo(tmp.path(), &[]);
        let task = CodeTask::new("x", tmp.path(), overrides()).unwrap();
        assert_eq!(task.test_cmd.as_deref(), Some(expected), "{file}");
    }
}

/// The tests near a change run alone; a failure the base has too is not the change's, and the
/// change is back in its file after the base run.
#[cfg(not(windows))]
#[test]
fn a_failure_the_base_has_too_is_not_the_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir_all(&checkout).unwrap();
    let test = "import unittest\nfrom calc import add, sub\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    \
                def test_sub(self):\n        self.assertEqual(sub(3, 2), 1)\n";
    fs::write(checkout.join("test_calc.py"), test).unwrap();
    fs::write(checkout.join("test_other.py"), "raise SystemExit(1)\n").unwrap();
    let calc = "def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a + b\n";
    fs::write(checkout.join("calc.py"), calc).unwrap();
    repo(&checkout, &[]);
    let given = Overrides {
        test_cmd: Some("python3 -m unittest {tests}".into()),
        ..overrides()
    };
    let mut task = CodeTask::new("x", &checkout, given).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let fixed = calc.replace("a - b", "a + b");
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("calc.py"), &fixed).unwrap();
    let (run, shown) = match task.test_until(None, true) {
        Err(e) if e.to_string().starts_with("the sandbox is unavailable") => {
            assert!(std::env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
            return println!("skipped: {e}");
        }
        ran => ran.unwrap(),
    };
    assert_eq!((run.passed, run.failed), (1, 0), "{shown}");
    assert_eq!(run.shared.len(), 1, "{shown}");
    assert!(
        shown.contains("still failing, as before your change: "),
        "{shown}"
    );
    assert_eq!(fs::read_to_string(wt.join("calc.py")).unwrap(), fixed);
}

/// A base run happens in a checkout of its own, in place and in a worktree alike: the folder's
/// bytes are the run's while the command runs and after it, a crash at any moment leaves them
/// so; the checkout holds the base's text of the changed files, none it lacks, and the rest as
/// the folder has it now.
#[test]
fn a_base_run_never_touches_the_folder() {
    for in_place in [true, false] {
        let tmp = tempfile::tempdir().unwrap();
        let checkout = tmp.path().join("checkout");
        fs::create_dir_all(checkout.join("app")).unwrap();
        fs::write(checkout.join("app/calc.py"), "v = 1\n").unwrap();
        fs::write(checkout.join("app/other.py"), "o = 1\n").unwrap();
        repo(&checkout, &[]);
        let data = tmp.path().join("data");
        let task = match in_place {
            true => {
                let mut task =
                    CodeTask::in_folder("x", &checkout.join("app"), &data, overrides()).unwrap();
                task.checkpoint().unwrap();
                task
            }
            false => {
                let mut task = CodeTask::new("x", &checkout, overrides()).unwrap();
                task.add_worktree(&data, 0).unwrap();
                task
            }
        };
        let dir = task.dir().unwrap().to_path_buf();
        let at = if in_place { "" } else { "app/" };
        fs::write(dir.join(format!("{at}calc.py")), "v = 2\n").unwrap();
        fs::write(dir.join(format!("{at}other.py")), "o = 2\n").unwrap();
        fs::write(dir.join(format!("{at}new.py")), "n = 1\n").unwrap();
        let files = [format!("{at}calc.py"), format!("{at}new.py")];
        let folder = |f: &str| fs::read_to_string(dir.join(format!("{at}{f}"))).ok();
        let now = || (folder("calc.py"), folder("other.py"), folder("new.py"));
        let was = now();
        let seen = task
            .at_base(&files, |base, kept| {
                assert_eq!(now(), was, "the folder changed during the base run");
                let read = |f: &str| fs::read_to_string(base.join(format!("{at}{f}"))).ok();
                assert_eq!(kept, [format!("{at}calc.py")]);
                (read("calc.py"), read("other.py"), read("new.py"))
            })
            .unwrap();
        assert_eq!(seen, (Some("v = 1\n".into()), Some("o = 2\n".into()), None));
        assert_eq!(now(), was, "in place: {in_place}");
    }
}

#[test]
fn the_packet_flags_a_path_outside_the_allowed_paths() {
    let (tmp, mut task) = started(Some(vec!["src".into()]));
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("src/lib.rs"), "fixed\n").unwrap();
    fs::write(wt.join("src/new.rs"), "new\n").unwrap();
    fs::write(wt.join("README.md"), "changed\n").unwrap();
    // A sibling whose name only starts with an allowed path is outside it.
    fs::write(wt.join("srcx.rs"), "no\n").unwrap();
    // An attribute the agent writes cannot turn its change into "Binary files differ".
    fs::write(wt.join(".gitattributes"), "*.rs -diff\n").unwrap();
    let packet = task.packet("Fix the parser", "Body.").unwrap();
    assert_eq!(
        packet.scope,
        [
            Scope {
                path: ".gitattributes".into(),
                inside: false
            },
            Scope {
                path: "README.md".into(),
                inside: false
            },
            Scope {
                path: "src/lib.rs".into(),
                inside: true
            },
            Scope {
                path: "src/new.rs".into(),
                inside: true
            },
            Scope {
                path: "srcx.rs".into(),
                inside: false
            },
        ]
    );
    assert!(packet.diff.contains("+fixed"), "{}", packet.diff);
    assert!(packet.diff.contains("+new"), "{}", packet.diff);
    // The worktree's commit is on the iris branch; the user's checkout never saw it.
    let checkout = tmp.path().join("checkout");
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "x\n"
    );
    assert!(!checkout.join("src/new.rs").exists());
    assert_eq!(git(&checkout, &["status", "--porcelain"]), "");
}

#[test]
fn the_packet_renders_and_travels_as_a_patch() {
    let (_tmp, mut task) = started(Some(vec!["src".into()]));
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("src/lib.rs"), "fixed\n").unwrap();
    fs::write(wt.join("README.md"), "changed\n").unwrap();
    let mut packet = task.packet("Fix the parser", "Parses again.").unwrap();
    packet.tests_before = Some(TestRun {
        passed: 3,
        failed: 1,
        failing: vec!["parse::empty".into()],
        shared: Vec::new(),
        tail: "1 failed".into(),
    });
    packet.commands_run = vec!["cargo test".into()];
    let shown = packet.render();
    for part in [
        "Fix the parser",
        "Parses again.",
        "tests before: 3 passed, 1 failed: parse::empty",
        "tests after: not run",
        "outside the allowed paths: README.md",
        "inside: src/lib.rs",
        "$ cargo test",
        "+fixed",
    ] {
        assert!(shown.contains(part), "{part:?} in\n{shown}");
    }
    let patch = TypedResult::Patch(packet);
    let json = serde_json::to_value(&patch).unwrap();
    assert_eq!(json["type"], "patch");
    assert_eq!(serde_json::from_value::<TypedResult>(json).unwrap(), patch);
}

/// A rename a command made (`mv`, there is no move tool) merges as a rename: the new file goes in
/// with the deletion, so the user's file is never lost.
#[test]
fn a_rename_by_command_keeps_the_new_file() {
    let (tmp, mut task) = started(None);
    let checkout = tmp.path().join("checkout");
    let wt = task.worktree.clone().unwrap();
    fs::rename(wt.join("src/lib.rs"), wt.join("src/core.rs")).unwrap();
    let packet = task.packet("Rename", "").unwrap();
    let paths: Vec<&str> = packet.scope.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(paths, ["src/core.rs", "src/lib.rs"], "{}", packet.diff);
    let approval = packet.approve(true).unwrap();
    task.apply(&packet, approval).unwrap();
    assert_eq!(
        fs::read_to_string(checkout.join("src/core.rs")).unwrap(),
        "x\n"
    );
    assert!(!checkout.join("src/lib.rs").exists());
}

#[test]
fn apply_merges_only_the_approved_packet() {
    let (tmp, mut task) = started(None);
    let checkout = tmp.path().join("checkout");
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("src/lib.rs"), "fixed\n").unwrap();
    let packet = task.packet("Fix the parser", "Body.").unwrap();

    // A human's no gives no approval, so apply cannot be called and the checkout stays.
    assert!(packet.approve(false).is_none());
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "x\n"
    );

    // An approval is for the diff the human saw: once the branch moves, it no longer applies.
    let stale = packet.approve(true).unwrap();
    fs::write(wt.join("src/lib.rs"), "fixed again\n").unwrap();
    let moved = task.packet("Fix the parser", "Body.").unwrap();
    let refused = task.apply(&moved, stale).unwrap_err().to_string();
    assert!(
        refused.contains("not the diff that was approved"),
        "{refused}"
    );
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "x\n"
    );

    let approval = moved.approve(true).unwrap();
    task.apply(&moved, approval).unwrap();
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "fixed again\n"
    );
    assert_eq!(
        git(&checkout, &["log", "-1", "--format=%s"]).trim_end(),
        "Fix the parser"
    );
    assert_eq!(git(&checkout, &["status", "--porcelain"]), "");
}

#[test]
fn a_merge_that_conflicts_is_undone_and_an_empty_title_still_commits() {
    let (tmp, mut task) = started(None);
    let checkout = tmp.path().join("checkout");
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("src/lib.rs"), "agent\n").unwrap();
    let packet = task.packet("", "").unwrap();
    assert!(packet.diff.contains("+agent"), "{}", packet.diff);
    // The user changed the same line meanwhile.
    fs::write(checkout.join("src/lib.rs"), "user\n").unwrap();
    git(&checkout, &["commit", "-q", "-am", "user"]);

    let approval = packet.approve(true).unwrap();
    task.apply(&packet, approval).unwrap_err();
    assert!(!checkout.join(".git/MERGE_HEAD").exists());
    assert_eq!(git(&checkout, &["status", "--porcelain"]), "");
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "user\n"
    );
}

#[cfg(unix)]
#[test]
fn the_repository_hooks_never_run() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir(&checkout).unwrap();
    repo(&checkout, &[]);
    let ran = tmp.path().join("ran");
    for hook in ["post-checkout", "pre-commit", "post-commit", "post-merge"] {
        let path = checkout.join(".git/hooks").join(hook);
        fs::write(
            &path,
            format!("#!/bin/sh\necho {hook} >> '{}'\n", ran.display()),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let mut task = CodeTask::new("x", &checkout, overrides()).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("README.md"), "changed\n").unwrap();
    let packet = task.packet("Change", "").unwrap();
    task.apply(&packet, packet.approve(true).unwrap()).unwrap();
    assert_eq!(
        fs::read_to_string(checkout.join("README.md")).unwrap(),
        "changed\n"
    );
    assert!(!ran.exists(), "{}", fs::read_to_string(&ran).unwrap());
}

#[test]
fn apply_leaves_the_users_own_merge_alone() {
    let (tmp, mut task) = started(None);
    let checkout = tmp.path().join("checkout");
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("README.md"), "agent\n").unwrap();
    let packet = task.packet("Agent", "").unwrap();
    // The user is part-way through a merge of their own, their resolution staged.
    git(&checkout, &["checkout", "-q", "-b", "side"]);
    fs::write(checkout.join("src/side.rs"), "side\n").unwrap();
    git(&checkout, &["add", "."]);
    git(&checkout, &["commit", "-q", "-m", "side"]);
    git(&checkout, &["checkout", "-q", "main"]);
    git(
        &checkout,
        &["merge", "-q", "--no-ff", "--no-commit", "side"],
    );
    fs::write(checkout.join("src/lib.rs"), "resolved\n").unwrap();
    git(&checkout, &["add", "src/lib.rs"]);

    let refused = task
        .apply(&packet, packet.approve(true).unwrap())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("in the middle of a merge"), "{refused}");
    assert!(checkout.join(".git/MERGE_HEAD").exists());
    assert_eq!(
        fs::read_to_string(checkout.join("src/lib.rs")).unwrap(),
        "resolved\n"
    );
    assert_eq!(
        fs::read_to_string(checkout.join("README.md")).unwrap(),
        "toy\n"
    );
}

#[test]
fn apply_refuses_a_checkout_that_left_the_base() {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir(&checkout).unwrap();
    repo(&checkout, &[]);
    // `release` stays behind a commit of main's that the packet never shows.
    git(&checkout, &["branch", "release"]);
    fs::write(checkout.join("feature.rs"), "unreleased\n").unwrap();
    git(&checkout, &["add", "."]);
    git(&checkout, &["commit", "-q", "-m", "feature"]);
    let mut task = CodeTask::new("x", &checkout, overrides()).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("README.md"), "agent\n").unwrap();
    let packet = task.packet("Agent", "").unwrap();
    git(&checkout, &["checkout", "-q", "release"]);
    let before = git(&checkout, &["rev-parse", "HEAD"]);

    let refused = task
        .apply(&packet, packet.approve(true).unwrap())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("does not descend from"), "{refused}");
    assert_eq!(git(&checkout, &["rev-parse", "HEAD"]), before);
    assert!(!checkout.join("feature.rs").exists());
}

/// What `in_folder`'s checkpoint and undo see in `dir`, any folder: a file it tracks, one it
/// does not track yet, one it ignores.
fn project(dir: &Path) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/kept.txt"), "kept\n").unwrap();
    fs::write(dir.join(".gitignore"), "*.log\n").unwrap();
}

/// The same task done in place, in a repository's top, in a folder of one and in a folder git
/// does not track: the checkpoint branch holds the folder as it was, untracked files too, and
/// leaves the user's HEAD, index and files alone; the diff is the run's, never an ignored file;
/// undo puts back each file the run changed, made or removed, and only those.
#[test]
fn in_place_checkpoints_any_folder_and_undo_puts_back_only_what_the_run_changed() {
    for kind in ["top", "sub", "plain"] {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let root = tmp.path().join("work");
        fs::create_dir_all(&root).unwrap();
        let dir = match kind {
            "plain" => root.clone(),
            "top" => root.clone(),
            _ => root.join("app"),
        };
        project(&dir);
        if kind != "plain" {
            repo(&root, &[]);
        }
        fs::write(dir.join("src/new.txt"), "untracked\n").unwrap();
        fs::write(dir.join("run.log"), "ignored\n").unwrap();
        let head = (kind != "plain").then(|| git(&root, &["rev-parse", "HEAD"]));
        let mut task = CodeTask::in_folder("Tidy the files", &dir, &data, overrides()).unwrap();
        task.checkpoint().unwrap();
        let folder = task.folder.clone().unwrap();
        let branch = folder.branch.clone().unwrap();
        assert!(
            branch.starts_with("larch/checkpoint-tidy-the-files-"),
            "{branch}"
        );
        match kind {
            "plain" => assert!(!dir.join(".git").exists(), "{kind}"),
            _ => {
                assert!(git(&root, &["branch", "--list", &branch]).contains(&branch));
                assert_eq!(
                    head.as_deref(),
                    Some(git(&root, &["rev-parse", "HEAD"]).as_str())
                );
                let staged = git(&root, &["diff", "--cached", "--name-only"]);
                assert_eq!(staged, "", "{kind}");
                assert!(git(&root, &["status", "--porcelain"]).contains("new.txt"));
            }
        }
        // The run: a change, a removal, a new file.
        fs::write(dir.join("src/kept.txt"), "changed\n").unwrap();
        fs::remove_file(dir.join("src/new.txt")).unwrap();
        fs::write(dir.join("src/made.txt"), "made\n").unwrap();
        assert!(task.changed().unwrap(), "{kind}");
        let packet = task.packet("Tidy", "").unwrap();
        let paths: Vec<&str> = packet.scope.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            ["src/kept.txt", "src/made.txt", "src/new.txt"],
            "{kind}"
        );
        assert!(!packet.diff.contains("run.log"), "{}", packet.diff);
        // The user's own change after the run, to a file it did not touch, stays.
        fs::write(dir.join("run.log"), "the user's\n").unwrap();
        let mut put = task.undo().unwrap();
        put.sort();
        assert_eq!(
            put,
            ["src/kept.txt", "src/made.txt", "src/new.txt"],
            "{kind}"
        );
        assert_eq!(
            fs::read_to_string(dir.join("src/kept.txt")).unwrap(),
            "kept\n"
        );
        assert_eq!(
            fs::read_to_string(dir.join("src/new.txt")).unwrap(),
            "untracked\n"
        );
        assert!(!dir.join("src/made.txt").exists(), "{kind}");
        assert_eq!(
            fs::read_to_string(dir.join("run.log")).unwrap(),
            "the user's\n"
        );
        assert!(!task.changed().unwrap(), "{kind}");
    }
}

/// Undo never loses what the user changed after the run: a file the run changed that changed
/// again is refused by name, and nothing is put back.
#[test]
fn undo_refuses_a_file_the_user_changed_after_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("work");
    project(&dir);
    let mut task =
        CodeTask::in_folder("Edit", &dir, &tmp.path().join("data"), overrides()).unwrap();
    task.checkpoint().unwrap();
    fs::write(dir.join("src/kept.txt"), "the run's\n").unwrap();
    fs::write(dir.join("src/made.txt"), "made\n").unwrap();
    task.packet("Edit", "").unwrap();
    fs::write(dir.join("src/kept.txt"), "the user's\n").unwrap();
    let refused = task.undo().unwrap_err().to_string();
    assert!(
        refused.contains("changed again since the run") && refused.contains("src/kept.txt"),
        "{refused}"
    );
    assert!(dir.join("src/made.txt").exists());
    assert_eq!(
        fs::read_to_string(dir.join("src/kept.txt")).unwrap(),
        "the user's\n"
    );
}

/// A command that removes the repository (`rm -rf .git`), or makes an empty one in its place,
/// does not take the undo with it: the checkpoint's copy under the data folder puts the folder
/// back, in a repository's top and in a folder of one.
#[test]
fn undo_survives_a_run_that_removes_the_repository() {
    for (kind, again) in [("top", false), ("sub", false), ("top", true)] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("work");
        fs::create_dir_all(&root).unwrap();
        let dir = if kind == "top" {
            root.clone()
        } else {
            root.join("app")
        };
        project(&dir);
        repo(&root, &[]);
        let data = tmp.path().join("data");
        let mut task = CodeTask::in_folder("Edit", &dir, &data, overrides()).unwrap();
        task.checkpoint().unwrap();
        fs::write(dir.join("src/kept.txt"), "the run's\n").unwrap();
        fs::write(dir.join("src/made.txt"), "made\n").unwrap();
        fs::remove_dir_all(root.join(".git")).unwrap();
        if again {
            git(&root, &["init", "-q"]);
        }
        let packet = task.packet("Edit", "").unwrap();
        assert!(packet.diff.contains("the run's"), "{kind}: {}", packet.diff);
        let mut put = task.undo().unwrap();
        put.sort();
        assert_eq!(put, ["src/kept.txt", "src/made.txt"], "{kind} {again}");
        assert_eq!(
            fs::read_to_string(dir.join("src/kept.txt")).unwrap(),
            "kept\n"
        );
        assert!(!dir.join("src/made.txt").exists(), "{kind} {again}");
    }
}

/// The backup sees the folder as the checkpoint did: a file the repository's own exclude
/// ignores stays out of the packet and survives undo, and CRLF bytes under core.autocrlf come
/// back as they were, after the run removed `.git`.
#[test]
fn a_backup_keeps_the_repository_s_excludes_and_line_endings() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("work");
    project(&dir);
    repo(&dir, &[]);
    git(&dir, &["config", "core.autocrlf", "true"]);
    fs::write(dir.join("crlf.txt"), "kept\r\n").unwrap();
    git(&dir, &["add", "crlf.txt"]);
    git(&dir, &["commit", "-q", "-m", "crlf"]);
    fs::write(dir.join(".git/info/exclude"), "secret.txt\n").unwrap();
    fs::write(dir.join("secret.txt"), "the user's\n").unwrap();
    let data = tmp.path().join("data");
    let mut task = CodeTask::in_folder("Edit", &dir, &data, overrides()).unwrap();
    task.checkpoint().unwrap();
    fs::write(dir.join("crlf.txt"), "changed\r\n").unwrap();
    fs::remove_dir_all(dir.join(".git")).unwrap();
    let packet = task.packet("Edit", "").unwrap();
    assert!(!packet.diff.contains("secret"), "{}", packet.diff);
    task.undo().unwrap();
    assert_eq!(fs::read(dir.join("secret.txt")).unwrap(), b"the user's\n");
    assert_eq!(fs::read(dir.join("crlf.txt")).unwrap(), b"kept\r\n");
}

/// The backup reads the folder's names and bytes as the checkpoint did: a configured excludes
/// file stays below info/exclude in precedence and apart from it, a case-insensitive ignore
/// still matches, and a configured attributes file still turns LF back into CRLF.
#[test]
fn a_backup_keeps_configured_ignore_and_attribute_files_and_ignore_case() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("work");
    project(&dir);
    repo(&dir, &[]);
    let rules = tmp.path().join("rules");
    fs::write(&rules, "!secret.txt\nother.txt\n").unwrap();
    let attributes = tmp.path().join("attributes");
    fs::write(&attributes, "*.crlf text eol=crlf\n").unwrap();
    git(
        &dir,
        &["config", "core.excludesFile", rules.to_str().unwrap()],
    );
    git(
        &dir,
        &[
            "config",
            "core.attributesFile",
            attributes.to_str().unwrap(),
        ],
    );
    git(&dir, &["config", "core.ignoreCase", "true"]);
    fs::write(dir.join("kept.crlf"), "kept\r\n").unwrap();
    git(&dir, &["add", "kept.crlf"]);
    git(&dir, &["commit", "-q", "-m", "crlf"]);
    // No final newline: the excludes file's first rule must not join this one.
    fs::write(dir.join(".git/info/exclude"), "secret.txt").unwrap();
    fs::write(dir.join(".gitignore"), "*.log\nloud.txt\n").unwrap();
    for name in ["secret.txt", "other.txt", "LOUD.txt"] {
        fs::write(dir.join(name), "the user's\n").unwrap();
    }
    let mut task =
        CodeTask::in_folder("Edit", &dir, &tmp.path().join("data"), overrides()).unwrap();
    task.checkpoint().unwrap();
    fs::write(dir.join("kept.crlf"), "changed\r\n").unwrap();
    fs::remove_dir_all(dir.join(".git")).unwrap();
    let packet = task.packet("Edit", "").unwrap();
    let paths: Vec<&str> = packet.scope.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(paths, ["kept.crlf"], "{}", packet.diff);
    task.undo().unwrap();
    for name in ["secret.txt", "other.txt", "LOUD.txt"] {
        assert_eq!(fs::read(dir.join(name)).unwrap(), b"the user's\n", "{name}");
    }
    assert_eq!(fs::read(dir.join("kept.crlf")).unwrap(), b"kept\r\n");
}

/// The folder's git still standing does not make it the one used: a gc that drops the run's
/// end state before the undo, or the old HEAD with the index removed, falls back or rebuilds
/// from the checkpoint.
#[test]
fn undo_survives_a_gc_of_the_folder_s_own_repository() {
    for case in ["gc after packet", "reset, gc, no index"] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("work");
        project(&dir);
        repo(&dir, &[]);
        let mut task =
            CodeTask::in_folder("Edit", &dir, &tmp.path().join("data"), overrides()).unwrap();
        task.checkpoint().unwrap();
        if case == "reset, gc, no index" {
            git(&dir, &["reset", "-q", "--hard", &task.base_sha]);
            git(&dir, &["reflog", "expire", "--expire=now", "--all"]);
            git(&dir, &["gc", "-q", "--prune=now"]);
            fs::remove_file(dir.join(".git/index")).unwrap();
        }
        fs::write(dir.join("src/kept.txt"), "the run's\n").unwrap();
        task.packet("Edit", "").unwrap();
        if case == "gc after packet" {
            git(&dir, &["gc", "-q", "--prune=now"]);
        }
        assert_eq!(task.undo().unwrap(), ["src/kept.txt"], "{case}");
        assert_eq!(
            fs::read_to_string(dir.join("src/kept.txt")).unwrap(),
            "kept\n",
            "{case}"
        );
    }
}

/// A url rewrite the run plants through an include the user's config already had never sends
/// the backup's copy to a repository of the run's, whose hook would run outside the sandbox.
#[cfg(unix)]
#[test]
fn a_planted_url_rewrite_never_picks_where_the_backup_goes() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("work");
    project(&dir);
    repo(&dir, &[]);
    git(&dir, &["config", "include.path", "../local-gitconfig"]);
    let data = tmp.path().join("data");
    let mut task = CodeTask::in_folder("Edit", &dir, &data, overrides()).unwrap();
    task.checkpoint().unwrap();
    let backup = task.folder.clone().unwrap().backup.unwrap();
    let sink = dir.join("sink.git");
    git(&dir, &["init", "-q", "--bare", sink.to_str().unwrap()]);
    let marker = tmp.path().join("ran");
    let hook = sink.join("hooks/pre-receive");
    fs::write(&hook, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
    let rewrite = format!("url.{}.insteadOf", sink.display());
    let file = dir.join("local-gitconfig");
    git(
        &dir,
        &[
            "config",
            "--file",
            file.to_str().unwrap(),
            &rewrite,
            backup.to_str().unwrap(),
        ],
    );
    fs::write(dir.join("src/kept.txt"), "the run's\n").unwrap();
    task.packet("Edit", "").unwrap();
    assert!(!marker.exists());
    fs::remove_dir_all(dir.join(".git")).unwrap();
    assert!(task.undo().unwrap().contains(&"src/kept.txt".to_string()));
    assert_eq!(
        fs::read_to_string(dir.join("src/kept.txt")).unwrap(),
        "kept\n"
    );
}

/// A folder done in place is never one whose checkpoint would take far too much or the agent's
/// own: a drive's top, the home folder, the data folder or one holding it.
#[test]
fn in_place_refuses_a_drive_top_the_home_folder_and_the_data_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let top = tmp.path().ancestors().last().unwrap().to_path_buf();
    let home = std::env::home_dir().unwrap();
    for (dir, why) in [
        (top.as_path(), "a drive's top"),
        (home.as_path(), "the home folder"),
        (data.as_path(), "Larch's data folder"),
        (tmp.path(), "Larch's data folder"),
    ] {
        let refused = CodeTask::in_folder("x", dir, &data, overrides())
            .unwrap_err()
            .to_string();
        assert!(refused.contains(why), "{}: {refused}", dir.display());
    }
}

/// A change that removes what a test imports, by a rename (calc.py to arith.py) or by making
/// its package a file, fails that test, and the base, which has it, does not: the failure is
/// the change's, never the base's too.
#[cfg(not(windows))]
#[test]
fn a_removed_module_is_back_in_the_base_run() {
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let add = "def add(a, b):\n    return a + b\n";
    for package in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let checkout = tmp.path().join("checkout");
        let module = if package {
            "calc/__init__.py"
        } else {
            "calc.py"
        };
        fs::create_dir_all(checkout.join("calc")).unwrap();
        fs::write(checkout.join(module), add).unwrap();
        fs::write(checkout.join("test_calc.py"), test).unwrap();
        repo(&checkout, &[]);
        let given = Overrides {
            test_cmd: Some("python3 -m unittest {tests}".into()),
            ..overrides()
        };
        let mut task = CodeTask::new("x", &checkout, given).unwrap();
        task.add_worktree(&tmp.path().join("data"), 0).unwrap();
        let wt = task.worktree.clone().unwrap();
        fs::rename(wt.join(module), wt.join("arith.py")).unwrap();
        if package {
            fs::remove_dir(wt.join("calc")).unwrap();
            fs::write(wt.join("calc"), "x\n").unwrap();
        }
        let (run, shown) = match task.test_until(None, true) {
            Err(e) if e.to_string().starts_with("the sandbox is unavailable") => {
                assert!(std::env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
                return println!("skipped: {e}");
            }
            ran => ran.unwrap(),
        };
        assert_eq!(run.failed, 1, "{shown}");
        assert!(!shown.contains("as before your change"), "{shown}");
        assert!(!shown.contains("could not compare"), "{shown}");
    }
}

/// A base checkout is gone after its run, one the run made a read-only folder in too, and a
/// stale one a killed run left in the tmp is swept by the next.
#[cfg(unix)]
#[test]
fn a_base_checkout_never_outlives_its_run() {
    use std::os::unix::fs::PermissionsExt;
    let stale = tempfile::Builder::new()
        .prefix("larch-base-")
        .tempdir()
        .unwrap()
        .keep();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(7200);
    fs::File::open(&stale).unwrap().set_modified(old).unwrap();
    let (_tmp, task) = started(None);
    let wt = task.worktree.clone().unwrap();
    fs::write(wt.join("src/lib.rs"), "changed\n").unwrap();
    let left = task
        .at_base(&["src/lib.rs".into()], |base, _| {
            let ro = base.join("ro");
            fs::create_dir(&ro).unwrap();
            fs::write(ro.join("f"), "x").unwrap();
            fs::set_permissions(&ro, fs::Permissions::from_mode(0o500)).unwrap();
            base.to_path_buf()
        })
        .unwrap();
    assert!(!left.exists(), "{}", left.display());
    assert!(!stale.exists(), "{}", stale.display());
}

/// A change of a test file alone is compared too: a test in it that fails on the base as well
/// (it needs the network) is not the change's.
#[cfg(not(windows))]
#[test]
fn a_change_of_tests_alone_is_compared_with_the_base() {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir_all(&checkout).unwrap();
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_net(self):\n        self.fail('needs the network')\n";
    fs::write(checkout.join("test_calc.py"), test).unwrap();
    fs::write(
        checkout.join("calc.py"),
        "def add(a, b):\n    return a + b\n",
    )
    .unwrap();
    repo(&checkout, &[]);
    let given = Overrides {
        test_cmd: Some("python3 -m unittest {tests}".into()),
        ..overrides()
    };
    let mut task = CodeTask::new("x", &checkout, given).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let wt = task.worktree.clone().unwrap();
    let more = "\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    fs::write(wt.join("test_calc.py"), format!("{test}{more}")).unwrap();
    let (run, shown) = match task.test_until(None, true) {
        Err(e) if e.to_string().starts_with("the sandbox is unavailable") => {
            assert!(std::env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
            return println!("skipped: {e}");
        }
        ran => ran.unwrap(),
    };
    assert_eq!((run.passed, run.failed), (1, 0), "{shown}");
    assert_eq!(run.shared.len(), 1, "{shown}");
    assert!(
        shown.contains("still failing, as before your change: "),
        "{shown}"
    );
}

/// A whole-suite command's failure with no run of the suite on the base to compare with could
/// not be compared: it neither holds a done nor passes it.
#[cfg(not(windows))]
#[test]
fn a_whole_suites_failure_without_a_base_run_is_not_compared() {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir_all(&checkout).unwrap();
    fs::write(checkout.join("test_net.py"), "raise SystemExit(1)\n").unwrap();
    repo(&checkout, &[]);
    let given = Overrides {
        test_cmd: Some("python3 test_net.py".into()),
        ..overrides()
    };
    let mut task = CodeTask::new("x", &checkout, given).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let (run, shown) = match task.test_until(None, true) {
        Err(e) if e.to_string().starts_with("the sandbox is unavailable") => {
            assert!(std::env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
            return println!("skipped: {e}");
        }
        ran => ran.unwrap(),
    };
    assert_eq!(run.failed, 1, "{shown}");
    assert!(shown.contains("could not compare with the base"), "{shown}");
}

/// Tests the model names run alone in the command's `{tests}`, quoted, instead of those near
/// the change, and their result is read as any run's.
#[cfg(not(windows))]
#[test]
fn named_tests_run_alone_and_are_parsed() {
    let tmp = tempfile::tempdir().unwrap();
    let checkout = tmp.path().join("checkout");
    fs::create_dir_all(checkout.join("tests")).unwrap();
    let pass =
        "import unittest\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n";
    fs::write(checkout.join("tests/test_x.py"), pass).unwrap();
    fs::write(checkout.join("tests/test_y.py"), "raise SystemExit(1)\n").unwrap();
    repo(&checkout, &[]);
    let given = Overrides {
        test_cmd: Some("python3 -m unittest {tests}".into()),
        ..overrides()
    };
    let mut task = CodeTask::new("x", &checkout, given).unwrap();
    task.add_worktree(&tmp.path().join("data"), 0).unwrap();
    let named = ["tests/test_x.py".to_string()];
    let (run, shown) = match task.test_these(None, true, &named) {
        Err(e) if e.to_string().starts_with("the sandbox is unavailable") => {
            assert!(std::env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
            return println!("skipped: {e}");
        }
        ran => ran.unwrap(),
    };
    assert_eq!((run.passed, run.failed), (1, 0), "{shown}");
}
