//! Code mode's view, search and edit, on folders the tests create.

use std::{fs, path::Path, process::Command};

use larch::toolcall::{Family, Mode, grammar};
use larch_agent::tool::{Permission, Registry, ToolCtx, ToolError};
use serde_json::{Value, json};

/// One allowed folder `root` inside a temp dir, with `secret.txt` beside it.
struct Fixture {
    tmp: tempfile::TempDir,
    ctx: ToolCtx,
}

impl Fixture {
    fn new() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::write(tmp.path().join("secret.txt"), "outside\n").unwrap();
        let ctx = ToolCtx {
            allowed: vec![root],
            data_dir: tmp.path().join("data"),
            source: "test".into(),
        };
        Fixture { tmp, ctx }
    }

    fn root(&self) -> &Path {
        &self.ctx.allowed[0]
    }

    fn write(&self, name: &str, text: &str) {
        let path = self.root().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.root().join(name)).unwrap()
    }

    fn call(&self, name: &str, args: Value) -> Result<String, ToolError> {
        Registry::builtin()
            .get(name)
            .unwrap()
            .call(&args, &self.ctx)
    }
}

fn invalid(r: Result<String, ToolError>) -> String {
    match r {
        Err(ToolError::Invalid(sentence)) => sentence,
        other => panic!("not invalid: {other:?}"),
    }
}

/// A view that starts inside a function shows it from its first line when it fits, so a big file
/// is not read in windows that cut its functions; one too big to fit starts where asked.
#[test]
fn a_view_inside_a_block_shows_the_block_whole() {
    let f = Fixture::new();
    let body: String = (0..20).map(|n| format!("    x{n} = {n}\n")).collect();
    let huge: String = (0..1000).map(|n| format!("    y{n} = {n}\n")).collect();
    f.write(
        "m.py",
        &format!("import os\n\ndef small():\n{body}\ndef huge():\n{huge}"),
    );
    let got = f
        .call("view", json!({"path": "m.py", "start": 12}))
        .unwrap();
    assert!(got.starts_with("3: def small():\n4:     x0 = 0"), "{got}");
    let got = f
        .call("view", json!({"path": "m.py", "start": 500}))
        .unwrap();
    assert!(got.starts_with("500:     y"), "{got}");
}

#[test]
fn view_numbers_a_window_of_lines_and_caps_it() {
    let f = Fixture::new();
    let text: String = (1..=1000).map(|n| format!("line {n}\n")).collect();
    f.write("big.txt", &text);
    // A window is the 400 lines from start, or from line 1 when the call gives none.
    for (args, first) in [
        (json!({"path": "big.txt", "start": 10}), 10),
        (json!({"path": "big.txt"}), 1),
    ] {
        let got = f.call("view", args).unwrap();
        let lines: Vec<_> = got.lines().collect();
        assert_eq!(lines.len(), 401, "{got}");
        assert_eq!(lines[0], format!("{first}: line {first}"));
        assert_eq!(lines[399], format!("{0}: line {0}", first + 399));
        assert_eq!(
            lines[400],
            format!("... more lines; view from start {}", first + 400)
        );
    }
    // The last window ends without a tail.
    let got = f
        .call("view", json!({"path": "big.txt", "start": 999}))
        .unwrap();
    assert_eq!(got, "999: line 999\n1000: line 1000");
    let past = invalid(f.call("view", json!({"path": "big.txt", "start": 1001})));
    assert!(past.contains("past the end"), "{past}");
    let folder = invalid(f.call("view", json!({"path": "."})));
    assert!(folder.contains("is a folder"), "{folder}");
    // Lines that fill the budget (4000 tokens, 4 bytes each) first stop there, and the hint
    // starts at the first line not shown.
    let long = "y".repeat(70);
    let text: String = (1..=1000).map(|n| format!("{n} {long}\n")).collect();
    f.write("wide.txt", &text);
    let got = f.call("view", json!({"path": "wide.txt"})).unwrap();
    assert!(got.len() <= 16000, "{}", got.len());
    let (shown, hint) = got.rsplit_once('\n').unwrap();
    let last: u64 = shown
        .lines()
        .last()
        .unwrap()
        .split(':')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(last < 400, "{last}");
    assert_eq!(
        hint,
        format!("... more lines; view from start {}", last + 1)
    );
}

#[test]
fn search_finds_lines_and_skips_what_gitignore_ignores() {
    let f = Fixture::new();
    // `foo[ab` never closes its bracket, so git ignores nothing with it.
    f.write(
        ".gitignore",
        "target/\n*.log\n/top.txt\n!keep.log\nfoo[ab\n",
    );
    f.write("fooa", "needle unclosed\n");
    f.write("src/main.rs", "fn main() {\n    needle();\n}\n");
    f.write("src/top.txt", "needle in a nested top.txt\n");
    f.write("top.txt", "needle at the top\n");
    f.write("target/debug/out.rs", "needle\n");
    f.write("run.log", "needle\n");
    f.write("keep.log", "needle kept\n");
    f.write("sub/.gitignore", "deep.rs\n");
    f.write("sub/deep.rs", "needle\n");
    f.write("sub/shallow.rs", "needle\n");
    f.write(".git/config", "needle\n");
    // A rule naming a Latin-1 file leaves the file's other rules working.
    fs::create_dir(f.root().join("vendor")).unwrap();
    fs::write(
        f.root().join("vendor/.gitignore"),
        b"caf\xe9.txt\nskip.txt\n",
    )
    .unwrap();
    f.write("vendor/skip.txt", "needle\n");
    f.write("vendor/a.txt", "needle\n");
    let got = f.call("search", json!({"pattern": "need+le"})).unwrap();
    assert_eq!(
        got,
        "fooa:1: needle unclosed\n\
         keep.log:1: needle kept\n\
         src/main.rs:2:     needle();\n\
         src/top.txt:1: needle in a nested top.txt\n\
         sub/shallow.rs:1: needle\n\
         vendor/a.txt:1: needle"
    );
    // A folder's search keeps the rules of the .gitignore files above it.
    let got = f
        .call("search", json!({"pattern": "needle", "path": "sub"}))
        .unwrap();
    assert_eq!(got, "sub/shallow.rs:1: needle");
    // A JavaScript literal is its pattern, and its i flag ignores case.
    let got = f.call("search", json!({"pattern": "/NEED+LE/gi", "path": "sub"}));
    assert_eq!(got.unwrap(), "sub/shallow.rs:1: needle");
    let none = f.call("search", json!({"pattern": "absent"})).unwrap();
    assert_eq!(none, "no line matches 'absent'");
    let bad = invalid(f.call("search", json!({"pattern": "("})));
    assert!(bad.contains("regex"), "{bad}");
}

#[test]
fn search_stops_at_its_cap() {
    let f = Fixture::new();
    let text: String = (1..=500).map(|n| format!("hit {n}\n")).collect();
    f.write("many.txt", &text);
    let got = f.call("search", json!({"pattern": "hit"})).unwrap();
    let lines: Vec<_> = got.lines().collect();
    assert_eq!(lines.len(), 101, "{got}");
    assert_eq!(lines[99], "many.txt:100: hit 100");
    assert!(lines[100].starts_with("... more matches"), "{got}");
    // Long hits fill the result budget (2000 tokens, 4 bytes each) first; the hint still shows.
    let long = "x".repeat(200);
    let text: String = (1..=100).map(|n| format!("hit {n} {long}\n")).collect();
    f.write("many.txt", &text);
    let got = f.call("search", json!({"pattern": "hit"})).unwrap();
    assert!(got.len() <= 8000, "{}", got.len());
    assert!(
        got.ends_with("... more matches; narrow the pattern or the path"),
        "{got}"
    );
}

#[cfg(unix)]
#[test]
fn search_refuses_a_fifo_instead_of_blocking_on_it() {
    let f = Fixture::new();
    let fifo = f.root().join("pipe");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let got = invalid(f.call("search", json!({"pattern": "x", "path": "pipe"})));
    assert!(got.contains("not a file"), "{got}");
    // Inside a folder it is skipped, as a link is.
    f.write("a.txt", "x\n");
    assert_eq!(
        f.call("search", json!({"pattern": "x"})).unwrap(),
        "a.txt:1: x"
    );
}

#[test]
fn edit_replaces_one_unique_match_and_refuses_none_or_many() {
    let f = Fixture::new();
    f.write("notes.txt", "alpha\nbeta\nbeta\n");
    let got = f
        .call(
            "edit",
            json!({"path": "notes.txt", "old_text": "alpha", "new_text": "gamma"}),
        )
        .unwrap();
    assert_eq!(got, "replaced 1 match in notes.txt at line 1");
    assert_eq!(f.read("notes.txt"), "gamma\nbeta\nbeta\n");
    let many = invalid(f.call(
        "edit",
        json!({"path": "notes.txt", "old_text": "beta", "new_text": "x"}),
    ));
    assert!(many.contains("2 matches"), "{many}");
    let none = invalid(f.call(
        "edit",
        json!({"path": "notes.txt", "old_text": "delta", "new_text": "x"}),
    ));
    assert!(none.contains("0 matches"), "{none}");
    // Overlapping matches count: `}}` is twice in `}}}`.
    f.write("braces.txt", "}}}\n");
    let overlap = invalid(f.call(
        "edit",
        json!({"path": "braces.txt", "old_text": "}}", "new_text": "}"}),
    ));
    assert!(overlap.contains("2 matches"), "{overlap}");
    // An edit that changes nothing is refused: code mode reads a done edit as a change.
    let same = invalid(f.call(
        "edit",
        json!({"path": "notes.txt", "old_text": "gamma", "new_text": "gamma"}),
    ));
    assert!(same.contains("the text is already in the file"), "{same}");
    // So is one made already, sent again: its text is gone and its new text there.
    let again = invalid(f.call(
        "edit",
        json!({"path": "notes.txt", "old_text": "alpha", "new_text": "gamma"}),
    ));
    assert!(
        again.starts_with("the change is already in the file") && again.contains("no progress"),
        "{again}"
    );
    assert_eq!(f.read("notes.txt"), "gamma\nbeta\nbeta\n");
    // The same checks run before the user is asked.
    let tool = Registry::builtin();
    let edit = tool.get("edit").unwrap();
    let args = json!({"path": "notes.txt", "old_text": "beta", "new_text": "x"});
    assert!(matches!(
        edit.prepare(&args, &f.ctx),
        Err(ToolError::Invalid(_))
    ));
}

/// An edit whose old_text is gone shows the lines around where its first line still is, not the
/// file's start.
#[test]
fn a_stale_edit_shows_the_lines_around_its_first_line() {
    let f = Fixture::new();
    // Line 3 holds the first line too, but not as a whole line.
    let text: String = (1..=80)
        .map(|n| match n {
            3 => "# x60 = 60 is below\n".into(),
            n => format!("x{n} = {n}\n"),
        })
        .collect();
    f.write("long.py", &text);
    let args = json!({"path": "long.py", "old_text": "  x60 = 60\nx61 = 0", "new_text": "x"});
    let stale = invalid(f.call("edit", args));
    assert!(stale.contains("now:\n40: x40 = 40\n"), "{stale}");
    assert!(stale.contains("\n60: x60 = 60\n"), "{stale}");
    assert!(
        stale.ends_with("... more lines; view from start 80"),
        "{stale}"
    );
}

fn python() -> bool {
    ["python3", "python"]
        .iter()
        // Ran, not just spawned: Windows' Store alias `python3` spawns and runs nothing.
        .any(|p| (Command::new(p).arg("--version").output()).is_ok_and(|o| o.status.success()))
}

#[test]
fn an_edit_that_breaks_py_compile_is_rolled_back_with_the_error() {
    if !python() {
        panic!("python is not installed; this gate needs it");
    }
    let f = Fixture::new();
    let text: String = (1..=20).map(|n| format!("x{n} = {n}\n")).collect();
    f.write("mod.py", &text);
    let err = invalid(f.call(
        "edit",
        json!({"path": "mod.py", "old_text": "x12 = 12", "new_text": "x12 = (12"}),
    ));
    assert_eq!(f.read("mod.py"), text, "not rolled back");
    assert!(err.contains("rolled back"), "{err}");
    assert!(err.contains("SyntaxError"), "{err}");
    // The ten lines around the error, numbered. Python 3.10+ places it at line 12, the
    // macOS system Python 3.9 at line 13: either window holds the edit.
    assert!(err.contains("12: x12 = (12"), "{err}");
    let numbered = |l: &&str| {
        l.split_once(": x")
            .is_some_and(|(n, _)| n.parse::<u32>().is_ok())
    };
    assert_eq!(err.lines().filter(numbered).count(), 10, "{err}");
    // No .pyc lands beside the file.
    assert!(!f.root().join("__pycache__").exists());
    // A file that already failed may stay failing.
    f.write("broken.py", "def f(:\n    pass\n");
    f.call(
        "edit",
        json!({"path": "broken.py", "old_text": "pass", "new_text": "return 1"}),
    )
    .unwrap();
    assert_eq!(f.read("broken.py"), "def f(:\n    return 1\n");
}

#[test]
fn an_edit_that_breaks_json_is_rolled_back_and_other_files_are_not_checked() {
    let f = Fixture::new();
    f.write("a.json", "{\n  \"k\": 1\n}\n");
    let err = invalid(f.call(
        "edit",
        json!({"path": "a.json", "old_text": "1", "new_text": "1,"}),
    ));
    assert!(err.contains("rolled back"), "{err}");
    assert_eq!(f.read("a.json"), "{\n  \"k\": 1\n}\n");
    // A minified file's one line is cut in the error, not returned whole.
    let min = format!("[{}0]", "1,".repeat(300_000));
    f.write("min.json", &min);
    let err = invalid(f.call(
        "edit",
        json!({"path": "min.json", "old_text": "0]", "new_text": "0,]"}),
    ));
    assert!(
        err.contains("rolled back") && err.len() < 4000,
        "{}",
        err.len()
    );
    assert_eq!(f.read("min.json"), min);
    f.write("a.txt", "{\n");
    f.call(
        "edit",
        json!({"path": "a.txt", "old_text": "{", "new_text": "{{"}),
    )
    .unwrap();
}

#[test]
fn paths_outside_the_allowed_folder_are_denied_for_all_three_tools() {
    let f = Fixture::new();
    let secret = f.tmp.path().join("secret.txt");
    let calls = [
        ("view", json!({"path": "../secret.txt"})),
        ("view", json!({"path": secret.to_str().unwrap()})),
        ("search", json!({"pattern": "outside", "path": ".."})),
        (
            "search",
            json!({"pattern": "outside", "path": "../secret.txt"}),
        ),
        (
            "edit",
            json!({"path": "../secret.txt", "old_text": "outside", "new_text": "in"}),
        ),
        (
            "replace_symbol",
            json!({"path": "../secret.txt", "name": "x", "new_text": "in"}),
        ),
    ];
    for (name, args) in calls {
        let got = f.call(name, args.clone());
        assert!(
            matches!(got, Err(ToolError::Denied(_))),
            "{name} {args}: {got:?}"
        );
    }
    assert_eq!(fs::read_to_string(&secret).unwrap(), "outside\n");
}

#[test]
fn each_tools_schema_passes_the_grammar() {
    let registry = Registry::builtin();
    for (name, permission) in [
        ("view", Permission::Read),
        ("search", Permission::Read),
        ("edit", Permission::WriteLocal),
        ("replace_symbol", Permission::WriteLocal),
    ] {
        let spec = registry.get(name).unwrap().spec();
        assert_eq!(spec.permission, permission, "{name}");
        assert!(spec.description.ends_with('.'), "{name}");
        jsonschema::validator_for(&spec.schema).unwrap();
        let tools = [spec.function()];
        for (family, xml) in [
            (Family::Qwen3, false),
            (Family::Qwen3, true),
            (Family::Llama3, false),
            (Family::Mistral, false),
        ] {
            grammar(family, &tools, xml, Mode::Required)
                .unwrap_or_else(|e| panic!("{name} {family:?} xml={xml}: {e}"));
        }
    }
}

#[test]
fn rustfmt_gates_an_edit_on_parse_errors_not_on_formatting() {
    let f = Fixture::new();
    f.write("lib.rs", "mod elsewhere;\nfn f() {}\n");
    // Badly formatted, and a module whose file is not there: neither is a parse error.
    f.call(
        "edit",
        json!({"path": "lib.rs", "old_text": "fn f() {}", "new_text": "fn f(){let x=1;}"}),
    )
    .unwrap();
    let err = invalid(f.call(
        "edit",
        json!({"path": "lib.rs", "old_text": "let x=1;", "new_text": "let x=(1;"}),
    ));
    assert!(
        err.contains("rolled back") && err.contains("2: fn f()"),
        "{err}"
    );
    assert_eq!(f.read("lib.rs"), "mod elsewhere;\nfn f(){let x=1;}\n");
}

#[test]
fn rustfmt_parses_each_file_in_its_crates_edition() {
    let f = Fixture::new();
    // 2024 reserves `gen`; 2021 refuses let chains. Each crate's own edition takes its code.
    f.write(
        "old/Cargo.toml",
        "[package]\nname = \"old\"\nedition = \"2021\"\n",
    );
    f.write("old/src/lib.rs", "fn f() {}\n");
    f.write("Cargo.toml", "[workspace.package]\nedition = \"2024\"\n");
    f.write(
        "new/Cargo.toml",
        "[package]\nname = \"new\"\nedition.workspace = true\n",
    );
    f.write("new/src/lib.rs", "fn f() {}\n");
    // The inline-table form inherits too; 2015 would refuse `async fn`.
    f.write(
        "inline/Cargo.toml",
        "[package]\nname = \"inline\"\nedition = { workspace = true }\n",
    );
    f.write("inline/src/lib.rs", "fn f() {}\n");
    f.call(
        "edit",
        json!({"path": "inline/src/lib.rs", "old_text": "fn f() {}", "new_text": "async fn f() {}"}),
    )
    .unwrap();
    f.call(
        "edit",
        json!({"path": "old/src/lib.rs", "old_text": "{}",
               "new_text": "{ let x: u32 = rng.gen(); }"}),
    )
    .unwrap();
    let chain = "{ if let Some(x) = a && let Some(y) = b { let _ = (x, y); } }";
    f.call(
        "edit",
        json!({"path": "new/src/lib.rs", "old_text": "{}", "new_text": chain}),
    )
    .unwrap();
    // The other way round, each is a parse error the file did not have, and is undone.
    let gen_in_2024 = f.call(
        "edit",
        json!({"path": "new/src/lib.rs", "old_text": "let _ = (x, y);",
               "new_text": "let _ = rng.gen();"}),
    );
    assert!(invalid(gen_in_2024).contains("rolled back"));
    let chain_in_2021 = f.call(
        "edit",
        json!({"path": "old/src/lib.rs", "old_text": "{ let x: u32 = rng.gen(); }",
               "new_text": chain}),
    );
    assert!(invalid(chain_in_2021).contains("rolled back"));
}

const GAME_JS: &str = "class Board {
  constructor() {
    this.cells = [];
  }

  dropDisc(col) {
    return '}';
  }
}

function render() {}
";

#[test]
fn replace_symbol_replaces_a_method_by_its_qualified_name() {
    let f = Fixture::new();
    f.write("game.js", GAME_JS);
    // No trailing newline: the symbol's lines are replaced whole, and the next line stays its own.
    let new = "  dropDisc(col) {\n    this.cells.push(col);\n    return col;\n  }";
    let got = f
        .call(
            "replace_symbol",
            json!({"path": "game.js", "name": "Board.dropDisc", "new_text": new}),
        )
        .unwrap();
    assert_eq!(
        got,
        "replaced method Board.dropDisc in game.js, lines 6-8, with lines 6-9"
    );
    assert_eq!(
        f.read("game.js"),
        GAME_JS.replace(
            "  dropDisc(col) {\n    return '}';\n  }\n",
            &format!("{new}\n")
        )
    );
    // A name it does not hold lists what it does, as the names to give.
    let none = invalid(f.call(
        "replace_symbol",
        json!({"path": "game.js", "name": "Board.drop", "new_text": "x"}),
    ));
    assert!(
        none.contains("no symbol 'Board.drop' in 'game.js'"),
        "{none}"
    );
    assert!(
        none.contains("method Board.dropDisc 6-9\nfunction render 12-12"),
        "{none}"
    );
    // Checked before the user is asked, as edit's match is.
    let tools = Registry::builtin();
    let args = json!({"path": "game.js", "name": "nope", "new_text": "x"});
    let refused = tools.get("replace_symbol").unwrap().prepare(&args, &f.ctx);
    assert!(matches!(refused, Err(ToolError::Invalid(_))), "{refused:?}");
}

#[test]
fn replace_symbol_lists_the_candidates_of_an_ambiguous_name() {
    let f = Fixture::new();
    let rs = "pub struct Board {\n    cells: Vec<u8>,\n}\n\nimpl Board {\n    fn new() -> Self {\n        Board { cells: vec![] }\n    }\n}\n";
    f.write("board.rs", rs);
    let both = invalid(f.call(
        "replace_symbol",
        json!({"path": "board.rs", "name": "Board", "new_text": "x"}),
    ));
    assert!(both.contains("'Board' names 2 symbols"), "{both}");
    assert!(both.contains("struct Board 1-3\nimpl Board 5-9"), "{both}");
    assert_eq!(f.read("board.rs"), rs);
    // A name as the list gives it picks one.
    f.call(
        "replace_symbol",
        json!({"path": "board.rs", "name": "impl Board::new",
               "new_text": "    fn new() -> Self {\n        Board { cells: Vec::new() }\n    }\n"}),
    )
    .unwrap();
    assert!(f.read("board.rs").contains("Board { cells: Vec::new() }"));
}

#[test]
fn a_replace_symbol_that_breaks_py_compile_is_rolled_back() {
    if !python() {
        panic!("python is not installed; this gate needs it");
    }
    let f = Fixture::new();
    let py =
        "class Board:\n    def drop(self, col):\n        return col\n\n\ndef main():\n    pass\n";
    f.write("game.py", py);
    let err = invalid(f.call(
        "replace_symbol",
        json!({"path": "game.py", "name": "Board.drop", "new_text": "    def drop(self, col:\n        return col\n"}),
    ));
    assert!(err.contains("rolled back"), "{err}");
    assert_eq!(f.read("game.py"), py);
}

#[test]
fn view_shows_a_whole_symbol_by_name() {
    let f = Fixture::new();
    f.write("game.js", GAME_JS);
    let got = f
        .call("view", json!({"path": "game.js", "symbol": "dropDisc"}))
        .unwrap();
    assert_eq!(got, "6:   dropDisc(col) {\n7:     return '}';\n8:   }");
    let none = invalid(f.call("view", json!({"path": "game.js", "symbol": "nope"})));
    assert!(none.contains("class Board 1-9"), "{none}");
}

#[test]
fn a_stale_edit_names_replace_symbol_for_a_file_with_symbols() {
    let f = Fixture::new();
    f.write("game.js", GAME_JS);
    let stale = invalid(f.call(
        "edit",
        json!({"path": "game.js", "old_text": "dropDisc(c) {", "new_text": "x"}),
    ));
    assert!(stale.contains("replace_symbol"), "{stale}");
}

#[test]
fn replace_symbol_keeps_the_code_that_shares_its_lines() {
    let f = Fixture::new();
    let cases = [
        (
            "index.html",
            "<p>Score: <span id=\"score\">0</span>!</p>\n<p>end</p>\n",
            "score",
            "<span id=\"score\">1</span>\n",
            "<p>Score: <span id=\"score\">1</span>!</p>\n<p>end</p>\n",
        ),
        (
            "style.css",
            ".a{color:red} .b{color:blue}\n",
            ".b",
            ".b{color:green}",
            ".a{color:red} .b{color:green}\n",
        ),
        (
            "lib.rs",
            "struct A; struct B;\n",
            "A",
            "struct A(u8);\n",
            "struct A(u8); struct B;\n",
        ),
    ];
    for (path, text, name, new, after) in cases {
        f.write(path, text);
        let args = json!({"path": path, "name": name, "new_text": new});
        f.call("replace_symbol", args).unwrap();
        assert_eq!(f.read(path), after, "{path}");
    }
}

#[test]
fn replace_symbol_takes_decorators_with_the_symbol() {
    let f = Fixture::new();
    f.write(
        "web.py",
        "import app\n\n\n@app.route('/')\ndef index():\n    return 1\n",
    );
    let shown = f
        .call("view", json!({"path": "web.py", "symbol": "index"}))
        .unwrap();
    assert!(
        shown.starts_with("4: @app.route('/')\n5: def index():"),
        "{shown}"
    );
    let new = "@app.route('/')\ndef index():\n    return 2\n";
    f.call(
        "replace_symbol",
        json!({"path": "web.py", "name": "index", "new_text": new}),
    )
    .unwrap();
    assert_eq!(f.read("web.py"), format!("import app\n\n\n{new}"));
}

/// lookup ranks the folder's named blocks by the query and shows the best whole, then the
/// blocks it uses: the element whose id its script reads and the rule of the class it sets.
#[test]
fn lookup_shows_the_best_block_whole_with_what_it_uses() {
    let f = Fixture::new();
    f.write(
        "index.html",
        "<html><body>\n<div id=\"board\" class=\"grid\"></div>\n<p id=\"status\">hi</p>\n</body></html>\n",
    );
    f.write(
        "style.css",
        ".grid { display: grid; }\n.cell { width: 40px; }\n.hidden { display: none; }\n",
    );
    f.write(
        "script.js",
        "function score(n) {\n  return n * 2;\n}\n\nfunction dropDisc(col) {\n  const board = \
         document.getElementById('board');\n  const cell = document.createElement('div');\n  \
         cell.classList.add('cell');\n  board.appendChild(cell);\n}\n\nfunction reset() {\n  \
         return 0;\n}\n\nfunction other() {\n  return 1;\n}\n\nfunction more() {\n  return \
         2;\n}\n\nfunction last() {\n  return 3;\n}\n",
    );
    let got = f.call("lookup", json!({"query": "drop disc"})).unwrap();
    assert!(
        got.starts_with("script.js, function dropDisc, lines 5-10:\n5: function dropDisc(col) {\n"),
        "{got}"
    );
    assert!(
        got.contains("used: index.html, element board, lines 2-2:\n2: <div id=\"board\""),
        "{got}"
    );
    assert!(
        got.contains("used: style.css, rule .cell, lines 2-2:\n2: .cell { width: 40px; }"),
        "{got}"
    );
    assert!(!got.contains(".hidden"), "{got}");
    let none = f.call("lookup", json!({"query": "zebra"})).unwrap();
    assert_eq!(
        none,
        "no block matches 'zebra': search takes a regex over every line"
    );
}

/// lookup reads Go too.
#[test]
fn lookup_finds_go() {
    let f = Fixture::new();
    f.write(
        "main.go",
        "package main\n\nfunc DropDisc(col int) int {\n\treturn col\n}\n",
    );
    let got = f.call("lookup", json!({"query": "drop disc"})).unwrap();
    assert!(
        got.starts_with("main.go, func DropDisc, lines 3-5:"),
        "{got}"
    );
}

/// With a block or two the query still decides: one it does not match is no hit.
#[test]
fn lookup_ranks_even_a_few_blocks() {
    let f = Fixture::new();
    f.write("a.js", "function alpha() {\n  return 1;\n}\n");
    let got = f.call("lookup", json!({"query": "zebra"})).unwrap();
    assert_eq!(
        got,
        "no block matches 'zebra': search takes a regex over every line"
    );
    let none = invalid(f.call("lookup", json!({"query": "!!!"})));
    assert!(none.starts_with("query has no words"), "{none}");
}

/// A call on a name uses the block of that name's type or module, not every block that shares
/// the method's name: `cache.get` is `Cache.get`, never the eight `Client.get`.
#[test]
fn lookup_uses_the_block_a_qualified_call_names() {
    let f = Fixture::new();
    f.write(
        "users.py",
        "def load_user(cache, key):\n    return cache.get(key)\n",
    );
    f.write(
        "store.py",
        "class Cache:\n    def get(self, key):\n        return key\n",
    );
    for n in 0..8 {
        f.write(
            &format!("client{n}.py"),
            "class Client:\n    def get(self, url):\n        return url\n",
        );
    }
    let got = f.call("lookup", json!({"query": "load user"})).unwrap();
    assert!(
        got.contains("used: store.py, def Cache.get, lines 2-3:"),
        "{got}"
    );
    assert!(!got.contains("used: client"), "{got}");
}

/// The index is kept between lookups and follows a file's change.
#[test]
fn lookup_follows_a_changed_file() {
    let f = Fixture::new();
    f.write("a.py", "def alpha():\n    return 1\n");
    let got = f.call("lookup", json!({"query": "zebra"})).unwrap();
    assert!(got.starts_with("no block matches"), "{got}");
    f.write(
        "a.py",
        "def alpha():\n    return 1\n\n\ndef zebra():\n    return 2\n",
    );
    let got = f.call("lookup", json!({"query": "zebra"})).unwrap();
    assert!(got.starts_with("a.py, def zebra, lines 5-6:"), "{got}");
}

/// replace_symbol on `path` (holding `text`) for `name` with `new`: the file is exactly `want`
/// after it, or, with `want` None, it is refused with the file unchanged. The map reads brackets
/// and indents, not a grammar, so where its span is wrong the replace must refuse, never drop or
/// leave code; where it is right, it must replace.
fn replaces(path: &str, text: &str, name: &str, new: &str, want: Option<&str>) {
    let f = Fixture::new();
    f.write(path, text);
    let got = f.call(
        "replace_symbol",
        json!({"path": path, "name": name, "new_text": new}),
    );
    match (got, want) {
        (Ok(done), Some(want)) => assert_eq!(f.read(path), want, "{done}"),
        (Err(ToolError::Invalid(why)), None) => assert_eq!(f.read(path), text, "{why}"),
        (got, want) => panic!("{name}: got {got:?}, want {want:?}"),
    }
}

/// The last method of a Python class, of a nested class, and a method a second class names too
/// are replaced: the class ends before the line end the new text takes.
#[test]
fn replace_symbol_replaces_the_last_method_of_a_python_class() {
    let text =
        "class A:\n    def f(self):\n        return 1\n\n    def g(self):\n        return 2\n";
    let new = "    def g(self):\n        return 3\n";
    replaces(
        "a.py",
        text,
        "A.g",
        new,
        Some(&text.replace("return 2", "return 3")),
    );
    let after = format!("{text}\n\ndef h():\n    return 4\n");
    replaces(
        "a.py",
        &after,
        "A.g",
        new,
        Some(&after.replace("return 2", "return 3")),
    );
    let meta =
        "class A:\n    x = 1\n\n    class Meta:\n        def m(self):\n            return 1\n";
    let new = "        def m(self):\n            return 2";
    replaces(
        "m.py",
        meta,
        "A.Meta.m",
        new,
        Some(&meta.replace("return 1", "return 2")),
    );
    let two = "class A:\n    def __str__(self):\n        return 'a'\n\n\nclass B:\n    def __str__(self):\n        return 'b'\n";
    let new = "    def __str__(self):\n        return 'A'\n";
    replaces(
        "t.py",
        two,
        "A.__str__",
        new,
        Some(&two.replace("'a'", "'A'")),
    );
}

/// An arrow whose body is a template literal, in code without semicolons: its span must not
/// take the next item with it.
#[test]
fn replace_symbol_never_drops_the_item_after_a_string_arrow() {
    let text = "const url = (id) => `/api/${id}`\nexport const load = async (id) => {\n  return fetch(url(id))\n}\n";
    let new = "const url = (id) => `/api/v2/${id}`";
    let want = text.replace("`/api/${id}`", "`/api/v2/${id}`");
    replaces("api.js", text, "url", new, Some(&want));
}

/// A chain on its own lines after an arrow (as Prettier writes it): the replace must not leave
/// the old chain behind the new text.
#[test]
fn replace_symbol_never_leaves_a_chain_behind() {
    let text = "const total = (xs) =>\n  xs\n    .map((x) => x * 2)\n    .reduce((a, b) => a + b, 0);\n\nfunction other() {\n  return 1;\n}\n";
    let new = "const total = (xs) =>\n  xs.reduce((a, b) => a + b, 0);";
    let want = text.replace(
        "const total = (xs) =>\n  xs\n    .map((x) => x * 2)\n    .reduce((a, b) => a + b, 0);",
        new,
    );
    replaces("total.js", text, "total", new, Some(&want));
}

/// A Go function whose return type has braces: the body is the block after them.
#[test]
fn replace_symbol_never_leaves_a_go_body_behind() {
    let text = "package main\n\nfunc load() map[string]interface{} {\n\treturn map[string]interface{}{\"a\": 1}\n}\n\nfunc other() int {\n\treturn 2\n}\n";
    let new = "func load() map[string]interface{} {\n\treturn nil\n}";
    let want = text.replace(
        "func load() map[string]interface{} {\n\treturn map[string]interface{}{\"a\": 1}\n}",
        new,
    );
    replaces("load.go", text, "load", new, Some(&want));
}

/// A TypeScript function whose return type has braces, which no syntax check covers.
#[test]
fn replace_symbol_never_leaves_a_ts_body_behind() {
    let text = "export async function f(): Promise<{ ok: boolean }> {\n  return { ok: true };\n}\n\nexport function g() {\n  return 1;\n}\n";
    let new = "export async function f(): Promise<{ ok: boolean }> {\n  return { ok: false };\n}";
    let want = text.replace(
        "export async function f(): Promise<{ ok: boolean }> {\n  return { ok: true };\n}",
        new,
    );
    replaces("f.ts", text, "f", new, Some(&want));
}

/// A Python function that ends in a multi-line string: the replace must take the whole string.
#[test]
fn replace_symbol_never_leaves_a_python_string_behind() {
    let text =
        "def page():\n    return \"\"\"\n<html>\n</html>\n\"\"\"\n\n\ndef other():\n    return 1\n";
    let new = "def page():\n    return \"<p>hi</p>\"";
    let want = text.replace(
        "def page():\n    return \"\"\"\n<html>\n</html>\n\"\"\"",
        new,
    );
    replaces("page.py", text, "page", new, Some(&want));
}

/// The guard leaves a right span alone: an object's method, followed by its comma, is replaced.
#[test]
fn replace_symbol_replaces_an_objects_method_before_its_comma() {
    let text =
        "const api = {\n  get() {\n    return 1;\n  },\n  set() {\n    return 2;\n  },\n};\n";
    let new = "  get() {\n    return 3;\n  }";
    let f = Fixture::new();
    f.write("api.js", text);
    let names: Vec<String> = larch_agent::code::map::symbols("api.js", text)
        .into_iter()
        .map(|s| s.name)
        .collect();
    let Some(name) = names.iter().find(|n| n.ends_with("get")) else {
        // The map reads no object methods: nothing to replace.
        return;
    };
    let got = f.call(
        "replace_symbol",
        json!({"path": "api.js", "name": name, "new_text": new}),
    );
    assert!(got.is_ok(), "{got:?} {names:?}");
    assert_eq!(f.read("api.js"), text.replace("return 1;", "return 3;"));
}

/// A docstring-only def takes a new docstring: the guard reads its span whole.
#[test]
fn replace_symbol_replaces_a_docstring_def() {
    let f = Fixture::new();
    let text =
        "def doc():\n    \"\"\"Old.\n\n    More.\n    \"\"\"\n\n\ndef other():\n    return 1\n";
    f.write("d.py", text);
    let new = "def doc():\n    \"\"\"New.\n\n    More.\n    \"\"\"";
    let got = f.call(
        "replace_symbol",
        json!({"path": "d.py", "name": "doc", "new_text": new}),
    );
    assert!(got.is_ok(), "{got:?}");
    assert_eq!(f.read("d.py"), text.replace("Old.", "New."));
}

/// A new text that ends on an indented comment is the def's own last line.
#[test]
fn replace_symbol_takes_a_new_text_ending_on_a_comment() {
    let f = Fixture::new();
    let text = "def f():\n    return 1\n\n\ndef g():\n    pass\n";
    f.write("c.py", text);
    let new = "def f():\n    return 2\n    # keep in step with g";
    let got = f.call(
        "replace_symbol",
        json!({"path": "c.py", "name": "f", "new_text": new}),
    );
    assert!(got.is_ok(), "{got:?}");
    assert_eq!(
        f.read("c.py"),
        text.replace(
            "    return 1\n",
            "    return 2\n    # keep in step with g\n"
        )
    );
}
