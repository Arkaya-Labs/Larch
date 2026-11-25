//! The code map: each named block of a source file (a function, a class and its methods, an impl,
//! a CSS rule, an HTML element with an id) and the lines it spans, found by bracket and indent
//! matching over the text with its comments and strings blanked out, so a brace in either counts
//! for nothing.

use std::{cmp::Reverse, collections::HashMap};

use regex::Regex;

/// A named block of a file and the lines it spans, 1-based and inclusive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    /// Its name, qualified by the blocks it is in: `Board::new`, `Board.dropDisc`, `.cell:hover`.
    pub name: String,
    /// What declares it: `fn`, `struct`, `impl`, `class`, `def`, `function`, `method`, `func`,
    /// `type`, `rule`, `at-rule`, `element`, ...
    pub kind: &'static str,
    /// Its first line: its decorators, attributes or doc comments, else its header.
    pub start_line: usize,
    /// Its last line: the closing brace or tag, or the last line indented under it.
    pub end_line: usize,
    /// The byte it starts at: its first line's start, or its first character for one that
    /// shares the line with other code (a CSS rule, an HTML element).
    pub start: usize,
    /// The byte past its last: its closing brace, tag or `;`, or its last line's last character.
    pub end: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Lang {
    Rust,
    Python,
    Js,
    Go,
    Css,
    Html,
}

/// A symbol as a language's reader finds it: byte offsets, its own name, and the name the blocks
/// inside it are qualified with when that is not its own (an impl's type).
struct Item {
    kind: &'static str,
    own: String,
    scope: Option<String>,
    start: usize,
    end: usize,
}

/// The kinds that are functions, one for another when a name is looked up.
const FUNCTIONS: [&str; 5] = ["fn", "def", "function", "method", "func"];

/// The words a lookup may start with to say which kind it means.
const KINDS: [&str; 16] = [
    "fn", "struct", "enum", "trait", "union", "mod", "impl", "def", "class", "function", "method",
    "func", "type", "rule", "at-rule", "element",
];

/// Words a lookup may carry from the header it copied, which say nothing of the kind.
const MODIFIERS: [&str; 7] = [
    "pub", "async", "export", "static", "default", "const", "unsafe",
];

/// Words after which a JS slash opens a regular expression, as an operator does.
const BEFORE_REGEX: [&str; 12] = [
    "return", "typeof", "case", "in", "of", "void", "yield", "await", "delete", "throw", "else",
    "do",
];

/// HTML elements that have no closing tag.
const VOID: [&str; 14] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// The tail `outline` keeps room for.
const MORE_BYTES: usize = 24;

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("a valid regex")
}

fn lang(path: &str) -> Option<Lang> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    Some(
        match name.rsplit_once('.')?.1.to_ascii_lowercase().as_str() {
            "rs" => Lang::Rust,
            "py" | "pyi" => Lang::Python,
            "js" | "mjs" | "cjs" | "jsx" | "ts" | "tsx" | "mts" | "cts" => Lang::Js,
            "go" => Lang::Go,
            "css" => Lang::Css,
            "html" | "htm" => Lang::Html,
            _ => return None,
        },
    )
}

/// Whether the map reads the file at `path`, by its extension.
pub(crate) fn reads(path: &str) -> bool {
    lang(path).is_some()
}

/// Where the bracket that the one at `close` in masked text `m` closes opens.
fn opener(m: &str, close: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, b) in m.as_bytes()[..=close].iter().enumerate().rev() {
        match b {
            b')' | b']' | b'}' => depth += 1,
            b'(' | b'[' | b'{' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The named blocks of `text`, the file at `path`, in file order, nested ones after the block
/// they are in; none for a language the map does not read (by `path`'s extension).
// ponytail: brackets and indents, no parser: a macro that makes items, a Go `type ( ... )` group,
// a JS template nested in a template's `${}` and an apostrophe in JSX text go unread; tree-sitter
// if a real repo needs them
pub fn symbols(path: &str, text: &str) -> Vec<Symbol> {
    let Some(lang) = lang(path) else {
        return vec![];
    };
    let (mut m, strings) = mask(text, lang);
    // A string's last byte is an operand where a JS or Go line may end: a JS arrow whose value
    // is a template ends at its line, not at the next item's.
    if matches!(lang, Lang::Js | Lang::Go) {
        let mut b = m.into_bytes();
        for &(_, end) in &strings {
            if b[end - 1] == b' ' {
                b[end - 1] = b'"';
            }
        }
        m = String::from_utf8(b).expect("an ASCII quote for an ASCII space");
    }
    let mut items = match lang {
        Lang::Rust => rust(&m),
        Lang::Python => python(&m, text, &strings),
        Lang::Js => js(&m),
        Lang::Go => go(&m),
        Lang::Css => css(&m),
        Lang::Html => html(&m),
    };
    // Decorators, attributes and doc comments directly above an item are part of it: a model
    // that viewed it gives them in its new text.
    let above: &[&str] = match lang {
        Lang::Rust => &["#[", "///"],
        Lang::Python | Lang::Js => &["@"],
        _ => &[],
    };
    for item in &mut items {
        let mut at = text[..item.start].rfind('\n').map_or(0, |n| n + 1);
        if !text[at..item.start].trim().is_empty() {
            continue;
        }
        while at > 0 {
            let mut up = text[..at - 1].rfind('\n').map_or(0, |n| n + 1);
            // A decorator whose arguments span lines: up to the line its last bracket opens on.
            if m[up..at].trim_end().ends_with([')', ']', '}']) {
                up = opener(&m, m[..at].trim_end().len() - 1)
                    .map_or(up, |o| m[..o].rfind('\n').map_or(0, |n| n + 1));
            }
            if !above
                .iter()
                .any(|p| text[up..at].trim_start().starts_with(p))
            {
                break;
            }
            at = up;
        }
        item.start = item.start.min(at);
    }
    items.sort_by_key(|i| (i.start, Reverse(i.end)));
    // Go names a method by its receiver; CSS rules and HTML ids stand alone.
    let sep = match lang {
        Lang::Rust => Some("::"),
        Lang::Python | Lang::Js => Some("."),
        _ => None,
    };
    let starts: Vec<usize> = std::iter::once(0)
        .chain(m.match_indices('\n').map(|(i, _)| i + 1))
        .collect();
    let line = |at: usize| starts.partition_point(|&s| s <= at);
    // The blocks the next item may be in: each one's end and the scope it gives its blocks.
    let mut open: Vec<(usize, String)> = Vec::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        while open.last().is_some_and(|(end, _)| *end < item.start) {
            open.pop();
        }
        let scope = item.scope.as_deref().unwrap_or(&item.own);
        let (name, scope) = match (sep, open.last()) {
            (Some(sep), Some((_, outer))) => (
                format!("{outer}{sep}{}", item.own),
                format!("{outer}{sep}{scope}"),
            ),
            _ => (item.own.clone(), scope.to_string()),
        };
        if sep.is_some() {
            open.push((item.end, scope));
        }
        out.push(Symbol {
            name,
            kind: item.kind,
            start_line: line(item.start),
            end_line: line(item.end),
            start: item.start,
            end: item.end + 1,
        });
    }
    out
}

/// The one symbol `query` names: a name, qualified or by its last part (`Board.dropDisc`,
/// `Foo::bar`, `drop_disc`), after an optional kind (`impl Foo`, `fn bar`) and before an
/// optional line (`impl Foo 11-24`, as `outline` lists it). Err with every symbol it names when
/// that is more than one, or with none when it names none.
pub fn find<'a>(symbols: &'a [Symbol], query: &str) -> Result<&'a Symbol, Vec<&'a Symbol>> {
    let mut q = query.trim();
    let mut line = None;
    if let Some(c) = re(r"\s+(\d+)(?:-\d+)?$").captures(q) {
        line = c[1].parse::<usize>().ok();
        q = q[..c.get(0).expect("the match").start()].trim_end();
    }
    let mut kind = None;
    while let Some((word, rest)) = q.split_once(char::is_whitespace) {
        match word {
            w if MODIFIERS.contains(&w) => {}
            w if KINDS.contains(&w) => kind = Some(w),
            _ => break,
        }
        q = rest.trim_start();
    }
    let norm = |s: &str| s.replace("::", ".");
    let q = norm(q.strip_suffix("()").unwrap_or(q));
    let here: Vec<&Symbol> = (symbols.iter())
        .filter(|s| line.is_none_or(|l| s.start_line == l))
        .collect();
    let exact = |s: &&&Symbol| {
        norm(&s.name) == q || (s.kind == "element" && q.strip_prefix('#') == Some(&s.name))
    };
    // Every block the bare name is in, a top-level one too: picking one silently picks wrong.
    let tail = format!(".{q}");
    let mut found: Vec<&Symbol> = (here.into_iter())
        .filter(|s| exact(&s) || norm(&s.name).ends_with(&tail))
        .collect();
    if let Some(kind) = kind {
        let same = |k: &str| k == kind || (FUNCTIONS.contains(&k) && FUNCTIONS.contains(&kind));
        let of_kind: Vec<&Symbol> = found.iter().filter(|s| same(s.kind)).copied().collect();
        // `impl Foo::bar` names the fn bar in impl Foo: the kind fits none, so it is no filter.
        if !of_kind.is_empty() {
            found = of_kind;
        }
    }
    match found[..] {
        [one] => Ok(one),
        _ => Err(found),
    }
}

/// `symbols` one per line, `kind name start-end`, in `tokens` (4 bytes each, as
/// `tool::truncate` counts): all of them, or else the ones no other contains, cut to the budget
/// with a last line saying how many are left out.
pub fn outline(symbols: &[&Symbol], tokens: u32) -> String {
    let cap = tokens as usize * 4;
    let shown = |s: &Symbol| format!("{} {} {}-{}", s.kind, s.name, s.start_line, s.end_line);
    let all: Vec<String> = symbols.iter().map(|s| shown(s)).collect();
    if all.iter().map(|l| l.len() + 1).sum::<usize>() <= cap + 1 {
        return all.join("\n");
    }
    let inside = |s: &Symbol, o: &Symbol| {
        o.start_line <= s.start_line
            && s.end_line <= o.end_line
            && (o.start_line, o.end_line) != (s.start_line, s.end_line)
    };
    let mut out = String::new();
    let mut kept = 0;
    for s in symbols
        .iter()
        .filter(|s| !symbols.iter().any(|o| inside(s, o)))
    {
        let line = shown(s);
        if out.len() + line.len() + 1 + MORE_BYTES > cap {
            break;
        }
        out += &line;
        out.push('\n');
        kept += 1;
    }
    out + &format!("... {} more symbols", symbols.len() - kept)
}

/// `text` with every comment and string (quotes too) turned to spaces, newlines kept, so each
/// byte stays where it was and a bracket left is code; and where each string (or JS regular
/// expression) starts and ends, an operand where a comment is none.
fn mask(text: &str, lang: Lang) -> (String, Vec<(usize, usize)>) {
    let b = text.as_bytes();
    let n = b.len();
    let mut out = b.to_vec();
    let mut strings = Vec::new();
    let c_like = matches!(lang, Lang::Rust | Lang::Js | Lang::Go);
    let ident = |at: usize| b[at].is_ascii_alphanumeric() || b[at] == b'_';
    // A JS slash after an operator, a `BEFORE_REGEX` word or at the start opens a regular
    // expression; after a name or a `)` it divides.
    let mut last = b'(';
    let mut i = 0;
    while i < n {
        let rest = &b[i..];
        let end = match b[i] {
            b'/' if c_like && rest.starts_with(b"//") => eol(b, i),
            b'#' if lang == Lang::Python => eol(b, i),
            b'/' if c_like || lang == Lang::Css => match rest.starts_with(b"/*") {
                true => comment_end(b, i, lang == Lang::Rust),
                false
                    if lang == Lang::Js
                        && (b"(,=:[!&|?{};+-*%<>~^".contains(&last)
                            || BEFORE_REGEX.contains(&word_before(text, i))) =>
                {
                    regex_end(b, i)
                }
                false => {
                    last = b'/';
                    i += 1;
                    continue;
                }
            },
            b'<' if lang == Lang::Html && rest.starts_with(b"<!--") => {
                seek(b, i + 4, b"-->").map_or(n, |e| e + 3)
            }
            b'"' | b'\''
                if lang == Lang::Python
                    && (rest.starts_with(b"\"\"\"") || rest.starts_with(b"'''")) =>
            {
                seek(b, i + 3, &rest[..3]).map_or(n, |e| e + 3)
            }
            // A char literal, or a lifetime, which is code.
            b'\'' if lang == Lang::Rust => match char_end(text, i) {
                Some(end) => end,
                None => {
                    i += 1;
                    continue;
                }
            },
            b'"' | b'\'' if lang != Lang::Html => quoted(b, i, lang == Lang::Rust, true),
            b'`' if lang == Lang::Js => quoted(b, i, true, true),
            b'`' if lang == Lang::Go => quoted(b, i, true, false),
            b'r' if lang == Lang::Rust && (i == 0 || !ident(i - 1)) => match raw_end(b, i) {
                Some(end) => end,
                None => {
                    last = b'r';
                    i += 1;
                    continue;
                }
            },
            c => {
                if !c.is_ascii_whitespace() {
                    last = c;
                }
                i += 1;
                continue;
            }
        };
        let end = end.min(n);
        for x in &mut out[i..end] {
            if *x != b'\n' {
                *x = b' ';
            }
        }
        let comment = rest.starts_with(b"//") || rest.starts_with(b"/*") || rest[0] == b'#';
        if !comment && !rest.starts_with(b"<!--") {
            strings.push((i, end));
        }
        last = b'"';
        i = end;
    }
    let out = String::from_utf8(out).expect("whole characters and ASCII bytes turn to spaces");
    (out, strings)
}

/// The word that ends right before `i`, past spaces: `return` in `return /x/`.
fn word_before(text: &str, i: usize) -> &str {
    let code = text[..i].trim_end();
    let word = code.trim_end_matches(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$');
    &code[word.len()..]
}

/// `js` with its comments, strings, template literals and regular expressions turned to spaces,
/// so what is left is code: its brackets and names, each where it was.
pub(crate) fn js_code(js: &str) -> String {
    mask(js, Lang::Js).0
}

/// Where the line at `i` ends: its newline, or the text's end.
fn eol(b: &[u8], i: usize) -> usize {
    seek(b, i, b"\n").unwrap_or(b.len())
}

fn seek(b: &[u8], from: usize, what: &[u8]) -> Option<usize> {
    (b.get(from..)?.windows(what.len()))
        .position(|w| w == what)
        .map(|p| p + from)
}

/// The end of the `/*` comment at `i`; Rust's nest.
fn comment_end(b: &[u8], i: usize, nest: bool) -> usize {
    let (mut depth, mut j) = (0usize, i);
    while j < b.len() {
        if b[j..].starts_with(b"/*") && (nest || depth == 0) {
            depth += 1;
            j += 2;
        } else if b[j..].starts_with(b"*/") {
            depth -= 1;
            j += 2;
            if depth == 0 {
                return j;
            }
        } else {
            j += 1;
        }
    }
    b.len()
}

/// The end of the string whose quote is at `i`: past its closing quote, or at the newline that
/// ends an unclosed one unless it spans lines.
fn quoted(b: &[u8], i: usize, multiline: bool, escapes: bool) -> usize {
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' if escapes => j += 1,
            q if q == b[i] => return j + 1,
            b'\n' if !multiline => return j,
            _ => {}
        }
        j += 1;
    }
    b.len()
}

/// The end of the JS regular expression at `i`, past its flags.
fn regex_end(b: &[u8], i: usize) -> usize {
    let (mut j, mut class) = (i + 1, false);
    while j < b.len() {
        match b[j] {
            b'\\' => j += 1,
            b'[' => class = true,
            b']' => class = false,
            b'/' if !class => {
                j += 1;
                while j < b.len() && b[j].is_ascii_alphabetic() {
                    j += 1;
                }
                return j;
            }
            b'\n' => return j,
            _ => {}
        }
        j += 1;
    }
    b.len()
}

/// The end of the Rust char literal at `i` (`'{'`, `'\n'`, `'\u{7d}'`), or None for a lifetime.
fn char_end(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if b.get(i + 1) == Some(&b'\\') {
        return seek(&b[..b.len().min(i + 12)], i + 2, b"'").map(|e| e + 1);
    }
    let len = text[i + 1..].chars().next()?.len_utf8();
    (b.get(i + 1 + len) == Some(&b'\'')).then_some(i + 2 + len)
}

/// The end of the raw string `r#"..."#` at `i`, or None when `r` starts no raw string.
fn raw_end(b: &[u8], i: usize) -> Option<usize> {
    let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
    let open = i + 1 + hashes;
    (b.get(open) == Some(&b'"')).then_some(())?;
    let mut close = vec![b'"'];
    close.extend(std::iter::repeat_n(b'#', hashes));
    Some(seek(b, open + 1, &close).map_or(b.len(), |e| e + close.len()))
}

/// Each bracket's index and the index of the one that closes it, in one pass: a bracket no
/// bracket closes is not in it. Each kind is counted apart, as a mismatch is no close.
fn pairs(m: &[u8]) -> HashMap<usize, usize> {
    let (mut pairs, mut open) = (HashMap::new(), [Vec::new(), Vec::new(), Vec::new()]);
    for (j, &c) in m.iter().enumerate() {
        match c {
            b'{' => open[0].push(j),
            b'(' => open[1].push(j),
            b'[' => open[2].push(j),
            b'}' | b')' | b']' => {
                let k = match c {
                    b'}' => 0,
                    b')' => 1,
                    _ => 2,
                };
                if let Some(o) = open[k].pop() {
                    pairs.insert(o, j);
                }
            }
            _ => {}
        }
    }
    pairs
}

/// The last byte of the item whose header ends at `from`: its block's closing brace, or the `;`
/// that ends it first outside brackets; with `asi`, also the end of the line its code ends on
/// when nothing there or at the next line's start says it goes on (a JS arrow's expression, a
/// Go `type ID int`). A block the header goes on after (`interface{} {`, `Promise<{ ok }> {`)
/// is a type's, not the body. None when its block never closes: no symbol beats one that runs
/// to the end of the file.
fn body_end(m: &[u8], pairs: &HashMap<usize, usize>, from: usize, asi: bool) -> Option<usize> {
    let (mut depth, mut last, mut j) = (0usize, None::<usize>, from);
    while j < m.len() {
        let c = m[j];
        match c {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth = depth.saturating_sub(1),
            b'{' if depth == 0 => {
                let close = *pairs.get(&j)?;
                let next = m[close + 1..].iter().find(|c| !matches!(c, b' ' | b'\t'));
                if !matches!(next, Some(b'{' | b'>' | b'|' | b'&' | b'[' | b',')) {
                    return Some(close);
                }
                (last, j) = (Some(close), close + 1);
                continue;
            }
            b';' if depth == 0 => return Some(j),
            b'\n' if asi && depth == 0 => {
                let next = m[j..].iter().find(|c| !c.is_ascii_whitespace());
                let goes_on = next.is_some_and(|c| b".?:&|+-*/%,=".contains(c));
                // A postfix `++` or `--` ends its operand (`() => count++`), not the operator a
                // `+` or `-` would be.
                let postfix = |l: usize| l > 0 && b"+-".contains(&m[l]) && m[l - 1] == m[l];
                let ends = |l: usize| !b"=>,(+-*/?:&|.".contains(&m[l]) || postfix(l);
                if let Some(l) = last.filter(|&l| ends(l) && !goes_on) {
                    return Some(l);
                }
            }
            _ => {}
        }
        if !c.is_ascii_whitespace() {
            last = Some(j);
        }
        j += 1;
    }
    last.filter(|_| asi)
}

/// `s` with its words one space apart.
fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rust(m: &str) -> Vec<Item> {
    let b = m.as_bytes();
    let pairs = pairs(b);
    let item = re(
        r"(?m)^[ \t]*(?:pub(?:[ \t]*\([^)\n]*\))?[ \t]+)?(?:(?:default|const|async|unsafe|extern)[ \t]+)*(fn|struct|enum|trait|union|mod)[ \t]+([A-Za-z_]\w*)",
    );
    let mut items: Vec<Item> = (item.captures_iter(m))
        .filter_map(|c| {
            let at = c.get(0).expect("the match");
            Some(Item {
                kind: match &c[1] {
                    "fn" => "fn",
                    "struct" => "struct",
                    "enum" => "enum",
                    "trait" => "trait",
                    "union" => "union",
                    _ => "mod",
                },
                own: c[2].into(),
                scope: None,
                start: at.start(),
                end: body_end(b, &pairs, at.end(), false)?,
            })
        })
        .collect();
    for at in re(r"(?m)^[ \t]*(?:unsafe[ \t]+)?impl\b").find_iter(m) {
        let Some(end) = body_end(b, &pairs, at.end(), false) else {
            continue;
        };
        let head = m[at.end()..end].split('{').next().unwrap_or_default();
        // `impl<T: Clone> fmt::Display for Wrap<T> where ...` is `fmt::Display for Wrap`.
        let (mut plain, mut depth, mut prev) = (String::new(), 0usize, ' ');
        for c in head.chars() {
            match c {
                '<' => depth += 1,
                '>' if prev != '-' && depth > 0 => depth -= 1,
                c if depth == 0 => plain.push(c),
                _ => {}
            }
            prev = c;
        }
        let plain = collapse(&plain);
        let own = plain.split(" where ").next().unwrap_or_default().trim();
        let ty = own.rsplit(" for ").next().unwrap_or(own);
        let ty = ty.rsplit("::").next().unwrap_or(ty);
        let ty = ty.trim_start_matches(['&', ' ']).trim_start_matches("mut ");
        items.push(Item {
            kind: "impl",
            own: own.into(),
            scope: Some(ty.trim_start_matches("dyn ").into()),
            start: at.start(),
            end,
        });
    }
    items
}

/// Python's defs and classes in `m`, `text` masked; a line that starts inside one of `strings`
/// (a multi-line string's) is its block's whatever its indent when the block opened that
/// string, and never when it opened past the block (`"""` at column 0 after a def, old code
/// left in a string).
fn python(m: &str, text: &str, strings: &[(usize, usize)]) -> Vec<Item> {
    let b = m.as_bytes();
    // Where the string a line starting at `at` is inside opened; `strings` are in order.
    let opened = |at: usize| {
        let i = strings.partition_point(|&(s, _)| s < at).checked_sub(1)?;
        let (s, e) = strings[i];
        (at < e).then_some(s)
    };
    // Each line's start and text, and how deep in brackets it starts: a continuation line at
    // column 0 inside a call's brackets ends no block.
    let mut lines = Vec::new();
    let (mut at, mut depth) = (0usize, 0isize);
    for line in m.split_inclusive('\n') {
        lines.push((at, line, depth));
        for c in line.bytes() {
            match c {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = (depth - 1).max(0),
                _ => {}
            }
        }
        at += line.len();
    }
    let index = |byte: usize| lines.partition_point(|l| l.0 <= byte) - 1;
    let head = re(r"(?m)^([ \t]*)(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)");
    let mut items = Vec::new();
    for c in head.captures_iter(m) {
        let at = c.get(0).expect("the match");
        let indent = c[1].len();
        // The header ends at its `:` outside brackets, maybe lines below.
        let mut depth = 0isize;
        let colon = (at.end()..b.len())
            .find(|&j| {
                match b[j] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b':' => return depth <= 0,
                    _ => {}
                }
                false
            })
            .unwrap_or(at.end());
        let mut last = index(colon);
        for (k, &(at, line, depth)) in lines.iter().enumerate().skip(last + 1) {
            // A line of a string the block opened, or after a `\`, goes on the one before,
            // whatever its indent.
            let inside = opened(at);
            if inside.is_some_and(|s| index(s) <= last) || lines[k - 1].1.trim_end().ends_with('\\')
            {
                last = k;
                continue;
            }
            if line.trim().is_empty() {
                // A comment or a string's first line indented under the header is the block's,
                // as its code is; a line of a string it did not open is not.
                let real = &text[at..at + line.len()];
                if inside.is_none()
                    && !real.trim().is_empty()
                    && real.len() - real.trim_start().len() > indent
                {
                    last = k;
                }
                continue;
            }
            if depth <= 0 && line.len() - line.trim_start().len() <= indent {
                break;
            }
            last = k;
        }
        let (start, line, _) = lines[last];
        let line = &text[start..start + line.len()];
        items.push(Item {
            kind: if &c[2] == "def" { "def" } else { "class" },
            own: c[3].into(),
            scope: None,
            start: at.start(),
            end: start + line.trim_end().len().saturating_sub(1),
        });
    }
    items
}

fn js(m: &str) -> Vec<Item> {
    let b = m.as_bytes();
    let pairs = pairs(b);
    let item = |kind, own: &str, start, end| Item {
        kind,
        own: own.into(),
        scope: None,
        start,
        end,
    };
    let mut items = Vec::new();
    let function = re(
        r"(?m)^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?function\b[ \t]*\*?[ \t]*([A-Za-z_$][\w$]*)",
    );
    for c in function.captures_iter(m) {
        let at = c.get(0).expect("the match");
        let Some(end) = body_end(b, &pairs, at.end(), false) else {
            continue;
        };
        items.push(item("function", &c[1], at.start(), end));
    }
    // `const f = (a) => ...`, `= x => ...`, `= async function (...) {`: a value that is a function.
    let bound = re(
        r"(?m)^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+([A-Za-z_$][\w$]*)[^=\n]*=[ \t]*(?:async[ \t]+)?",
    );
    let arrow = re(r"^\s*(?::[^=;{]*)?=>");
    let bare = re(r"^[A-Za-z_$][\w$]*[ \t]*=>");
    for c in bound.captures_iter(m) {
        let at = c.get(0).expect("the match");
        let rest = &m[at.end()..];
        let end = if rest.starts_with("function") {
            body_end(b, &pairs, at.end(), false)
        } else if rest.starts_with('(') {
            let Some(shut) = pairs.get(&at.end()).map(|s| s + 1) else {
                continue;
            };
            match arrow.find(&m[shut..]) {
                Some(a) => body_end(b, &pairs, shut + a.end(), true),
                None => continue,
            }
        } else if let Some(a) = bare.find(rest) {
            body_end(b, &pairs, at.end() + a.end(), true)
        } else {
            continue;
        };
        let Some(end) = end else {
            continue;
        };
        items.push(item("function", &c[1], at.start(), end));
    }
    let class = re(
        r"(?m)^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:abstract[ \t]+)?class[ \t]+([A-Za-z_$][\w$]*)",
    );
    let method = re(
        r"^[ \t]*(?:(?:static|async|get|set|public|private|protected|override|readonly|abstract)[ \t]+)*\*?[ \t]*(#?[A-Za-z_$][\w$]*)[ \t]*(?:=[ \t]*(?:async[ \t]+)?)?\(",
    );
    const KEYWORDS: [&str; 8] = [
        "if", "for", "while", "switch", "catch", "return", "function", "super",
    ];
    for c in class.captures_iter(m) {
        let at = c.get(0).expect("the match");
        let Some(end) = body_end(b, &pairs, at.end(), false) else {
            continue;
        };
        items.push(item("class", &c[1], at.start(), end));
        // Its body's brace, not one of a type argument it extends.
        let Some(open) = (at.end()..end).find(|o| pairs.get(o) == Some(&end)) else {
            continue;
        };
        // The lines directly in the class body, where its methods and fields start.
        let (mut depth, mut start) = (0isize, open + 1);
        for line in m[open + 1..end].split_inclusive('\n') {
            if depth == 0
                && let Some(c) = method.captures(line)
                && !KEYWORDS.contains(&&c[1])
            {
                let head = c.get(0).expect("the match");
                // A field's arrow (`handle = (e) => this.go(e)`) may end at its line, unlike a
                // method's body: without a `;` it would run on into the next method.
                let field = head.as_str().contains('=');
                if let Some(end) = body_end(b, &pairs, start + head.end() - 1, field) {
                    items.push(item("method", &c[1], start, end));
                }
            }
            depth += line.matches('{').count() as isize - line.matches('}').count() as isize;
            start += line.len();
        }
    }
    items
}

fn go(m: &str) -> Vec<Item> {
    let b = m.as_bytes();
    let pairs = pairs(b);
    let func = re(
        r"(?m)^func[ \t]*(?:\([ \t]*(?:[A-Za-z_]\w*[ \t]+)?\*?[ \t]*([A-Za-z_]\w*)[^)\n]*\)[ \t]*)?([A-Za-z_]\w*)",
    );
    let mut items: Vec<Item> = (func.captures_iter(m))
        .filter_map(|c| {
            let at = c.get(0).expect("the match");
            let (kind, own) = match c.get(1) {
                Some(recv) => ("method", format!("{}.{}", recv.as_str(), &c[2])),
                None => ("func", c[2].to_string()),
            };
            Some(Item {
                kind,
                own,
                scope: None,
                start: at.start(),
                end: body_end(b, &pairs, at.end(), false)?,
            })
        })
        .collect();
    for c in re(r"(?m)^type[ \t]+([A-Za-z_]\w*)").captures_iter(m) {
        let at = c.get(0).expect("the match");
        if let Some(end) = body_end(b, &pairs, at.end(), true) {
            items.push(Item {
                kind: "type",
                own: c[1].into(),
                scope: None,
                start: at.start(),
                end,
            });
        }
    }
    items
}

/// Each block's selector or at-rule, from where the last statement or block ended to its `{`.
fn css(m: &str) -> Vec<Item> {
    let (mut items, mut open, mut seg) = (Vec::new(), Vec::new(), 0);
    for (j, c) in m.bytes().enumerate() {
        match c {
            b'{' => {
                let head = &m[seg..j];
                open.push((seg + head.len() - head.trim_start().len(), collapse(head)));
                seg = j + 1;
            }
            b'}' => {
                if let Some((start, own)) = open.pop().filter(|(_, own)| !own.is_empty()) {
                    let kind = if own.starts_with('@') {
                        "at-rule"
                    } else {
                        "rule"
                    };
                    items.push(Item {
                        kind,
                        own,
                        scope: None,
                        start,
                        end: j,
                    });
                }
                seg = j + 1;
            }
            b';' => seg = j + 1,
            _ => {}
        }
    }
    items
}

/// Each element with an id, to its closing tag: the next one of its name at its own depth.
fn html(m: &str) -> Vec<Item> {
    let lower = m.to_ascii_lowercase();
    let tag = re(r#"<([A-Za-z][\w-]*)(?:\s[^>]*?)?\sid\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))"#);
    let mut items = Vec::new();
    for c in tag.captures_iter(m) {
        let at = c.get(0).expect("the match");
        let name = c[1].to_ascii_lowercase();
        let id = (c.get(2).or(c.get(3)).or(c.get(4))).map_or("", |id| id.as_str());
        let Some(gt) = m[at.end()..].find('>').map(|g| g + at.end()) else {
            continue;
        };
        let void = VOID.contains(&name.as_str()) || m[..gt].ends_with('/');
        let end = match void {
            true => gt,
            false => closing(&lower, &name, gt + 1).unwrap_or(gt),
        };
        items.push(Item {
            kind: "element",
            own: id.into(),
            scope: None,
            start: at.start(),
            end,
        });
    }
    items
}

/// The `>` of the tag that closes a `<name>` opened before `from`, in lowercased HTML.
fn closing(lower: &str, name: &str, from: usize) -> Option<usize> {
    let mut depth = 1;
    let mut j = from;
    while let Some(lt) = lower[j..].find('<') {
        let at = j + lt;
        let rest = &lower[at + 1..];
        let (shut, rest) = rest.strip_prefix('/').map_or((false, rest), |r| (true, r));
        let named = rest.strip_prefix(name).is_some_and(|after| {
            !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '-')
        });
        if named && shut {
            depth -= 1;
            if depth == 0 {
                return lower[at..].find('>').map(|g| at + g);
            }
        } else if named {
            depth += 1;
        }
        j = at + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each symbol as `kind name start-end`, in file order.
    fn spans(path: &str, text: &str) -> Vec<String> {
        (symbols(path, text).iter())
            .map(|s| format!("{} {} {}-{}", s.kind, s.name, s.start_line, s.end_line))
            .collect()
    }

    const RUST: &str = r##"//! doc { not a brace
use std::fmt;

/// A board.
pub struct Board {
    cells: Vec<u8>,
}

pub(crate) struct Unit;

impl Board {
    pub fn new() -> Self {
        let s = "}{ fn fake() {";
        let c = '{';
        Board { cells: vec![] }
    }

    fn drop_disc<'a>(&'a mut self, col: [u8; 2]) -> Option<&'a u8> {
        /* } nested /* } */ */
        fn helper(x: [u8; 2]) -> u8 { x[0] }
        let r = r#"}"#;
        None
    }
}

impl<T: Clone> fmt::Display for Wrap<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{}}")
    }
}

pub trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> String {
        String::new()
    }
}

enum E { A, B }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
"##;

    #[test]
    fn rust_items_impls_and_their_fns_span_their_braces() {
        assert_eq!(
            spans("src/board.rs", RUST),
            [
                "struct Board 4-7",
                "struct Unit 9-9",
                "impl Board 11-24",
                "fn Board::new 12-16",
                "fn Board::drop_disc 18-23",
                "fn Board::drop_disc::helper 20-20",
                "impl fmt::Display for Wrap 26-30",
                "fn Wrap::fmt 27-29",
                "trait Shape 32-37",
                "fn Shape::area 33-33",
                "fn Shape::name 34-36",
                "enum E 39-39",
                "mod tests 41-45",
                "fn tests::it_works 43-44",
            ]
        );
    }

    #[test]
    fn python_defs_and_classes_span_their_indented_lines() {
        let py = r#""""Module doc: def fake(): pass"""
import os


class Board:
    """A board.
def not_a_def():
"""

    def __init__(self, rows=6,
                 cols=7):
        self.cells = [[0] * cols
for _ in range(rows)]

    def drop(self, col):
        def inner(x):
            return x  # def nope():
        # comment
# stray
        return inner(col)


async def main():
    s = "class Fake:"
    return s
"#;
        assert_eq!(
            spans("game.py", py),
            [
                "class Board 5-20",
                "def Board.__init__ 10-13",
                "def Board.drop 15-20",
                "def Board.drop.inner 16-17",
                "def main 23-25",
            ]
        );
    }

    const JS: &str = r#"// function fake() {
const RE = /[{]/g;
let s = `template ${ "}" } {`;

class Board {
  constructor(rows) {
    this.rows = rows; // }
  }

  dropDisc(col) {
    if (col < 0) {
      return '}';
    }
    function inner() {
      return "{";
    }
    return inner();
  }

  static async load(url) {
    return fetch(url);
  }

  handle = (e) => {
    this.dropDisc(e.col);
  };
}

function render(board) {
  return board;
}

export const add = (a, b) => a + b;

const make = async function (x) {
  return x;
};

const notFn = (1 + 2) * 3;
"#;

    #[test]
    fn js_functions_classes_methods_and_arrows_span_their_bodies() {
        assert_eq!(
            spans("web/game.js", JS),
            [
                "class Board 5-27",
                "method Board.constructor 6-8",
                "method Board.dropDisc 10-18",
                "function Board.dropDisc.inner 14-16",
                "method Board.load 20-22",
                "method Board.handle 24-26",
                "function render 29-31",
                "function add 33-33",
                "function make 35-37",
            ]
        );
    }

    #[test]
    fn go_funcs_methods_and_types() {
        let go = "package main\n\nimport \"fmt\"\n\n// Board holds cells. func Fake() {\n\
                  type Board struct {\n\tcells [][]int\n}\n\ntype ID int\n\n\
                  func (b *Board) Drop(col int) error {\n\ts := \"}\"\n\tr := '{'\n\
                  \traw := `\n}`\n\treturn nil\n}\n\nfunc main() {\n\tfmt.Println(\"{\")\n}\n";
        assert_eq!(
            spans("main.go", go),
            [
                "type Board 6-8",
                "type ID 10-10",
                "method Board.Drop 12-18",
                "func main 20-22",
            ]
        );
    }

    #[test]
    fn css_rules_map_each_selector_to_its_block() {
        let css = "/* .fake { } */\nbody {\n  margin: 0;\n}\n\n.cell:hover,\n.cell.red {\n  \
                   content: \"}\";\n}\n\n@media (max-width: 600px) {\n  .cell {\n    width: 10px;\n  \
                   }\n}\n";
        assert_eq!(
            spans("style.css", css),
            [
                "rule body 2-4",
                "rule .cell:hover, .cell.red 6-9",
                "at-rule @media (max-width: 600px) 11-15",
                "rule .cell 12-14",
            ]
        );
    }

    #[test]
    fn html_elements_with_an_id_span_to_their_closing_tag() {
        let html = "<!doctype html>\n<html>\n<body>\n  <!-- <div id=\"ghost\"></div> -->\n  \
                    <div id=\"board\" class=\"b\">\n    <div class=\"row\"><div id=\"c1\"></div></div>\n    \
                    <div>\n    </div>\n  </div>\n  <input id=\"name\" type=\"text\">\n  \
                    <button id='go'>Go</button>\n</body>\n</html>\n";
        assert_eq!(
            spans("index.html", html),
            [
                "element board 5-9",
                "element c1 6-6",
                "element name 10-10",
                "element go 11-11",
            ]
        );
    }

    #[test]
    fn decorators_belong_to_what_they_decorate() {
        let py = "import app\n\n@app.route('/')\n@login\ndef index():\n    return 1\n";
        assert_eq!(spans("web.py", py), ["def index 3-6"]);
        let py =
            "@app.route(\n    '/x',\n    methods=['GET'],\n)\n@login\ndef x():\n    return 1\n";
        assert_eq!(spans("web.py", py), ["def x 1-7"]);
        let ts = "class A {\n  @Input()\n  name(x) {\n    return x;\n  }\n}\n";
        assert_eq!(spans("a.ts", ts), ["class A 1-6", "method A.name 2-5"]);
    }

    #[test]
    fn a_class_field_arrow_without_a_semicolon_ends_at_its_line() {
        let js = "class A {\n  handle = (e) => this.go(e)\n  render() {\n    return 1\n  }\n}\n";
        assert_eq!(
            spans("a.js", js),
            ["class A 1-6", "method A.handle 2-2", "method A.render 3-5"]
        );
    }

    #[test]
    fn a_regex_after_a_keyword_is_no_division() {
        let js = "function f(s) {\n  return /{/.test(s);\n}\nfunction g() {\n  return 1;\n}\n";
        assert_eq!(spans("a.js", js), ["function f 1-3", "function g 4-6"]);
    }

    #[test]
    fn an_unclosed_block_is_no_symbol_and_costs_linear_time() {
        let rs = "fn a() {\n    let x = 1;\n\nfn b() {}\n";
        assert_eq!(spans("a.rs", rs), ["fn b 4-4"]);
        let many = "fn a() {\n".repeat(20_000);
        let t = std::time::Instant::now();
        assert!(symbols("a.rs", &many).is_empty());
        assert!(t.elapsed().as_secs() < 2, "{:?}", t.elapsed());
    }

    #[test]
    fn a_file_of_another_language_has_no_symbols() {
        assert!(symbols("notes.txt", "fn main() {}\n").is_empty());
    }

    #[test]
    fn find_takes_qualified_names_kinds_and_line_numbers() {
        let rs = symbols("src/board.rs", RUST);
        let at = |q: &str| find(&rs, q).map(|s| s.start_line);
        assert_eq!(at("impl Board"), Ok(11));
        assert_eq!(at("struct Board"), Ok(4));
        assert_eq!(at("Board::drop_disc"), Ok(18));
        assert_eq!(at("drop_disc"), Ok(18));
        assert_eq!(at("impl Board::new"), Ok(12));
        assert_eq!(at("pub fn helper()"), Ok(20));
        assert_eq!(at("impl Display for Wrap"), Ok(26));
        assert_eq!(at("Board 11-24"), Ok(11));
        let both: Vec<_> = find(&rs, "Board").unwrap_err();
        assert_eq!(
            both.iter().map(|s| s.kind).collect::<Vec<_>>(),
            ["struct", "impl"]
        );
        assert!(find(&rs, "nope").unwrap_err().is_empty());
        let js = symbols("game.js", JS);
        assert_eq!(find(&js, "Board.dropDisc").map(|s| s.end_line), Ok(18));
        assert_eq!(find(&js, "dropDisc()").map(|s| s.end_line), Ok(18));
        let css = symbols("a.css", "@media (max-width: 600px) {\n  .cell {\n  }\n}\n");
        let listed = outline(&css.iter().collect::<Vec<_>>(), 100);
        let first = listed.lines().next().unwrap();
        assert_eq!(find(&css, first).map(|s| s.kind), Ok("at-rule"), "{first}");
        // A bare name in two blocks is ambiguous, though one of them is top-level.
        let py = symbols(
            "a.py",
            "def drop():\n    pass\n\nclass B:\n    def drop(self):\n        pass\n",
        );
        assert_eq!(find(&py, "drop").unwrap_err().len(), 2);
        assert_eq!(find(&py, "B.drop").map(|s| s.start_line), Ok(5));
        let html = symbols("a.html", "<p id=\"x\">\n</p>\n");
        assert_eq!(find(&html, "#x").map(|s| s.end_line), Ok(2));
    }

    #[test]
    fn outline_drops_nested_symbols_then_cuts_to_its_budget() {
        let rs = symbols("src/board.rs", RUST);
        let rs: Vec<&Symbol> = rs.iter().collect();
        let all = outline(&rs, 1000);
        assert_eq!(all.lines().count(), rs.len(), "{all}");
        assert!(all.contains("fn Board::new 12-16"), "{all}");
        let small = outline(&rs, 30);
        assert!(small.len() <= 120, "{small}");
        assert!(small.starts_with("struct Board 4-7\n"), "{small}");
        assert!(!small.contains("Board::new"), "{small}");
        assert!(small.ends_with("more symbols"), "{small}");
    }

    /// A JS arrow whose value is a template literal or string ends at its line, not at the next
    /// item's; one whose next line goes on with an operator (`.map`, `?`, `:`, `&&`, `||`, `+`)
    /// runs on to where the expression ends.
    #[test]
    fn a_js_arrow_ends_at_its_string_and_goes_on_past_a_continued_line() {
        let js = "const page = (t) => `<h1>${t}</h1>`\nconst next = (n) => n + 1\n\
                  const total = (xs) => xs\n  .map((x) => x * 2)\n  .filter(Boolean)\n\
                  const pick = (a) => a\n  ? 1\n  : 2\nconst both = (a, b) => a\n  && b\n  || 'x'\n\
                  const sum = (a) => a\n  + 1\nconst say = () => 'hi'\nconst last = 1\n";
        assert_eq!(
            spans("a.js", js),
            [
                "function page 1-1",
                "function next 2-2",
                "function total 3-5",
                "function pick 6-8",
                "function both 9-11",
                "function sum 12-13",
                "function say 14-14",
            ]
        );
    }

    /// A Python block's lines inside a multi-line string are its own: a docstring-only def, a
    /// def that returns a multi-line string, and one that formats one end where the string does;
    /// so is a comment indented under it, not one at its header's indent.
    #[test]
    fn a_python_block_runs_to_the_end_of_its_multiline_string() {
        let py = "def doc():\n    \"\"\"Doc.\n\nMore.\n\"\"\"\n\n\
                  def page():\n    return \"\"\"\n<h1>\n</h1>\n\"\"\"\n\n\
                  def query(x):\n    q = \"\"\"\nselect 1\n\"\"\".format(x)\n    return q\n\n\
                  def after():\n    pass\n    # the end of after\n# not after's\n\n\
                  def tail():\n    return 'a' \\\n'b'\n";
        assert_eq!(
            spans("a.py", py),
            [
                "def doc 1-5",
                "def page 7-11",
                "def query 13-17",
                "def after 19-21",
                "def tail 24-26",
            ]
        );
    }

    /// A brace of the return type or of a type argument in a header is not the body's: a Go
    /// func returning `map[string]interface{}`, a TS function returning `Promise<{ ok: boolean }>`
    /// and a class that extends `B<{ x: 1 }>` span their bodies.
    #[test]
    fn a_brace_in_a_header_type_is_not_the_body() {
        let go = "func f() map[string]interface{} {\n\treturn nil\n}\n\nfunc g() {\n}\n";
        assert_eq!(spans("a.go", go), ["func f 1-3", "func g 5-6"]);
        let ts = "async function f(): Promise<{ ok: boolean }> {\n  return { ok: true };\n}\n\
                  class A extends B<{ x: 1 }> {\n  run() {\n    return 1;\n  }\n}\n";
        assert_eq!(
            spans("a.ts", ts),
            ["function f 1-3", "class A 4-8", "method A.run 5-7"]
        );
    }

    /// A JS arrow whose value ends on a postfix `++` or `--` ends at its line: the operator is
    /// its operand's, not one the next line goes on from.
    #[test]
    fn a_js_arrow_ends_at_a_postfix_operator() {
        let js = "const bump = () => count++\nconst drop = () => n--\nconst g = () => 2\n";
        assert_eq!(
            spans("a.js", js),
            ["function bump 1-1", "function drop 2-2", "function g 3-3"]
        );
    }

    /// A multi-line string that opens past a def at its indent or less (old code left in a
    /// `"""` or `'''` at column 0) is not the def's, whose replace would drop it.
    #[test]
    fn a_python_string_past_a_block_is_not_its() {
        let py = "def a():\n    return 1\n\n\"\"\"\ndef old():\n    pass\n\"\"\"\n\n\
                  def b():\n    return 2\n'''\nold code\n'''\ndef c():\n    pass\n";
        assert_eq!(
            spans("a.py", py),
            ["def a 1-2", "def b 9-10", "def c 14-15"]
        );
        // With no blank line before the string, and at the class's indent after a method.
        let tight = "def f():\n    return 1\n\"\"\"\ndef old():\n    pass\n\"\"\"\n";
        assert_eq!(spans("a.py", tight), ["def f 1-2"]);
        let class =
            "class A:\n    def m(self):\n        return 1\n    \"\"\"\n    old\n    \"\"\"\n";
        assert_eq!(spans("a.py", class), ["class A 1-6", "def A.m 2-3"]);
    }
}
