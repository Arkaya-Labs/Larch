//! A page's title and main text as Markdown: a lenient HTML parse, the readability density
//! heuristic to pick the main block, then Markdown with links kept. No JavaScript runs.

use std::collections::HashMap;

use serde_json::Value;
use url::Url;

/// An element of the parsed page; the root is `#root`.
struct Elem {
    name: String,
    attrs: Vec<(String, String)>,
    kids: Vec<Node>,
}

enum Node {
    Text(String),
    Elem(Elem),
}

impl Elem {
    fn attr(&self, name: &str) -> Option<&str> {
        (self.attrs.iter()).find_map(|(k, v)| (k == name).then_some(v.as_str()))
    }

    fn elems(&self) -> impl Iterator<Item = &Elem> {
        self.kids.iter().filter_map(|k| match k {
            Node::Elem(e) => Some(e),
            Node::Text(_) => None,
        })
    }

    /// The first element named `name` here or below, depth first, outside icons (an svg has a
    /// `title` of its own).
    fn find(&self, name: &str) -> Option<&Elem> {
        if self.name == name {
            return Some(self);
        }
        if self.name == "svg" {
            return None;
        }
        self.elems().find_map(|e| e.find(name))
    }
}

/// Elements with no end tag.
const VOID: [&str; 14] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];
/// Elements whose content is text up to their end tag, never markup.
const RAW: [&str; 5] = ["script", "style", "title", "textarea", "xmp"];
/// Elements that are never the page's text: code, chrome and controls. Not `form`: ASP.NET and
/// SharePoint pages wrap their whole body in one, and came back empty; its controls go one by one
/// (a `textarea`'s default text too, "Write your comment here"). Not `label`: WebForms pages hold
/// real text in them.
const STRIP: [&str; 14] = [
    "head", "title", "script", "style", "noscript", "template", "svg", "nav", "footer", "aside",
    "button", "iframe", "select", "textarea",
];
/// Elements that start a new Markdown block.
const BLOCK: [&str; 35] = [
    "address",
    "article",
    "blockquote",
    "caption",
    "center",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
];
/// Open elements kept; deeper start tags are dropped, their text kept, so the walks below
/// cannot overflow the stack on a hostile page.
// ponytail: a dropped tag's end tag may close an outer one of its name; count dropped tags if a
// saved page that nests past 256 shows text in the wrong block
const MAX_DEPTH: usize = 256;

/// Readability's class and id hints.
const NEGATIVE: [&str; 21] = [
    "banner",
    "combx",
    "comment",
    "contact",
    "cookie",
    "footer",
    "footnote",
    "gdpr",
    "masthead",
    "media",
    "meta",
    "outbrain",
    "promo",
    "related",
    "share",
    "shoutbox",
    "sidebar",
    "sponsor",
    "skyscraper",
    "tags",
    "widget",
];
const POSITIVE: [&str; 11] = [
    "article", "blog", "body", "content", "entry", "hentry", "main", "page", "post", "story",
    "text",
];

/// The page's `<title>` (its first `h1` when that is empty) and its main block as Markdown, with
/// links made absolute against `base`.
pub fn html(page: &str, base: &Url) -> (String, String) {
    let root = parse(page);
    let plain = |e: &Elem| {
        let mut t = String::new();
        text(e, &mut t);
        collapse(&t)
    };
    let title = Some(root.find("title").map(plain).unwrap_or_default())
        .filter(|t| !t.is_empty())
        .or_else(|| root.find("h1").map(plain))
        .unwrap_or_default();
    let mut out = recipe(&root);
    block(main(&root), base, 0, &mut out);
    (title, out.join("\n\n"))
}

/// A JSON-LD `Recipe`'s ingredients and steps as Markdown blocks, which lead the text so the head
/// every reader sees has them: allrecipes and bettycrocker give the list nowhere else (a user's
/// cookie question, 27 Oct).
fn recipe(root: &Elem) -> Vec<String> {
    let mut scripts = Vec::new();
    all(root, "script", &mut scripts);
    let found = (scripts.into_iter())
        .filter(|e| e.attr("type").is_some_and(|t| t.contains("ld+json")))
        .filter_map(|e| match e.kids.first() {
            Some(Node::Text(t)) => serde_json::from_str::<Value>(t).ok(),
            _ => None,
        })
        .find_map(|v| recipe_in(&v).cloned());
    let Some(r) = found else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let ingredients: Vec<String> = (r["recipeIngredient"].as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .map(|i| format!("- {}", collapse(i)))
        .collect();
    if !ingredients.is_empty() {
        out.extend(["Ingredients:".into(), ingredients.join("\n")]);
    }
    let mut steps = Vec::new();
    recipe_steps(&r["recipeInstructions"], &mut steps);
    if !steps.is_empty() {
        let numbered: Vec<String> = (1..)
            .zip(&steps)
            .map(|(n, s)| format!("{n}. {s}"))
            .collect();
        out.extend(["Steps:".into(), numbered.join("\n")]);
    }
    out
}

/// The `Recipe` in `v`, a JSON-LD value: itself, an item of its list, its `@graph` or its
/// `mainEntity`.
fn recipe_in(v: &Value) -> Option<&Value> {
    match v {
        Value::Array(items) => items.iter().find_map(recipe_in),
        Value::Object(o) => {
            // "Recipe" or "http://schema.org/Recipe", alone or in a list.
            let named = |t: &Value| t.as_str().is_some_and(|t| t.ends_with("Recipe"));
            let is = (o.get("@type"))
                .is_some_and(|t| named(t) || t.as_array().is_some_and(|a| a.iter().any(named)));
            if is {
                Some(v)
            } else {
                (o.get("@graph").and_then(recipe_in))
                    .or_else(|| o.get("mainEntity").and_then(recipe_in))
            }
        }
        _ => None,
    }
}

/// The text of each step in `v`: a string, a `HowToStep`, or a `HowToSection`'s items.
fn recipe_steps(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(untagged(s)),
        Value::Array(items) => items.iter().for_each(|i| recipe_steps(i, out)),
        // A step's text, else a section's items, else a step's name alone.
        Value::Object(o) => match (o.get("text"), o.get("itemListElement"), o.get("name")) {
            (Some(Value::String(t)), _, _) | (None, None, Some(Value::String(t))) => {
                out.push(untagged(t))
            }
            (_, Some(items), _) => recipe_steps(items, out),
            _ => {}
        },
        _ => {}
    }
    out.retain(|s| !s.is_empty());
}

/// `s` without its HTML tags, its spaces collapsed: some sites write a step as "<p>...</p>".
fn untagged(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut tag = false;
    for c in s.chars() {
        match (c, tag) {
            ('<', _) => tag = true,
            ('>', true) => {
                tag = false;
                out.push(' ');
            }
            (_, false) => out.push(c),
            _ => {}
        }
    }
    collapse(&out)
}

/// Every element named `name` at or below `e`.
fn all<'a>(e: &'a Elem, name: &str, out: &mut Vec<&'a Elem>) {
    if e.name == name {
        out.push(e);
    }
    for kid in e.elems() {
        all(kid, name, out);
    }
}

fn parse(page: &str) -> Elem {
    let lower = page.to_ascii_lowercase();
    let mut stack = vec![Elem {
        name: "#root".into(),
        attrs: Vec::new(),
        kids: Vec::new(),
    }];
    let mut text = String::new();
    let mut at = 0;
    while let Some(lt) = page[at..].find('<').map(|i| at + i) {
        text.push_str(&page[at..lt]);
        let after = lt + 1;
        let rest = &lower[after..];
        let past = |from: usize, end: &str| {
            lower[from..]
                .find(end)
                .map_or(page.len(), |i| from + i + end.len())
        };
        if rest.starts_with("!--") {
            at = past(after + 3, "-->");
            continue;
        }
        if rest.starts_with(['!', '?']) {
            at = past(after, ">");
            continue;
        }
        let close = rest.starts_with('/');
        let name_at = after + usize::from(close);
        let name = tag_name(&lower[name_at..]);
        if name.is_empty() {
            // `< 3` and `</>` are text, as a browser shows them.
            text.push('<');
            at = after;
            continue;
        }
        flush_text(&mut text, &mut stack);
        if close {
            at = past(name_at, ">");
            if let Some(open) = stack
                .iter()
                .rposition(|e| e.name == name)
                .filter(|&i| i > 0)
            {
                pop_to(&mut stack, open);
            }
            continue;
        }
        let (attrs, self_closing, end) = tag_attrs(page, name_at + name.len());
        at = end;
        implied_ends(&mut stack, name);
        let mut elem = Elem {
            name: name.into(),
            attrs,
            kids: Vec::new(),
        };
        if RAW.contains(&name) {
            let close = format!("</{name}");
            let stop = lower[at..].find(&close).map_or(page.len(), |i| at + i);
            elem.kids.push(Node::Text(decode(&page[at..stop])));
            at = past(stop, ">");
        } else if !(VOID.contains(&name) || self_closing) && stack.len() <= MAX_DEPTH {
            stack.push(elem);
            continue;
        } else if !VOID.contains(&name) && !self_closing {
            continue;
        }
        let top = stack.last_mut().expect("the root stays");
        top.kids.push(Node::Elem(elem));
    }
    text.push_str(&page[at..]);
    flush_text(&mut text, &mut stack);
    pop_to(&mut stack, 1);
    stack.pop().expect("the root stays")
}

/// A tag name at the start of `s`: a letter, then letters, digits, `-` and `:`.
fn tag_name(s: &str) -> &str {
    if !s.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return "";
    }
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == ':'))
        .unwrap_or(s.len());
    &s[..end]
}

/// The attributes from `at` to the tag's `>`, names lowercased and values decoded; whether the
/// tag ends `/>`; and the offset past the `>`.
fn tag_attrs(page: &str, mut at: usize) -> (Vec<(String, String)>, bool, usize) {
    let b = page.as_bytes();
    let mut attrs = Vec::new();
    let mut self_closing = false;
    loop {
        while at < b.len() && (b[at].is_ascii_whitespace() || b[at] == b'/') {
            self_closing = b[at] == b'/';
            at += 1;
        }
        if at >= b.len() || b[at] == b'>' {
            return (attrs, self_closing, (at + 1).min(b.len()));
        }
        self_closing = false;
        let start = at;
        while at < b.len() && !b[at].is_ascii_whitespace() && !b"=>/".contains(&b[at]) {
            at += 1;
        }
        let name = page[start..at].to_ascii_lowercase();
        while at < b.len() && b[at].is_ascii_whitespace() {
            at += 1;
        }
        let mut value = "";
        if b.get(at) == Some(&b'=') {
            at += 1;
            while at < b.len() && b[at].is_ascii_whitespace() {
                at += 1;
            }
            let quote = b.get(at).copied().filter(|q| *q == b'"' || *q == b'\'');
            let from = at + usize::from(quote.is_some());
            let end = match quote {
                Some(q) => page[from..].find(q as char).map_or(b.len(), |i| from + i),
                None => page[from..]
                    .find(|c: char| c.is_ascii_whitespace() || c == '>')
                    .map_or(b.len(), |i| from + i),
            };
            value = &page[from..end];
            at = (end + usize::from(quote.is_some())).min(b.len());
        }
        attrs.push((name, decode(value)));
    }
}

fn flush_text(text: &mut String, stack: &mut [Elem]) {
    if !text.is_empty() {
        let top = stack.last_mut().expect("the root stays");
        top.kids.push(Node::Text(decode(text)));
        text.clear();
    }
}

/// Closes the open elements from `open` up, each into its parent.
fn pop_to(stack: &mut Vec<Elem>, open: usize) {
    while stack.len() > open {
        let e = stack.pop().expect("longer than open");
        let top = stack.last_mut().expect("open is at least 1");
        top.kids.push(Node::Elem(e));
    }
}

/// The end tags HTML leaves out: a new item closes the open one in its list, row or table, and a
/// block closes an open paragraph.
fn implied_ends(stack: &mut Vec<Elem>, name: &str) {
    let (closes, stops): (&[&str], &[&str]) = match name {
        "li" => (&["li"], &["ul", "ol"]),
        "dt" | "dd" => (&["dt", "dd"], &["dl"]),
        // Closing the row closes its open cell with it.
        "tr" => (&["tr"], &["table", "thead", "tbody", "tfoot"]),
        "td" | "th" => (&["td", "th"], &["tr", "table"]),
        "option" => (&["option"], &["select"]),
        _ if BLOCK.contains(&name) => (&["p"], &BLOCK),
        _ => return,
    };
    for i in (1..stack.len()).rev() {
        let open = stack[i].name.as_str();
        if closes.contains(&open) {
            pop_to(stack, i);
            return;
        }
        if stops.contains(&open) {
            return;
        }
    }
}

/// `s` with its character references decoded; an unknown one stays as written.
// ponytail: the common named references only; the full table of 2,231 if pages show `&name;`
fn decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        // A reference is short: look for its `;` nearby, never to the end of the text.
        let near = &rest.as_bytes()[1..];
        let semi = (near.iter().take(33))
            .position(|&b| b == b';')
            .map(|i| i + 1);
        let decoded = semi.and_then(|semi| {
            let name = &rest[1..semi];
            let c = match name.strip_prefix('#') {
                Some(num) => {
                    let n = match num.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16),
                        None => num.parse(),
                    };
                    n.ok().map(|n| {
                        char::from_u32(n)
                            .filter(|&c| c != '\0')
                            .unwrap_or('\u{FFFD}')
                    })
                }
                None => named(name),
            };
            c.map(|c| (c, semi + 1))
        });
        match decoded {
            Some((c, len)) => {
                if c != '\u{AD}' {
                    out.push(c);
                }
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn named(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{A0}',
        "shy" => '\u{AD}',
        "ndash" => '\u{2013}',
        "mdash" => '\u{2014}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "sbquo" => '\u{201A}',
        "ldquo" => '\u{201C}',
        "rdquo" => '\u{201D}',
        "bdquo" => '\u{201E}',
        "hellip" => '\u{2026}',
        "laquo" => '\u{AB}',
        "raquo" => '\u{BB}',
        "middot" => '\u{B7}',
        "bull" => '\u{2022}',
        "copy" => '\u{A9}',
        "reg" => '\u{AE}',
        "trade" => '\u{2122}',
        "deg" => '\u{B0}',
        "times" => '\u{D7}',
        "divide" => '\u{F7}',
        "plusmn" => '\u{B1}',
        "euro" => '\u{20AC}',
        "pound" => '\u{A3}',
        "yen" => '\u{A5}',
        "cent" => '\u{A2}',
        "sect" => '\u{A7}',
        "para" => '\u{B6}',
        "larr" => '\u{2190}',
        "rarr" => '\u{2192}',
        "uarr" => '\u{2191}',
        "darr" => '\u{2193}',
        "le" => '\u{2264}',
        "ge" => '\u{2265}',
        "ne" => '\u{2260}',
        "minus" => '\u{2212}',
        "prime" => '\u{2032}',
        "thinsp" => '\u{2009}',
        "ensp" => '\u{2002}',
        "emsp" => '\u{2003}',
        "zwj" => '\u{200D}',
        "zwnj" => '\u{200C}',
        _ => return None,
    })
}

/// The text below `e`, blocks set apart by spaces, without the stripped elements.
fn text(e: &Elem, out: &mut String) {
    for kid in &e.kids {
        match kid {
            Node::Text(t) => out.push_str(t),
            Node::Elem(k) if STRIP.contains(&k.name.as_str()) => {}
            Node::Elem(k) => {
                let gap = BLOCK.contains(&k.name.as_str()) || k.name == "br";
                if gap {
                    out.push(' ');
                }
                text(k, out);
                if gap {
                    out.push(' ');
                }
            }
        }
    }
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Characters of text below `e`, and those inside links.
fn lengths(e: &Elem, in_link: bool) -> (usize, usize) {
    let mut n = (0, 0);
    for kid in &e.kids {
        let (all, linked) = match kid {
            Node::Text(t) => {
                let all = t.chars().filter(|c| !c.is_whitespace()).count();
                (all, if in_link { all } else { 0 })
            }
            Node::Elem(k) if STRIP.contains(&k.name.as_str()) => (0, 0),
            Node::Elem(k) => lengths(k, in_link || k.name == "a"),
        };
        n = (n.0 + all, n.1 + linked);
    }
    n
}

/// A scored element, its score and its ancestors, root first.
type Scored<'a> = (&'a Elem, f64, Vec<&'a Elem>);

/// Readability's main block: each paragraph of 25 characters or more scores its parent and, half,
/// its grandparent by its commas and length; the best score, cut by the share of the text in
/// links, wins. When three of the next five come within 75 % of it, their common ancestor wins
/// instead; else its parent, when that comes within 75 %. Nothing scores: the body, or the page.
// ponytail: no sibling merge; add Readability's high-scoring siblings if goldens show articles
// split across sibling blocks
fn main(root: &Elem) -> &Elem {
    let mut scores = HashMap::new();
    score(root, &mut Vec::new(), &mut scores);
    // Ties go to the earlier block: in page order, never the map's.
    let mut ranked: Vec<(usize, Scored)> = (scores.into_values())
        .map(|(order, (e, s, up))| {
            let (all, linked) = lengths(e, false);
            (
                order,
                (e, s * (1.0 - linked as f64 / all.max(1) as f64), up),
            )
        })
        .collect();
    ranked.sort_by(|(i, a), (j, b)| b.1.total_cmp(&a.1).then(i.cmp(j)));
    let ranked: Vec<Scored> = ranked.into_iter().map(|(_, s)| s).collect();
    let Some((top, best, up)) = ranked.first() else {
        return root.find("body").unwrap_or(root);
    };
    let close: Vec<&Scored> = (ranked.iter().skip(1).take(5))
        .filter(|(_, s, _)| *s >= 0.75 * best)
        .collect();
    if close.len() >= 3 {
        let shared = (up.iter().rev())
            .take_while(|a| !matches!(a.name.as_str(), "body" | "#root"))
            .find(|a| {
                let has = |c: &&&Scored| c.2.iter().any(|u| std::ptr::eq(*u, **a));
                close.iter().filter(has).count() >= 3
            });
        if let Some(shared) = shared {
            return shared;
        }
    }
    // A block that holds most of its parent's score is a part of it, as a lead is of an article.
    if let Some(parent) = (ranked.iter()).find(|c| up.last().is_some_and(|p| std::ptr::eq(*p, c.0)))
        && parent.1 >= 0.75 * best
    {
        return parent.0;
    }
    top
}

fn score<'a>(
    e: &'a Elem,
    ancestors: &mut Vec<&'a Elem>,
    scores: &mut HashMap<*const Elem, (usize, Scored<'a>)>,
) {
    if STRIP.contains(&e.name.as_str()) {
        return;
    }
    let paragraph = match e.name.as_str() {
        "p" | "pre" | "td" => true,
        "div" => !e.elems().any(|k| BLOCK.contains(&k.name.as_str())),
        _ => false,
    };
    if paragraph {
        let mut t = String::new();
        text(e, &mut t);
        let t = collapse(&t);
        let len = t.chars().count();
        if len >= 25 {
            let points = 1.0 + t.matches(',').count() as f64 + (len / 100).min(3) as f64;
            // Readability makes a div holding one paragraph that paragraph: its hints never count.
            let scoring = (0..ancestors.len())
                .rev()
                .filter(|&i| !wraps_one_p(ancestors[i]));
            for (share, i) in [1.0, 0.5].into_iter().zip(scoring) {
                let up = ancestors[i];
                let order = scores.len();
                let entry = (scores.entry(up as *const Elem))
                    .or_insert_with(|| (order, (up, initial(up), ancestors[..i].to_vec())));
                entry.1.1 += points * share;
            }
        }
    }
    ancestors.push(e);
    for kid in e.elems() {
        score(kid, ancestors, scores);
    }
    ancestors.pop();
}

/// A `div` whose only content is one `p`.
fn wraps_one_p(e: &Elem) -> bool {
    let mut kids = e.kids.iter().filter(|k| match k {
        Node::Text(t) => !t.trim().is_empty(),
        Node::Elem(_) => true,
    });
    e.name == "div"
        && matches!((kids.next(), kids.next()), (Some(Node::Elem(p)), None) if p.name == "p")
}

/// Readability's starting score: the tag's, plus 25 for a hopeful class or id, less 25 for a
/// hopeless one.
fn initial(e: &Elem) -> f64 {
    let tag = match e.name.as_str() {
        "div" => 5.0,
        "pre" | "td" | "blockquote" => 3.0,
        "address" | "ol" | "ul" | "dl" | "dd" | "dt" | "li" | "form" => -3.0,
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "th" => -5.0,
        _ => 0.0,
    };
    let hints = format!(
        "{} {}",
        e.attr("class").unwrap_or(""),
        e.attr("id").unwrap_or("")
    )
    .to_lowercase();
    let hint = |words: &[&str]| words.iter().any(|w| hints.contains(w));
    tag + if hint(&NEGATIVE) { -25.0 } else { 0.0 } + if hint(&POSITIVE) { 25.0 } else { 0.0 }
}

/// Quote levels written as `>`; a deeper quote's text stays in the last. Each level prefixes
/// every line under it again, so a 2 MB page of 250 nested quotes took a minute and 1 GB.
const MAX_QUOTES: usize = 8;

/// `e` as Markdown blocks, inside `quotes` levels of quote.
fn block(e: &Elem, base: &Url, quotes: usize, out: &mut Vec<String>) {
    match e.name.as_str() {
        name @ ("h1" | "h2" | "h3" | "h4" | "h5" | "h6") => {
            let mut line = String::new();
            inlines(&e.kids, base, &mut line);
            let line = collapse(&line);
            if !line.is_empty() {
                let level = usize::from(name.as_bytes()[1] - b'0');
                out.push(format!("{} {line}", "#".repeat(level)));
            }
        }
        "pre" => {
            let mut code = String::new();
            raw(e, &mut code);
            // HTML drops a newline right after `<pre>`.
            let code = code.strip_prefix('\n').unwrap_or(&code).trim_end();
            if !code.is_empty() {
                out.push(format!("```\n{code}\n```"));
            }
        }
        "ul" | "ol" => {
            let items: Vec<String> = (e.elems().filter(|k| k.name == "li"))
                .enumerate()
                .filter_map(|(i, li)| {
                    let mark = if e.name == "ol" {
                        format!("{}. ", i + 1)
                    } else {
                        "- ".into()
                    };
                    let mut sub = Vec::new();
                    blocks(&li.kids, base, quotes, &mut sub);
                    let body = sub.join("\n");
                    let indent = format!("\n{}", " ".repeat(mark.len()));
                    (!body.is_empty()).then(|| format!("{mark}{}", body.replace('\n', &indent)))
                })
                .collect();
            if !items.is_empty() {
                out.push(items.join("\n"));
            }
        }
        "table" => {
            let mut rows = Vec::new();
            table_rows(e, &mut rows);
            let rows: Vec<Vec<Vec<String>>> = (rows.iter())
                .map(|tr| {
                    (tr.elems().filter(|c| c.name == "td" || c.name == "th"))
                        .map(|cell| {
                            let mut sub = Vec::new();
                            blocks(&cell.kids, base, quotes, &mut sub);
                            sub
                        })
                        .collect()
                })
                .collect();
            // At most one cell of text a row lays a page out (paulgraham.com's essays sit in
            // one): its paragraphs stay paragraphs.
            if (rows.iter()).all(|r| r.iter().filter(|c| !c.is_empty()).count() <= 1) {
                out.extend(rows.into_iter().flatten().flatten());
                return;
            }
            let lines: Vec<String> = (rows.iter())
                .map(|r| {
                    (r.iter().map(|cell| cell.join(" ")))
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .filter(|row| !row.trim_matches([' ', '|']).is_empty())
                .collect();
            if !lines.is_empty() {
                out.push(lines.join("\n"));
            }
        }
        "blockquote" if quotes >= MAX_QUOTES => blocks(&e.kids, base, quotes, out),
        "blockquote" => {
            let mut sub = Vec::new();
            blocks(&e.kids, base, quotes + 1, &mut sub);
            if !sub.is_empty() {
                let joined = sub.join("\n\n");
                let quoted: Vec<String> = (joined.lines())
                    .map(|l| {
                        if l.is_empty() {
                            ">".into()
                        } else {
                            format!("> {l}")
                        }
                    })
                    .collect();
                out.push(quoted.join("\n"));
            }
        }
        "hr" => {}
        _ => blocks(&e.kids, base, quotes, out),
    }
}

/// The rows of `table`, through its sections but not into a table inside a cell.
fn table_rows<'a>(e: &'a Elem, rows: &mut Vec<&'a Elem>) {
    for kid in e.elems() {
        match kid.name.as_str() {
            "tr" => rows.push(kid),
            "thead" | "tbody" | "tfoot" => table_rows(kid, rows),
            _ => {}
        }
    }
}

/// Inline elements with no Markdown of their own, whose `<br>` ends a paragraph as their parent's
/// does: paulgraham.com's essays are one `<font>`.
const PLAIN: [&str; 2] = ["font", "span"];

/// `kids` as Markdown blocks: runs of text and inline elements become paragraphs, and a `<br>`
/// ends one.
fn blocks(kids: &[Node], base: &Url, quotes: usize, out: &mut Vec<String>) {
    let mut line = String::new();
    paragraphs(kids, base, quotes, &mut line, out);
    flush(&mut line, out);
}

/// `kids` into `line`, each paragraph it ends pushed to `out`.
fn paragraphs(kids: &[Node], base: &Url, quotes: usize, line: &mut String, out: &mut Vec<String>) {
    for kid in kids {
        match kid {
            Node::Text(t) => line.push_str(t),
            Node::Elem(e) if STRIP.contains(&e.name.as_str()) => {}
            Node::Elem(e) if e.name == "br" => flush(line, out),
            Node::Elem(e) if BLOCK.contains(&e.name.as_str()) => {
                flush(line, out);
                block(e, base, quotes, out);
            }
            Node::Elem(e) if PLAIN.contains(&e.name.as_str()) => {
                paragraphs(&e.kids, base, quotes, line, out)
            }
            Node::Elem(e) => inline(e, base, line),
        }
    }
}

/// `line` as a paragraph of `out`, when it has text.
fn flush(line: &mut String, out: &mut Vec<String>) {
    let t = collapse(line);
    if !t.is_empty() {
        out.push(t);
    }
    line.clear();
}

fn inlines(kids: &[Node], base: &Url, line: &mut String) {
    for kid in kids {
        match kid {
            Node::Text(t) => line.push_str(t),
            Node::Elem(e) if STRIP.contains(&e.name.as_str()) => {}
            // A block inside an inline element (a card's link around a div) runs on as text.
            Node::Elem(e) if e.name == "br" || BLOCK.contains(&e.name.as_str()) => {
                line.push(' ');
                inlines(&e.kids, base, line);
                line.push(' ');
            }
            Node::Elem(e) => inline(e, base, line),
        }
    }
}

/// `e` as inline Markdown: links, emphasis and code; images dropped.
fn inline(e: &Elem, base: &Url, line: &mut String) {
    let mut inner = String::new();
    inlines(&e.kids, base, &mut inner);
    let wrap = |open: &str, close: &str| {
        let core = inner.trim();
        if core.is_empty() {
            return inner.clone();
        }
        let lead = &inner[..inner.len() - inner.trim_start().len()];
        let trail = &inner[inner.trim_end().len()..];
        format!("{lead}{open}{}{close}{trail}", collapse(core))
    };
    let md = match e.name.as_str() {
        // A link within the page itself is its text.
        "a" => match (e.attr("href").map(str::trim))
            .filter(|href| !href.starts_with('#'))
            .and_then(|href| base.join(href).ok())
            .filter(|url| matches!(url.scheme(), "http" | "https"))
        {
            Some(url) => wrap("[", &format!("]({url})")),
            None => inner,
        },
        "strong" | "b" => wrap("**", "**"),
        "em" | "i" => wrap("*", "*"),
        "code" | "kbd" | "samp" | "tt" => wrap("`", "`"),
        "img" => String::new(),
        _ => inner,
    };
    line.push_str(&md);
}

/// The text below `e` exactly as written, for `<pre>`.
fn raw(e: &Elem, out: &mut String) {
    for kid in &e.kids {
        match kid {
            Node::Text(t) => out.push_str(t),
            Node::Elem(k) if k.name == "br" => out.push('\n'),
            Node::Elem(k) => raw(k, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(page: &str) -> (String, String) {
        html(page, &Url::parse("https://example.org/a/b.html").unwrap())
    }

    #[test]
    fn the_article_wins_over_a_longer_link_list_and_scripts() {
        let links: String = (0..40)
            .map(|i| {
                format!("<li><a href='/r{i}'>Related story number {i}, a long headline</a></li>")
            })
            .collect();
        let page = format!(
            "<html><head><title> Big  news &amp; more </title><script>var s = '</div><p>no';</script>\
             <style>p {{ color: red }}</style></head><body><nav><a href='/'>Home</a></nav>\
             <div class='sidebar'><ul>{links}</ul></div>\
             <article><h1>The <em>real</em> story</h1>\
             <p>First paragraph, with a comma, and a <a href='/x?a=1&amp;b=2'>link</a> in it.\
             <p>Second paragraph, also long enough to count, with <strong> bold </strong>text.</p>\
             <!-- <p>hidden comment</p> --></article><footer>(c) site</footer></body></html>"
        );
        let (title, text) = md(&page);
        assert_eq!(title, "Big news & more");
        assert_eq!(
            text,
            "# The *real* story\n\n\
             First paragraph, with a comma, and a [link](https://example.org/x?a=1&b=2) in it.\n\n\
             Second paragraph, also long enough to count, with **bold** text."
        );
    }

    /// A page whose body is one `<form>` (ASP.NET WebForms, SharePoint) keeps its text; the
    /// form's controls still go.
    #[test]
    fn a_body_wrapped_in_a_form_keeps_its_text() {
        let page = "<html><head><title>Hours</title></head><body>\
             <form id='aspnetForm' method='post'><input type='hidden' name='__VIEWSTATE' value='x'>\
             <div class='main'><h1>Hours</h1>\
             <p>The office is open Monday to Friday, from nine in the morning to five.</p>\
             <select name='lang'><option>English</option></select>\
             <textarea name='note'>Write your comment here</textarea>\
             <button type='submit'>Search</button></div></form></body></html>";
        let (title, text) = md(page);
        assert_eq!(title, "Hours");
        assert_eq!(
            text,
            "# Hours\n\nThe office is open Monday to Friday, from nine in the morning to five."
        );
    }

    #[test]
    fn blocks_lists_code_tables_and_quotes_become_markdown() {
        let page = "<body><div class='content'><p>Intro paragraph, long enough to be scored here.</p>\
             <ul><li>one<li>two <code>x</code></ul><ol><li>first</li><li>second</li></ol>\
             <pre><code>fn main() {\n    a &lt; b;\n}\n</code></pre>\
             <table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td>1</td></tr></table>\
             <blockquote><p>quoted</p><p>twice</p></blockquote>\
             <p>line one<br>line two &#8212; &#x2014; &hellip;&nbsp;end</p>\
             <p><a href='#top'>top</a> <a href='javascript:x()'>js</a> <a href='mailto:a@b'>m</a></p></div></body>";
        let (title, text) = md(page);
        assert_eq!(title, "");
        assert_eq!(
            text,
            "Intro paragraph, long enough to be scored here.\n\n- one\n- two `x`\n\n1. first\n2. second\n\n\
             ```\nfn main() {\n    a < b;\n}\n```\n\nk | v\na | 1\n\n> quoted\n>\n> twice\n\n\
             line one\n\nline two \u{2014} \u{2014} \u{2026} end\n\ntop js m"
        );
    }

    #[test]
    fn a_br_in_a_font_ends_a_paragraph() {
        // paulgraham.com's markup: an essay is one <font> in a layout table, its paragraphs
        // split by <br /><br />.
        let page = "<html><head><title>How to Do Great Work</title></head><body>\
             <table width='100%'><tr valign='top'><td bgcolor='#FF9900' width='174'><img src='m.gif'></td>\
             <td><a href='index.html'><img src='logo.gif'></a><br><br>\
             <table width='435'><tr><td><img src='t.gif' alt='How to Do Great Work'><br><br>\
             <font size='2' face='verdana'>July 2023<br /><br />If you collected lists of \
             techniques, you would find a few, and they would hold.<br /><br />The first step is \
             to <i>decide</i> what to work on, <span>and then<br />to work.</span><br /><br />\
             Curiosity, delight and the desire to do something impressive.</font>\
             </td></tr></table></td></tr></table></body></html>";
        let (title, text) = md(page);
        assert_eq!(title, "How to Do Great Work");
        assert_eq!(
            text,
            "July 2023\n\nIf you collected lists of techniques, you would find a few, and they \
             would hold.\n\nThe first step is to *decide* what to work on, and then\n\nto work.\n\n\
             Curiosity, delight and the desire to do something impressive."
        );
    }

    #[test]
    fn a_recipe_in_json_ld_leads_the_text() {
        // allrecipes' shape: the ingredients are in the page's JSON-LD alone.
        let allrecipes = r#"<html><head><title>Best Chocolate Chip Cookies</title>
<script type="application/ld+json">[{"@context":"http://schema.org","@type":["Recipe"],
"recipeIngredient":["1 cup butter, softened","2 eggs","3 cups all-purpose flour"],
"recipeInstructions":[{"@type":"HowToStep","text":"Preheat the oven to 350 degrees F."},
{"@type":"HowToStep","text":"Beat the butter and the sugars."}]}]</script>
<script type="application/ld+json">{"@type":"BreadcrumbList"}</script></head>
<body><article><p>These cookies are crisp at the edges, chewy in the middle, loved by all.</p>
</article></body></html>"#;
        assert_eq!(
            md(allrecipes).1,
            "Ingredients:\n\n- 1 cup butter, softened\n- 2 eggs\n- 3 cups all-purpose flour\n\n\
             Steps:\n\n1. Preheat the oven to 350 degrees F.\n2. Beat the butter and the sugars.\n\n\
             These cookies are crisp at the edges, chewy in the middle, loved by all."
        );
        // A recipe plugin's: in an @graph, its steps in sections, and a page with no recipe.
        let graph = r#"<script type="application/ld+json">{"@graph":[{"@type":"WebPage"},
{"@type":"Recipe","recipeIngredient":["1 egg"],"recipeInstructions":[{"@type":"HowToSection",
"itemListElement":[{"@type":"HowToStep","text":"Whisk  it."}]},"Bake it."]}]}</script>
<p>A cookie page with a recipe card, long enough to be read.</p>"#;
        assert_eq!(
            md(graph).1,
            "Ingredients:\n\n- 1 egg\n\nSteps:\n\n1. Whisk it.\n2. Bake it.\n\n\
             A cookie page with a recipe card, long enough to be read."
        );
        assert!(
            md("<p>No recipe here, only a paragraph to read.</p>")
                .1
                .starts_with("No recipe")
        );
        // A page's mainEntity, a full type URL, a step named but not texted, and HTML in a step.
        let entity = r#"<script type="application/ld+json">{"@type":"WebPage","mainEntity":
{"@type":"http://schema.org/Recipe","recipeIngredient":["1 egg"],"recipeInstructions":[
{"@type":"HowToStep","name":"Whisk it."},"<p>Bake it.</p><p>Cool it.</p>"]}}</script>
<p>A cookie page with a recipe card, long enough to be read.</p>"#;
        assert_eq!(
            md(entity).1,
            "Ingredients:\n\n- 1 egg\n\nSteps:\n\n1. Whisk it.\n2. Bake it. Cool it.\n\n\
             A cookie page with a recipe card, long enough to be read."
        );
    }

    #[test]
    fn of_two_blocks_that_score_the_same_the_first_wins_every_time() {
        let p = "<p>A paragraph, with a comma, long enough to count.</p>";
        let page = format!("<body><div>First{p}{p}</div><div>Second{p}{p}</div></body>");
        for _ in 0..20 {
            assert!(md(&page).1.starts_with("First"));
        }
    }

    #[test]
    fn the_title_falls_back_to_the_first_h1_never_an_icons_title() {
        assert_eq!(md("<h1>Only <b>heading</b></h1><p>x</p>").0, "Only heading");
        assert_eq!(md("<svg><title>Menu</title></svg><h1>Real</h1>").0, "Real");
    }

    #[test]
    fn uppercase_tags_unclosed_paragraphs_and_stray_ends_parse() {
        let (_, text) = md(
            "<BODY><DIV ID=main><P>One, two, three, four, five, six.<P>Seven, eight.</span></DIV></BODY>",
        );
        assert_eq!(text, "One, two, three, four, five, six.\n\nSeven, eight.");
    }

    #[test]
    fn rows_and_cells_without_end_tags_stay_rows() {
        let (_, text) = md("<table><tr><td>a<td>b<tr><td>c<td>d</table>");
        assert_eq!(text, "a | b\nc | d");
    }

    #[test]
    fn a_page_of_ampersands_decodes_in_linear_time() {
        let page = format!("<p>{}</p>", "&".repeat(300_000));
        let start = std::time::Instant::now();
        assert_eq!(md(&page).1.len(), 300_000);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    /// Quotes nest to `MAX_QUOTES` levels; a deeper one's text stays in the last, so a page of
    /// 250 nested quotes is read in linear time and its text kept.
    #[test]
    fn quotes_nest_to_a_bound_and_keep_their_text() {
        let article = |inner: String| format!("<body><article>{inner}</article></body>");
        let (_, text) = md(&article(
            "<blockquote><p>one</p><blockquote><p>two</p></blockquote></blockquote>".into(),
        ));
        assert_eq!(text, "> one\n>\n> > two");
        let page = article(format!("{}<p>deep</p>", "<blockquote>".repeat(250)));
        let deepest = format!("{}deep", "> ".repeat(MAX_QUOTES));
        assert_eq!(md(&page).1, deepest);
        // Each paragraph is its line and a blank one, each `MAX_QUOTES` prefixes at most.
        let n = 200_000;
        let page = article("<blockquote>".repeat(250) + &"<p>x</p>".repeat(n));
        let started = std::time::Instant::now();
        let (_, text) = md(&page);
        assert!(text.len() <= n * (4 * MAX_QUOTES + 4), "{}", text.len());
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    #[test]
    fn deep_nesting_neither_overflows_the_stack_nor_loses_text() {
        let page = format!("{}<p>deep text, kept.</p>", "<div>".repeat(100_000));
        assert!(md(&page).1.contains("deep text, kept."));
    }
}
