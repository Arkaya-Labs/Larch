//! Item 7 (toolcall-opt): an outside tool's description and its parameters' descriptions in a
//! short, regular form, made when the server connects: the purpose (the first sentences), no
//! example values (small models copy them into arguments: "ollama/react" from a description's
//! "e.g. facebook/react"), whitespace collapsed, at most `MAX_CHARS`. The original stays for
//! validation; only what the model reads changes.

use serde_json::Value;

/// The most characters a compiled tool description keeps.
pub(crate) const MAX_CHARS: usize = 200;
/// And a parameter's.
pub(crate) const MAX_PARAM_CHARS: usize = 120;

/// `text` without its examples: parenthesised or bracketed "e.g."/"for example"/"such as" asides,
/// and sentences or clauses that start with one; whitespace collapsed; cut at the last sentence
/// end within `max` characters (or the last word, with "…").
pub(crate) fn compile(text: &str, max: usize) -> String {
    let mut s = text.replace(['\n', '\r', '\t'], " ");
    // Asides in brackets that give examples.
    let aside =
        regex::Regex::new(r"(?i)\s*[\(\[][^\)\]]*\b(e\.g\.|eg\.|for example|for instance|such as|example:|examples:)[^\)\]]*[\)\]]")
            .expect("valid");
    s = aside.replace_all(&s, "").into_owned();
    // A clause or sentence that starts with an example marker, to the sentence end.
    let clause = regex::Regex::new(
        r"(?i)[,;:]?\s*\b(e\.g\.|eg\.|for example|for instance|example:|examples:)[^.]*\.?",
    )
    .expect("valid");
    s = clause.replace_all(&s, ".").into_owned();
    // Code fences and inline backticks around examples stay as text; collapse whitespace.
    let ws = regex::Regex::new(r"\s+").expect("valid");
    s = ws.replace_all(&s, " ").trim().to_owned();
    s = s.replace(" .", ".").replace("..", ".");
    if s.chars().count() <= max {
        return s;
    }
    let cut: String = s.chars().take(max).collect();
    match cut.rfind(". ") {
        Some(i) if i > max / 3 => cut[..=i].trim().to_owned(),
        _ => match cut.rfind(' ') {
            Some(i) => format!("{}…", cut[..i].trim_end()),
            None => cut,
        },
    }
}

/// `schema` with the description of every schema inside it compiled (`compile`,
/// `MAX_PARAM_CHARS`): its properties', theirs, and those in items, anyOf/oneOf/allOf and $defs.
/// Nested ones count as much as the top's: Notion's search and create-pages kept 4-5k
/// characters in them, and two such tools no longer fit a step's `OUTSIDE_CHARS` together.
pub(crate) fn compile_schema(schema: &Value) -> Value {
    let mut out = schema.clone();
    compile_within(&mut out);
    out
}

/// Compiles the descriptions of the schemas `schema` holds, and of theirs.
fn compile_within(schema: &mut Value) {
    let Some(o) = schema.as_object_mut() else {
        return;
    };
    for (key, v) in o.iter_mut() {
        match (key.as_str(), v) {
            ("properties" | "$defs" | "definitions", Value::Object(m)) => {
                m.values_mut().for_each(compile_sub)
            }
            ("items" | "anyOf" | "oneOf" | "allOf" | "prefixItems", Value::Array(a)) => {
                a.iter_mut().for_each(compile_sub)
            }
            ("items" | "additionalProperties", v @ Value::Object(_)) => compile_sub(v),
            _ => {}
        }
    }
}

fn compile_sub(s: &mut Value) {
    let short = (s.get("description").and_then(Value::as_str)).map(|d| compile(d, MAX_PARAM_CHARS));
    if let Some(d) = short {
        s["description"] = d.into();
    }
    compile_within(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_go_and_the_purpose_stays_short() {
        assert_eq!(
            compile(
                "GitHub repository in owner/repo format (e.g. \"facebook/react\").",
                200
            ),
            "GitHub repository in owner/repo format."
        );
        assert_eq!(
            compile(
                "Time zone name, for example America/New_York. Required.",
                200
            ),
            "Time zone name. Required."
        );
        let long = "Search the docs. ".repeat(30);
        let got = compile(&long, 200);
        assert!(got.chars().count() <= 200 && got.ends_with('.'), "{got}");
        assert_eq!(compile("a\n\n  b", 200), "a b");
    }

    /// Descriptions nested in objects, arrays and anyOf are compiled as the top's are; a property
    /// named `description` is a property, and the schema's own keywords stay.
    #[test]
    fn nested_descriptions_are_compiled_too() {
        let long = format!("{} (e.g. \"x\").", "Words ".repeat(40).trim_end());
        let schema = serde_json::json!({"type": "object", "properties": {
            "filters": {"type": "object", "description": long, "properties": {
                "range": {"type": "object", "properties": {
                    "start": {"type": "string", "pattern": "^\\d{4}$", "description": long}}}}},
            "parent": {"anyOf": [{"type": "object", "properties": {
                "page_id": {"type": "string", "description": long}}}]},
            "pages": {"type": "array", "items": {"type": "object", "properties": {
                "description": {"type": "string", "description": long}}}}}});
        let got = compile_schema(&schema);
        let p = &got["properties"];
        let short = compile(&long, MAX_PARAM_CHARS);
        assert!(short.chars().count() <= MAX_PARAM_CHARS && !short.contains("e.g."));
        assert_eq!(p["filters"]["description"], short.as_str());
        let start = &p["filters"]["properties"]["range"]["properties"]["start"];
        assert_eq!(start["description"], short.as_str());
        assert_eq!(start["pattern"], "^\\d{4}$");
        assert_eq!(
            p["parent"]["anyOf"][0]["properties"]["page_id"]["description"],
            short.as_str()
        );
        let item = &p["pages"]["items"]["properties"]["description"];
        assert_eq!(
            item,
            &serde_json::json!({"type": "string", "description": short})
        );
    }
}
