//! Tool calls read from the model's text in its own format, shared by `larch serve` and the agent.
//! Every error's `Display` is the sentence an agent loop feeds back to the model.

use std::fmt;

use serde_json::{Value, json};

use crate::api::{Grammar, GrammarError, json_schema_regex};

/// The tool-call format a model was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
    Qwen3,
    /// `<|python_tag|>{"name": ..., "parameters": {...}}`, or that object bare, ending the reply.
    Llama3,
    /// `[TOOL_CALLS][{"name": ..., "arguments": {...}}]`, or that array bare, ending the reply.
    Mistral,
    /// Inkling 2's messages: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,
    /// "args":{...}}<|end_message|>`, one message per call, beside `<|content_thinking|>` and
    /// `<|content_text|>` messages of the same shape. Every tag is a control token, so the reply
    /// reads only where the detokenizer keeps [`INKLING_TAGS`] as text.
    Inkling,
}

/// Inkling 2's message tags, which its replies are read by: llama.cpp's `preserved_tokens`
/// (common/parsers/inkling.cpp) with the text kinds Splash's chat format reads as the answer's.
pub const INKLING_TAGS: [&str; 7] = [
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_tool_error|>",
    "<|content_invoke_tool_text|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
];

impl Family {
    /// The text that opens a call in this format.
    pub fn marker(self) -> &'static str {
        match self {
            Self::Qwen3 => "<tool_call>",
            Self::Llama3 => "<|python_tag|>",
            Self::Mistral => "[TOOL_CALLS]",
            Self::Inkling => "<|content_invoke_tool_json|>",
        }
    }

    /// Each text that opens a call in this format, with its closing tag: the marker, and for
    /// Qwen3 the `<function-call>` wrapper Qwen3-30B-A3B writes at times outside the grammar
    /// (twice on the M5's source-files-003, 31 Oct), read as the same call.
    pub fn wrappers(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Qwen3 => &[
                ("<tool_call>", "</tool_call>"),
                ("<function-call>", "</function-call>"),
            ],
            Self::Llama3 => &[("<|python_tag|>", "")],
            Self::Mistral => &[("[TOOL_CALLS]", "")],
            Self::Inkling => &[("<|content_invoke_tool_json|>", "<|end_message|>")],
        }
    }
}

/// One tool call: the name and its arguments object.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// The tool's name as the model wrote it.
    pub name: String,
    /// The arguments object.
    pub args: Value,
}

/// A malformed reply; `Display` is the message the model gets back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The reply ended inside `<think>`, usually at `max_tokens`.
    UnclosedThinking,
    /// The call is not JSON; `byte` is where the JSON parser stopped, counted from 1.
    InvalidJson {
        /// serde_json's reason, without its position.
        message: String,
        /// Position in the call's JSON text.
        byte: usize,
    },
    /// The call is JSON but has no `name` string.
    MissingName,
    /// The arguments are not a JSON object.
    ArgsNotObject,
    /// More calls than were read: a second one in a format that takes one per reply, or a Qwen
    /// call after `Parsed::later`'s that does not read. The first is still returned.
    MultipleCalls,
    /// An XML-body call's `<function=...>` or `<parameter=...>` tag is broken; says how.
    MalformedTag(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnclosedThinking => f.write_str(
                "the reply ran out of space inside <think>; answer or call a tool directly, without thinking first",
            ),
            Self::InvalidJson { message, byte } => {
                write!(f, "tool call is not valid JSON: {message} at byte {byte}")
            }
            Self::MissingName => f.write_str("tool call has no \"name\" string"),
            Self::ArgsNotObject => f.write_str("tool call arguments must be a JSON object"),
            Self::MultipleCalls => f.write_str(
                "only one tool call per reply: the first call is used and the others are ignored",
            ),
            Self::MalformedTag(why) => write!(f, "tool call has a malformed tag: {why}"),
        }
    }
}

/// One reply, split into thinking, the text before the call, the call and what went wrong.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    /// The text up to `</think>`, after the reply's `<think>` or the template's; it goes to the
    /// audit log, never to the user.
    pub thinking: Option<String>,
    /// The text before the call, or the whole answer when there is no call.
    pub text_before: String,
    /// The first call, if any.
    pub call: Option<ToolCall>,
    /// The calls after the first, in order: the Qwen format writes one block per call.
    pub later: Vec<ToolCall>,
    /// What was wrong, if anything; `MultipleCalls` comes with `call` set.
    pub error: Option<ParseError>,
}

/// Splits a reply into thinking, text and its tool calls in `family`'s format.
/// `think_open` says the rendered prompt ends with an open `<think>`: the reply up to its first
/// `</think>` is then thinking. Otherwise only a leading `<think>...</think>` block is.
pub fn parse(family: Family, text: &str, think_open: bool) -> Parsed {
    if family == Family::Inkling {
        return parse_inkling(text);
    }
    let mut parsed = Parsed::default();
    let mut rest = text.trim_start();
    let block = if think_open {
        Some(rest)
    } else {
        rest.strip_prefix("<think>")
    };
    if let Some(block) = block {
        // Markers inside a closed block are the model considering a call, not making one.
        let Some((thinking, after)) = block.split_once("</think>") else {
            parsed.thinking = Some(block.trim().into());
            parsed.error = Some(ParseError::UnclosedThinking);
            return parsed;
        };
        parsed.thinking = Some(thinking.trim().into());
        rest = after;
    }
    let Some((before, found)) = find_call(family, rest) else {
        parsed.text_before = rest.trim().into();
        return parsed;
    };
    parsed.text_before = before.trim().into();
    match found {
        Ok((call, mut more)) => {
            parsed.call = Some(call);
            // Qwen writes one block per call; a later one that does not read ends them, also when
            // a good one follows it.
            while family == Family::Qwen3
                && let Some(Some((before, Ok((call, next))))) = more.map(|r| find_call(family, r))
                && !(family.wrappers().iter()).any(|(open, _)| before.contains(open))
            {
                parsed.later.push(call);
                more = next;
            }
            parsed.error = more.map(|_| ParseError::MultipleCalls);
        }
        Err(error) => parsed.error = Some(error),
    }
    parsed
}

/// An Inkling 2 reply, message by message: its thinking messages joined as the thinking, its text
/// messages (`content_text`, a tool error, a text-invoked tool's text) joined as the answer, and its
/// call messages in order. A reply without tags (a detokenizer that hid them) is all answer. A
/// message cut off at `max_tokens` keeps what it holds: thinking as `UnclosedThinking`, a call's JSON
/// checked as written.
fn parse_inkling(text: &str) -> Parsed {
    const OPEN: &str = "<|message_model|>";
    const END: &str = "<|end_message|>";
    let mut parsed = Parsed::default();
    let (mut thinking, mut answer, mut calls) = (Vec::new(), Vec::new(), Vec::new());
    // Text before the first message has no kind; the model opens its own messages, so it is the
    // answer's, as a reply without tags is.
    let mut messages = text.split(OPEN);
    if let Some(head) = messages.next().map(str::trim)
        && !head.is_empty()
    {
        answer.push(head.trim_end_matches(END).trim());
    }
    for message in messages {
        let (body, closed) = match message.split_once(END) {
            Some((body, rest)) => {
                // Text between messages (none in the format) stays the answer's.
                let rest = rest.trim();
                if !rest.is_empty() {
                    answer.push(rest);
                }
                (body, true)
            }
            None => (message, false),
        };
        if let Some(t) = body.strip_prefix("<|content_thinking|>") {
            thinking.push(t.trim());
            if !closed {
                parsed.error = Some(ParseError::UnclosedThinking);
            }
        } else if let Some((_name, json)) = body.split_once(Family::Inkling.marker()) {
            calls.push(first_json(json).and_then(|(v, _)| to_call(v)));
        } else {
            let t = [
                "<|content_text|>",
                "<|content_tool_error|>",
                "<|content_invoke_tool_text|>",
            ]
            .iter()
            .find_map(|tag| body.strip_prefix(tag))
            .unwrap_or(body);
            answer.push(t.trim());
        }
    }
    parsed.thinking = (!thinking.is_empty()).then(|| thinking.join("\n\n"));
    parsed.text_before = answer.join("\n\n");
    let mut calls = calls.into_iter();
    match calls.next() {
        Some(Ok(call)) => parsed.call = Some(call),
        Some(Err(e)) => parsed.error = Some(e),
        None => {}
    }
    for call in calls {
        match call {
            Ok(c) if parsed.call.is_some() => parsed.later.push(c),
            // A later call that does not read ends them, as in the Qwen format.
            _ => {
                parsed.error.get_or_insert(ParseError::MultipleCalls);
                break;
            }
        }
    }
    parsed
}

/// A call, and the text after it when another call follows (empty outside the Qwen format).
type Found<'a> = Result<(ToolCall, Option<&'a str>), ParseError>;

/// The text before the call, and the call; `None` when the reply has no call.
fn find_call(family: Family, text: &str) -> Option<(&str, Found<'_>)> {
    match family {
        Family::Qwen3 => {
            // Prose may name the tag before the call, so each tag is tried in turn; a broken
            // call is reported only when no tag holds a call, at the first tag whose text starts
            // like a call (a tag in prose is followed by words), else at the first tag.
            // ponytail: a call that opens with <function= and no <tool_call> (Qwen3-Coder does at
            // times, llama.cpp reads it) is text; read it when IR-07d's parse rate shows it.
            fn more(rest: &str) -> Option<&str> {
                let wrappers = Family::Qwen3.wrappers().iter();
                wrappers
                    .clone()
                    .any(|(o, _)| rest.contains(o))
                    .then_some(rest)
            }
            let (mut braced, mut first) = (None, None);
            let mut tags: Vec<(usize, &str, &str)> = (family.wrappers().iter())
                .flat_map(|&(open, close)| {
                    text.match_indices(open)
                        .map(move |(at, _)| (at, open, close))
                })
                .collect();
            tags.sort_unstable();
            for (at, tag, close) in tags {
                let after = &text[at + tag.len()..];
                if let Some(body) = after.trim_start().strip_prefix("<function=") {
                    let call = xml_call(body).map(|(c, rest)| (c, more(rest)));
                    if call.is_ok() {
                        return Some((&text[..at], call));
                    }
                    braced.get_or_insert((&text[..at], call));
                    continue;
                }
                // The JSON ends where its value ends, so a "</tool_call>" inside a string
                // argument is text. A broken call is read up to its closing tag; an unclosed tag
                // (the step hit max_tokens) still gets its JSON checked.
                let json = first_json(after).or_else(|e| match after.split_once(close) {
                    Some((inside, _)) => first_json(inside),
                    None => Err(e),
                });
                let call = json.and_then(|(v, rest)| Ok((to_call(v)?, more(rest))));
                if call.is_ok() {
                    return Some((&text[..at], call));
                }
                let slot = if after.trim_start().starts_with('{') {
                    &mut braced
                } else {
                    &mut first
                };
                slot.get_or_insert((&text[..at], call));
            }
            braced.or(first)
        }
        Family::Mistral => {
            // A link or any other bracket after the call is text, not a second call.
            let more = |rest: &str| {
                rest.contains(family.marker())
                    || first_json(rest).is_ok_and(|(v, _)| v.is_array() && call_shaped(&v))
            };
            let Some((before, after)) = text.split_once(family.marker()) else {
                return bare_call(text, more);
            };
            let call = first_json(after).and_then(|(v, rest)| first_call(v, more(rest)));
            Some((before, call))
        }
        // parse() reads Inkling's messages itself.
        Family::Inkling => None,
        Family::Llama3 => {
            // Other JSON after the call (a result in a code fence) is not a second call.
            let more = |rest: &str| {
                let rest = rest.trim_start().trim_start_matches(';');
                rest.contains(family.marker()) || starts_call(rest)
            };
            if let Some((before, after)) = text.split_once(family.marker()) {
                let call = first_json(after)
                    .and_then(|(v, rest)| Ok((to_call(v)?, more(rest).then_some(""))));
                return Some((before, call));
            }
            if names_call(text)
                && let Err(e) = first_json(text)
            {
                return Some(("", Err(e)));
            }
            bare_call(text, more)
        }
    }
}

/// The Qwen3.5/3.6 XML body after its `<function=`: the call and the text after `</function>`.
/// Each value is the text between the newlines that wrap it, kept as a string (`coerce` types it).
fn xml_call(body: &str) -> Result<(ToolCall, &str), ParseError> {
    let (name, mut rest) = tag_name(body, "<function=")?;
    if name.is_empty() {
        return Err(ParseError::MissingName);
    }
    let mut args = serde_json::Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix("</function>") {
            let call = ToolCall {
                name: name.into(),
                args: Value::Object(args),
            };
            return Ok((call, after));
        }
        let Some(param) = rest.strip_prefix("<parameter=") else {
            return Err(ParseError::MalformedTag(match rest.is_empty() {
                true => format!("<function={name}> has no </function>"),
                false => format!("<function={name}> holds text outside <parameter=...> tags"),
            }));
        };
        let (key, after) = tag_name(param, "<parameter=")?;
        if key.is_empty() {
            return Err(ParseError::MalformedTag("<parameter=> has no name".into()));
        }
        let Some((value, after)) = after.split_once("</parameter>") else {
            let why = format!("<parameter={key}> has no </parameter>");
            return Err(ParseError::MalformedTag(why));
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        args.insert(key.into(), value.strip_suffix('\n').unwrap_or(value).into());
        rest = after;
    }
}

/// The name in a tag that `open` starts, and the text after its `>`.
fn tag_name<'a>(text: &'a str, open: &str) -> Result<(&'a str, &'a str), ParseError> {
    match text.split_once('>') {
        Some((name, rest)) if !name.contains(['<', '\n']) => Ok((name.trim(), rest)),
        _ => Err(ParseError::MalformedTag(format!(
            "{open} is not closed by >"
        ))),
    }
}

/// Whether `text` starts with a call: JSON shaped like one, or a broken one (cut off at
/// `max_tokens`, say).
fn starts_call(text: &str) -> bool {
    match first_json(text) {
        Ok((value, _)) => call_shaped(&value),
        Err(_) => names_call(text),
    }
}

/// Whether `text` starts with an object and names a tool and its arguments: when it is not JSON,
/// a broken call; a name alone (a package manifest) is an answer.
fn names_call(text: &str) -> bool {
    text.trim_start().starts_with('{')
        && text.contains("\"name\"")
        && (text.contains("\"parameters\"") || text.contains("\"arguments\""))
}

/// A call whose marker never reached the text (the detokenizer drops control tokens such as
/// `<|python_tag|>` and `[TOOL_CALLS]`): an object with a name and arguments, or an array that
/// starts with one, at the start of a line and with only another call after it. Any other JSON,
/// such as an example in a code fence, is an answer.
fn bare_call(text: &str, more: impl Fn(&str) -> bool) -> Option<(&str, Found<'_>)> {
    line_starts(text).find_map(|at| {
        let line = text[at..].trim_start();
        if !line.starts_with(['{', '[']) {
            return None;
        }
        let (value, rest) = first_json(line).ok()?;
        let more = more(rest);
        let last = more || rest.trim().is_empty();
        (call_shaped(&value) && last).then(|| (&text[..at], first_call(value, more)))
    })
}

/// The byte where each line of `text` starts, where a bare call may begin.
pub fn line_starts(text: &str) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0).chain(text.match_indices('\n').map(|(i, _)| i + 1))
}

/// Whether `value` is an object with a name and arguments, or an array that starts with one.
fn call_shaped(value: &Value) -> bool {
    let head = match value {
        Value::Array(calls) => calls.first(),
        one => Some(one),
    };
    head.is_some_and(|h| {
        h.get("name").is_some() && (h.get("arguments").is_some() || h.get("parameters").is_some())
    })
}

/// The first call in an object or an array of them, and whether another follows.
fn first_call(value: Value, more: bool) -> Found<'static> {
    let (one, more) = match value {
        Value::Array(calls) => {
            let more = more || calls.len() > 1;
            (calls.into_iter().next().unwrap_or(Value::Null), more)
        }
        one => (one, more),
    };
    Ok((to_call(one)?, more.then_some("")))
}

/// The first JSON value in `text` and the text after it.
fn first_json(text: &str) -> Result<(Value, &str), ParseError> {
    let text = text.trim();
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    let error = match values.next() {
        Some(Ok(value)) => return Ok((value, &text[values.byte_offset()..])),
        Some(Err(e)) => e,
        None => serde_json::from_str::<Value>(text).expect_err("no value in blank text"),
    };
    let shown = error.to_string();
    let message = shown.rsplit_once(" at line ").map_or(&*shown, |(m, _)| m);
    // serde_json counts lines from 1 and bytes within the line; the model gets one offset.
    let before: usize = text
        .split_inclusive('\n')
        .take(error.line().saturating_sub(1))
        .map(str::len)
        .sum();
    Err(ParseError::InvalidJson {
        message: message.into(),
        byte: before + error.column(),
    })
}

fn to_call(value: Value) -> Result<ToolCall, ParseError> {
    let Value::Object(mut o) = value else {
        return Err(ParseError::MissingName);
    };
    let Some(Value::String(name)) = o.remove("name") else {
        return Err(ParseError::MissingName);
    };
    // Each family has its own key; accepting both costs nothing and saves a retry.
    let args = o
        .remove("arguments")
        .or_else(|| o.remove("parameters"))
        .or_else(|| o.remove("args"))
        .unwrap_or_else(|| json!({}));
    if !args.is_object() {
        return Err(ParseError::ArgsNotObject);
    }
    Ok(ToolCall { name, args })
}

/// Whether a model's chat template asks for the XML body (`<function=NAME>` and
/// `<parameter=KEY>` tags inside `<tool_call>`: Qwen3.5/3.6, Qwen3-Coder), as llama.cpp detects it.
pub fn xml_body(template: &str) -> bool {
    template.contains("<function=") && template.contains("<parameter=")
}

/// Types an XML-body call's string arguments: each one whose schema in `tools` (OpenAI function
/// specs) admits more than a string is read as JSON (number, integer, boolean, null, object,
/// array); a value that does not read stays a string, for the agent's validation to report.
/// JSON-body arguments come typed, so it changes nothing there, except a non-string value the
/// model quoted (`"3"` for an integer), which it reads as the agent's `ToolRegistry::coerce`
/// does for integers. Callers run it after `parse`.
pub fn coerce(call: &mut ToolCall, tools: &[Value]) {
    let Some(f) = (tools.iter().map(|t| &t["function"])).find(|f| f["name"] == call.name.as_str())
    else {
        return;
    };
    let params = &f["parameters"];
    let (Some(props), Some(args)) = (params["properties"].as_object(), call.args.as_object_mut())
    else {
        return;
    };
    for (key, value) in args {
        let (Some(schema), Some(text)) = (props.get(key), value.as_str()) else {
            continue;
        };
        let types = types(schema, params, 0);
        // A value of a type the schema does not admit stays as the model wrote it.
        let admits = |v: &Value| {
            let t = json_type(v);
            (types.as_ref()).is_none_or(|ts| {
                t != "string" && (ts.contains(&t) || t == "integer" && ts.contains(&"number"))
            })
        };
        if let Ok(v) = serde_json::from_str(text)
            && admits(&v)
        {
            *value = v;
        }
    }
}

/// How [`grammar`] holds a reply to a tool call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Free text until the family's marker, then one call through its close, re-armed after it.
    Auto,
    /// A call from the first answer token; more in the Qwen format.
    Required,
    /// [`Mode::Required`] with this one tool.
    Named(String),
}

/// The grammar that makes the model's call in `family`'s format (the XML body when `xml`, see
/// [`xml_body`]; Qwen3 only) parse as a call of one of `tools` (OpenAI function specs) whose
/// arguments fit its `parameters`.
///
/// - Auto triggers at `family.marker()`, and the pattern runs through the call's close: Qwen3's
///   `</tool_call>`, Mistral's `]`, Llama3's object, after which its reply ends on a stop token
///   (a triggered grammar re-arms after the call rather than forcing the end). A bare Llama3
///   object or Mistral array, written without its marker, stays unconstrained: a `{` trigger
///   would constrain every JSON in prose.
/// - Required is untriggered: the reply is the call, after at most two newlines; the Llama3 and
///   Mistral marker is optional, as the detokenizer may hide it. With thinking on, the caller sets
///   `trigger` to `Some("</think>".into())`, so the reply thinks freely and the call follows the
///   think block; a prompt that already ends in `</think>` needs nothing more.
/// - JSON body: `{"name": "<tool>", "arguments": <json_schema_regex(parameters)>}` (Llama3:
///   `"parameters"`; Mistral: the template's `"id"` may follow) as the template spaces it.
///   XML body: `<function=NAME>`, the properties as `<parameter=KEY>` in schema order with the
///   required ones present (any names when the schema lists no properties), a value that admits
///   a string any text without `</parameter>` (one of a string `enum` or `const`), any other
///   `json_schema_regex` of its schema, then `</function>`.
///   A tool without `parameters` takes an empty object, as llama-server reads it.
///
/// An unknown `Named` tool, no tools, a tool without a name and a schema `json_schema_regex`
/// refuses are errors, never an unconstrained grammar.
// ponytail: under Auto with thinking on, a marker written inside the think block arms the pattern
// too; arm only after </think> when IR-07d's real-model rows show calls forced inside thinking.
pub fn grammar(
    family: Family,
    tools: &[Value],
    xml: bool,
    mode: Mode,
) -> Result<Grammar, GrammarError> {
    if xml && family != Family::Qwen3 {
        return Err(GrammarError::Unsupported(format!(
            "the XML tool-call body in the {family:?} format"
        )));
    }
    let (mut calls, mut names) = (Vec::new(), Vec::new());
    for t in tools {
        let f = &t["function"];
        let Some(name) = f["name"].as_str() else {
            return Err(GrammarError::Unsupported(format!(
                "tool without a name: {t}"
            )));
        };
        if matches!(&mode, Mode::Named(n) if n != name) {
            continue;
        }
        names.push(name);
        let mut params = (f.get("parameters").cloned())
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        if let Some(o) = params.as_object_mut() {
            o.entry("type").or_insert_with(|| "object".into());
        }
        // Inkling's template writes a call's arguments with their keys sorted (`dictsort`), so its
        // model writes them so too.
        if family == Family::Inkling {
            sort_properties(&mut params);
        }
        // Checked for the XML body too, so both bodies refuse the same schemas.
        let args = json_schema_regex(&params)?;
        calls.push(match xml {
            true => xml_call_regex(name, &params)?,
            false => {
                let name_json = escape(&json!(name).to_string());
                if family == Family::Inkling {
                    // As the template writes it, compact. The message's name before the marker is
                    // the caller's: Auto triggers at the marker, Required writes it.
                    format!(
                        r#"\{{"name":{name_json},"args":{args}\}}{}"#,
                        escape("<|end_message|>")
                    )
                } else {
                    let (key, id) = match family {
                        Family::Llama3 => ("parameters", ""),
                        Family::Mistral => ("arguments", r#"(?:, "id": "[a-zA-Z0-9]{9}")?"#),
                        _ => ("arguments", ""),
                    };
                    format!(r#"\{{"name": {name_json}, "{key}": {args}{id}\}}"#)
                }
            }
        });
    }
    if calls.is_empty() {
        return Err(match mode {
            Mode::Named(n) => GrammarError::Unsupported(format!("no tool named {n}")),
            _ => GrammarError::Empty,
        });
    }
    let call = format!("(?:{})", calls.join("|"));
    if family == Family::Inkling {
        // Each call is a message of its own. Auto triggers at the marker, after the message's
        // name, and holds the body; a required call opens its message, named for its tool, from
        // the first answer token, and more may follow.
        if mode == Mode::Auto {
            return Ok(Grammar {
                trigger: Some(family.marker().into()),
                pattern: call,
            });
        }
        let (open, marker) = (escape("<|message_model|>"), escape(family.marker()));
        let named: Vec<String> = (names.iter().zip(&calls))
            .map(|(name, body)| format!("{open}{}{marker}{body}", escape(name)))
            .collect();
        let one = format!("(?:{})", named.join("|"));
        return Ok(Grammar {
            trigger: None,
            pattern: match mode {
                Mode::Required => format!("{one}+"),
                _ => one,
            },
        });
    }
    let pattern = match family {
        Family::Qwen3 => format!(r"\n{call}\n</tool_call>"),
        Family::Mistral => format!(r" ?\[{call}\]"),
        _ => call,
    };
    let marker = escape(family.marker());
    Ok(match (mode, family) {
        (Mode::Auto, _) => Grammar {
            trigger: Some(family.marker().into()),
            pattern,
        },
        // Qwen writes one block per call, so a reply may hold several; a forced tool takes one.
        (Mode::Required, Family::Qwen3) => Grammar {
            trigger: None,
            pattern: format!(r"\n{{0,2}}{marker}{pattern}(?:\n{marker}{pattern})*"),
        },
        (_, Family::Qwen3) => Grammar {
            trigger: None,
            pattern: format!(r"\n{{0,2}}{marker}{pattern}"),
        },
        _ => Grammar {
            trigger: None,
            pattern: format!(r"\n{{0,2}}(?:{marker})?{pattern}"),
        },
    })
}

/// `schema`'s `properties`, and those of every object nested in it, in key order.
fn sort_properties(schema: &mut Value) {
    match schema {
        Value::Object(o) => {
            if let Some(Value::Object(props)) = o.get_mut("properties") {
                let mut sorted: Vec<(String, Value)> = std::mem::take(props).into_iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                props.extend(sorted);
            }
            o.values_mut().for_each(sort_properties);
        }
        Value::Array(a) => a.iter_mut().for_each(sort_properties),
        _ => {}
    }
}

/// One tool's XML body, `<function=NAME>` through `</function>`.
fn xml_call_regex(name: &str, params: &Value) -> Result<String, GrammarError> {
    let (free, open) = (no_close_tag(), format!("<function={}>\n", escape(name)));
    let Some(props) = params["properties"].as_object() else {
        if ["$ref", "anyOf", "oneOf"]
            .iter()
            .any(|k| params.get(k).is_some())
        {
            let why = format!("the XML body of {name} needs its properties at the top level");
            return Err(GrammarError::Unsupported(why));
        }
        let param = format!("<parameter=[^<>\n]+>\n{free}\n</parameter>\n");
        return Ok(format!("{open}(?:{param})*</function>"));
    };
    let required = params["required"].as_array().map_or(&[][..], Vec::as_slice);
    let mut out = open;
    for (key, schema) in props {
        let strings = types(schema, params, 0).is_some_and(|ts| ts.contains(&"string"));
        let value = match (strings, &schema["enum"], &schema["const"]) {
            (true, Value::Array(e), _) if e.iter().all(Value::is_string) => {
                let e: Vec<String> = e.iter().filter_map(Value::as_str).map(escape).collect();
                format!("(?:{})", e.join("|"))
            }
            (true, _, Value::String(c)) => escape(c),
            (true, ..) => free.clone(),
            (false, ..) => {
                // Referenced from the whole schema, so the value's local `$ref`s still resolve.
                let mut root = params.clone();
                let pointer = key.replace('~', "~0").replace('/', "~1");
                root["$ref"] = format!("#/properties/{pointer}").into();
                json_schema_regex(&root)?
            }
        };
        let param = format!("<parameter={}>\n{value}\n</parameter>\n", escape(key));
        out += &match required.contains(&json!(key)) {
            true => param,
            false => format!("(?:{param})?"),
        };
    }
    Ok(out + "</function>")
}

/// Any text without `</parameter>`: plain chars, or a `<` and partial close tags each broken by a
/// char that does not continue it, ending in at most one partial tag.
fn no_close_tag() -> String {
    let rest: Vec<char> = "/parameter>".chars().collect();
    let prefix = (rest[..rest.len() - 1].iter().rev())
        .fold(String::new(), |inner, c| format!("(?:{c}{inner})?"));
    let broken = rest
        .iter()
        .rev()
        .fold(String::new(), |inner, c| match inner.is_empty() {
            true => format!("[^{c}<]"),
            false => format!("(?:[^{c}<]|{c}{inner})"),
        });
    let lead = format!("<(?:{prefix}<)*");
    format!("(?:[^<]|{lead}{broken})*(?:{lead}{prefix})?")
}

/// `text` as a regex that matches it literally: regex_syntax::escape's set.
// ponytail: a copy of regex_syntax::escape, as this lane's task may not touch the facade's
// Cargo.toml or the engine's grammar.rs; call it when either links or re-exports it.
fn escape(text: &str) -> String {
    text.chars().fold(String::new(), |mut out, c| {
        if r"\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
        out
    })
}

/// The JSON types `schema` admits, from its `type`, `enum` or `const` values, or the members of
/// its `anyOf`/`oneOf`, with local `$ref`s resolved in `root`. None takes any value: the schema
/// names no type, or its references nest past eight.
fn types<'a>(schema: &'a Value, root: &'a Value, depth: u8) -> Option<Vec<&'a str>> {
    if depth > 8 {
        return None;
    }
    if let Some(r) = schema["$ref"].as_str() {
        let target = r.strip_prefix('#').and_then(|p| root.pointer(p))?;
        return types(target, root, depth + 1);
    }
    if let Some(Value::Array(members)) = schema.get("anyOf").or(schema.get("oneOf")) {
        let each = members.iter().map(|m| types(m, root, depth + 1));
        return each.collect::<Option<Vec<_>>>().map(|ts| ts.concat());
    }
    match (&schema["type"], &schema["enum"], schema.get("const")) {
        (Value::String(t), ..) => Some(vec![t.as_str()]),
        (Value::Array(ts), ..) => Some(ts.iter().filter_map(Value::as_str).collect()),
        (_, Value::Array(e), _) => Some(e.iter().map(json_type).collect()),
        (_, _, Some(c)) => Some(vec![json_type(c)]),
        _ => None,
    }
}

/// The JSON Schema type name of `v`.
fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// The format family for a model architecture (`ModelConfig.arch` or GGUF `general.architecture`),
/// unless config.toml names one. Qwen 3.5 (`qwen35`, `qwen35moe`) writes the Qwen format with the
/// XML body (`xml_body`). Mistral GGUFs say `llama`, so Mistral needs the override. `None`
/// for an architecture with no row and no override, or an override that names no family: that
/// model's chats run without tools.
pub fn family_for(arch: &str, config_override: Option<&str>) -> Option<Family> {
    match config_override
        .unwrap_or(arch)
        .to_ascii_lowercase()
        .as_str()
    {
        "qwen2" | "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe" => Some(Family::Qwen3),
        "llama" | "llama3" => Some(Family::Llama3),
        "mistral" => Some(Family::Mistral),
        "inkling2" | "inkling2_loop" => Some(Family::Inkling),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The XML body as vLLM's Qwen3 parser reads it (vllm/parser/qwen3.py, which serves
    // `qwen3_coder`, at 059e2de6d81750551dc8fdc82365eb031bab1086) and llama.cpp's Qwen3-Coder
    // format writes it (common/chat.cpp, common/parsers/qwen3-coder.cpp at
    // d854d98bae598388ecea1ed84396587e5edc326d).
    const XML: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

    fn qwen(text: &str) -> Parsed {
        parse(Family::Qwen3, text, false)
    }

    #[test]
    fn qwen_35_ggufs_take_the_qwen_format() {
        // #369: a Qwen 3.5 chat ran with no tools.
        for arch in ["qwen35", "qwen35moe"] {
            assert_eq!(family_for(arch, None), Some(Family::Qwen3), "{arch}");
        }
    }

    #[test]
    fn a_function_call_wrapper_is_a_qwen_call() {
        // The M5's source-files-003 reply, twice (31 Oct): printed, not run.
        let text = "<function-call>\n{\"name\": \"recall\", \"arguments\": {\"query\": \"my name\", \"source\": \"fact\"}}\n</function-call>";
        let p = qwen(text);
        assert_eq!(p.error, None);
        let call = p.call.unwrap();
        assert_eq!(
            (call.name.as_str(), &call.args["query"]),
            ("recall", &json!("my name"))
        );
        // Its closing tag ends a broken call's JSON, as </tool_call> does.
        let broken = qwen("<function-call>{\"name\": \"recall\", }</function-call>");
        assert!(
            matches!(broken.error, Some(ParseError::InvalidJson { .. })),
            "{broken:?}"
        );
        // Two calls in either wrapper are one reply's calls, in order.
        let two = qwen(&format!(
            "{text}\n<tool_call>\n{{\"name\": \"recall\", \"arguments\": {{\"query\": \"x\"}}}}\n</tool_call>"
        ));
        assert_eq!(two.later.len(), 1, "{two:?}");
    }

    #[test]
    fn xml_body_parses() {
        let p = qwen(&format!("Let me check.\n{XML}"));
        assert_eq!(p.text_before, "Let me check.");
        assert_eq!(p.error, None);
        let call = ToolCall {
            name: "get_weather".into(),
            args: json!({"city": "New York", "days": "3"}),
        };
        assert_eq!(p.call, Some(call));
        // One wrapping newline goes on each side; other closing tags inside a value are text.
        let text = "<tool_call>\n<function=write>\n<parameter=body>\n\nline</tool_call>\n</function>\n\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            qwen(text).call.unwrap().args,
            json!({"body": "\nline</tool_call>\n</function>\n"})
        );
        // No parameters, and cut off after </function> (as an unclosed JSON call still reads).
        let p = qwen("<tool_call>\n<function=now>\n</function>");
        assert_eq!((p.call.unwrap().args, p.error), (json!({}), None));
    }

    #[test]
    fn xml_parse_errors() {
        let err = |text: &str| qwen(text).error;
        let tag = |why: &str| Some(ParseError::MalformedTag(why.into()));
        assert_eq!(
            err(&format!("<think>hmm {XML}")),
            Some(ParseError::UnclosedThinking)
        );
        assert_eq!(
            err("<tool_call>\n<function=>\n</function>\n</tool_call>"),
            Some(ParseError::MissingName)
        );
        let p = qwen(&format!("{XML}\n{XML}"));
        assert_eq!((p.error, p.later.len()), (None, 1));
        assert_eq!(p.call.unwrap().name, "get_weather");
        // A broken call between two good ones ends them there, and is said to be left out.
        let p = qwen(&format!("{XML}\n<tool_call>\n<function=f>\n{XML}"));
        assert_eq!(
            (p.error.clone(), p.later.len()),
            (Some(ParseError::MultipleCalls), 0)
        );
        // A later call that does not read is left out, and said to be.
        let p = qwen(&format!("{XML}\n<tool_call>\n<function=f>\n"));
        assert_eq!(
            (p.error, p.later.len()),
            (Some(ParseError::MultipleCalls), 0)
        );
        // InvalidJson and ArgsNotObject have no XML form: no JSON is read and the arguments are
        // always an object. The new error is for a broken tag.
        assert_eq!(
            err(
                "<tool_call>\n<function=get_weather\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
            ),
            tag("<function= is not closed by >")
        );
        assert_eq!(
            err("<tool_call>\n<function=f>\n<parameter=city\nParis\n</parameter>\n</function>"),
            tag("<parameter= is not closed by >")
        );
        assert_eq!(
            err("<tool_call>\n<function=f>\n<parameter=>\nParis\n</parameter>\n</function>"),
            tag("<parameter=> has no name")
        );
        assert_eq!(
            err("<tool_call>\n<function=f>\n<parameter=city>\nParis\n</function>\n</tool_call>"),
            tag("<parameter=city> has no </parameter>")
        );
        assert_eq!(
            err("<tool_call>\n<function=f>\n<parameter=city>\nParis\n</parameter>\n"),
            tag("<function=f> has no </function>")
        );
        assert_eq!(
            err("<tool_call>\n<function=f>\n{\"city\": \"Paris\"}\n</function>\n</tool_call>"),
            tag("<function=f> holds text outside <parameter=...> tags")
        );
        assert_eq!(
            ParseError::MalformedTag("x".into()).to_string(),
            "tool call has a malformed tag: x"
        );
    }

    #[test]
    fn coerce_reads_each_type() {
        let tools = [
            json!({"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {
                "n": {"type": "number"}, "i": {"type": "integer"}, "b": {"type": "boolean"},
                "z": {"type": "null"}, "o": {"type": "object"}, "a": {"type": "array"},
                "s": {"type": "string"}, "e": {"enum": ["1", "2"]}, "u": {"type": ["integer", "string"]},
                "any": {}, "bad": {"type": "integer"}}}}}),
        ];
        let mut call = ToolCall {
            name: "f".into(),
            args: json!({"n": "1.5", "i": "-2", "b": "true", "z": "null", "o": "{\"k\": [1]}",
                "a": "[1, \"x\"]", "s": "42", "e": "1", "u": "7", "any": "{}", "bad": "three",
                "extra": "5"}),
        };
        coerce(&mut call, &tools);
        let typed = json!({"n": 1.5, "i": -2, "b": true, "z": null, "o": {"k": [1]}, "a": [1, "x"],
            "s": "42", "e": "1", "u": 7, "any": {}, "bad": "three", "extra": "5"});
        assert_eq!(call.args, typed);
        // A JSON-body call comes typed, and an unknown tool's call is left alone.
        coerce(&mut call, &tools);
        assert_eq!(call.args, typed);
        let mut other = ToolCall {
            name: "g".into(),
            args: json!({"i": "1"}),
        };
        coerce(&mut other, &tools);
        assert_eq!(other.args, json!({"i": "1"}));
    }

    /// A union with string keeps a value only as a type it admits; `anyOf`, `oneOf` and local
    /// `$ref`s count, as Pydantic writes `Optional[str]`.
    #[test]
    fn coerce_keeps_only_admitted_types() {
        let tools = [
            json!({"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {
                "zip": {"type": ["string", "null"]},
                "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "ref": {"$ref": "#/$defs/n"}, "rs": {"$ref": "#/$defs/s"}},
            "$defs": {"n": {"type": "integer"}, "s": {"type": "string"}}}}}),
        ];
        let mut c = call(
            "f",
            json!({"zip": "12345", "opt": "123", "ref": "4", "rs": "5"}),
        );
        coerce(&mut c, &tools);
        assert_eq!(
            c.args,
            json!({"zip": "12345", "opt": "123", "ref": 4, "rs": "5"})
        );
        let mut c = call("f", json!({"zip": "null", "opt": "null"}));
        coerce(&mut c, &tools);
        assert_eq!(c.args, json!({"zip": null, "opt": null}));
    }

    #[test]
    fn xml_grammar_takes_unions_enums_and_free_form_objects() {
        let params = json!({"type": "object", "properties": {
            "unit": {"type": "string", "enum": ["c", "f"]},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["unit"]});
        let tools = [
            json!({"type": "function", "function": {"name": "t", "parameters": params}}),
            json!({"type": "function", "function": {"name": "free", "parameters": {}}}),
        ];
        let g = grammar(Family::Qwen3, &tools, true, Mode::Required).unwrap();
        let xml = |c: &ToolCall| render(Family::Qwen3, true, c);
        assert!(ok(
            &g,
            &xml(&call("t", json!({"unit": "c", "note": "raw text"})))
        ));
        assert!(!ok(&g, &xml(&call("t", json!({"unit": "kelvin"})))));
        let free = call("free", json!({"any": "thing", "n": "2"}));
        assert!(ok(&g, &xml(&free)));
        assert_eq!(parse(Family::Qwen3, &xml(&free), false).call, Some(free));
        // A schema whose properties sit behind a top-level reference has no XML form here.
        let behind =
            json!({"$ref": "#/$defs/a", "$defs": {"a": {"type": "object", "properties": {}}}});
        let tool = json!({"type": "function", "function": {"name": "r", "parameters": behind}});
        let g = grammar(Family::Qwen3, &[tool], true, Mode::Auto);
        assert!(matches!(g, Err(GrammarError::Unsupported(_))), "{g:?}");
    }

    /// Mistral's template writes the call's id after its arguments; the grammar lets it.
    #[test]
    fn mistral_calls_may_carry_an_id() {
        let tools = tools();
        let text = r#"[TOOL_CALLS][{"name": "now", "arguments": {}, "id": "call00001"}]"#;
        for mode in [Mode::Auto, Mode::Required] {
            let g = grammar(Family::Mistral, &tools, false, mode).unwrap();
            assert!(ok(&g, text));
        }
        assert_eq!(
            parse(Family::Mistral, text, false).call.unwrap().name,
            "now"
        );
    }

    #[test]
    fn xml_body_detects_the_template() {
        // Qwen3's template asks for the JSON body; Qwen3.5's, as Qwen3-Coder's, for the XML one.
        let qwen3 =
            r#"{{- '<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}"#;
        let qwen35 = r"{{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}{%- for args_name, args_value in tool_call.arguments|items %}{{- '<parameter=' + args_name + '>\n' }}";
        assert!(!xml_body(qwen3));
        assert!(xml_body(qwen35));
    }

    const BODIES: [(Family, bool); 5] = [
        (Family::Qwen3, false),
        (Family::Qwen3, true),
        (Family::Llama3, false),
        (Family::Mistral, false),
        (Family::Inkling, false),
    ];

    fn tools() -> Vec<Value> {
        let weather = json!({"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
            "limit": {"$ref": "#/$defs/count"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["city", "days"], "$defs": {"count": {"type": "integer", "minimum": 0}}});
        vec![
            json!({"type": "function", "function": {"name": "get_weather", "description": "w",
                "parameters": weather}}),
            json!({"type": "function", "function": {"name": "now", "description": "t"}}),
        ]
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            args,
        }
    }

    /// `c` as the family's template writes a call: a string value raw in the XML body, any
    /// other as JSON.
    fn render(family: Family, xml: bool, c: &ToolCall) -> String {
        let (name, args) = (json!(c.name), &c.args);
        match (family, xml) {
            (Family::Qwen3, true) => {
                let params: String = (args.as_object().unwrap().iter())
                    .map(|(k, v)| match v {
                        Value::String(s) => format!("<parameter={k}>\n{s}\n</parameter>\n"),
                        v => format!("<parameter={k}>\n{v}\n</parameter>\n"),
                    })
                    .collect();
                let name = &c.name;
                format!("<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>")
            }
            (Family::Qwen3, false) => {
                format!("<tool_call>\n{{\"name\": {name}, \"arguments\": {args}}}\n</tool_call>")
            }
            (Family::Llama3, _) => {
                format!("<|python_tag|>{{\"name\": {name}, \"parameters\": {args}}}")
            }
            (Family::Mistral, _) => {
                format!("[TOOL_CALLS][{{\"name\": {name}, \"arguments\": {args}}}]")
            }
            (Family::Inkling, _) => {
                // The template's `dictsort` order.
                let mut sorted: Vec<_> = args.as_object().unwrap().iter().collect();
                sorted.sort_by_key(|(k, _)| *k);
                let args: serde_json::Map<String, Value> = sorted
                    .into_iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let args = Value::Object(args);
                format!(
                    "<|message_model|>{}<|content_invoke_tool_json|>{{\"name\":{name},\"args\":{args}}}<|end_message|>",
                    c.name
                )
            }
        }
    }

    fn ok(g: &Grammar, text: &str) -> bool {
        crate::api::grammar::matches(g, text).unwrap()
    }

    #[test]
    fn grammar_round_trips_every_family_body_and_mode() {
        let tools = tools();
        let calls = [
            call(
                "get_weather",
                json!({"city": "New York", "days": 3, "metric": true, "limit": 5, "tags": ["a", "b"]}),
            ),
            call("get_weather", json!({"city": "Oslo\nwest <b>", "days": 0})),
            call("now", json!({})),
        ];
        for (family, xml) in BODIES {
            for c in &calls {
                for mode in [Mode::Auto, Mode::Required, Mode::Named(c.name.clone())] {
                    let g = grammar(family, &tools, xml, mode.clone()).unwrap();
                    let rendered = render(family, xml, c);
                    let mut texts = vec![format!("Checking.\n{rendered}")];
                    if mode != Mode::Auto {
                        texts = vec![rendered.clone()];
                        // The control token the detokenizer may hide: the bare form passes too.
                        if !matches!(family, Family::Qwen3 | Family::Inkling) {
                            texts.push(rendered.replacen(family.marker(), "", 1));
                        }
                    }
                    for text in texts {
                        let what = format!("{family:?} xml={xml} {mode:?}: {text:?}");
                        assert!(ok(&g, &text), "{what}");
                        let mut parsed = parse(family, &text, false).call.expect(&what);
                        coerce(&mut parsed, &tools);
                        assert_eq!(&parsed, c, "{what}");
                    }
                }
            }
        }
        // With thinking on, Required is armed at </think>; a marker in the thinking is free.
        let mut g = grammar(Family::Qwen3, &tools, true, Mode::Required).unwrap();
        g.trigger = Some("</think>".into());
        let text = format!(
            "<think>\nuse <tool_call> maybe\n</think>\n\n{}",
            render(Family::Qwen3, true, &calls[0])
        );
        assert!(ok(&g, &text));
        let mut parsed = parse(Family::Qwen3, &text, false).call.unwrap();
        coerce(&mut parsed, &tools);
        assert_eq!(parsed, calls[0]);
    }

    /// The Qwen format writes one block per call: Auto re-arms after each and Required repeats,
    /// and the parser reads every call in order.
    #[test]
    fn qwen_replies_hold_several_calls() {
        let tools = tools();
        let (a, b) = (
            call("get_weather", json!({"city": "Oslo", "days": 1})),
            call("now", json!({})),
        );
        for xml in [false, true] {
            let two = format!(
                "{}\n{}",
                render(Family::Qwen3, xml, &a),
                render(Family::Qwen3, xml, &b)
            );
            for mode in [Mode::Auto, Mode::Required] {
                let g = grammar(Family::Qwen3, &tools, xml, mode.clone()).unwrap();
                assert!(ok(&g, &two), "xml={xml} {mode:?}");
            }
            let g = grammar(Family::Qwen3, &tools, xml, Mode::Named("now".into())).unwrap();
            assert!(!ok(&g, &two), "xml={xml}: a forced tool takes one call");
            let mut p = qwen(&format!("Both.\n{two}"));
            assert_eq!((p.text_before.as_str(), p.error.clone()), ("Both.", None));
            for c in p.call.iter_mut().chain(&mut p.later) {
                coerce(c, &tools);
            }
            assert_eq!(
                (p.call, p.later),
                (Some(a.clone()), vec![b.clone()]),
                "xml={xml}"
            );
        }
    }

    #[test]
    fn grammar_rejects_wrong_name_missing_arg_wrong_type() {
        let tools = tools();
        let bad = [
            call("get_wether", json!({"city": "Oslo", "days": 1})),
            call("get_weather", json!({"city": "Oslo"})),
            call("get_weather", json!({"city": "Oslo", "days": "three"})),
            call(
                "get_weather",
                json!({"city": "Oslo", "days": 1, "limit": true}),
            ),
        ];
        for (family, xml) in BODIES {
            for mode in [
                Mode::Auto,
                Mode::Required,
                Mode::Named("get_weather".into()),
            ] {
                let g = grammar(family, &tools, xml, mode.clone()).unwrap();
                for c in &bad {
                    let text = render(family, xml, c);
                    assert!(!ok(&g, &text), "{family:?} xml={xml} {mode:?}: {text:?}");
                }
            }
            // Named holds the reply to that one tool.
            let g = grammar(family, &tools, xml, Mode::Named("now".into())).unwrap();
            let weather = call("get_weather", json!({"city": "Oslo", "days": 1}));
            assert!(!ok(&g, &render(family, xml, &weather)));
        }
    }

    #[test]
    fn xml_string_values_stop_only_at_the_close_tag() {
        let g = grammar(Family::Qwen3, &tools(), true, Mode::Required).unwrap();
        let text = |city: &str| {
            render(
                Family::Qwen3,
                true,
                &call("get_weather", json!({"city": city, "days": 1})),
            )
        };
        for city in [
            "",
            "<",
            "<<",
            "</par",
            "</</parameter",
            "a\n</parameter b",
            "</parameter",
        ] {
            assert!(ok(&g, &text(city)), "{city:?}");
        }
        for city in ["a</parameter>b", "<</parameter>", "</parameter>"] {
            assert!(!ok(&g, &text(city)), "{city:?}");
        }
    }

    #[test]
    fn grammar_errors() {
        let tools = tools();
        let unsupported =
            |r: Result<Grammar, GrammarError>| matches!(r, Err(GrammarError::Unsupported(_)));
        let named = Mode::Named("nope".into());
        assert!(unsupported(grammar(Family::Qwen3, &tools, false, named)));
        assert_eq!(
            grammar(Family::Qwen3, &[], false, Mode::Auto),
            Err(GrammarError::Empty)
        );
        assert!(unsupported(grammar(
            Family::Llama3,
            &tools,
            true,
            Mode::Auto
        )));
        let nameless = json!({"type": "function", "function": {"description": "x"}});
        assert!(unsupported(grammar(
            Family::Qwen3,
            &[nameless],
            false,
            Mode::Auto
        )));
        // json_schema_regex's refusals hold for both bodies, also on a string-only schema the XML
        // body writes as free text.
        let all_of =
            json!({"type": "object", "properties": {"a": {"allOf": [{"type": "integer"}]}}});
        let missing =
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]});
        for params in [all_of, missing] {
            let tool = json!({"type": "function", "function": {"name": "f", "parameters": params}});
            for xml in [false, true] {
                let g = grammar(Family::Qwen3, std::slice::from_ref(&tool), xml, Mode::Auto);
                assert!(unsupported(g), "{params} xml={xml}");
            }
        }
    }

    /// Inkling 2's reply as its template lays a turn out (data/inkling2.jinja in larch-model): its
    /// messages read by kind, as llama.cpp's parser (common/parsers/inkling.cpp) reads them.
    #[test]
    fn inkling_replies_read_by_message() {
        let ink = |t: &str| parse(Family::Inkling, t, false);
        let think = "<|message_model|><|content_thinking|>Need the weather.<|end_message|>";
        let text = "<|message_model|><|content_text|>Checking.<|end_message|>";
        let a = "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|>";
        let b = "<|message_model|>now<|content_invoke_tool_json|>{\"name\":\"now\",\"args\":{}}<|end_message|>";
        let p = ink(&format!("{think}{text}{a}{b}"));
        assert_eq!(p.thinking.as_deref(), Some("Need the weather."));
        assert_eq!(p.text_before, "Checking.");
        assert_eq!(p.call, Some(call("get_weather", json!({"city": "Oslo"}))));
        assert_eq!(p.later, [call("now", json!({}))]);
        assert_eq!(p.error, None);
        // Reasoning may come again after text; both parts join.
        let p = ink(&format!(
            "{think}{text}{think}<|message_model|><|content_text|>Done.<|end_message|>"
        ));
        assert_eq!(
            p.thinking.as_deref(),
            Some("Need the weather.\n\nNeed the weather.")
        );
        assert_eq!(
            (p.text_before.as_str(), p.call),
            ("Checking.\n\nDone.", None)
        );
        // Cut off at max_tokens inside its thinking.
        let p = ink("<|message_model|><|content_thinking|>Still think");
        assert_eq!(p.thinking.as_deref(), Some("Still think"));
        assert_eq!(p.error, Some(ParseError::UnclosedThinking));
        // A broken call, and text whose tags a detokenizer hid.
        let p = ink("<|message_model|>x<|content_invoke_tool_json|>{\"name\":");
        assert!(
            matches!(p.error, Some(ParseError::InvalidJson { .. })),
            "{p:?}"
        );
        assert_eq!(ink("plain answer").text_before, "plain answer");
        for arch in ["inkling2", "inkling2_loop"] {
            assert_eq!(family_for(arch, None), Some(Family::Inkling));
        }
    }
}
