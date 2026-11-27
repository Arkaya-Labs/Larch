//! Chat templates (architecture 8): the GGUF's Jinja `tokenizer.chat_template`, rendered with minijinja
//! the way llama.cpp's `common_chat_templates_apply` renders it.

use std::borrow::Cow;
use std::time::{SystemTime, UNIX_EPOCH};

use larch_records::LoadError;
use minijinja::value::{Kwargs, Rest, Value as Jinja, ValueKind};
use minijinja::{Environment, Error, ErrorKind, context};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::Tokenizer;
use crate::gguf::{Gguf, Value};

/// `LoadError::Template` codes.
pub mod code {
    /// The template does not compile.
    pub const SYNTAX: u16 = 1;
}

/// Template instructions per render: a 1000-message conversation takes well under 1 %, and a template
/// that loops forever fails in about half a second instead of hanging the caller.
const FUEL: u64 = 10_000_000;

/// llama.cpp's template for a GGUF that has none.
const CHATML: &str = "{%- for message in messages -%}\n  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n  {{- '<|im_start|>assistant\n' -}}\n{%- endif -%}";

/// One chat message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// One tool call of an assistant message.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ToolCall {
    /// The tool's name.
    pub name: String,
    /// The arguments: a JSON object, or its JSON text as OpenAI sends it, which the render parses.
    pub arguments: Json,
    /// What a `tool` message's `tool_call_id` names; empty (or absent when read) is no ID, as in
    /// llama.cpp.
    #[serde(default)]
    pub id: String,
}

/// A [`Message`] with the tool fields the templates read, for [`ChatTemplate::render_tools`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// The text; empty for an assistant message that only calls tools.
    pub content: String,
    /// An assistant message's calls; empty is none.
    pub tool_calls: Vec<ToolCall>,
    /// The call a `tool` message answers; empty is none.
    pub tool_call_id: String,
}

/// A message as llama.cpp's `common_chat_msg::to_json_oaicompat` hands it to the template, keys in
/// its order: templates print it with `tojson` and `items()`.
#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<Function<'a, Call<'a>>>>,
}

/// OpenAI's wrapper around a tool call or a tool spec.
#[derive(Serialize)]
struct Function<'a, F> {
    r#type: &'static str,
    function: F,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
}

#[derive(Serialize)]
struct Call<'a> {
    name: &'a str,
    arguments: Cow<'a, Json>,
}

#[derive(Serialize)]
struct Spec<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Json,
}

/// A compiled chat template and the token texts it may print.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    add_bos: bool,
    add_eos: bool,
    /// The GGUF has no template, so ChatML stands in, as in llama.cpp.
    pub fallback: bool,
    /// `date_string` for templates that print today's date (Llama 3.x), `%d %b %Y`.
    pub date: String,
}

impl ChatTemplate {
    /// The GGUF's `tokenizer.chat_template`, or ChatML when it has none.
    pub fn from_gguf(g: &Gguf, tok: &Tokenizer) -> Result<ChatTemplate, LoadError> {
        match g.get("tokenizer.chat_template").and_then(Value::as_str) {
            Some(src) if !src.is_empty() && src != "chatml" => ChatTemplate::new(src, tok),
            _ => Ok(ChatTemplate {
                fallback: true,
                ..ChatTemplate::new(CHATML, tok)?
            }),
        }
    }

    /// Compiles `src` with llama.cpp's Jinja settings: trim and lstrip blocks, the trailing newline
    /// kept, Python string methods, `raise_exception`.
    // ponytail: fuel bounds a template's time, not its memory; a hostile GGUF template can still
    // allocate without limit while rendering. Upgrade: cap the output when GGUFs come from untrusted
    // sources.
    pub fn new(src: &str, tok: &Tokenizer) -> Result<ChatTemplate, LoadError> {
        let mut env = Environment::new();
        env.set_fuel(Some(FUEL));
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_filter("tojson", tojson);
        env.add_template_owned("chat", without_generation_tags(src).into_owned())
            .map_err(|_| LoadError::Template(code::SYNTAX))?;
        let text = |id: Option<u32>| {
            id.map(|i| String::from_utf8_lossy(tok.decode_piece(i)).into_owned())
                .unwrap_or_default()
        };
        let days = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() / 86_400);
        Ok(ChatTemplate {
            env,
            bos: text(tok.bos()),
            eos: text(tok.eos()),
            add_bos: tok.add_bos(),
            add_eos: tok.add_eos(),
            fallback: false,
            // ponytail: UTC date; llama.cpp uses local time, so renders differ between local and UTC
            // midnight. Upgrade: take the date from the caller when bench's prompt-hash check trips.
            date: date_string(days as i64),
        })
    }

    /// The prompt for `messages`, ending with the assistant header when `add_generation_prompt`. A
    /// leading BOS (trailing EOS) is dropped when the tokenizer adds it, as llama.cpp does.
    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        let messages = messages
            .iter()
            .map(|m| Msg {
                role: &m.role,
                content: &m.content,
                tool_call_id: None,
                tool_calls: None,
            })
            .collect();
        self.apply(messages, Vec::new(), add_generation_prompt, None)
    }

    /// [`render`](Self::render) with tool calls and results in `messages` and `tools`, each an OpenAI
    /// function spec (`{"type": "function", "function": {"name", "description", "parameters"}}`),
    /// in the shape llama.cpp hands the template: a spec as llama-server reads it (a missing
    /// description is empty, missing parameters `{}`), a message as `to_json_oaicompat` builds it.
    /// With no tools and no tool fields it renders exactly as `render`.
    // ponytail: llama.cpp steps the Qwen 3, Llama 3.x and tool-calling Mistral templates do not need
    // here: keeping string arguments as strings for a template whose probe finds no object arguments
    // (this parses them for every template, so text that is not JSON fails even there), renaming
    // `developer` to `system`, merging the system message into the next message for a template
    // without a system role (m7's, which raises here on a system message), and a GGUF's
    // `tokenizer.chat_template.tool_use` in place of the template when tools are given. Upgrade: the
    // merge when a caller sends m7 a system message; the others when a model that needs them joins
    // bench/models.toml.
    pub fn render_tools(
        &self,
        messages: &[ToolMessage],
        tools: &[Json],
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        self.render_tools_thinking(messages, tools, add_generation_prompt, None)
    }

    /// [`render_tools`](Self::render_tools) with the template's `enable_thinking` set to
    /// `thinking`; None is the default of llama.cpp's `common_chat_templates_apply`, true. The
    /// fixture `tests/data/qwen3.jinja` answers false with an empty think block after the
    /// generation prompt; `tests/data/llama31.jinja` and `tests/data/no-roundtrip/mistral.jinja`
    /// do not read it.
    // ponytail: `render_tools` keeps its form without `thinking` (None) for its caller in
    // crates/larch/src/api.rs, as Rust has no default arguments. Upgrade: fold `render_tools` into
    // this method when #167 passes `GenParams.thinking` there.
    // ponytail: llama-server's own default is false for a template it finds no thinking markers in;
    // None stays true for those. Upgrade: when a pinned GGUF's template reads `enable_thinking`
    // without such markers.
    pub fn render_tools_thinking(
        &self,
        messages: &[ToolMessage],
        tools: &[Json],
        add_generation_prompt: bool,
        thinking: Option<bool>,
    ) -> Result<String, Error> {
        let messages = messages
            .iter()
            .map(|m| {
                let calls = m
                    .tool_calls
                    .iter()
                    .map(|c| {
                        let arguments = match &c.arguments {
                            Json::String(s) => {
                                Cow::Owned(serde_json::from_str(s).map_err(|e| {
                                    Error::new(
                                        ErrorKind::InvalidOperation,
                                        format!("tool call arguments are not JSON: {e}"),
                                    )
                                })?)
                            }
                            a => Cow::Borrowed(a),
                        };
                        Ok(Function {
                            r#type: "function",
                            function: Call {
                                name: &c.name,
                                arguments,
                            },
                            id: Some(c.id.as_str()).filter(|id| !id.is_empty()),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Msg {
                    role: &m.role,
                    content: &m.content,
                    tool_call_id: Some(m.tool_call_id.as_str()).filter(|id| !id.is_empty()),
                    tool_calls: Some(calls).filter(|c| !c.is_empty()),
                })
            })
            .collect::<Result<_, Error>>()?;
        let no_parameters = Json::Object(Default::default());
        let tools = tools
            .iter()
            .map(|t| {
                let f = &t["function"];
                let description = f.get("description").map_or(Some(""), Json::as_str);
                match (t["type"].as_str(), f["name"].as_str(), description) {
                    (Some("function"), Some(name), Some(description)) => Ok(Function {
                        r#type: "function",
                        function: Spec {
                            name,
                            description,
                            parameters: f.get("parameters").unwrap_or(&no_parameters),
                        },
                        id: None,
                    }),
                    _ => Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("not an OpenAI function spec: {t}"),
                    )),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.apply(messages, tools, add_generation_prompt, thinking)
    }

    /// `messages` and `tools` through the template with llama.cpp's other variables.
    fn apply(
        &self,
        messages: Vec<Msg>,
        tools: Vec<Function<Spec>>,
        add_generation_prompt: bool,
        thinking: Option<bool>,
    ) -> Result<String, Error> {
        let ctx = context! {
            messages,
            bos_token => self.bos,
            eos_token => self.eos,
            // llama.cpp's `common_chat_templates_inputs::enable_thinking` defaults to true.
            enable_thinking => thinking.unwrap_or(true),
            date_string => self.date,
        };
        // llama.cpp defines `tools` only when there are some.
        let ctx = if tools.is_empty() {
            ctx
        } else {
            context! { tools, ..ctx }
        };
        // llama.cpp leaves `add_generation_prompt` undefined rather than false.
        let ctx = if add_generation_prompt {
            context! { add_generation_prompt => true, ..ctx }
        } else {
            ctx
        };
        let mut out = self.env.get_template("chat")?.render(ctx)?;
        if self.add_bos && out.starts_with(&self.bos) {
            out.drain(..self.bos.len());
        }
        if self.add_eos && out.ends_with(&self.eos) {
            out.truncate(out.len() - self.eos.len());
        }
        Ok(out)
    }
}

/// `src` with transformers' `{% generation %}` and `{% endgeneration %}` tags (which mark the
/// assistant's tokens for training masks) as `{% if true %}` and `{% endif %}`, so the text between
/// renders as llama.cpp renders it: it ignores both tags. The whitespace markers are kept, so trim
/// works as written. Inkling 2's template wraps each assistant message in them.
fn without_generation_tags(src: &str) -> Cow<'_, str> {
    if !src.contains("generation") {
        return Cow::Borrowed(src);
    }
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(at) = rest.find("{%") {
        out.push_str(&rest[..at]);
        let tag = &rest[at..];
        let Some(end) = tag.find("%}") else {
            break;
        };
        let inner = &tag[2..end];
        let (open, body) = match inner.strip_prefix(['-', '+']) {
            Some(b) => (&inner[..1], b),
            None => ("", inner),
        };
        let (body, close) = match body.strip_suffix(['-', '+']) {
            Some(b) => (b, &body[b.len()..]),
            None => (body, ""),
        };
        let word = match body.trim() {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        match word {
            Some(w) => out.push_str(&format!("{{%{open} {w} {close}%}}")),
            None => out.push_str(&tag[..end + 2]),
        }
        rest = &tag[end + 2..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// llama.cpp's Jinja `tojson`: Python's separators (`, ` and `: `; `,` and newlines with `indent`)
/// unless `separators` names them, keys in the value's order (byte order with `sort_keys`), floats
/// as C++ writes them, strings escaped as JSON and nothing more (no HTML escaping, UTF-8 kept unless
/// `ensure_ascii`). Its arguments as llama.cpp takes them: `ensure_ascii`, `indent`, `separators`
/// then `sort_keys`, by keyword or position; an indent that is negative or not an integer is none,
/// and `separators` that is not a list is the default. Inkling 2's template passes both.
fn tojson(value: &Jinja, args: Rest<Jinja>, kwargs: Kwargs) -> Result<String, Error> {
    if args.len() > 4 {
        return Err(Error::new(
            ErrorKind::TooManyArguments,
            "tojson takes at most 4 arguments",
        ));
    }
    let pos = |i: usize| args.get(i).cloned();
    let (ensure_ascii, indent, separators, sort_keys) = (pos(0), pos(1), pos(2), pos(3));
    let ascii = kwargs
        .get::<Option<Jinja>>("ensure_ascii")?
        .or(ensure_ascii)
        .is_some_and(|v| v.is_true());
    let indent = kwargs
        .get::<Option<Jinja>>("indent")?
        .or(indent)
        .filter(Jinja::is_integer)
        .and_then(|v| v.as_i64())
        // llama.cpp casts to `int`, so a huge indent wraps there instead of exhausting memory.
        .map(|n| n as i32)
        .and_then(|n| usize::try_from(n).ok());
    let sort = kwargs
        .get::<Option<Jinja>>("sort_keys")?
        .or(sort_keys)
        .is_some_and(|v| v.is_true());
    // llama.cpp reads the first two items of a list as strings; anything else keeps the default.
    let given: Vec<String> = match kwargs.get::<Option<Jinja>>("separators")?.or(separators) {
        Some(v) if v.kind() == ValueKind::Seq => v.try_iter()?.map(|s| s.to_string()).collect(),
        _ => Vec::new(),
    };
    kwargs.assert_all_used()?;
    let item = given.first().cloned().unwrap_or_else(|| match indent {
        Some(_) => ",".into(),
        None => ", ".into(),
    });
    let key = given.get(1).cloned().unwrap_or_else(|| ": ".into());
    let f = Format {
        indent,
        ascii,
        sort,
        item: &item,
        key: &key,
    };
    let mut out = String::new();
    write_json(&mut out, value, &f, 0)?;
    Ok(out)
}

/// How [`tojson`] writes: its arguments, resolved.
struct Format<'a> {
    indent: Option<usize>,
    ascii: bool,
    sort: bool,
    item: &'a str,
    key: &'a str,
}

fn write_json(out: &mut String, v: &Jinja, f: &Format, depth: usize) -> Result<(), Error> {
    let (indent, ascii) = (f.indent, f.ascii);
    let map = match v.kind() {
        ValueKind::Seq | ValueKind::Iterable => false,
        ValueKind::Map => true,
        ValueKind::String => {
            write_str(out, v.as_str().unwrap_or_default(), ascii);
            return Ok(());
        }
        kind => {
            out.push_str(&match kind {
                ValueKind::Undefined | ValueKind::None => "null".into(),
                ValueKind::Bool => v.is_true().to_string(),
                ValueKind::Number if v.is_integer() => v.to_string(),
                ValueKind::Number => format_g(f64::try_from(v.clone())?),
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("tojson cannot write a {kind}"),
                    ));
                }
            });
            return Ok(());
        }
    };
    // Tool specs nest a few levels; a value that holds itself would recurse until the stack overflows.
    if depth == 128 {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson nests deeper than 128 levels",
        ));
    }
    let newline = |out: &mut String, depth: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', n * depth));
        }
    };
    out.push(if map { '{' } else { '[' });
    let mut items: Vec<Jinja> = v.try_iter()?.collect();
    if map && f.sort {
        // Python's json.dumps(sort_keys=True), as llama.cpp sorts: UTF-8 byte order of the keys.
        items.sort_by_cached_key(|k| k.to_string());
    }
    let mut empty = true;
    for item in items {
        if !empty {
            out.push_str(f.item);
        }
        empty = false;
        newline(out, depth + 1);
        if map {
            write_str(out, &item.to_string(), ascii);
            out.push_str(f.key);
            write_json(out, &v.get_item(&item)?, f, depth + 1)?;
        } else {
            write_json(out, &item, f, depth + 1)?;
        }
    }
    if !empty {
        newline(out, depth);
    }
    out.push(if map { '}' } else { ']' });
    Ok(())
}

/// `s` as a JSON string: serde_json escapes exactly llama.cpp's set (`"`, `\` and control
/// characters), and `ascii` also escapes the rest as UTF-16 `\uXXXX`, as `ensure_ascii` does.
fn write_str(out: &mut String, s: &str, ascii: bool) {
    for c in Json::from(s).to_string().chars() {
        if ascii && !c.is_ascii() {
            for u in c.encode_utf16(&mut [0; 2]) {
                out.push_str(&format!("\\u{u:04x}"));
            }
        } else {
            out.push(c);
        }
    }
}

/// `x` as C++'s `ostream << double` writes it, which is printf's `%g`: 6 significant digits, the
/// exponent form below 1e-4 and from 1e6, trailing zeros dropped.
// ponytail: a NaN is `nan` whatever its sign, as macOS writes it; glibc writes `-nan` when the sign
// bit is set, as on x86 after inf - inf. Upgrade: when a template prints a NaN.
fn format_g(x: f64) -> String {
    if !x.is_finite() {
        return x.to_string().to_lowercase();
    }
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        }
    };
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` writes an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` writes an integer exponent");
    if (-4..6).contains(&exp) {
        trim(format!("{x:.*}", (5 - exp) as usize))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa.to_owned()), exp.abs())
    }
}

/// Days since 1970-01-01 as `%d %b %Y` (Howard Hinnant's `civil_from_days`).
fn date_string(days: i64) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{d:02} {} {y}", MONTHS[m as usize - 1])
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_string() {
        assert_eq!(super::date_string(0), "01 Jan 1970");
        assert_eq!(super::date_string(11_016), "29 Feb 2000");
        assert_eq!(super::date_string(20_732), "06 Oct 2026");
    }
}
