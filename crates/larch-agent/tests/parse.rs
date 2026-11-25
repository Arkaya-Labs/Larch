//! Parser fixtures per format family, the family table and argument validation. Every `Display`
//! string is asserted exactly: the model reads it.

use larch_agent::parse::{
    Family, ParseError, Parsed, ToolCall, ValidateError, family_for, parse, validate,
};
use larch_agent::tool::{Permission, ToolSpec};
use serde_json::json;

fn read_call() -> ToolCall {
    ToolCall {
        name: "read_file".into(),
        args: json!({"path": "src/lib.rs"}),
    }
}

fn call_only(family: Family, text: &str) -> Parsed {
    let p = parse(family, text, false);
    assert_eq!(p.error, None, "{text}");
    p
}

#[test]
fn each_family_reads_its_own_wrapper() {
    let cases = [
        (
            Family::Qwen3,
            "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}\n</tool_call>",
        ),
        (
            Family::Llama3,
            "<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}",
        ),
        (
            Family::Llama3,
            "{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}",
        ),
        (
            Family::Mistral,
            "[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}]",
        ),
    ];
    for (family, text) in cases {
        let p = call_only(family, text);
        assert_eq!(p.call, Some(read_call()), "{text}");
        assert_eq!(p.text_before, "");
        assert_eq!(p.thinking, None);
    }
}

#[test]
fn a_call_whose_marker_the_detokenizer_dropped_is_still_a_call() {
    // `[TOOL_CALLS]` and `<|python_tag|>` are control tokens, which add no text to a reply.
    let log = ToolCall {
        name: "log".into(),
        args: json!({"branch": "main", "n": 1}),
    };
    let cases = [
        (
            Family::Mistral,
            "[{\"name\": \"log\", \"arguments\": {\"branch\": \"main\", \"n\": 1}}]",
            "",
        ),
        (
            Family::Mistral,
            "I will look.\n[{\"name\": \"log\", \"arguments\": {\"branch\": \"main\", \"n\": 1}}]",
            "I will look.",
        ),
        (
            Family::Llama3,
            "Sure.\n{\"name\": \"log\", \"parameters\": {\"branch\": \"main\", \"n\": 1}}",
            "Sure.",
        ),
    ];
    for (family, text, before) in cases {
        let p = call_only(family, text);
        assert_eq!(p.call.as_ref(), Some(&log), "{text}");
        assert_eq!(p.text_before, before);
    }
    let two = "[{\"name\": \"log\", \"arguments\": {\"branch\": \"main\", \"n\": 1}}, {\"name\": \"log\", \"arguments\": {}}]";
    let p = parse(Family::Mistral, two, false);
    assert_eq!(
        (p.call, p.error),
        (Some(log), Some(ParseError::MultipleCalls))
    );
    // JSON that is not shaped like a call, that does not start a line, or that text follows (a
    // code fence, an explanation), stays an answer.
    for (family, text) in [
        (Family::Mistral, "[{\"name\": \"Alice\", \"age\": 3}]"),
        (Family::Mistral, "[1, 2, 3]"),
        (
            Family::Llama3,
            "Call it as {\"name\": \"log\", \"parameters\": {}} next time.",
        ),
        (
            Family::Llama3,
            "Here is the format:\n```json\n{\"name\": \"write_file\", \"parameters\": {\"path\": \"a\", \"text\": \"x\"}}\n```",
        ),
        (
            Family::Mistral,
            "Here is the format:\n```json\n[{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}]\n```",
        ),
        (
            Family::Llama3,
            "Two ways:\n{\"name\": \"read_file\", \"parameters\": {\"path\": \"a\"}} reads it, and the other way writes it.",
        ),
        (
            Family::Llama3,
            "The loop sends this:\n```\n{\"name\": \"read_file\", \"parameters\": {\"path\": \"a\"}}\n{\"result\": \"ok\"}\n```",
        ),
    ] {
        let p = call_only(family, text);
        assert_eq!(p.call, None, "{text}");
        assert_eq!(p.text_before, text);
    }
}

#[test]
fn a_missing_brace_is_invalid_json_with_the_byte() {
    let cases = [
        (
            Family::Qwen3,
            "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}</tool_call>",
            "tool call is not valid JSON: EOF while parsing an object at byte 57",
        ),
        (
            Family::Llama3,
            "{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}",
            "tool call is not valid JSON: EOF while parsing an object at byte 58",
        ),
        (
            Family::Mistral,
            "[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}]",
            "tool call is not valid JSON: expected `,` or `}` at byte 59",
        ),
    ];
    for (family, text, shown) in cases {
        let p = parse(family, text, false);
        assert_eq!(p.call, None, "{text}");
        let error = p.error.expect(text);
        assert!(matches!(error, ParseError::InvalidJson { .. }), "{error:?}");
        assert_eq!(error.to_string(), shown);
    }
}

#[test]
fn a_call_without_a_name_or_with_bad_arguments_says_which() {
    let p = parse(
        Family::Qwen3,
        "<tool_call>{\"arguments\": {}}</tool_call>",
        false,
    );
    assert_eq!(p.error, Some(ParseError::MissingName));
    assert_eq!(
        ParseError::MissingName.to_string(),
        "tool call has no \"name\" string"
    );
    let p = parse(
        Family::Mistral,
        "[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": \"src/lib.rs\"}]",
        false,
    );
    assert_eq!(p.error, Some(ParseError::ArgsNotObject));
    assert_eq!(
        ParseError::ArgsNotObject.to_string(),
        "tool call arguments must be a JSON object"
    );
}

#[test]
fn a_call_without_arguments_gets_an_empty_object() {
    let p = call_only(Family::Qwen3, "<tool_call>{\"name\": \"log\"}</tool_call>");
    assert_eq!(p.call.unwrap().args, json!({}));
}

#[test]
fn a_thinking_block_before_the_call_is_split_off() {
    let p = call_only(
        Family::Qwen3,
        "<think>\nThe user wants the file.\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}\n</tool_call>",
    );
    assert_eq!(p.thinking.as_deref(), Some("The user wants the file."));
    assert_eq!(p.call, Some(read_call()));
    assert_eq!(p.text_before, "");
}

#[test]
fn a_marker_inside_a_closed_think_block_is_not_a_call() {
    let cases = [
        (
            Family::Qwen3,
            "<think>Maybe <tool_call>{\"name\": \"write_file\", \"arguments\": {}}</tool_call>? No.</think>\nThe file is fine.",
        ),
        (
            Family::Llama3,
            "<think>Maybe <|python_tag|>{\"name\": \"write_file\", \"parameters\": {}}? No.</think>\nThe file is fine.",
        ),
        (
            Family::Mistral,
            "<think>Maybe [TOOL_CALLS][{\"name\": \"write_file\", \"arguments\": {}}]? No.</think>\nThe file is fine.",
        ),
    ];
    for (family, text) in cases {
        let p = call_only(family, text);
        assert_eq!(p.call, None, "{text}");
        assert_eq!(p.text_before, "The file is fine.");
        assert!(p.thinking.unwrap().contains("write_file"));
    }
}

#[test]
fn a_close_tag_alone_ends_thinking_the_template_opened() {
    // A template that opens <think> in the prompt leaves the reply only the close tag.
    let opened = |text: &str| {
        let p = parse(Family::Qwen3, text, true);
        assert_eq!(p.error, None, "{text}");
        p
    };
    let p = opened(
        "Maybe <tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"text\": \"x\"}}</tool_call>? No.</think>\nThe file is fine.",
    );
    assert_eq!(p.call, None);
    assert_eq!(p.text_before, "The file is fine.");
    assert!(p.thinking.unwrap().ends_with("? No."));
    let p = opened(
        "The user wants the file.\n</think>\n\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}</tool_call>",
    );
    assert_eq!(p.thinking.as_deref(), Some("The user wants the file."));
    assert_eq!(p.call, Some(read_call()));
    // Cut at max_tokens before the close tag: all of it is thinking.
    let p = parse(
        Family::Qwen3,
        "Let me weigh the options. The user asked about",
        true,
    );
    assert_eq!(
        p.thinking.as_deref(),
        Some("Let me weigh the options. The user asked about")
    );
    assert_eq!(p.text_before, "");
    assert_eq!(p.error, Some(ParseError::UnclosedThinking));
}

#[test]
fn a_close_tag_without_an_opened_block_is_text() {
    for family in [Family::Qwen3, Family::Llama3, Family::Mistral] {
        let text = "The parser splits at </think> and keeps the rest.";
        let p = call_only(family, text);
        assert_eq!(p.thinking, None);
        assert_eq!(p.text_before, text);
    }
    let p = call_only(
        Family::Qwen3,
        "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"text\": \"Qwen ends thinking with </think> on its own line.\"}}</tool_call>",
    );
    let text = p.call.unwrap().args["text"].clone();
    assert_eq!(text, "Qwen ends thinking with </think> on its own line.");
    assert_eq!(p.text_before, "");
}

#[test]
fn an_unclosed_think_block_tells_the_model_it_ran_out_of_space() {
    let p = parse(
        Family::Qwen3,
        "<think>\nLet me go through every file one by one",
        false,
    );
    assert_eq!(p.call, None);
    assert_eq!(
        p.thinking.as_deref(),
        Some("Let me go through every file one by one")
    );
    assert_eq!(p.error, Some(ParseError::UnclosedThinking));
    assert_eq!(
        ParseError::UnclosedThinking.to_string(),
        "the reply ran out of space inside <think>; answer or call a tool directly, without thinking first"
    );
}

#[test]
fn text_before_a_call_is_kept_for_the_ui() {
    let cases = [
        (
            Family::Qwen3,
            "I will read it.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}\n</tool_call>",
        ),
        (
            Family::Llama3,
            "I will read it.\n<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}",
        ),
        (
            Family::Mistral,
            "I will read it.\n[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}]",
        ),
    ];
    for (family, text) in cases {
        let p = call_only(family, text);
        assert_eq!(p.call, Some(read_call()), "{text}");
        assert_eq!(p.text_before, "I will read it.");
    }
}

#[test]
fn plain_text_is_no_call() {
    for family in [Family::Qwen3, Family::Llama3, Family::Mistral] {
        let p = call_only(family, "  The answer is 4.\n");
        assert_eq!(p.call, None);
        assert_eq!(p.text_before, "The answer is 4.");
    }
    // A JSON answer that is not shaped like a call stays an answer, also when it is not strict
    // JSON and names something.
    for text in [
        "{\"answer\": 4}",
        "{\n  // the app's manifest\n  \"name\": \"my-app\",\n  \"version\": \"1.0.0\"\n}\nThat is a package manifest with a comment.",
    ] {
        let p = call_only(Family::Llama3, text);
        assert_eq!(p.call, None);
        assert_eq!(p.text_before, text);
    }
}

#[test]
fn only_the_first_of_two_calls_counts() {
    // The Qwen format writes one block per call, so it keeps both.
    let qwen = "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}</tool_call>\n<tool_call>{\"name\": \"log\", \"arguments\": {}}</tool_call>";
    let p = parse(Family::Qwen3, qwen, false);
    assert_eq!((p.call, p.error), (Some(read_call()), None));
    assert_eq!(
        p.later.iter().map(|c| &c.name[..]).collect::<Vec<_>>(),
        ["log"]
    );
    let cases = [
        (
            Family::Llama3,
            "{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}; {\"name\": \"log\", \"parameters\": {}}",
        ),
        (
            Family::Llama3,
            "<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}<|python_tag|>{\"name\": \"log\", \"parameters\": {}}",
        ),
        (
            Family::Mistral,
            "[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}, {\"name\": \"log\", \"arguments\": {}}]",
        ),
    ];
    for (family, text) in cases {
        let p = parse(family, text, false);
        assert_eq!(p.call, Some(read_call()), "{text}");
        assert_eq!(p.error, Some(ParseError::MultipleCalls), "{text}");
    }
    assert_eq!(
        ParseError::MultipleCalls.to_string(),
        "only one tool call per reply: the first call is used and the others are ignored"
    );
}

#[test]
fn the_family_table() {
    let rows = [
        ("qwen2", None, Family::Qwen3),
        ("qwen3moe", None, Family::Qwen3),
        ("qwen3", None, Family::Qwen3),
        ("Qwen3", None, Family::Qwen3),
        ("llama", None, Family::Llama3),
        ("Llama", None, Family::Llama3),
        ("mistral", None, Family::Mistral),
        // Mistral GGUFs say `llama`; config.toml names the family.
        ("llama", Some("mistral"), Family::Mistral),
        ("qwen3", Some("llama3"), Family::Llama3),
        ("llama", Some("Qwen3"), Family::Qwen3),
    ];
    for (arch, over, family) in rows {
        assert_eq!(family_for(arch, over), Some(family), "{arch} {over:?}");
    }
    // No row and no override: no known tool-call format, so the chat runs without tools.
    assert_eq!(family_for("gemma3", None), None);
    // An unknown override is a config mistake: no tools, rather than a format the model never learned.
    assert_eq!(family_for("llama", Some("mistral-v3")), None);
}

fn calendar() -> ToolSpec {
    ToolSpec {
        name: "calendar_events".into(),
        description: "List events in a date range.".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "n": {"type": "integer", "minimum": 1}
            },
            "required": ["from", "to"],
            "additionalProperties": false
        }),
        permission: Permission::Read,
        untrusted_results: false,
        max_result_tokens: 100,
        max_args_tokens: 256,
    }
}

fn check(name: &str, args: serde_json::Value) -> Result<(), ValidateError> {
    validate(
        &ToolCall {
            name: name.into(),
            args,
        },
        &calendar(),
    )
}

#[test]
fn validation_messages_are_the_feedback_sentences() {
    assert_eq!(
        check("calendar_events", json!({"from": "a", "to": "b"})),
        Ok(())
    );
    let shown = |r: Result<(), ValidateError>| r.unwrap_err().to_string();
    assert_eq!(
        shown(check("calender_events", json!({}))),
        "unknown tool 'calender_events'; did you mean 'calendar_events'?"
    );
    assert_eq!(
        shown(check("weather", json!({}))),
        "unknown tool 'weather'; available: calendar_events"
    );
    assert_eq!(
        shown(check("calendar_events", json!({"from": "2026-10-13"}))),
        "missing required argument 'to'"
    );
    assert_eq!(
        shown(check(
            "calendar_events",
            json!({"from": "a", "to": "b", "n": "three"})
        )),
        "argument 'n' must be of type integer"
    );
    assert_eq!(
        shown(check(
            "calendar_events",
            json!({"from": "a", "to": "b", "n": 0})
        )),
        "argument 'n' is invalid: 0 is less than the minimum of 1"
    );
    assert_eq!(
        shown(check(
            "calendar_events",
            json!({"from": "a", "to": "b", "form": "c"})
        )),
        "unknown argument 'form'; arguments: from, n, to"
    );
}

#[test]
fn unknown_tool_lists_every_name_when_none_is_close() {
    let e = ValidateError::UnknownTool {
        name: "mail_send".into(),
        available: vec!["list_dir".into(), "read_file".into()],
    };
    assert_eq!(
        e.to_string(),
        "unknown tool 'mail_send'; available: list_dir, read_file"
    );
}

#[test]
fn a_short_unknown_name_gets_the_list_not_a_far_guess() {
    let available = ["list_dir", "read_file", "log", "recall"]
        .map(String::from)
        .to_vec();
    let shown = |name: &str| {
        ValidateError::UnknownTool {
            name: name.into(),
            available: available.clone(),
        }
        .to_string()
    };
    let list = "available: list_dir, read_file, log, recall";
    assert_eq!(shown("ls"), format!("unknown tool 'ls'; {list}"));
    assert_eq!(shown("read"), format!("unknown tool 'read'; {list}"));
    assert_eq!(shown("logs"), "unknown tool 'logs'; did you mean 'log'?");
    assert_eq!(
        shown("read_fiel"),
        "unknown tool 'read_fiel'; did you mean 'read_file'?"
    );
}

#[test]
fn a_mistral_call_counts_twice_only_for_a_second_call() {
    let call = "[TOOL_CALLS][{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}]";
    let p = call_only(
        Family::Mistral,
        &format!("{call}\n[docs](https://example.invalid)"),
    );
    assert_eq!(p.call, Some(read_call()));
    let p = parse(
        Family::Mistral,
        &format!("{call}\n[{{\"name\": \"log\", \"arguments\": {{}}}}]"),
        false,
    );
    assert_eq!(
        (p.call, p.error),
        (Some(read_call()), Some(ParseError::MultipleCalls))
    );
}

#[test]
fn a_llama_call_counts_twice_only_for_a_second_call() {
    let call = "{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/lib.rs\"}}";
    // JSON after the call that is not a call is not a second one.
    let p = call_only(
        Family::Llama3,
        &format!("<|python_tag|>{call}\n{{\"result\": \"ok\"}}"),
    );
    assert_eq!(p.call, Some(read_call()));
    // A second call cut off at max_tokens is still a second call.
    for text in [
        format!("{call}\n{{\"name\": \"log\", \"parameters\": {{\"branch\""),
        format!("<|python_tag|>{call}; {{\"name\": \"log\", \"parameters\": {{"),
    ] {
        let p = parse(Family::Llama3, &text, false);
        assert_eq!(
            (p.call, p.error),
            (Some(read_call()), Some(ParseError::MultipleCalls)),
            "{text}"
        );
    }
}

#[test]
fn a_tag_named_in_prose_before_the_call_is_text() {
    let p = call_only(
        Family::Qwen3,
        "I will use a <tool_call> block.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}</tool_call>",
    );
    assert_eq!(p.call, Some(read_call()));
    assert_eq!(p.text_before, "I will use a <tool_call> block.");
    // A broken call after it is reported where its JSON breaks, not at the tag in prose.
    let p = parse(
        Family::Qwen3,
        "I will use a <tool_call> block.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"</tool_call>",
        false,
    );
    assert_eq!(p.call, None);
    assert_eq!(
        p.error.map(|e| e.to_string()).as_deref(),
        Some("tool call is not valid JSON: EOF while parsing an object at byte 47")
    );
    assert_eq!(p.text_before, "I will use a <tool_call> block.");
}

#[test]
fn a_closing_tag_inside_a_string_argument_is_text() {
    let p = parse(
        Family::Qwen3,
        "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"t.rs\", \"text\": \"a </tool_call> b\"}}</tool_call>",
        false,
    );
    assert_eq!(p.error, None);
    assert_eq!(
        p.call,
        Some(ToolCall {
            name: "write_file".into(),
            args: json!({"path": "t.rs", "text": "a </tool_call> b"}),
        })
    );
}

#[test]
fn xml_body_values_arrive_typed_and_python_literals_read_where_the_schema_admits_them() {
    let tools = [
        json!({"type": "function", "function": {"name": "t", "parameters": {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"},
                "off": {"type": "boolean"},
                "gone": {"type": ["integer", "null"]},
                "n": {"type": "integer"},
                "word": {"type": "string"},
            },
        }}}),
    ];
    let mut call = ToolCall {
        name: "t".into(),
        args: json!({"flag": "True", "off": "false", "gone": "None", "n": "3", "word": "False"}),
    };
    larch_agent::parse::coerce(&mut call, &tools);
    let want = json!({"flag": true, "off": false, "gone": null, "n": 3, "word": "False"});
    assert_eq!(call.args, want);
}
