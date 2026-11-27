//! Tokenizer, chat templates and tokcheck against llama.cpp (architecture 15). Every expected token
//! list and render in `tests/data` comes from `tools/llama-logits/tokdump` (llama.cpp b11458, CPU):
//!   tokdump tokens <gguf> tests/data/prompts.jsonl > tests/data/<name>.tokens.jsonl
//!   tokdump chat <gguf> <template | -> tests/data/chat.json "26 Jul 2024" > tests/data/<name>.chat.jsonl
//!   tokdump chat <gguf> <template | -> tests/data/chat-tools.json "26 Jul 2024" > tests/data/<name>.tools.jsonl
//! For `synth-qwen2.qwen3` only, the last two once with `true` after the date and once with
//! `false`, the second appended, write `synth-qwen2.qwen3.thinking.chat.jsonl` and
//! `synth-qwen2.qwen3.thinking.tools.jsonl`.
//! `no-roundtrip/mistral.jinja` is llama.cpp's tool-calling Mistral template (`models/templates` at
//! b11458, sha256 e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272). It puts the
//! system message in the last user turn, so with a system message no turn's prompt is a prefix of
//! the next. It stays out of the top level, whose `*.jinja` the facade's tests take as templates
//! that keep the KV across turns (D14). The synthetic vocabularies are generated here;
//! `LARCH_VOCAB_OUT=<dir> cargo test -p larch-model --test tokenizer -- --ignored write_vocabs`
//! writes them for tokdump. Real vocabularies, outside git:
//! `LARCH_LLAMA_CPP=<llama.cpp checkout> cargo test -p larch-model --test tokenizer -- --ignored
//! llama_cpp_vocabs`, and the merge gate `tools/llama-logits/model-dumps.sh <dir of the GGUFs>`.
//! A `.tokens.jsonl` dump holds each prompt with its tokens: the `--ref` of `larch tokcheck`.

use larch_model::gguf::{Gguf, Value, write};
use larch_model::{
    ChatTemplate, Detokenizer, Message, TokcheckError, Tokenizer, ToolCall, ToolMessage, tiny,
    tokcheck,
};
use larch_records::LoadError;
use serde_json::Value as Json;

const DATE: &str = "26 Jul 2024";

fn data(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("tests/data/{name}: {e}"))
}

/// GPT-2's byte-level alphabet, which byte-level BPE vocabularies are written in.
fn byte_chars() -> [char; 256] {
    let mut next = 0x100;
    let mut map = ['\0'; 256];
    for (b, c) in map.iter_mut().enumerate() {
        *c = if matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
            b as u8 as char
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    map
}

fn byte_level(s: &str) -> String {
    let map = byte_chars();
    s.bytes().map(|b| map[b as usize]).collect()
}

/// Words whose left-to-right merge chains make the synthetic vocabularies' merges, in rank order.
const WORDS: [&str; 30] = [
    "he", "in", " the", "the", "ing", " hello", " world", "er", " there", "re", "Hello", "  ",
    "    ", "\n", "\n\n", " \n", "12", "123", " 1", "'s", "'re", "日本", "é", " {\n", "()",
    " print", "ou", " you", " 世界", "!\n",
];

/// `qwen35`'s extra words: merges across a letter and its combining mark, and across punctuation
/// and a mark, which split apart under `qwen2`.
const MARK_WORDS: [&str; 4] = ["e\u{301}", "u\u{308}", "हि", ">\u{301}"];

/// (text, GGUF token type): control 3, user-defined 4, unknown 2, unused 5; `<|endoftext|>` is
/// normal in the file and made control by its text.
const SPECIAL: [(&str, i32); 13] = [
    ("<|begin_of_text|>", 3),
    ("<|end_of_text|>", 3),
    ("<|eot_id|>", 3),
    ("<|start_header_id|>", 3),
    ("<|end_header_id|>", 3),
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<|endoftext|>", 1),
    ("<tool_call>", 4),
    ("<|im", 4),
    ("<|unknown_tok|>", 2),
    ("<|fim_pad|>", 3),
    ("<|unused|>", 5),
];

/// The hyperparameters llama.cpp wants even when it loads only the vocabulary.
fn hparams() -> Vec<(String, Value)> {
    let u = |k: &str, v| (format!("llama.{k}"), Value::U32(v));
    vec![
        ("general.architecture".into(), Value::Str("llama".into())),
        u("block_count", 1),
        u("context_length", 512),
        u("embedding_length", 64),
        u("feed_forward_length", 128),
        u("attention.head_count", 4),
        u("attention.head_count_kv", 4),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
    ]
}

fn strs(v: impl IntoIterator<Item = String>) -> Value {
    Value::Arr(v.into_iter().map(Value::Str).collect())
}

/// A byte-level BPE vocabulary: the 256 byte tokens, the merge chains of [`WORDS`] (and
/// [`MARK_WORDS`] for `qwen35`), [`SPECIAL`].
fn bpe_vocab(pre: &str) -> Vec<u8> {
    write(&bpe_kv(pre), NO_TENSORS)
}

const NO_TENSORS: &[(&str, Vec<u64>, u32, Vec<u8>)] = &[];

/// [`bpe_vocab`]'s metadata.
fn bpe_kv(pre: &str) -> Vec<(String, Value)> {
    let mut tokens: Vec<String> = byte_chars().map(String::from).to_vec();
    let mut merges = Vec::new();
    let marks: &[&str] = if pre == "qwen35" { &MARK_WORDS } else { &[] };
    for w in WORDS.iter().chain(marks) {
        let cs: Vec<char> = byte_level(w).chars().collect();
        for j in 1..cs.len() {
            let (l, r): (String, String) = (cs[..j].iter().collect(), cs[j].to_string());
            if !merges.contains(&format!("{l} {r}")) {
                merges.push(format!("{l} {r}"));
                tokens.push(format!("{l}{r}"));
            }
        }
    }
    let first_special = tokens.len() as u32;
    tokens.extend(SPECIAL.iter().map(|(t, _)| t.to_string()));
    let types = (0..tokens.len())
        .map(|i| match i.checked_sub(first_special as usize) {
            Some(s) => Value::I32(SPECIAL[s].1),
            None => Value::I32(1),
        })
        .collect();
    let mut kv = hparams();
    kv.extend([
        ("tokenizer.ggml.model".into(), Value::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::Str(pre.into())),
        ("tokenizer.ggml.tokens".into(), strs(tokens)),
        ("tokenizer.ggml.token_type".into(), Value::Arr(types)),
        ("tokenizer.ggml.merges".into(), strs(merges)),
        (
            "tokenizer.ggml.bos_token_id".into(),
            Value::U32(first_special),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            Value::U32(first_special + 2),
        ),
    ]);
    if pre == "qwen2" {
        // Qwen defaults to no BOS; this vocabulary asks for it.
        kv.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
    }
    kv
}

/// An SPM vocabulary in Mistral's shape: `<unk> <s> </s>`, 256 byte tokens, scored pieces; it also
/// asks for EOS.
fn spm_vocab() -> Vec<u8> {
    write(&spm_kv(), NO_TENSORS)
}

/// [`spm_vocab`]'s metadata.
fn spm_kv() -> Vec<(String, Value)> {
    let mut tokens: Vec<(String, f32, i32)> = vec![
        ("<unk>".into(), 0.0, 2),
        ("<s>".into(), 0.0, 3),
        ("</s>".into(), 0.0, 3),
    ];
    tokens.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), 0.0, 6)));
    tokens.extend([("[INST]".into(), 0.0, 3), ("[/INST]".into(), 0.0, 3)]);
    // Higher score merges first; single characters never merge on their own.
    let pieces = "▁t he ▁the in ing ▁▁ ▁▁▁▁ ll llo ▁he ▁hello ▁w or ld ▁world ▁wor 12 123 日本 ▁日本 er re ▁you ou ▁y";
    tokens.extend(
        pieces
            .split(' ')
            .enumerate()
            .map(|(i, p)| (p.to_string(), -(i as f32), 1)),
    );
    let singles = ('!'..='~')
        .map(String::from)
        .chain(["▁", "日", "本", "é"].map(String::from));
    tokens.extend(singles.map(|c| (c, -1000.0, 1)));
    let mut kv = hparams();
    kv.extend([
        ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
        (
            "tokenizer.ggml.tokens".into(),
            strs(tokens.iter().map(|t| t.0.clone())),
        ),
        (
            "tokenizer.ggml.scores".into(),
            Value::Arr(tokens.iter().map(|t| Value::F32(t.1)).collect()),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Arr(tokens.iter().map(|t| Value::I32(t.2)).collect()),
        ),
        ("tokenizer.ggml.add_eos_token".into(), Value::Bool(true)),
    ]);
    kv
}

fn vocabs() -> [(&'static str, Vec<u8>); 4] {
    [
        ("synth-llama3", bpe_vocab("llama-bpe")),
        ("synth-qwen2", bpe_vocab("qwen2")),
        ("synth-qwen35", bpe_vocab("qwen35")),
        ("synth-spm", spm_vocab()),
    ]
}

fn tokenizer(gguf: &[u8]) -> (Gguf, Tokenizer) {
    let g = Gguf::parse(gguf).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    (g, tok)
}

/// The header of the GGUF at `path` and its tokenizer; `Gguf::read` skips the tensor data.
fn tokenizer_file(path: &str) -> (Gguf, Tokenizer) {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let len = f.metadata().unwrap().len();
    let g = Gguf::read(std::io::BufReader::new(f), len).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    (g, tok)
}

#[test]
#[ignore = "writes the synthetic vocabularies to LARCH_VOCAB_OUT for tokdump"]
fn write_vocabs() {
    let dir = std::env::var("LARCH_VOCAB_OUT").expect("LARCH_VOCAB_OUT=<dir>");
    for (name, gguf) in vocabs() {
        std::fs::write(format!("{dir}/{name}.gguf"), gguf).unwrap();
    }
}

/// `tokcheck` against `<name>.tokens.jsonl`, which must be a dump of exactly `prompts.jsonl`, so an
/// edited prompt cannot keep a stale dump passing.
fn check_tokens(tok: &Tokenizer, name: &str) {
    let (prompts, dump) = (data("prompts.jsonl"), data(&format!("{name}.tokens.jsonl")));
    let prompt = |l: &str| {
        serde_json::from_str::<Json>(l)
            .unwrap()
            .get("prompt")
            .cloned()
    };
    assert!(
        prompts.lines().map(prompt).eq(dump.lines().map(prompt)),
        "{name}: not a dump of prompts.jsonl"
    );
    assert_eq!(tokcheck(tok, &dump), Ok(prompts.lines().count()), "{name}");
}

#[test]
fn synthetic_vocabs_tokenise_like_llama() {
    for (name, gguf) in vocabs() {
        check_tokens(&tokenizer(&gguf).1, name);
    }
}

/// SPM special-token texts in the vocabularies here; a prompt holding one splits into fragments that
/// each get SPM's leading space, which [`check_lossless`] does not model.
const SPM_SPECIAL: [&str; 5] = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]"];

/// Encoding loses nothing: the tokens' pieces spell each prompt back. SPM adds its leading space and
/// reads `▁` as a space, as llama.cpp does.
fn check_lossless(tok: &Tokenizer, spm: bool, prompts: &str) {
    for line in prompts.lines() {
        let v: Json = serde_json::from_str(line).unwrap();
        let text = v["prompt"].as_str().unwrap();
        let want = match spm {
            false => text.to_owned(),
            true if SPM_SPECIAL.iter().any(|s| text.contains(s)) => continue,
            true if text.is_empty() => String::new(),
            true => format!(" {}", text.replace('▁', " ")),
        };
        let spelt: Vec<u8> = tok
            .encode(text, false)
            .iter()
            .flat_map(|&id| tok.decode_piece(id).to_vec())
            .collect();
        assert_eq!(String::from_utf8(spelt).unwrap(), want, "{text:?}");
    }
}

#[test]
fn synthetic_vocabs_lose_nothing() {
    let prompts = data("prompts.jsonl");
    for (name, gguf) in vocabs() {
        check_lossless(&tokenizer(&gguf).1, name == "synth-spm", &prompts);
    }
}

/// A vocabulary that cannot spell every byte would drop text (llama.cpp drops the byte for BPE and
/// aborts for SPM), so loading refuses it.
#[test]
fn vocabulary_missing_a_byte_is_refused() {
    // Byte 0x00 is `Ā` in the byte-level alphabet.
    for (gguf, byte) in [(bpe_vocab("llama-bpe"), "Ā"), (spm_vocab(), "<0x80>")] {
        let mut g = Gguf::parse(&gguf).unwrap();
        let Some(Value::Arr(tokens)) = g.kv.get_mut("tokenizer.ggml.tokens") else {
            unreachable!("the vocabularies here have tokens")
        };
        let i = tokens
            .iter()
            .position(|t| t.as_str() == Some(byte))
            .unwrap();
        tokens[i] = Value::Str("[gone]".into());
        assert_eq!(
            Tokenizer::from_gguf(&g).err(),
            Some(LoadError::BadGguf(larch_model::gguf::code::KEY, 0)),
            "{byte}"
        );
    }
}

#[test]
fn tokcheck_reports_the_first_difference() {
    let (_, tok) = tokenizer(&bpe_vocab("llama-bpe"));
    let line =
        |p: &str, t: &[u32]| serde_json::json!({ "prompt": p, "tokens": t }).to_string() + "\n";
    let first = line("Hello world", &tok.encode("Hello world", true));
    let b = tok.encode(" the", true);
    assert_eq!(tokcheck(&tok, &(first.clone() + &line(" the", &b))), Ok(2));

    let mut wrong = b.clone();
    wrong[1] += 1;
    assert_eq!(
        tokcheck(&tok, &(first.clone() + &line(" the", &wrong))),
        Err(TokcheckError::Mismatch {
            prompt: 1,
            position: 1,
            ours: Some(b[1]),
            llama: Some(wrong[1]),
        })
    );
    assert_eq!(
        tokcheck(&tok, &(first.clone() + &line(" the", &b[..1]))),
        Err(TokcheckError::Mismatch {
            prompt: 1,
            position: 1,
            ours: Some(b[1]),
            llama: None,
        })
    );
    // An id that is not a u32, and a line without its prompt.
    for bad in [
        "{\"prompt\": \" the\", \"tokens\": [-1]}",
        "{\"tokens\": []}",
    ] {
        assert_eq!(
            tokcheck(&tok, &format!("{first}{bad}\n")),
            Err(TokcheckError::BadLine(2))
        );
    }
    assert_eq!(tokcheck(&tok, ""), Err(TokcheckError::Empty));
}

/// `Session` shares both across threads (architecture 10.1.1).
#[test]
fn shared_across_threads() {
    fn check<T: Send + Sync>() {}
    check::<Tokenizer>();
    check::<ChatTemplate>();
}

#[test]
fn eog_and_special_ids() {
    let (_, tok) = tokenizer(&bpe_vocab("llama-bpe"));
    let id = |t: &str| match tok.encode(t, false)[..] {
        [id] => id,
        ref ids => panic!("{t} is {ids:?}"),
    };
    assert_eq!(tok.bos(), Some(id("<|begin_of_text|>")));
    assert_eq!(tok.eos(), Some(id("<|eot_id|>")));
    let mut eog = vec![
        id("<|eot_id|>"),
        id("<|end_of_text|>"),
        id("<|im_end|>"),
        id("<|endoftext|>"),
        id("<|fim_pad|>"),
    ];
    eog.sort_unstable();
    assert_eq!(tok.eog(), eog);
    assert_eq!(tok.decode_piece(id("<|eot_id|>")), b"<|eot_id|>");
    assert_eq!(tok.decode_piece(u32::MAX), b"");
}

/// Feeds `ids` one at a time; returns the pieces of text and whether a stop string ended them.
fn stream(tok: &Tokenizer, stops: &[&str], ids: &[u32]) -> (Vec<String>, bool) {
    let mut d = Detokenizer::new(stops.iter().map(|s| s.to_string()).collect());
    let mut out = Vec::new();
    for &id in ids {
        let (text, stop) = d.push(tok, id);
        out.push(text);
        if stop {
            return (out, true);
        }
    }
    out.push(d.finish());
    (out, false)
}

#[test]
fn detokenizer_waits_for_whole_characters() {
    let (_, tok) = tokenizer(&spm_vocab());
    let byte = |b: u8| 3 + u32::from(b);
    // 世 is E4 B8 96 and has no token: three byte tokens.
    let ids = [byte(0xe4), byte(0xb8), byte(0x96), byte(b'!')];
    assert_eq!(stream(&tok, &[], &ids).0, ["", "", "世", "!", ""]);
    // A lone continuation byte is not UTF-8; a sequence cut off at the end neither.
    assert_eq!(
        stream(&tok, &[], &[byte(0x96), byte(0xe4)]).0,
        ["\u{fffd}", "", "\u{fffd}"]
    );
    // Control tokens show nothing.
    assert_eq!(stream(&tok, &[], &[2, byte(b'a')]).0, ["", "a", ""]);
}

#[test]
fn detokenizer_holds_back_and_cuts_stop_strings() {
    let (_, tok) = tokenizer(&spm_vocab());
    let ids = |s: &str| tok.encode(s, false);
    // "▁hello" "▁world": the stop string "lo wo" spans the two tokens.
    let hw: Vec<u32> = ids("hello world");
    assert_eq!(hw.len(), 2);
    assert_eq!(
        stream(&tok, &["lo wo"], &hw),
        (vec![" hel".to_string(), String::new()], true)
    );
    // A tail that only might start a stop string is released when it does not.
    assert_eq!(stream(&tok, &["lo!"], &hw).0, [" hel", "lo world", ""]);
    // Held text comes out at the end; an empty stop string is ignored.
    assert_eq!(
        stream(&tok, &["world!", ""], &hw).0,
        [" hello", " ", "world"]
    );
}

/// The fixed six-turn conversation behind the template dumps.
fn conversation() -> Vec<Message> {
    let v: Json = serde_json::from_str(&data("chat.json")).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| Message {
            role: m["role"].as_str().unwrap().into(),
            content: m["content"].as_str().unwrap().into(),
        })
        .collect()
}

/// [`conversation`] as tool messages without tool fields.
fn conversation_as_tool_messages() -> Vec<ToolMessage> {
    conversation()
        .into_iter()
        .map(|m| ToolMessage {
            role: m.role,
            content: m.content,
            ..Default::default()
        })
        .collect()
}

/// Every render tokdump made, as llama.cpp rendered it, against ours.
fn check_renders(t: &ChatTemplate, dump: &str) {
    let conv = conversation();
    let lines: Vec<Json> = dump
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(lines.len() >= 4, "user turns plus the whole conversation");
    for l in lines {
        let k = l["messages"].as_u64().unwrap() as usize;
        let add_gen = l["add_generation_prompt"].as_bool().unwrap();
        assert_eq!(
            t.render(&conv[..k], add_gen).unwrap(),
            l["prompt"].as_str().unwrap(),
            "{k} messages"
        );
    }
}

/// What `qwen3.jinja` adds after the generation prompt when `enable_thinking` is false.
const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

/// tokdump's renders of `conv` with `enable_thinking` true and false over the same prompts, against
/// `render_tools_thinking`. False gives the generation prompt the empty think block and true does
/// not, so a dump or a render that ignores the flag fails.
fn check_thinking_renders(t: &ChatTemplate, conv: &[ToolMessage], tools: &[Json], dump: &str) {
    let mut halves: [Vec<(usize, bool)>; 2] = Default::default();
    for l in dump.lines() {
        let l: Json = serde_json::from_str(l).unwrap();
        let k = l["messages"].as_u64().unwrap() as usize;
        let add_gen = l["add_generation_prompt"].as_bool().unwrap();
        let thinking = l["enable_thinking"].as_bool().unwrap();
        let prompt = l["prompt"].as_str().unwrap();
        halves[usize::from(thinking)].push((k, add_gen));
        let at = format!("{k} messages, thinking {thinking}");
        assert_eq!(
            t.render_tools_thinking(&conv[..k], tools, add_gen, Some(thinking))
                .unwrap(),
            prompt,
            "{at}"
        );
        if add_gen {
            assert_eq!(prompt.ends_with(EMPTY_THINK), !thinking, "{at}");
        }
    }
    assert!(
        !halves[0].is_empty() && halves[0] == halves[1],
        "both values over the same prompts: {halves:?}"
    );
}

/// tokdump's renders of `chat.json` (`chat`) and `chat-tools.json` (`tools`), each with
/// `enable_thinking` true and false, against ours. `chat.json` goes in as tool messages without
/// tool fields, which render as [`ChatTemplate::render`] renders the messages
/// (`empty_tool_fields_render_as_none`).
fn check_thinking(t: &ChatTemplate, chat: &str, tools: &str) {
    check_thinking_renders(t, &conversation_as_tool_messages(), &[], chat);
    let (conv, specs) = tool_conversation();
    check_thinking_renders(t, &conv, &specs, tools);
}

/// D14: every turn's prompt tokenises to a prefix of the next turn's, so the engine reuses the whole
/// previous prompt from the parked KV.
fn check_roundtrip(t: &ChatTemplate, tok: &Tokenizer) {
    let conv = conversation();
    let mut turns: Vec<Vec<u32>> = (1..conv.len())
        .filter(|&k| conv[k - 1].role == "user")
        .map(|k| tok.encode(&t.render(&conv[..k], true).unwrap(), true))
        .collect();
    turns.push(tok.encode(&t.render(&conv, false).unwrap(), true));
    for w in turns.windows(2) {
        assert!(
            w[1].starts_with(&w[0]),
            "{:?} is not a prefix of {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn templates_render_like_llama() {
    // The templates the target GGUFs carry. Qwen3 drops reasoning from earlier turns, so only Llama 3.1
    // and Qwen2.5 promise the roundtrip.
    for (vocab, tmpl, name, roundtrip) in [
        (
            bpe_vocab("llama-bpe"),
            "llama31.jinja",
            "synth-llama3.llama31",
            true,
        ),
        (
            bpe_vocab("qwen2"),
            "qwen25.jinja",
            "synth-qwen2.qwen25",
            true,
        ),
        (
            bpe_vocab("qwen2"),
            "qwen3.jinja",
            "synth-qwen2.qwen3",
            false,
        ),
    ] {
        let (_, tok) = tokenizer(&vocab);
        let mut t = ChatTemplate::new(&data(tmpl), &tok).unwrap();
        t.date = DATE.into();
        check_renders(&t, &data(&format!("{name}.chat.jsonl")));
        if roundtrip {
            check_roundtrip(&t, &tok);
        }
    }
}

/// `thinking` is the template's `enable_thinking`, which `qwen3.jinja` reads, with tools and
/// without.
#[test]
fn templates_read_the_thinking_flag() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let mut t = ChatTemplate::new(&data("qwen3.jinja"), &tok).unwrap();
    t.date = DATE.into();
    check_thinking(
        &t,
        &data("synth-qwen2.qwen3.thinking.chat.jsonl"),
        &data("synth-qwen2.qwen3.thinking.tools.jsonl"),
    );
    // The flag printed, which `qwen3.jinja` cannot show for None: tokdump (b11458) renders this
    // template for one user message "a" on synth-qwen2.gguf as `True`, with `true` as `True` and
    // with `false` as `False`.
    let flag = ChatTemplate::new("{{ enable_thinking }}", &tok).unwrap();
    let a = [Message {
        role: "user".into(),
        content: "a".into(),
    }];
    assert_eq!(flag.render(&a, true).unwrap(), "True");
    let a = [ToolMessage {
        role: "user".into(),
        content: "a".into(),
        ..Default::default()
    }];
    for (thinking, printed) in [(true, "True"), (false, "False")] {
        let ours = flag.render_tools_thinking(&a, &[], true, Some(thinking));
        assert_eq!(ours.unwrap(), printed, "thinking {thinking}");
    }
}

/// `tiny::TOKENIZER_GGUF`: the synth-llama3 vocabulary with the Llama 3.1 template in the file, so
/// llama.cpp's dumps of that vocabulary and template are its reference (tokdump renders the file's own
/// template, `-`, exactly as `synth-llama3.llama31.chat.jsonl`).
fn tiny_tokenizer() -> Vec<u8> {
    let mut kv = bpe_kv("llama-bpe");
    kv.push((
        "tokenizer.chat_template".into(),
        Value::Str(data("llama31.jinja")),
    ));
    write(&kv, NO_TENSORS)
}

#[test]
#[ignore = "rewrites tests/data/tiny-tokenizer.gguf"]
fn write_tiny_tokenizer() {
    std::fs::write(tiny::TOKENIZER_GGUF, tiny_tokenizer()).unwrap();
}

#[test]
fn tiny_tokenizer_fixture() {
    let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
    assert!(
        bytes == tiny_tokenizer(),
        "stale fixture: cargo test -p larch-model --test tokenizer -- --ignored write_tiny_tokenizer"
    );
    assert!(bytes.len() < 16 << 10, "{} bytes", bytes.len());
    let (g, tok) = tokenizer(&bytes);
    check_tokens(&tok, "synth-llama3");
    let mut t = ChatTemplate::from_gguf(&g, &tok).unwrap();
    assert!(!t.fallback);
    t.date = DATE.into();
    check_renders(&t, &data("synth-llama3.llama31.chat.jsonl"));
    check_roundtrip(&t, &tok);
}

#[test]
fn template_without_source_is_chatml() {
    let (g, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::from_gguf(&g, &tok).unwrap();
    assert!(t.fallback);
    let m = |role: &str, content: &str| Message {
        role: role.into(),
        content: content.into(),
    };
    assert_eq!(
        t.render(&[m("user", "hi")], true).unwrap(),
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(ChatTemplate::new("{% if %}", &tok).is_err());
    let endless = "{% for i in range(99999) %}{% for j in range(99999) %}{% endfor %}{% endfor %}";
    assert!(
        ChatTemplate::new(endless, &tok)
            .unwrap()
            .render(&[], true)
            .is_err()
    );
}

/// `chat-tools.json`: system, user, an assistant message with one tool call and its result, with
/// two tools in OpenAI's function shape.
fn tool_conversation() -> (Vec<ToolMessage>, Vec<Json>) {
    let v: Json = serde_json::from_str(&data("chat-tools.json")).unwrap();
    let messages = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| ToolMessage {
            role: m["role"].as_str().unwrap().into(),
            content: m["content"].as_str().unwrap().into(),
            tool_calls: m
                .get("tool_calls")
                .map_or_else(Vec::new, |c| serde_json::from_value(c.clone()).unwrap()),
            tool_call_id: m["tool_call_id"].as_str().unwrap_or_default().into(),
        })
        .collect();
    (messages, v["tools"].as_array().unwrap().clone())
}

/// Both renders tokdump made of `chat-tools.json` (the user turn and the tool result, each with the
/// generation prompt), as llama.cpp rendered them, against ours: with the call's arguments as an
/// object, and as OpenAI sends them, a JSON string, which llama.cpp parses and renders the same.
fn check_tool_renders(t: &ChatTemplate, dump: &str) {
    let (conv, tools) = tool_conversation();
    let mut wire = conv.clone();
    for c in wire.iter_mut().flat_map(|m| &mut m.tool_calls) {
        c.arguments = Json::String(c.arguments.to_string());
    }
    let lines: Vec<Json> = dump
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "the user turn and the tool result");
    for l in lines {
        let k = l["messages"].as_u64().unwrap() as usize;
        let add_gen = l["add_generation_prompt"].as_bool().unwrap();
        for (conv, form) in [(&conv, "object"), (&wire, "string")] {
            assert_eq!(
                t.render_tools(&conv[..k], &tools, add_gen).unwrap(),
                l["prompt"].as_str().unwrap(),
                "{k} messages, {form} arguments"
            );
        }
    }
}

/// String arguments that are not JSON fail the render, as in llama.cpp.
#[test]
fn string_arguments_must_be_json() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::new(&data("qwen3.jinja"), &tok).unwrap();
    let (mut conv, tools) = tool_conversation();
    conv[2].tool_calls[0].arguments = Json::String("{\"city\": \"Paris\"".into());
    assert!(t.render_tools(&conv[..3], &tools, false).is_err());
}

/// A template with no tool support, in the form of m7's: llama.cpp's error on m7 quotes this text,
/// with `raise_exception(` at line 1, column 125. User and assistant must alternate, so a system or
/// tool message raises.
const NO_TOOLS: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

/// The whole of `chat-tools.json` fails, as llama.cpp's render of it does: the template's
/// `raise_exception` is the render's error.
fn check_tools_refused(t: &ChatTemplate) {
    let (conv, tools) = tool_conversation();
    let e = t.render_tools(&conv, &tools, true).unwrap_err();
    assert!(
        e.to_string()
            .contains("Conversation roles must alternate user/assistant"),
        "{e}"
    );
}

#[test]
fn template_without_tools_refuses_them() {
    let (_, tok) = tokenizer(&spm_vocab());
    check_tools_refused(&ChatTemplate::new(NO_TOOLS, &tok).unwrap());
}

#[test]
fn templates_render_tools_like_llama() {
    // The Qwen and Llama templates as their GGUFs carry them; Mistral is llama.cpp's tool-calling
    // Mistral template from models/templates at b11458, which needs 9-character call IDs.
    for (vocab, tmpl, name) in [
        (bpe_vocab("qwen2"), "qwen3.jinja", "synth-qwen2.qwen3"),
        (bpe_vocab("qwen2"), "qwen25.jinja", "synth-qwen2.qwen25"),
        (
            bpe_vocab("llama-bpe"),
            "llama31.jinja",
            "synth-llama3.llama31",
        ),
        (
            spm_vocab(),
            "no-roundtrip/mistral.jinja",
            "synth-spm.mistral",
        ),
    ] {
        let (_, tok) = tokenizer(&vocab);
        let mut t = ChatTemplate::new(&data(tmpl), &tok).unwrap();
        t.date = DATE.into();
        check_tool_renders(&t, &data(&format!("{name}.tools.jsonl")));
    }
}

/// ChatML has no tool syntax: llama.cpp drops the tools and the call, and keeps the tool message.
#[test]
fn fallback_renders_tools_like_llama() {
    let (g, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::from_gguf(&g, &tok).unwrap();
    assert!(t.fallback);
    check_tool_renders(&t, &data("synth-qwen2.chatml.tools.jsonl"));
}

#[test]
fn tiny_tokenizer_renders_tools() {
    let (g, tok) = tokenizer(&std::fs::read(tiny::TOKENIZER_GGUF).unwrap());
    let mut t = ChatTemplate::from_gguf(&g, &tok).unwrap();
    t.date = DATE.into();
    check_tool_renders(&t, &data("synth-llama3.llama31.tools.jsonl"));
}

/// An empty tool list, no calls and an empty call ID render as no tool fields at all: llama.cpp
/// leaves all three undefined, and Llama 3.1 tests `'tool_calls' in message`.
#[test]
fn empty_tool_fields_render_as_none() {
    for (vocab, tmpl, name) in [
        (
            bpe_vocab("llama-bpe"),
            "llama31.jinja",
            "synth-llama3.llama31",
        ),
        (bpe_vocab("qwen2"), "qwen3.jinja", "synth-qwen2.qwen3"),
    ] {
        let (_, tok) = tokenizer(&vocab);
        let mut t = ChatTemplate::new(&data(tmpl), &tok).unwrap();
        t.date = DATE.into();
        let conv = conversation_as_tool_messages();
        for l in data(&format!("{name}.chat.jsonl")).lines() {
            let l: Json = serde_json::from_str(l).unwrap();
            let k = l["messages"].as_u64().unwrap() as usize;
            let add_gen = l["add_generation_prompt"].as_bool().unwrap();
            assert_eq!(
                t.render_tools(&conv[..k], &[], add_gen).unwrap(),
                l["prompt"].as_str().unwrap(),
                "{name}, {k} messages"
            );
        }
    }
}

/// A message reaches the template as llama.cpp's `to_json_oaicompat` builds it: OpenAI's key order,
/// and no `tool_call_id` or call `id` key when they are empty.
#[test]
fn messages_reach_the_template_like_llama() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::new("{{ messages | tojson }}", &tok).unwrap();
    let call = |id: &str| ToolCall {
        name: "now".into(),
        arguments: serde_json::json!({"tz": "UTC"}),
        id: id.into(),
    };
    let m = |calls, call_id: &str| ToolMessage {
        role: "assistant".into(),
        content: String::new(),
        tool_calls: calls,
        tool_call_id: call_id.into(),
    };
    assert_eq!(
        t.render_tools(
            &[m(vec![call("")], ""), m(vec![call("c1")], "c0")],
            &[],
            false
        )
        .unwrap(),
        "[{\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"type\": \"function\", \
         \"function\": {\"name\": \"now\", \"arguments\": {\"tz\": \"UTC\"}}}]}, {\"role\": \
         \"assistant\", \"content\": \"\", \"tool_call_id\": \"c0\", \"tool_calls\": [{\"type\": \
         \"function\", \"function\": {\"name\": \"now\", \"arguments\": {\"tz\": \"UTC\"}}, \"id\": \
         \"c1\"}]}]"
    );
}

/// A call read from a model's `{"name", "arguments"}` has no ID: it reads as the empty one.
#[test]
fn tool_call_reads_without_an_id() {
    let c: ToolCall = serde_json::from_str(r#"{"name": "now", "arguments": {}}"#).unwrap();
    assert_eq!(c.id, "");
}

/// llama-server's rules for a tool spec: `type` must be `function`, the function needs a name, and
/// a description must be a string; a missing description is empty and missing parameters are `{}`
/// (llama.cpp's render below).
#[test]
fn tool_spec_must_be_an_openai_function() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::new(&data("qwen3.jinja"), &tok).unwrap();
    let hi = [ToolMessage {
        role: "user".into(),
        content: "hi".into(),
        ..Default::default()
    }];
    for spec in [
        serde_json::json!({"function": {"name": "now"}}),
        serde_json::json!({"type": "web", "function": {"name": "now"}}),
        serde_json::json!({"type": "function", "function": {"description": "no name"}}),
        serde_json::json!({"type": "function"}),
        serde_json::json!({"type": "function", "function": {"name": "now", "description": null}}),
    ] {
        assert!(
            t.render_tools(&hi, std::slice::from_ref(&spec), true)
                .is_err(),
            "{spec}"
        );
    }
    let bare = serde_json::json!({"type": "function", "function": {"name": "now"}});
    let prompt = t.render_tools(&hi, &[bare], true).unwrap();
    assert!(
        prompt.contains(
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n</tools>"
        ),
        "{prompt}"
    );
}

/// `tojson` escapes as llama.cpp's Jinja does: JSON's short escapes, `\u00XX` for other control
/// characters, and nothing else (no HTML escaping, UTF-8 kept). The expected text is tokdump's render
/// of this template at b11458.
#[test]
fn tojson_writes_like_llama() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let src = "{%- for message in messages -%}\n{{- '<|im_start|>' + message.role + '\\n' + (message.content | tojson) + '<|im_end|>\\n' -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n{{- '<|im_start|>assistant\\n' -}}\n{%- endif -%}";
    let t = ChatTemplate::new(src, &tok).unwrap();
    let m = Message {
        role: "user".into(),
        content: "a\u{1}\u{8}\u{c}\r\t\"\\/é<>&'\n".into(),
    };
    assert_eq!(
        t.render(&[m], true).unwrap(),
        "<|im_start|>user\n\"a\\u0001\\b\\f\\r\\t\\\"\\\\/é<>&'\\n\"<|im_end|>\n<|im_start|>assistant\n"
    );
}

/// `tojson` of what the tool data does not hold: an integer above 2^53, a float, none, empty
/// containers with `indent`, an iterable, keys out of order, and floats in C++'s 6 significant
/// digits, infinities and NaN. The expected text is tokdump's render at b11458.
#[test]
fn tojson_writes_values_like_llama() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let render = |src: &str| ChatTemplate::new(src, &tok).unwrap().render(&[], false);
    assert_eq!(
        render(
            "{{ {\"n\": 9007199254740993, \"f\": 0.5, \"z\": none, \"d\": {}, \"e\": []} \
             | tojson(indent=2) }}\n{{ range(2) | tojson }}\n{{ [0.123456789, 1000000.0, \
             100000.0, 0.0001, 0.00001234567, 2.0, -1.5, 123456789.0] | tojson }}"
        )
        .unwrap(),
        "{\n  \"n\": 9007199254740993,\n  \"f\": 0.5,\n  \"z\": null,\n  \"d\": {},\n  \"e\": []\n}\n\
         [0, 1]\n[0.123457, 1e+06, 100000, 0.0001, 1.23457e-05, 2, -1.5, 1.23457e+08]"
    );
    // 1e200 squared overflows to infinity. The NaN is negated because x86 makes inf - inf with the
    // sign bit set, which glibc prints as `-nan`; macOS prints any NaN as `nan`.
    let b = format!("1{}.0", "0".repeat(200));
    assert_eq!(
        render(
            &[
                "{% set b = ",
                &b,
                " %}{{ (b*b)|tojson }}|{{ (0-b*b)|tojson }}|{{ (-(b*b-b*b))|tojson }}"
            ]
            .concat()
        )
        .unwrap(),
        "inf|-inf|nan"
    );
}

/// `tojson`'s arguments as llama.cpp's Jinja takes them: `ensure_ascii` then `indent` by position
/// (so `tojson(2)` escapes non-ASCII text), a negative indent or a float as none, an indent cut to
/// C's `int` as llama.cpp casts it, and `sort_keys=false`. The expected text is tokdump's render at
/// b11458.
#[test]
fn tojson_takes_llamas_arguments() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::new(
        "{{ \"é😀\" | tojson(2) }}\n{{ \"é\" | tojson(ensure_ascii=false) }}\n\
         {{ \"é😀\" | tojson(ensure_ascii=true) }}\n{{ [1, [2]] | tojson(indent=-1) }}\n\
         {{ [1, {\"a\": 2}] | tojson(indent=0) }}\n{{ {\"b\": 1, \"a\": [3]} | tojson(false, 1) }}\n\
         {{ [] | tojson(sort_keys=false) }}\n{{ {\"é\": \"ü\"} | tojson(true) }}\n\
         {{ [1] | tojson(indent=2.0) }}\n{{ [1] | tojson(indent=4611686018427387904) }}",
        &tok,
    )
    .unwrap();
    assert_eq!(
        t.render(&[], false).unwrap(),
        "\"\\u00e9\\ud83d\\ude00\"\n\"é\"\n\"\\u00e9\\ud83d\\ude00\"\n[1, [2]]\n[\n1,\n{\n\"a\": 2\n}\n]\n\
         {\n \"b\": 1,\n \"a\": [\n  3\n ]\n}\n[]\n{\"\\u00e9\": \"\\u00fc\"}\n[1]\n[\n1\n]"
    );
}

/// A GGUF template can build a value that holds itself: `tojson` fails on it instead of
/// overflowing the stack.
#[test]
fn tojson_refuses_a_value_that_holds_itself() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let src = "{% set ns = namespace() %}{% set ns.me = ns %}{{ ns | tojson }}";
    assert!(
        ChatTemplate::new(src, &tok)
            .unwrap()
            .render(&[], false)
            .is_err()
    );
}

/// llama.cpp's own vocabulary files (`models/` at b11458): real Llama 3, Qwen2 and Llama 2 (SPM)
/// vocabularies, checkable without the model GGUFs. (dump name, file name)
const LLAMA_CPP_VOCABS: [(&str, &str); 3] = [
    ("llama-cpp-llama-bpe", "ggml-vocab-llama-bpe.gguf"),
    ("llama-cpp-qwen2", "ggml-vocab-qwen2.gguf"),
    ("llama-cpp-llama-spm", "ggml-vocab-llama-spm.gguf"),
];

#[test]
#[ignore = "needs a llama.cpp checkout at b11458: LARCH_LLAMA_CPP=<dir>"]
fn llama_cpp_vocabs() {
    let dir = std::env::var("LARCH_LLAMA_CPP").expect("LARCH_LLAMA_CPP=<llama.cpp checkout>");
    let prompts = data("prompts.jsonl");
    for (name, file) in LLAMA_CPP_VOCABS {
        let (_, tok) = tokenizer_file(&format!("{dir}/models/{file}"));
        check_tokens(&tok, name);
        check_lossless(&tok, name.ends_with("spm"), &prompts);
    }
}

/// The requirement-R4 models of the merge gate (`tools/llama-logits/model-dumps.sh`, which writes
/// their dumps and `models.sha256`) by opaque fixture ID, and whether the template roundtrip applies:
/// the gate reads `<LARCH_MODELS>/<id>.gguf`, the ID-to-file mapping stays outside the repo, and
/// `models.sha256` pins each file's bytes.
const MODELS: [(&str, bool); 4] = [("m1", true), ("m2", false), ("m3", false), ("m4", true)];

#[test]
#[ignore = "needs the model GGUFs: LARCH_MODELS=<dir>"]
fn models() {
    let dir = std::env::var("LARCH_MODELS").expect("LARCH_MODELS=<dir of the GGUFs>");
    let prompts = data("prompts.jsonl");
    for (name, roundtrip) in MODELS {
        let (g, tok) = tokenizer_file(&format!("{dir}/{name}.gguf"));
        check_tokens(&tok, name);
        check_lossless(&tok, false, &prompts);
        let mut t = ChatTemplate::from_gguf(&g, &tok).unwrap();
        t.date = DATE.into();
        check_renders(&t, &data(&format!("{name}.chat.jsonl")));
        if roundtrip {
            check_roundtrip(&t, &tok);
        }
    }
}

/// The tool renders of the m1 and m2 GGUFs' own templates against tokdump's, which
/// `tools/llama-logits/model-dumps.sh` writes next to the GGUFs, outside git, before it runs this (a
/// name apart from `models`, which the script selects by substring):
///   tokdump chat <dir>/<id>.gguf - tests/data/chat-tools.json "26 Jul 2024" > <dir>/<id>.tools.jsonl
/// m7's template has no tool support: llama.cpp raises on the tool conversation, so there is no
/// render to compare, and larch's render must fail the same way.
#[test]
#[ignore = "needs the GGUFs and their tool renders: LARCH_MODELS=<dir>"]
fn gguf_templates_render_tools() {
    let dir = std::env::var("LARCH_MODELS").expect("LARCH_MODELS=<dir of the GGUFs>");
    for id in ["m1", "m2"] {
        let (g, tok) = tokenizer_file(&format!("{dir}/{id}.gguf"));
        let mut t = ChatTemplate::from_gguf(&g, &tok).unwrap();
        t.date = DATE.into();
        let dump = format!("{dir}/{id}.tools.jsonl");
        check_tool_renders(&t, &std::fs::read_to_string(&dump).expect(&dump));
    }
    let (g, tok) = tokenizer_file(&format!("{dir}/m7.gguf"));
    check_tools_refused(&ChatTemplate::from_gguf(&g, &tok).unwrap());
}

/// m2's own template with `enable_thinking` true and false against tokdump's renders, which
/// `tools/llama-logits/model-dumps.sh` writes next to the GGUF, outside git, before it runs this
/// (`<c>` is `chat` for tests/data/chat.json and `tools` for tests/data/chat-tools.json):
///   tokdump chat <dir>/m2.gguf - tests/data/<conversation> "26 Jul 2024" true > <dir>/m2.thinking.<c>.jsonl
///   tokdump chat <dir>/m2.gguf - tests/data/<conversation> "26 Jul 2024" false >> <dir>/m2.thinking.<c>.jsonl
#[test]
#[ignore = "needs the m2 GGUF and its thinking renders: LARCH_MODELS=<dir>"]
fn gguf_template_reads_thinking() {
    let dir = std::env::var("LARCH_MODELS").expect("LARCH_MODELS=<dir of the GGUFs>");
    let (g, tok) = tokenizer_file(&format!("{dir}/m2.gguf"));
    let mut t = ChatTemplate::from_gguf(&g, &tok).unwrap();
    t.date = DATE.into();
    let read = |c: &str| {
        let dump = format!("{dir}/m2.thinking.{c}.jsonl");
        std::fs::read_to_string(&dump).expect(&dump)
    };
    check_thinking(&t, &read("chat"), &read("tools"));
}

/// Inkling 2's template (`data/inkling2.jinja`, the release's, which Unsloth's Mainline_Support GGUFs
/// hold byte for byte) renders: transformers' `{% generation %}` tags, which llama.cpp ignores,
/// `tojson(sort_keys=true, separators=(",", ":"))` for the tool declaration and `dictsort` for the
/// call's arguments. The expected text is Python Jinja2's render with the tags dropped and
/// `tojson` as `json.dumps`.
#[test]
fn inkling2_template_renders_tools_and_calls() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let t = ChatTemplate::new(&data("inkling2.jinja"), &tok).unwrap();
    let call = ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"unit": "c", "city": "Paris"}),
        id: "call_1".into(),
    };
    let msgs = [
        ("system", "Be brief.", vec![], ""),
        ("user", "Weather in Paris?", vec![], ""),
        ("assistant", "", vec![call], ""),
        ("tool", "{\"temp\": 21}", vec![], "call_1"),
    ]
    .map(|(role, content, tool_calls, id)| ToolMessage {
        role: role.into(),
        content: content.into(),
        tool_calls,
        tool_call_id: id.into(),
    });
    let tools = [
        serde_json::json!({"type": "function", "function": {"name": "get_weather",
        "description": "Weather now", "parameters": {"type": "object", "properties":
        {"city": {"type": "string"}, "unit": {"type": "string"}}, "required": ["city"]}}}),
    ];
    assert_eq!(
        t.render_tools(&msgs, &tools, true).unwrap(),
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Weather now\",\"name\":\"get_weather\",\
         \"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"}},\
         \"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>\
         <|message_system|><|content_text|>Be brief.<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>Weather in Paris?<|end_message|>\
         <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":\
         {\"city\":\"Paris\",\"unit\":\"c\"}}<|end_message|><|content_model_end_sampling|>\
         <|message_tool|>get_weather<|content_text|>{\"temp\": 21}<|end_message|>"
    );
}

/// `tojson`'s `separators` and `sort_keys`, by keyword and by position (after `ensure_ascii` and
/// `indent`), as llama.cpp's Jinja takes them: keys in byte order, nested objects too.
#[test]
fn tojson_sorts_keys_and_takes_separators() {
    let (_, tok) = tokenizer(&bpe_vocab("qwen2"));
    let render = |src: &str| {
        ChatTemplate::new(src, &tok)
            .unwrap()
            .render(&[], false)
            .unwrap()
    };
    let v = "{\"b\": [1, {\"z\": 1, \"a\": 2}], \"a\": 0, \"B\": 3}";
    assert_eq!(
        render(&format!(
            "{{{{ {v} | tojson(sort_keys=true, separators=(\",\", \":\")) }}}}"
        )),
        "{\"B\":3,\"a\":0,\"b\":[1,{\"a\":2,\"z\":1}]}"
    );
    assert_eq!(
        render(&format!(
            "{{{{ {v} | tojson(false, none, [\";\", \"=\"], true) }}}}"
        )),
        "{\"B\"=3;\"a\"=0;\"b\"=[1;{\"a\"=2;\"z\"=1}]}"
    );
    assert_eq!(
        render(&format!("{{{{ {v} | tojson(sort_keys=false) }}}}")),
        "{\"b\": [1, {\"z\": 1, \"a\": 2}], \"a\": 0, \"B\": 3}"
    );
    assert_eq!(
        render("{%- generation -%} x {%- endgeneration %}|{% generation %}y{% endgeneration -%} z"),
        "x|yz"
    );
}

/// Inkling 2's message tags are control tokens its tokenizer shows as text, as llama-server does
/// for preserved tokens, so its tool calls reach the parser; its EOS and the same tag in another
/// architecture's vocabulary stay hidden.
#[test]
fn inkling_tags_show_as_text() {
    for (arch, shown) in [
        ("inkling2_loop", true),
        ("inkling2", true),
        ("llama", false),
    ] {
        let mut kv = spm_kv();
        let tokens = kv
            .iter_mut()
            .find(|(k, _)| k == "tokenizer.ggml.tokens")
            .unwrap();
        let Value::Arr(tokens) = &mut tokens.1 else {
            unreachable!()
        };
        let tag = tokens.len() as u32;
        tokens.extend(
            ["<|message_model|>", "<|content_model_end_sampling|>"].map(|t| Value::Str(t.into())),
        );
        for (k, v) in &mut kv {
            match (k.as_str(), v) {
                ("tokenizer.ggml.token_type", Value::Arr(t)) => {
                    t.extend([Value::I32(3), Value::I32(3)])
                }
                ("tokenizer.ggml.scores", Value::Arr(s)) => {
                    s.extend([Value::F32(0.0), Value::F32(0.0)])
                }
                ("general.architecture", v) => *v = Value::Str(arch.into()),
                _ => {}
            }
        }
        let (_, tok) = tokenizer(&write(&kv, NO_TENSORS));
        let mut d = Detokenizer::new(Vec::new());
        let text = d.push(&tok, tag).0 + &d.push(&tok, tag + 1).0 + &d.finish();
        assert_eq!(text, if shown { "<|message_model|>" } else { "" }, "{arch}");
    }
}
