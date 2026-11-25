#![no_main]
use std::sync::LazyLock;

use libfuzzer_sys::fuzz_target;
use larch_model::{Detokenizer, Tokenizer, gguf::Gguf};

/// The tokenizer and its vocabulary size (the Tokenizer does not expose one).
static TOK: LazyLock<(Tokenizer, usize)> = LazyLock::new(|| {
    let bytes = include_bytes!("../../crates/larch-model/tests/data/tiny-tokenizer.gguf");
    let g = Gguf::parse(bytes).unwrap();
    let n = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_arr())
        .unwrap()
        .len();
    (Tokenizer::from_gguf(&g).unwrap(), n)
});

fuzz_target!(|data: &[u8]| {
    let Some((&flags, text)) = data.split_first() else {
        return;
    };
    let (tok, vocab) = &*TOK;
    let ids = tok.encode(&String::from_utf8_lossy(text), flags & 1 == 1);
    assert!(
        ids.iter().all(|&id| (id as usize) < *vocab),
        "id out of vocab: {ids:?}"
    );
    let mut d = Detokenizer::new(vec![]);
    for id in ids {
        tok.decode_piece(id);
        d.push(tok, id);
    }
    d.finish();
});
