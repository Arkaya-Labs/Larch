//! Writes the seed corpus under `corpus/` (run from `fuzz/`).
use std::fs;

use larch_model::{Arch, tiny};

fn main() {
    let (gguf, tok) = ("corpus/gguf", "corpus/tokenizer");
    fs::create_dir_all(gguf).unwrap();
    fs::create_dir_all(tok).unwrap();
    for (name, arch) in [
        ("llama", Arch::Llama),
        ("qwen2", Arch::Qwen2),
        ("qwen3", Arch::Qwen3),
    ] {
        fs::write(format!("{gguf}/{name}"), tiny::gguf(arch, 1)).unwrap();
    }
    fs::copy(tiny::TOKENIZER_GGUF, format!("{gguf}/tiny-tokenizer")).unwrap();
    let prompts = fs::read_to_string("../crates/larch-model/tests/data/prompts.jsonl").unwrap();
    for (i, line) in prompts.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        // The target reads byte 0 as add_special, so each seed leads with a set flag.
        let mut seed = vec![1u8];
        seed.extend_from_slice(v["prompt"].as_str().unwrap().as_bytes());
        fs::write(format!("{tok}/{i:04}"), seed).unwrap();
    }
}
