#![no_main]
use libfuzzer_sys::fuzz_target;
use larch_model::{ModelConfig, Tokenizer, gguf::Gguf};

fuzz_target!(|data: &[u8]| {
    // Only "no panic, no hang, no huge allocation" is checked: an Err is the expected outcome.
    if let Ok(g) = Gguf::parse(data) {
        let _ = ModelConfig::from_gguf(&g);
        let _ = Tokenizer::from_gguf(&g);
    }
});
