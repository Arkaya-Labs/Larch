//! `larch tokcheck -m <gguf> --ref <tokens.jsonl>` (architecture 10.2, 15): larch's tokens for the
//! prompts of a llama.cpp tokenisation dump (`tools/llama-logits` tokdump, one prompt and its tokens
//! per line) against the dump's. A mismatch is an error naming the first differing prompt and
//! position, so the command exits 1.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

use anyhow::Context;
use larch_model::gguf::Gguf;
use larch_model::{Tokenizer, tokcheck};

/// Checks every prompt of `reference` and prints how many matched; reads only the GGUF's header.
pub fn run(gguf: &Path, reference: &Path) -> anyhow::Result<()> {
    let f = File::open(gguf).with_context(|| gguf.display().to_string())?;
    let g = Gguf::read(BufReader::new(&f), f.metadata()?.len())?;
    let tok = Tokenizer::from_gguf(&g)?;
    let text = fs::read_to_string(reference).with_context(|| reference.display().to_string())?;
    let n = tokcheck(&tok, &text)?;
    println!("tokcheck: all {n} prompts match llama.cpp");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_missing_gguf_is_an_error_naming_it() {
        let p = std::path::Path::new("no-such.gguf");
        let e = super::run(p, p).unwrap_err();
        assert!(e.to_string().contains("no-such.gguf"), "{e}");
    }
}
