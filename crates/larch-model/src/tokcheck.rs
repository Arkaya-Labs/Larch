//! `larch tokcheck` (architecture 10.2, 15): larch's tokens for a prompt set against llama.cpp's.

use std::fmt;

use serde_json::Value;

use crate::Tokenizer;

/// Why a tokcheck failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokcheckError {
    /// 1-based line of the reference that is not a `{"prompt", "tokens"}` line.
    BadLine(usize),
    /// The reference has no prompts, so a pass would check nothing.
    Empty,
    /// The first prompt (0-based) that tokenises differently, at its first differing position; `None`
    /// is past the end of that side's tokens.
    Mismatch {
        prompt: usize,
        position: usize,
        ours: Option<u32>,
        llama: Option<u32>,
    },
}

impl fmt::Display for TokcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokcheckError::BadLine(line) => write!(f, "reference line {line}: bad format"),
            TokcheckError::Empty => write!(f, "no prompts"),
            TokcheckError::Mismatch {
                prompt,
                position,
                ours,
                llama,
            } => write!(
                f,
                "prompt {prompt}: first difference at token {position}: larch {ours:?}, llama.cpp {llama:?}"
            ),
        }
    }
}

impl std::error::Error for TokcheckError {}

/// Tokenises the `prompt` of each `{"prompt": "...", "tokens": [...]}` line of `reference` (a
/// `tools/llama-logits` tokdump dump) with BOS/EOS as the model asks (llama.cpp's
/// `llama_tokenize(add_special, parse_special)`) and compares the result with that line's `tokens`.
/// `Ok` is the number of prompts, at least one, all equal.
pub fn tokcheck(tok: &Tokenizer, reference: &str) -> Result<usize, TokcheckError> {
    let mut n = 0;
    for (i, line) in reference.lines().enumerate() {
        let bad = TokcheckError::BadLine(i + 1);
        let v: Value = serde_json::from_str(line).map_err(|_| bad)?;
        let text = v["prompt"].as_str().ok_or(bad)?;
        let llama = v["tokens"]
            .as_array()
            .ok_or(bad)?
            .iter()
            .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()).ok_or(bad))
            .collect::<Result<Vec<_>, _>>()?;
        check(tok, i, text, &llama)?;
        n += 1;
    }
    if n == 0 {
        return Err(TokcheckError::Empty);
    }
    Ok(n)
}

/// Prompt `prompt` (0-based): larch's tokens for `text` against llama.cpp's.
pub(crate) fn check(
    tok: &Tokenizer,
    prompt: usize,
    text: &str,
    llama: &[u32],
) -> Result<(), TokcheckError> {
    let ours = tok.encode(text, true);
    if ours == llama {
        return Ok(());
    }
    let position = ours.iter().zip(llama).take_while(|(a, b)| a == b).count();
    Err(TokcheckError::Mismatch {
        prompt,
        position,
        ours: ours.get(position).copied(),
        llama: llama.get(position).copied(),
    })
}
