//! The tokenizer from `tokenizer.ggml.*` (architecture 8): ports of llama.cpp's byte-level BPE (the
//! Llama 3 and Qwen pre-tokenisers) and SPM, so a prompt becomes the ids llama.cpp feeds (checked by
//! [`crate::tokcheck`] against a llama.cpp dump), and the streaming [`Detokenizer`].

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::sync::LazyLock;

use larch_records::LoadError;
use regex_syntax::hir::{Class, HirKind};

use crate::gguf::{Gguf, Value, code};

/// GGUF token types (`tokenizer.ggml.token_type`) as llama.cpp acts on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    User,
    Byte,
    /// Unused or undefined: never split out, decodes to nothing.
    Other,
}

/// llama.cpp's end-of-generation token texts for the families larch loads; llama.cpp also forces
/// them to control tokens.
const EOG_TEXTS: [&str; 9] = [
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "</s>",
];

/// GPT-2's byte-level alphabet: printable Latin-1 stands for itself, the other 68 bytes for U+0100 on.
static BYTE_CHARS: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut out = ['\0'; 256];
    let mut next = 0x100;
    for (b, c) in out.iter_mut().enumerate() {
        *c = if matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
            b as u8 as char
        } else {
            next += 1;
            char::from_u32(next - 1).expect("below U+0144")
        };
    }
    out
});

// llama.cpp's Unicode tables are version 15.1; later letters and digits are not letters or digits to it.
static LETTER: LazyLock<Vec<(char, char)>> = LazyLock::new(|| class(r"[\p{L}&&\p{Age=15.1}]"));
static NUMBER: LazyLock<Vec<(char, char)>> = LazyLock::new(|| class(r"[\p{N}&&\p{Age=15.1}]"));
/// Qwen 3.5's letter runs: letters and combining marks.
static LETTER_MARK: LazyLock<Vec<(char, char)>> =
    LazyLock::new(|| class(r"[[\p{L}\p{M}]&&\p{Age=15.1}]"));
/// Combining marks, which GPT-4o's pre-tokeniser counts as letters of either case.
static MARK: LazyLock<Vec<(char, char)>> = LazyLock::new(|| class(r"[\p{M}&&\p{Age=15.1}]"));

/// The ranges of a Unicode class, from the regex crate's tables.
fn class(re: &str) -> Vec<(char, char)> {
    match regex_syntax::parse(re).expect("a valid class").into_kind() {
        HirKind::Class(Class::Unicode(c)) => {
            c.ranges().iter().map(|r| (r.start(), r.end())).collect()
        }
        _ => unreachable!("a Unicode class"),
    }
}

fn in_class(ranges: &[(char, char)], c: char) -> bool {
    let i = ranges.partition_point(|&(_, hi)| hi < c);
    ranges.get(i).is_some_and(|&(lo, _)| lo <= c)
}

/// A GGUF model's tokenizer: llama.cpp's `llama_tokenize(add_special, parse_special = true)` and its
/// token pieces.
pub struct Tokenizer {
    spm: bool,
    /// The longest digit run the BPE pre-tokeniser keeps in one word: Llama 3 3, Qwen 1.
    digits: usize,
    /// Qwen 3.5: letter runs also take combining marks (`\p{M}`), which punctuation runs leave out.
    marks: bool,
    /// Inkling 2 (`inkling2`): GPT-4o's pre-tokeniser ([`gpt4o_split`]) in place of [`pre_split`].
    gpt4o: bool,
    /// Llama 3: a pre-tokenised word that is a token is emitted whole, without merges.
    ignore_merges: bool,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    bos: Option<u32>,
    eos: Option<u32>,
    /// Sorted.
    eog: Vec<u32>,
    kind: Vec<Kind>,
    score: Vec<f32>,
    ids: HashMap<String, u32>,
    /// BPE: (left, right) → (rank, merged token).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// BPE: the token of each byte's byte-level character; SPM: its `<0xXX>` token.
    byte_ids: [u32; 256],
    /// Control, user-defined and unknown tokens, longest text first: llama.cpp splits them out in this
    /// order before BPE or SPM sees the text.
    special: Vec<(String, u32)>,
    piece: Vec<Vec<u8>>,
}

enum Frag<'a> {
    Text(&'a str),
    Tok(u32),
}

impl Tokenizer {
    /// Reads `tokenizer.ggml.*`: byte-level BPE with the `llama-bpe`, `qwen2` or `qwen35`
    /// pre-tokeniser, or SPM.
    /// `UnsupportedArch` for another tokenizer; `BadGguf(KEY, 0)` for a missing or malformed key,
    /// including a vocabulary without a token for every byte.
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, LoadError> {
        let bad = LoadError::BadGguf(code::KEY, 0);
        let key = |k: &str| g.get(&format!("tokenizer.ggml.{k}"));
        let model = key("model").and_then(Value::as_str).ok_or(bad)?;
        let pre = key("pre").and_then(Value::as_str);
        // (spm, digits, marks, ignore_merges, add_bos, default bos and eos), as llama.cpp's vocab load.
        let (spm, digits, marks, ignore_merges, mut add_bos, (def_bos, def_eos)) =
            match (model, pre) {
                ("llama", _) => (true, 0, false, false, true, (1, 2)),
                ("gpt2", Some("llama-bpe" | "llama3" | "llama-v3")) => {
                    (false, 3, false, true, true, (11, 11))
                }
                ("gpt2", Some("qwen2")) => (false, 1, false, false, false, (11, 11)),
                ("gpt2", Some("qwen35")) => (false, 1, true, false, false, (11, 11)),
                ("gpt2", Some("inkling2")) => (false, 3, false, true, false, (11, 11)),
                _ => return Err(LoadError::UnsupportedArch),
            };
        let gpt4o = pre == Some("inkling2");
        let text = key("tokens")
            .and_then(Value::as_arr)
            .ok_or(bad)?
            .iter()
            .enumerate()
            .map(|(i, v)| match v.as_str() {
                Some("") => Ok(format!("[EMPTY_{i}]")),
                Some(t) => Ok(t.to_owned()),
                None => Err(bad),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let n = text.len();
        u32::try_from(n).map_err(|_| bad)?;
        // A per-token array of at least `n` entries, or `none` when the key is absent.
        let per_token = |k: &str, none| -> Result<Vec<Value>, LoadError> {
            match key(k).and_then(Value::as_arr) {
                None if key(k).is_none() => Ok(vec![none; n]),
                Some(a) if a.len() >= n => Ok(a[..n].to_vec()),
                _ => Err(bad),
            }
        };
        let score = per_token("scores", Value::F32(0.0))?
            .iter()
            .map(|v| match *v {
                Value::F32(s) => Ok(s),
                Value::I32(s) => Ok(s as f32),
                _ => Err(bad),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut kind = per_token("token_type", Value::I32(1))?
            .iter()
            .map(|v| match *v {
                Value::I32(1) => Ok(Kind::Normal),
                Value::I32(2) => Ok(Kind::Unknown),
                Value::I32(3) => Ok(Kind::Control),
                Value::I32(4) => Ok(Kind::User),
                Value::I32(6) => Ok(Kind::Byte),
                Value::I32(_) => Ok(Kind::Other),
                _ => Err(bad),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ids: HashMap<String, u32> = text
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        // A special token id from the GGUF; out of range keeps the default, as llama.cpp.
        let id = |k: &str, default: Option<u32>| {
            let default = default.filter(|&d| (d as usize) < n);
            match key(k).and_then(Value::as_u64) {
                Some(v) if v < n as u64 => Some(v as u32),
                _ => default,
            }
        };
        // llama.cpp's inkling2 has no BOS, whatever the file says.
        let bos = id("bos_token_id", Some(def_bos)).filter(|_| !gpt4o);
        let eos = id("eos_token_id", Some(def_eos));
        if let Some(Value::Bool(b)) = key("add_bos_token") {
            add_bos = *b;
        }
        let add_eos = matches!(key("add_eos_token"), Some(Value::Bool(true)));
        let add_space_prefix = match key("add_space_prefix") {
            Some(Value::Bool(b)) => *b,
            _ => spm,
        };
        if add_bos && bos.is_none() || add_eos && eos.is_none() {
            return Err(bad);
        }
        let mut eog: Vec<u32> = [eos, id("eot_token_id", None), id("eom_token_id", None)]
            .into_iter()
            .flatten()
            .collect();
        for t in EOG_TEXTS {
            if let Some(&i) = ids.get(t) {
                kind[i as usize] = Kind::Control;
                eog.push(i);
            }
        }
        eog.sort_unstable();
        eog.dedup();

        let mut merges = HashMap::new();
        if !spm {
            let list = key("merges").and_then(Value::as_arr).ok_or(bad)?;
            for (rank, m) in list.iter().enumerate() {
                let m = m.as_str().ok_or(bad)?;
                // llama.cpp splits at the first space after the first byte.
                let Some(sp) = m.bytes().skip(1).position(|b| b == b' ').map(|p| p + 1) else {
                    continue;
                };
                let (l, r) = (&m[..sp], &m[sp + 1..]);
                if let (Some(&a), Some(&b), Some(&c)) =
                    (ids.get(l), ids.get(r), ids.get(&format!("{l}{r}")))
                {
                    merges.entry((a, b)).or_insert((rank as u32, c));
                }
            }
        }
        // Every byte needs a token, so no text is lost: llama.cpp drops a byte it cannot spell (BPE)
        // or aborts unless a one-character token stands in (SPM); larch refuses such a vocabulary.
        let byte_ids: [u32; 256] = (0..256)
            .map(|b| {
                let t = if spm {
                    format!("<0x{b:02X}>")
                } else {
                    BYTE_CHARS[b].to_string()
                };
                ids.get(&t).copied()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(bad)?
            .try_into()
            .expect("256 bytes");
        let mut special: Vec<(String, u32)> = (0..n)
            .filter(|&i| matches!(kind[i], Kind::Control | Kind::User | Kind::Unknown))
            .map(|i| (text[i].clone(), i as u32))
            .collect();
        special.sort_by_key(|(t, _)| Reverse(t.len()));

        let byte_of: HashMap<char, u8> = (0..=255u8).map(|b| (BYTE_CHARS[b as usize], b)).collect();
        let piece = text
            .iter()
            .zip(&kind)
            .map(|(t, k)| match k {
                Kind::Control | Kind::Unknown | Kind::User => t.as_bytes().to_vec(),
                Kind::Normal if spm => t.replace('▁', " ").into_bytes(),
                // A character outside the byte-level alphabet stands for itself.
                Kind::Normal => t
                    .chars()
                    .flat_map(|c| match byte_of.get(&c) {
                        Some(&b) => vec![b],
                        None => c.to_string().into_bytes(),
                    })
                    .collect(),
                Kind::Byte => {
                    vec![
                        t.get(3..5)
                            .and_then(|h| u8::from_str_radix(h, 16).ok())
                            .unwrap_or(0),
                    ]
                }
                Kind::Other => Vec::new(),
            })
            .collect();

        Ok(Tokenizer {
            spm,
            digits,
            marks,
            gpt4o,
            ignore_merges,
            add_bos,
            add_eos,
            add_space_prefix,
            bos,
            eos,
            eog,
            kind,
            score,
            ids,
            merges,
            byte_ids,
            special,
            piece,
        })
    }

    /// llama.cpp's `llama_tokenize(text, add_special, parse_special = true)`: control, user-defined and
    /// unknown token texts become those tokens, the rest goes through BPE or SPM; `add_special` adds
    /// BOS and EOS where the GGUF asks for them.
    // ponytail: llama.cpp's algorithm on std maps, ~2.5 ms per 10k tokens for Llama 3 and ~10 ms for
    // SPM. Upgrade: a faster hasher and one-pass special-token search when re-tokenising shows in TTFT.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            out.extend(self.bos);
        }
        let mut prev_special = true;
        for f in self.partition(text) {
            match f {
                Frag::Tok(id) => {
                    out.push(id);
                    prev_special = true;
                }
                Frag::Text(s) if self.spm => {
                    let lead = if self.add_space_prefix && prev_special {
                        " "
                    } else {
                        ""
                    };
                    self.spm_encode(&format!("{lead}{s}").replace(' ', "▁"), &mut out);
                    prev_special = false;
                }
                Frag::Text(s) => {
                    let words = match self.gpt4o {
                        true => gpt4o_split(s),
                        false => pre_split(s, self.digits, self.marks),
                    };
                    for w in words {
                        self.bpe_word(w, &mut out);
                    }
                }
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// The bytes token `id` stands for, control tokens as their text (llama.cpp's
    /// `token_to_piece(special = true)`); empty for an id outside the vocabulary.
    pub fn decode_piece(&self, id: u32) -> &[u8] {
        self.piece.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// The BOS token, if the vocabulary has one.
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// The EOS token, if the vocabulary has one.
    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    /// Whether `encode(.., add_special)` starts with BOS, and so a template's leading BOS is dropped.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether `encode(.., add_special)` ends with EOS.
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// The tokens that end generation, sorted: EOS, EOT, EOM and llama.cpp's end-of-generation texts.
    pub fn eog(&self) -> &[u32] {
        &self.eog
    }

    /// Control and unknown tokens show no text in generated output (llama.cpp's `special = false`).
    fn hidden(&self, id: u32) -> bool {
        matches!(
            self.kind.get(id as usize),
            Some(Kind::Control | Kind::Unknown)
        )
    }

    /// llama.cpp's `tokenizer_st_partition`: each special token in turn, longest first, cuts every
    /// remaining text fragment at all its occurrences.
    fn partition<'a>(&self, text: &'a str) -> Vec<Frag<'a>> {
        let mut frags = if text.is_empty() {
            Vec::new()
        } else {
            vec![Frag::Text(text)]
        };
        for (t, id) in &self.special {
            let (t, id) = (t.as_str(), *id);
            if !text.contains(t) {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 2);
            for f in frags {
                let Frag::Text(s) = f else {
                    next.push(f);
                    continue;
                };
                let mut at = 0;
                for (m, _) in s.match_indices(t) {
                    if m > at {
                        next.push(Frag::Text(&s[at..m]));
                    }
                    next.push(Frag::Tok(id));
                    at = m + t.len();
                }
                if at < s.len() {
                    next.push(Frag::Text(&s[at..]));
                }
            }
            frags = next;
        }
        frags
    }

    /// llama.cpp's `llm_tokenizer_bpe_session` on one pre-tokenised word: one symbol per byte, then the
    /// lowest-ranked merge of adjacent symbols, leftmost first, until none applies.
    fn bpe_word(&self, w: &str, out: &mut Vec<u32>) {
        if self.ignore_merges {
            let enc: String = w.bytes().map(|b| BYTE_CHARS[b as usize]).collect();
            if let Some(&id) = self.ids.get(&enc) {
                out.push(id);
                return;
            }
        }
        struct Sym {
            id: u32,
            /// 0 once merged into the symbol on its left.
            len: usize,
            prev: usize,
            next: usize,
        }
        let n = w.len();
        let mut sym: Vec<Sym> = w
            .bytes()
            .enumerate()
            .map(|(i, b)| Sym {
                id: self.byte_ids[b as usize],
                len: 1,
                prev: i.wrapping_sub(1),
                next: i + 1,
            })
            .collect();
        let rank = |sym: &[Sym], l: usize, r: usize| {
            let (a, b) = (sym.get(l)?.id, sym.get(r)?.id);
            self.merges.get(&(a, b)).copied()
        };
        let mut heap = BinaryHeap::new();
        for i in 1..n {
            if let Some((k, _)) = rank(&sym, i - 1, i) {
                heap.push(Reverse((k, i - 1, i)));
            }
        }
        while let Some(Reverse((k, l, r))) = heap.pop() {
            // Stale unless both symbols are still adjacent and still form this merge.
            if sym[l].len == 0 || sym[l].next != r {
                continue;
            }
            let Some((_, merged)) = rank(&sym, l, r).filter(|&(k2, _)| k2 == k) else {
                continue;
            };
            sym[l].id = merged;
            sym[l].len += sym[r].len;
            sym[r].len = 0;
            let after = sym[r].next;
            sym[l].next = after;
            if let Some(s) = sym.get_mut(after) {
                s.prev = l;
            }
            let (p, nx) = (sym[l].prev, sym[l].next);
            if let Some((k, _)) = rank(&sym, p, l) {
                heap.push(Reverse((k, p, l)));
            }
            if let Some((k, _)) = rank(&sym, l, nx) {
                heap.push(Reverse((k, l, nx)));
            }
        }
        out.extend(sym.iter().filter(|s| s.len > 0).map(|s| s.id));
    }

    /// llama.cpp's `llm_tokenizer_spm_session`: one symbol per char, then repeatedly the adjacent pair
    /// whose concatenation is the highest-scoring token, leftmost first; leftovers become byte tokens.
    fn spm_encode(&self, text: &str, out: &mut Vec<u32>) {
        struct Sym {
            start: usize,
            /// 0 once merged into the symbol on its left.
            len: usize,
            prev: usize,
            next: usize,
        }
        let mut sym: Vec<Sym> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, c))| Sym {
                start,
                len: c.len_utf8(),
                prev: i.wrapping_sub(1),
                next: i + 1,
            })
            .collect();
        let bigram = |sym: &[Sym], l: usize, r: usize| {
            let (a, b) = (sym.get(l)?, sym.get(r)?);
            let s = &text[a.start..a.start + a.len + b.len];
            let id = *self.ids.get(s)?;
            Some(Bigram {
                score: self.score[id as usize],
                left: l,
                right: r,
                size: s.len(),
            })
        };
        let mut heap: BinaryHeap<Bigram> = (1..sym.len())
            .filter_map(|i| bigram(&sym, i - 1, i))
            .collect();
        while let Some(Bigram {
            left: l,
            right: r,
            size,
            ..
        }) = heap.pop()
        {
            if sym[l].len == 0 || sym[r].len == 0 || sym[l].len + sym[r].len != size {
                continue;
            }
            sym[l].len += sym[r].len;
            sym[r].len = 0;
            let after = sym[r].next;
            sym[l].next = after;
            if let Some(s) = sym.get_mut(after) {
                s.prev = l;
            }
            let (p, nx) = (sym[l].prev, sym[l].next);
            heap.extend(bigram(&sym, p, l));
            heap.extend(bigram(&sym, l, nx));
        }
        let mut i = 0;
        while let Some(s) = sym.get(i) {
            let t = &text[s.start..s.start + s.len];
            match self.ids.get(t) {
                Some(&id) => out.push(id),
                None => out.extend(t.bytes().map(|b| self.byte_ids[b as usize])),
            }
            i = s.next;
        }
    }
}

/// An SPM merge candidate: the highest-scoring token first, then the leftmost.
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then(o.left.cmp(&self.left))
    }
}

/// llama.cpp's hand-written pre-tokeniser for the Llama 3 regex (`digits` 3) and Qwen2's (`digits`
/// 1): `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,digits}|
/// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. With `marks`, Qwen 3.5's: `[\p{L}\p{M}]+` for
/// `\p{L}+` and `[^\s\p{L}\p{M}\p{N}]+` for `[^\s\p{L}\p{N}]+`.
fn pre_split(text: &str, digits: usize, marks: bool) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| cs.get(i).map(|&(_, c)| c);
    // ASCII first: most text is, and the table search costs more than the rest of the split. No
    // ASCII character is a mark.
    let letters: &[(char, char)] = if marks { &LETTER_MARK } else { &LETTER };
    let word =
        |i| at(i).is_some_and(|c| c.is_ascii_alphabetic() || !c.is_ascii() && in_class(letters, c));
    let number =
        |i| at(i).is_some_and(|c| c.is_ascii_digit() || !c.is_ascii() && in_class(&NUMBER, c));
    let space = |i| at(i).is_some_and(char::is_whitespace);
    let byte = |i: usize| cs.get(i).map_or(text.len(), |&(b, _)| b);
    let mut words = Vec::new();
    let mut start = 0;
    let mut cut = |end: usize| {
        if end > start {
            words.push(&text[byte(start)..byte(end)]);
        }
        start = end;
    };
    let mut pos = 0;
    while pos < n {
        let c = cs[pos].1;
        if c == '\'' && pos + 1 < n {
            let c1 = cs[pos + 1].1.to_ascii_lowercase();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                pos += 2;
                cut(pos);
                continue;
            }
            if let Some(c2) = at(pos + 2).map(|c| c.to_ascii_lowercase())
                && matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l'))
            {
                pos += 3;
                cut(pos);
                continue;
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+, with marks [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(c == '\r' || c == '\n' || number(pos)) && (word(pos) || word(pos + 1)) {
            pos += 1;
            while word(pos) {
                pos += 1;
            }
            cut(pos);
            continue;
        }
        // \p{N}{1,digits}
        if number(pos) {
            let mut ini = pos;
            while number(pos) {
                pos += 1;
                if pos - ini >= digits {
                    cut(pos);
                    ini = pos;
                }
            }
            cut(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+[\r\n]*, with marks <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*; past the
        // end counts as a match, as in llama.cpp.
        let other = |i| i >= n || !(space(i) || word(i) || number(i));
        if other(if c == ' ' { pos + 1 } else { pos }) {
            pos += usize::from(c == ' ');
            while pos < n && other(pos) {
                pos += 1;
            }
            while matches!(at(pos), Some('\r' | '\n')) {
                pos += 1;
            }
            cut(pos);
            continue;
        }
        let mut ws = 0;
        let mut last_newline = None;
        while space(pos + ws) {
            if matches!(cs[pos + ws].1, '\r' | '\n') {
                last_newline = Some(pos + ws + 1);
            }
            ws += 1;
        }
        // \s*[\r\n]+, then \s+(?!\S), then \s+
        pos = match last_newline {
            Some(end) => end,
            None if ws > 1 && pos + ws < n => pos + ws - 1,
            None => pos + ws.max(1),
        };
        cut(pos);
    }
    words
}

/// GPT-4o's pre-tokeniser as llama.cpp b12723 runs it (`LLAMA_VOCAB_PRE_TYPE_GPT4O`): one regex,
/// `[^\r\n\p{L}\p{N}]?U*L+C?|[^\r\n\p{L}\p{N}]?U+L*C?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|
/// \s*[\r\n]+|\s+(?!\S)|\s+`, with U a letter or mark other than a-z, L one other than A-Z (so every
/// letter outside ASCII and every mark is both, llama.cpp's stand-in for `\p{Lu}` and `\p{Ll}`) and C
/// `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll` or `'d` in either case. std::regex matches it on text whose
/// characters outside ASCII are collapsed to their category, the first alternative that matches
/// winning, each quantifier greedy and given back one at a time; every character starts a match.
fn gpt4o_split(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| cs.get(i).map(|&(_, c)| c);
    let letter = |c: char| c.is_ascii_alphabetic() || !c.is_ascii() && in_class(&LETTER, c);
    let number = |c: char| c.is_ascii_digit() || !c.is_ascii() && in_class(&NUMBER, c);
    let mark = |c: char| !c.is_ascii() && in_class(&MARK, c);
    let is = |i: usize, f: &dyn Fn(char) -> bool| at(i).is_some_and(f);
    let space = |i| is(i, &char::is_whitespace);
    // [^\r\n\p{L}\p{N}], U and L.
    let lead = |i| is(i, &|c| !(c == '\r' || c == '\n' || letter(c) || number(c)));
    let upper = |i| is(i, &|c| (letter(c) || mark(c)) && !c.is_ascii_lowercase());
    let lower = |i| is(i, &|c| (letter(c) || mark(c)) && !c.is_ascii_uppercase());
    let run = |mut i: usize, f: &dyn Fn(usize) -> bool| {
        let from = i;
        while i < n && f(i) {
            i += 1;
        }
        i - from
    };
    // C?: its length at i, the alternatives in order.
    let suffix = |i: usize| {
        let low = |k: usize| at(i + k).map(|c| c.to_ascii_lowercase());
        if at(i) != Some('\'') {
            return 0;
        }
        match (low(1), low(2)) {
            (Some('s' | 't'), _) => 2,
            (Some('r'), Some('e')) | (Some('v'), Some('e')) => 3,
            (Some('m'), _) => 2,
            (Some('l'), Some('l')) => 3,
            (Some('d'), _) => 2,
            _ => 0,
        }
    };
    // The match starting at i, as its end.
    let matched = |i: usize| -> usize {
        let leads: &[usize] = if lead(i) { &[1, 0] } else { &[0] };
        // [^\r\n\p{L}\p{N}]?U*L+C?: the longest U run that leaves an L after it.
        for &p in leads {
            let j = i + p;
            for k in (0..=run(j, &upper)).rev() {
                let l = run(j + k, &lower);
                if l > 0 {
                    return j + k + l + suffix(j + k + l);
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?U+L*C?
        for &p in leads {
            let j = i + p;
            let k = run(j, &upper);
            if k > 0 {
                let l = run(j + k, &lower);
                return j + k + l + suffix(j + k + l);
            }
        }
        // \p{N}{1,3}
        let digits = run(i, &|i| is(i, &number)).min(3);
        if digits > 0 {
            return i + digits;
        }
        // ' '?[^\s\p{L}\p{N}]+[\r\n/]*
        let other = |i| is(i, &|c| !(c.is_whitespace() || letter(c) || number(c)));
        for p in if at(i) == Some(' ') { [1, 0] } else { [0, 0] } {
            let o = run(i + p, &other);
            if o > 0 {
                let j = i + p + o;
                return j + run(j, &|i| matches!(at(i), Some('\r' | '\n' | '/')));
            }
        }
        let w = run(i, &space);
        // \s*[\r\n]+: through the run's last line break.
        if let Some(q) = (i..i + w)
            .rev()
            .find(|&q| matches!(at(q), Some('\r' | '\n')))
        {
            return q + 1;
        }
        // \s+(?!\S), else \s+.
        if i + w < n && w > 1 {
            i + w - 1
        } else {
            i + w.max(1)
        }
    };
    let byte = |i: usize| cs.get(i).map_or(text.len(), |&(b, _)| b);
    let mut words = Vec::new();
    let mut i = 0;
    while i < n {
        let end = matched(i);
        words.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    words
}

/// Streams generated tokens as text: holds back an incomplete UTF-8 sequence and any tail that may
/// still grow into a stop string, and ends the text before the first stop string.
pub struct Detokenizer {
    stops: Vec<String>,
    /// An incomplete UTF-8 sequence at most.
    bytes: Vec<u8>,
    /// Decoded text not returned yet because a stop string may start in it.
    held: String,
}

impl Detokenizer {
    /// Empty stop strings are ignored.
    pub fn new(stops: Vec<String>) -> Detokenizer {
        Detokenizer {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            bytes: Vec::new(),
            held: String::new(),
        }
    }

    /// Feeds one generated token: returns the text now safe to show and whether a stop string ended
    /// the text (the stop string is never returned). Control tokens add no text, as in llama-server.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> (String, bool) {
        if !tok.hidden(id) {
            self.bytes.extend_from_slice(tok.decode_piece(id));
        }
        loop {
            match std::str::from_utf8(&self.bytes) {
                Ok(s) => {
                    self.held.push_str(s);
                    self.bytes.clear();
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    self.held
                        .push_str(std::str::from_utf8(&self.bytes[..ok]).expect("valid prefix"));
                    let Some(bad) = e.error_len() else {
                        self.bytes.drain(..ok);
                        break;
                    };
                    self.held.push(char::REPLACEMENT_CHARACTER);
                    self.bytes.drain(..ok + bad);
                }
            }
        }
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let out = self.held[..at].to_owned();
            self.held.clear();
            self.bytes.clear();
            return (out, true);
        }
        let keep = self
            .stops
            .iter()
            .map(|s| overlap(&self.held, s))
            .max()
            .unwrap_or(0);
        let out = self.held[..self.held.len() - keep].to_owned();
        self.held.drain(..self.held.len() - keep);
        (out, false)
    }

    /// The text still held back when generation ends; a trailing partial UTF-8 sequence becomes U+FFFD.
    pub fn finish(&mut self) -> String {
        let mut out = std::mem::take(&mut self.held);
        out.push_str(&String::from_utf8_lossy(&self.bytes));
        self.bytes.clear();
        out
    }
}

/// The longest tail of `text` that is a proper prefix of `stop`.
fn overlap(text: &str, stop: &str) -> usize {
    (1..stop.len().min(text.len() + 1))
        .rev()
        .find(|&k| {
            stop.as_bytes()
                .starts_with(&text.as_bytes()[text.len() - k..])
        })
        .unwrap_or(0)
}
