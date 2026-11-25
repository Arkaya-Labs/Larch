//! The model oracle (architecture 15): llama.cpp's teacher-forced dumps from `tools/llama-logits`
//! (`llama-logits`), a run compared against one, and the KL bound and tie margin calibrated on
//! llama.cpp's own run-to-run differences.
//!
//! At each position, with p the reference and q the run under test, both log-probabilities over the
//! whole vocabulary, KL = Σ p_i (log p_i − log q_i) over the reference's top 64 ids i, plus one term
//! for the mass outside them, (1 − P) ln((1 − P) / (1 − Q)) with P and Q the two sides' mass on those
//! ids. An id missing from q's top 64 takes q's 64th log-probability, which is at least its own, so that
//! term is a lower bound on its true value. The sum is the KL between the two distributions coarsened to
//! p's top 64 ids and one bucket for the rest (each side renormalised if an f32 log-sum-exp moved its
//! total off 1): never negative beyond rounding, identical rows score 0, a run sharper than the reference
//! cannot score below 0 and cancel another position's error, and with no id missing it is a lower bound
//! on the full KL.
//!
//! A position is decisive when the reference's top-1 log-probability leads its second by more than the
//! margin (a top of one entry always is). A run passes ([`Report::passes`]) when its top-1 is the
//! reference's at ≥ 99 % of the decisive positions and its mean KL over all positions is at most the
//! bound; its top-1 over all positions is reported beside. The bound and the margin come from llama.cpp
//! against itself ([`calibrate`]), the reference paired with other runs of it (another backend, or the
//! same at another micro-batch): the bound is 1.5 × the largest pair's p99 KL over positions, the
//! margin the largest reference top-1/top-2 gap at which any pair's top-1 differs. llama.cpp's own
//! backends flip top-1 at near-ties, so 99 % at every position is a gate the reference itself misses.

use std::fmt;
use std::fs::File;
use std::io::BufRead;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::Tokenizer;
use crate::tokcheck::{self, TokcheckError};

/// Candidates kept per position.
pub const TOP_K: usize = 64;
/// The dump format `llama-logits` writes (`larch_oracle` in its header).
pub const FORMAT: u64 = 1;
/// Lowest top-1 agreement that passes.
pub const TOP1_MIN: f64 = 0.99;
/// The bound over the largest cross-backend p99.
pub const BOUND_FACTOR: f64 = 1.5;
/// Calibration runs on the first 100 prompts of the bench prompt set...
pub const CALIBRATION_PROMPTS: usize = 100;
/// ...of which at least 4 are longer than 1100 tokens.
pub const LONG_PROMPTS: usize = 4;
/// Tokens a long prompt exceeds.
pub const LONG_TOKENS: usize = 1100;
/// What an f32 log-sum-exp may overshoot by: a top log-probability that much above 0, a top 64 whose
/// probabilities sum that much over 1.
const F32_SLACK: f64 = 1e-3;
/// The smallest tail an f32 top 64 resolves: below it, 1 − P or 1 − Q is rounding.
const TAIL_FLOOR: f64 = f32::EPSILON as f64;

/// One position's top 64: (token id, log-probability over the whole vocabulary), most likely first.
pub type Top = Vec<(u32, f32)>;

/// Why a dump, a run or a calibration set was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum OracleError {
    /// Reading the dump failed.
    Io(std::io::ErrorKind),
    /// 1-based line that is not JSON of the dump's shape, or a prompt line out of order.
    BadLine(usize),
    /// A header field that is missing or invalid.
    BadHeader(&'static str),
    /// The dump is of another GGUF.
    WrongModel { expected: String, found: String },
    /// The header's prompt count and the prompts given differ (a cut-off dump, a short run).
    PromptCount { expected: usize, found: usize },
    /// Prompt `prompt` (0-based) has `positions` rows for its `tokens` tokens.
    Positions {
        prompt: usize,
        tokens: usize,
        positions: usize,
    },
    /// An invalid token, id or log-probability at a prompt's position.
    BadValue {
        prompt: usize,
        position: usize,
        what: &'static str,
    },
    /// Dumps of one calibration set differ in this field, or (`run`) another dump has the reference's
    /// header; or no other run was given.
    Mismatch(&'static str),
    /// A calibration set that is not the protocol's: its prompts, and how many exceed [`LONG_TOKENS`].
    CalibrationSet { prompts: usize, long: usize },
    /// The worst pair's p99 KL is not positive, so it gives no bound.
    NoBound(f64),
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OracleError::Io(k) => write!(f, "reading the dump: {k}"),
            OracleError::BadLine(l) => write!(f, "dump line {l}: bad format"),
            OracleError::BadHeader(k) => write!(f, "dump header: missing or invalid {k}"),
            OracleError::WrongModel { expected, found } => {
                write!(f, "dump of model {found}, expected {expected}")
            }
            OracleError::PromptCount { expected, found } => {
                write!(f, "{found} prompts, expected {expected}")
            }
            OracleError::Positions {
                prompt,
                tokens,
                positions,
            } => write!(
                f,
                "prompt {prompt}: {positions} positions for {tokens} tokens"
            ),
            OracleError::BadValue {
                prompt,
                position,
                what,
            } => write!(f, "prompt {prompt} position {position}: {what}"),
            OracleError::Mismatch(k) => write!(f, "calibration dumps differ: {k}"),
            OracleError::CalibrationSet { prompts, long } => write!(
                f,
                "calibration needs {CALIBRATION_PROMPTS} prompts with {LONG_PROMPTS} over \
                 {LONG_TOKENS} tokens; got {prompts} with {long}"
            ),
            OracleError::NoBound(p99) => {
                write!(f, "the worst backend pair's p99 KL is {p99:e}: no bound")
            }
        }
    }
}

impl std::error::Error for OracleError {}

/// A dump's first line: what reproduces the run (the whole line is kept in `raw`).
#[derive(Clone, Debug)]
pub struct Header {
    /// The llama.cpp release the tool was built against.
    pub llama_cpp_tag: String,
    /// Its commit, as llama.cpp's build info gives it.
    pub llama_cpp_commit: String,
    /// Of the GGUF.
    pub model_sha256: String,
    /// Of the whole prompt file (the dump may cover only its first prompts).
    pub prompts_sha256: String,
    /// The ggml backend that computed it, by registry name: `CPU`, `CUDA`, `MTL` (Metal), ...
    pub backend: String,
    /// Ids and tokens are below it.
    pub n_vocab: u32,
    /// Prompt lines that follow.
    pub prompts: usize,
    /// The whole header: devices, context parameters, threads, flash attention, ...
    pub raw: Value,
}

/// One prompt of a dump.
#[derive(Clone, Debug)]
pub struct Prompt {
    /// The text llama.cpp tokenised (for a chat line, the rendered prompt).
    pub text: String,
    /// llama.cpp's tokens for `text`, which the model was fed; [`IMAGE_ROW`] at an image's rows.
    pub tokens: Vec<u32>,
    /// The images of a `--mmproj` dump, in order.
    pub images: Vec<ImageRows>,
    /// One per token: the prediction for the token after it.
    pub top: Vec<Top>,
}

/// The token of a row fed an image's embedding row rather than a token.
pub const IMAGE_ROW: u32 = u32::MAX;

/// Prompt `p`'s positions a vision gate scores: every text row, and each image's last row (it
/// predicts the image's closing text). An image's other rows predict nothing that is ever sampled,
/// and llama.cpp disagrees with itself there (InternVL3.5-1B: top-1 0.91 on them, 0.993 on text).
pub fn text_positions(p: &Prompt) -> Vec<usize> {
    (0..p.tokens.len())
        .filter(|&i| {
            p.tokens[i] != IMAGE_ROW || p.tokens.get(i + 1).is_none_or(|&t| t != IMAGE_ROW)
        })
        .collect()
}

/// `dump` with only each prompt's [`text_positions`] (its `tokens` and `top`; `images` kept as
/// they were, for provenance).
pub fn text_only(dump: &Dump) -> Dump {
    let mut d = dump.clone();
    for p in &mut d.prompts {
        let keep = text_positions(p);
        p.tokens = keep.iter().map(|&i| p.tokens[i]).collect();
        p.top = keep.iter().map(|&i| p.top[i].clone()).collect();
    }
    d
}

/// One image of a `--mmproj` dump: rows `at..at + n` of its prompt, fed the embedding rows that
/// `embd` (beside the dump) holds, f32 `[n, d_model]`, encoded from `file` (relative to the prompts).
/// On an M-RoPE model (Qwen-VL), `grid` is its rows' (nx, ny) and `n_pos` the positions it takes.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRows {
    pub at: usize,
    pub n: usize,
    pub embd: String,
    pub file: String,
    pub grid: Option<(u32, u32)>,
    pub n_pos: Option<u32>,
}

/// A parsed, checked `llama-logits` dump.
#[derive(Clone, Debug)]
pub struct Dump {
    /// The run's provenance.
    pub header: Header,
    /// In the prompt file's order.
    pub prompts: Vec<Prompt>,
}

/// The sha256 of the file at `path`, lowercase hex, from the platform's hasher (`sha256sum`,
/// `shasum`, or Windows' `certutil`, which may space its digest's bytes): no dependency for a
/// digest every machine already computes. If no hasher gives one, the first that failed (started and
/// failed, or could not be started for any reason but being absent) is the error, and `NotFound` only
/// if every one is absent.
pub fn sha256(path: &Path) -> std::io::Result<String> {
    digest(path, &HASHERS)
}

/// Command and leading arguments of each platform hasher, in the order [`sha256`] tries them.
const HASHERS: [(&str, &[&str]); 3] = [
    ("sha256sum", &[]),
    ("shasum", &["-a", "256"]),
    ("certutil", &["-hashfile"]),
];

/// [`sha256`] with `hashers` (command, leading arguments) tried in order.
fn digest(path: &Path, hashers: &[(&str, &[&str])]) -> std::io::Result<String> {
    File::open(path)?; // a missing file is that error, not "no hasher"
    // Absolute, so a path starting with `-` is never read as an option.
    let p = std::path::absolute(path)?;
    let mut failed = None;
    for &(cmd, args) in hashers {
        let mut c = Command::new(cmd);
        c.args(args).arg(&p);
        if cmd == "certutil" {
            c.arg("SHA256");
        }
        let out = match c.output() {
            Ok(out) => out,
            // Not installed: the next one may be.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                failed.get_or_insert_with(|| std::io::Error::new(e.kind(), format!("{cmd}: {e}")));
                continue;
            }
        };
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                // GNU sha256sum starts the line with `\` when the path holds one (Windows paths).
                let line = line.trim_start_matches('\\');
                let first = line.split_whitespace().next().unwrap_or_default();
                let joined: String = line.split_whitespace().collect();
                let mut digest = [first, &joined].into_iter();
                if let Some(d) = digest.find(|s| is_sha256(&s.to_ascii_lowercase())) {
                    return Ok(d.to_ascii_lowercase());
                }
            }
        }
        failed.get_or_insert_with(|| {
            let stderr = String::from_utf8_lossy(&out.stderr);
            std::io::Error::other(format!("{cmd} ({}): {}", out.status, stderr.trim()))
        });
    }
    Err(failed.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no sha256sum, shasum or certutil to hash it with",
        )
    }))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A position's top 64 is exactly that: 64 distinct in-vocabulary ids, finite log-probabilities ≤ 0
/// in descending order, whose probabilities sum to at most 1 (so they are normalised over the whole
/// vocabulary), each bound with [`F32_SLACK`].
fn check_top(top: &[(u32, f32)], n_vocab: u32) -> Result<(), &'static str> {
    if top.len() != TOP_K.min(n_vocab as usize) {
        return Err("not 64 entries");
    }
    if top.iter().any(|&(id, _)| id >= n_vocab) {
        return Err("id outside the vocabulary");
    }
    if !top
        .iter()
        .all(|&(_, lp)| lp.is_finite() && f64::from(lp) <= F32_SLACK)
    {
        return Err("log-probability not finite and <= 0");
    }
    if top.windows(2).any(|w| w[0].1 < w[1].1) {
        return Err("not in descending order");
    }
    let mut ids: Vec<u32> = top.iter().map(|e| e.0).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err("repeated id");
    }
    if top.iter().map(|&(_, lp)| f64::from(lp).exp()).sum::<f64>() > 1.0 + F32_SLACK {
        return Err("probabilities sum over 1");
    }
    Ok(())
}

/// `[[id, logprob], ...]`; `None` if it is not that shape (values are checked by [`check_top`]).
fn parse_top(v: &Value) -> Option<Top> {
    v.as_array()?
        .iter()
        .map(|e| match e.as_array()?.as_slice() {
            [id, lp] => Some((u32::try_from(id.as_u64()?).ok()?, lp.as_f64()? as f32)),
            _ => None,
        })
        .collect()
}

fn parse_header(line: &str) -> Result<Header, OracleError> {
    let raw: Value = serde_json::from_str(line).map_err(|_| OracleError::BadLine(1))?;
    let s = |k: &'static str| match raw[k].as_str() {
        Some(v) if !v.is_empty() => Ok(v.to_owned()),
        _ => Err(OracleError::BadHeader(k)),
    };
    let sha = |k| s(k).and_then(|v| is_sha256(&v).then_some(v).ok_or(OracleError::BadHeader(k)));
    if raw["larch_oracle"].as_u64() != Some(FORMAT) {
        return Err(OracleError::BadHeader("larch_oracle"));
    }
    if raw["top_k"].as_u64() != Some(TOP_K as u64) {
        return Err(OracleError::BadHeader("top_k"));
    }
    let n_vocab = raw["n_vocab"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or(OracleError::BadHeader("n_vocab"))?;
    let prompts = raw["prompts"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(OracleError::BadHeader("prompts"))?;
    Ok(Header {
        llama_cpp_tag: s("llama_cpp_tag")?,
        llama_cpp_commit: s("llama_cpp_commit")?,
        model_sha256: sha("model_sha256")?,
        prompts_sha256: sha("prompts_sha256")?,
        backend: s("backend")?,
        n_vocab,
        prompts,
        raw,
    })
}

/// Prompt line `index` (0-based; 1-based `line` of the file).
fn parse_prompt(line: &str, n: usize, index: usize, n_vocab: u32) -> Result<Prompt, OracleError> {
    let bad = OracleError::BadLine(n);
    let v: Value = serde_json::from_str(line).map_err(|_| bad.clone())?;
    if v["index"].as_u64() != Some(index as u64) {
        return Err(bad);
    }
    let text = v["prompt"].as_str().ok_or(bad.clone())?.to_owned();
    let images = match v.get("images") {
        None => Vec::new(),
        Some(a) => a
            .as_array()
            .ok_or(bad.clone())?
            .iter()
            .map(|i| {
                let u32_at = |v: &Value| v.as_u64().and_then(|v| u32::try_from(v).ok());
                let grid = match i.get("grid") {
                    None => None,
                    Some(g) => match g.as_array()?.as_slice() {
                        [x, y] => Some((u32_at(x)?, u32_at(y)?)),
                        _ => return None,
                    },
                };
                let n_pos = match i.get("n_pos") {
                    None => None,
                    Some(v) => Some(u32_at(v)?),
                };
                let n = usize::try_from(i["n"].as_u64()?).ok()?;
                // A grid holds the image's rows.
                if grid.is_some_and(|(x, y)| x as usize * y as usize != n)
                    || grid.is_some() != n_pos.is_some()
                {
                    return None;
                }
                Some(ImageRows {
                    at: usize::try_from(i["at"].as_u64()?).ok()?,
                    n,
                    embd: i["embd"].as_str()?.to_owned(),
                    file: i["file"].as_str()?.to_owned(),
                    grid,
                    n_pos,
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(bad.clone())?,
    };
    // -1 marks an image's row, and only there.
    let tokens = v["tokens"]
        .as_array()
        .ok_or(bad.clone())?
        .iter()
        .enumerate()
        .map(|(r, t)| match t.as_i64() {
            Some(-1) if images.iter().any(|i| (i.at..i.at + i.n).contains(&r)) => Some(IMAGE_ROW),
            _ => t.as_u64().and_then(|t| u32::try_from(t).ok()),
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(bad.clone())?;
    let top = v["top64"]
        .as_array()
        .ok_or(bad.clone())?
        .iter()
        .map(parse_top)
        .collect::<Option<Vec<_>>>()
        .ok_or(bad)?;
    let p = Prompt {
        text,
        tokens,
        images,
        top,
    };
    if let Some(position) = p
        .tokens
        .iter()
        .position(|&t| t >= n_vocab && t != IMAGE_ROW)
    {
        return Err(OracleError::BadValue {
            prompt: index,
            position,
            what: "token outside the vocabulary",
        });
    }
    check_rows(index, p.tokens.len(), &p.top, n_vocab)?;
    Ok(p)
}

/// One prompt's rows: one valid top 64 per token.
fn check_rows(prompt: usize, tokens: usize, top: &[Top], n_vocab: u32) -> Result<(), OracleError> {
    if top.len() != tokens {
        return Err(OracleError::Positions {
            prompt,
            tokens,
            positions: top.len(),
        });
    }
    for (position, t) in top.iter().enumerate() {
        check_top(t, n_vocab).map_err(|what| OracleError::BadValue {
            prompt,
            position,
            what,
        })?;
    }
    Ok(())
}

impl Dump {
    /// Reads a dump of the GGUF whose sha256 is `model_sha256` (hex, either case), refusing anything
    /// malformed: a bad header or line, another model, a prompt missing or out of order, a position
    /// missing, or an id or log-probability that is not a valid top 64 (see the module doc).
    pub fn read(r: impl BufRead, model_sha256: &str) -> Result<Dump, OracleError> {
        let mut lines = r.lines();
        let mut next = || {
            lines
                .next()
                .transpose()
                .map_err(|e| OracleError::Io(e.kind()))
        };
        let header = parse_header(&next()?.ok_or(OracleError::BadLine(1))?)?;
        if header.model_sha256 != model_sha256.to_ascii_lowercase() {
            return Err(OracleError::WrongModel {
                expected: model_sha256.to_owned(),
                found: header.model_sha256,
            });
        }
        let mut prompts = Vec::new();
        while let Some(line) = next()? {
            let index = prompts.len();
            prompts.push(parse_prompt(&line, index + 2, index, header.n_vocab)?);
        }
        if prompts.len() != header.prompts {
            return Err(OracleError::PromptCount {
                expected: header.prompts,
                found: prompts.len(),
            });
        }
        Ok(Dump { header, prompts })
    }

    /// larch's tokens for each prompt's text against the dump's, which are llama.cpp's (`tokcheck`).
    pub fn tokcheck(&self, tok: &Tokenizer) -> Result<usize, TokcheckError> {
        for (i, p) in self.prompts.iter().enumerate() {
            tokcheck::check(tok, i, &p.text, &p.tokens)?;
        }
        match self.prompts.len() {
            0 => Err(TokcheckError::Empty),
            n => Ok(n),
        }
    }
}

/// Top-1 agreement and KL of a run against a reference, over a set of positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    /// Positions compared.
    pub positions: usize,
    /// Positions whose top-1 is the reference's (or tied with it in the reference).
    pub top1: usize,
    /// Positions whose reference top-1 log-probability leads its second by more than the margin.
    pub decisive: usize,
    /// Of those, the positions counted in `top1`.
    pub decisive_top1: usize,
    /// The largest reference top-1/top-2 gap at a position not counted in `top1`; 0 if none.
    pub flip_gap: f64,
    /// 0 for no positions, as are the two below.
    pub mean_kl: f64,
    /// Nearest rank: the ⌈0.99 n⌉-th smallest.
    pub p99_kl: f64,
    /// The largest position's KL.
    pub max_kl: f64,
}

/// `n / of`; 0 for none.
fn rate(n: usize, of: usize) -> f64 {
    if of == 0 { 0.0 } else { n as f64 / of as f64 }
}

impl Stats {
    /// From each position's (KL, top-1 agreement, reference top-1/top-2 gap).
    fn new(rows: &[(f64, bool, f64)], margin: f64) -> Stats {
        let mut s: Vec<f64> = rows.iter().map(|r| r.0).collect();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        let rank = (n * 99).div_ceil(100);
        let decisive = || rows.iter().filter(|r| r.2 > margin);
        Stats {
            positions: n,
            top1: rows.iter().filter(|r| r.1).count(),
            decisive: decisive().count(),
            decisive_top1: decisive().filter(|r| r.1).count(),
            flip_gap: rows
                .iter()
                .filter(|r| !r.1)
                .map(|r| r.2)
                .fold(0.0, f64::max),
            mean_kl: if n == 0 {
                0.0
            } else {
                s.iter().sum::<f64>() / n as f64
            },
            p99_kl: s.get(rank.saturating_sub(1)).copied().unwrap_or(0.0),
            max_kl: s.last().copied().unwrap_or(0.0),
        }
    }

    /// `top1 / positions`; 0 for no positions.
    pub fn top1_rate(&self) -> f64 {
        rate(self.top1, self.positions)
    }

    /// `decisive_top1 / decisive`: what the gate reads; 0 for no decisive positions.
    pub fn decisive_top1_rate(&self) -> f64 {
        rate(self.decisive_top1, self.decisive)
    }
}

/// A run against a dump: the gate's numbers over all positions, and per prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Over every position of every prompt: what the gate reads.
    pub all: Stats,
    /// Diagnostics only: the gate reads `all`.
    pub prompts: Vec<Stats>,
}

impl Report {
    /// The gate of one backend and mode: top-1 ≥ 99 % of the decisive positions and mean KL ≤ `bound`
    /// over all (no decisive positions fails).
    pub fn passes(&self, bound: f64) -> bool {
        self.all.decisive_top1_rate() >= TOP1_MIN && self.all.mean_kl <= bound
    }
}

/// The KL at one position (module doc); `p` and `q` are checked tops.
pub fn kl(p: &[(u32, f32)], q: &[(u32, f32)]) -> f64 {
    let floor = q.last().map_or(f32::NEG_INFINITY, |e| e.1);
    let (mut kl, mut pm, mut qm) = (0.0, 0.0, 0.0);
    for &(id, lp) in p {
        let lq = f64::from(q.iter().find(|e| e.0 == id).map_or(floor, |e| e.1));
        let lp = f64::from(lp);
        kl += lp.exp() * (lp - lq);
        (pm, qm) = (pm + lp.exp(), qm + lq.exp());
    }
    // Each side's rest outside p's top 64; under TAIL_FLOOR it is rounding, not mass. q needs a rest
    // wherever p has one, at least TAIL_FLOOR.
    let rest = |m: f64| if 1.0 - m > TAIL_FLOOR { 1.0 - m } else { 0.0 };
    let p_rest = rest(pm);
    let q_rest = if p_rest > 0.0 {
        (1.0 - qm).max(TAIL_FLOOR)
    } else {
        rest(qm)
    };
    if p_rest > 0.0 {
        kl += p_rest * (p_rest / q_rest).ln();
    }
    // Both sides renormalised to their totals (1 unless an f32 log-sum-exp or the floor moved them),
    // so this is the KL between two distributions over {p's top 64 ids, the rest}.
    let (zp, zq) = (pm + p_rest, qm + q_rest);
    kl / zp + (zq / zp).ln()
}

/// Whether `q`'s top-1 is `p`'s, where a tie at the top of either side counts for any id in it.
fn top1(p: &[(u32, f32)], q: &[(u32, f32)]) -> bool {
    let (best, lp) = p[0];
    q[0].0 == best
        || p.iter().any(|e| e.0 == q[0].0 && e.1 == lp)
        || q.iter().any(|e| e.0 == best && e.1 == q[0].1)
}

/// A run fed the dump's tokens against the dump: `run[p][t]` is its top 64 at prompt p's position t,
/// with the dump's shape; positions whose reference gap exceeds `margin` (nats) are decisive. Both
/// sides' rows are checked as [`Dump::read`] checks them, since a `Dump`'s fields are public.
pub fn compare(
    reference: &Dump,
    run: &[impl AsRef<[Top]>],
    margin: f64,
) -> Result<Report, OracleError> {
    if run.len() != reference.prompts.len() {
        return Err(OracleError::PromptCount {
            expected: reference.prompts.len(),
            found: run.len(),
        });
    }
    let n_vocab = reference.header.n_vocab;
    let gap = |p: &[(u32, f32)]| {
        p.get(1)
            .map_or(f64::INFINITY, |e| f64::from(p[0].1) - f64::from(e.1))
    };
    let (mut all, mut prompts) = (Vec::new(), Vec::new());
    for (i, (p, q)) in reference.prompts.iter().zip(run).enumerate() {
        let q = q.as_ref();
        check_rows(i, p.tokens.len(), &p.top, n_vocab)?;
        check_rows(i, p.tokens.len(), q, n_vocab)?;
        let rows: Vec<_> = p
            .top
            .iter()
            .zip(q)
            .map(|(p, q)| (kl(p, q), top1(p, q), gap(p)))
            .collect();
        prompts.push(Stats::new(&rows, margin));
        all.extend(rows);
    }
    Ok(Report {
        all: Stats::new(&all, margin),
        prompts,
    })
}

/// llama.cpp against itself across runs on one model and prompt set.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// Each other run's backend and stats against the reference, at margin 0.
    pub pairs: Vec<(String, Stats)>,
    /// [`BOUND_FACTOR`] × the largest pair's p99.
    pub bound: f64,
    /// The largest pair's [`Stats::flip_gap`].
    pub margin: f64,
}

/// The KL bound and tie margin from llama.cpp's dumps of one model: `reference` (the dump larch's runs
/// are then compared against) paired with each of `others`, other runs of it (another backend, or the
/// same at another micro-batch). All must share the model, prompt file, llama.cpp pin, vocabulary and
/// tokens, each other dump's header must differ from the reference's, and the set must be the
/// protocol's: [`CALIBRATION_PROMPTS`] prompts, [`LONG_PROMPTS`] of them over [`LONG_TOKENS`] tokens. A
/// worst p99 ≤ 0 (the runs agree to rounding) is refused rather than turned into a bound nothing
/// passes.
pub fn calibrate(reference: &Dump, others: &[Dump]) -> Result<Calibration, OracleError> {
    let r = &reference.header;
    let long = reference
        .prompts
        .iter()
        .filter(|p| p.tokens.len() > LONG_TOKENS)
        .count();
    if reference.prompts.len() != CALIBRATION_PROMPTS || long < LONG_PROMPTS {
        return Err(OracleError::CalibrationSet {
            prompts: reference.prompts.len(),
            long,
        });
    }
    if others.is_empty() {
        return Err(OracleError::Mismatch("no other run"));
    }
    let mut pairs = Vec::new();
    for o in others {
        let h = &o.header;
        let same = [
            ("model_sha256", h.model_sha256 == r.model_sha256),
            ("prompts_sha256", h.prompts_sha256 == r.prompts_sha256),
            ("llama_cpp_tag", h.llama_cpp_tag == r.llama_cpp_tag),
            ("llama_cpp_commit", h.llama_cpp_commit == r.llama_cpp_commit),
            ("n_vocab", h.n_vocab == r.n_vocab),
            (
                "tokens",
                o.prompts.len() == reference.prompts.len()
                    && o.prompts
                        .iter()
                        .zip(&reference.prompts)
                        .all(|(a, b)| a.tokens == b.tokens),
            ),
            ("run", h.raw != r.raw),
        ];
        if let Some(&(field, _)) = same.iter().find(|s| !s.1) {
            return Err(OracleError::Mismatch(field));
        }
        let run: Vec<&[Top]> = o.prompts.iter().map(|p| &p.top[..]).collect();
        pairs.push((h.backend.clone(), compare(reference, &run, 0.0)?.all));
    }
    let worst = pairs.iter().map(|p| p.1.p99_kl).fold(f64::MIN, f64::max);
    if worst <= 0.0 {
        return Err(OracleError::NoBound(worst));
    }
    Ok(Calibration {
        margin: pairs.iter().map(|p| p.1.flip_gap).fold(0.0, f64::max),
        bound: BOUND_FACTOR * worst,
        pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hasher that cannot be started for a reason other than absence is that hasher's failure, kept
    /// unless a later one hashes; one that is absent is skipped. A directory cannot be spawned on any
    /// OS (access denied), and the real hashers follow it where one is due to succeed.
    #[test]
    fn a_hasher_that_cannot_start_is_its_failure() {
        let dir = std::env::temp_dir().join(format!("larch-digest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("file");
        std::fs::write(&file, b"abc").unwrap();
        let (locked, absent) = (dir.to_str().unwrap(), dir.join("absent"));
        let absent = (absent.to_str().unwrap(), &[][..]);
        let e = digest(&file, &[absent, (locked, &[]), absent]).unwrap_err();
        assert_ne!(e.kind(), std::io::ErrorKind::NotFound, "{e}");
        assert!(e.to_string().contains(locked), "{e}");
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let hashers = [&[(locked, &[][..])][..], &HASHERS[..]].concat();
        assert_eq!(digest(&file, &hashers).unwrap(), abc);
        let e = digest(&file, &[absent]).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
