//! The KL bound and tie margin calibration (architecture 15) over `llama-logits` dumps of one GGUF:
//!   cargo run --release -p larch-model --example oracle-calibrate -- <model sha256> <reference.jsonl>
//!       <other.jsonl>...
//! The reference is the dump larch is gated against, the others llama.cpp's other runs of it (other
//! backends, or the same at another micro-batch). Dumps are plain JSON lines in regular files
//! (`zstd -dc x.jsonl.zst > x.jsonl`), not pipes: each is hashed, then read. Prints each pair's
//! numbers, the bound and the margin to stderr and the proposed `bench/oracle-bounds.json` entry to
//! stdout, whose `dump_sha256`s are digests of the decompressed JSON lines, not of the `.zst`.

use std::fs::File;
use std::io::BufReader;

use larch_model::oracle::{self, Dump};
use serde_json::json;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [sha, reference, others @ ..] = &args[..] else {
        return Err(
            "usage: oracle-calibrate <model sha256> <reference.jsonl> <other.jsonl>...".into(),
        );
    };
    // Which runs these were: a saved entry names its reference's dump, not only its backend. The
    // hasher reads a pipe to its end, which would leave nothing for the read below.
    let digest = |path: &String| {
        let meta = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
        if !meta.is_file() {
            return Err(format!(
                "{path}: not a regular file; decompress to a file first (zstd -dc x.jsonl.zst > \
                 x.jsonl)"
            ));
        }
        oracle::sha256(path.as_ref()).map_err(|e| format!("{path}: {e}"))
    };
    let reference_sha = digest(reference)?;
    let others_sha = others.iter().map(digest).collect::<Result<Vec<_>, _>>()?;
    let read = |path: &String| {
        let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        Dump::read(BufReader::new(f), sha).map_err(|e| format!("{path}: {e}"))
    };
    let reference = read(reference)?;
    let others = others.iter().map(read).collect::<Result<Vec<_>, _>>()?;
    // LARCH_ORACLE_TEXT_ONLY=1: a vision dump's text positions and each image's last row
    // (oracle::text_positions), by the same rules: 1.5 x the worst pair's p99, the largest flip gap.
    // LARCH_ORACLE_ALL_ROWS=1: every position of a vision dump by those rules, without the text
    // oracle's long-prompt requirement (Qwen 3.5's image prompts are all under 1100 tokens).
    let text = std::env::var("LARCH_ORACLE_TEXT_ONLY").is_ok_and(|v| v == "1");
    let all_rows = std::env::var("LARCH_ORACLE_ALL_ROWS").is_ok_and(|v| v == "1");
    if text || all_rows {
        let pick = |d: &Dump| {
            if text {
                oracle::text_only(d)
            } else {
                d.clone()
            }
        };
        let what = if text {
            "text positions"
        } else {
            "all positions"
        };
        let r = pick(&reference);
        let (mut bound, mut margin, mut pairs) = (0.0f64, 0.0f64, Vec::new());
        for (o, dump_sha) in others.iter().zip(&others_sha) {
            let o = pick(o);
            let tops: Vec<_> = o.prompts.iter().map(|p| p.top.clone()).collect();
            let s = oracle::compare(&r, &tops, 0.0)
                .map_err(|e| e.to_string())?
                .all;
            eprintln!(
                "{} vs {} ({what}): {} positions, top-1 {:.4}, flip gap {:.4}, KL mean {:.3e} \
                 p99 {:.3e} max {:.3e}",
                o.header.backend,
                r.header.backend,
                s.positions,
                s.top1_rate(),
                s.flip_gap,
                s.mean_kl,
                s.p99_kl,
                s.max_kl
            );
            bound = bound.max(oracle::BOUND_FACTOR * s.p99_kl);
            margin = margin.max(s.flip_gap);
            pairs.push(json!({"backend": o.header.backend, "dump_sha256": dump_sha,
                "positions": s.positions, "top1": s.top1_rate(), "flip_gap": s.flip_gap,
                "mean_kl": s.mean_kl, "p99_kl": s.p99_kl, "max_kl": s.max_kl}));
        }
        eprintln!("bound {bound:.6e}, margin {margin:.6} nats");
        let positions = if text {
            "text rows and each image's last row (oracle::text_positions)"
        } else {
            "every row, image rows included"
        };
        let entry = json!({"positions": positions,
            "reference_dump_sha256": reference_sha, "pairs": pairs,
            "bound_rule": "1.5 x max over pairs of the p99 (nearest rank) over those positions",
            "bound": bound, "margin": margin});
        println!("{entry:#}");
        return Ok(());
    }
    let c = oracle::calibrate(&reference, &others).map_err(|e| e.to_string())?;
    let r = &reference.header;
    let mut pairs = Vec::new();
    for (((backend, s), o), dump_sha) in c.pairs.iter().zip(&others).zip(others_sha) {
        eprintln!(
            "{backend} vs {}: {} positions, top-1 {:.4}, flip gap {:.4}, KL mean {:.3e} p99 {:.3e} \
             max {:.3e}",
            r.backend,
            s.positions,
            s.top1_rate(),
            s.flip_gap,
            s.mean_kl,
            s.p99_kl,
            s.max_kl
        );
        let h = &o.header.raw;
        pairs.push(json!({
            "backend": backend, "device": h["device"], "n_batch": h["n_batch"],
            "n_ubatch": h["n_ubatch"], "dump_sha256": dump_sha, "positions": s.positions,
            "top1": s.top1_rate(), "flip_gap": s.flip_gap, "mean_kl": s.mean_kl, "p99_kl": s.p99_kl,
            "max_kl": s.max_kl,
        }));
    }
    eprintln!("bound {:.6e}, margin {:.6} nats", c.bound, c.margin);
    let entry = json!({
        "model_sha256": r.model_sha256,
        "prompts_sha256": r.prompts_sha256,
        "prompts": reference.prompts.len(),
        "llama_cpp": format!("{} ({})", r.llama_cpp_tag, r.llama_cpp_commit),
        "reference_backend": r.backend,
        "reference": {
            "backend": r.backend, "device": r.raw["device"], "n_batch": r.raw["n_batch"],
            "n_ubatch": r.raw["n_ubatch"], "dump_sha256": reference_sha,
        },
        "pairs": pairs,
        "kl": "per position, KL(p_ref || q) over the reference top 64 plus one bucket for the rest; an \
               id outside q's top 64 takes q's 64th logprob (a lower bound on that term)",
        "bound_rule": "1.5 x max over pairs of the p99 (nearest rank) over positions",
        "bound": c.bound,
        "margin_rule": "largest reference top-1/top-2 gap (nats) at which any pair's top-1 differs",
        "margin": c.margin,
        "gate": "per backend and mode: top-1 >= 0.99 on positions whose reference gap exceeds the \
                 margin, and mean KL over all positions <= bound",
    });
    println!("{entry:#}");
    Ok(())
}
