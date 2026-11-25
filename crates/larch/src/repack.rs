//! `larch repack -m <gguf> -o <out> --layers <list>`: the routed experts (gate, up and down) of the
//! listed layers rewritten to Q2_K ([`larch_model::q2k::repack`]), everything else copied byte for
//! byte, as a standard GGUF that llama.cpp loads, so `larch parity` can use a llama.cpp dump of the
//! same file. With `--cold-fraction F --histogram <file>`, each layer (all, without `--layers`)
//! keeps its hot experts and moves the fraction its routing histogram counts least to Q2_K beside
//! them ([`larch_model::q2k::repack_cold`]): a file only larch loads.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use larch_model::gguf::Gguf;
use larch_model::{Histogram, ModelConfig, MoeCfg};

/// `larch repack` options.
#[derive(Args, Debug)]
pub struct Options {
    /// The source GGUF.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Where the repacked GGUF goes.
    #[arg(short = 'o', long)]
    pub out: PathBuf,
    /// Layers whose routed experts become Q2_K: comma-separated numbers and ranges (`28-39,4`).
    /// Required without `--cold-fraction`; with it, every layer when left out.
    // The path, not `Vec`: clap takes a bare `Vec` field as repeated values, not one parsed list.
    #[arg(long, value_parser = layers, required_unless_present = "cold_fraction")]
    pub layers: Option<std::vec::Vec<u32>>,
    /// The fraction of each layer's experts, the coldest by `--histogram`, that become Q2_K beside
    /// the hot ones.
    #[arg(long, requires = "histogram")]
    pub cold_fraction: Option<f64>,
    /// The model's routing histogram (`<gguf>.experts`, written by a recording run).
    #[arg(long, requires = "cold_fraction")]
    pub histogram: Option<PathBuf>,
}

/// A list of layer numbers and inclusive ranges, sorted, each once.
fn layers(s: &str) -> Result<Vec<u32>> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let (a, b): (u32, u32) = (a.trim().parse()?, b.trim().parse()?);
        if a > b || b - a >= 1 << 16 {
            bail!("{part}: a range runs upward, over fewer than 65536 layers");
        }
        v.extend(a..=b);
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

/// Repacks and reports what was written.
pub fn run(o: &Options) -> Result<()> {
    let what = || format!("repack {}", o.model.display());
    let (Some(f), Some(h)) = (o.cold_fraction, &o.histogram) else {
        let layers = o.layers.as_deref().unwrap_or_default();
        larch_model::q2k::repack(&o.model, &o.out, layers).with_context(what)?;
        println!(
            "repack: {} layers' experts as Q2_K in {}",
            layers.len(),
            o.out.display()
        );
        return Ok(());
    };
    let file = File::open(&o.model).with_context(what)?;
    let len = file.metadata()?.len();
    let g = Gguf::read(BufReader::new(file), len).with_context(what)?;
    let c = ModelConfig::from_gguf(&g)?;
    let Some(m) = MoeCfg::from_gguf(&g, c.arch)? else {
        bail!("{}: no routed experts", o.model.display());
    };
    let Some(hist) = Histogram::read(h, len, c.n_layers, m.experts) else {
        bail!("{}: not this model's routing histogram", h.display());
    };
    let all: Vec<u32> = (0..c.n_layers).collect();
    let layers = o.layers.as_deref().unwrap_or(&all);
    larch_model::q2k::repack_cold(&o.model, &o.out, layers, f, &hist).with_context(what)?;
    println!(
        "repack: the coldest {f} of {} layers' experts as Q2_K beside the hot ones in {}",
        layers.len(),
        o.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        o: super::Options,
    }

    #[test]
    fn options_parse() {
        let c = Cli::try_parse_from([
            "repack", "-m", "a.gguf", "-o", "b.gguf", "--layers", "1-2,5",
        ])
        .unwrap();
        assert_eq!(c.o.layers.as_deref(), Some(&[1, 2, 5][..]));
        assert!(Cli::try_parse_from(["repack", "-m", "a", "-o", "b", "--layers", "x"]).is_err());
        let c = Cli::try_parse_from([
            "repack",
            "-m",
            "a",
            "-o",
            "b",
            "--cold-fraction",
            "0.2",
            "--histogram",
            "h",
        ])
        .unwrap();
        assert_eq!((c.o.layers, c.o.cold_fraction), (None, Some(0.2)));
        // Neither a layer list nor a fraction, or a fraction without its histogram.
        assert!(Cli::try_parse_from(["repack", "-m", "a", "-o", "b"]).is_err());
        assert!(
            Cli::try_parse_from(["repack", "-m", "a", "-o", "b", "--cold-fraction", "0.2"])
                .is_err()
        );
    }

    #[test]
    fn layer_lists_take_numbers_and_ranges() {
        assert_eq!(super::layers("28-30,4,29").unwrap(), [4, 28, 29, 30]);
        assert_eq!(super::layers("7").unwrap(), [7]);
        for bad in ["", "3-1", "a", "1-", "-1", "1,,2", "0-4294967295"] {
            assert!(super::layers(bad).is_err(), "{bad}");
        }
    }
}
