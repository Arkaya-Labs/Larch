//! The faster copy of a model that a GPU holds only in part (`larch faster`, and the desktop's
//! "Faster on this computer"): the experts its routing histogram counts least are stored at 2 bits
//! beside the others ([`larch_model::q2k::repack_cold_progress`]), so more of the model fits on the
//! GPU. Only a file this binary carries a histogram for is offered: the histogram picks the experts
//! that lose precision, and the quality cost was measured with that histogram on that file.
//!
//! The copy is made safely: the source is hashed first, free disk space is checked, the copy is
//! written under a temporary name, checked, and renamed into place only once it passes. A failed or
//! stopped build leaves nothing behind, and the source is never changed.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use larch_model::gguf::Gguf;
use larch_model::{Histogram, ModelConfig, MoeCfg, q2k};

/// The share of each layer's experts stored at 2 bits. 5070 Ti, m13, 25 Nov 2026: see
/// bench/results/35b-vs-llamacpp-2026-11-25.md for its speed and its KL against the original.
pub const COLD_FRACTION: f64 = 0.35;

/// A model file the binary carries a routing histogram for.
pub struct Known {
    /// The file's length and LFS SHA-256 (bench/models.toml).
    pub size: u64,
    pub sha256: &'static str,
    /// The `<gguf>.experts` histogram the copy is planned from.
    routes: &'static [u8],
}

/// m13 (bench/models.toml): its histogram was recorded by the expert cache on a 5070 Ti, 20 Nov 2026.
// ponytail: one file; add a row (its histogram under bench/routes) when another model measures
// faster as a copy with its KL in budget.
const KNOWN: &[Known] = &[Known {
    size: 17_117_951_808,
    sha256: "eaef9194110fa86144b003e22f5058a3fb2810be0828c805ff0bec21601d2db8",
    routes: include_bytes!("../../../bench/routes/m13.experts"),
}];

/// The extra free space a build keeps beyond the copy's own size.
const DISK_MARGIN: u64 = 1 << 30;

/// The part of a build `progress` reports: hashing the source, then writing the copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Check,
    Write,
}

/// Why a build stopped. Its text is for people: the desktop shows it as is.
#[derive(Debug)]
pub enum Error {
    /// No histogram for a file of this size.
    NotOffered,
    /// The size matched but the bytes did not: another file, or a damaged download.
    NotThisFile,
    /// The free space at the copy's folder, and what the build needs.
    Disk {
        need: u64,
        have: u64,
    },
    /// Something that is not a faster copy is already where the copy would go.
    Occupied(PathBuf),
    /// The caller stopped it.
    Stopped,
    /// The written copy did not pass its check.
    Check(String),
    Io(io::Error),
    Repack(q2k::RepackError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gb = |b: u64| b as f64 / f64::from(1u32 << 30);
        match self {
            Error::NotOffered => f.write_str("There is no faster version of this model."),
            Error::NotThisFile => f.write_str(
                "This file is not the download the faster version was made for. Download the model again.",
            ),
            Error::Disk { need, have } => write!(
                f,
                "The faster version needs {:.1} GB of free disk space and {:.1} GB is free.",
                gb(*need),
                gb(*have)
            ),
            Error::Occupied(p) => write!(
                f,
                "{} is already there and is not a faster version. Move it, or choose another place.",
                p.display()
            ),
            Error::Stopped => f.write_str("Stopped."),
            Error::Check(e) => write!(f, "The faster version did not pass its check: {e}"),
            Error::Io(e) => write!(f, "The faster version could not be made: {e}"),
            Error::Repack(e) => write!(f, "The faster version could not be made: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// The histogram row for `src`, by its length alone (cheap, for offering); [`build`] checks the
/// bytes.
pub fn known(src: &Path) -> Option<&'static Known> {
    let len = fs::metadata(src).ok()?.len();
    KNOWN.iter().find(|k| k.size == len)
}

/// `dst`'s name while it is written and checked: never one a model list or a load picks up.
fn unverified(dst: &Path) -> PathBuf {
    let mut p = dst.as_os_str().to_owned();
    p.push(".unverified");
    PathBuf::from(p)
}

/// Removes what an interrupted build left at `dst`: the unchecked copy and the repack's own
/// temporary files (`<unverified>.<pid>.tmp`).
pub fn clean(dst: &Path) {
    let part = unverified(dst);
    let _ = fs::remove_file(&part);
    let (Some(dir), Some(name)) = (part.parent(), part.file_name()) else {
        return;
    };
    let prefix = format!("{}.", name.to_string_lossy());
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.starts_with(&prefix) && n.ends_with(".tmp") {
            let _ = fs::remove_file(e.path());
        }
    }
}

/// Makes the faster copy of `src` at `dst`; `progress(stage, done, total)` reports bytes, and
/// `false` from it stops the build. `dst` appears only once the copy passed [`check`]; an existing
/// `dst` that passes is kept as it is, and any other file there is refused, never replaced: it may
/// be the source itself, or anything else a person keeps there.
pub fn build(
    src: &Path,
    dst: &Path,
    progress: &mut dyn FnMut(Stage, u64, u64) -> bool,
) -> Result<(), Error> {
    let k = known(src).ok_or(Error::NotOffered)?;
    let dir = match dst.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    build_with(src, dst, k, fs2::available_space(dir)?, progress)
}

/// [`build`] for `k`'s file, with `have` bytes free at `dst`'s folder.
fn build_with(
    src: &Path,
    dst: &Path,
    k: &Known,
    have: u64,
    progress: &mut dyn FnMut(Stage, u64, u64) -> bool,
) -> Result<(), Error> {
    clean(dst);
    if dst.exists() {
        return check(dst, k).map_err(|_| Error::Occupied(dst.to_owned()));
    }
    // The copy is smaller than its source: the source's size bounds it.
    let need = k.size + DISK_MARGIN;
    if have < need {
        return Err(Error::Disk { need, have });
    }
    let sum = larch_fetch::sha256_file_progress(src, &mut |n| progress(Stage::Check, n, k.size))
        .map_err(|e| match e.kind() {
            io::ErrorKind::Interrupted => Error::Stopped,
            _ => Error::Io(e),
        })?;
    if sum != k.sha256 {
        return Err(Error::NotThisFile);
    }
    let (_, _, c, m) = header(src)?;
    let hist = Histogram::parse(k.routes, k.size, c.n_layers, m.experts)
        .ok_or_else(|| Error::Check("the carried histogram is not this model's".into()))?;
    let part = unverified(dst);
    let layers: Vec<u32> = (0..c.n_layers).collect();
    let mut write = |done, total| progress(Stage::Write, done, total);
    q2k::repack_cold_progress(src, &part, &layers, COLD_FRACTION, &hist, &mut write).map_err(
        |e| match e {
            q2k::RepackError::Stopped => Error::Stopped,
            e => Error::Repack(e),
        },
    )?;
    let published = check(&part, k).and_then(|()| Ok(fs::rename(&part, dst)?));
    if published.is_err() {
        let _ = fs::remove_file(&part);
    }
    published
}

/// `path`'s header, its length and its model's config and MoE shape.
fn header(path: &Path) -> Result<(Gguf, u64, ModelConfig, MoeCfg), Error> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    let bad = |e: &dyn fmt::Display| Error::Check(e.to_string());
    let g = Gguf::read(BufReader::new(file), len).map_err(|e| bad(&e))?;
    let c = ModelConfig::from_gguf(&g).map_err(|e| bad(&e))?;
    let m = MoeCfg::from_gguf(&g, c.arch).map_err(|e| bad(&e))?;
    let m = m.ok_or_else(|| Error::Check("no routed experts".into()))?;
    Ok((g, len, c, m))
}

/// Whether `path` is a whole faster copy of `k`'s file: it reads as a GGUF of the same model, every
/// layer lists the cold experts the histogram picks, and each tensor lies inside the file.
pub fn check(path: &Path, k: &Known) -> Result<(), Error> {
    let bad = |e: String| Err(Error::Check(e));
    let (g, len, c, m) = header(path)?;
    let Some(hist) = Histogram::parse(k.routes, k.size, c.n_layers, m.experts) else {
        return bad("another model's shape".into());
    };
    let n = (COLD_FRACTION * f64::from(m.experts)) as usize;
    for l in 0..c.n_layers {
        let counts = &hist.counts[(l * m.experts) as usize..][..m.experts as usize];
        let mut want: Vec<u64> = (0..u64::from(m.experts)).collect();
        want.sort_by_key(|&e| (counts[e as usize], e));
        let mut want = want[..n].to_vec();
        want.sort_unstable();
        let have: Option<Vec<u64>> = g
            .get(&q2k::cold_key(l))
            .and_then(|v| v.as_arr())
            .and_then(|a| a.iter().map(|x| x.as_u64()).collect());
        if have.as_deref() != Some(&want[..]) {
            return bad(format!("layer {l}: not the expected cold experts"));
        }
        for p in ["gate", "up", "down"] {
            if g.tensor(&q2k::cold_name(l, p)).is_none() {
                return bad(format!("layer {l}: no cold {p} experts"));
            }
        }
    }
    let end = g
        .tensors
        .iter()
        .map(|t| t.offset + t.bytes)
        .max()
        .unwrap_or(0);
    if g.data_offset + end > len {
        return bad("shorter than its tensors".into());
    }
    Ok(())
}

/// `larch faster` options.
#[derive(clap::Args, Debug)]
pub struct Options {
    /// The downloaded model.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Where the faster copy goes; beside the model as `<name>.faster.gguf` by default.
    #[arg(short = 'o', long)]
    pub out: Option<PathBuf>,
}

/// Where `larch faster` writes the copy of `src` by default.
pub fn default_out(src: &Path) -> PathBuf {
    src.with_extension("faster.gguf")
}

/// Builds the faster copy, printing its progress to stderr and its path to stdout.
pub fn run(o: &Options) -> anyhow::Result<()> {
    let out = o.out.clone().unwrap_or_else(|| default_out(&o.model));
    let started = std::time::Instant::now();
    let mut last = (Stage::Check, u64::MAX);
    build(&o.model, &out, &mut |stage, done, total| {
        let pct = done * 100 / total.max(1);
        if (stage, pct) != last {
            last = (stage, pct);
            let what = match stage {
                Stage::Check => "checking the download",
                Stage::Write => "writing the faster copy",
            };
            eprint!(
                "\r{what}: {pct:3} % ({:.0} s)   ",
                started.elapsed().as_secs_f64()
            );
        }
        true
    })?;
    eprintln!();
    println!("{}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use larch_model::{Arch, Histogram, tiny};

    use super::{Error, Known, Stage, build_with, check};

    /// The tiny MoE model in a fresh folder, and a row for it whose histogram counts expert `e` of
    /// every layer `EXPERTS - e` times, so the coldest is the highest id (a uniform one picks 0).
    fn setup(name: &str) -> (PathBuf, PathBuf, Known) {
        let dir = std::env::temp_dir().join(format!("larch-faster-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.gguf");
        let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
        fs::write(&src, &bytes).unwrap();
        let mut h = Histogram::uniform(bytes.len() as u64, tiny::LAYERS, tiny::EXPERTS);
        let e = u64::from(tiny::EXPERTS);
        h.counts = (0..h.counts.len() as u64).map(|i| e - i % e).collect();
        let hp = dir.join("h.experts");
        h.write(&hp).unwrap();
        let k = Known {
            size: bytes.len() as u64,
            sha256: Box::leak(larch_fetch::sha256_bytes(&bytes).into_boxed_str()),
            routes: Box::leak(fs::read(&hp).unwrap().into_boxed_slice()),
        };
        fs::remove_file(&hp).unwrap();
        (src, dir.join("m.faster.gguf"), k)
    }

    /// The folder's files other than the source.
    fn others(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = (fs::read_dir(dir).unwrap().flatten())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "m.gguf")
            .collect();
        v.sort();
        v
    }

    #[test]
    fn a_build_writes_a_checked_copy_and_leaves_the_source() {
        let (src, dst, k) = setup("ok");
        let before = fs::read(&src).unwrap();
        let mut stages = Vec::new();
        build_with(&src, &dst, &k, u64::MAX, &mut |s, d, t| {
            assert!(d <= t);
            stages.push(s);
            true
        })
        .unwrap();
        assert!(stages.contains(&Stage::Check) && stages.contains(&Stage::Write));
        check(&dst, &k).unwrap();
        assert_eq!(fs::read(&src).unwrap(), before);
        assert_eq!(others(dst.parent().unwrap()), ["m.faster.gguf"]);
        // A second build keeps the copy that passes.
        build_with(&src, &dst, &k, 0, &mut |_, _, _| true).unwrap();
        fs::remove_dir_all(dst.parent().unwrap()).unwrap();
    }

    #[test]
    fn another_file_of_the_same_size_is_refused() {
        let (src, dst, mut k) = setup("hash");
        k.sha256 = "00";
        let r = build_with(&src, &dst, &k, u64::MAX, &mut |_, _, _| true);
        assert!(matches!(r, Err(Error::NotThisFile)), "{r:?}");
        assert!(others(dst.parent().unwrap()).is_empty());
        fs::remove_dir_all(dst.parent().unwrap()).unwrap();
    }

    #[test]
    fn too_little_disk_is_refused_before_anything_is_written() {
        let (src, dst, k) = setup("disk");
        let r = build_with(&src, &dst, &k, k.size, &mut |_, _, _| panic!("no work"));
        assert!(
            matches!(r, Err(Error::Disk { have, .. }) if have == k.size),
            "{r:?}"
        );
        assert!(others(dst.parent().unwrap()).is_empty());
        fs::remove_dir_all(dst.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_stopped_build_leaves_nothing() {
        for stop in [Stage::Check, Stage::Write] {
            let (src, dst, k) = setup("stop");
            let r = build_with(&src, &dst, &k, u64::MAX, &mut |s, _, _| s != stop);
            assert!(matches!(r, Err(Error::Stopped)), "{stop:?}: {r:?}");
            assert!(others(dst.parent().unwrap()).is_empty(), "{stop:?}");
            fs::remove_dir_all(dst.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn what_a_crash_left_is_cleaned() {
        let (src, dst, k) = setup("crash");
        let dir = dst.parent().unwrap();
        fs::write(dir.join("m.faster.gguf.unverified"), b"half").unwrap();
        fs::write(dir.join("m.faster.gguf.unverified.4242.tmp"), b"half").unwrap();
        build_with(&src, &dst, &k, u64::MAX, &mut |_, _, _| true).unwrap();
        check(&dst, &k).unwrap();
        assert_eq!(others(dir), ["m.faster.gguf"]);
        fs::remove_dir_all(dir).unwrap();
    }

    /// A file at the copy's place that is not a copy is left alone: another file of the person's,
    /// or the source itself (`larch faster -m m.gguf -o m.gguf`).
    #[test]
    fn a_file_in_the_way_is_refused_and_kept() {
        let (src, dst, k) = setup("occupied");
        fs::write(&dst, b"not a gguf").unwrap();
        let e = build_with(&src, &dst, &k, u64::MAX, &mut |_, _, _| true).unwrap_err();
        assert!(matches!(e, Error::Occupied(_)), "{e}");
        assert_eq!(fs::read(&dst).unwrap(), b"not a gguf");
        let before = fs::read(&src).unwrap();
        let e = build_with(&src, &src, &k, u64::MAX, &mut |_, _, _| true).unwrap_err();
        assert!(matches!(e, Error::Occupied(_)), "{e}");
        assert_eq!(fs::read(&src).unwrap(), before, "the source is untouched");
        fs::remove_dir_all(dst.parent().unwrap()).unwrap();
    }

    /// The carried row is m13's in bench/models.toml, where model identifiers live: the binary
    /// ships without that file, so the two are kept equal here.
    #[test]
    fn the_known_file_is_m13_of_the_bench_models() {
        let toml = include_str!("../../../bench/models.toml");
        let m13 = &toml[toml.find("[m13]").unwrap()..];
        let m13 = &m13[..m13[1..].find("\n[").map_or(m13.len(), |i| i + 1)];
        let k = &super::KNOWN[0];
        assert!(m13.contains(&format!("sha256 = \"{}\"", k.sha256)), "{m13}");
        assert!(m13.contains(&format!("size = {}\n", k.size)), "{m13}");
    }

    #[test]
    fn a_copy_with_other_cold_experts_fails_its_check() {
        let (src, dst, k) = setup("other");
        let other = Histogram::uniform(k.size, tiny::LAYERS, tiny::EXPERTS);
        larch_model::q2k::repack_cold(&src, &dst, &[0, 1], super::COLD_FRACTION, &other).unwrap();
        assert!(matches!(check(&dst, &k), Err(Error::Check(_))));
        fs::remove_dir_all(dst.parent().unwrap()).unwrap();
    }
}
