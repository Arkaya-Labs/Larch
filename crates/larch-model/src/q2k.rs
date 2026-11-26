//! Q2_K quantization (GGML type 10) without an importance matrix: `larch repack`'s cold experts.
//!
//! A block of 256 weights is 16 runs of 16, each `d sc q - dmin m` with q a 2-bit code and sc, m
//! 4-bit integers; d and dmin are the block's f16 scales. Bytes 0-15 hold sc | m << 4 per run,
//! 16-79 the codes (run `8n + 2u + l / 16`'s weight l in bits 2u of byte `16 + 32n + l % 32`, n the
//! half of 128), 80 d and 82 dmin.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use larch_kernels::{Buffer, CPU, DType, Device, KError, Stream, Tensor, ops};
use larch_records::LoadError;

use crate::Histogram;
use crate::gguf::{self, Gguf};
use crate::transformer::f16_bits;
use crate::vision::f16_value;

/// Bytes of a Q2_K block of 256 weights.
pub const BLOCK_BYTES: usize = 84;

/// Step divisors the fit of a run tries around its min/max range's 3: wider steps clip nothing,
/// narrower ones clip the extremes for finer levels in between.
const TRIALS: [f32; 9] = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.35, 3.5];

/// `x` as Q2_K blocks; its length must be a whole number of 256-weight blocks.
pub fn quantize(x: &[f32]) -> Vec<u8> {
    assert!(x.len().is_multiple_of(256), "a partial Q2_K block");
    x.chunks_exact(256).flat_map(block).collect()
}

/// Why [`repack`] stopped; it leaves no output file behind.
#[derive(Debug)]
pub enum RepackError {
    /// Reading the source or writing the output.
    Io(io::Error),
    /// The source is not a GGUF the reader takes.
    Load(LoadError),
    /// What was asked of the file is not in it, or cannot be Q2_K.
    Plan(String),
    /// Decoding the source experts on the CPU reference.
    Kernel(KError),
    /// The progress callback asked it to stop.
    Stopped,
}

impl std::fmt::Display for RepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepackError::Io(e) => write!(f, "{e}"),
            RepackError::Load(e) => write!(f, "{e}"),
            RepackError::Plan(e) => f.write_str(e),
            RepackError::Kernel(e) => write!(f, "CPU decode: {e:?}"),
            RepackError::Stopped => f.write_str("stopped"),
        }
    }
}

impl std::error::Error for RepackError {}

impl From<io::Error> for RepackError {
    fn from(e: io::Error) -> Self {
        RepackError::Io(e)
    }
}
impl From<LoadError> for RepackError {
    fn from(e: LoadError) -> Self {
        RepackError::Load(e)
    }
}
impl From<KError> for RepackError {
    fn from(e: KError) -> Self {
        RepackError::Kernel(e)
    }
}

/// GGML's Q2_K and BF16 type ids.
const GGML_Q2_K: u32 = 10;
const GGML_BF16: u32 = 30;

/// The metadata key listing a mixed layer's cold experts (u32, ascending), whose rows are its
/// `ffn_{gate,up,down}_cold_exps` tensors; its `ffn_*_exps` tensors hold the other experts, in id order.
pub fn cold_key(layer: u32) -> String {
    format!("larch.cold_experts.{layer}")
}

/// A mixed layer's cold experts' tensor of projection `p` (gate, up or down).
pub fn cold_name(layer: u32, p: &str) -> String {
    format!("blk.{layer}.ffn_{p}_cold_exps.weight")
}

/// Writes `src` to `dst` with the routed experts (gate, up and down) of `layers` as Q2_K, quantized
/// from their CPU-reference decode; the metadata and every other tensor are copied byte for byte, in
/// a standard GGUF v3 that llama.cpp reads. Written through a synced temporary file beside `dst`
/// (`<dst>.<pid>.tmp`, removed on any error; an interrupted run leaves it, to delete by hand), then
/// published only where `dst` does not exist: a repack never replaces a file, its source included.
pub fn repack(src: &Path, dst: &Path, layers: &[u32]) -> Result<(), RepackError> {
    write(
        src,
        dst,
        layers,
        &|_, _, e| Some((0..e).collect()),
        &mut |_, _| true,
    )
}

/// [`repack`] of `fraction` of each of `layers`' experts, those `hist` counts least (the lower id
/// first on a tie): they move to [`cold_name`] tensors as Q2_K, the others stay in the layer's
/// tensors at their own type, and [`cold_key`] lists them. A file with such a layer is larch's
/// alone: llama.cpp reads neither the cold tensors nor the hot ones' shape.
pub fn repack_cold(
    src: &Path,
    dst: &Path,
    layers: &[u32],
    fraction: f64,
    hist: &Histogram,
) -> Result<(), RepackError> {
    repack_cold_progress(src, dst, layers, fraction, hist, &mut |_, _| true)
}

/// [`repack_cold`] that reports `progress(written, total)` output bytes after each tensor; a
/// `false` from it stops the repack ([`RepackError::Stopped`]), with no output left behind.
pub fn repack_cold_progress(
    src: &Path,
    dst: &Path,
    layers: &[u32],
    fraction: f64,
    hist: &Histogram,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), RepackError> {
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(RepackError::Plan(format!(
            "cold fraction {fraction} not in (0, 1)"
        )));
    }
    let pick = |layer: u32, len: u64, experts: u64| -> Option<Vec<u64>> {
        if layer >= hist.layers || experts != u64::from(hist.experts) {
            return None; // past the histogram's layers, or another expert count
        }
        let n = (fraction * experts as f64) as usize;
        let counts = &hist.counts[layer as usize * experts as usize..][..experts as usize];
        let mut by: Vec<u64> = (0..experts).collect();
        by.sort_by_key(|&e| (counts[e as usize], e));
        let mut cold = by[..n].to_vec();
        cold.sort_unstable();
        (n > 0 && hist.file_len == len).then_some(cold)
    };
    write(src, dst, layers, &pick, progress)
}

/// The planner of [`repack`] and [`repack_cold`]: `cold(layer, src length, experts)` names the
/// experts of a layer that go to Q2_K, all of them for a whole layer (which keeps its tensors' names
/// and shapes), else the ones that move to the cold tensors; `None` refuses the layer.
fn write(
    src: &Path,
    dst: &Path,
    layers: &[u32],
    cold: &dyn Fn(u32, u64, u64) -> Option<Vec<u64>>,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), RepackError> {
    // The one check that also catches `dst` naming `src`: a repack is lossy, so it must never replace a model.
    if std::fs::symlink_metadata(dst).is_ok() {
        return Err(RepackError::Plan(format!(
            "{}: exists; repack writes a new file and never replaces one",
            dst.display()
        )));
    }
    let mut f = BufReader::new(File::open(src)?);
    let len = f.get_ref().metadata()?.len();
    let g = Gguf::read(&mut f, len)?;
    if g.get("general.alignment")
        .is_some_and(|a| a.as_u64() != Some(32))
    {
        return Err(RepackError::Plan(
            "a general.alignment other than 32".into(),
        ));
    }
    // A mixed layer's tensors are not its experts' whole tensors; an empty array's element type,
    // which gguf::Value drops, would change in the copy.
    if let Some(l) = layers.iter().find(|&&l| g.get(&cold_key(l)).is_some()) {
        return Err(RepackError::Plan(format!("layer {l} is already mixed")));
    }
    if let Some(k) = (g.kv.iter()).find(|(_, v)| v.as_arr().is_some_and(<[_]>::is_empty)) {
        return Err(RepackError::Plan(format!(
            "{}: an empty array, whose type the copy would lose",
            k.0
        )));
    }
    let plan = |t: &gguf::TensorInfo| -> Option<(u32, &'static str)> {
        layers.iter().find_map(|&l| {
            ["gate", "up", "down"]
                .into_iter()
                .find(|p| t.name == format!("blk.{l}.ffn_{p}_exps.weight"))
                .map(|p| (l, p))
        })
    };
    let mut kv: Vec<(String, gguf::Value)> =
        g.kv.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let mut out = Vec::with_capacity(g.tensors.len());
    for t in &g.tensors {
        let elems: u64 = t.dims.iter().product();
        let Some((l, p)) = plan(t) else {
            let bytes = match t.ggml_type {
                ty if gguf::dtype(ty).is_ok() => t.bytes,
                GGML_BF16 => 2 * elems,
                ty => {
                    return Err(RepackError::Plan(format!(
                        "{}: GGML type {ty} unsized",
                        t.name
                    )));
                }
            };
            out.push(Piece {
                src: t,
                name: t.name.clone(),
                dims: t.dims.clone(),
                ty: t.ggml_type,
                bytes,
                from: Source::Copy,
            });
            continue;
        };
        let e = t.dims.get(2).copied().unwrap_or(0);
        if t.dims.len() != 3
            || e == 0
            || !t.dims[0].is_multiple_of(256)
            || gguf::dtype(t.ggml_type).is_err()
        {
            return Err(RepackError::Plan(format!(
                "{}: not Q2_K-able experts",
                t.name
            )));
        }
        let Some(c) = cold(l, len, e) else {
            return Err(RepackError::Plan(format!(
                "layer {l}: no cold experts for this file"
            )));
        };
        let q2 = |n: u64| elems / e * n / 256 * BLOCK_BYTES as u64;
        let mut dims = t.dims.clone();
        if c.len() as u64 == e {
            out.push(Piece {
                src: t,
                name: t.name.clone(),
                dims,
                ty: GGML_Q2_K,
                bytes: q2(e),
                from: Source::Q2k(c),
            });
            continue;
        }
        if p == "gate" {
            kv.push((
                cold_key(l),
                gguf::Value::Arr(c.iter().map(|&x| gguf::Value::U32(x as u32)).collect()),
            ));
        }
        let hot: Vec<u64> = (0..e).filter(|x| !c.contains(x)).collect();
        dims[2] = hot.len() as u64;
        let hot_bytes = t.bytes / e * hot.len() as u64;
        out.push(Piece {
            src: t,
            name: t.name.clone(),
            dims: dims.clone(),
            ty: t.ggml_type,
            bytes: hot_bytes,
            from: Source::Hot(hot),
        });
        dims[2] = c.len() as u64;
        out.push(Piece {
            src: t,
            name: cold_name(l, p),
            dims,
            ty: GGML_Q2_K,
            bytes: q2(c.len() as u64),
            from: Source::Q2k(c),
        });
    }
    if let Some(l) = layers.iter().find(|&&l| {
        ["gate", "up", "down"]
            .iter()
            .any(|p| g.tensor(&format!("blk.{l}.ffn_{p}_exps.weight")).is_none())
    }) {
        return Err(RepackError::Plan(format!(
            "layer {l}: its routed experts are not in the file"
        )));
    }
    kv.sort_by(|a, b| a.0.cmp(&b.0));
    let table: Vec<(&str, Vec<u64>, u32, u64)> = out
        .iter()
        .map(|o| (o.name.as_str(), o.dims.clone(), o.ty, o.bytes))
        .collect();
    let mut tmp = dst.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let total: u64 = out.iter().map(|o| o.bytes).sum();
    let mut emit = || -> Result<(), RepackError> {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&gguf::header(&kv, &table))?;
        let mut at = 0u64;
        for o in &out {
            let start = g.data_offset + o.src.offset;
            let e = o.src.dims.get(2).copied().unwrap_or(1);
            let slice = o.src.bytes / e;
            let mut read = |ids: &[u64]| -> io::Result<Vec<u8>> {
                let mut raw = vec![0u8; (slice * ids.len() as u64) as usize];
                for (&x, r) in ids.iter().zip(raw.chunks_mut(slice as usize)) {
                    f.seek(SeekFrom::Start(start + x * slice))?;
                    f.read_exact(r)?;
                }
                Ok(raw)
            };
            match &o.from {
                Source::Copy => {
                    f.seek(SeekFrom::Start(start))?;
                    if io::copy(&mut (&mut f).take(o.bytes), &mut w)? != o.bytes {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                }
                Source::Hot(ids) => w.write_all(&read(ids)?)?,
                Source::Q2k(ids) => {
                    let raw = read(ids)?;
                    let dtype = gguf::dtype(o.src.ggml_type)?;
                    let k = o.src.dims[0] as usize;
                    let rows = raw.len() / dtype.bytes(k as u64) as usize;
                    w.write_all(&to_q2_k(&raw, dtype, rows, k)?)?;
                }
            }
            at += o.bytes;
            let pad = at.next_multiple_of(32) - at;
            w.write_all(&vec![0u8; pad as usize])?;
            at += pad;
            // `at` counts the alignment padding as well, which `total` leaves out.
            if !progress(at.min(total), total) {
                return Err(RepackError::Stopped);
            }
        }
        w.into_inner()
            .map_err(io::IntoInnerError::into_error)?
            .sync_all()?;
        // A hard link publishes without replacing: it fails if `dst` appeared since the check above.
        std::fs::hard_link(&tmp, dst)?;
        sync_dir(dst)?;
        Ok(())
    };
    let r = emit();
    let _ = std::fs::remove_file(&tmp); // published through its link, or failed: either way not needed
    r
}

/// Makes `path`'s new directory entry durable (a no-op where directories cannot be synced).
pub fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Where an output tensor's bytes come from, in its source tensor `src`.
enum Source {
    /// All of them.
    Copy,
    /// These experts' slices, in order.
    Hot(Vec<u64>),
    /// These experts' rows, quantized to Q2_K.
    Q2k(Vec<u64>),
}

/// One tensor of the output: its table entry and its bytes' source.
struct Piece<'g> {
    src: &'g gguf::TensorInfo,
    name: String,
    dims: Vec<u64>,
    ty: u32,
    bytes: u64,
    from: Source,
}

/// `raw`, `rows` rows of `k` weights of `dtype`, as Q2_K: rows decoded to f32 on the CPU reference
/// (embed) and quantized, a share of the rows on each of up to 8 threads.
fn to_q2_k(raw: &[u8], dtype: DType, rows: usize, k: usize) -> Result<Vec<u8>, RepackError> {
    const CHUNK: usize = 64;
    let (rb, ob) = (dtype.bytes(k as u64) as usize, k / 256 * BLOCK_BYTES);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let per = rows.div_ceil(threads).max(1);
    let mut out = vec![0u8; rows * ob];
    let work = |src: &[u8], dst: &mut [u8]| -> Result<(), RepackError> {
        let dev = Device::open(CPU)?;
        let s = Stream::new(&dev)?;
        let w = Buffer::alloc(&dev, (CHUNK * rb) as u64)?;
        let ids = Buffer::alloc(&dev, 4 * CHUNK as u64)?;
        let y = Buffer::alloc(&dev, (4 * CHUNK * k) as u64)?;
        s.upload(&ids, 0, &(0..CHUNK as i32).collect::<Vec<_>>())?;
        let mut f = vec![0f32; CHUNK * k];
        for (rs, ds) in src.chunks(CHUNK * rb).zip(dst.chunks_mut(CHUNK * ob)) {
            let n = (rs.len() / rb) as i64;
            s.upload(&w, 0, rs)?;
            let table = Tensor::new(&w, 0, dtype, &[n, k as i64])?;
            let idt = Tensor::new(&ids, 0, DType::I32, &[n])?;
            let o = Tensor::new(&y, 0, DType::F32, &[n, k as i64])?;
            ops::embed(&s, &table, &idt, &o)?;
            let f = &mut f[..n as usize * k];
            s.download(&y, 0, f)?;
            s.sync()?;
            // quantize has no code for them: a NaN would zero its run, an inf make the block's scale infinite.
            if let Some(i) = f.iter().position(|v| !v.is_finite()) {
                return Err(RepackError::Plan(format!(
                    "a source expert weight is {} (row {} of a chunk): the file is corrupt",
                    f[i],
                    i / k
                )));
            }
            ds.copy_from_slice(&quantize(f));
        }
        Ok(())
    };
    std::thread::scope(|sc| {
        let jobs: Vec<_> = raw
            .chunks(per * rb)
            .zip(out.chunks_mut(per * ob))
            .map(|(src, dst)| sc.spawn(move || work(src, dst)))
            .collect();
        jobs.into_iter()
            .try_for_each(|j| j.join().expect("a repack worker panicked"))
    })?;
    Ok(out)
}

/// The code of `v` at scale `dl` and min `ml`, as the decoder reads it back (`dl q - ml`).
fn code(v: f32, dl: f32, ml: f32) -> u8 {
    if dl > 0.0 {
        ((v + ml) / dl).round().clamp(0.0, 3.0) as u8
    } else {
        0
    }
}

/// The squared error of `x` coded at `dl` and `ml`, each weight's value computed as the decoder
/// computes it.
fn err(x: &[f32], dl: f32, ml: f32) -> f32 {
    x.iter()
        .map(|&v| (dl * f32::from(code(v, dl, ml)) - ml - v).powi(2))
        .sum()
}

/// The step s >= 0 and min m >= 0 that fit the run `x` best as `s q - m`: for each trial step, the
/// codes it gives and then the least-squares (s, m) of those codes, twice.
fn fit(x: &[f32]) -> (f32, f32) {
    let lo = x.iter().fold(0f32, |a, &v| a.min(v));
    let hi = x.iter().fold(lo, |a, &v| a.max(v));
    if hi == lo {
        return (0.0, -lo);
    }
    let mut best = (f32::INFINITY, (hi - lo) / 3.0, -lo);
    for t in TRIALS {
        let (mut s, mut m) = ((hi - lo) / t, -lo);
        for _ in 0..2 {
            let (mut n, mut sq, mut sqq, mut sx, mut sqx) = (0f32, 0f32, 0f32, 0f32, 0f32);
            for &v in x {
                let q = f32::from(code(v, s, m));
                (n, sq, sqq, sx, sqx) = (n + 1.0, sq + q, sqq + q * q, sx + v, sqx + q * v);
            }
            let den = n * sqq - sq * sq;
            if den <= 0.0 {
                break; // one code for all: the step is free, keep the last
            }
            let ls = (n * sqx - sq * sx) / den;
            let b = (sx - ls * sq) / n;
            // The format's min is never positive: refit the step alone at 0.
            (s, m) = if b > 0.0 { (sqx / sqq, 0.0) } else { (ls, -b) };
            if s.is_nan() || s <= 0.0 {
                break;
            }
            let e = err(x, s, m);
            if e < best.0 {
                best = (e, s, m);
            }
        }
    }
    (best.1, best.2)
}

/// One block of 256 weights: each run's fit, d and dmin a 15th of the largest step and min, and
/// each run's 4-bit sc and m whichever of their two nearest integers code it best.
fn block(x: &[f32]) -> [u8; BLOCK_BYTES] {
    let fits: Vec<(f32, f32)> = x.chunks_exact(16).map(fit).collect();
    let (d, dmin) = (
        f16_bits(fits.iter().fold(0f32, |a, f| a.max(f.0)) / 15.0),
        f16_bits(fits.iter().fold(0f32, |a, f| a.max(f.1)) / 15.0),
    );
    let (dv, mv) = (f16_value(d), f16_value(dmin));
    let near = |v: f32, unit: f32| -> [u8; 2] {
        let r = if unit > 0.0 { v / unit } else { 0.0 };
        [r.floor(), r.ceil()].map(|i| i.clamp(0.0, 15.0) as u8)
    };
    let mut out = [0u8; BLOCK_BYTES];
    for (j, (run, &(s, m))) in x.chunks_exact(16).zip(&fits).enumerate() {
        let mut best = (f32::INFINITY, 0u8, 0u8);
        for sc in near(s, dv) {
            for mm in near(m, mv) {
                let e = err(run, dv * f32::from(sc), mv * f32::from(mm));
                if e < best.0 {
                    best = (e, sc, mm);
                }
            }
        }
        let (dl, ml) = (dv * f32::from(best.1), mv * f32::from(best.2));
        out[j] = best.1 | best.2 << 4;
        // Run j = 8n + 2u + h holds weights 128n + 32u + 16h + [0, 16), in bits 2u of qs[32n + 16h..].
        let (n, u, h) = (j / 8, j % 8 / 2, j % 2);
        for (l, &v) in run.iter().enumerate() {
            out[16 + 32 * n + 16 * h + l] |= code(v, dl, ml) << (2 * u);
        }
    }
    out[80..82].copy_from_slice(&d.to_le_bytes());
    out[82..84].copy_from_slice(&dmin.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use larch_kernels::{Buffer, DType, Device, Stream, Tensor, ops};

    /// `x`'s Q2_K blocks decoded by the CPU reference (embed, f32), as llama.cpp decodes them.
    fn decode(q: &[u8], rows: usize, k: usize) -> Vec<f32> {
        let dev = Device::open(-1).unwrap();
        let s = Stream::new(&dev).unwrap();
        let (w, ids, out) = (
            Buffer::alloc(&dev, q.len() as u64).unwrap(),
            Buffer::alloc(&dev, 4 * rows as u64).unwrap(),
            Buffer::alloc(&dev, 4 * (rows * k) as u64).unwrap(),
        );
        s.upload(&w, 0, q).unwrap();
        s.upload(&ids, 0, &(0..rows as i32).collect::<Vec<_>>())
            .unwrap();
        let (r, c) = (rows as i64, k as i64);
        let table = Tensor::new(&w, 0, DType::Q2K, &[r, c]).unwrap();
        let idt = Tensor::new(&ids, 0, DType::I32, &[r]).unwrap();
        let o = Tensor::new(&out, 0, DType::F32, &[r, c]).unwrap();
        ops::embed(&s, &table, &idt, &o).unwrap();
        let mut y = vec![0f32; rows * k];
        s.download(&out, 0, &mut y).unwrap();
        s.sync().unwrap();
        y
    }

    /// A uniform in (0, 1) from an LCG.
    fn uniform(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// The baseline a fit must beat: each 16 weights' min (at most 0) and max as its range, its
    /// scale and min then rounded to 4 bits of the block's largest, as Q2_K stores them.
    fn naive(x: &[f32]) -> Vec<f32> {
        let f16 = |v: f32| crate::vision::f16_value(crate::transformer::f16_bits(v));
        let mut y = Vec::with_capacity(x.len());
        for b in x.chunks_exact(256) {
            let fits: Vec<(f32, f32)> = b
                .chunks_exact(16)
                .map(|s| {
                    let lo = s.iter().fold(0f32, |a, &v| a.min(v));
                    let hi = s.iter().fold(lo, |a, &v| a.max(v));
                    ((hi - lo) / 3.0, -lo)
                })
                .collect();
            let d = f16(fits.iter().fold(0f32, |a, f| a.max(f.0)) / 15.0);
            let dmin = f16(fits.iter().fold(0f32, |a, f| a.max(f.1)) / 15.0);
            for (s, &(sc, m)) in b.chunks_exact(16).zip(&fits) {
                let dl = d * if d > 0.0 { (sc / d).round() } else { 0.0 };
                let ml = dmin * if dmin > 0.0 { (m / dmin).round() } else { 0.0 };
                for &v in s {
                    let q = if dl > 0.0 {
                        ((v + ml) / dl).round().clamp(0.0, 3.0)
                    } else {
                        0.0
                    };
                    y.push(dl * q - ml);
                }
            }
        }
        y
    }

    /// The fit's RMS error against the min/max baseline's at most: the step search and least squares
    /// take off more than the choice of rounding alone (0.91 and 0.89 of it on these rows).
    const BEATS_NAIVE: f64 = 0.85;

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        let e: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        e / a.len() as f64
    }

    fn rms(a: &[f32]) -> f64 {
        (a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// Gaussian weights: 4 levels of an affine fit per 16 leave about a third of the spread (the
    /// best 2-bit quantizer of a Gaussian has an RMS error of 0.34 sigma), and the fit beats the
    /// min/max one by [`BEATS_NAIVE`].
    #[test]
    fn q2_k_quantize_stays_within_its_error_on_random_rows() {
        let (rows, k) = (8, 2048);
        let mut seed = 1u64;
        let x: Vec<f32> = (0..rows * k)
            .map(|_| {
                let (u, v) = (uniform(&mut seed), uniform(&mut seed));
                0.02 * (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
            })
            .collect();
        let q = super::quantize(&x);
        assert_eq!(q.len(), rows * k / 256 * super::BLOCK_BYTES);
        let y = decode(&q, rows, k);
        let rel = mse(&x, &y).sqrt() / rms(&x);
        assert!(rel < 0.36, "relative RMS error {rel}");
        assert!(mse(&x, &y) < BEATS_NAIVE * BEATS_NAIVE * mse(&x, &naive(&x)));
    }

    /// Real-shaped weights: rows of an expert's gate/up width with Laplace values, a scale per row
    /// that spans 16x, and an outlier in every 256 (one block's largest scale).
    #[test]
    fn q2_k_quantize_stays_within_its_error_on_real_shaped_rows() {
        let (rows, k) = (16, 2048);
        let mut seed = 2u64;
        let mut x = Vec::with_capacity(rows * k);
        for r in 0..rows {
            let scale = 0.004 * 2f32.powf(r as f32 / 4.0);
            for i in 0..k {
                let u = uniform(&mut seed) - 0.5;
                let lap = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
                x.push(if i % 256 == 77 { 12.0 * scale } else { lap });
            }
        }
        let y = decode(&super::quantize(&x), rows, k);
        let rel = mse(&x, &y).sqrt() / rms(&x);
        assert!(rel < 0.45, "relative RMS error {rel}");
        assert!(mse(&x, &y) < BEATS_NAIVE * BEATS_NAIVE * mse(&x, &naive(&x)));
    }

    /// Exact cases: zeros stay 0, a constant run of 16 keeps its value, and weights of one sign
    /// leave the other side's range unused.
    #[test]
    fn q2_k_quantize_keeps_zero_and_constant_runs() {
        let zero = vec![0f32; 256];
        assert_eq!(decode(&super::quantize(&zero), 1, 256), zero);
        let mut x = vec![-0.5f32; 256];
        x[16..32].fill(0.25);
        let y = decode(&super::quantize(&x), 1, 256);
        assert!(
            y[..16].iter().all(|&v| (v + 0.5).abs() < 0.5 / 30.0),
            "{:?}",
            &y[..16]
        );
        assert!(
            y[16..32].iter().all(|&v| (v - 0.25).abs() < 0.5 / 30.0),
            "{:?}",
            &y[16..32]
        );
        let pos: Vec<f32> = (0..256).map(|i| (i % 16) as f32 / 15.0).collect();
        let y = decode(&super::quantize(&pos), 1, 256);
        assert!(y.iter().all(|&v| v >= 0.0), "{y:?}");
    }
}
