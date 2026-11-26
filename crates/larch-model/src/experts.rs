//! The expert cache's routing histogram (plan v2 E4): how often each MoE layer routed to each of its
//! experts, kept beside the model file so the next load places the hot experts on the device.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::gguf::{self, Gguf};
use crate::{ModelConfig, MoeCfg};

/// The file's first bytes: "VVEH" and format 1.
const MAGIC: [u8; 8] = *b"VVEH\x01\0\0\0";
/// The magic, the model file's length, the layer and expert counts.
const HEADER: usize = 8 + 8 + 4 + 4;

/// Per (layer, expert) routing counts of one model file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    /// The model file's length: a histogram written for another file is foreign.
    pub file_len: u64,
    /// The model's MoE layers.
    pub layers: u32,
    /// The routed experts of each layer.
    pub experts: u32,
    /// `[layers * experts]`, layer-major: how many routed slots picked each expert.
    pub counts: Vec<u64>,
}

impl Histogram {
    /// No counts yet: every expert weighs the same.
    pub fn uniform(file_len: u64, layers: u32, experts: u32) -> Histogram {
        Histogram {
            file_len,
            layers,
            experts,
            counts: vec![0; layers as usize * experts as usize],
        }
    }

    /// What a load plans from: the histogram at `path` or, without a usable one, uniform. `None`
    /// when `g`'s expert tensors, in every layer, are not the loader's shape for `m`, or together
    /// hold more bytes than the file: nothing else bounds `expert_count`, which sizes this and the
    /// plan, and each tensor is bounded by the file but may share its bytes with another, while a
    /// real file holds every (layer, expert) pair's bytes apart. The load behind the plan then
    /// refuses the file.
    pub fn for_load(
        g: &Gguf,
        c: &ModelConfig,
        m: MoeCfg,
        path: &Path,
        file_len: u64,
    ) -> Option<Histogram> {
        let (layers, experts) = (c.n_layers, m.experts);
        let (e, ef, d) = (i64::from(experts), i64::from(m.ffn), i64::from(c.d_model));
        let shapes = [
            ("gate", [e, ef, d]),
            ("up", [e, ef, d]),
            ("down", [e, d, ef]),
        ];
        let mut held = 0u64;
        let holds = (0..layers).all(|i| {
            shapes.iter().all(|(p, s)| {
                g.tensor(&format!("blk.{i}.ffn_{p}_exps.weight"))
                    .filter(|t| t.shape() == s && gguf::dtype(t.ggml_type).is_ok())
                    .is_some_and(|t| {
                        held = held.saturating_add(t.bytes);
                        held <= file_len
                    })
            })
        });
        holds.then(|| {
            Histogram::read(path, file_len, layers, experts)
                .unwrap_or_else(|| Histogram::uniform(file_len, layers, experts))
        })
    }

    /// Where the histogram of the model file at `model` lives: beside it, as `<file>.experts`.
    pub fn path(model: &Path) -> PathBuf {
        let mut p = model.as_os_str().to_owned();
        p.push(".experts");
        PathBuf::from(p)
    }

    /// The histogram at `path` when it is whole and this model's (`file_len`, `layers`,
    /// `experts`); `None` when it is missing, corrupt or another model's, and the caller plans
    /// uniform. The file is untrusted: its length must be exactly what the model's shape gives
    /// before anything is read, so a crafted header never sizes an allocation.
    pub fn read(path: &Path, file_len: u64, layers: u32, experts: u32) -> Option<Histogram> {
        let n = (layers as usize).checked_mul(experts as usize)?;
        let want = n.checked_mul(8)?.checked_add(HEADER)?;
        if fs::metadata(path).ok()?.len() != want as u64 {
            return None;
        }
        Histogram::parse(&fs::read(path).ok()?, file_len, layers, experts)
    }

    /// [`Histogram::read`] of bytes already in memory: a histogram the binary carries.
    pub fn parse(b: &[u8], file_len: u64, layers: u32, experts: u32) -> Option<Histogram> {
        let n = (layers as usize).checked_mul(experts as usize)?;
        let want = n.checked_mul(8)?.checked_add(HEADER)?;
        if b.len() != want {
            return None;
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
        if b[..8] != MAGIC || u64_at(8) != file_len || u32_at(16) != layers || u32_at(20) != experts
        {
            return None;
        }
        Some(Histogram {
            file_len,
            layers,
            experts,
            counts: (0..n).map(|i| u64_at(HEADER + 8 * i)).collect(),
        })
    }

    /// Writes it to `path` through a synced temporary file beside it, so a crash leaves the old one
    /// whole.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut b = Vec::with_capacity(HEADER + 8 * self.counts.len());
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&self.file_len.to_le_bytes());
        b.extend_from_slice(&self.layers.to_le_bytes());
        b.extend_from_slice(&self.experts.to_le_bytes());
        for c in &self.counts {
            b.extend_from_slice(&c.to_le_bytes());
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".{}.tmp", std::process::id()));
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&b)?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::Histogram;

    #[test]
    fn a_histogram_round_trips_and_a_bad_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("larch-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = Histogram::path(&dir.join("m.gguf"));
        assert!(p.ends_with("m.gguf.experts"));
        assert_eq!(Histogram::read(&p, 9, 2, 3), None, "missing");
        let h = Histogram {
            counts: vec![5, 0, 1, 2, 3, u64::MAX],
            ..Histogram::uniform(9, 2, 3)
        };
        h.write(&p).unwrap();
        assert_eq!(Histogram::read(&p, 9, 2, 3), Some(h.clone()));
        // Another model's: its length, or its shape.
        assert_eq!(Histogram::read(&p, 10, 2, 3), None);
        assert_eq!(Histogram::read(&p, 9, 3, 2), None);
        assert_eq!(Histogram::read(&p, 9, 1, 3), None);
        assert_eq!(Histogram::read(&p, 9, u32::MAX, u32::MAX), None);
        // Truncated, padded, or a flipped magic.
        let good = std::fs::read(&p).unwrap();
        for bad in [
            good[..good.len() - 1].to_vec(),
            [&good[..], &[0]].concat(),
            [&b"X"[..], &good[1..]].concat(),
        ] {
            std::fs::write(&p, bad).unwrap();
            assert_eq!(Histogram::read(&p, 9, 2, 3), None);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
