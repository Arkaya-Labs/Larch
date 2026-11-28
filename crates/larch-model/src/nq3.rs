//! Q3_K matrices rewritten at load as [`larch_kernels::DType::Nq3`] (larch.h `LARCH_NQ3`) for the
//! Metal kernels: Q3_K's values bit for bit and the same bytes, moved where those kernels decode
//! them with a few masks. A load converts a matrix chunk by chunk into its device buffer, so the
//! memory it holds afterwards is what the zero-copy map held.

/// The GGML type id of Q3_K.
const GGML_Q3_K: u32 = 11;
/// Bytes of a Q3_K block of 256 weights.
const Q3_K_BYTES: usize = 110;

/// Whether a tensor of `ggml_type` with these GGML dims (row length first) is rewritten: Q3_K
/// whose rows hold an even number of blocks, 2 or more dims (a matrix or experts).
pub fn converts(ggml_type: u32, dims: &[u64]) -> bool {
    ggml_type == GGML_Q3_K && dims.len() >= 2 && dims[0].is_multiple_of(512)
}

/// Q3_K rows (`q3k`, each a run of 110-byte blocks, an even number of them) rewritten in place
/// order as nq3: the same bytes, a pair of blocks at a time. Weight `i` of a block is its
/// sub-block `c = i / 32` (`n = c / 4`, bit pair `u = c % 4`, `l = i % 32`): low bits
/// `qs[32 n + l] >> 2 u`, high bit `hmask[l]` bit `4 n + u`, code `q + 4` (0..7). In the pair,
/// input `p` (0..511) is code `4 f + e` of item `j` of unit `p / 128`, `f = p % 128 / 16`,
/// `j = p % 16 / 4`, `e = p % 4`, at bytes `48 unit + 12 j` as pair `P = 2 f + e / 2`: bits
/// `3 (P % 5)` (`e` even) or `16 + 3 (P % 5)` (odd) of word `P / 5` for `P < 15`, else bit `b`
/// of the code at bit 15 (`e` even) or 31 (odd) of word `b`. Then the two blocks' 12 scale bytes
/// and their f16 d, as they are.
pub fn from_q3_k(q3k: &[u8]) -> Vec<u8> {
    const PAIR: usize = 2 * Q3_K_BYTES;
    assert!(
        q3k.len().is_multiple_of(PAIR),
        "nq3 takes pairs of Q3_K blocks"
    );
    let mut out = vec![0u8; q3k.len()];
    // A load converts gigabytes: split across the cores, a run of whole pairs each.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = q3k.len().div_ceil(PAIR).div_ceil(threads).max(1) * PAIR;
    std::thread::scope(|sc| {
        for (src, dst) in q3k.chunks(per).zip(out.chunks_mut(per)) {
            sc.spawn(move || {
                for (src, dst) in src.chunks_exact(PAIR).zip(dst.chunks_exact_mut(PAIR)) {
                    pair(src, dst);
                }
            });
        }
    });
    out
}

/// One pair of Q3_K blocks (`src`, 220 bytes) as nq3 (`dst`).
fn pair(src: &[u8], dst: &mut [u8]) {
    // The pair's 512 codes in input order.
    let mut codes = [0u8; 512];
    for (b, q) in src.chunks_exact(Q3_K_BYTES).enumerate() {
        dst[192 + 12 * b..][..12].copy_from_slice(&q[96..108]);
        dst[216 + 2 * b..][..2].copy_from_slice(&q[108..110]);
        let (hm, qs) = (&q[..32], &q[32..96]);
        for n in 0..2 {
            for u in 0..4 {
                let c = &mut codes[256 * b + 128 * n + 32 * u..][..32];
                for l in 0..32 {
                    c[l] = (qs[32 * n + l] >> (2 * u)) & 3 | ((hm[l] >> (4 * n + u)) & 1) << 2;
                }
            }
        }
    }
    for unit in 0..4 {
        for j in 0..4 {
            let mut w = [0u32; 3];
            for f in 0..8 {
                for e in 0..4 {
                    let code = u32::from(codes[128 * unit + 16 * f + 4 * j + e]);
                    let (p, hi) = (2 * f + e / 2, 16 * (e % 2));
                    if p < 15 {
                        w[p / 5] |= code << (3 * (p % 5) + hi);
                    } else {
                        for (bit, word) in w.iter_mut().enumerate() {
                            *word |= (code >> bit & 1) << (15 + hi);
                        }
                    }
                }
            }
            for (k, word) in w.iter().enumerate() {
                dst[48 * unit + 12 * j + 4 * k..][..4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use larch_kernels::{Buffer, DType, Device, Stream, Tensor, ops};

    /// Rows of `dtype` decoded by the CPU reference (embed, f32).
    fn decode(q: &[u8], dtype: DType, rows: usize, k: usize) -> Vec<f32> {
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
        let table = Tensor::new(&w, 0, dtype, &[r, c]).unwrap();
        let idt = Tensor::new(&ids, 0, DType::I32, &[r]).unwrap();
        let o = Tensor::new(&out, 0, DType::F32, &[r, c]).unwrap();
        ops::embed(&s, &table, &idt, &o).unwrap();
        let mut y = vec![0f32; rows * k];
        s.download(&out, 0, &mut y).unwrap();
        s.sync().unwrap();
        y
    }

    /// Random Q3_K blocks with finite f16 d: every code, high bit and scale pattern occurs.
    fn q3_k(rows: usize, k: usize) -> Vec<u8> {
        let mut seed = 5u64;
        let mut q: Vec<u8> = (0..rows * k / 256 * 110)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 56) as u8
            })
            .collect();
        for b in q.chunks_exact_mut(110) {
            b[109] &= 0x3b; // d's exponent below 15: finite
        }
        q
    }

    #[test]
    fn nq3_decodes_to_q3_ks_values_bit_for_bit() {
        let (rows, k) = (5, 2048);
        let q = q3_k(rows, k);
        let n = super::from_q3_k(&q);
        assert_eq!(n.len(), q.len());
        let (a, b) = (
            decode(&q, DType::Q3K, rows, k),
            decode(&n, DType::Nq3, rows, k),
        );
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b));
    }

    #[test]
    fn converts_only_q3_k_matrices_of_whole_pairs() {
        assert!(super::converts(11, &[2048, 512]));
        assert!(super::converts(11, &[512, 2048, 256]));
        assert!(!super::converts(11, &[256, 64])); // one block a row
        assert!(!super::converts(11, &[2048])); // a vector
        assert!(!super::converts(12, &[2048, 512])); // Q4_K
    }
}
