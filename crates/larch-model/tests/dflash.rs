//! The DFlash drafter on the CPU reference against the NumPy reference of z-lab's model_mlx.py
//! (`tests/dflash/gen.py`, which writes both fixtures): three rounds on one sequence, with an
//! observe past one op's 16 rows, a short block and a cache cut back.

use std::path::Path;

use larch_kernels::{Buffer, CPU, DType, Device, Stream, Tensor};
use larch_model::gguf::{Gguf, Value};
use larch_model::{DflashDrafter, tiny};
use larch_records::LoadError;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

#[test]
fn drafter_matches_the_reference() {
    let (drafter, ..) = rounds("dflash", 64);
    assert_eq!(
        (drafter.target_layers(), drafter.block()),
        (&[1, 3, 5][..], 16)
    );
}

/// DFlash2 against the NumPy reference of Splash's draft (gen.py): its convs, a block whose rows
/// see each other, the ring of its window wrapped and cut back, and the selector's guesses wherever
/// the reference's best candidate leads by more than the logits' tolerance.
#[test]
fn dflash2_matches_the_reference() {
    let (drafter, cache) = rounds("dflash2", 1024);
    assert_eq!(drafter.block(), 8);
    // A ring of the 20-token window, not the 1024 positions asked for.
    assert!(cache.len() > drafter.shape().cache_positions(1024));
}

/// Runs fixture `name`'s rounds on a cache for `positions`, checking each against the reference.
fn rounds(name: &str, positions: u32) -> (DflashDrafter, larch_model::DraftCache) {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let path = Path::new(DATA).join(format!("{name}-tiny.gguf"));
    let drafter = DflashDrafter::load(&path, &dev, &s).unwrap();

    let bytes = std::fs::read(Path::new(DATA).join(format!("{name}-case.gguf"))).unwrap();
    let g = Gguf::parse(&bytes).unwrap();
    let int = |k: &str| g.get(k).and_then(Value::as_u64).unwrap() as u32;
    // Each case tensor in a buffer of its own.
    let bufs: Vec<(String, Buffer, DType, Vec<i64>)> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let b = Buffer::alloc(&dev, t.bytes).unwrap();
            s.upload(&b, 0, &bytes[at..at + t.bytes as usize]).unwrap();
            let dt = larch_model::gguf::dtype(t.ggml_type).unwrap();
            (t.name.clone(), b, dt, t.shape())
        })
        .collect();
    let tensor = |name: &str| {
        let (_, b, dt, shape) = bufs.iter().find(|e| e.0 == name).unwrap();
        Tensor::new(b, 0, *dt, shape).unwrap()
    };
    let f32s = |name: &str| -> Option<Vec<f32>> {
        let t = g.tensor(name)?;
        let at = (g.data_offset + t.offset) as usize;
        Some(
            bytes[at..at + t.bytes as usize]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    };
    let (embed, head) = (tensor("token_embd.weight"), tensor("output.weight"));
    let vocab = head.shape()[0] as usize;

    let mut cache = drafter.cache(&dev, &s, positions).unwrap();
    for i in 0..int("case.rounds") {
        let key = |k: &str| int(&format!("case.{i}.{k}"));
        cache.truncate(key("truncate"));
        drafter
            .observe(
                &s,
                &mut cache,
                &tensor(&format!("taps.{i}")),
                key("first")..key("end"),
            )
            .unwrap();
        let n = key("guesses");
        let (ids, conf) = drafter
            .propose(&s, &cache, &embed, &head, key("anchor"), &[], n)
            .unwrap();
        // A chance for each guess, of its own candidate's softmax.
        assert_eq!(conf.len(), ids.len(), "{name} round {i}");
        assert!(
            conf.iter().all(|&p| p > 0.0 && p <= 1.0),
            "{name} round {i}: {conf:?}"
        );
        let got = drafter.last_logits(&s, n).unwrap();
        let want = f32s(&format!("logits.{i}")).unwrap();
        // f16 activations against f64: 0.5 % of the logits' spread, and the guess wherever the
        // reference's top two are further apart than that.
        let spread = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tol = 0.005 * spread;
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= tol,
            "{name} round {i}: logits off by {worst}, spread {spread}"
        );
        match (f32s(&format!("guesses.{i}")), f32s(&format!("margins.{i}"))) {
            // The selector: each guess follows the one before, so only up to the first near tie.
            (Some(guesses), Some(margins)) => {
                let sure = margins.iter().take_while(|&&m| m > 2.0 * tol).count();
                let want: Vec<u32> = guesses[..sure].iter().map(|&g| g as u32).collect();
                assert_eq!(ids[..sure], want, "{name} round {i}");
                assert_eq!(ids.len(), n as usize);
            }
            _ => {
                for (r, row) in want.chunks(vocab).enumerate() {
                    let mut order: Vec<usize> = (0..vocab).collect();
                    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    if row[order[0]] - row[order[1]] > 2.0 * tol {
                        assert_eq!(ids[r], order[0] as u32, "round {i} guess {r}");
                    }
                }
            }
        }
        println!("{name} round {i}: worst {worst:.4} of spread {spread:.2}, guesses {ids:?}");
    }
    (drafter, cache)
}

/// A mask token past the vocabulary is refused at load, whatever its width: `propose` feeds it to
/// `embed`, which does not check ids on the device.
#[test]
fn a_mask_token_past_the_vocabulary_is_refused() {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let path = std::env::temp_dir().join(format!("larch-mask-{}.gguf", std::process::id()));
    let load = |mask| {
        std::fs::write(&path, tiny::dflash_gguf_with(7, "mask_token_id", mask)).unwrap();
        DflashDrafter::load(&path, &dev, &s)
    };
    assert!(load(Value::U32(tiny::VOCAB - 1)).is_ok());
    for bad in [
        Value::U32(tiny::VOCAB),
        Value::U64(1 << 63),
        Value::U64(u64::MAX),
    ] {
        assert!(matches!(load(bad), Err(LoadError::BadGguf(..))));
    }
    std::fs::remove_file(&path).unwrap();
}

/// Window lists the drafter cannot run are refused at load: mixed windows, a list of the wrong
/// length or type, and a window whose table would not fit a cache.
#[test]
fn unsupported_sliding_windows_are_refused() {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let path = std::env::temp_dir().join(format!("larch-window-{}.gguf", std::process::id()));
    let load = |windows| {
        std::fs::write(
            &path,
            tiny::dflash_gguf_with(7, "attention.sliding_window", windows),
        )
        .unwrap();
        DflashDrafter::load(&path, &dev, &s)
    };
    let arr = |w: &[u32]| Value::Arr(w.iter().map(|&w| Value::U32(w)).collect());
    assert!(load(arr(&[24, 0])).is_ok());
    assert!(load(arr(&[0, 0])).is_ok());
    for bad in [
        arr(&[24, 32]),
        arr(&[24]),
        arr(&[24, 0, 0]),
        Value::U32(24),
        arr(&[(1 << 31) - 1, 0]),
    ] {
        assert!(matches!(load(bad), Err(LoadError::BadGguf(..))));
    }
    std::fs::remove_file(&path).unwrap();
}

/// The converter's `--quant q4_k` drafter (gen.py's dflash2-q4k.gguf): its Q4_K matrices, beside
/// the Q8_0 ones whose rows are not whole Q4_K blocks, load and propose on the CPU backend.
#[test]
fn a_q4_k_drafter_proposes() {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let path = Path::new(DATA).join("dflash2-q4k.gguf");
    let g = Gguf::parse(&std::fs::read(&path).unwrap()).unwrap();
    let ty = |n: &str| larch_model::gguf::dtype(g.tensor(n).unwrap().ggml_type).unwrap();
    assert_eq!(
        (ty("blk.0.attn_q.weight"), ty("blk.0.attn_output.weight")),
        (DType::Q4K, DType::Q8_0)
    );
    let drafter = DflashDrafter::load(&path, &dev, &s).unwrap();
    let (vocab, d) = (128, 256);
    // One Q8_0 table as both embedding and head: scales of 2^-7 (f16 0x2000), patterned values.
    let table: Vec<u8> = (0..vocab * d / 32)
        .flat_map(|b| {
            [0, 0x20]
                .into_iter()
                .chain((0..32).map(move |j| (b * 31 + j * 57) as u8))
        })
        .collect();
    let buf = Buffer::alloc(&dev, table.len() as u64).unwrap();
    s.upload(&buf, 0, &table).unwrap();
    let embed = Tensor::new(&buf, 0, DType::Q8_0, &[vocab, d]).unwrap();
    let taps: Vec<f32> = (0..4 * d).map(|i| (i % 7) as f32 - 3.0).collect();
    let tbuf = Buffer::alloc(&dev, 4 * taps.len() as u64).unwrap();
    s.upload(&tbuf, 0, &taps).unwrap();
    let taps = Tensor::new(&tbuf, 0, DType::F32, &[4, d]).unwrap();

    let mut cache = drafter.cache(&dev, &s, 64).unwrap();
    drafter.observe(&s, &mut cache, &taps, 0..4).unwrap();
    let (ids, conf) = drafter
        .propose(&s, &cache, &embed, &embed, 5, &[], 7)
        .unwrap();
    assert_eq!((ids.len(), conf.len()), (7, 7));
    assert!(conf.iter().all(|&p| p > 0.0 && p <= 1.0), "{conf:?}");
    assert!(ids.iter().all(|&t| t < vocab as u32));
    let logits = drafter.last_logits(&s, 7).unwrap();
    assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|&v| v != 0.0));
}
