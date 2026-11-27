//! The safe API over the CPU reference: every op once, copies, and the checks that run before any
//! FFI call. Dtypes follow the op contract (architecture 6.1).

use bytemuck::Pod;
use larch_kernels::DType::{F16, F32, I32, Q8_0};
#[cfg(any(unix, windows))]
use larch_kernels::Mmap;
use larch_kernels::{
    Buffer, CPU, CPU_COMPUTE, DType, Device, Event, Graph, KError, KvPool, SampleParams, Stream,
    Tensor, ops,
};

fn cpu() -> (Device, Stream) {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    (dev, s)
}

fn up<T: Pod>(dev: &Device, s: &Stream, data: &[T]) -> Buffer {
    let b = Buffer::alloc(dev, size_of_val(data) as u64).unwrap();
    s.upload(&b, 0, data).unwrap();
    b
}

fn t<'a>(b: &'a Buffer, dtype: DType, shape: &[i64]) -> Tensor<'a> {
    Tensor::new(b, 0, dtype, shape).unwrap()
}

/// f16 bits of `x`, for zero and the normal values used here.
fn h(x: f32) -> u16 {
    let b = x.to_bits();
    if x == 0.0 {
        return 0;
    }
    ((b >> 16) & 0x8000 | (((b >> 23) & 0xff) - 112) << 10 | (b >> 13) & 0x3ff) as u16
}

/// The first `n` f16 values of `b`, as f32.
fn halves(s: &Stream, b: &Buffer, n: usize) -> Vec<f32> {
    let mut v = vec![0u16; n];
    s.download(b, 0, &mut v).unwrap();
    v.iter()
        .map(|&x| {
            let (exp, man) = (u32::from(x >> 10 & 0x1f), u32::from(x & 0x3ff));
            let mag = match exp {
                0 => man as f32 / (1 << 24) as f32, // zero and subnormals
                31 if man == 0 => f32::INFINITY,
                31 => f32::NAN,
                _ => f32::from_bits((exp + 112) << 23 | man << 13),
            };
            if x & 0x8000 != 0 { -mag } else { mag }
        })
        .collect()
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert!(
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol),
        "{got:?} != {want:?}"
    );
}

/// Each op once with a known answer, so a swapped argument in `ops` fails here.
#[test]
fn every_op_once() {
    let (dev, s) = cpu();
    assert_eq!(dev.props().unwrap().backend, larch_kernels::Backend::Cpu);

    let table = up(&dev, &s, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(h));
    let ids = up(&dev, &s, &[2i32, 0]);
    let out = Buffer::alloc(&dev, 16).unwrap();
    let (tt, it) = (t(&table, F16, &[3, 2]), t(&ids, I32, &[2]));
    ops::embed(&s, &tt, &it, &t(&out, F32, &[2, 2])).unwrap();
    let mut got = [0f32; 4];
    s.download(&out, 0, &mut got).unwrap();
    assert_eq!(got, [5.0, 6.0, 1.0, 2.0]);

    // rms([2, 2, 2, 2]) = 2, so both norms give w.
    let x = up(&dev, &s, &[2.0f32; 4]);
    let w = up(&dev, &s, &[1.0f32, 2.0, 3.0, 4.0]);
    let out = Buffer::alloc(&dev, 8).unwrap();
    let (xt, wt) = (t(&x, F32, &[1, 4]), t(&w, F32, &[4]));
    ops::rmsnorm(&s, &xt, &wt, 0.0, &t(&out, F16, &[1, 4])).unwrap();
    close(&halves(&s, &out, 4), &[1.0, 2.0, 3.0, 4.0], 0.0);
    let qk = up(&dev, &s, &[h(2.0); 4]);
    ops::qk_norm(&s, &t(&qk, F16, &[1, 1, 4]), &wt, 0.0).unwrap();
    close(&halves(&s, &qk, 4), &[1.0, 2.0, 3.0, 4.0], 0.0);

    // One pair at position 1 turns by 1 radian.
    let qk = up(&dev, &s, &[1.0, 0.0].map(h));
    let pos = up(&dev, &s, &[1i32]);
    let p = ops::RopeParams {
        n_dims: 2,
        neox: false,
        n_ctx_orig: 4096,
        freq_base: 10000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        freq_factors: None,
    };
    ops::rope(&s, &t(&qk, F16, &[1, 1, 2]), &t(&pos, I32, &[1]), &p).unwrap();
    close(&halves(&s, &qk, 2), &[1f32.cos(), 1f32.sin()], 1e-3);

    // The same pair under rope_multi: its one section takes position t (1); h, w and e are 0.
    let qk = up(&dev, &s, &[1.0, 0.0].map(h));
    let pos4 = up(&dev, &s, &[1i32, 0, 0, 0]);
    let pos4 = t(&pos4, I32, &[1, 4]);
    ops::rope_multi(
        &s,
        &t(&qk, F16, &[1, 1, 2]),
        &pos4,
        &p,
        [1, 0, 0, 0],
        ops::RopeMode::Imrope,
    )
    .unwrap();
    close(&halves(&s, &qk, 2), &[1f32.cos(), 1f32.sin()], 1e-3);
    // Under VISION n_dims is half the head: the pair turns by its section's position.
    let qk = up(&dev, &s, &[1.0, 0.0].map(h));
    let vision = ops::RopeParams { n_dims: 1, ..p };
    ops::rope_multi(
        &s,
        &t(&qk, F16, &[1, 1, 2]),
        &pos4,
        &vision,
        [1, 0, 0, 0],
        ops::RopeMode::Vision,
    )
    .unwrap();
    close(&halves(&s, &qk, 2), &[1f32.cos(), 1f32.sin()], 1e-3);

    // A Q8_0 row of 32 ones (scale 1, every quant 1).
    let mut q8 = [1u8; 34];
    q8[..2].copy_from_slice(&h(1.0).to_le_bytes());
    let wq = up(&dev, &s, &q8);
    let wqt = t(&wq, Q8_0, &[1, 32]);
    let out = Buffer::alloc(&dev, 64).unwrap();
    ops::dequant(&s, &wqt, &t(&out, F16, &[1, 32])).unwrap();
    close(&halves(&s, &out, 32), &[1.0; 32], 0.0);
    let x = up(&dev, &s, &[h(1.0); 17 * 32]);
    let bias = up(&dev, &s, &[0.5f32]);
    let out = Buffer::alloc(&dev, 34).unwrap();
    let (xt, bt) = (t(&x, F16, &[17, 32]), t(&bias, F32, &[1]));
    let x1 = xt.rows(0, 1).unwrap();
    ops::mmvq(&s, &wqt, &x1, Some(&bt), &t(&out, F16, &[1, 1])).unwrap();
    close(&halves(&s, &out, 1), &[32.5], 0.0);
    // More than 16 rows is dequant + gemm's job.
    let o17 = t(&out, F16, &[17, 1]);
    assert_eq!(
        ops::mmvq(&s, &wqt, &xt, None, &o17).err(),
        Some(KError::InvalidArg)
    );

    // Two softmax-gated experts: e^0 : e^ln3 is 1 : 3, so expert 1 leads with weight 0.75.
    let logits = up(&dev, &s, &[0.0f32, 3f32.ln()]);
    let routed = Buffer::alloc(&dev, 16).unwrap();
    let (rids, rw) = (
        Tensor::new(&routed, 0, I32, &[1, 2]).unwrap(),
        Tensor::new(&routed, 8, F32, &[1, 2]).unwrap(),
    );
    ops::moe_route(&s, &t(&logits, F32, &[1, 2]), 0.0, &rids, &rw).unwrap();
    let mut got = [0u32; 4];
    s.download(&routed, 0, &mut got).unwrap();
    assert_eq!(got[..2], [1, 0]);
    close(&[got[2], got[3]].map(f32::from_bits), &[0.75, 0.25], 1e-6);
    // Expert 0 is the ones row above, expert 1 the same quants at scale 2: x · e = 32 and 64.
    let mut q8x2 = [q8, q8].concat();
    q8x2[34..36].copy_from_slice(&h(2.0).to_le_bytes());
    let we = up(&dev, &s, &q8x2);
    let wet = t(&we, Q8_0, &[2, 32]);
    let out = Buffer::alloc(&dev, 8).unwrap();
    ops::mmvq_id(&s, &wet, &x1, &rids, None, None, &t(&out, F16, &[1, 2, 1])).unwrap();
    close(&halves(&s, &out, 2), &[64.0, 32.0], 0.0);
    // The combine: 0.75 * 64 + 0.25 * 32.
    let one = t(&out, F16, &[1, 1]);
    ops::mmvq_id(&s, &wet, &x1, &rids, None, Some(&rw), &one).unwrap();
    close(&halves(&s, &out, 1), &[56.0], 0.0);

    // 2 * (1 * 3 + 2 * 4) + 1; alpha and beta swapped would give 1.
    let a = up(&dev, &s, &[1.0, 2.0].map(h));
    let b_t = up(&dev, &s, &[3.0, 4.0].map(h));
    let bias = up(&dev, &s, &[1.0f32]);
    let c = Buffer::alloc(&dev, 2).unwrap();
    let (at, btt, bt) = (
        t(&a, F16, &[1, 2]),
        t(&b_t, F16, &[1, 2]),
        t(&bias, F32, &[1]),
    );
    ops::gemm(&s, &at, &btt, Some(&bt), &t(&c, F16, &[1, 1]), 2.0, 0.0).unwrap();
    close(&halves(&s, &c, 1), &[23.0], 0.0);

    // silu(2) * 1; swapped, silu(1) * 2.
    let gate = up(&dev, &s, &[h(2.0)]);
    let upv = up(&dev, &s, &[h(1.0)]);
    let out = Buffer::alloc(&dev, 2).unwrap();
    let (gt, ut) = (t(&gate, F16, &[1, 1]), t(&upv, F16, &[1, 1]));
    ops::silu_mul(&s, &gt, &ut, &t(&out, F16, &[1, 1])).unwrap();
    close(&halves(&s, &out, 1), &[2.0 / (1.0 + (-2f32).exp())], 1e-3);

    // The f32 residual plus an f16 delta, in place.
    let res = up(&dev, &s, &[1.0f32, 2.0]);
    let delta = up(&dev, &s, &[3.0, 4.0].map(h));
    let rt = t(&res, F32, &[1, 2]);
    ops::add(&s, &rt, &t(&delta, F16, &[1, 2]), &rt).unwrap();
    let mut got = [0f32; 2];
    s.download(&res, 0, &mut got).unwrap();
    assert_eq!(got, [4.0, 6.0]);

    // One cached position: attention returns its v (k and v swapped would return [1, 0]).
    let (kb, vb) = (
        Buffer::alloc(&dev, 8).unwrap(),
        Buffer::alloc(&dev, 8).unwrap(),
    );
    let pool = KvPool {
        k: &kb,
        v: &vb,
        n_blocks: 1,
        block_tokens: 2,
        n_layers: 1,
        n_kv_heads: 1,
        head_dim: 2,
        dtype: F16,
    };
    let k = up(&dev, &s, &[1.0, 0.0].map(h));
    let v = up(&dev, &s, &[0.0, 2.0].map(h));
    let (zero, one) = (up(&dev, &s, &[0i32]), up(&dev, &s, &[1i32]));
    let (q, vt) = (t(&k, F16, &[1, 1, 2]), t(&v, F16, &[1, 1, 2]));
    let (z, o, table) = (
        t(&zero, I32, &[1]),
        t(&one, I32, &[1]),
        t(&zero, I32, &[1, 1]),
    );
    ops::kv_write(&s, &q, &vt, &z, &pool, 0).unwrap();
    let shared = KvPool { v: &kb, ..pool }; // v overwriting k
    assert_eq!(
        ops::kv_write(&s, &q, &vt, &z, &shared, 0).err(),
        Some(KError::InvalidArg)
    );
    let out = Buffer::alloc(&dev, 4).unwrap();
    let ot = t(&out, F16, &[1, 1, 2]);
    ops::attn_decode(&s, &q, &pool, 0, &table, &o, 0, 1.0, &ot).unwrap();
    close(&halves(&s, &out, 2), &[0.0, 2.0], 0.0);
    s.upload(&out, 0, &[0u16; 2]).unwrap();
    ops::attn_prefill(&s, &q, &pool, 0, &table, &z, &o, 1.0, &ot).unwrap();
    close(&halves(&s, &out, 2), &[0.0, 2.0], 0.0);
    let flat = KvPool {
        block_tokens: 0,
        ..pool
    };
    assert_eq!(
        ops::kv_write(&s, &q, &vt, &z, &flat, 0).err(),
        Some(KError::InvalidArg)
    );

    // The top 2 of [0, 3, 1, 2], then -inf and id -1; the LSE is over all four.
    let logits = up(&dev, &s, &[0.0f32, 3.0, 1.0, 2.0]);
    let out = Buffer::alloc(&dev, 129 * 4).unwrap();
    let vals = Tensor::new(&out, 0, F32, &[1, 64]).unwrap();
    let ids = Tensor::new(&out, 256, I32, &[1, 64]).unwrap();
    let lse = Tensor::new(&out, 512, F32, &[1]).unwrap();
    let p = SampleParams {
        temperature: 1.0,
        top_k: 2,
        repeat_penalty: 1.0,
        penalty_window: 0,
    };
    let lt = t(&logits, F32, &[1, 4]);
    ops::sample_prep(&s, &lt, &[p], None, &vals, &ids, &lse).unwrap();
    let mut got = [0f32; 129];
    s.download(&out, 0, &mut got).unwrap();
    assert_eq!(got[..3], [3.0, 2.0, f32::NEG_INFINITY]);
    let id = |j: usize| got[64 + j].to_bits() as i32;
    assert_eq!([id(0), id(1), id(2)], [1, 3, -1]);
    let want = [0f32, 3.0, 1.0, 2.0]
        .iter()
        .map(|x| x.exp())
        .sum::<f32>()
        .ln();
    close(&got[128..], &[want], 1e-5);
}

#[test]
fn checks_before_ffi() {
    let (dev, s) = cpu();
    let buf = Buffer::alloc(&dev, 64).unwrap();
    assert_eq!(
        Tensor::new(&buf, 4, DType::F32, &[4, 4]).err(),
        Some(KError::InvalidArg)
    ); // past the end
    assert_eq!(
        Tensor::new(&buf, 0, DType::Q8_0, &[1, 31]).err(),
        Some(KError::InvalidArg)
    ); // part block
    assert_eq!(
        Tensor::new(&buf, 0, DType::F32, &[]).err(),
        Some(KError::InvalidArg)
    );
    assert!(s.upload(&buf, 60, &[0u32; 2]).is_err());
    let t = Tensor::new(&buf, 0, DType::F32, &[4, 4]).unwrap();
    assert!(t.rows(3, 2).is_err());
    let rows = t.rows(1, 2).unwrap();
    assert_eq!((rows.offset(), rows.shape()), (16, &[2i64, 4][..]));
    assert_eq!(t.reshape(&[2, 8]).unwrap().shape(), &[2, 8]);
    assert!(t.reshape(&[2, 4]).is_err());
    // Offsets suit the dtype: whole elements, 2 bytes for a quant block.
    assert!(Tensor::new(&buf, 2, DType::F32, &[1, 4]).is_err());
    assert!(Tensor::new(&buf, 2, F16, &[1, 4]).is_ok());
    assert!(Tensor::new(&buf, 1, DType::Q8_0, &[1, 32]).is_err());
    assert!(Tensor::new(&buf, 2, DType::Q8_0, &[1, 32]).is_ok());

    // Handles of another device: a backend would cast them to its own types. b is f16, so only the
    // device check can reject the call.
    let (_dev2, s2) = cpu();
    let b16 = Tensor::new(&buf, 0, F16, &[4, 4]).unwrap();
    assert_eq!(ops::add(&s2, &t, &b16, &t).err(), Some(KError::InvalidArg));
    let (e1, e2) = (Event::record(&s).unwrap(), Event::record(&s2).unwrap());
    assert_eq!(e1.elapsed_ns(&e2).err(), Some(KError::InvalidArg));
    assert!(e1.elapsed_ns(&Event::record(&s).unwrap()).is_ok());
    let mut dst = [0u32; 2];
    assert!(s2.download(&buf, 0, &mut dst).is_err());
    assert!(s.download(&buf, 60, &mut dst).is_err());
    assert_eq!(Device::open(-3).err(), Some(KError::InvalidArg)); // -1 and -2 are the CPU devices

    // sample_prep reads one SampleParams per logits row.
    let logits = Tensor::new(&buf, 0, DType::F32, &[2, 8]).unwrap();
    let out = Buffer::alloc(&dev, 2 * 64 * 4 * 2 + 8).unwrap();
    let vals = Tensor::new(&out, 0, DType::F32, &[2, 64]).unwrap();
    let ids = Tensor::new(&out, 512, DType::I32, &[2, 64]).unwrap();
    let lse = Tensor::new(&out, 1024, DType::F32, &[2]).unwrap();
    let p = SampleParams {
        temperature: 0.0,
        top_k: 1,
        repeat_penalty: 1.0,
        penalty_window: 0,
    };
    s.upload(&buf, 0, &[0f32; 16]).unwrap(); // new buffers hold NaN poison, which sample_prep rejects
    assert_eq!(
        ops::sample_prep(&s, &logits, &[p], None, &vals, &ids, &lse).err(),
        Some(KError::InvalidArg)
    );
    ops::sample_prep(&s, &logits, &[p, p], None, &vals, &ids, &lse).unwrap();
}

/// Views whose byte size overflows: rejected, never wrapped into range (release) or a panic (debug).
#[test]
fn overflowing_views_are_rejected() {
    let (dev, _) = cpu();
    let buf = Buffer::alloc(&dev, 64).unwrap();
    for shape in [
        &[1i64 << 62][..],
        &[0, 1 << 40, 1 << 40],
        &[0, 1 << 32, 1 << 32],
    ] {
        assert_eq!(
            Tensor::new(&buf, 0, DType::F32, shape).err(),
            Some(KError::InvalidArg),
            "{shape:?}"
        );
    }
    // A 1-D quant tensor has no whole-block row.
    assert!(Tensor::new(&buf, 0, DType::Q8_0, &[32]).is_err());
    let t = Tensor::new(&buf, 0, DType::F32, &[4, 4]).unwrap();
    assert!(t.rows(1, i64::MAX).is_err());
    assert!(t.rows(i64::MAX, 1).is_err());

    // A pool of 2^63 f16 elements per buffer, rejected in Rust: last_error() is thread-local and only
    // set by a failing FFI call, so it stays empty on a fresh thread if the backend was never asked.
    let (err, detail) = std::thread::spawn(|| {
        let (dev, s) = cpu();
        let buf = Buffer::alloc(&dev, 64).unwrap();
        let v = Buffer::alloc(&dev, 64).unwrap();
        let kv = Tensor::new(&buf, 0, F16, &[1, 2, 1]).unwrap();
        let slots = Tensor::new(&buf, 0, I32, &[1]).unwrap();
        let pool = KvPool {
            k: &buf,
            v: &v,
            n_blocks: 1 << 31,
            block_tokens: 1 << 31,
            n_layers: 1,
            n_kv_heads: 2,
            head_dim: 1,
            dtype: F16,
        };
        let err = ops::kv_write(&s, &kv, &kv, &slots, &pool, 0).err();
        (err, larch_kernels::last_error())
    })
    .join()
    .unwrap();
    assert_eq!((err, detail.as_str()), (Some(KError::InvalidArg), ""));
}

#[test]
fn cpu_has_no_graphs() {
    let (_dev, s) = cpu();
    let mut ran = false;
    let g = Graph::capture(&s, |_| {
        ran = true;
        Ok(())
    });
    assert!(matches!(g, Ok(None)) && !ran);
    assert!(!s.capturing());
}

/// The CPU device the engine computes on opens beside the reference and runs ops on its own buffers.
#[test]
fn the_compute_cpu_runs_ops() {
    let dev = Device::open(CPU_COMPUTE).unwrap();
    let s = Stream::new(&dev).unwrap();
    let (a, b) = (
        up(&dev, &s, &[1.5f32, -2.0]),
        up(&dev, &s, &[h(0.25), h(4.0)]),
    );
    let x = t(&a, F32, &[1, 2]);
    ops::add(&s, &x, &t(&b, F16, &[1, 2]), &x).unwrap();
    let mut got = [0f32; 2];
    s.download(&a, 0, &mut got).unwrap();
    assert_eq!(got, [1.75, 2.0]);
}

/// A rejected partial capture executes none of its writes and leaves the stream reusable. Run this
/// explicitly on the graph-test device; CPU-only jobs exercise the unsupported backend above.
#[test]
fn unsupported_capture_discards_partial_work() {
    let Ok(ordinal) = std::env::var("LARCH_TEST_DEVICE") else {
        return;
    };
    let dev = Device::open(ordinal.parse().unwrap()).unwrap();
    let s = Stream::new(&dev).unwrap();
    let x = up(&dev, &s, &[1.0_f32, 2.0]);
    let delta = up(&dev, &s, &[h(3.0), h(4.0)]);
    s.sync().unwrap();
    let xt = t(&x, F32, &[1, 2]);
    let dt = t(&delta, F16, &[1, 2]);
    let graph = Graph::capture(&s, |s| {
        ops::add(s, &xt, &dt, &xt)?;
        Err(KError::Unsupported)
    })
    .unwrap();
    assert!(graph.is_none());
    let mut actual = [0.0_f32; 2];
    s.download(&x, 0, &mut actual).unwrap();
    s.sync().unwrap();
    assert_eq!(actual, [1.0, 2.0]);
    ops::add(&s, &xt, &dt, &xt).unwrap();
    s.download(&x, 0, &mut actual).unwrap();
    s.sync().unwrap();
    assert_eq!(actual, [4.0, 6.0]);
    let graph = Graph::capture(&s, |s| ops::add(s, &xt, &dt, &xt)).unwrap();
    if let Some(graph) = graph {
        graph.launch(&s).unwrap();
        s.download(&x, 0, &mut actual).unwrap();
        s.sync().unwrap();
        assert_eq!(actual, [7.0, 10.0]);
    }
    assert!(matches!(
        Graph::capture(&s, |_| Err(KError::InvalidArg)),
        Err(KError::InvalidArg)
    ));
}

#[test]
fn a_cut_outside_a_capture_is_invalid() {
    let (_dev, s) = cpu();
    assert_eq!(s.cut(7), Err(KError::InvalidArg));
}

/// Two pieces with a host step between them (the cut's tag picks it) replay the eager sequence's
/// bytes, twice. Run with LARCH_TEST_DEVICE set and --ignored on the graph-test device.
#[test]
#[ignore = "needs a graph device: LARCH_TEST_DEVICE"]
fn pieces_with_a_host_step_between_replay_as_eager() {
    let Ok(ordinal) = std::env::var("LARCH_TEST_DEVICE") else {
        return;
    };
    let dev = Device::open(ordinal.parse().unwrap()).unwrap();
    let s = Stream::new(&dev).unwrap();
    let delta = up(&dev, &s, &[h(3.0), h(4.0)]);
    let dt = t(&delta, F16, &[1, 2]);
    let host = |b: &Buffer, tag: u64| {
        let mut v = [0f32; 2];
        s.download(b, 0, &mut v).unwrap();
        s.upload(b, 0, &v.map(|x| x * tag as f32 - 0.1)).unwrap();
    };
    let bits = |b: &Buffer| {
        let mut v = [0f32; 2];
        s.download(b, 0, &mut v).unwrap();
        v.map(f32::to_bits)
    };
    let (eager, replayed) = (up(&dev, &s, &[1f32, 2.0]), up(&dev, &s, &[1f32, 2.0]));
    let (et, rt) = (t(&eager, F32, &[1, 2]), t(&replayed, F32, &[1, 2]));
    let (cut, last) = Graph::capture_pieces(&s, |s| {
        ops::add(s, &rt, &dt, &rt)?;
        s.cut(3)?;
        ops::add(s, &rt, &dt, &rt)
    })
    .unwrap()
    .expect("the device captures");
    assert_eq!(cut.len(), 1);
    assert_eq!(bits(&replayed), bits(&eager), "a capture runs nothing");
    for _ in 0..2 {
        ops::add(&s, &et, &dt, &et).unwrap();
        host(&eager, 3);
        ops::add(&s, &et, &dt, &et).unwrap();
        for (g, tag) in &cut {
            g.launch(&s).unwrap();
            host(&replayed, *tag);
        }
        last.launch(&s).unwrap();
        assert_eq!(bits(&replayed), bits(&eager));
    }
}

/// The debug-only device-loss op exists on CUDA alone: the CPU reference answers Unsupported.
#[cfg(any(debug_assertions, feature = "fault-inject"))]
#[test]
fn fault_device_lost_is_unsupported_on_cpu() {
    let (_dev, s) = cpu();
    assert_eq!(
        larch_kernels::fault_device_lost(&s),
        Err(KError::Unsupported)
    );
}

/// With a GPU backend built: its events and the CPU reference's never meet in one backend call.
#[test]
fn gpu_events_against_cpu_ones() {
    if larch_kernels::device_count().unwrap() == 0 {
        return;
    }
    // A listed GPU this build cannot use (below the floor, an old driver) skips like no GPU.
    let gpu = match Device::open(0) {
        Err(KError::Unsupported | KError::Driver { .. }) => return,
        g => g.unwrap(),
    };
    let gs = Stream::new(&gpu).unwrap();
    let (_dev, s) = cpu();
    let (g, c) = (Event::record(&gs).unwrap(), Event::record(&s).unwrap());
    assert_eq!(g.elapsed_ns(&c).err(), Some(KError::InvalidArg));
    assert_eq!(c.elapsed_ns(&g).err(), Some(KError::InvalidArg));
    assert!(g.elapsed_ns(&Event::record(&gs).unwrap()).is_ok());
}

/// A GPU that cannot open fails with a reason, also with no GPU driver installed (GPU-less CI), where
/// CUDA 13's runtime has no error strings and formatting one crashed the process.
#[test]
fn unopenable_gpu_fails_with_a_reason() {
    let e = Device::open(99).err().expect("no backend has GPU 99");
    // Rust refuses no driver (Linux: version 0) or an old one: the versions are the reason.
    assert!(
        !larch_kernels::last_error().is_empty()
            || matches!(e, KError::Driver { need, .. } if need > 0),
        "{e:?}"
    );
}

/// A temp file removed on drop, so a failed assert leaves nothing behind for the next run.
#[cfg(any(unix, windows))]
struct Temp(std::path::PathBuf);

#[cfg(any(unix, windows))]
impl Temp {
    fn new(name: &str, bytes: &[u8]) -> Temp {
        let path = std::env::temp_dir().join(format!("larch-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        Temp(path)
    }

    fn map(&self) -> std::io::Result<Mmap> {
        Mmap::map(&std::fs::File::open(&self.0).unwrap())
    }
}

#[cfg(any(unix, windows))]
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0); // a failed test may hold the map still
    }
}

/// A mapped file uploads as its bytes, on every platform that maps (unix, Windows), and the file
/// deletes once the map is gone.
#[cfg(any(unix, windows))]
#[test]
fn upload_map_reads_the_file() {
    let bytes: Vec<u8> = (0..70_000).map(|i| (i % 251) as u8).collect();
    let file = Temp::new("upmap", &bytes);
    let map = file.map().unwrap();
    assert_eq!(map.bytes(), bytes.len() as u64);
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let buf = Buffer::alloc(&dev, 1000).unwrap();
    s.upload_map(&buf, 0, &map, 65_000..66_000).unwrap();
    let mut got = vec![0u8; 1000];
    s.download(&buf, 0, &mut got).unwrap();
    assert_eq!(got, bytes[65_000..66_000]);
    drop(map);
    std::fs::remove_file(&file.0).unwrap();
}

/// An empty file has nothing to map, on every platform that maps.
#[cfg(any(unix, windows))]
#[test]
fn an_empty_file_does_not_map() {
    assert!(Temp::new("empty", b"").map().is_err());
}

/// A file of `len` bytes, byte i = i % 251, mapped; the name is gone on return (the map keeps it).
#[cfg(unix)]
fn mapped(name: &str, len: usize) -> Mmap {
    let path = std::env::temp_dir().join(format!("larch-{name}-{}", std::process::id()));
    std::fs::write(&path, (0..len).map(|i| (i % 251) as u8).collect::<Vec<_>>()).unwrap();
    let map = Mmap::map(&std::fs::File::open(&path).unwrap());
    std::fs::remove_file(&path).unwrap();
    map.unwrap()
}

/// 64 KiB: a page boundary for 4, 16 and 64 KiB pages.
#[cfg(unix)]
const PAGE: u64 = 1 << 16;

/// wrap_host's own checks run before the backend (last_error stays empty on a fresh thread); a valid
/// range reaches it, and the CPU reference answers Unsupported.
#[cfg(unix)]
#[test]
fn wrap_host_checks_before_ffi() {
    let map = mapped("wrap", 3 * PAGE as usize + 100);
    assert_eq!(map.bytes(), 3 * PAGE + 100);
    let (refused, detail, cpu) = std::thread::spawn(move || {
        let dev = Device::open(CPU).unwrap();
        let len = map.bytes();
        // not page-aligned, empty, past the file, empty at a page
        let refused = [1..PAGE, 0..0, PAGE..len + 1, PAGE..PAGE]
            .map(|r| Buffer::wrap_host(&dev, &map, r).err());
        let detail = larch_kernels::last_error();
        (
            refused,
            detail,
            Buffer::wrap_host(&dev, &map, PAGE..len).err(),
        )
    })
    .join()
    .unwrap();
    assert_eq!(refused, [Some(KError::InvalidArg); 4]);
    assert_eq!((detail.as_str(), cpu), ("", Some(KError::Unsupported)));
}

/// view_host's own checks run before the backend: an empty range, one past the buffer, a buffer that
/// is not a host buffer and another device's host buffer are refused; a valid range reaches it, and the
/// CPU reference answers Unsupported.
#[test]
fn view_host_checks_before_ffi() {
    let (refused, detail, cpu) = std::thread::spawn(|| {
        let dev = Device::open(CPU).unwrap();
        let host = Buffer::alloc_host(&dev, 64).unwrap();
        let plain = Buffer::alloc(&dev, 64).unwrap();
        let other = Device::open(CPU).unwrap();
        let theirs = Buffer::alloc_host(&other, 64).unwrap();
        let refused = [
            Buffer::view_host(&dev, &host, 8..8).err(),
            Buffer::view_host(&dev, &host, 8..65).err(),
            Buffer::view_host(&dev, &plain, 0..64).err(),
            Buffer::view_host(&dev, &theirs, 0..64).err(),
        ];
        let detail = larch_kernels::last_error();
        (refused, detail, Buffer::view_host(&dev, &host, 8..64).err())
    })
    .join()
    .unwrap();
    assert_eq!(refused, [Some(KError::InvalidArg); 4]);
    assert_eq!((detail.as_str(), cpu), ("", Some(KError::Unsupported)));
}

/// On CUDA (GPU 0): a view of a page-locked host buffer reads back its bytes in place, keeps the host
/// buffer alive past its own handle, and refuses uploads. Skips without a CUDA GPU.
#[test]
fn a_host_view_on_cuda_reads_the_host_bytes() {
    let dev = match Device::open(0) {
        Ok(d) if d.props().unwrap().backend == larch_kernels::Backend::Cuda => d,
        _ => return,
    };
    let s = Stream::new(&dev).unwrap();
    let host = Buffer::alloc_host(&dev, 4096).unwrap();
    let cpu = Device::open(CPU).unwrap();
    let cs = Stream::new(&cpu).unwrap();
    let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    cs.upload(&host, 0, &src).unwrap();
    let view = Buffer::view_host(&dev, &host, 1024..4096).unwrap();
    drop(host);
    let mut got = vec![0u8; 3072];
    s.download(&view, 0, &mut got).unwrap();
    s.sync().unwrap();
    assert_eq!(got, src[1024..]);
    assert_eq!(
        s.upload(&view, 0, &[0u8; 4]).err(),
        Some(KError::InvalidArg)
    );
}

/// On Metal (GPU 0): a wrapped range reads back as the file after the caller's map handle is gone, and
/// is never written. Skips without a Metal GPU.
#[cfg(unix)]
#[test]
fn wrapped_file_on_metal() {
    let dev = match Device::open(0) {
        Ok(d) if d.props().unwrap().backend == larch_kernels::Backend::Metal => d,
        _ => return,
    };
    let s = Stream::new(&dev).unwrap();
    let map = mapped("metal", 2 * PAGE as usize + 100);
    let w = Buffer::wrap_host(&dev, &map, PAGE..map.bytes()).unwrap();
    drop(map);
    let mut got = vec![0u8; (PAGE + 100) as usize];
    s.download(&w, 0, &mut got).unwrap();
    assert!(
        got.iter()
            .enumerate()
            .all(|(i, &b)| b == ((i as u64 + PAGE) % 251) as u8)
    );
    assert_eq!(s.upload(&w, 0, &[0u8; 4]).err(), Some(KError::InvalidArg));
    let (x, h) = (
        Buffer::alloc(&dev, 16).unwrap(),
        Buffer::alloc(&dev, 8).unwrap(),
    );
    let (xt, ht) = (t(&x, F32, &[1, 4]), t(&h, F16, &[1, 4]));
    let wt = Tensor::new(&w, 0, F32, &[1, 4]).unwrap();
    assert_eq!(ops::add(&s, &xt, &ht, &wt).err(), Some(KError::InvalidArg));
}

/// A column view keeps its parent's row stride through the FFI: `gemm_ex` on two heads of a
/// `[2, 6]` matrix, written back into the middle columns of another.
#[test]
fn column_views_keep_the_row_stride() {
    let (dev, s) = cpu();
    let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let xb = up(&dev, &s, &x);
    let cb = up(&dev, &s, &[0f32; 8]);
    let (xt, ct) = (t(&xb, F32, &[2, 6]), t(&cb, F32, &[2, 4]));
    // c[:, 1..3] = x[:, 0..2] · x[:, 4..6]ᵀ
    let (a, b, c) = (
        xt.cols(0, 2).unwrap(),
        xt.cols(4, 2).unwrap(),
        ct.cols(1, 2).unwrap(),
    );
    ops::gemm_ex(&s, &a, &b, &c, false, true, 1.0, 0.0).unwrap();
    let mut got = [0f32; 8];
    s.download(&cb, 0, &mut got).unwrap();
    s.sync().unwrap();
    assert_eq!(got, [0.0, 5.0, 11.0, 0.0, 0.0, 59.0, 137.0, 0.0]);
    assert_eq!(c.reshape(&[4]).err(), Some(KError::InvalidArg));
    assert_eq!(xt.cols(5, 2).err(), Some(KError::InvalidArg));
}

/// A host buffer of one device ([`Buffer::alloc_host`]) is any CPU device's stream's own until it is
/// shared ([`Buffer::share_host`]); then it is read-only, and a copy source by range for its own
/// device's streams ([`Stream::upload_from`]). A plain buffer is never shared, and a range past the
/// end is refused before the backend.
#[test]
fn a_host_buffer_is_every_cpu_streams_own_then_a_copy_source() {
    let (reference, rs) = cpu();
    let mut host = Buffer::alloc_host(&reference, 64).unwrap();
    let compute = Device::open(CPU_COMPUTE).unwrap();
    let cs = Stream::new(&compute).unwrap();
    let data: Vec<u8> = (0..64).collect();
    cs.upload(&host, 0, &data).unwrap();
    let src = host.share_host().unwrap();
    assert_eq!(cs.upload(&host, 0, &data), Err(KError::InvalidArg));
    let dst = Buffer::alloc(&reference, 32).unwrap();
    rs.upload_from(&dst, 0, &src, 8..40).unwrap();
    rs.sync().unwrap();
    let mut got = vec![0u8; 32];
    rs.download(&dst, 0, &mut got).unwrap();
    rs.sync().unwrap();
    assert_eq!(got, data[8..40]);
    // Its CPU readers still take it, as an op's input.
    let mut back = vec![0u8; 64];
    cs.download(&host, 0, &mut back).unwrap();
    assert_eq!(back, data);
    let mut plain = Buffer::alloc(&reference, 64).unwrap();
    assert!(plain.share_host().is_none());
    assert_eq!(
        rs.upload_from(&dst, 0, &src, 40..72),
        Err(KError::InvalidArg)
    );
    assert_eq!(
        rs.upload_from(&dst, 8, &src, 0..32),
        Err(KError::InvalidArg)
    );
    // Another device's plain buffer is not a CPU stream's own.
    assert_eq!(cs.upload(&plain, 0, &data), Err(KError::InvalidArg));
}
