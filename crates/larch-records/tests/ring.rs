#![cfg(not(loom))]

use bytemuck::Zeroable;
use larch_records::*;
use proptest::prelude::*;

/// Every field is derived from the seq, so a torn read cannot pass `check`.
fn sample(seq: u64) -> EngineSample {
    EngineSample {
        step: seq,
        dur_ns: !seq,
        host_ns: seq.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        n_running: seq as u32,
        tot_rejected: seq ^ 0xA5A5,
        sm_occupancy: MF32::estimated(seq as f32),
        ..EngineSample::zeroed()
    }
}

fn span(seq: u64) -> RequestSpan {
    RequestSpan {
        req_id: seq,
        t_done_ns: !seq,
        session_id: seq ^ 7,
        ..RequestSpan::zeroed()
    }
}

/// Even seqs are EngineSamples and odd ones RequestSpans, so decoding of mixed kinds is covered.
fn push(w: &mut RingWriter, seq: u64) {
    let h = RecordHeader {
        engine_id: 99,
        t_mono_ns: seq,
        ..RecordHeader::zeroed()
    };
    let got = if seq.is_multiple_of(2) {
        w.push(h, &sample(seq))
    } else {
        w.push(h, &span(seq))
    };
    assert_eq!(got, seq);
}

fn check(r: &Record, seq: u64) {
    assert_eq!(r.header().seq, seq);
    assert_eq!(r.header().t_mono_ns, seq);
    assert_eq!(r.header().engine_id, 99);
    match r {
        Record::EngineSample(h, b) => {
            assert_eq!(
                (h.kind(), h.layout),
                (RecordKind::EngineSample, EngineSample::LAYOUT_VERSION)
            );
            assert_eq!(*b, sample(seq));
        }
        Record::RequestSpan(h, b) => {
            assert_eq!(h.kind(), RecordKind::RequestSpan);
            assert_eq!(*b, span(seq));
        }
        other => panic!("unexpected record {other:?}"),
    }
}

#[test]
fn push_then_read() {
    let (mut w, handle) = ring(8);
    let mut r = handle.reader(ReadFrom::Oldest);
    assert_eq!(r.next(), Read::Empty);
    for s in 0..5 {
        push(&mut w, s);
    }
    assert_eq!(handle.head(), 5);
    for s in 0..5 {
        match r.next() {
            Read::Record(rec) => check(&rec, s),
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(r.next(), Read::Empty);
    assert_eq!(r.dropped_total(), 0);

    let mut latest = handle.reader(ReadFrom::Latest);
    assert_eq!(latest.next(), Read::Empty);
    push(&mut w, 5);
    assert!(matches!(latest.next(), Read::Record(rec) if rec.header().seq == 5));
}

#[test]
fn lagged_reader_skips_ahead_and_counts() {
    let (mut w, handle) = ring(4);
    let mut r = handle.reader(ReadFrom::Seq(0));
    for s in 0..10 {
        push(&mut w, s);
    }
    // head 10, capacity 4: resume at 10 + 1 - 4 = 7.
    assert_eq!(r.next(), Read::Lagged { dropped: 7 });
    let mut out = Vec::new();
    assert_eq!(
        r.drain(&mut out, 100),
        DrainStats {
            records: 3,
            dropped: 0
        }
    );
    for (rec, s) in out.iter().zip(7..) {
        check(rec, s);
    }
    assert_eq!(r.dropped_total(), 7);

    // ReadFrom::Oldest after wrap starts at head - capacity.
    let mut oldest = handle.reader(ReadFrom::Oldest);
    assert_eq!(oldest.position(), 6);
    assert!(matches!(oldest.next(), Read::Record(rec) if rec.header().seq == 6));
}

#[test]
fn drain_respects_max() {
    let (mut w, handle) = ring(16);
    let mut r = handle.reader(ReadFrom::Oldest);
    for s in 0..10 {
        push(&mut w, s);
    }
    let mut out = Vec::new();
    assert_eq!(r.drain(&mut out, 4).records, 4);
    assert_eq!(r.drain(&mut out, 100).records, 6);
    assert_eq!(out.len(), 10);
}

#[test]
#[should_panic(expected = "power of two")]
fn capacity_must_be_a_power_of_two() {
    let _ = ring(3);
}

#[test]
fn handle_and_reader_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<RingHandle>();
    send::<RingReader>();
    send::<RingWriter>();
}

/// One writer and two readers on real threads, with a ring small enough to lag constantly.
#[test]
fn concurrent_readers_never_see_torn_or_reordered_records() {
    const N: u64 = 200_000;
    let (mut w, handle) = ring(64);
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let mut r = handle.reader(ReadFrom::Seq(0));
            std::thread::spawn(move || {
                let (mut records, mut dropped, mut last) = (0u64, 0u64, None);
                while r.position() < N {
                    match r.next() {
                        Read::Record(rec) => {
                            let seq = rec.header().seq;
                            assert!(last.is_none_or(|l| seq > l));
                            check(&rec, seq);
                            assert_eq!(r.position(), seq + 1);
                            last = Some(seq);
                            records += 1;
                        }
                        Read::Lagged { dropped: d } => dropped += d,
                        Read::Empty => std::hint::spin_loop(),
                    }
                }
                assert_eq!(records + dropped, N);
                assert_eq!(r.dropped_total(), dropped);
                records
            })
        })
        .collect();
    for s in 0..N {
        push(&mut w, s);
    }
    for t in readers {
        assert!(t.join().unwrap() > 0);
    }
}

#[derive(Clone, Debug)]
enum Op {
    Push(u8),
    Read(usize, u8),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1u8..20).prop_map(Op::Push),
        (0usize..2, 1u8..20).prop_map(|(r, n)| Op::Read(r, n))
    ]
}

proptest! {
    /// Single-threaded interleavings against a model: records come in seq order, each one intact,
    /// and every seq a reader passes is either read or counted as dropped, exactly once.
    #[test]
    fn ordering_and_drop_accounting(log2_cap in 1u32..5, ops in prop::collection::vec(op(), 1..60)) {
        let cap = 1u64 << log2_cap;
        let (mut w, handle) = ring(cap as usize);
        let mut readers = [handle.reader(ReadFrom::Oldest), handle.reader(ReadFrom::Oldest)];
        let mut cursor = [0u64; 2];
        let mut read = [0u64; 2];
        let mut head = 0u64;
        for op in ops {
            match op {
                Op::Push(n) => for _ in 0..n {
                    push(&mut w, head);
                    head += 1;
                },
                Op::Read(i, n) => for _ in 0..n {
                    let got = readers[i].next();
                    let c = cursor[i];
                    if c >= head {
                        prop_assert_eq!(got, Read::Empty);
                    } else if head - c > cap {
                        let to = head + 1 - cap;
                        prop_assert_eq!(got, Read::Lagged { dropped: to - c });
                        cursor[i] = to;
                    } else {
                        let Read::Record(rec) = got else { panic!("expected seq {c}, got {got:?}") };
                        check(&rec, c);
                        cursor[i] = c + 1;
                        read[i] += 1;
                    }
                    prop_assert_eq!(readers[i].position(), cursor[i]);
                    prop_assert_eq!(read[i] + readers[i].dropped_total(), cursor[i]);
                },
            }
            prop_assert_eq!(handle.head(), head);
        }
    }
}
