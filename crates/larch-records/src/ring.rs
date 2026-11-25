//! The single-producer, many-reader ring (4.4). A seqlock per slot, every access atomic, so there
//! is no `unsafe`, no data race and no UB. The producer never waits and never allocates; a reader
//! that falls behind gets `Lagged` instead of slowing it down.

#[cfg(loom)]
use loom::sync::{
    Arc,
    atomic::{AtomicU64, Ordering, fence},
};
#[cfg(not(loom))]
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering, fence},
};

use crate::{Record, RecordBody, RecordHeader, SLOT_WORDS};

/// 65 536 slots × 256 B = 16 MB.
pub const DEFAULT_CAPACITY: usize = 65_536;

/// Holds seq `s` once `stamp == 2s + 2`; `2s + 1` while it is being written.
struct Slot<const W: usize> {
    stamp: AtomicU64,
    words: [AtomicU64; W],
}

/// The seqlock protocol over slots of `W` words. The ring uses `W = SLOT_WORDS`; the loom test
/// runs this same code with fewer words so the model stays small enough to check exhaustively.
struct Core<const W: usize> {
    slots: Box<[Slot<W>]>,
    mask: u64,
    /// The next seq the writer will push.
    head: AtomicU64,
}

enum Next<const W: usize> {
    Words([u64; W]),
    Empty,
    Lagged { dropped: u64 },
}

impl<const W: usize> Core<W> {
    fn new(capacity_pow2: usize) -> Self {
        assert!(
            capacity_pow2 >= 2 && capacity_pow2.is_power_of_two(),
            "ring capacity must be a power of two >= 2, got {capacity_pow2}"
        );
        let slots = (0..capacity_pow2)
            .map(|_| Slot {
                stamp: AtomicU64::new(0),
                words: std::array::from_fn(|_| AtomicU64::new(0)),
            })
            .collect();
        Core {
            slots,
            mask: capacity_pow2 as u64 - 1,
            head: AtomicU64::new(0),
        }
    }

    fn capacity(&self) -> u64 {
        self.mask + 1
    }

    fn head(&self) -> u64 {
        self.head.load(Ordering::Acquire)
    }

    /// Writer, seq `s`. One writer, seqs in order.
    fn write(&self, s: u64, words: [u64; W]) {
        let slot = &self.slots[(s & self.mask) as usize];
        slot.stamp.store(2 * s + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        for (dst, w) in slot.words.iter().zip(words) {
            dst.store(w, Ordering::Relaxed);
        }
        slot.stamp.store(2 * s + 2, Ordering::Release);
        self.head.store(s + 1, Ordering::Release);
    }

    /// Reader at `*cursor`; moves it past what was read or dropped.
    fn read(&self, cursor: &mut u64) -> Next<W> {
        let s = *cursor;
        let want = 2 * s + 2;
        let slot = &self.slots[(s & self.mask) as usize];
        let stamp = slot.stamp.load(Ordering::Acquire);
        if stamp < want {
            return Next::Empty;
        }
        if stamp == want {
            let mut words = [0u64; W];
            for (dst, w) in words.iter_mut().zip(&slot.words) {
                *dst = w.load(Ordering::Relaxed);
            }
            fence(Ordering::Acquire);
            if slot.stamp.load(Ordering::Relaxed) == want {
                *cursor = s + 1;
                return Next::Words(words);
            }
        }
        // Overwritten before or during the copy: skip to one past the oldest slot, which the
        // writer could be rewriting next.
        let to = (self.head() + 1).saturating_sub(self.capacity()).max(s + 1);
        *cursor = to;
        Next::Lagged { dropped: to - s }
    }
}

/// Creates a ring of `capacity_pow2` slots (a power of two, at least 2).
pub fn ring(capacity_pow2: usize) -> (RingWriter, RingHandle) {
    let core = Arc::new(Core::new(capacity_pow2));
    (
        RingWriter {
            core: core.clone(),
            next: 0,
        },
        RingHandle { core },
    )
}

/// The only producer. Not `Clone`; it is moved into the engine thread (section 3, invariant 1).
///
/// ```compile_fail
/// let (w, _) = larch_records::ring(2);
/// let _second_writer = w.clone();
/// ```
pub struct RingWriter {
    core: Arc<Core<SLOT_WORDS>>,
    next: u64,
}

impl RingWriter {
    /// Writes one record and returns its seq. Stamps `kind`, `layout` and `seq` into the header.
    pub fn push<R: RecordBody>(&mut self, mut header: RecordHeader, body: &R) -> u64 {
        let s = self.next;
        header.kind = R::KIND.raw();
        header.layout = R::LAYOUT_VERSION;
        header.seq = s;
        self.core.write(s, Record::encode(&header, body));
        self.next = s + 1;
        s
    }
}

/// Where a new reader starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadFrom {
    /// The next record pushed.
    Latest,
    /// The oldest record still in the ring.
    Oldest,
    /// This seq; if it was already overwritten, the first read is `Lagged`.
    Seq(u64),
}

/// Shares the ring with readers. `Send + Sync`.
#[derive(Clone)]
pub struct RingHandle {
    core: Arc<Core<SLOT_WORDS>>,
}

impl RingHandle {
    pub fn reader(&self, from: ReadFrom) -> RingReader {
        let head = self.head();
        let cursor = match from {
            ReadFrom::Latest => head,
            ReadFrom::Oldest => head.saturating_sub(self.core.capacity()),
            ReadFrom::Seq(s) => s,
        };
        RingReader {
            core: self.core.clone(),
            cursor,
            dropped_total: 0,
        }
    }

    /// The next seq the writer will push (= the number of records pushed so far).
    pub fn head(&self) -> u64 {
        self.core.head()
    }
}

/// The result of one [`RingReader::next`].
// A record is returned by value: boxing it would allocate on every read.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Read {
    Record(Record),
    /// Nothing new yet.
    Empty,
    /// The writer overwrote records before this reader got to them; the reader skipped ahead.
    Lagged {
        dropped: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrainStats {
    pub records: usize,
    pub dropped: u64,
}

/// One per consumer.
pub struct RingReader {
    core: Arc<Core<SLOT_WORDS>>,
    cursor: u64,
    dropped_total: u64,
}

impl RingReader {
    // Not `Iterator::next`: an empty ring and a lag are answers too, not the end.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Read {
        match self.core.read(&mut self.cursor) {
            Next::Words(words) => Read::Record(Record::decode(&words)),
            Next::Empty => Read::Empty,
            Next::Lagged { dropped } => {
                self.dropped_total += dropped;
                Read::Lagged { dropped }
            }
        }
    }

    /// Reads until the ring is empty or `max` records were appended to `out`.
    pub fn drain(&mut self, out: &mut Vec<Record>, max: usize) -> DrainStats {
        let mut stats = DrainStats::default();
        while stats.records < max {
            match self.next() {
                Read::Record(r) => {
                    out.push(r);
                    stats.records += 1;
                }
                Read::Lagged { dropped } => stats.dropped += dropped,
                Read::Empty => break,
            }
        }
        stats
    }

    pub fn dropped_total(&self) -> u64 {
        self.dropped_total
    }

    /// The seq this reader will read next.
    pub fn position(&self) -> u64 {
        self.cursor
    }
}

/// Model-checks the protocol with 1 writer and 2 readers (4.4). Run with
/// `RUSTFLAGS="--cfg loom" cargo test -p larch-records --lib --release`.
/// Two words per slot instead of 32: the protocol code is the same, and every word a reader loads
/// multiplies the executions loom has to explore.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::{Arc, Core, Next};

    /// Both words derive from the seq, so a torn read cannot pass.
    fn words(seq: u64) -> [u64; 2] {
        [seq, !seq]
    }

    fn reader(core: Arc<Core<2>>) -> loom::thread::JoinHandle<()> {
        loom::thread::spawn(move || {
            let (mut cursor, mut records, mut dropped, mut last) = (0u64, 0u64, 0u64, None);
            for _ in 0..3 {
                let before = cursor;
                match core.read(&mut cursor) {
                    Next::Words(w) => {
                        assert_eq!(w, words(before), "torn read");
                        assert!(last.is_none_or(|l| before > l), "out of order");
                        assert_eq!(cursor, before + 1);
                        last = Some(before);
                        records += 1;
                    }
                    Next::Lagged { dropped: d } => {
                        assert!(d > 0 && cursor == before + d);
                        dropped += d;
                    }
                    Next::Empty => assert_eq!(cursor, before),
                }
            }
            assert_eq!(records + dropped, cursor, "drop accounting");
        })
    }

    #[test]
    fn one_writer_two_readers() {
        let mut model = loom::model::Builder::new();
        // Bound 1 explores every interleaving with at most one forced switch (seconds) and fails
        // when the re-check or either fence is removed. LOOM_MAX_PREEMPTIONS=2 goes deeper.
        model.preemption_bound = model.preemption_bound.or(Some(1));
        model.check(|| {
            // Capacity 2 and 3 writes, so the last write reuses seq 0's slot while a reader may
            // still be copying it.
            let core = Arc::new(Core::<2>::new(2));
            let readers = [reader(core.clone()), reader(core.clone())];
            for seq in 0..3 {
                core.write(seq, words(seq));
            }
            for r in readers {
                r.join().unwrap();
            }
        });
    }
}
