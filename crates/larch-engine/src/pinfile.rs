//! `.pin` files (architecture section 8): a pinned document's KV and recurrent state, appended a
//! segment at a time as its read goes and read back a layer at a time. Unstable: may change before
//! 1.0; the version field lets the format change.
//!
//! The pin file is append-only, little endian: a header, then one record per segment of
//! [`PinInfo::segment`] tokens (the last may be shorter).
//!
//! | section | bytes |
//! |---|---|
//! | header | magic `LARCHPN`, version u32, GGUF file id u64, projector file id u64 (0: no image rows), kernel ABI u32, build id (16: the sha256 of the engine version and source commit), KV layers u32, block bytes u64, KV dtype u32, tokens per block u32, state bytes u64, segment tokens u32, then the sha256 of all of that (32) |
//! | record | token count n u32, the n ids as u32, per layer each block's K then V, then the sha256 of the whole record (32) |
//!
//! The recurrent state sits in a small sidecar, replaced whole (temp file and rename) when it
//! moves: magic `LARCHST`, version u32, the header's sha256 (32), tokens covered u64, state bytes
//! u64, the state after the last covered token (none for a dense model), then the sha256 of all of
//! that. An import takes the complete records up to the tokens the sidecar covers; a torn record
//! after them is ignored.

use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

use larch_records::EngineError;
use sha2::{Digest, Sha256};

use crate::{PinInfo, PinOp, PinReply, invalid};

/// Runs one engine step of a pin's persistence ([`PinOp`]).
pub(crate) type Ops<'a> = &'a mut dyn FnMut(PinOp) -> Result<PinReply, EngineError>;

const MAGIC: &[u8; 8] = b"LARCH_PN";
const STATE_MAGIC: &[u8; 8] = b"LARCH_ST";
const VERSION: u32 = 1;
const BUILD_LEN: usize = 16;
/// The header's bytes before its checksum.
const HEAD_LEN: usize = 8 + 4 + 8 + 8 + 4 + BUILD_LEN + 4 + 8 + 4 + 4 + 8 + 4;
/// Bytes read and hashed at a time from an id section.
const STEP: usize = 64 << 10;

/// Why a `.pin` was not written or read. Unstable: may change before 1.0.
#[derive(Debug)]
pub enum PinError {
    /// Nothing to export yet, or a file that is not this engine's to load: another model, layout,
    /// build or document, a bad checksum, a truncated file or a size no engine would write. The
    /// caller reads the document instead.
    Miss,
    /// The file is good but the pool (or host memory) has no room for it now: the caller keeps
    /// the file and tries again once something is unpinned.
    NoRoom,
    /// The reader or writer failed.
    Io(io::Error),
    /// The engine refused or failed a step. `IMPORT_TORN` is one: a reload broke the import off,
    /// and the file is good to import again.
    Engine(EngineError),
}

impl From<io::Error> for PinError {
    fn from(e: io::Error) -> Self {
        match e.kind() {
            io::ErrorKind::UnexpectedEof => PinError::Miss,
            _ => PinError::Io(e),
        }
    }
}

impl From<EngineError> for PinError {
    fn from(e: EngineError) -> Self {
        match e {
            // `IMPORT_TORN` is no miss: it stays the engine's, and the caller keeps the file.
            EngineError::InvalidRequest(invalid::PIN_CHANGED) => PinError::Miss,
            EngineError::InvalidRequest(invalid::PINS_FULL | invalid::DOC_TOO_LONG) => {
                PinError::NoRoom
            }
            e => PinError::Engine(e),
        }
    }
}

fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// The build id the header holds: KV from another version or commit is not this build's, a
/// kernel fix included.
fn build(info: &PinInfo) -> [u8; BUILD_LEN] {
    let id = Sha256::digest([info.version, info.build].join("\0"));
    id[..BUILD_LEN].try_into().expect("16")
}

/// The header this engine writes (without its checksum): the same bytes whenever it is asked.
fn header(info: &PinInfo) -> Vec<u8> {
    let mut h = Vec::with_capacity(HEAD_LEN);
    h.extend(MAGIC);
    h.extend(VERSION.to_le_bytes());
    h.extend(info.file.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    h.extend(info.abi.to_le_bytes());
    h.extend(build(info));
    h.extend(info.layers.to_le_bytes());
    h.extend(info.block_bytes.to_le_bytes());
    h.extend(info.dtype.to_le_bytes());
    h.extend(info.block_tokens.to_le_bytes());
    h.extend(info.state_bytes.to_le_bytes());
    h.extend(info.segment.to_le_bytes());
    debug_assert_eq!(h.len(), HEAD_LEN);
    h
}

/// Writes `bytes` and feeds them to `hash`.
fn put(w: &mut dyn Write, hash: &mut Sha256, bytes: &[u8]) -> io::Result<()> {
    hash.update(bytes);
    w.write_all(bytes)
}

/// Where an import's time went, printed as `larch_pin_import,<phase>,<ms>` lines (as the
/// engine's `larch_pin_stall`): `reload` (the documents plan) once it is done, then, once the
/// import that took blocks ends (finished or failed), `read` and `checksum` on this thread,
/// `upload` the engine steps that copy the layers to the device, and `total` from the import's
/// start, its finish included. A file refused before the reload prints nothing.
#[derive(Default)]
struct Phases {
    read: Duration,
    checksum: Duration,
    upload: Duration,
}

/// Reads `len` bytes into `buf` (reused) and feeds them to `hash`, timing both into `t`; `len` is
/// bounded by the caller.
fn take(
    r: &mut dyn Read,
    hash: &mut Sha256,
    buf: &mut Vec<u8>,
    len: usize,
    t: &mut Phases,
) -> Result<(), PinError> {
    buf.clear();
    buf.try_reserve_exact(len).map_err(|_| PinError::NoRoom)?;
    buf.resize(len, 0);
    let began = Instant::now();
    r.read_exact(buf)?;
    let read = Instant::now();
    hash.update(&buf[..]);
    t.read += read - began;
    t.checksum += read.elapsed();
    Ok(())
}

/// Reads the 32 bytes that must equal `hash`'s digest.
fn check(r: &mut dyn Read, hash: Sha256) -> Result<(), PinError> {
    let mut want = [0; 32];
    r.read_exact(&mut want)?;
    if <[u8; 32]>::from(hash.finalize()) == want {
        Ok(())
    } else {
        Err(PinError::Miss)
    }
}

/// Appends pin `id`'s segments from token `from` (0, or what an earlier call returned) up to its
/// last exportable token to `w`, with the header first when `from` is 0; returns where it ended.
/// Only the new segments' KV leave the device. Nothing new returns `from`; nothing at all is `Miss`.
pub(crate) fn export(ops: Ops, id: u64, from: u32, w: &mut dyn Write) -> Result<u32, PinError> {
    let PinReply::Info(info) = ops(PinOp::Info)? else {
        unreachable!("info")
    };
    let from = from as usize;
    let PinReply::Chain { count, ids, .. } = ops(PinOp::Chain {
        id,
        from,
        state: false,
    })?
    else {
        unreachable!("chain")
    };
    if count <= from {
        return if from == 0 {
            Err(PinError::Miss)
        } else {
            Ok(from as u32)
        };
    }
    let seg = info.segment as usize;
    if !from.is_multiple_of(seg) {
        return Err(PinError::Miss);
    }
    if from == 0 {
        let head = header(&info);
        w.write_all(&head)?;
        w.write_all(&sha(&head))?;
    }
    let mut at = from;
    while at < count {
        let end = (at + seg).min(count);
        let mut hash = Sha256::new();
        put(w, &mut hash, &((end - at) as u32).to_le_bytes())?;
        let bytes: Vec<u8> = ids[at - from..end - from]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        put(w, &mut hash, &bytes)?;
        for layer in 0..info.layers {
            let PinReply::Bytes(b) = ops(PinOp::Layer {
                id,
                from: at,
                to: end,
                layer,
            })?
            else {
                unreachable!("layer")
            };
            put(w, &mut hash, &b)?;
        }
        w.write_all(&hash.finalize())?;
        at = end;
    }
    Ok(count as u32)
}

/// Writes pin `id`'s state sidecar: the state after the last token a hybrid checkpoint holds (a
/// dense model's is empty, over its whole chain). Returns the tokens it covers. The pin file must
/// hold at least that many before the sidecar is put in place; if [`export`] ended short of it, append
/// again first.
pub(crate) fn export_state(ops: Ops, id: u64, w: &mut dyn Write) -> Result<u32, PinError> {
    let PinReply::Info(info) = ops(PinOp::Info)? else {
        unreachable!("info")
    };
    let PinReply::Chain { count, state, .. } = ops(PinOp::Chain {
        id,
        from: usize::MAX,
        state: true,
    })?
    else {
        unreachable!("chain")
    };
    if count == 0 || state.len() as u64 != info.state_bytes {
        return Err(PinError::Miss);
    }
    let mut hash = Sha256::new();
    put(w, &mut hash, STATE_MAGIC)?;
    put(w, &mut hash, &VERSION.to_le_bytes())?;
    put(w, &mut hash, &sha(&header(&info)))?;
    put(w, &mut hash, &(count as u64).to_le_bytes())?;
    put(w, &mut hash, &info.state_bytes.to_le_bytes())?;
    put(w, &mut hash, &state)?;
    w.write_all(&hash.finalize())?;
    Ok(count as u32)
}

fn le<const N: usize>(b: &[u8], at: &mut usize) -> [u8; N] {
    let out = b[*at..*at + N].try_into().expect("sized");
    *at += N;
    out
}

/// Loads the pin file `pin` and its sidecar `state` into the cache of `namespace` under `id` (a
/// new one when None) and pins `tokens` up to what the sidecar covers; returns the id and the
/// tokens pinned. `prepare` runs once the files have passed their header checks and before any
/// block is taken (the documents plan's reload). The files must be this engine's: model file,
/// build, ABI, KV layout, dtype, state size, segment length, no image rows, every checksum good and the ids a
/// prefix of `tokens`; else `Miss` with nothing published. A torn record ends a dense model's
/// import there; a hybrid's state belongs to the full count, so it is `Miss`. Every size is
/// checked against the engine before a byte goes to the device. Memory: one layer of a segment,
/// and the engine's staging of half of it.
pub(crate) fn import(
    ops: Ops,
    id: Option<u64>,
    tokens: &[u32],
    namespace: u64,
    pin: &mut dyn Read,
    state: &mut dyn Read,
    prepare: &mut dyn FnMut() -> Result<(), EngineError>,
) -> Result<(u64, usize), PinError> {
    let PinReply::Info(info) = ops(PinOp::Info)? else {
        unreachable!("info")
    };
    let (began, mut t) = (Instant::now(), Phases::default());
    let mut head = [0u8; HEAD_LEN + 32];
    pin.read_exact(&mut head)?;
    if sha(&head[..HEAD_LEN]) != head[HEAD_LEN..] || head[..HEAD_LEN] != header(&info)[..] {
        return Err(PinError::Miss);
    }
    // The header named this engine's segment, a multiple of the block.
    let (seg, bt) = (info.segment as usize, info.block_tokens as usize);
    // The sidecar: its sizes against the engine's before anything is allocated from them.
    let mut hash = Sha256::new();
    let mut buf = Vec::new();
    take(state, &mut hash, &mut buf, 8 + 4 + 32 + 8 + 8, &mut t)?;
    let mut at = 0;
    let magic: [u8; 8] = le(&buf, &mut at);
    let version = u32::from_le_bytes(le(&buf, &mut at));
    let head_sha: [u8; 32] = le(&buf, &mut at);
    let covered = u64::from_le_bytes(le(&buf, &mut at));
    let state_len = u64::from_le_bytes(le(&buf, &mut at));
    if &magic != STATE_MAGIC
        || version != VERSION
        || head_sha != head[HEAD_LEN..]
        || state_len != info.state_bytes
        || covered == 0
        || covered % bt as u64 != 0
        || covered > tokens.len() as u64
    {
        return Err(PinError::Miss);
    }
    let covered = covered as usize;
    take(state, &mut hash, &mut buf, state_len as usize, &mut t)?;
    check(state, hash)?;
    let state_bytes = std::mem::take(&mut buf);
    let reload = Instant::now();
    prepare()?;
    eprintln!("larch_pin_import,reload,{}", reload.elapsed().as_millis());
    let PinReply::Id(id) = ops(PinOp::Begin {
        id,
        namespace,
        tokens: tokens[..covered].to_vec(),
    })?
    else {
        unreachable!("begin")
    };
    // Whatever ends the read, an error or a reader that panics, frees the blocks taken.
    let mut job = Importing {
        ops,
        id,
        live: true,
    };
    let done = read_segments(
        &mut *job.ops,
        id,
        tokens,
        covered,
        seg,
        &info,
        pin,
        info.state_bytes == 0,
        &mut t,
    );
    let ended = done.and_then(|done| {
        // Finish takes the job, finished or refused.
        job.live = false;
        (job.ops)(PinOp::Finish(id, state_bytes, done))?;
        Ok((id, done))
    });
    let phases = [
        ("read", t.read),
        ("checksum", t.checksum),
        ("upload", t.upload),
    ];
    for (phase, d) in phases.into_iter().chain([("total", began.elapsed())]) {
        eprintln!("larch_pin_import,{phase},{}", d.as_millis());
    }
    ended
}

/// An import's blocks, taken by [`PinOp::Begin`], until it is finished.
struct Importing<'a> {
    ops: Ops<'a>,
    id: u64,
    live: bool,
}

impl Drop for Importing<'_> {
    fn drop(&mut self) {
        if self.live {
            // The engine may be gone (a failed engine answers nothing): nothing to free then.
            let _ = (self.ops)(PinOp::Abort(self.id));
        }
    }
}

/// Reads and uploads the records covering `covered` tokens; returns the tokens loaded, short of
/// `covered` only when `torn_ok` and a record after the first is not whole.
#[allow(clippy::too_many_arguments)]
fn read_segments(
    ops: Ops,
    id: u64,
    tokens: &[u32],
    covered: usize,
    seg: usize,
    info: &PinInfo,
    r: &mut dyn Read,
    torn_ok: bool,
    t: &mut Phases,
) -> Result<usize, PinError> {
    let bt = info.block_tokens as usize;
    let mut at = 0;
    let mut buf = Vec::new();
    while at < covered {
        let n = seg.min(covered - at);
        match read_record(ops, id, &tokens[at..at + n], at / bt, info, r, &mut buf, t) {
            Ok(()) => at += n,
            Err(PinError::Miss) if torn_ok && at > 0 => break,
            Err(e) => return Err(e),
        }
    }
    Ok(at)
}

/// One record: its count, ids and checksum checked, its layers uploaded from block `from_block`.
#[allow(clippy::too_many_arguments)]
fn read_record(
    ops: Ops,
    id: u64,
    want: &[u32],
    from_block: usize,
    info: &PinInfo,
    r: &mut dyn Read,
    buf: &mut Vec<u8>,
    t: &mut Phases,
) -> Result<(), PinError> {
    let mut hash = Sha256::new();
    take(r, &mut hash, buf, 4, t)?;
    if u32::from_le_bytes(buf[..4].try_into().expect("4")) as usize != want.len() {
        return Err(PinError::Miss);
    }
    let mut seen = 0;
    while seen < want.len() {
        let n = (STEP / 4).min(want.len() - seen);
        take(r, &mut hash, buf, n * 4, t)?;
        let same = (buf.chunks_exact(4))
            .zip(&want[seen..])
            .all(|(c, t)| c == t.to_le_bytes());
        if !same {
            return Err(PinError::Miss);
        }
        seen += n;
    }
    let layer_len = (want.len() / info.block_tokens as usize)
        .checked_mul(2 * info.block_bytes as usize)
        .ok_or(PinError::Miss)?;
    for layer in 0..info.layers {
        take(r, &mut hash, buf, layer_len, t)?;
        let began = Instant::now();
        ops(PinOp::Upload {
            id,
            layer,
            from_block,
            bytes: std::mem::take(buf),
        })?;
        t.upload += began.elapsed();
    }
    check(r, hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory the host cannot give is no room, not a bad file: the caller keeps the file.
    #[test]
    fn a_read_the_host_cannot_hold_is_no_room() {
        let (mut hash, mut buf) = (Sha256::new(), Vec::new());
        let got = take(
            &mut io::empty(),
            &mut hash,
            &mut buf,
            usize::MAX,
            &mut Phases::default(),
        );
        assert!(matches!(got, Err(PinError::NoRoom)));
        let full = EngineError::InvalidRequest(invalid::PINS_FULL);
        assert!(matches!(PinError::from(full), PinError::NoRoom));
        let bad = EngineError::InvalidRequest(invalid::PIN_CHANGED);
        assert!(matches!(PinError::from(bad), PinError::Miss));
        // A reload broke the import off: the file is good, not a miss to delete (#591 review).
        let torn = EngineError::InvalidRequest(invalid::IMPORT_TORN);
        assert!(matches!(
            PinError::from(torn),
            PinError::Engine(EngineError::InvalidRequest(invalid::IMPORT_TORN))
        ));
    }

    /// A `.pin` from another build (same version, other commit) is a `Miss`, nothing read past its
    /// header.
    #[test]
    fn a_pin_file_from_another_build_is_a_miss() {
        let info = |build| PinInfo {
            file: 1,
            layers: 1,
            block_bytes: 8,
            dtype: 1,
            state_bytes: 0,
            block_tokens: 16,
            segment: 16,
            abi: 1,
            version: "0.0.0",
            build,
        };
        let (a, b) = (info("aaaaaaa"), info("bbbbbbb"));
        assert_eq!(header(&a), header(&info("aaaaaaa")));
        assert_ne!(header(&a), header(&b));
        let mut file = header(&a);
        file.extend(sha(&file));
        let mut ops = |op: PinOp| match op {
            PinOp::Info => Ok(PinReply::Info(b.clone())),
            _ => unreachable!("nothing past the header"),
        };
        let got = import(
            &mut ops,
            None,
            &[0; 16],
            0,
            &mut &file[..],
            &mut io::empty(),
            &mut || Ok(()),
        );
        assert!(matches!(got, Err(PinError::Miss)));
    }

    /// A reader that panics, or errs, after the blocks were taken frees them: an `Abort` runs.
    #[test]
    fn an_import_whose_reader_fails_aborts() {
        struct Panics;
        impl Read for Panics {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                panic!("the reader failed")
            }
        }
        let info = PinInfo {
            file: 1,
            layers: 1,
            block_bytes: 8,
            dtype: 1,
            state_bytes: 0,
            block_tokens: 16,
            segment: 16,
            abi: 1,
            version: "0.0.0",
            build: "aaaaaaa",
        };
        let head = header(&info);
        let mut file = head.clone();
        file.extend(sha(&head));
        let mut side = Vec::new();
        side.extend(STATE_MAGIC);
        side.extend(VERSION.to_le_bytes());
        side.extend(sha(&head));
        side.extend(16u64.to_le_bytes());
        side.extend(0u64.to_le_bytes());
        side.extend(sha(&side));
        for panics in [true, false] {
            let mut seen = Vec::new();
            let mut ops = |op: PinOp| {
                seen.push(op.name());
                match op {
                    PinOp::Info => Ok(PinReply::Info(info.clone())),
                    PinOp::Begin { .. } => Ok(PinReply::Id(7)),
                    _ => Ok(PinReply::Done),
                }
            };
            let pin: &mut dyn Read = &mut (&file[..]).chain(Panics);
            let short: &mut dyn Read = &mut &file[..];
            let reader = if panics { pin } else { short };
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                import(
                    &mut ops,
                    None,
                    &[0; 16],
                    0,
                    reader,
                    &mut &side[..],
                    &mut || Ok(()),
                )
            }));
            assert_eq!(got.is_err(), panics);
            if !panics {
                // The file ends after its header: a torn first record is a Miss.
                assert!(matches!(got, Ok(Err(PinError::Miss))));
            }
            assert_eq!(seen, ["info", "begin", "abort"], "panics: {panics}");
        }
    }
}
