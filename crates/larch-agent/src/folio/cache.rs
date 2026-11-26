//! The disk cache of pinned documents, `<data>/doc-cache/<key>.pin` with its `<key>.state` sidecar:
//! file names, the budget and its least-recently-used eviction, and the file lifecycle around the
//! engine's `export_pin` and `import_pin`, which own the format and refuse a file that does not
//! match (`PinError::Miss`). A pin is append-only: each export appends a segment to the open file
//! and replaces only the small state sidecar (by temp file and rename); a torn trailing segment is
//! ignored by the engine. Only files here are ever deleted; a user's documents never are.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

/// Tokens between exports while a document is first read: one state checkpoint each.
pub const EXPORT_EVERY: u32 = 8192;

/// The default budget: 8 GiB.
const DEFAULT_BUDGET: u64 = 8 << 30;

/// Free space left alone on the data volume: the budget stops this short of the volume's end, and
/// no segment is appended below it.
pub const MIN_FREE: u64 = 2 << 30;

/// Why a cache operation failed: the disk, or the caller's own write or import.
#[derive(Debug)]
pub enum Failed<E> {
    /// The cache folder or file could not be used.
    Io(io::Error),
    /// The closure the caller passed failed.
    Engine(E),
}

/// The cache folder of `data`.
pub fn dir(data: &Path) -> PathBuf {
    data.join("doc-cache")
}

/// The file name of a pin: the hash of the document's token ids and the model's path, so another
/// model or another text never finds it. A partial read is kept under the whole document's key.
pub fn key(tokens: &[u32], model: &Path) -> String {
    let mut bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    bytes.extend(model.as_os_str().as_encoded_bytes());
    format!("{}.pin", larch::bench::hex(&larch::bench::sha256(&bytes)))
}

/// The read rate file of `model`: the tokens and seconds of its last first read (`keep_read_rate`).
fn rate_file(dir: &Path, model: &Path) -> PathBuf {
    let hash = larch::bench::sha256(model.as_os_str().as_encoded_bytes());
    dir.join(format!("{}.rate", larch::bench::hex(&hash)))
}

/// Tokens a second `model` read its last first read at, for `folio::in_time`; `None` before one,
/// or for a torn file.
pub fn read_rate(dir: &Path, model: &Path) -> Option<f64> {
    let text = fs::read_to_string(rate_file(dir, model)).ok()?;
    let (tokens, secs) = text.split_once(' ')?;
    let (tokens, secs): (f64, f64) = (tokens.parse().ok()?, secs.parse().ok()?);
    (tokens > 0.0 && secs > 0.0).then(|| tokens / secs)
}

/// Records that `model` read `tokens` in `secs`, by temp file and rename.
pub fn keep_read_rate(dir: &Path, model: &Path, tokens: u32, secs: f64) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    write!(tmp, "{tokens} {secs}")?;
    tmp.persist(rate_file(dir, model))
        .map(drop)
        .map_err(|e| e.error)
}

/// The bytes the cache may hold: `setting` or 8 GiB, but never more than the volume's `free` bytes
/// less `MIN_FREE`.
pub fn budget(setting: Option<u64>, free: u64) -> u64 {
    setting
        .unwrap_or(DEFAULT_BUDGET)
        .min(free.saturating_sub(MIN_FREE))
}

/// The free bytes of the volume `dir` is on.
pub fn free(dir: &Path) -> io::Result<u64> {
    fs2::available_space(dir)
}

/// Whether a first read that has done `done` of `total` tokens, last exported at `exported`,
/// exports now: at each multiple of `EXPORT_EVERY` passed, at the end, and when the read `stopped`
/// with something new to keep.
pub fn export_due(exported: u32, done: u32, total: u32, stopped: bool) -> bool {
    done > exported && (stopped || done >= total || done / EXPORT_EVERY > exported / EXPORT_EVERY)
}

/// The state sidecar of the pin `key`.
fn state(dir: &Path, key: &str) -> PathBuf {
    dir.join(key).with_extension("state")
}

/// Appends a segment to the pin `key` through `write` (the engine's `export_pin`), which writes
/// the new segment at the end of the open file and gives the new state; the state sidecar is then
/// replaced by temp file and rename. The pin file itself is never rewritten: a failed or killed
/// append leaves a torn trailing segment the engine ignores, and the old state. With no room (the
/// volume's free space at `MIN_FREE` or less) nothing is written and this is `Ok(false)`: the
/// document stays in memory only, and the read goes on.
pub fn append<E>(
    dir: &Path,
    key: &str,
    write: impl FnOnce(&mut File) -> Result<Vec<u8>, E>,
) -> Result<bool, Failed<E>> {
    fs::create_dir_all(dir).map_err(Failed::Io)?;
    append_with(dir, key, free(dir).map_err(Failed::Io)?, write)
}

/// `append` on a volume with `free` bytes free.
fn append_with<E>(
    dir: &Path,
    key: &str,
    free: u64,
    write: impl FnOnce(&mut File) -> Result<Vec<u8>, E>,
) -> Result<bool, Failed<E>> {
    if free <= MIN_FREE {
        return Ok(false);
    }
    let mut file = File::options()
        .create(true)
        .append(true)
        .open(dir.join(key))
        .map_err(Failed::Io)?;
    let new = write(&mut file).map_err(Failed::Engine)?;
    file.sync_all().map_err(Failed::Io)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir).map_err(Failed::Io)?;
    tmp.write_all(&new).map_err(Failed::Io)?;
    tmp.as_file().sync_all().map_err(Failed::Io)?;
    tmp.persist(state(dir, key))
        .map_err(|e| Failed::Io(e.error))?;
    Ok(true)
}

/// Reads the pin `key` through `import` (the engine's `import_pin`), which gets the file and its
/// state sidecar (`None` when there is none) and says `Ok(None)` for a file it refuses: a corrupt
/// or mismatched file is deleted with its state and read as a miss, and the document is read
/// again. A hit counts as a use for eviction.
pub fn load<T, E>(
    dir: &Path,
    key: &str,
    import: impl FnOnce(&mut File, Option<&[u8]>) -> Result<Option<T>, E>,
) -> Result<Option<T>, Failed<E>> {
    let path = dir.join(key);
    // Open to write as well: a hit sets the file's time, which Windows refuses through a handle
    // opened only to read ("Access is denied"). `source::touch` opens its blobs the same way.
    let mut file = match File::options().read(true).write(true).open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Failed::Io(e)),
    };
    let side = fs::read(state(dir, key)).ok();
    match import(&mut file, side.as_deref()).map_err(Failed::Engine)? {
        Some(hit) => {
            file.set_modified(SystemTime::now()).map_err(Failed::Io)?;
            Ok(Some(hit))
        }
        None => {
            drop(file);
            let _ = fs::remove_file(state(dir, key));
            fs::remove_file(&path).map_err(Failed::Io)?;
            Ok(None)
        }
    }
}

/// The documents plan chat `chat` last held its pin on (`<chat>.plan` here), which a restart
/// that reopens the chat loads straight into ([`larch_engine::EngineOptions::documents`]): one
/// load instead of the default plan's and then the documents reload's. With it, the context the
/// chat routed its documents against then, which it routes against again on that plan: the plan
/// made room for the set, which must still be pinned, not taken as plain text.
pub fn plan(dir: &Path, chat: &str) -> Option<(larch_engine::DocumentsPlan, u32)> {
    let text = fs::read_to_string(dir.join(format!("{chat}.plan"))).ok()?;
    let n: Vec<u32> = text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    // A torn or foreign record is no plan: the restart loads as before and reloads for the pin.
    let &[context, seats, route] = &n[..] else {
        return None;
    };
    let plan = larch_engine::DocumentsPlan {
        max_context: context,
        seats,
    };
    (seats > 0 && route > 0 && route <= context).then_some((plan, route))
}

/// Records `plan` and its routing context as chat `chat`'s ([`plan`]) by temp file and rename, or
/// forgets them when the chat holds no pin. Unchanged, nothing is written.
pub fn keep_plan(
    dir: &Path,
    chat: &str,
    plan: Option<(larch_engine::DocumentsPlan, u32)>,
) -> io::Result<()> {
    let path = dir.join(format!("{chat}.plan"));
    match plan {
        Some(p) if self::plan(dir, chat) == Some(p) => Ok(()),
        Some((p, route)) => {
            fs::create_dir_all(dir)?;
            let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
            write!(tmp, "{} {} {route}", p.max_context, p.seats)?;
            tmp.persist(path).map(drop).map_err(|e| e.error)
        }
        None => match fs::remove_file(path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        },
    }
}

/// Records which documents (`folio::sha`) the pin `key` was read from, in `<key>.docs`, so
/// that deleting a document can find the pin (`forget`).
pub fn tag(dir: &Path, key: &str, hashes: &[String]) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join(key).with_extension("docs"), hashes.join("\n"))
}

/// Deletes every pin tagged with one of `hashes`, and its tag. Returns how many pins went.
pub fn forget(dir: &Path, hashes: &[String]) -> io::Result<usize> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(0);
    };
    let mut gone = 0;
    for entry in entries {
        let tag = entry?.path();
        if tag.extension().is_none_or(|e| e != "docs") {
            continue;
        }
        let tagged = fs::read_to_string(&tag)?;
        if tagged.lines().any(|h| hashes.iter().any(|g| g == h)) {
            let _ = fs::remove_file(tag.with_extension("state"));
            match fs::remove_file(tag.with_extension("pin")) {
                Ok(()) => gone += 1,
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                _ => {}
            }
            fs::remove_file(&tag)?;
        }
    }
    Ok(gone)
}

/// Deletes the least recently used pins until those left take at most `budget` bytes; returns
/// how many went.
pub fn evict(dir: &Path, budget: u64) -> io::Result<usize> {
    let mut pins = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() && entry.path().extension().is_some_and(|e| e == "pin") {
            pins.push((meta.modified()?, meta.len(), entry.path()));
        }
    }
    pins.sort();
    let mut total: u64 = pins.iter().map(|p| p.1).sum();
    let mut gone = 0;
    for (_, len, path) in pins {
        if total <= budget {
            break;
        }
        let _ = fs::remove_file(path.with_extension("docs"));
        let _ = fs::remove_file(path.with_extension("state"));
        fs::remove_file(path)?;
        total -= len;
        gone += 1;
    }
    Ok(gone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A read rate comes back per model, the last one kept wins, and a torn file is no rate.
    #[test]
    fn a_read_rate_is_the_models_last() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (Path::new("/m/a.gguf"), Path::new("/m/b.gguf"));
        assert_eq!(read_rate(dir.path(), a), None);
        keep_read_rate(dir.path(), a, 10_000, 4.0).unwrap();
        keep_read_rate(dir.path(), a, 12_000, 4.0).unwrap();
        assert_eq!(read_rate(dir.path(), a), Some(3000.0));
        assert_eq!(read_rate(dir.path(), b), None);
        fs::write(rate_file(dir.path(), b), "12000").unwrap();
        assert_eq!(read_rate(dir.path(), b), None);
    }

    /// A chat's plan record comes back as written; a torn or foreign one is no plan; and
    /// forgetting it leaves none.
    #[test]
    fn a_plan_record_round_trips_and_a_torn_one_is_no_plan() {
        let dir = tempfile::tempdir().unwrap();
        let p = larch_engine::DocumentsPlan {
            max_context: 131_072,
            seats: 2,
        };
        keep_plan(dir.path(), "c", Some((p, 32_768))).unwrap();
        assert_eq!(plan(dir.path(), "c"), Some((p, 32_768)));
        for torn in [
            "131072 2",
            "131072 2 327 9",
            "131072 0 32768",
            "4096 2 32768",
            "x 2 3",
        ] {
            fs::write(dir.path().join("c.plan"), torn).unwrap();
            assert_eq!(plan(dir.path(), "c"), None, "{torn}");
        }
        keep_plan(dir.path(), "c", None).unwrap();
        assert!(!dir.path().join("c.plan").exists());
        keep_plan(dir.path(), "c", None).unwrap();
    }

    /// A deleted chat leaves no plan record behind (`source::forget_chat`), another chat's stays.
    #[test]
    fn a_deleted_chat_takes_its_plan_record() {
        let data = tempfile::tempdir().unwrap();
        let p = larch_engine::DocumentsPlan {
            max_context: 32_768,
            seats: 2,
        };
        for chat in ["c", "d"] {
            keep_plan(&dir(data.path()), chat, Some((p, 8192))).unwrap();
        }
        super::super::source::forget_chat(data.path(), "c").unwrap();
        assert_eq!(plan(&dir(data.path()), "c"), None);
        assert_eq!(plan(&dir(data.path()), "d"), Some((p, 8192)));
    }

    /// A deleted chat takes the pin files and PDF bytes of the links it read, but not those of a
    /// link another chat read too (#590 F10).
    #[test]
    fn a_deleted_chat_takes_its_links_pins_and_bytes_but_not_a_shared_one() {
        use super::super::source::{self, Entry};
        let data = tempfile::tempdir().unwrap();
        let link = |source: &str, sha: &str| Entry {
            letter: String::new(),
            name: source.into(),
            source: source.into(),
            sha256: sha.into(),
            reps: Vec::new(),
            pages: 1,
            stamp: None,
        };
        source::add(data.path(), "c", link("https://a.org/own.pdf", "own")).unwrap();
        source::add(data.path(), "c", link("https://a.org/both.pdf", "both")).unwrap();
        source::add(data.path(), "d", link("https://a.org/both.pdf", "both")).unwrap();
        let d = dir(data.path());
        fs::create_dir_all(&d).unwrap();
        for (key, sha) in [("k1", "own"), ("k2", "both")] {
            fs::write(d.join(key).with_extension("pin"), b"pin").unwrap();
            tag(&d, key, &[sha.to_owned()]).unwrap();
            source::keep(data.path(), sha, b"%PDF").unwrap();
        }
        source::forget_chat(data.path(), "c").unwrap();
        assert!(!d.join("k1.pin").exists() && !source::blob(data.path(), "own").exists());
        assert!(d.join("k2.pin").exists() && source::blob(data.path(), "both").exists());
    }

    fn put(dir: &Path, name: &str, len: usize, age: u64) {
        let p = dir.join(name);
        fs::write(&p, vec![0; len]).unwrap();
        let t = SystemTime::now() - Duration::from_secs(age);
        File::options()
            .write(true)
            .open(p)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn a_hit_reads_the_file_and_its_state_and_a_miss_finds_none() {
        let d = tempfile::tempdir().unwrap();
        let wrote = |w: &mut File| w.write_all(b"pin").map(|()| b"state".to_vec());
        assert!(append_with(d.path(), "k.pin", 3 << 30, wrote).unwrap());
        let read = |f: &mut File, s: Option<&[u8]>| {
            io::read_to_string(f).map(|t| Some((t, s.map(<[u8]>::to_vec))))
        };
        assert_eq!(
            load(d.path(), "k.pin", read).unwrap(),
            Some(("pin".to_owned(), Some(b"state".to_vec())))
        );
        assert!(load(d.path(), "other.pin", read).unwrap().is_none());
    }

    #[test]
    fn a_second_export_appends_and_only_the_state_is_replaced() {
        let d = tempfile::tempdir().unwrap();
        let seg = |name: &'static [u8], state: &'static [u8]| {
            move |w: &mut File| w.write_all(name).map(|()| state.to_vec())
        };
        append_with(d.path(), "k.pin", 3 << 30, seg(b"one", b"s1")).unwrap();
        let first = fs::metadata(d.path().join("k.pin")).unwrap().len();
        append_with(d.path(), "k.pin", 3 << 30, seg(b"two", b"s2")).unwrap();
        // The pin grew by the new segment and keeps the old bytes; the state is the latest one.
        assert_eq!(
            (first, fs::read(d.path().join("k.pin")).unwrap()),
            (3, b"onetwo".to_vec())
        );
        assert_eq!(fs::read(d.path().join("k.state")).unwrap(), b"s2");
        // Nothing but the pin and its state is left beside them (no temp file).
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), 2);
    }

    #[test]
    fn a_failed_append_keeps_the_old_state_and_with_no_room_nothing_is_written() {
        let d = tempfile::tempdir().unwrap();
        let ok = |w: &mut File| w.write_all(b"one").map(|()| b"s1".to_vec());
        append_with(d.path(), "k.pin", 3 << 30, ok).unwrap();
        // A torn segment: the write failed half way; the state still names the first segment.
        let r = append_with(d.path(), "k.pin", 3 << 30, |w: &mut File| {
            w.write_all(b"to").unwrap();
            Err("disk")
        });
        assert!(matches!(r, Err(Failed::Engine("disk"))));
        assert_eq!(fs::read(d.path().join("k.state")).unwrap(), b"s1");
        // Two GiB free or less: skipped, the read goes on, nothing is written.
        let no_room = append_with(d.path(), "j.pin", MIN_FREE, |_: &mut File| {
            Ok::<_, ()>(Vec::new())
        });
        assert!(matches!(no_room, Ok(false)));
        assert!(!d.path().join("j.pin").exists());
    }

    #[test]
    fn eviction_takes_the_least_recently_used_first() {
        let d = tempfile::tempdir().unwrap();
        put(d.path(), "old.pin", 10, 300);
        put(d.path(), "mid.pin", 10, 200);
        put(d.path(), "new.pin", 10, 100);
        fs::write(d.path().join("old.state"), "s").unwrap();
        fs::write(d.path().join("note.txt"), "not a pin").unwrap();
        assert_eq!(evict(d.path(), 20).unwrap(), 1);
        assert!(!d.path().join("old.pin").exists() && d.path().join("mid.pin").exists());
        assert!(!d.path().join("old.state").exists());
        // A hit makes the oldest the newest.
        load::<_, ()>(d.path(), "mid.pin", |_, _| Ok(Some(()))).unwrap();
        put(d.path(), "late.pin", 10, 0);
        assert_eq!(evict(d.path(), 20).unwrap(), 1);
        assert!(!d.path().join("new.pin").exists() && d.path().join("note.txt").exists());
    }

    #[test]
    fn a_refused_file_is_deleted_and_read_as_a_miss() {
        let d = tempfile::tempdir().unwrap();
        put(d.path(), "k.pin", 4, 0);
        assert!(
            load::<(), ()>(d.path(), "k.pin", |_, _| Ok(None))
                .unwrap()
                .is_none()
        );
        assert!(!d.path().join("k.pin").exists());
    }

    #[test]
    fn the_budget_is_the_setting_or_8_gib_but_two_gib_short_of_a_full_volume() {
        assert_eq!(budget(None, 100 << 30), 8 << 30);
        assert_eq!(budget(None, 6 << 30), 4 << 30);
        assert_eq!(budget(Some(5), 200 << 30), 5);
        assert_eq!(budget(Some(50 << 30), 10 << 30), 8 << 30);
        assert_eq!(budget(None, 1 << 30), 0);
    }

    #[test]
    fn a_first_read_exports_every_8192_tokens_at_the_end_and_on_stop() {
        let due: Vec<u32> = (1..=20_000u32)
            .step_by(512)
            .scan(0, |last, done| {
                let d = export_due(*last, done, 20_000, false);
                if d {
                    *last = done;
                }
                Some(d.then_some(done))
            })
            .flatten()
            .collect();
        assert_eq!(due, vec![8193, 16385]);
        assert!(export_due(16385, 20_000, 20_000, false));
        assert!(!export_due(20_000, 20_000, 20_000, false));
        // A stop with something new since the last export keeps it.
        assert!(export_due(8193, 9000, 20_000, true) && !export_due(9000, 9000, 20_000, true));
    }
}
