//! "Faster on this computer": the faster copy of a model the GPU holds only in part
//! ([`larch::faster`]), made once in the background while chats keep running on the original,
//! switched to when it is ready, and switched back from on the same card.
//!
//! The copy lives in `<data>/faster`, which no model list reads: the card of the original offers
//! it and says which of the two runs. A copy is used only after it loaded: a load that fails
//! deletes it and loads the original again.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, PoisonError};
use std::time::{Duration, Instant};

use larch::faster::{self, Stage};

use crate::contract::{Faster, FasterDone, FasterFailed, FasterProgress};
use crate::{App, Event, Inner};

/// How often a build says how far it got.
const TICK: Duration = Duration::from_millis(250);

/// The folder the copies go to, under the app's data.
const DIR: &str = "faster";

impl Inner {
    /// Where the copy of `src` goes.
    fn faster_path(&self, src: &Path) -> PathBuf {
        let name = src.file_stem().unwrap_or_default().to_string_lossy();
        self.data.join(DIR).join(format!("{name}.faster.gguf"))
    }

    /// The card's offer for `src` of `size_bytes`: a CUDA build (where the copy's 2-bit experts
    /// load and it was measured faster) on a GPU that cannot hold the whole original, and a file
    /// [`larch::faster`] carries a histogram for.
    // ponytail: measured on one 16 GB sm_120 card (bench/results/35b-vs-llamacpp-2026-11-25.md);
    // narrow it by card if a smaller or older GPU measures the copy slower.
    pub(crate) fn faster_offer(
        &self,
        src: &Path,
        size_bytes: u64,
        gpu_total: Option<u64>,
    ) -> Option<Faster> {
        let here = larch_kernels::BACKEND == "cuda" && gpu_total.is_some_and(|t| t < size_bytes);
        (here && faster::known(src).is_some()).then(|| {
            let copy = self.faster_path(src);
            Faster {
                copy: copy.is_file().then(|| copy.to_string_lossy().into_owned()),
                building: (self.building.lock().unwrap_or_else(PoisonError::into_inner))
                    .iter()
                    .any(|(p, _)| p == src),
            }
        })
    }
}

impl App {
    /// The faster version offered for the model at `path` on this computer, as its listing shows
    /// it; None where there is none.
    pub fn faster_for(&self, path: &str) -> Option<Faster> {
        let size = std::fs::metadata(path).ok()?.len();
        let gpu = self.inner.gpu().as_ref().map(|g| g.1);
        self.inner.faster_offer(Path::new(path), size, gpu)
    }

    /// Starts making the faster copy of the model at `path` on a thread of its own; its progress
    /// and end arrive as `faster:*` events. Chats keep running on the loaded model meanwhile.
    pub fn build_faster(&self, path: &str) -> Result<(), String> {
        let src = PathBuf::from(path);
        if faster::known(&src).is_none() {
            return Err(faster::Error::NotOffered.to_string());
        }
        let stop = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = (self.inner.building.lock()).unwrap_or_else(PoisonError::into_inner);
            if jobs.iter().any(|(p, _)| *p == src) {
                return Err("The faster version is already being made.".into());
            }
            jobs.push((src.clone(), stop.clone()));
        }
        let dst = self.inner.faster_path(&src);
        let inner = self.inner.clone();
        let spawned = std::thread::Builder::new()
            .name("larch-faster".into())
            .spawn(move || inner.build_faster(&src, &dst, &stop));
        if let Err(e) = spawned {
            self.inner.end_build(Path::new(path));
            return Err(format!("The faster version could not start: {e}"));
        }
        Ok(())
    }

    /// Stops the faster copy of `path` being made; `faster:failed` (cancelled) follows.
    pub fn cancel_faster(&self, path: &str) -> Result<(), String> {
        let jobs = (self.inner.building.lock()).unwrap_or_else(PoisonError::into_inner);
        let (_, stop) = (jobs.iter().find(|(p, _)| p == Path::new(path)))
            .ok_or("The faster version is not being made.")?;
        stop.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Loads the faster copy of the model at `path` (`on`) or the original. A copy that fails to
    /// load is deleted and the original loaded again; the error says so.
    pub fn use_faster(&self, path: &str, on: bool) -> Result<(), String> {
        if !on {
            return self.load_model(path);
        }
        let copy = self.inner.faster_path(Path::new(path));
        if !copy.is_file() {
            return Err("The faster version is not made yet.".into());
        }
        self.switch_or_back(path, &copy)
    }

    /// Loads `copy` in place of the original at `path`, or, when it fails, deletes it and loads
    /// the original again.
    pub(crate) fn switch_or_back(&self, path: &str, copy: &Path) -> Result<(), String> {
        let Err(why) = self.load_model(&copy.to_string_lossy()) else {
            return Ok(());
        };
        // A refusal before the old model was unloaded (a reply being written) leaves it running,
        // and the copy may well load next time.
        if self.app_state().model.as_deref() == Some(path) {
            return Err(why);
        }
        let _ = std::fs::remove_file(copy);
        let back = self.load_model(path);
        Err(match back {
            Ok(()) => format!(
                "The faster version did not load, so Larch went back to the original. ({why})"
            ),
            Err(e) => {
                format!("The faster version did not load ({why}), nor did the original ({e}).")
            }
        })
    }
}

impl Inner {
    /// One build, on its own thread, ending in `faster:done` or `faster:failed`.
    fn build_faster(&self, src: &Path, dst: &Path, stop: &AtomicBool) {
        let path = src.to_string_lossy().into_owned();
        let started = Instant::now();
        let mut last = Instant::now() - TICK;
        let mut progress = |stage: Stage, done: u64, total: u64| {
            if last.elapsed() >= TICK {
                last = Instant::now();
                // Hashing reads the source once and writing reads it again: the two stages as one
                // bar, each half of it.
                let half = |d: u64| d as f64 / total.max(1) as f64 / 2.;
                let fraction = match stage {
                    Stage::Check => half(done),
                    Stage::Write => 0.5 + half(done),
                };
                (self.emit)(Event::FasterProgress(FasterProgress {
                    path: path.clone(),
                    fraction,
                    seconds: started.elapsed().as_secs_f64(),
                }));
            }
            !stop.load(Ordering::Relaxed)
        };
        let made = std::fs::create_dir_all(dst.parent().unwrap_or(dst))
            .map_err(faster::Error::from)
            .and_then(|()| faster::build(src, dst, &mut progress));
        self.end_build(src);
        (self.emit)(match made {
            Ok(()) => Event::FasterDone(FasterDone {
                path,
                copy: dst.to_string_lossy().into_owned(),
            }),
            Err(e) => Event::FasterFailed(FasterFailed {
                path,
                cancelled: matches!(e, faster::Error::Stopped),
                message: e.to_string(),
            }),
        });
    }

    fn end_build(&self, src: &Path) {
        (self.building.lock().unwrap_or_else(PoisonError::into_inner)).retain(|(p, _)| p != src);
    }
}
