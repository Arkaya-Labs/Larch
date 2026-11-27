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
use larch_records::knob_id;

use crate::contract::{Faster, FasterDone, FasterFailed, FasterProgress};
use crate::{App, Event, Inner};

/// How often a build says how far it got.
const TICK: Duration = Duration::from_millis(250);

/// The folder the copies go to, under the app's data.
const DIR: &str = "faster";

impl Inner {
    /// Where the copy of `src` goes.
    pub(crate) fn faster_path(&self, src: &Path) -> PathBuf {
        let name = src.file_stem().unwrap_or_default().to_string_lossy();
        self.data.join(DIR).join(format!("{name}.faster.gguf"))
    }

    /// The card's offer for `src`: a CUDA build (where the copy's 2-bit experts load and it was
    /// measured faster), a file [`larch::faster`] carries a histogram for, and a GPU that cannot
    /// hold the whole original but can hold the copy ([`faster::Known::offered_on`]).
    // ponytail: measured on one 16 GB sm_120 card (bench/results/35b-vs-llamacpp-2026-11-25.md);
    // elsewhere switch_or_back keeps the copy only where it loads whole with graphs. Narrow the
    // offer by card if one passes that check and still measures the copy slower.
    pub(crate) fn faster_offer(&self, src: &Path, gpu_total: Option<u64>) -> Option<Faster> {
        let k = faster::known(src)?;
        let here = larch_kernels::BACKEND == "cuda" && gpu_total.is_some_and(|t| k.offered_on(t));
        here.then(|| {
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
        let gpu = self.inner.gpu().as_ref().map(|g| g.1);
        self.inner.faster_offer(Path::new(path), gpu)
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
            // Two libraries' files of the same name share one copy: one build at a time for it.
            if jobs
                .iter()
                .any(|(p, _)| self.inner.faster_path(p) == self.inner.faster_path(&src))
            {
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
        (self.inner.stop_build(Path::new(path)))
            .then_some(())
            .ok_or_else(|| "The faster version is not being made.".into())
    }

    /// Loads the faster copy of the model at `path` (`on`) or the original. A copy that does not
    /// run faster here goes back to the original ([`App::switch_or_back`]); the error says why.
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

    /// Loads `copy` in place of the original at `path`, and keeps it only when it runs as it was
    /// measured faster: loaded whole on the GPU, with its decode graphs on. A copy that fails to
    /// load and no longer passes its check is deleted; one that loads short of room (another
    /// program holding part of the GPU) is kept for later. Either way the original is loaded again
    /// and the error says why.
    pub(crate) fn switch_or_back(&self, path: &str, copy: &Path) -> Result<(), String> {
        let before = self.app_state().model;
        let why = match self.load_model(&copy.to_string_lossy()) {
            Ok(()) if self.runs_faster() => return Ok(()),
            Ok(()) => {
                "The faster version had too little free GPU memory to run faster, so Larch went \
                 back to the original. Close programs using the GPU and try again."
                    .to_string()
            }
            // Refused before anything was unloaded (a reply being written, a load under way): the
            // model that ran still runs, and the copy may well load next time.
            Err(why) if before.is_some() && self.app_state().model == before => return Err(why),
            // A copy that still passes its check is whole: the load failed for want of room or of
            // the device, so it is kept for later. One that does not is deleted.
            Err(why) => {
                let whole =
                    faster::known(Path::new(path)).is_some_and(|k| faster::check(copy, k).is_ok());
                if !whole {
                    let _ = std::fs::remove_file(copy);
                }
                format!(
                    "The faster version did not load, so Larch went back to the original. ({why})"
                )
            }
        };
        match self.load_model(path) {
            Ok(()) => Err(why),
            Err(e) => Err(format!("{why} The original did not load either ({e}).")),
        }
    }

    /// Whether the loaded copy runs as it was measured faster than the original
    /// (bench/results/35b-vs-llamacpp-2026-11-25.md): every expert layer on the GPU and decode
    /// graphs on. With 500 MB of the card held by another program, the load kept one expert layer
    /// in host memory and ran slower than the original, graphs and all; with less room the fit
    /// turns `decode_graphs` off, and a backend that cannot replay graphs (the CPU reference)
    /// keeps it at 0.
    fn runs_faster(&self) -> bool {
        let st = self.inner.lock();
        st.larch().is_ok_and(|l| {
            l.host_expert_layers() == 0
                && l.knobs().effective().values[knob_id::DECODE_GRAPHS as usize] != 0.0
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
        // The copies' folder is the app's own: a copy there that no longer passes its check (a
        // disk error since) is made again rather than refused.
        if let Some(k) = faster::known(src)
            && dst.is_file()
            && faster::check(dst, k).is_err()
        {
            let _ = std::fs::remove_file(dst);
        }
        let made = std::fs::create_dir_all(dst.parent().unwrap_or(dst))
            .map_err(faster::Error::from)
            .and_then(|()| faster::build(src, dst, &mut progress))
            // Stopped after the last check (the model deleted meanwhile): the copy goes too, or
            // nothing would list or delete it.
            .and_then(|()| match stop.load(Ordering::SeqCst) {
                true => {
                    let _ = std::fs::remove_file(dst);
                    Err(faster::Error::Stopped)
                }
                false => Ok(()),
            });
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

    /// Stops the build of `src`'s copy, if one runs: whether one did.
    pub(crate) fn stop_build(&self, src: &Path) -> bool {
        let jobs = self.building.lock().unwrap_or_else(PoisonError::into_inner);
        let job = jobs.iter().find(|(p, _)| p == src);
        job.inspect(|(_, stop)| stop.store(true, Ordering::SeqCst))
            .is_some()
    }

    fn end_build(&self, src: &Path) {
        (self.building.lock().unwrap_or_else(PoisonError::into_inner)).retain(|(p, _)| p != src);
    }
}
