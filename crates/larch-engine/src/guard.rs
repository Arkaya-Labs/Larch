//! The guardrail's statistics (architecture 9.9): per-token decode time (`dur_ns / n_decode_seqs`) of
//! decode-dominant steps excluding graph capture warm-ups, bucketed by batch size. A watch snapshots
//! the p90 of the last 200 such steps per bucket, then compares a rolling p90 over 32 steps of the
//! same bucket against `guardrail_ratio` × that baseline. An A/B re-snapshots from arm A's latest
//! steps at each flip to arm B, so traffic drift moves both sides (#213). The engine acts on a trip;
//! this module only measures.

use std::collections::VecDeque;

/// Decode steps the baseline looks back over.
const HISTORY: usize = 200;
/// Steps in the rolling window; a bucket's baseline also needs this many samples.
const WINDOW: usize = 32;
/// Decode steps a change stays watched.
const WATCH_STEPS: u64 = 400;
/// Batch-size buckets: 1, 2, 3-4, 5-8, ..., 33-64 (`max_batch_seqs` is at most 64).
const BUCKETS: usize = 7;
/// The nearest-rank percentile compared, window against baseline. A p99 of 32 steps is their
/// maximum, so one slow step on a jittery GPU tripped it (#173); p90 of 32 ignores the 3 slowest,
/// and a slowdown on more than 3 of 32 steps, or a steady one past the ratio, still trips.
const PERCENTILE: usize = 90;

/// What a watch guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// An applied change: its revision, on `knob`.
    Rev { rev: u64, knob: u16 },
    /// The running A/B, on its arm-B steps.
    Ab,
}

struct Watch {
    target: Target,
    baseline: [Option<u64>; BUCKETS],
    window: [VecDeque<u64>; BUCKETS],
    /// Decode steps left; None for an A/B (watched while it runs).
    left: Option<u64>,
}

impl Watch {
    /// Adds a sample of bucket `b`. True when that bucket's window is full and its p90 exceeds
    /// `ratio` × the bucket's baseline; a bucket with too few baseline samples is not guarded.
    fn trips(&mut self, b: usize, ns: u64, ratio: f64) -> bool {
        let win = &mut self.window[b];
        if win.len() == WINDOW {
            win.pop_front();
        }
        win.push_back(ns);
        if let Some(left) = &mut self.left {
            *left -= 1;
        }
        win.len() == WINDOW
            && self.baseline[b].is_some_and(|base| {
                nearest_rank(win.iter().copied(), PERCENTILE) as f64 > ratio * base as f64
            })
    }
}

#[derive(Default)]
pub(crate) struct Guard {
    /// (bucket, ns per token) of the last decode-dominant steps.
    history: VecDeque<(usize, u64)>,
    /// Watched changes, oldest first.
    changes: Vec<Watch>,
    ab: Option<Watch>,
    /// `history` gained a step since the A/B's baseline was taken.
    stale: bool,
}

pub(crate) fn bucket(n_decode: u32) -> usize {
    ((u32::BITS - (n_decode - 1).leading_zeros()) as usize).min(BUCKETS - 1)
}

/// Nearest-rank p99.
pub(crate) fn p99(v: impl Iterator<Item = u64>) -> u64 {
    nearest_rank(v, 99)
}

/// Nearest-rank percentile `p` of a non-empty sample.
fn nearest_rank(v: impl Iterator<Item = u64>, p: usize) -> u64 {
    let mut v: Vec<u64> = v.collect();
    v.sort_unstable();
    v[(v.len() * p).div_ceil(100) - 1]
}

impl Guard {
    /// Each bucket's p90 over `history`; None for a bucket with fewer than a window of samples.
    fn baseline(&self) -> [Option<u64>; BUCKETS] {
        std::array::from_fn(|b| {
            let n = self.history.iter().filter(|h| h.0 == b).count();
            (n >= WINDOW).then(|| {
                nearest_rank(
                    self.history.iter().filter(|h| h.0 == b).map(|h| h.1),
                    PERCENTILE,
                )
            })
        })
    }

    /// Snapshots the baseline and starts watching `target`: one more change, or the A/B.
    pub(crate) fn watch(&mut self, target: Target) {
        let w = Watch {
            target,
            baseline: self.baseline(),
            window: Default::default(),
            left: matches!(target, Target::Rev { .. }).then_some(WATCH_STEPS),
        };
        match target {
            Target::Rev { .. } => self.changes.push(w),
            Target::Ab => self.ab = Some(w),
        }
    }

    /// Stops watching the changes to `knob`: a later revision superseded them.
    pub(crate) fn unwatch(&mut self, knob: u16) {
        self.changes
            .retain(|w| !matches!(w.target, Target::Rev { knob: k, .. } if k == knob));
    }

    /// Watching a change or the A/B: a reload would drop the watch.
    pub(crate) fn watching(&self) -> bool {
        !self.changes.is_empty() || self.ab.is_some()
    }

    pub(crate) fn unwatch_ab(&mut self) {
        self.ab = None;
    }

    /// Records one step and returns every watch that tripped, oldest first; those stop. An arm-B
    /// step counts for the A/B only; every other step counts for each watched change and the
    /// baseline history, so arm B's cost is never blamed on a change or taken as normal. Capture
    /// warm-ups neither train the baseline nor enter a window or consume its watch lifetime.
    pub(crate) fn step(
        &mut self,
        n_decode: u32,
        n_prefill_tokens: u32,
        dur_ns: u64,
        capture_warmup: bool,
        arm_b: bool,
        ratio: f64,
    ) -> Vec<Target> {
        if capture_warmup || n_decode == 0 || n_prefill_tokens > n_decode {
            return Vec::new();
        }
        let (b, ns) = (bucket(n_decode), dur_ns / u64::from(n_decode));
        if arm_b {
            // During an A/B every non-B step is arm A's: re-snapshot once per flip, not per step.
            if self.stale {
                let baseline = self.baseline();
                if let Some(w) = &mut self.ab {
                    w.baseline = baseline;
                }
                self.stale = false;
            }
            let tripped = self.ab.as_mut().is_some_and(|w| w.trips(b, ns, ratio));
            return if tripped {
                self.ab.take().map(|w| w.target).into_iter().collect()
            } else {
                Vec::new()
            };
        }
        // Arm-B steps stay out: arm B is the experiment, never a later change's baseline.
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back((b, ns));
        self.stale = true;
        let mut trips = Vec::new();
        self.changes.retain_mut(|w| {
            if w.trips(b, ns, ratio) {
                trips.push(w.target);
                return false;
            }
            w.left != Some(0)
        });
        trips
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` decode steps of one sequence at `ns` per token; the first step's trips.
    fn run(g: &mut Guard, n: usize, ns: u64, arm_b: bool) -> Vec<Target> {
        (0..n)
            .map(|_| g.step(1, 0, ns, false, arm_b, 1.5))
            .find(|t| !t.is_empty())
            .unwrap_or_default()
    }

    fn rev(rev: u64) -> Target {
        Target::Rev { rev, knob: 1 }
    }

    /// Callers bound their windows themselves (the watchdog keeps its own 200); p99 takes any length.
    #[test]
    fn p99_keeps_nearest_rank_at_window_boundaries() {
        for (len, expected) in [
            (1, 1),
            (99, 99),
            (100, 99),
            (101, 100),
            (HISTORY as u64, 198),
            (HISTORY as u64 + 1, 199),
        ] {
            assert_eq!(p99((1..=len).rev()), expected);
        }
    }

    #[test]
    fn capture_warmups_stay_out_of_the_baseline() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        // 21 outliers would raise the nearest-rank p90 of 200 samples.
        for _ in 0..21 {
            assert_eq!(g.step(1, 0, 1_000_000, true, false, 1.5), []);
        }
        g.watch(rev(1));
        assert_eq!(run(&mut g, 32, 1600, false), [rev(1)]);
    }

    #[test]
    fn capture_warmups_stay_out_of_watched_windows() {
        for (target, arm_b) in [(rev(1), false), (Target::Ab, true)] {
            let mut g = Guard::default();
            run(&mut g, 200, 1000, false);
            g.watch(target);
            assert_eq!(g.step(1, 0, 1_000_000, true, arm_b, 1.5), []);
            assert_eq!(run(&mut g, 31, 1600, arm_b), [], "window not full");
            assert_eq!(run(&mut g, 1, 1600, arm_b), [target]);
        }
    }

    #[test]
    fn capture_warmups_do_not_consume_a_changes_watch_lifetime() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(rev(1));
        assert_eq!(run(&mut g, 368, 1000, false), []);
        for _ in 0..32 {
            assert_eq!(g.step(1, 0, 1000, true, false, 1.5), []);
        }
        assert_eq!(run(&mut g, 32, 1600, false), [rev(1)]);
    }

    /// The window's p90 against the baseline's p90 (#173): up to 3 slow steps in 32 are noise, a
    /// fourth trips, and a steady slowdown past the ratio trips even over a baseline with a tail.
    #[test]
    fn trips_once_the_window_p90_exceeds_the_ratio() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(rev(7));
        assert_eq!(run(&mut g, 100, 1400, false), [], "1.4x is under 1.5x");
        assert_eq!(
            run(&mut g, 3, 9000, false),
            [],
            "3 slow steps in 32 are noise"
        );
        assert_eq!(run(&mut g, 1, 9000, false), [rev(7)], "a 4th is a slowdown");
        assert_eq!(run(&mut g, 40, 5000, false), [], "a trip stops the watch");

        let mut g = Guard::default();
        run(&mut g, 180, 1000, false);
        run(&mut g, 20, 3000, false);
        g.watch(rev(8));
        let steady = run(&mut g, 32, 1600, false);
        assert_eq!(
            steady,
            [rev(8)],
            "a steady 1.6x trips past the baseline's tail"
        );
    }

    #[test]
    fn needs_a_full_window() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(rev(1));
        assert_eq!(run(&mut g, 31, 9000, false), []);
        assert_eq!(run(&mut g, 1, 9000, false), [rev(1)]);
    }

    #[test]
    fn a_change_is_watched_for_400_decode_steps() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(rev(1));
        // Prefill-dominant steps neither count nor trip.
        assert_eq!(g.step(1, 512, 1_000_000, false, false, 1.5), []);
        run(&mut g, 400, 1000, false);
        assert_eq!(run(&mut g, 64, 9000, false), [], "the watch ended");
    }

    #[test]
    fn each_change_is_watched_and_arm_b_is_never_blamed_on_one() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        let other = Target::Rev { rev: 2, knob: 2 };
        g.watch(rev(1));
        g.watch(other);
        g.watch(Target::Ab);
        assert_eq!(
            run(&mut g, 40, 9000, true),
            [Target::Ab],
            "arm B trips the A/B"
        );
        assert_eq!(
            run(&mut g, 31, 9000, false),
            [],
            "the changes' windows are not full"
        );
        let both = run(&mut g, 1, 9000, false);
        assert_eq!(both, [rev(1), other], "both trip at the same boundary");
    }

    #[test]
    fn a_superseded_change_is_not_watched() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        let other = Target::Rev { rev: 2, knob: 2 };
        g.watch(rev(1));
        g.watch(other);
        g.unwatch(1);
        assert_eq!(run(&mut g, 64, 9000, false), [other]);
    }

    #[test]
    fn an_ab_counts_arm_b_steps_only() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(Target::Ab);
        assert_eq!(run(&mut g, 600, 9000, false), [], "arm A is never compared");
        assert_eq!(run(&mut g, 32, 9000, true), [], "arm B at arm A's pace");
        assert_eq!(run(&mut g, 32, 14400, true), [Target::Ab], "1.6x arm A");
    }

    /// Traffic that drifts during an A/B moves both arms (#213): arm B is judged against arm A's
    /// latest steps, so a drift past the ratio does not trip, and arm B slower than arm A still does.
    #[test]
    fn an_ab_judges_arm_b_against_arm_as_latest_steps() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(Target::Ab);
        assert_eq!(run(&mut g, 200, 1000, true), []);
        assert_eq!(run(&mut g, 200, 2000, false), [], "arm A drifts to 2x");
        assert_eq!(run(&mut g, 200, 2000, true), [], "arm B drifted with it");
        assert_eq!(run(&mut g, 200, 2000, false), []);
        assert_eq!(run(&mut g, 32, 3200, true), [Target::Ab], "1.6x arm A");
    }

    #[test]
    fn buckets_compare_like_with_like() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(rev(1));
        // Batch 8 has no baseline, so it is not guarded however slow.
        let slow = (0..64)
            .map(|_| g.step(8, 0, 80_000, false, false, 1.5))
            .find(|t| !t.is_empty());
        assert_eq!(slow, None);
        assert_eq!(bucket(1), 0);
        assert_eq!(bucket(2), 1);
        assert_eq!((bucket(3), bucket(4), bucket(5)), (2, 2, 3));
        assert_eq!(bucket(64), BUCKETS - 1);
    }

    /// Arm B's steps never become a later change's baseline: an arm B slow enough to leave it
    /// guarding 1.4x more would let a later change regress that much further.
    #[test]
    fn arm_b_steps_stay_out_of_the_baseline() {
        let mut g = Guard::default();
        run(&mut g, 200, 1000, false);
        g.watch(Target::Ab);
        assert_eq!(run(&mut g, 50, 1400, true), [], "arm B under its trip");
        g.unwatch_ab();
        g.watch(rev(3));
        assert_eq!(
            run(&mut g, 32, 1600, false),
            [rev(3)],
            "arm A's baseline, not arm B's"
        );
    }
}
