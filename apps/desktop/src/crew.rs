//! Helpers in a chat: the workers a tool runs side by side (a research report's section writers,
//! the papers it reads). The tree at the top of the chat spawns them as rays, each in its own mark
//! and tone, and shows them working; the reply only lists their states as steps.

use std::f32::consts::{PI, TAU};
use std::hash::{BuildHasher, RandomState};
use std::time::Instant;

use gpui::{
    AnyElement, App, Bounds, Hsla, IntoElement, ParentElement, PathBuilder, Pixels, Point, Styled,
    Window, canvas, div, fill, point, px, size,
};
use serde_json::Value;

use crate::theme::{Theme, fade};
use crate::tree::{Tree, tree};
use crate::ui::{Step, StepState, u};

/// One helper the chat started.
#[derive(Clone, Debug)]
pub struct Helper {
    /// The part it was given, in the user's words.
    pub task: String,
    /// What it is doing now, or what it found once done.
    pub now: String,
    pub st: StepState,
    /// How far it is, 0 to 1, when it can tell.
    pub frac: Option<f32>,
}

impl Helper {
    /// A helper as the chat event says it stands.
    pub fn from_chat(h: larch_app::contract::ChatHelper) -> Self {
        let (st, now) = match h.state.as_str() {
            "queued" => (StepState::No, "Waiting for a free seat"),
            "done" => (StepState::Ok, ""),
            "failed" => (StepState::Fail, "Stopped without an answer"),
            _ => (StepState::Run, ""),
        };
        Helper {
            task: h.task,
            now: now.into(),
            st,
            frac: None,
        }
    }
}

/// The helpers of one reply.
#[derive(Clone, Debug)]
pub struct Crew {
    pub helpers: Vec<Helper>,
    /// When they were split off: the tree spawns them from then.
    pub since: Instant,
    /// This reply's draw of marks and tones (`look`).
    seed: u64,
}

impl Crew {
    /// The helpers `helpers`, split off now.
    pub fn new(helpers: Vec<Helper>) -> Self {
        Crew {
            helpers,
            since: Instant::now(),
            seed: RandomState::new().hash_one(0u8),
        }
    }

    /// The time the picture shows: since the split, or long settled when still.
    fn now(&self, still: bool) -> f32 {
        if still {
            1e3
        } else {
            self.since.elapsed().as_secs_f32()
        }
    }
}

/// Whether `h` has yet to end: waiting for a seat or working; a failed or stopped one has ended.
pub fn open(h: &Helper) -> bool {
    matches!(h.st, StepState::Run | StepState::No)
}

fn queued(h: &Helper) -> bool {
    matches!(h.st, StepState::No | StepState::Stop)
}

/// How far a helper is: its own share when it can tell, a slow swell while it works when not.
fn reach(h: &Helper, i: usize, t: f32) -> f32 {
    match h.st {
        StepState::Ok => 1.,
        StepState::Run => h
            .frac
            .unwrap_or_else(|| 0.5 + 0.12 * (t * 0.9 + i as f32 * 1.7).sin()),
        StepState::Wait | StepState::Fail => h.frac.unwrap_or(0.5),
        StepState::No | StepState::Stop => 0.,
    }
}

/// How far helper `i` has come out of the tree, 0 to 1: one after another, then settled.
fn spawn(i: usize, t: f32) -> f32 {
    let x = ((t - 0.6 - 0.35 * i as f32) / 0.7).clamp(0., 1.);
    x * x * (3. - 2. * x)
}

/// The tree's status and its second line while it runs `c`.
pub fn status(c: &Crew, still: bool) -> (String, Option<String>) {
    let n = c.helpers.len();
    let open: Vec<&Helper> = c.helpers.iter().filter(|h| open(h)).collect();
    if (0..n).any(|i| spawn(i, c.now(still)) < 1.) {
        return (format!("Splitting into {n} helpers"), None);
    }
    match open.as_slice() {
        [] => ("Joining what they found".into(), None),
        [h] => ("Waiting on 1 helper".into(), Some(h.task.clone())),
        _ => (
            format!("Waiting on {} helpers", open.len()),
            Some(format!("{} of {n} done", n - open.len())),
        ),
    }
}

/// The reply's record of the helpers, as steps: the split, one row per helper, and the wait.
pub fn steps(c: &Crew) -> Vec<Step> {
    let step = |text: String, sum: String, st: StepState| Step {
        name: "helper".into(),
        args: Value::Null,
        now: text.clone(),
        done: text,
        sum,
        st,
        count: 1,
    };
    let n = c.helpers.len();
    let mut out = vec![step(
        format!("Split into {n} helpers"),
        String::new(),
        StepState::Ok,
    )];
    out.extend(
        c.helpers
            .iter()
            .map(|h| step(h.task.clone(), h.now.clone(), h.st)),
    );
    let open = c.helpers.iter().filter(|h| open(h)).count();
    if open > 0 {
        out.push(step(
            format!("Waiting on {open} of {n}"),
            String::new(),
            StepState::Run,
        ));
    }
    out
}

/// The chat's tree `s` of size `z` with `c`'s helpers round it; `still` holds all motion.
pub fn crest(th: &Theme, s: Tree, c: &Crew, z: f32, still: bool) -> AnyElement {
    rays(th, s, c, z, still)
}

const MARKS: usize = 6;
const TONES: usize = 4;

/// Helper `i`'s mark and tone in a crew drawn by `seed`: tones go round so neighbours differ, and
/// the mark steps one further each lap, so no two of the first `MARKS * TONES` share both.
// ponytail: 24 distinct looks; a 25th helper repeats one. Add marks or tones if crews get bigger.
fn look(seed: u64, i: usize) -> (usize, usize) {
    let mut marks: [usize; MARKS] = std::array::from_fn(|k| k);
    let mut x = seed | 1;
    for k in (1..MARKS).rev() {
        // xorshift64: enough to shuffle six marks.
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        marks.swap(k, (x % (k as u64 + 1)) as usize);
    }
    let lap = i / TONES;
    (
        marks[(i + lap) % MARKS],
        (i + (seed >> 32) as usize) % TONES,
    )
}

/// Tone `k`, none like the tree: forest, madder, river and gold, from the website's palette.
fn tone(th: &Theme, k: usize) -> Hsla {
    let light = [0x356345, 0xA4452F, 0x2C6F78, 0xB0820F];
    let dark = [0x91C8AC, 0xE08A6E, 0x7CC0C6, 0xE2BC5A];
    gpui::rgb(if th.dark { dark } else { light }[k]).into()
}

/// Mark `m` at `c`, turned by `a`: a diamond, a petal, three seeds, a square, a triangle or a
/// cross.
fn motif(win: &mut Window, c: Point<Pixels>, m: usize, a: f32, col: Hsla) {
    let at = |x: f32, y: f32| {
        let (s, k) = a.sin_cos();
        c + point(px(x * k - y * s), px(x * s + y * k))
    };
    let mut pb = PathBuilder::fill();
    let poly = |pb: &mut PathBuilder, pts: &[(f32, f32)]| {
        pb.move_to(at(pts[0].0, pts[0].1));
        for &(x, y) in &pts[1..] {
            pb.line_to(at(x, y));
        }
    };
    match m {
        0 => poly(&mut pb, &[(0., -5.5), (4., 0.), (0., 5.5), (-4., 0.)]),
        1 => {
            pb.move_to(at(0., -6.));
            pb.curve_to(at(0., 6.), at(5., 0.));
            pb.curve_to(at(0., -6.), at(-5., 0.));
        }
        2 => {
            for k in 0..3 {
                let b = a + k as f32 * TAU / 3.;
                dot(
                    win,
                    c + point(px(b.sin() * 3.6), px(-b.cos() * 3.6)),
                    1.9,
                    col,
                );
            }
            return;
        }
        3 => poly(
            &mut pb,
            &[(-3.6, -3.6), (3.6, -3.6), (3.6, 3.6), (-3.6, 3.6)],
        ),
        4 => poly(&mut pb, &[(0., -5.), (4.6, 3.4), (-4.6, 3.4)]),
        _ => poly(
            &mut pb,
            &[
                (-1.6, -5.),
                (1.6, -5.),
                (1.6, -1.6),
                (5., -1.6),
                (5., 1.6),
                (1.6, 1.6),
                (1.6, 5.),
                (-1.6, 5.),
                (-1.6, 1.6),
                (-5., 1.6),
                (-5., -1.6),
                (-1.6, -1.6),
            ],
        ),
    }
    pb.close();
    if let Ok(p) = pb.build() {
        win.paint_path(p, col);
    }
}

/// An arc round `c` of radius `r` from the top, `f` of the way round.
fn arc(win: &mut Window, c: Point<Pixels>, r: f32, f: f32, width: f32, col: Hsla) {
    let steps = (48. * f).ceil().max(1.) as usize;
    let mut pb = PathBuilder::stroke(px(width));
    for j in 0..=steps {
        let a = -TAU / 4. + TAU * f * j as f32 / steps as f32;
        let p = c + point(px(a.cos() * r), px(a.sin() * r));
        if j == 0 { pb.move_to(p) } else { pb.line_to(p) }
    }
    if let Ok(p) = pb.build() {
        win.paint_path(p, col);
    }
}

/// How much taller than the tree alone its helpers' rays make the crest.
pub const REACH: f32 = 104.;

/// The crest's width, the arc's half-width and depth, and a helper's radius.
const W: f32 = 560.;
const RX: f32 = 215.;
const RY: f32 = 64.;
const RAD: f32 = 14.;

/// Where helper `i` of `n` sits round a tree of size `z`: along a low arc under it.
fn spot(i: usize, n: usize, z: f32) -> (f32, f32) {
    let a = PI * (0.92 - 0.84 * (i as f32 + 0.5) / n as f32);
    (W / 2. + RX * a.cos(), z / 2. + z * 0.12 + RY * a.sin())
}

/// The width each helper's name gets under it: the room to its neighbours on its line, up to
/// 160. None when a name would not fit a readable width: the marks stand alone and the reply's
/// list names them.
fn label_w(n: usize, z: f32) -> Option<f32> {
    let room = (1..n)
        .map(|i| (spot(i - 1, n, z), spot(i, n, z)))
        // A line of 11.5 px text: names this far apart vertically never touch.
        .filter(|(a, b)| (a.1 - b.1).abs() < 17.)
        .map(|(a, b)| (a.0 - b.0).abs() - 8.)
        .fold(160f32, f32::min);
    (room >= 96.).then_some(room)
}

/// The tree sends one long ray to each helper. A helper is its own mark in its own tone, never a
/// small tree: a ring that fills round as it gets further, a filled seal once done, a dashed ring
/// while it waits for a seat. Light runs along a working ray toward its helper.
fn rays(th: &Theme, s: Tree, c: &Crew, z: f32, still: bool) -> AnyElement {
    let th2 = *th;
    let helpers = c.helpers.clone();
    let n = helpers.len().max(1);
    let (w, h) = (W, z + REACH);
    let (cx, cy) = (w / 2., z / 2.);
    let rad = RAD;
    let (since, seed) = (c.since, c.seed);
    let picture = canvas(
        |_, _, _| (),
        move |b: Bounds<Pixels>, _, win: &mut Window, _: &mut App| {
            let t = if still {
                1e3
            } else {
                since.elapsed().as_secs_f32()
            };
            if !still {
                win.request_animation_frame();
            }
            let o = b.origin;
            for (i, hp) in helpers.iter().enumerate() {
                let to = spot(i, n, z);
                let (dx, dy) = (to.0 - cx, to.1 - cy);
                let len = (dx * dx + dy * dy).sqrt();
                let (a0, a1) = (z * 0.4 / len, 1. - (rad + 4.) / len);
                let at = |f: f32| {
                    let f = a0 + (a1 - a0) * f;
                    o + point(px(cx + dx * f), px(cy + dy * f))
                };
                let grown = spawn(i, t);
                let (mark, k) = look(seed, i);
                let col = tone(&th2, k);
                let mut track = PathBuilder::stroke(px(1.));
                if queued(hp) {
                    track = track.dash_array(&[px(2.), px(4.)]);
                }
                track.move_to(at(0.));
                track.line_to(at(grown.max(0.01)));
                if let Ok(p) = track.build() {
                    win.paint_path(p, fade(th2.line, 1.));
                }
                if grown < 1. {
                    continue;
                }
                let me = o + point(px(to.0), px(to.1));
                let r = reach(hp, i, t);
                match hp.st {
                    StepState::No | StepState::Stop => {
                        let mut ring = PathBuilder::stroke(px(1.2)).dash_array(&[px(3.), px(3.)]);
                        ring.move_to(me + point(px(rad), px(0.)));
                        for j in 1..=48 {
                            let a = j as f32 * TAU / 48.;
                            ring.line_to(me + point(px(a.cos() * rad), px(a.sin() * rad)));
                        }
                        if let Ok(p) = ring.build() {
                            win.paint_path(p, fade(th2.line, 1.));
                        }
                        motif(win, me, mark, 0., col.opacity(0.35));
                        continue;
                    }
                    StepState::Ok => {
                        dot(win, me, rad, col);
                        motif(win, me, mark, 0., th2.card.into());
                    }
                    _ => {
                        dot(win, me, rad, th2.card.into());
                        arc(win, me, rad, 1., 1.2, col.opacity(0.25));
                        arc(win, me, rad, r, 2.4, col);
                        // A working helper's mark turns slowly, each at its own pace.
                        let turn = if still {
                            0.
                        } else {
                            t * (0.5 + 0.15 * i as f32)
                        };
                        motif(win, me, mark, turn, col);
                    }
                }
                let mut lit = PathBuilder::stroke(px(1.6));
                lit.move_to(at(0.));
                lit.line_to(at(if hp.st == StepState::Ok { 1. } else { r }));
                if let Ok(p) = lit.build() {
                    win.paint_path(p, col.opacity(0.8));
                }
                if hp.st == StepState::Run && !still {
                    // A glint leaves the tree and runs out to its helper.
                    let g = (t * (0.45 + 0.08 * i as f32) + i as f32 * 0.37).fract();
                    for k in 0..5 {
                        let f = (g - k as f32 * 0.03).max(0.);
                        let a = 0.95 - k as f32 * 0.18;
                        dot(win, at(f), 2.3 - k as f32 * 0.35, fade(th2.sun, a));
                    }
                }
            }
        },
    )
    .w(px(w))
    .h(px(h));
    let t = c.now(still);
    let mut el = div()
        .relative()
        .w(px(w))
        .h(px(h))
        .flex_none()
        .child(picture)
        .child(
            div()
                .absolute()
                .left(px(cx - z / 2.))
                .top(px(0.))
                .child(tree(s, z, 225., still, *th, th.app)),
        );
    let Some(lw) = label_w(n, z) else {
        return el.into_any_element();
    };
    for (i, hp) in c.helpers.iter().enumerate() {
        if spawn(i, t) < 1. {
            continue;
        }
        let (x, y) = spot(i, n, z);
        el = el.child(
            div()
                .absolute()
                .left(px(x - lw / 2.))
                .top(px(y + rad + 4.))
                .w(px(lw))
                .flex()
                .justify_center()
                .child(
                    div()
                        .truncate()
                        .text_size(u(11.5))
                        .text_color(if queued(hp) { th.ink3 } else { th.ink2 })
                        .child(hp.task.clone()),
                ),
        );
    }
    el.into_any_element()
}

fn dot(win: &mut Window, c: Point<Pixels>, r: f32, color: Hsla) {
    win.paint_quad(
        fill(Bounds::centered_at(c, size(px(2. * r), px(2. * r))), color).corner_radii(px(r)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(st: StepState) -> Helper {
        Helper {
            task: "t".into(),
            now: String::new(),
            st,
            frac: None,
        }
    }

    #[test]
    fn the_reply_lists_the_split_each_helper_and_the_wait() {
        let c = Crew::new(vec![h(StepState::Ok), h(StepState::Run), h(StepState::No)]);
        let s = steps(&c);
        assert_eq!(s.len(), 5);
        assert_eq!(s[0].done, "Split into 3 helpers");
        assert_eq!(s[4].now, "Waiting on 2 of 3");
        assert_eq!(status(&c, true).0, "Waiting on 2 helpers");
        let done = Crew::new(vec![h(StepState::Ok)]);
        assert_eq!(steps(&done).len(), 2);
        assert_eq!(status(&done, true).0, "Joining what they found");
    }

    /// No two helpers of a crew share both mark and tone, neighbours never share a tone, and
    /// another reply draws them anew.
    #[test]
    fn every_helper_looks_its_own() {
        for seed in [0, 1, 0xdead_beef_cafe, u64::MAX] {
            let looks: Vec<_> = (0..MARKS * TONES).map(|i| look(seed, i)).collect();
            let mut seen = looks.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), looks.len(), "seed {seed}");
            assert!(looks.windows(2).all(|w| w[0].1 != w[1].1));
        }
        assert_ne!(
            (0..8).map(|i| look(1, i)).collect::<Vec<_>>(),
            (0..8)
                .map(|i| look(0xdead_beef_cafe, i))
                .collect::<Vec<_>>()
        );
    }

    /// A few helpers keep their names under them; a crowd's would collide, so they go to the list.
    #[test]
    fn names_show_only_where_they_fit() {
        for z in [64., 92.] {
            assert_eq!(label_w(1, z), Some(160.));
            assert!(label_w(5, z).is_some());
            assert_eq!(label_w(6, z), None);
        }
    }

    /// A failed helper has ended: the tree waits on the rest, and a turn whose others are done
    /// shows no wait.
    #[test]
    fn a_failed_helper_is_not_waited_on() {
        let c = Crew::new(vec![
            h(StepState::Fail),
            h(StepState::Run),
            h(StepState::Ok),
        ]);
        assert_eq!(status(&c, true).0, "Waiting on 1 helper");
        assert_eq!(steps(&c)[4].now, "Waiting on 1 of 3");
        let ended = Crew::new(vec![
            h(StepState::Fail),
            h(StepState::Ok),
            h(StepState::Stop),
        ]);
        assert_eq!(status(&ended, true).0, "Joining what they found");
        assert_eq!(steps(&ended).len(), 4);
    }
}
