//! The assistant: a larch of seven boughs of almond petals on a trunk that runs to a bud, pale
//! gold at the top deepening to ember at the foot. It says what it is doing with its petals: gold
//! climbs the boughs while it works, petals open one by one while it writes, a slow lift rises up
//! the still tree while it thinks, the top boughs reach toward you when it needs you, petals droop
//! and one falls on a problem. Every state is also written as text beside it, so nothing is said
//! by motion alone. A port of the design's petal tree P2 (`boughTree`, `petalMove`, 23 Nov).

use std::f32::consts::PI;
use std::time::Instant;

use gpui::{
    App, Bounds, IntoElement, PathBuilder, Pixels, Point, Rgba, Styled, Window, canvas, point, px,
};

use crate::theme::{Theme, mix};

/// What the assistant is doing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mood {
    Ready,
    Listen,
    Think,
    Work,
    Write,
    Need,
    Done,
    Err,
    Load,
    Rest,
}

impl Mood {
    /// Every state, in the order Settings shows them.
    pub const ALL: [Mood; 10] = [
        Mood::Ready,
        Mood::Listen,
        Mood::Think,
        Mood::Work,
        Mood::Write,
        Mood::Need,
        Mood::Done,
        Mood::Err,
        Mood::Load,
        Mood::Rest,
    ];

    /// The state in words, which is always shown beside the tree.
    pub fn label(self) -> &'static str {
        match self {
            Mood::Ready => "Ready",
            Mood::Listen => "Waiting",
            Mood::Think => "Thinking",
            Mood::Work => "Working",
            Mood::Write => "Writing",
            Mood::Need => "Needs you",
            Mood::Done => "Done",
            Mood::Err => "Problem",
            Mood::Load => "Loading",
            Mood::Rest => "Resting",
        }
    }

    /// Whether this state moves on its own, so the window must keep drawing frames. Only the
    /// working states do: a mark breathing at idle keeps a laptop's GPU at full clock, ~4 W warmer,
    /// and the heat costs the next reply its boost. The rest settle after their entry motion.
    pub fn moves(self) -> bool {
        matches!(self, Mood::Think | Mood::Work | Mood::Write | Mood::Load)
    }
}

/// A mood and when it began: one-shot motions (the lean, the gold sweep) run from that instant.
#[derive(Clone, Copy)]
pub struct Tree {
    pub mood: Mood,
    pub since: Instant,
    prev: Mood,
    /// How far into the last mood (scaled seconds) the switch came: the ease starts from that
    /// frame, so a gold band mid-climb does not vanish as Working turns to Done.
    prev_t: f32,
}

impl Tree {
    /// The tree in `mood` since the app started, so a tree made afresh each frame keeps moving.
    pub fn new(mood: Mood) -> Tree {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        Tree {
            mood,
            since: *START.get_or_init(Instant::now),
            prev: mood,
            prev_t: 0.,
        }
    }

    /// Moves to `mood`; the tiers ease from the old pose to the new one.
    pub fn set(&mut self, mood: Mood) {
        if mood != self.mood {
            let t = self.since.elapsed().as_secs_f32();
            self.prev_t = if self.mood.moves() { t } else { t.min(SETTLE) } * SPEED;
            self.prev = self.mood;
            self.mood = mood;
            self.since = Instant::now();
        }
    }
}

fn ease(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}

/// cubic-bezier(.3,1.4,.5,1): the overshoot the design's lean and pose changes use.
fn back(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    let c = 1.7;
    1. + (c + 1.) * (t - 1.).powi(3) + c * (t - 1.).powi(2)
}

/// How much faster than the design's timings the tree moves.
const SPEED: f32 = 1.4;
/// Seconds a state's entry motion runs before a tree that does not move holds still.
const SETTLE: f32 = 1.5;
/// Tiers, top to foot.
const TIERS: usize = 7;

/// What a state does to one tier's colour, over the gold it has.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Tint {
    None,
    /// Needs you: the top tier glows ember, pulsing toward gold while it settles.
    Ember(f32),
    Madder,
    Rest,
}

/// One tier's pose: how gold it is, how far it sways, grows, drops and lifts (in tree sizes), how
/// much of it has appeared, and its tint.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Pose {
    gold: f32,
    sway: f32,
    grow: f32,
    drop: f32,
    lift: f32,
    show: f32,
    tint: Tint,
}

/// Tier `k` of `TIERS` (0 = top) in `mood`, `t` seconds in; `still` gives each state's held pose.
fn pose(mood: Mood, t: f32, k: usize, still: bool) -> Pose {
    let f = k as f32 / (TIERS - 1) as f32;
    let kf = k as f32;
    let mut p = Pose {
        gold: 0.,
        sway: 0.,
        grow: 1.,
        drop: 0.,
        lift: 0.,
        show: 1.,
        tint: Tint::None,
    };
    match mood {
        Mood::Ready if !still => p.sway = 0.006 * (t * 1.1 + kf).sin(),
        Mood::Listen => p.lift = 0.01,
        // Thinking holds the tree still (a sway read as a dance): tiers catch the light one by
        // one, each on its own slow beat.
        Mood::Think if !still => p.gold = 0.5 * (t * 1.6 + kf * 2.4).sin().max(0.).powi(3),
        Mood::Work if !still => {
            // A band of gold climbs from the foot to the top, the tier it passes swelling a little.
            let c = 1. - (t * 0.5) % 1.3;
            let lit = (1. - (c - f).abs() * 4.).max(0.);
            p.gold = lit;
            p.grow = 1. + 0.06 * lit;
            p.lift = 0.012 * lit;
        }
        Mood::Write if !still => {
            let w = 0.5 + 0.5 * (t * 2.2 - kf * 0.9).sin();
            p.gold = w * w * 0.8;
        }
        Mood::Need if k == 0 => {
            p.tint = Tint::Ember(if still {
                0.2
            } else {
                0.5 + 0.5 * (t * 5.).sin()
            })
        }
        // Gold sweeps up from the foot and stays.
        Mood::Done => {
            p.gold = if still {
                1.
            } else {
                ease(t - (TIERS - 1 - k) as f32 * 0.06)
            }
        }
        Mood::Err => {
            p.drop = 0.04 * if k % 2 == 1 { 1. } else { 0.4 };
            p.tint = Tint::Madder;
        }
        // Tiers appear from the foot up, then go and come again.
        Mood::Load if !still => {
            let c = (t * 0.5) % 1.3;
            p.show = (c * TIERS as f32 - (TIERS - 1 - k) as f32).clamp(0., 1.);
        }
        Mood::Rest => {
            p.tint = Tint::Rest;
            p.grow = 0.92;
        }
        _ => {}
    }
    p
}

/// Tier `k`'s colour in pose `p`.
fn color(th: &Theme, k: usize, p: Pose) -> Rgba {
    let f = k as f32 / (TIERS - 1) as f32;
    let c = mix(mix(th.gold, th.ember, f * 0.85), th.gold, p.gold);
    match p.tint {
        Tint::None => c,
        Tint::Ember(w) => mix(th.ember, th.gold, w),
        Tint::Madder => mix(c, th.madder, 0.6),
        Tint::Rest => mix(c, th.ink2, 0.5),
    }
}

/// Done's pop: the whole tree swells 7 % and back over its first second.
fn pop(mood: Mood, t: f32, still: bool) -> f32 {
    if mood == Mood::Done && !still && t < 1.4 {
        1. + 0.07 * (t / 1.4 * PI).sin()
    } else {
        1.
    }
}

/// Needs you: the upper tiers lean toward the ask, `side` -1 for left and 1 for right, in tree
/// sizes at the top tier.
fn lean(mood: Mood, t: f32, still: bool, side: f32) -> f32 {
    match mood {
        Mood::Need if still => 0.045 * side,
        Mood::Need => back((t - 0.2) / 0.6) * 0.045 * side,
        _ => 0.,
    }
}

/// The tree, `z` pixels square, on a surface of colour `paper` (each petal's rim is cut in
/// it). `dir` is the direction of the ask in degrees clockwise from up (225 = down left).
/// `still` holds each state's pose (Animation off, or reduced motion).
pub fn tree(s: Tree, z: f32, dir: f32, still: bool, th: Theme, paper: Rgba) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |b: Bounds<Pixels>, _, win: &mut Window, _: &mut App| {
            paint(b, win, s, z, dir.to_radians(), still, th, paper)
        },
    )
    .size(px(z))
    .flex_none()
}

/// How one petal of tier `k` moves in `mood`, `t` scaled seconds in: `turn` tips it (radians, +
/// down), `glow` lights it gold, `fall` drops it (0 to 1). `j` counts out from the trunk, `right`
/// is the side facing the ask, `per` is the tier's petals a side. Small on purpose: it must read at
/// 32 px without fuss.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Petal {
    turn: f32,
    glow: f32,
    fall: f32,
}

#[allow(clippy::too_many_arguments)]
fn petal(mood: Mood, t: f32, k: usize, j: usize, right: bool, per: usize, still: bool) -> Petal {
    let (kf, jf) = (k as f32, j as f32);
    let f = kf / (TIERS - 1) as f32;
    let mut p = Petal::default();
    match mood {
        // One breath, then still.
        Mood::Ready if !still => {
            p.turn = 0.03 * (t * 1.2 + kf * 0.7 + jf).sin() * (1. - t / 2.).max(0.)
        }
        // Lifted, listening.
        Mood::Listen => p.turn = -0.1,
        // A slow lift rises up the tree while its petals catch the light; the tree holds still.
        Mood::Think if !still => {
            p.turn = -0.09
                * (t * 1.4 - (TIERS - 1 - k) as f32 * 0.55 - jf * 0.4)
                    .sin()
                    .max(0.);
            let i = jf * 2. + if right { 1. } else { 0. };
            p.glow = 0.55 * (t * 1.7 + kf * 2.1 + i * 1.3).sin().max(0.).powi(4);
        }
        // The tier the gold band is passing lifts.
        Mood::Work if !still => {
            let c = 1. - (t * 0.5) % 1.3;
            p.turn = -0.16 * (1. - (c - f).abs() * 4.).max(0.);
        }
        // Petals open one by one, foot to top.
        Mood::Write if !still => {
            let i = ((TIERS - 1 - k) * per + j) as f32;
            let c = (t * 6.) % ((TIERS * per) as f32 + 4.);
            let lit = (1. - (c - i).abs() * 0.8).max(0.);
            (p.turn, p.glow) = (-0.2 * lit, 0.3 * lit);
        }
        // The top boughs reach toward the ask and glow.
        Mood::Need if k <= 2 => {
            let r = if still { 1. } else { back((t - 0.2) / 0.6) };
            p.turn = if right { -0.16 * r } else { 0. };
            p.glow = 0.2 * r;
        }
        // Gold sweeps up from the foot, the petals rise with it, then settle.
        Mood::Done => {
            let (u, back) = if still {
                (1., 1.)
            } else {
                (
                    ease(t - (TIERS - 1 - k) as f32 * 0.06),
                    ease((t - 1.2) / 0.6),
                )
            };
            (p.turn, p.glow) = (-0.22 * u + 0.12 * back, 0.4 * u);
        }
        // Drooped, a brief shiver, and one petal falls.
        Mood::Err => {
            p.turn = 0.4
                + if still {
                    0.
                } else {
                    0.05 * (t * 14.).sin() * (1. - t / 0.6).max(0.)
                };
            if k == 3 && j == per - 1 && right {
                p.fall = if still { 1. } else { ease(t / 1.2) };
            }
        }
        // Folded down.
        Mood::Rest => p.turn = 0.55,
        _ => {}
    }
    p
}

/// Petals a side on tier `k`: two at the top, five at the foot.
fn per(k: usize) -> usize {
    2 + (3. * k as f32 / (TIERS - 1) as f32).round() as usize
}

/// An almond (pointed both ends, the Arkaya lotus petal) centred at `c`, `len` long and `wid`
/// across, turned `a` from pointing right.
fn almond(c: Point<Pixels>, len: f32, wid: f32, a: f32, pb: &mut PathBuilder) {
    let (sn, cs) = a.sin_cos();
    let at = |x: f32, y: f32| c + point(px(x * cs - y * sn), px(x * sn + y * cs));
    pb.move_to(at(-len / 2., 0.));
    pb.curve_to(at(len / 2., 0.), at(0., -wid));
    pb.curve_to(at(-len / 2., 0.), at(0., wid));
    pb.close();
}

/// Paints an almond in `col`, cut from what is under it by a rim of `paper` when `rim` > 0.
#[allow(clippy::too_many_arguments)]
fn paint_almond(
    win: &mut Window,
    c: Point<Pixels>,
    len: f32,
    wid: f32,
    a: f32,
    col: Rgba,
    paper: Rgba,
    rim: f32,
) {
    if rim > 0. {
        let mut edge = PathBuilder::stroke(px(rim));
        almond(c, len, wid, a, &mut edge);
        if let Ok(path) = edge.build() {
            win.paint_path(path, paper);
        }
    }
    let mut body = PathBuilder::fill();
    almond(c, len, wid, a, &mut body);
    if let Ok(path) = body.build() {
        win.paint_path(path, col);
    }
}

#[allow(clippy::too_many_arguments)]
fn paint(
    b: Bounds<Pixels>,
    win: &mut Window,
    s: Tree,
    z: f32,
    dir: f32,
    still: bool,
    th: Theme,
    paper: Rgba,
) {
    if !still && s.mood.moves() || s.since.elapsed().as_secs_f32() < SETTLE {
        win.request_animation_frame();
    }
    let mut t = s.since.elapsed().as_secs_f32();
    // A settled tree holds the pose of its last frame, so a later repaint does not jump.
    if !s.mood.moves() {
        t = t.min(SETTLE);
    }
    let t = t * SPEED;
    // The pose eases over .45 s from the last state's pose at the switch.
    let blend = if still { 1. } else { back(t / 0.45) };
    let lerp = |a: f32, b: f32| a + (b - a) * blend;
    let side = (dir.sin() / std::f32::consts::FRAC_1_SQRT_2).clamp(-1., 1.);
    // The design reaches .47 above its centre and .48 below, petals .46 to a side.
    let zz = z * 0.98;
    let c = b.center();
    let at = |x: f32, y: f32| -> Point<Pixels> { c + point(px(x), px(y)) };
    let pop = pop(s.mood, t, still);
    let lean = lerp(
        lean(s.prev, s.prev_t, still, side),
        lean(s.mood, t, still, side),
    ) * zz;
    let top = -0.47 * zz;
    // Rims cut petal from petal; under 40 px they would only blur, so small trees go without.
    let rim = if z >= 40. { (zz * 0.014).max(1.) } else { 0. };
    // Without rims, small petals go plumper so the boughs still read as boughs, not lines.
    let plump = if z >= 40. { 0.36 } else { 0.5 };
    // The trunk runs up through the canopy to the bud, tapering, so every bough hangs from it.
    let mut trunk = PathBuilder::fill();
    let (tb, tt, ty) = (0.03 * zz, 0.008 * zz, top + 0.04 * zz);
    trunk.move_to(at(-tb, 0.48 * zz));
    trunk.line_to(at(lean - tt, ty));
    trunk.line_to(at(lean + tt, ty));
    trunk.line_to(at(tb, 0.48 * zz));
    trunk.close();
    if let Ok(path) = trunk.build() {
        win.paint_path(
            path,
            if s.mood == Mood::Rest {
                th.wood2
            } else {
                th.wood
            },
        );
    }
    for k in (0..TIERS).rev() {
        let (p0, p1) = (pose(s.prev, s.prev_t, k, still), pose(s.mood, t, k, still));
        let p = Pose {
            gold: lerp(p0.gold, p1.gold),
            sway: lerp(p0.sway, p1.sway),
            grow: lerp(p0.grow, p1.grow),
            drop: lerp(p0.drop, p1.drop),
            lift: lerp(p0.lift, p1.lift),
            show: lerp(p0.show, p1.show).clamp(0., 1.),
            tint: p1.tint,
        };
        if p.show <= 0.01 {
            continue;
        }
        let f = k as f32 / (TIERS - 1) as f32;
        let y = top + zz * (0.08 + 0.64 * f) + (p.drop - p.lift) * zz;
        let x = lean * (1. - f) + p.sway * zz;
        let w = zz * (0.1 + 0.34 * f) * pop * p.show * p.grow;
        let len = zz * (0.1 + 0.06 * f) * pop * p.show;
        let n = per(k);
        for right in [true, false] {
            let sg = if right { 1. } else { -1. };
            for j in (0..n).rev() {
                let (q0, q1) = (
                    petal(s.prev, s.prev_t, k, j, right, n, still),
                    petal(s.mood, t, k, j, right, n, still),
                );
                let turn = lerp(q0.turn, q1.turn);
                let glow = lerp(q0.glow, q1.glow).clamp(0., 1.);
                let droop = 0.32 + 0.5 * j as f32 / n as f32 + turn;
                let a = if right { droop } else { PI - droop };
                let d = 0.012 * zz + (j as f32 + 0.35) / n as f32 * w * 0.78;
                let (bx, by) = (x + sg * d - a.cos() * len * 0.45, y + 0.012 * zz * j as f32);
                let by = by + q1.fall * 0.18 * zz;
                // A petal's own light joins its tier's gold; the colour eases as the shape does.
                let lit = |q: Pose| Pose {
                    gold: 1. - (1. - q.gold) * (1. - glow),
                    ..q
                };
                let mut col = mix(
                    color(&th, k, lit(p0)),
                    color(&th, k, lit(p1)),
                    blend.min(1.),
                );
                if q1.fall > 0. {
                    col = mix(col, th.madder, 0.4);
                    col.a *= 1. - q1.fall * 0.6;
                }
                let centre = at(bx + a.cos() * len * 0.5, by + a.sin() * len * 0.5);
                paint_almond(win, centre, len, len * plump, a, col, paper, rim);
            }
        }
    }
    // The bud on top: pale gold, ember while it needs you.
    let bud = match s.mood {
        Mood::Need if !still => mix(th.ember, th.gold, 0.5 + 0.5 * (t * 5.).sin()),
        Mood::Need => th.ember,
        Mood::Err => th.madder,
        Mood::Rest => th.wood2,
        _ => th.gold,
    };
    paint_almond(
        win,
        at(lean, top + 0.02 * zz),
        0.11 * zz * pop,
        0.04 * zz,
        PI / 2.,
        bud,
        paper,
        rim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs you: only the top tier turns ember, and the tree leans toward the ask.
    #[test]
    fn need_marks_the_top_tier_and_leans_toward_the_ask() {
        assert!(matches!(
            pose(Mood::Need, 3., 0, false).tint,
            Tint::Ember(_)
        ));
        assert_eq!(pose(Mood::Need, 3., 1, false).tint, Tint::None);
        assert!(lean(Mood::Need, 5., false, -1.) < -0.04);
        assert_eq!(lean(Mood::Ready, 5., false, -1.), 0.);
    }

    /// Working: the gold band starts at the foot and has reached the top 1.5 s later.
    #[test]
    fn work_gold_climbs_from_the_foot() {
        assert!(pose(Mood::Work, 0., TIERS - 1, false).gold > 0.9);
        assert!(pose(Mood::Work, 0., 0, false).gold < 0.01);
        assert!(pose(Mood::Work, 1.9, 0, false).gold > 0.5);
    }

    /// Loading shows the foot before the top; Done ends all gold.
    #[test]
    fn load_grows_from_the_foot_and_done_ends_gold() {
        let at = 0.5;
        assert!(pose(Mood::Load, at, TIERS - 1, false).show > pose(Mood::Load, at, 0, false).show);
        assert!((0..TIERS).all(|k| pose(Mood::Done, 5., k, false).gold == 1.));
    }

    /// Leaving a moving state eases from where it was, not from its still pose: Working's gold
    /// band does not drop out on the first frame of Done.
    #[test]
    fn a_switch_starts_from_the_last_frame() {
        let mut tree = Tree::new(Mood::Work);
        std::thread::sleep(std::time::Duration::from_millis(5));
        tree.set(Mood::Done);
        assert!(tree.prev_t > 0.);
        let mut still = Tree::new(Mood::Rest);
        still.set(Mood::Ready);
        assert!(still.prev_t <= SETTLE * SPEED);
    }

    /// An idle window stops drawing: only working states ask for frames forever.
    #[test]
    fn only_working_states_keep_drawing() {
        let moving: Vec<_> = Mood::ALL.into_iter().filter(|m| m.moves()).collect();
        assert_eq!(moving, [Mood::Think, Mood::Work, Mood::Write, Mood::Load]);
    }

    /// Petals: a problem drops exactly one; thinking only lifts them and lights some; resting
    /// folds every one down.
    #[test]
    fn a_problem_drops_one_petal_and_thinking_only_lifts() {
        let all = |m: Mood, t: f32| {
            (0..TIERS).flat_map(move |k| {
                (0..per(k))
                    .flat_map(move |j| [true, false].map(|r| petal(m, t, k, j, r, per(k), false)))
            })
        };
        assert_eq!(all(Mood::Err, 2.).filter(|p| p.fall > 0.99).count(), 1);
        assert!(all(Mood::Think, 1.).all(|p| p.turn <= 0. && p.fall == 0.));
        assert!((0..30).any(|i| all(Mood::Think, i as f32 * 0.1).any(|p| p.glow > 0.2)));
        assert!(all(Mood::Rest, 0.).all(|p| p.turn > 0.5));
    }

    #[test]
    fn thinking_lights_tiers_without_moving_them() {
        let poses = (0..40)
            .flat_map(|i| (0..TIERS).map(move |k| pose(Mood::Think, i as f32 * 0.1, k, false)));
        let (mut lit, mut moved) = (false, false);
        for p in poses {
            lit |= p.gold > 0.2;
            moved |= p.sway != 0. || p.lift != 0. || p.grow != 1.;
        }
        assert!(lit && !moved);
    }
}
