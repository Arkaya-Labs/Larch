//! The six themes of the Larch design (the house design tokens, kept outside this repository): the website's palette,
//! forest green for what you press, gold only for what is running.

use gpui::{Hsla, Rgba, rgb};
use serde::{Deserialize, Serialize};

/// Every colour a screen uses.
#[derive(Clone, Copy)]
pub struct Theme {
    pub dark: bool,
    pub page: Rgba,
    pub app: Rgba,
    pub side: Rgba,
    pub card: Rgba,
    pub sunk: Rgba,
    pub ink: Rgba,
    pub ink2: Rgba,
    pub ink3: Rgba,
    pub line: Rgba,
    pub cur: Rgba,
    pub eyebrow: Rgba,
    pub keep: Rgba,
    pub lose: Rgba,
    pub accent: Rgba,
    pub on_accent: Rgba,
    /// Gold: the running model, chat memory in the GPU bar, the current page's icon block.
    pub sun: Rgba,
    /// The tree: the pale gold of its top, the ember of its foot, its problem red and its trunk
    /// (and the trunk at rest). Ember also marks what needs you.
    pub gold: Rgba,
    pub ember: Rgba,
    pub madder: Rgba,
    pub wood: Rgba,
    pub wood2: Rgba,
    /// Code blocks and the write previews: the night sky on every theme.
    pub code_bg: Rgba,
    pub code_ink: Rgba,
}

/// A light theme.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
pub enum Light {
    #[default]
    Paper,
    Porcelain,
    HighContrast,
}

/// A dark theme. All three share the forest accent, the gold sun and the paper-white ink; only the
/// night they sit on changes.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
pub enum Dark {
    /// The website's navy night sky.
    #[default]
    Night,
    /// Evergreen, the forest accent's own dark. Takes over the brown Dusk saved before.
    #[serde(alias = "Dusk")]
    Forest,
    /// Warm neutral graphite, for screens that want no colour cast.
    Graphite,
}

/// The dark card on every theme: the assistant's popover, the toast and the tour's card. Its fill,
/// its ink and its eyebrow.
pub const NOTE: u32 = 0x2a2521;
pub const NOTE_INK: u32 = 0xece6d8;
pub const NOTE_EYEBROW: u32 = 0xd9b382;

/// The sun gold, then the tree's gold, ember, madder, wood and resting wood.
const TREE: (u32, [u32; 5]) = (0xF7C531, [0xF0C24E, 0xC24E1A, 0xA4452F, 0x6B4314, 0x9A6A36]);
const TREE_DARK: (u32, [u32; 5]) = (0xF7C531, [0xF7D57A, 0xF08A4B, 0xE08A6E, 0xB98A5A, 0xD3A877]);

#[allow(clippy::too_many_arguments)]
fn make(
    dark: bool,
    surf: [u32; 5],
    ink: [u32; 3],
    line: u32,
    cur: u32,
    eyebrow: u32,
    keep: u32,
    lose: u32,
    accent: (u32, u32),
    tree: (u32, [u32; 5]),
) -> Theme {
    Theme {
        dark,
        page: rgb(surf[0]),
        app: rgb(surf[1]),
        side: rgb(surf[2]),
        card: rgb(surf[3]),
        sunk: rgb(surf[4]),
        ink: rgb(ink[0]),
        ink2: rgb(ink[1]),
        ink3: rgb(ink[2]),
        line: rgb(line),
        cur: rgb(cur),
        eyebrow: rgb(eyebrow),
        keep: rgb(keep),
        lose: rgb(lose),
        accent: rgb(accent.0),
        on_accent: rgb(accent.1),
        sun: rgb(tree.0),
        gold: rgb(tree.1[0]),
        ember: rgb(tree.1[1]),
        madder: rgb(tree.1[2]),
        wood: rgb(tree.1[3]),
        wood2: rgb(tree.1[4]),
        code_bg: rgb(if dark { 0x0a0e1a } else { 0x0e1322 }),
        code_ink: rgb(0xece6d8),
    }
}

impl Theme {
    /// The light theme `l`.
    pub fn light(l: Light) -> Theme {
        let green = (0x356345, 0xffffff);
        match l {
            Light::Paper => make(
                false,
                [0xf3ebdc, 0xfbf7ef, 0xeee6d8, 0xfffaf2, 0xf4ede1],
                [0x2a2521, 0x5e554b, 0x8d8477],
                0xdacfbd,
                0xdedfc8,
                0x784527,
                0x3c7f55,
                0xb8275a,
                green,
                TREE,
            ),
            Light::Porcelain => make(
                false,
                [0xEDEDE8, 0xFAFAF7, 0xF1F1EC, 0xFFFFFF, 0xF4F4F0],
                [0x0B0B0C, 0x55545A, 0x8A8890],
                0xE4E2DC,
                0xE3EDE2,
                0x356345,
                0x2E7D4F,
                0xB3261E,
                green,
                TREE,
            ),
            Light::HighContrast => make(
                false,
                [0xffffff, 0xffffff, 0xffffff, 0xffffff, 0xf0f0f0],
                [0x000000, 0x1a1a1a, 0x404040],
                0x000000,
                0xdfeee3,
                0x0f3d24,
                0x0b6b2f,
                0xa3001b,
                (0x0f3d24, 0xffffff),
                (0xE0A000, [0xE0A000, 0xA33A00, 0xA3001B, 0x3D2400, 0x6B4314]),
            ),
        }
    }

    /// The dark theme `d`.
    pub fn dark(d: Dark) -> Theme {
        match d {
            Dark::Forest => make(
                true,
                [0x0b1410, 0x0f1a15, 0x13201a, 0x17261f, 0x1d2e26],
                [0xe9eee6, 0xb3bfb4, 0x85948a],
                0x2a3b32,
                0x24402f,
                0xd8bd84,
                0x7fca98,
                0xef7e9a,
                (0x91c8ac, 0x0f1a15),
                TREE_DARK,
            ),
            Dark::Graphite => make(
                true,
                [0x111213, 0x161718, 0x1b1c1e, 0x202124, 0x27282c],
                [0xecebe7, 0xb4b2ad, 0x8a8883],
                0x323438,
                0x2a3a33,
                0xd4b48a,
                0x74c08f,
                0xef7ea2,
                (0x91c8ac, 0x161718),
                TREE_DARK,
            ),
            Dark::Night => make(
                true,
                [0x0a0e1a, 0x0e1322, 0x121927, 0x171e2c, 0x1c2838],
                [0xece6d8, 0xbdb5a6, 0x847d72],
                0x2c3649,
                0x293b3b,
                0xd9b382,
                0x6fbf8a,
                0xec6a9c,
                (0x91c8ac, 0x0e1322),
                TREE_DARK,
            ),
        }
    }
}

impl Theme {
    /// The block a print icon sits on: gold on the current page, a quiet wash elsewhere.
    pub fn block(&self, on: bool) -> Rgba {
        match (on, self.dark) {
            (true, false) => mix(self.sun, self.ember, 0.12),
            (true, true) => mix(self.sun, self.page, 0.3),
            (false, false) => mix(self.line, self.sun, 0.18),
            (false, true) => self.line,
        }
    }
}

/// `c` at opacity `a`.
pub fn fade(c: Rgba, a: f32) -> Hsla {
    Hsla::from(c).opacity(a)
}

/// The colour `t` of the way from `a` to `b`.
pub fn mix(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0., 1.);
    Rgba {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_saved_dusk_opens_as_forest() {
        assert_eq!(
            serde_json::from_str::<Dark>("\"Dusk\"").unwrap(),
            Dark::Forest
        );
    }
}
