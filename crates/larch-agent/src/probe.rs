//! The probe: a logistic regression on the residual stream of a prompt's last token at two layers
//! (`Larch::activations`), which says before any token is generated whether a turn needs a tool.
//! `larch-agent probe train` fits it on the bench's tool rows and scores it on the held-out rows;
//! the file lives in the data folder, never in the repository.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use larch::{EngineError, Larch, Prompt};
use larch_model::Message;
use serde::{Deserialize, Serialize};

/// The probe file's schema name.
pub const SCHEMA: &str = "larch-agent-probe/1";

/// The probability at or above which a turn gets tools.
// ponytail: 0.5 with balanced class weights; pick it on the multi-step rows when the Mac's probe
// scores show a better cut
pub const THRESHOLD: f32 = 0.5;

/// Passes of gradient descent a fit takes.
const STEPS: usize = 2000;

/// The L2 weight of a fit: with a few hundred rows and thousands of features the rows separate, and
/// this keeps the weights finite.
// ponytail: one fixed value; sweep it on the held-out AUROC if the Mac's AUROC is under 0.9
const L2: f64 = 1e-2;

/// A fitted probe, as its file holds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    /// `SCHEMA`.
    pub schema: String,
    /// The layers (0-based) whose last-token rows, in this order, make one feature row.
    pub layers: Vec<u16>,
    /// Each feature's mean over the training rows.
    pub mean: Vec<f32>,
    /// One over each feature's standard deviation there.
    pub scale: Vec<f32>,
    /// The weights of the standardized features.
    pub weights: Vec<f32>,
    /// The bias.
    pub bias: f32,
    /// A probability at or above this gives the turn its tools.
    pub threshold: f32,
    /// The training rows.
    pub rows: usize,
    /// The AUROC on the held-out rows, never on the training rows; `None` with a class missing there.
    pub auroc: Option<f64>,
}

/// Where a model's probe file is: `<data_dir>/agent/probe/<model-sha256>.json`.
pub fn path(data_dir: &Path, model_sha256: &str) -> PathBuf {
    data_dir
        .join("agent")
        .join("probe")
        .join(format!("{model_sha256}.json"))
}

/// The two layers a probe reads on a model of `n_layers`: half and three quarters of the way
/// down, once each.
// ponytail: the layers the published probes read best; try others if the Mac's AUROC is low
pub fn layers(n_layers: u16) -> Vec<u16> {
    // Three quarters, multiplied first: 30 layers is 22, not 21 (30 / 4 * 3).
    let three_quarters = u16::try_from(u32::from(n_layers) * 3 / 4).unwrap_or(u16::MAX);
    let mut layers = vec![n_layers / 2, three_quarters];
    layers.retain(|&l| l < n_layers);
    layers.dedup();
    layers
}

/// The prompt a probe reads: the system prompt and the turn's text, rendered as a chat's first
/// turn is, without tools.
// ponytail: this turn alone, not the chat's earlier turns, as the bench rows it is trained on are
// single turns; give it the history when the multi-step rows show a turn that hangs on an earlier one
pub fn prompt(system: &str, user: &str) -> Prompt {
    let message = |role: &str, content: &str| Message {
        role: role.into(),
        content: content.into(),
    };
    Prompt::Messages(vec![message("system", system), message("user", user)])
}

/// One feature row: `layers`' last-token rows of `prompt`, joined.
pub fn features(larch: &Larch, prompt: Prompt, layers: &[u16]) -> Result<Vec<f32>, EngineError> {
    // A prefill runs on the engine as a generation does, so it holds a seat while it runs; a
    // foreground one, as the router it feeds, which never waits for it.
    let seats = crate::runtime::inference::SeatAdmission::of(larch);
    let class = larch::api::Class::Foreground;
    let _seat = seats.take(1, class, &std::sync::atomic::AtomicBool::default())?;
    Ok(larch.activations(prompt, layers)?.concat())
}

/// Why a probe file cannot be used.
#[derive(Debug)]
pub enum ProbeError {
    /// It could not be read.
    Io(PathBuf, io::Error),
    /// It is not a probe this build reads, or its parts do not fit together.
    Invalid(PathBuf, String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Invalid(path, why) => write!(f, "{}: {why}", path.display()),
        }
    }
}

impl std::error::Error for ProbeError {}

impl Probe {
    /// Reads and checks a probe file for a model of `n_layers` layers.
    pub fn load(path: &Path, n_layers: u16) -> Result<Probe, ProbeError> {
        let invalid = |why: String| ProbeError::Invalid(path.into(), why);
        let text = fs::read_to_string(path).map_err(|e| ProbeError::Io(path.into(), e))?;
        let probe: Probe = serde_json::from_str(&text).map_err(|e| invalid(e.to_string()))?;
        if probe.schema != SCHEMA {
            return Err(invalid(format!("schema '{}', not {SCHEMA}", probe.schema)));
        }
        let n = probe.weights.len();
        if probe.layers.is_empty() || n == 0 || !n.is_multiple_of(probe.layers.len()) {
            return Err(invalid(format!(
                "{n} weights do not split over {} layers",
                probe.layers.len()
            )));
        }
        if probe.mean.len() != n || probe.scale.len() != n {
            return Err(invalid(
                "the mean, scale and weights differ in length".into(),
            ));
        }
        if let Some(l) = probe.layers.iter().find(|&&l| l >= n_layers) {
            return Err(invalid(format!(
                "layer {l} is past the model's {n_layers} layers"
            )));
        }
        let numbers = (probe.mean.iter().chain(&probe.scale).chain(&probe.weights))
            .chain([&probe.bias, &probe.threshold]);
        if !numbers.into_iter().all(|x| x.is_finite()) {
            return Err(invalid("a number is not finite".into()));
        }
        Ok(probe)
    }

    /// The probability that the turn whose features are `x` needs a tool.
    pub fn score(&self, x: &[f32]) -> Result<f32, String> {
        if x.len() != self.weights.len() {
            return Err(format!(
                "the model gave {} features; the probe takes {}",
                x.len(),
                self.weights.len()
            ));
        }
        let z: f64 = (x.iter().zip(&self.mean).zip(&self.scale).zip(&self.weights))
            .map(|(((&x, &m), &s), &w)| f64::from((x - m) * s) * f64::from(w))
            .sum::<f64>()
            + f64::from(self.bias);
        Ok(sigmoid(z) as f32)
    }

    /// Fits a probe on `rows` (features, needs a tool) for `layers`, by full-batch gradient descent
    /// on the standardized features with both classes weighted alike; the AUROC is the caller's to
    /// set from held-out rows. Every row must have the first row's length.
    pub fn fit(rows: &[(Vec<f32>, bool)], layers: Vec<u16>) -> Result<Probe, String> {
        let d = rows.first().map(|(x, _)| x.len()).unwrap_or(0);
        if d == 0 || rows.iter().any(|(x, _)| x.len() != d) {
            return Err("the rows are empty or differ in length".into());
        }
        let n = rows.len() as f64;
        let pos = rows.iter().filter(|(_, y)| *y).count() as f64;
        if pos == 0.0 || pos == n {
            return Err("the rows hold one class only".into());
        }
        let mut mean = vec![0f64; d];
        for (x, _) in rows {
            for (m, &v) in mean.iter_mut().zip(x) {
                *m += f64::from(v) / n;
            }
        }
        let mut var = vec![0f64; d];
        for (x, _) in rows {
            for ((s, &m), &v) in var.iter_mut().zip(&mean).zip(x) {
                *s += (f64::from(v) - m).powi(2) / n;
            }
        }
        // A feature that never changes gets weight 0 through a scale of 0.
        let scale: Vec<f64> = (var.iter())
            .map(|&v| if v > 1e-12 { 1.0 / v.sqrt() } else { 0.0 })
            .collect();
        let z: Vec<Vec<f64>> = (rows.iter())
            .map(|(x, _)| {
                (x.iter().zip(&mean).zip(&scale))
                    .map(|((&v, &m), &s)| (f64::from(v) - m) * s)
                    .collect()
            })
            .collect();
        let weight = |y: bool| {
            if y {
                n / (2.0 * pos)
            } else {
                n / (2.0 * (n - pos))
            }
        };
        // A row's squared norm is at most d once standardized, so this step never overshoots.
        let lr = 1.0 / (0.25 * d as f64 * (n / (2.0 * pos.min(n - pos))) + L2);
        let (mut w, mut b) = (vec![0f64; d], 0f64);
        for _ in 0..STEPS {
            let mut gw: Vec<f64> = w.iter().map(|&w| L2 * w).collect();
            let mut gb = 0.0;
            for (zx, (_, y)) in z.iter().zip(rows) {
                let dot: f64 = zx.iter().zip(&w).map(|(a, b)| a * b).sum();
                let err = weight(*y) * (sigmoid(dot + b) - f64::from(u8::from(*y))) / n;
                for (g, &v) in gw.iter_mut().zip(zx) {
                    *g += err * v;
                }
                gb += err;
            }
            for (w, g) in w.iter_mut().zip(&gw) {
                *w -= lr * g;
            }
            b -= lr * gb;
        }
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        Ok(Probe {
            schema: SCHEMA.into(),
            layers,
            mean: f32s(&mean),
            scale: f32s(&scale),
            weights: f32s(&w),
            bias: b as f32,
            threshold: THRESHOLD,
            rows: rows.len(),
            auroc: None,
        })
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// The area under the ROC curve of `scored` (score, needs a tool): the share of (tool, no tool)
/// pairs the score orders right, a tie half; `None` without both classes.
pub fn auroc(scored: &[(f32, bool)]) -> Option<f64> {
    let pos: Vec<f32> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    let neg: Vec<f32> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut right = 0.0;
    for &p in &pos {
        for &q in &neg {
            right += match p.partial_cmp(&q) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Equal) => 0.5,
                _ => 0.0,
            };
        }
    }
    Some(right / (pos.len() * neg.len()) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20 rows of 3 features: the first is the label's sign plus noise that never crosses 0, the
    /// others noise that says nothing.
    fn toy() -> Vec<(Vec<f32>, bool)> {
        (0..20)
            .map(|i| {
                let y = i % 2 == 0;
                let jitter = (i % 5) as f32 * 0.1;
                let sign = if y { 1.0 } else { -1.0 };
                (
                    vec![sign * (0.5 + jitter), (i % 3) as f32, (i % 7) as f32 * 0.5],
                    y,
                )
            })
            .collect()
    }

    #[test]
    fn the_probe_separates_a_20_row_toy_set_and_scores_it() {
        let rows = toy();
        let probe = Probe::fit(&rows, vec![0]).unwrap();
        let scored: Vec<(f32, bool)> = (rows.iter())
            .map(|(x, y)| (probe.score(x).unwrap(), *y))
            .collect();
        assert_eq!(auroc(&scored), Some(1.0));
        // Each row lands on its side of the threshold, and a new row far out does too.
        for (s, y) in &scored {
            assert_eq!(*s >= THRESHOLD, *y, "{scored:?}");
        }
        assert!(probe.score(&[3.0, 1.0, 1.0]).unwrap() > 0.9);
        assert!(probe.score(&[-3.0, 1.0, 1.0]).unwrap() < 0.1);
        assert!(probe.score(&[1.0]).is_err());
    }

    #[test]
    fn a_fit_needs_both_classes_and_rows_of_one_length() {
        let one: Vec<_> = toy().into_iter().filter(|r| r.1).collect();
        assert!(Probe::fit(&one, vec![0]).is_err());
        let mut ragged = toy();
        ragged[3].0.pop();
        assert!(Probe::fit(&ragged, vec![0]).is_err());
        assert!(Probe::fit(&[], vec![0]).is_err());
    }

    #[test]
    fn auroc_counts_ordered_pairs_and_ties_half() {
        assert_eq!(auroc(&[(0.9, true), (0.1, false)]), Some(1.0));
        assert_eq!(auroc(&[(0.1, true), (0.9, false)]), Some(0.0));
        assert_eq!(auroc(&[(0.5, true), (0.5, false)]), Some(0.5));
        assert_eq!(auroc(&[(0.5, true)]), None);
    }

    #[test]
    fn the_layers_are_half_and_three_quarters_down_once_each() {
        assert_eq!(layers(32), [16, 24]);
        assert_eq!(layers(36), [18, 27]);
        assert_eq!(layers(1), [0]);
        // Multiplied before the division: 6 is not [3, 3] (one tap), nor 30 [15, 21].
        assert_eq!(layers(6), [3, 4]);
        assert_eq!(layers(14), [7, 10]);
        assert_eq!(layers(30), [15, 22]);
        assert_eq!(layers(2), [1]);
        assert_eq!(layers(3), [1, 2]);
        assert_eq!(layers(u16::MAX), [32767, 49151]);
        assert!(layers(0).is_empty());
    }

    #[test]
    fn a_probe_file_that_does_not_fit_the_model_is_refused_with_the_reason() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("p.json");
        let probe = Probe::fit(&toy(), vec![0]).unwrap();
        let write = |p: &Probe| fs::write(&file, serde_json::to_string(p).unwrap()).unwrap();
        write(&probe);
        assert_eq!(Probe::load(&file, 1).unwrap(), probe);
        let why = |n| Probe::load(&file, n).unwrap_err().to_string();
        assert!(
            why(0).contains("layer 0 is past the model's 0 layers"),
            "{}",
            why(0)
        );
        write(&Probe {
            mean: vec![0.0],
            ..probe.clone()
        });
        assert!(why(1).contains("differ in length"), "{}", why(1));
        write(&Probe {
            schema: "other".into(),
            ..probe
        });
        assert!(why(1).contains("schema 'other'"), "{}", why(1));
        fs::write(&file, "{").unwrap();
        assert!(matches!(
            Probe::load(&file, 1),
            Err(ProbeError::Invalid(..))
        ));
        assert!(matches!(
            Probe::load(&dir.path().join("none.json"), 1),
            Err(ProbeError::Io(..))
        ));
    }
}
