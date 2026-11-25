//! The host half of sampling (architecture 9.6), over the top-64 candidates `sample_prep` returns
//! (largest first, ties to the lower id, repetition penalty applied). llama.cpp's order: top-k, then
//! top-p on the untempered distribution, then temperature. Logprobs use the full-vocabulary
//! log-sum-exp, so they are exact for the returned tokens.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::RngCore;

use crate::{SampledToken, SamplingParams};

pub(crate) const CANDIDATES: usize = 64;

/// One row of `sample_prep` output.
pub(crate) struct Row<'a> {
    pub(crate) vals: &'a [f32],
    pub(crate) ids: &'a [i32],
    pub(crate) lse: f32,
}

pub(crate) fn sample(
    row: &Row,
    p: &SamplingParams,
    logprobs: u8,
    ban: &[u32],
    rng: &mut ChaCha8Rng,
) -> SampledToken {
    check(row, p, logprobs, ban, None, rng)
}

/// [`sample`] that first checks a guessed token (speculative decoding with a deterministic drafter,
/// q one-hot): it keeps the guess with probability p(guess), else draws from p with the guess
/// removed and renormalised, so the token returned is distributed exactly as `sample`'s. p is the
/// distribution `sample` draws from (after the ban, top-k, top-p and temperature); a guess outside
/// it has p 0 and is never kept. With no guess this is `sample`, draw for draw. A greedy row
/// ignores the guess: the caller keeps it when it equals the pick.
pub(crate) fn check(
    row: &Row,
    p: &SamplingParams,
    logprobs: u8,
    ban: &[u32],
    guess: Option<u32>,
    rng: &mut ChaCha8Rng,
) -> SampledToken {
    // ignore_eos (9.7): a banned id has a -inf logit, as under llama-server's ignore_eos, so it is
    // never drawn. Logprobs keep the full-vocabulary LSE and the top list still shows it, as
    // llama-server's default n_probs reports the model's distribution.
    if p.temperature <= 0.0 {
        let pick = row
            .ids
            .iter()
            .position(|&id| !ban.contains(&(id as u32)))
            .expect("all candidates banned");
        return SampledToken {
            id: row.ids[pick] as u32,
            logprob: row.vals[pick] - row.lse,
            top: top(row, usize::from(logprobs)),
        };
    }
    let (mut vals, mut ids) = ([0f32; CANDIDATES], [-1i32; CANDIDATES]);
    let masked;
    let pool = if ban.is_empty() {
        row
    } else {
        let mut n = 0;
        for (&id, &v) in row.ids.iter().zip(row.vals) {
            if !ban.contains(&(id as u32)) {
                (vals[n], ids[n]) = (v, id);
                n += 1;
            }
        }
        masked = Row {
            vals: &vals[..n],
            ids: &ids[..n],
            lse: row.lse,
        };
        &masked
    };
    // Candidates past the vocabulary are padded with id -1.
    let valid = pool.ids.iter().take_while(|&&id| id >= 0).count().max(1);
    let k = if p.top_k == 0 {
        valid
    } else {
        (p.top_k as usize).min(valid)
    };
    let pick = {
        let logit = |i: usize| f64::from(pool.vals[i] - pool.vals[0]);
        // top-p: the shortest prefix whose softmax mass reaches top_p.
        let total: f64 = (0..k).map(|i| logit(i).exp()).sum();
        let mut mass = 0.0;
        let n = (0..k)
            .take_while(|&i| {
                let before = mass;
                mass += logit(i).exp() / total;
                before < f64::from(p.top_p)
            })
            .count()
            .max(1);
        // Temperature, then draw.
        let w = |i: usize| (logit(i) / f64::from(p.temperature)).exp();
        let mut unit = || (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let g = guess.and_then(|g| (0..n).position(|i| pool.ids[i] as u32 == g));
        match g {
            Some(g) if unit() * (0..n).map(w).sum::<f64>() < w(g) => g,
            _ => {
                // The residual: p without the rejected guess.
                let w = |i: usize| if Some(i) == g { 0.0 } else { w(i) };
                let u = unit() * (0..n).map(w).sum::<f64>();
                let mut acc = 0.0;
                (0..n)
                    .position(|i| {
                        acc += w(i);
                        u < acc
                    })
                    .unwrap_or_else(|| (0..n).rev().find(|&i| Some(i) != g).unwrap_or(n - 1))
            }
        }
    };
    SampledToken {
        id: pool.ids[pick] as u32,
        logprob: pool.vals[pick] - row.lse,
        // Top logprobs are the most likely tokens whatever the sampling filter.
        top: top(row, usize::from(logprobs)),
    }
}

/// The row's `n` most likely tokens, most likely first, with their logprobs; never padding.
pub(crate) fn top(row: &Row, n: usize) -> Vec<(u32, f32)> {
    row.ids
        .iter()
        .zip(row.vals)
        .take_while(|(id, _)| **id >= 0)
        .take(n)
        .map(|(&id, &v)| (id as u32, v - row.lse))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;

    fn params(temperature: f32, top_p: f32, top_k: u32) -> SamplingParams {
        SamplingParams {
            temperature,
            top_p,
            top_k,
            repeat_penalty: 1.0,
            seed: None,
        }
    }

    #[test]
    fn greedy_top_p_and_temperature() {
        let vals = [3.0f32, 2.0, 1.0, -1.0];
        let ids = [7, 3, 9, -1];
        let lse = (3f32.exp() + 2f32.exp() + 1f32.exp() + (-1f32).exp()).ln();
        let row = Row {
            vals: &vals,
            ids: &ids,
            lse,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let g = sample(&row, &params(0.0, 1.0, 0), 2, &[], &mut rng);
        assert_eq!(g.id, 7);
        assert!((g.logprob - (3.0 - lse)).abs() < 1e-6);
        assert_eq!(g.top.len(), 2);
        // Top logprobs ignore top_k but never include padding.
        assert_eq!(
            sample(&row, &params(0.0, 1.0, 1), 20, &[], &mut rng)
                .top
                .len(),
            3
        );
        // top_p 0.5: the first candidate alone holds 66 % of the mass.
        assert!((0..100).all(|_| sample(&row, &params(1.0, 0.5, 0), 0, &[], &mut rng).id == 7));
        // top_k 2 at a high temperature: only ids 7 and 3, both drawn.
        let drawn: Vec<u32> = (0..400)
            .map(|_| sample(&row, &params(5.0, 1.0, 2), 0, &[], &mut rng).id)
            .collect();
        assert!(
            drawn.iter().all(|&i| i == 7 || i == 3) && drawn.contains(&3) && drawn.contains(&7)
        );
        // Padding ids (-1) are never drawn.
        assert!(
            (0..400).all(|_| sample(&row, &params(50.0, 1.0, 0), 0, &[], &mut rng).id != u32::MAX)
        );
    }

    /// ignore_eos: a banned id is never drawn, greedy or not; logprobs keep the full-vocabulary
    /// LSE and the top list still shows the banned id.
    #[test]
    fn banned_ids_are_never_drawn() {
        let vals = [3.0f32, 2.0, 1.0, -1.0];
        let ids = [7, 3, 9, -1];
        let lse = (3f32.exp() + 2f32.exp() + 1f32.exp() + (-1f32).exp()).ln();
        let row = Row {
            vals: &vals,
            ids: &ids,
            lse,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let g = sample(&row, &params(0.0, 1.0, 0), 3, &[7], &mut rng);
        assert_eq!(g.id, 3);
        assert!((g.logprob - (2.0 - lse)).abs() < 1e-6);
        assert_eq!(g.top[0].0, 7);
        assert!(
            (0..400).all(|_| sample(&row, &params(50.0, 1.0, 0), 0, &[7, 9], &mut rng).id == 3)
        );
    }

    /// Accept-or-resample draws each token with `sample`'s probability, whatever the guess: a
    /// likely one, an unlikely one, one top-k cut, one outside the candidates.
    #[test]
    fn a_checked_guess_keeps_the_distribution() {
        let vals = [2.0f32, 1.5, 0.5, -1.0];
        let ids = [7, 3, 9, 4];
        let row = Row {
            vals: &vals,
            ids: &ids,
            lse: 0.0,
        };
        let p = params(0.8, 1.0, 3);
        let w: Vec<f64> = vals[..3]
            .iter()
            .map(|&v| (f64::from(v - vals[0]) / 0.8).exp())
            .collect();
        let total: f64 = w.iter().sum();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 200_000;
        for guess in [7, 9, 4, 100] {
            let mut counts = [0u32; 4];
            for _ in 0..n {
                let id = check(&row, &p, 0, &[], Some(guess), &mut rng).id;
                counts[ids.iter().position(|&i| i as u32 == id).unwrap()] += 1;
            }
            assert_eq!(counts[3], 0);
            for i in 0..3 {
                let f = f64::from(counts[i]) / f64::from(n);
                assert!(
                    (f - w[i] / total).abs() < 0.005,
                    "guess {guess}: {counts:?}"
                );
            }
        }
    }
}
