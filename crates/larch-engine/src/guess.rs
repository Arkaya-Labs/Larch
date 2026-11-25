//! Speculative decoding's guesses: a [`Drafter`] proposes the tokens that follow a sequence, the
//! engine checks them in one multi-row step (sched.rs, engine.rs), and [`Costs`] sizes each check
//! from the measured step time and the sequence's acceptance ([`Hits`]).

use larch_kernels::{Buffer, DType, Device, KError, Stream, Tensor};
use larch_model::{
    DflashDrafter, DraftCache, MTP_GUESSES, Model, MtpCache, MtpDrafter, Taps, Transformer,
};

use crate::guard;
use crate::sched::Namespace;

/// Proposes guessed tokens for a decoding sequence; the check step keeps only those the model
/// itself would have produced, so a drafter changes speed, never output.
pub trait Drafter {
    /// Appends to `out` up to `max` tokens guessed to follow `tokens` (the prompt, then the
    /// generated tokens; the last one is not fed yet) of request `id`, which runs under KV
    /// namespace `ns`. Appending none means no guess this step.
    fn propose(&mut self, id: u64, ns: Namespace, tokens: &[u32], max: usize, out: &mut Vec<u32>);

    /// For a drafter whose guesses cost the same at any count and that scores them: proposes as
    /// [`Drafter::propose`] does, with its chance of each guess in `conf`, and returns true. False,
    /// proposing nothing, for any other drafter.
    fn propose_scored(
        &mut self,
        _id: u64,
        _ns: Namespace,
        _tokens: &[u32],
        _max: usize,
        _out: &mut Vec<u32>,
        _conf: &mut Vec<f32>,
    ) -> bool {
        false
    }

    /// Whether its proposals depend on other requests' outputs, which a seeded sampled request
    /// must not take: its guesses decide how its RNG is drawn.
    fn uses_history(&self) -> bool {
        false
    }

    /// Request `id`, which may have proposed under `ns`, ended having generated `output`.
    fn ended(&mut self, _id: u64, _ns: Namespace, _output: &[u32]) {}
}

/// Prompt lookup: finds the latest earlier occurrence of the sequence's last 3 tokens (else 2) and
/// proposes what followed it. The baseline [`crate::suffix::SuffixDrafter`] replaced as the
/// default: the tests measure the suffix drafts against it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PromptLookup;

/// The n-gram lengths tried, longest first: a 3-token match predicts better, a 2-token one hits more.
#[cfg(test)]
const NGRAMS: [usize; 2] = [3, 2];

#[cfg(test)]
impl Drafter for PromptLookup {
    fn propose(&mut self, _: u64, _: Namespace, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        // ponytail: a linear scan per decode step (an 8k context is ~8k compares, microseconds);
        // index the n-grams if contexts reach 100k tokens.
        for n in NGRAMS {
            if tokens.len() <= n {
                continue;
            }
            let tail = &tokens[tokens.len() - n..];
            let found = tokens[..tokens.len() - 1]
                .windows(n)
                .rposition(|w| w == tail);
            if let Some(at) = found {
                let from = at + n;
                out.extend_from_slice(&tokens[from..(from + max).min(tokens.len())]);
                return;
            }
        }
    }
}

/// The most sequences one engine drafts for with its drafter: each holds a cache of the full
/// context, live or parked with its session. A sequence past them guesses with the scheduler's
/// drafter.
// ponytail: capped at four (the brief's "up to 4 chats"); raise when serving more chats with a
// drafter shows in a bench.
pub(crate) const DRAFT_SEATS: usize = 4;

/// The draft seats a load allocates and its fit reserves for a startup `max_batch_seqs`: a live
/// raise leaves the extra sequences to the scheduler's drafter, so one chat pays for one cache.
pub(crate) fn draft_seats(max_batch_seqs: u32) -> usize {
    (max_batch_seqs as usize).clamp(1, DRAFT_SEATS)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seat {
    Free,
    /// Request `id`'s cache, parking under `session` (0: none) when it ends.
    Live {
        id: u64,
        session: u64,
    },
    /// A session's last turn, from engine time `at`: its next turn takes it over.
    Parked {
        session: u64,
        at: u64,
    },
}

/// A load's drafter and each seat's cache.
// One a load: boxing the larger variant saves nothing.
#[allow(clippy::large_enum_variant)]
enum Kind {
    /// A DFlash GGUF beside the target, with every batch row's taps (f32 `[target_layers * rows,
    /// d_model]`, [`Taps::out`]'s layout) and their rows (i32 `0..rows`).
    Dflash {
        d: DflashDrafter,
        caches: Vec<DraftCache>,
        taps: Buffer,
        iota: Buffer,
        d_model: i64,
    },
    /// The target's own MTP block, which holds its taps.
    Mtp {
        d: MtpDrafter,
        caches: Vec<MtpCache>,
    },
}

/// The drafter of a loaded model (a [`DflashDrafter`] or the target's [`MtpDrafter`]) and its
/// sequences' context. Every position a sequence commits (its prompt chunks, decode rows and a
/// check's kept rows) goes through [`Draft::observe`] from the step's taps, so a cache holds
/// exactly its sequence's KV positions; a sequence whose step starts past its cache (a prefix it
/// did not compute here) drops it and guesses with the scheduler's drafter until it ends.
pub(crate) struct Draft {
    kind: Kind,
    seats: Vec<Seat>,
}

/// MTP guesses a check tries when no cap is asked: llama.cpp's `--draft-max` default.
const MTP_DEFAULT_GUESSES: u32 = 3;

/// The guesses a check tries with a DFlash drafter of `block` rows, else the MTP block: `asked`
/// up to the drafter's cap, or its default when 0 asks none. Load and fit size checks by it.
pub(crate) fn drafted(asked: u32, block: Option<u32>) -> u32 {
    let (cap, default) = match block {
        Some(b) => (b - 1, b - 1),
        None => (MTP_GUESSES, MTP_DEFAULT_GUESSES),
    };
    if asked > 0 { asked.min(cap) } else { default }
}

impl Draft {
    /// `seats` caches and the taps beside DFlash drafter `d` ([`crate::fit::drafter_bytes`]).
    pub(crate) fn dflash(
        d: DflashDrafter,
        dev: &Device,
        s: &Stream,
        d_model: u32,
        rows: usize,
        max_context: u32,
        seats: usize,
    ) -> Result<Draft, KError> {
        let caches = (0..seats)
            .map(|_| d.cache(dev, s, max_context))
            .collect::<Result<_, _>>()?;
        let k = d.target_layers().len();
        let taps = Buffer::alloc(dev, 4 * (k * rows * d_model as usize) as u64)?;
        let iota = Buffer::alloc(dev, 4 * rows as u64)?;
        s.upload(&iota, 0, &(0..rows as i32).collect::<Vec<_>>())?;
        Ok(Draft {
            kind: Kind::Dflash {
                d,
                caches,
                taps,
                iota,
                d_model: i64::from(d_model),
            },
            seats: vec![Seat::Free; seats],
        })
    }

    /// `seats` caches beside the target's MTP drafter `d` ([`crate::fit::mtp_bytes`]).
    pub(crate) fn mtp(
        d: MtpDrafter,
        t: &Transformer,
        s: &Stream,
        seats: usize,
    ) -> Result<Draft, KError> {
        let caches = (0..seats)
            .map(|_| d.cache(t, s))
            .collect::<Result<_, _>>()?;
        Ok(Draft {
            kind: Kind::Mtp { d, caches },
            seats: vec![Seat::Free; seats],
        })
    }

    /// The most guesses a check tries for an `asked` cap (0: the drafter's own): DFlash's block
    /// less the anchor, the MTP block's chain length.
    pub(crate) fn guesses(&self, asked: u32) -> u32 {
        drafted(asked, self.block())
    }

    /// A DFlash drafter's block rows; None for the MTP block.
    pub(crate) fn block(&self) -> Option<u32> {
        match &self.kind {
            Kind::Dflash { d, .. } => Some(d.block()),
            Kind::Mtp { .. } => None,
        }
    }

    /// The taps of a step of `rows` batch rows: every row at DFlash's target layers, or every
    /// row's final hidden state for the MTP block.
    pub(crate) fn taps(&self, rows: usize) -> Result<Taps<'_>, KError> {
        let (d, taps, iota, d_model) = match &self.kind {
            Kind::Mtp { d, .. } => return d.taps(rows),
            Kind::Dflash {
                d,
                taps,
                iota,
                d_model,
                ..
            } => (d, taps, iota, *d_model),
        };
        let k = d.target_layers().len() as i64;
        let n = rows as i64;
        Ok(Taps {
            layers: d.target_layers(),
            final_state: false,
            rows: Tensor::new(iota, 0, DType::I32, &[n])?,
            out: Tensor::new(taps, 0, DType::F32, &[k * n, d_model])?,
        })
    }

    /// Positions seat `i`'s cache holds.
    fn len(&self, i: usize) -> u32 {
        match &self.kind {
            Kind::Dflash { caches, .. } => caches[i].len(),
            Kind::Mtp { caches, .. } => caches[i].len(),
        }
    }

    fn seat(&self, id: u64) -> Option<usize> {
        (self.seats.iter()).position(|s| matches!(*s, Seat::Live { id: i, .. } if i == id))
    }

    /// Request `id`'s seat, its cache cut back to `start`, the first position its step feeds; a
    /// new seat at position 0, or its session's parked one, else none.
    fn sync(&mut self, id: u64, session: u64, start: u32) -> Option<usize> {
        let at = self.seat(id).or_else(|| {
            let parked = (session != 0)
                .then(|| {
                    (self.seats.iter())
                        .position(|s| matches!(*s, Seat::Parked { session: p, .. } if p == session))
                })
                .flatten();
            let free = || {
                let oldest = (self.seats.iter().enumerate())
                    .filter_map(|(i, s)| match *s {
                        Seat::Parked { at, .. } => Some((at, i)),
                        _ => None,
                    })
                    .min();
                (self.seats.iter().position(|s| *s == Seat::Free)).or(oldest.map(|o| o.1))
            };
            // A parked turn's positions are this session's prompt as far as the KV it reuses.
            // ponytail: not checked token by token; a session whose history was edited just
            // guesses worse (a guess never changes output).
            let i = parked.or_else(|| (start == 0).then(free).flatten())?;
            self.seats[i] = Seat::Live { id, session };
            Some(i)
        })?;
        // A cut a ring cannot keep empties it, so the check follows the cut.
        match &mut self.kind {
            Kind::Dflash { caches, .. } => caches[at].truncate(start),
            Kind::Mtp { caches, .. } => caches[at].truncate(start),
        }
        if self.len(at) < start {
            self.seats[at] = Seat::Free;
            return None;
        }
        Some(at)
    }

    /// Feeds request `id`'s committed rows of the step whose taps [`Self::taps`] described for
    /// `n` rows: batch rows `rows`, from position `start`, holding `tokens` (the MTP block reads
    /// them). `untapped`: the step ran without the drafter's taps, so the request loses its seat.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe(
        &mut self,
        s: &Stream,
        model: &dyn Model,
        id: u64,
        session: u64,
        start: usize,
        n: usize,
        rows: std::ops::Range<u32>,
        tokens: &[u32],
        untapped: bool,
    ) -> Result<(), KError> {
        if untapped {
            if let Some(i) = self.seat(id) {
                self.seats[i] = Seat::Free;
            }
            return Ok(());
        }
        let Some(i) = self.sync(id, session, start as u32) else {
            return Ok(());
        };
        match &mut self.kind {
            Kind::Dflash {
                d,
                caches,
                taps,
                d_model,
                ..
            } => {
                let k = d.target_layers().len() as i64;
                let taps = Tensor::new(taps, 0, DType::F32, &[k * n as i64, *d_model])?;
                d.observe(s, &mut caches[i], &taps, rows)
            }
            Kind::Mtp { d, caches } => {
                let t = model.transformer().ok_or(KError::InvalidArg)?;
                d.observe(t, s, &mut caches[i], n, rows, tokens)
            }
        }
    }

    /// Releases the seats of requests that ended (`gone`): a session's parks, at engine time `now`,
    /// replacing an older park of it.
    pub(crate) fn finished(&mut self, gone: impl Fn(u64) -> bool, now: u64) {
        for i in 0..self.seats.len() {
            let Seat::Live { id, session } = self.seats[i] else {
                continue;
            };
            if !gone(id) {
                continue;
            }
            for s in &mut self.seats {
                if matches!(*s, Seat::Parked { session: p, .. } if p == session && p != 0) {
                    *s = Seat::Free;
                }
            }
            self.seats[i] = match session {
                0 => Seat::Free,
                session => Seat::Parked { session, at: now },
            };
        }
    }

    /// Each seated request and the positions its cache holds (tests).
    #[cfg(test)]
    pub(crate) fn live(&self) -> Vec<(u64, u32)> {
        (self.seats.iter().enumerate())
            .filter_map(|(i, s)| match *s {
                Seat::Live { id, .. } => Some((id, self.len(i))),
                _ => None,
            })
            .collect()
    }

    /// Each seated request, its positions, and the logits of two guesses after token 1 (tests).
    #[cfg(test)]
    pub(crate) fn guess_logits(&self, s: &Stream, model: &dyn Model) -> Vec<(u64, u32, Vec<f32>)> {
        let guess = |(id, len): (u64, u32)| {
            let i = self.seat(id).expect("a seated request");
            let logits = match &self.kind {
                Kind::Dflash { d, caches, .. } => {
                    let (embed, head) = model.embed_and_head().expect("a model that embeds");
                    d.propose(s, &caches[i], &embed, &head, 1, 2).unwrap();
                    d.last_logits(s, 2).unwrap()
                }
                Kind::Mtp { d, caches } => {
                    let t = model.transformer().expect("the model with the block");
                    d.propose(t, s, &caches[i], 1, 2).unwrap();
                    d.last_logits(s).unwrap()
                }
            };
            (id, len, logits)
        };
        self.live().into_iter().map(guess).collect()
    }

    /// The sessions parked (tests).
    #[cfg(test)]
    pub(crate) fn parked(&self) -> Vec<u64> {
        (self.seats.iter())
            .filter_map(|s| match *s {
                Seat::Parked { session, .. } => Some(session),
                _ => None,
            })
            .collect()
    }
}

/// One step's [`Drafter`]: the load's drafter for a request whose cache holds every position but
/// the last token's, else nothing, and the scheduler's own drafter proposes. The load's drafter
/// reads the whole context, quoted text included, so it is not combined with that drafter on one
/// sequence: picking between two proposals needs a confidence neither gives, and a second check
/// costs a step (Stage 5's adaptive length).
pub(crate) struct DraftStep<'a> {
    pub(crate) draft: &'a Draft,
    pub(crate) s: &'a Stream,
    pub(crate) model: &'a dyn Model,
    /// The first device error, which fails the step after scheduling.
    pub(crate) err: Option<KError>,
}

impl DraftStep<'_> {
    /// [`Drafter::propose`], with a DFlash drafter's chance of each guess in `conf`.
    fn guesses(
        &mut self,
        id: u64,
        tokens: &[u32],
        max: usize,
        out: &mut Vec<u32>,
        conf: &mut Vec<f32>,
    ) {
        let dr = self.draft;
        let seated = dr
            .seat(id)
            .filter(|&i| dr.len(i) as usize + 1 == tokens.len());
        let Some(i) = seated.filter(|_| self.err.is_none()) else {
            return;
        };
        let anchor = *tokens.last().expect("a decoding sequence has tokens");
        let got = match &dr.kind {
            Kind::Dflash { d, caches, .. } => {
                let Some((embed, head)) = self.model.embed_and_head() else {
                    return;
                };
                let n = max.min(d.block() as usize - 1) as u32;
                d.propose(self.s, &caches[i], &embed, &head, anchor, n)
            }
            Kind::Mtp { d, caches } => {
                let Some(t) = self.model.transformer() else {
                    return;
                };
                let n = max.min(MTP_GUESSES as usize) as u32;
                d.propose(t, self.s, &caches[i], anchor, n)
                    .map(|g| (g, Vec::new()))
            }
        };
        match got {
            Ok((g, p)) => {
                out.extend(g);
                conf.extend(p);
            }
            Err(e) => self.err = Some(e),
        }
    }
}

impl Drafter for DraftStep<'_> {
    fn propose(&mut self, id: u64, _: Namespace, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        self.guesses(id, tokens, max, out, &mut Vec::new());
    }

    fn propose_scored(
        &mut self,
        id: u64,
        _: Namespace,
        tokens: &[u32],
        max: usize,
        out: &mut Vec<u32>,
        conf: &mut Vec<f32>,
    ) -> bool {
        // DFlash computes its whole block whatever it is asked; the MTP block's chain grows with it.
        let dflash = matches!(self.draft.kind, Kind::Dflash { .. });
        if dflash {
            self.guesses(id, tokens, max, out, conf);
        }
        dflash
    }
}

/// Guesses a sequence's hit rate is judged over.
const WINDOW: u32 = 32;
/// Decode steps a sequence first stops guessing for after a low hit rate; each further low window
/// doubles it up to [`MAX_BACKOFF`], and a good one resets it.
const BACKOFF: u32 = 16;
const MAX_BACKOFF: u32 = 1024;
/// What a sequence's acceptance evidence keeps a round: about its last 10 rounds count. A step
/// without a check leaves it alone; a sequence the pick leaves plain retries by probing.
const ACCEPT_DECAY: f64 = 0.9;
/// The highest acceptance one-guess probes can show, every one kept: a probe runs only where
/// this would flip the pick (11/12 at this decay).
pub(crate) const PROBE_TOP: f64 = {
    let kept = 1.0 / (1.0 - ACCEPT_DECAY);
    (kept + 1.0) / (kept + 2.0)
};

/// One sequence's guess record: below 1 kept in 4 judged over a window it stops guessing for a
/// while, so a sequence whose guesses miss runs plain decode steps, as it did before guessing. Its
/// [`Hits::acceptance`] sizes each check by the measured cost ([`Costs::pick`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hits {
    tried: u32,
    kept: u32,
    /// Decode steps left without guessing.
    skip: u32,
    backoff: u32,
    /// Decayed kept guesses, and rounds that stopped at a wrong guess.
    accepted: f64,
    missed: f64,
    /// Decayed draft confidence of the guesses scored rounds judged, those guesses, and the
    /// scored rounds.
    conf: f64,
    judged: f64,
    scored: u32,
}

impl Default for Hits {
    fn default() -> Hits {
        Hits {
            tried: 0,
            kept: 0,
            skip: 0,
            backoff: BACKOFF,
            accepted: 0.0,
            missed: 0.0,
            conf: 0.0,
            judged: 0.0,
            scored: 0,
        }
    }
}

impl Hits {
    /// Whether this decode step may guess; counts a skipped step.
    pub(crate) fn may_guess(&mut self) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        true
    }

    /// Records a check round of `tried` guesses that kept `kept`.
    pub(crate) fn round(&mut self, tried: u32, kept: u32) {
        self.accepted = self.accepted * ACCEPT_DECAY + f64::from(kept);
        self.missed = self.missed * ACCEPT_DECAY + f64::from(u8::from(kept < tried));
        // Guesses past the first miss were never judged.
        self.refused(kept + u32::from(kept < tried), kept);
    }

    /// Records `tried` guesses of a round that kept `kept` toward the hit rate only: a check the
    /// model refused for room says nothing about the guesses.
    pub(crate) fn refused(&mut self, tried: u32, kept: u32) {
        self.tried += tried;
        self.kept += kept;
        if self.tried < WINDOW {
            return;
        }
        // ponytail: fixed 1 in 4 still overrides a pick below it at small batches (one chat at
        // a = 0.2 picks k=1 at #281's row cost), worth under 1/3 token a step; gate it on the pick
        // if a bench shows that band.
        if self.kept * 4 < self.tried {
            self.skip = self.backoff;
            self.backoff = (self.backoff * 2).min(MAX_BACKOFF);
        } else {
            self.backoff = BACKOFF;
        }
        (self.tried, self.kept) = (0, 0);
    }

    /// The chance its next guess is kept: each round keeps guesses until its first miss, so
    /// kept / (kept + misses), from a prior of one of each (1/2).
    pub(crate) fn acceptance(&self) -> f64 {
        (self.accepted + 1.0) / (self.accepted + self.missed + 2.0)
    }

    /// Records the draft's chance of each guess a round judged (up to its first miss, as
    /// [`Hits::round`] counts them); none is no round.
    pub(crate) fn scored(&mut self, conf: &[f32]) {
        if conf.is_empty() {
            return;
        }
        let sum: f64 = conf.iter().map(|&p| f64::from(p).clamp(0.0, 1.0)).sum();
        self.conf = self.conf * ACCEPT_DECAY + sum;
        self.judged = self.judged * ACCEPT_DECAY + conf.len() as f64;
        self.scored = self.scored.saturating_add(1);
    }

    /// What scales the draft's confidence to the chance a guess is kept: the acceptance over the
    /// mean confidence of the guesses judged, within [0.5, 1.5], once [`MIN_SAMPLES`] scored rounds
    /// show it; 1 before.
    // ponytail: the acceptance counts the suffix drafts' rounds too (LARCH_DRAFT_HYBRID); keep a
    // scored rounds' own if hybrid with confidences shows in a bench.
    pub(crate) fn calibration(&self) -> f64 {
        if self.scored < MIN_SAMPLES {
            return 1.0;
        }
        (self.acceptance() * self.judged / self.conf).clamp(0.5, 1.5)
    }
}

/// Buckets of a step's decode rows and of its guessed rows: 0, 1, 2, 3-4, 5-8, 9-16, 17 and up.
const BUCKETS: usize = 7;
/// Samples a bucket needs before its cost counts, and scored rounds before a calibration does.
const MIN_SAMPLES: u32 = 4;
/// The weight of a new sample once a bucket has 1 / COST_WEIGHT of them: about its last 8 count.
const COST_WEIGHT: f64 = 0.125;
/// Every this many steps that may check explores ([`Costs::pick`]).
pub(crate) const EXPLORE: u64 = 16;

fn bucket(n: usize) -> usize {
    match n {
        0 => 0,
        n => (1 + guard::bucket(n as u32)).min(BUCKETS - 1),
    }
}

/// The tokens a check of `k` guesses emits on average when each is kept with chance `a`, the
/// first miss ending it: 1 + a + ... + a^k.
pub(crate) fn expected(a: f64, k: usize) -> f64 {
    (0..=k as i32).map(|i| a.powi(i)).sum()
}

/// The tokens a check of the first `k` guesses emits on average when guess i is kept with chance
/// c_i = `scale` times its draft confidence (at most 1), given those before it were: 1 + c_0 +
/// c_0 c_1 + ... (EAGLE-2's draft confidence as acceptance).
pub(crate) fn chained(conf: &[f32], scale: f64, k: usize) -> f64 {
    let c = conf[..k]
        .iter()
        .map(|&p| (f64::from(p).clamp(0.0, 1.0) * scale).min(1.0));
    1.0 + c
        .scan(1.0, |run, c| {
            *run *= c;
            Some(*run)
        })
        .sum::<f64>()
}

/// Measured step time by (decode rows, guessed rows) bucket, an exponential average over the
/// steps that run anyway: plain decode steps fill 0 guessed rows, check steps the rest. The 0
/// column prices a plain step of this load as it runs, which on a guessing hybrid is eager at the
/// pinned splits: it compares k against k=0 of the same load, never against guessing off.
// ponytail: a bucket no step picks keeps its last measurement and buckets ignore context length,
// so a check bucket measured at 20k KV tokens stays high after a short chat refreshes k=0; age
// the buckets if a bench shows speculation off where it pays (long then short chats, a batch or
// clock change).
#[derive(Clone, Debug, Default)]
pub(crate) struct Costs {
    /// (mean ns, samples).
    t: [[(f64, u32); BUCKETS]; BUCKETS],
    /// Tests' fixed tables ignore the CPU's real step times.
    #[cfg(test)]
    frozen: bool,
}

impl Costs {
    /// Records a step of `decode` decode rows (checked sequences included) and `guessed` guessed
    /// rows that took `ns`; 0 is no measurement.
    pub(crate) fn record(&mut self, decode: usize, guessed: usize, ns: u64) {
        #[cfg(test)]
        if self.frozen {
            return;
        }
        if ns == 0 {
            return;
        }
        let (mean, n) = &mut self.t[bucket(decode)][bucket(guessed)];
        *n = n.saturating_add(1);
        // The first samples average evenly, so the first one does not linger.
        *mean += (ns as f64 - *mean) * (1.0 / f64::from(*n)).max(COST_WEIGHT);
    }

    fn get(&self, decode: usize, guessed: usize) -> Option<f64> {
        let (mean, n) = self.t[bucket(decode)][bucket(guessed)];
        (n >= MIN_SAMPLES).then_some(mean)
    }

    /// [`Costs::get`], else for an unmeasured bucket between two measured ones of the same decode
    /// bucket, their ns interpolated linearly at `guessed` rows; never past the largest measured.
    // A bucket stands at its top row count (0, 1, 2, 4, 8, 16, 32), as `fixed` prices it: a check
    // at the drafter's cap sits at or near its bucket's top (7 guesses read as 8).
    fn estimate(&self, decode: usize, guessed: usize) -> Option<f64> {
        if let Some(ns) = self.get(decode, guessed) {
            return Some(ns);
        }
        let row = &self.t[bucket(decode)];
        let top = |b: usize| f64::from((1u32 << b) / 2);
        let g = bucket(guessed);
        let lo = (0..g).rev().find(|&b| row[b].1 >= MIN_SAMPLES)?;
        let hi = (g + 1..BUCKETS).find(|&b| row[b].1 >= MIN_SAMPLES)?;
        let (x0, x1) = (top(lo), top(hi));
        let f = (guessed as f64 - x0) / (x1 - x0);
        Some(row[lo].0 + (row[hi].0 - row[lo].0) * f)
    }

    /// The guesses, up to `max`, for one sequence of a step of `decode` decoding sequences beside
    /// `guessed` guessed rows already planned, whose guesses emit `gain` tokens beyond one a
    /// sequence: the k whose step emits the most expected tokens per measured ns at acceptance
    /// `a` (an unmeasured k between measured ones priced by interpolation), the smaller on a tie.
    /// 0 when the step's cost without it is unmeasured. Exploring, the largest k whose bucket is
    /// unmeasured, if any, so every bucket gets measured.
    pub(crate) fn pick(
        &self,
        decode: usize,
        guessed: usize,
        gain: f64,
        a: f64,
        max: usize,
        explore: bool,
    ) -> usize {
        self.pick_by(decode, guessed, gain, |k| expected(a, k), max, explore)
    }

    /// [`Costs::pick`] with `e(k)` the tokens a check of k guesses emits on average.
    pub(crate) fn pick_by(
        &self,
        decode: usize,
        guessed: usize,
        gain: f64,
        e: impl Fn(usize) -> f64,
        max: usize,
        explore: bool,
    ) -> usize {
        if explore
            && let Some(k) = (1..=max)
                .rev()
                .find(|&k| self.get(decode, guessed + k).is_none())
        {
            return k;
        }
        if self.get(decode, guessed).is_none() {
            return 0;
        }
        let mut best = (0, 0.0);
        for k in 0..=max {
            let Some(ns) = self.estimate(decode, guessed + k) else {
                continue;
            };
            let per_ns = (decode as f64 + gain + e(k) - 1.0) / ns;
            if per_ns > best.1 {
                best = (k, per_ns);
            }
        }
        best.0
    }

    /// The (mean ns, samples) of a bucket (tests).
    #[cfg(test)]
    pub(crate) fn sample(&self, decode: usize, guessed: usize) -> (f64, u32) {
        self.t[bucket(decode)][bucket(guessed)]
    }

    /// A frozen table: `ns(decode, guessed)` at each bucket's largest row count (tests).
    #[cfg(test)]
    pub(crate) fn fixed(ns: impl Fn(usize, usize) -> f64) -> Costs {
        let top = |b: usize| if b == 0 { 0 } else { 1 << (b - 1) };
        let mut c = Costs {
            frozen: true,
            ..Costs::default()
        };
        for (d, row) in c.t.iter_mut().enumerate() {
            for (g, cell) in row.iter_mut().enumerate() {
                *cell = (ns(top(d), top(g)), MIN_SAMPLES);
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn propose(tokens: &[u32], max: usize) -> Vec<u32> {
        let mut out = Vec::new();
        PromptLookup.propose(0, (0, 0), tokens, max, &mut out);
        out
    }

    #[test]
    fn prompt_lookup_copies_what_followed_the_latest_match() {
        // "1 2 3" appears twice; the later occurrence wins.
        assert_eq!(
            propose(&[1, 2, 3, 4, 5, 1, 2, 3, 9, 8, 7, 1, 2, 3], 2),
            [9, 8]
        );
        // No 3-gram match: the 2-gram "5 6" falls back.
        assert_eq!(propose(&[5, 6, 7, 0, 5, 6], 8), [7, 0, 5, 6]);
        // Nothing matches, or too short.
        assert!(propose(&[1, 2, 3, 4], 8).is_empty());
        assert!(propose(&[1, 1], 8).is_empty());
        // A run proposes its own continuation, bounded by `max` and the text.
        assert_eq!(propose(&[7, 7, 7, 7], 8), [7]);
    }

    #[test]
    fn a_low_hit_rate_backs_off_and_retries() {
        let misses = |h: &mut Hits| (0..WINDOW).for_each(|_| h.round(8, 0));
        let mut h = Hits::default();
        h.round(8, 0);
        assert!(h.may_guess());
        (0..3).for_each(|_| h.round(8, 1));
        (0..25).for_each(|_| h.round(8, 0));
        // 3 of 32 judged kept: off for 16 steps, then on again.
        assert!((0..16).all(|_| !h.may_guess()));
        assert!(h.may_guess());
        misses(&mut h);
        assert!((0..32).all(|_| !h.may_guess()));
        assert!(h.may_guess());
        // A good window resets the backoff.
        (0..16).for_each(|_| h.round(8, 1));
        misses(&mut h);
        assert!((0..16).all(|_| !h.may_guess()));
        assert!(h.may_guess());
    }

    /// Guesses past a round's first miss were never judged: keeping 1 of 8 and then missing is
    /// 1 in 2 of those judged, where `pick` may well want those checks.
    #[test]
    fn guesses_past_the_first_miss_do_not_count_toward_backoff() {
        let mut h = Hits::default();
        for _ in 0..100 {
            h.round(8, 1);
        }
        assert!(h.may_guess());
    }

    #[test]
    fn acceptance_counts_guesses_kept_before_the_first_miss() {
        let mut h = Hits::default();
        assert_eq!(h.acceptance(), 0.5);
        // All 8 kept: no miss.
        h.round(8, 8);
        assert_eq!(h.acceptance(), 9.0 / 10.0);
        // 2 kept, then a miss: the guesses past it were never judged.
        h.round(8, 2);
        let (k, m) = (8.0 * ACCEPT_DECAY + 2.0, 1.0);
        assert!((h.acceptance() - (k + 1.0) / (k + m + 2.0)).abs() < 1e-12);
        // A refusal is no evidence.
        let a = h.acceptance();
        h.refused(8, 0);
        assert_eq!(h.acceptance(), a);
    }

    #[test]
    fn buckets_double() {
        let b: Vec<usize> = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 64].map(bucket).into();
        assert_eq!(b, [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6]);
        assert!((expected(0.5, 2) - 1.75).abs() < 1e-12);
        assert_eq!(expected(0.0, 8), 1.0);
    }

    /// k maximises the step's expected tokens per ns: none when a check costs more than its rows
    /// can emit, the most when rows are nearly free, between where acceptance sets it.
    #[test]
    fn pick_maximises_tokens_per_ns() {
        // A row costs what a step does: no check pays even at acceptance 1, as on the CPU.
        let rows = Costs::fixed(|d, g| (d + g) as f64);
        assert_eq!(rows.pick(1, 0, 0.0, 1.0, 8, false), 0);
        // Rows nearly free: as many as allowed at a high acceptance, none at 0.
        let free = Costs::fixed(|d, g| 100.0 + (d + g) as f64);
        assert_eq!(free.pick(1, 0, 0.0, 0.9, 8, false), 8);
        assert_eq!(free.pick(1, 0, 0.0, 0.0, 8, false), 0);
        // Rows at a quarter of a step (CUDA's 8-row check ~1.45x a row, #281, at 1/16 a row):
        // high acceptance takes the 8 bucket, low takes few or none.
        let cuda = Costs::fixed(|d, g| 1.0 + 0.25 * (d + g) as f64);
        assert_eq!(cuda.pick(1, 0, 0.0, 0.95, 8, false), 8);
        assert_eq!(cuda.pick(1, 0, 0.0, 0.6, 8, false), 2);
        assert_eq!(cuda.pick(1, 0, 0.0, 0.1, 8, false), 0);
        // At 8 chats a step's rows dwarf what one sequence's guesses add.
        assert_eq!(cuda.pick(8, 0, 0.0, 0.5, 8, false), 0);
        // Unmeasured: no guess, unless exploring, which takes the largest unmeasured bucket.
        let mut c = Costs::default();
        assert_eq!(c.pick(1, 0, 0.0, 1.0, 8, false), 0);
        assert_eq!(c.pick(1, 0, 0.0, 1.0, 8, true), 8);
        for _ in 0..MIN_SAMPLES {
            c.record(1, 8, 10);
            c.record(1, 0, 10);
        }
        assert_eq!(c.pick(1, 0, 0.0, 1.0, 8, true), 4);
        assert_eq!(c.pick(1, 0, 0.0, 1.0, 8, false), 8);
        c.record(1, 0, 0);
        assert_eq!(c.t[1][0], (10.0, MIN_SAMPLES));
    }

    /// E(k) chains each guess's calibrated confidence, and the pick follows where it falls off.
    #[test]
    fn confidence_sizes_the_check() {
        assert!((chained(&[0.5, 0.5], 1.0, 2) - expected(0.5, 2)).abs() < 1e-12);
        assert_eq!(chained(&[0.9], 1.0, 0), 1.0);
        // Scaled chances stay chances: 1 + 1 + 0.75.
        assert!((chained(&[0.8, 0.5, 0.1], 1.5, 2) - 2.75).abs() < 1e-12);
        let cuda = Costs::fixed(|d, g| 1.0 + 0.25 * (d + g) as f64);
        let pick = |c: &[f32]| cuda.pick_by(1, 0, 0.0, |k| chained(c, 1.0, k), c.len(), false);
        assert_eq!(pick(&[0.95; 8]), 8);
        assert_eq!(pick(&[0.9, 0.5, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]), 2);
        assert_eq!(pick(&[0.05; 8]), 0);
        // An overconfident draft (0.75 where half are kept) is scaled down once rounds show it.
        let mut h = Hits::default();
        for _ in 0..MIN_SAMPLES {
            assert_eq!(h.calibration(), 1.0);
            h.round(8, 1);
            h.scored(&[0.75, 0.75]);
        }
        assert!((h.calibration() - 0.5 / 0.75).abs() < 1e-12);
    }

    /// With only plain steps and full checks measured, a short check is priced between them, so
    /// a confident first two guesses check 2 instead of all or none.
    #[test]
    fn unmeasured_checks_interpolate_between_measured_ones() {
        let mut c = Costs {
            frozen: true,
            ..Costs::default()
        };
        c.t[bucket(1)][bucket(0)] = (22.5e6, MIN_SAMPLES);
        c.t[bucket(1)][bucket(7)] = (50e6, MIN_SAMPLES);
        let conf = [0.9, 0.9, 0.1, 0.05, 0.05, 0.05, 0.05];
        let e = |k| chained(&conf, 1.0, k);
        assert_eq!(c.pick_by(1, 0, 0.0, e, 7, false), 2);
        assert!((c.estimate(1, 2).unwrap() - (22.5e6 + 27.5e6 * 2.0 / 8.0)).abs() < 1.0);
        // Measured wins, and nothing is priced past the largest measured bucket.
        c.t[bucket(1)][bucket(2)] = (40e6, MIN_SAMPLES);
        assert_eq!(c.estimate(1, 2), Some(40e6));
        assert_eq!(c.estimate(1, 9), None);
    }
}
