//! Suffix drafts (SuffixDecoding, Oliaro et al., 2024): a suffix automaton over each request's
//! tokens, and one per namespace over recent finished outputs. The longest match of the sequence's
//! tail proposes its most frequent continuation while that continuation's estimated probability
//! holds. Agent loops quote tool output and code back, and repeat earlier turns' outputs; no draft
//! model is needed to copy them.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use crate::guess::Drafter;
use crate::sched::Namespace;

/// Ends each output in the global automaton: never a vocabulary id, so never matched or proposed.
const SEP: u32 = u32::MAX;
/// The shortest match that proposes, as prompt lookup's 2-gram: one token predicts too little.
const MIN_MATCH: u32 = 2;
/// The tail tokens a match is looked for in: a longer match than this predicts no better.
const TAIL: usize = 32;
/// Draft tokens a matched token buys: a short match proposes a short draft.
const PER_MATCH: usize = 2;
/// Drafting stops once the draft's estimated probability (the product of each token's share of
/// its context's continuations) falls below this.
const FLOOR: f32 = 0.3;
/// Suffix links a new position's occurrence count climbs.
// ponytail: a run of one repeated token makes the climb as long as the run (quadratic); past 256
// links the shortest contexts undercount, which only weakens their drafts. Count offline (a pass
// over the link tree) if runs that long show up in agent transcripts.
const COUNT_LINKS: usize = 256;
/// Tokens of finished outputs the global automata hold, across namespaces.
// ponytail: 64k tokens (~10 MB of states); raise it when an agent bench shows global matches
// evicted before their reuse.
const GLOBAL_TOKENS: usize = 1 << 16;

const ROOT: u32 = 0;
const NONE: u32 = u32::MAX;

/// A suffix automaton state: the strings sharing one set of end positions.
#[derive(Clone)]
struct State {
    /// The longest string's length.
    len: u32,
    /// The state of the longest suffix in another set; NONE at the root.
    link: u32,
    /// Occurrences: the end positions' count.
    count: u32,
    /// Transitions by token, sorted.
    next: Vec<(u32, u32)>,
}

/// A suffix automaton over a token string, built one token at a time.
struct Sam {
    st: Vec<State>,
    last: u32,
    /// Tokens pushed.
    n: usize,
}

impl Default for Sam {
    fn default() -> Sam {
        let root = State {
            len: 0,
            link: NONE,
            count: 0,
            next: Vec::new(),
        };
        Sam {
            st: vec![root],
            last: ROOT,
            n: 0,
        }
    }
}

impl Sam {
    fn go(&self, v: u32, c: u32) -> Option<u32> {
        let next = &self.st[v as usize].next;
        let i = next.binary_search_by_key(&c, |e| e.0).ok()?;
        Some(next[i].1)
    }

    fn set(&mut self, v: u32, c: u32, to: u32) {
        let next = &mut self.st[v as usize].next;
        match next.binary_search_by_key(&c, |e| e.0) {
            Ok(i) => next[i].1 = to,
            Err(i) => next.insert(i, (c, to)),
        }
    }

    fn link(&self, v: u32) -> u32 {
        self.st[v as usize].link
    }

    fn len(&self, v: u32) -> u32 {
        self.st[v as usize].len
    }

    /// Appends token `c` (the textbook online construction), counting its end position on every
    /// state it ends.
    fn push(&mut self, c: u32) {
        let cur = self.st.len() as u32;
        self.st.push(State {
            len: self.len(self.last) + 1,
            link: ROOT,
            count: 0,
            next: Vec::new(),
        });
        let mut p = self.last;
        while p != NONE && self.go(p, c).is_none() {
            self.set(p, c, cur);
            p = self.link(p);
        }
        if p != NONE {
            let q = self.go(p, c).expect("p has c");
            if self.len(p) + 1 == self.len(q) {
                self.st[cur as usize].link = q;
            } else {
                let clone = self.st.len() as u32;
                let mut st = self.st[q as usize].clone();
                st.len = self.len(p) + 1;
                self.st.push(st);
                while p != NONE && self.go(p, c) == Some(q) {
                    self.set(p, c, clone);
                    p = self.link(p);
                }
                self.st[q as usize].link = clone;
                self.st[cur as usize].link = clone;
            }
        }
        self.last = cur;
        self.n += 1;
        let mut v = cur;
        for _ in 0..COUNT_LINKS {
            if v == ROOT {
                break;
            }
            self.st[v as usize].count += 1;
            v = self.link(v);
        }
    }

    /// Appends a finished output and its end.
    fn output(&mut self, output: &[u32]) {
        output.iter().for_each(|&t| self.push(t));
        self.push(SEP);
    }

    /// The state of the longest suffix of `tail` that some token followed, and that suffix's
    /// length.
    fn find(&self, tail: &[u32]) -> (u32, u32) {
        let (mut v, mut l) = (ROOT, 0);
        for &c in tail {
            loop {
                if let Some(w) = self.go(v, c) {
                    (v, l) = (w, l + 1);
                    break;
                }
                if v == ROOT {
                    break;
                }
                v = self.link(v);
                l = self.len(v);
            }
        }
        // The tail's own end has no continuation yet: a shorter suffix that occurred before.
        while v != ROOT && self.st[v as usize].next.is_empty() {
            v = self.link(v);
            l = self.len(v);
        }
        (v, l)
    }

    /// Appends up to `room` tokens following state `v`, the most frequent first.
    fn draft(&self, mut v: u32, room: usize, out: &mut Vec<u32>) {
        let mut p = 1.0f32;
        for _ in 0..room {
            let next = &self.st[v as usize].next;
            let count = |e: &&(u32, u32)| self.st[e.1 as usize].count;
            let total: u64 = next.iter().map(|e| u64::from(count(&e))).sum();
            // The end of an output loses ties: it must not hide a continuation another output had.
            let Some(&(c, w)) = next.iter().max_by_key(|e| (count(e), e.0 != SEP)) else {
                return;
            };
            p *= self.st[w as usize].count as f32 / total.max(1) as f32;
            if c == SEP || p < FLOOR {
                return;
            }
            out.push(c);
            v = w;
        }
    }
}

/// The default prompt-side [`Drafter`]: suffix drafts from the request's own tokens or, on a
/// longer match, from its namespace's recent outputs. Per-request automata are freed when their
/// request ends; the global ones hold token ids only, never text, and are kept per namespace, so
/// nothing crosses a cache namespace or an adapter version.
#[derive(Default)]
pub(crate) struct SuffixDrafter {
    reqs: HashMap<u64, Sam>,
    /// Finished outputs, oldest first, [`GLOBAL_TOKENS`] at most.
    recent: VecDeque<(Namespace, Vec<u32>)>,
    held: usize,
    /// Each namespace's automaton over its outputs in `recent`, built when first needed.
    global: HashMap<Namespace, Sam>,
}

impl Drafter for SuffixDrafter {
    fn propose(&mut self, id: u64, ns: Namespace, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        // ponytail: the automaton spans the whole context (~150 B a token, built at the first
        // guess); index only the last 32k tokens if long contexts with guessing show in RAM or
        // plan time.
        let own = self.reqs.entry(id).or_default();
        // Committed tokens only grow; anything else starts over.
        if tokens.len() < own.n {
            *own = Sam::default();
        }
        for &t in &tokens[own.n..] {
            own.push(t);
        }
        let tail = &tokens[tokens.len().saturating_sub(TAIL)..];
        let own: &Sam = own;
        let mine = (own, own.find(tail));
        let theirs = global(&mut self.global, &self.recent, ns).map(|g| (g, g.find(tail)));
        // The longer match first; the other if it drafts nothing (an output's end, say).
        let (first, second) = match theirs {
            Some(t) if t.1.1 > mine.1.1 => (t, Some(mine)),
            t => (mine, t),
        };
        for (sam, (v, l)) in [Some(first), second].into_iter().flatten() {
            if l >= MIN_MATCH && out.is_empty() {
                sam.draft(v, max.min(PER_MATCH * l as usize), out);
            }
        }
    }

    fn uses_history(&self) -> bool {
        true
    }

    fn ended(&mut self, id: u64, ns: Namespace, output: &[u32]) {
        self.reqs.remove(&id);
        if output.is_empty() {
            return;
        }
        if let Some(g) = self.global.get_mut(&ns) {
            g.output(output);
        }
        self.held += output.len();
        self.recent.push_back((ns, output.to_vec()));
        // Evicting a quarter at once rebuilds each automaton once per quarter, not per output.
        if self.held > GLOBAL_TOKENS {
            while self.held > GLOBAL_TOKENS / 4 * 3 {
                let (ns, old) = self.recent.pop_front().expect("held tokens");
                self.held -= old.len();
                self.global.remove(&ns);
            }
        }
    }
}

/// Namespace `ns`'s automaton over its outputs in `recent`, if it has any.
// ponytail: built in the planning step that first needs it (up to 48k tokens after an eviction),
// and a namespace without outputs scans `recent` each call; build off the step if a bench shows
// the spike.
fn global<'a>(
    global: &'a mut HashMap<Namespace, Sam>,
    recent: &VecDeque<(Namespace, Vec<u32>)>,
    ns: Namespace,
) -> Option<&'a Sam> {
    match global.entry(ns) {
        Entry::Occupied(g) => Some(g.into_mut()),
        Entry::Vacant(v) => {
            let mut outputs = recent.iter().filter(|r| r.0 == ns).peekable();
            outputs.peek()?;
            let g = v.insert(Sam::default());
            outputs.for_each(|o| g.output(&o.1));
            Some(g)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guess::PromptLookup;

    const NS: Namespace = (0, 0);

    fn propose(
        d: &mut dyn Drafter,
        id: u64,
        ns: Namespace,
        tokens: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        d.propose(id, ns, tokens, max, &mut out);
        out
    }

    #[test]
    fn copies_what_followed_an_earlier_match() {
        let mut d = SuffixDrafter::default();
        // "4 5" matches; a match of 2 drafts up to 4 tokens, to the end of the text.
        assert_eq!(
            propose(&mut d, 1, NS, &[4, 5, 6, 7, 8, 9, 4, 5], 8),
            [6, 7, 8, 9]
        );
        // The same request again, grown by one token: the automaton extends.
        assert_eq!(
            propose(&mut d, 1, NS, &[4, 5, 6, 7, 8, 9, 4, 5, 6], 8),
            [7, 8, 9, 4, 5, 6]
        );
        // Nothing earlier matches two tokens, or the text is too short.
        assert!(propose(&mut d, 2, NS, &[1, 2, 3, 4, 1], 8).is_empty());
        assert!(propose(&mut d, 3, NS, &[1, 1], 8).is_empty());
        // A run proposes its own continuation, bounded by the text and `max`.
        assert_eq!(propose(&mut d, 4, NS, &[7, 7, 7, 7], 8), [7]);
        assert_eq!(propose(&mut d, 5, NS, &[4, 5, 6, 7, 8, 9, 4, 5], 2), [6, 7]);
    }

    #[test]
    fn the_most_frequent_continuation_wins_and_rare_ones_draft_nothing() {
        // "1 2" went on as "3 9" twice, as "4 8" last: prompt lookup copies the latest.
        let t = [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 4, 8, 1, 2];
        assert_eq!(
            propose(&mut SuffixDrafter::default(), 1, NS, &t, 8),
            [3, 9, 1, 2]
        );
        assert_eq!(propose(&mut PromptLookup, 1, NS, &t, 8), [4, 8, 1, 2]);
        // Four continuations, a quarter each: below the floor.
        let t = [1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 2];
        assert!(propose(&mut SuffixDrafter::default(), 1, NS, &t, 8).is_empty());
    }

    #[test]
    fn finished_outputs_draft_for_their_namespace_only() {
        let mut d = SuffixDrafter::default();
        let output = [20, 21, 22, 23, 24, 25, 26];
        assert!(propose(&mut d, 1, NS, &[1, 2, 3], 8).is_empty());
        d.ended(1, NS, &output);
        assert!(d.reqs.is_empty());
        // A later request reaching "20 21" drafts the output's continuation; another namespace
        // (another adapter version) does not.
        assert_eq!(propose(&mut d, 2, NS, &[9, 20, 21], 8), [22, 23, 24, 25]);
        assert!(propose(&mut d, 3, (0, 1), &[9, 20, 21], 8).is_empty());
        // A draft stops at an output's end, and then the shorter own match drafts; a longer own
        // match beats a shorter global one.
        assert_eq!(propose(&mut d, 4, NS, &[24, 25, 26], 8), Vec::<u32>::new());
        assert_eq!(
            propose(&mut d, 6, NS, &[9, 25, 26, 3, 24, 25, 26], 8),
            [3, 24, 25, 26]
        );
        assert_eq!(
            propose(&mut d, 5, NS, &[7, 20, 21, 30, 7, 20, 21], 8),
            [30, 7, 20, 21]
        );
        for id in 2..7 {
            d.ended(id, NS, &[]);
        }
        assert!(d.reqs.is_empty());
    }

    #[test]
    fn an_output_end_does_not_win_a_tie() {
        let mut d = SuffixDrafter::default();
        d.ended(1, NS, &[1, 2]);
        d.ended(2, NS, &[1, 2, 3]);
        // "1 2" ended one output and went on in another: the continuation drafts.
        assert_eq!(propose(&mut d, 3, NS, &[9, 1, 2], 8), [3]);
    }

    #[test]
    fn the_global_automata_stay_bounded() {
        let mut d = SuffixDrafter::default();
        for i in 0..300u32 {
            let ns = (0, u64::from(i % 3));
            let output: Vec<u32> = (0..1000).map(|j| (i * 7 + j) % 5000).collect();
            d.ended(u64::from(i), ns, &output);
            assert!(propose(&mut d, 1000, ns, &output[..50], 8).len() <= 8);
            d.ended(1000, ns, &[]);
            assert!(d.held <= GLOBAL_TOKENS);
            assert_eq!(d.held, d.recent.iter().map(|r| r.1.len()).sum::<usize>());
            let states: usize = d.global.values().map(|s| s.st.len()).sum();
            assert!(
                states <= 2 * (d.held + d.recent.len()) + d.global.len(),
                "{states}"
            );
        }
        // The oldest outputs went; the newest still drafts.
        assert!(d.recent.len() < 300 && d.recent.len() >= GLOBAL_TOKENS / 1000 / 2);
        assert!(d.reqs.is_empty());
    }
}
