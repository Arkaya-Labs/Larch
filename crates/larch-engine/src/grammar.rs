//! Grammar-constrained sampling (IR-07a): a request's pattern compiles once to an anchored dense
//! DFA over the output's bytes, and each step samples only tokens whose bytes keep it alive. A
//! state's allowed tokens are computed on its first visit and cached with the compiled pattern,
//! which the facade shares across requests that send the same pattern.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use rand_chacha::ChaCha8Rng;
use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::nfa::thompson;
use regex_automata::util::{primitives::StateID, start};
use regex_automata::{Anchored, MatchKind};
use serde_json::{Map, Value};

use crate::sample::{Row, sample, top};
use crate::{MAX_SCORE_IDS, SampledToken, SamplingParams};

/// The DFA's (and its determinization's) memory limit; a pattern past it is refused.
const DFA_LIMIT: usize = 32 << 20;
/// Whitespace between JSON tokens: one space, or one newline and at most 16 spaces of indent, so
/// a model cannot loop on whitespace.
const WS: &str = r"(?: |\n {0,16})?";
const STRING: &str = r#""(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*""#;
// Bounded so serde_json reads every value as its type: an integer fits an i64 (and is not -0,
// which it reads as a float), a number an f64
// (a fraction of at most 17 digits, an exponent of two); unbounded digits would also let a model
// loop forever, as whitespace would.
const INTEGER: &str = r"(?:0|-?[1-9][0-9]{0,17})";
const NUMBER: &str = r"-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,17})?(?:[eE][+-]?[0-9]{1,2})?";
/// How deep an untyped value nests: a regex cannot recurse.
const ANY_DEPTH: u32 = 3;

/// A constraint on a request's output text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grammar {
    /// None constrains from the first generated token. Some(t) samples freely until the output
    /// text holds `t` (bytes after it in the same token are the pattern's), then constrains; a
    /// completed pattern re-arms (free until `t` shows again). Only generated text counts: a
    /// prompt that ends with `t` arms nothing, so send such a request untriggered.
    pub trigger: Option<String>,
    /// A regex over the output's bytes, matched whole from where the constraint starts. Stop
    /// tokens are allowed only where it matches; an untriggered pattern that can go no further
    /// allows only them.
    pub pattern: String,
}

/// Why a grammar or schema was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// An empty pattern or trigger.
    Empty,
    /// The pattern is not a regex: the parser's reason.
    Pattern(String),
    /// The DFA would pass its size limit; the request is refused, never run unconstrained.
    TooLarge,
    /// A schema keyword or shape with no regex here (recursive `$ref`, `allOf`, a remote `$ref`).
    Unsupported(String),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty grammar pattern or trigger"),
            Self::Pattern(why) => write!(f, "grammar pattern is not a regex: {why}"),
            Self::TooLarge => f.write_str("grammar pattern is too large"),
            Self::Unsupported(what) => write!(f, "unsupported schema: {what}"),
        }
    }
}

impl std::error::Error for GrammarError {}

/// The JSON a schema describes, as a [`Grammar::pattern`]: `type` (and type lists), `properties`
/// in schema order with `required` (optional ones skippable), `enum`, `const`, `items`,
/// `anyOf`/`oneOf` and local `$ref`. `additionalProperties` is always false: no other key is
/// written. Keywords that only narrow a value (`pattern`, `minLength`, `minimum`, ...) are
/// ignored and the type still holds; an untyped value is any JSON value nested at most
/// three deep. Whitespace between tokens and the digits of numbers are bounded (see `WS`,
/// `INTEGER`, `NUMBER`).
pub fn json_schema_regex(schema: &Value) -> Result<String, GrammarError> {
    value(schema, schema, &mut Vec::new())
}

fn unsupported(what: impl fmt::Display) -> GrammarError {
    GrammarError::Unsupported(what.to_string())
}

fn value(s: &Value, root: &Value, refs: &mut Vec<String>) -> Result<String, GrammarError> {
    let o = match s {
        Value::Bool(true) => return Ok(any(ANY_DEPTH)),
        Value::Object(o) => o,
        other => return Err(unsupported(format!("schema {other}"))),
    };
    if let Some(r) = o.get("$ref") {
        let r = r.as_str().ok_or_else(|| unsupported("non-string $ref"))?;
        if refs.iter().any(|x| x == r) {
            return Err(unsupported(format!("recursive $ref {r}")));
        }
        let target = (r.strip_prefix('#'))
            .and_then(|p| root.pointer(p))
            .ok_or_else(|| unsupported(format!("$ref {r}")))?;
        refs.push(r.into());
        let out = value(target, root, refs);
        refs.pop();
        return out;
    }
    if o.contains_key("allOf") {
        return Err(unsupported("allOf"));
    }
    if let Some(c) = o.get("const") {
        return Ok(literal(c));
    }
    if let Some(e) = o.get("enum") {
        let e = e.as_array().ok_or_else(|| unsupported("non-array enum"))?;
        return alt(e.iter().map(|v| Ok(literal(v))));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(v) = o.get(key) {
            let v = v.as_array().ok_or_else(|| unsupported(key))?;
            return alt(v.iter().map(|v| value(v, root, refs)));
        }
    }
    match o.get("type") {
        Some(Value::String(t)) => typed(t, o, root, refs),
        Some(Value::Array(ts)) => alt(ts.iter().map(|t| match t {
            Value::String(t) => typed(t, o, root, refs),
            other => Err(unsupported(format!("type {other}"))),
        })),
        None if o.contains_key("properties") => typed("object", o, root, refs),
        None if o.contains_key("items") => typed("array", o, root, refs),
        None => Ok(any(ANY_DEPTH)),
        Some(other) => Err(unsupported(format!("type {other}"))),
    }
}

fn typed(
    t: &str,
    o: &Map<String, Value>,
    root: &Value,
    refs: &mut Vec<String>,
) -> Result<String, GrammarError> {
    Ok(match t {
        "string" => STRING.into(),
        "number" => NUMBER.into(),
        "integer" => INTEGER.into(),
        "boolean" => "(?:true|false)".into(),
        "null" => "null".into(),
        "array" => array(&match o.get("items") {
            Some(items) => value(items, root, refs)?,
            None => any(ANY_DEPTH - 1),
        }),
        "object" => match o.get("properties") {
            Some(Value::Object(props)) => {
                let required: Vec<&str> = (o.get("required").and_then(Value::as_array))
                    .map_or(Vec::new(), |r| r.iter().filter_map(Value::as_str).collect());
                if let Some(k) = required.iter().find(|k| !props.contains_key(**k)) {
                    return Err(unsupported(format!("required {k} has no property")));
                }
                let items = (props.iter())
                    .map(|(k, v)| {
                        let item = format!(
                            "{}{WS}:{WS}{}",
                            literal(&Value::String(k.clone())),
                            value(v, root, refs)?
                        );
                        Ok((item, required.contains(&k.as_str())))
                    })
                    .collect::<Result<Vec<_>, GrammarError>>()?;
                format!(r"\{{{WS}{}{WS}\}}", members(&items, 0, false))
            }
            Some(other) => return Err(unsupported(format!("properties {other}"))),
            None => object(&any(ANY_DEPTH - 1)),
        },
        other => return Err(unsupported(format!("type {other}"))),
    })
}

/// Properties `items[i..]` in order, each `(regex, required)`; `after` says one was written
/// before, so the next takes a comma. Linear in the count once a property is written, quadratic
/// over a leading run of optional ones.
fn members(items: &[(String, bool)], i: usize, after: bool) -> String {
    let Some((item, required)) = items.get(i) else {
        return String::new();
    };
    let comma = if after {
        format!("{WS},{WS}")
    } else {
        String::new()
    };
    match (required, after) {
        (true, _) => format!("{comma}{item}{}", members(items, i + 1, true)),
        (false, true) => format!("(?:{comma}{item})?{}", members(items, i + 1, true)),
        (false, false) => format!(
            "(?:{item}{}|{})",
            members(items, i + 1, true),
            members(items, i + 1, false)
        ),
    }
}

fn array(item: &str) -> String {
    format!(r"\[{WS}(?:{item}(?:{WS},{WS}{item})*)?{WS}\]")
}

fn object(v: &str) -> String {
    let member = format!("{STRING}{WS}:{WS}{v}");
    format!(r"\{{{WS}(?:{member}(?:{WS},{WS}{member})*)?{WS}\}}")
}

/// Any JSON value nested at most `depth` deep.
fn any(depth: u32) -> String {
    let scalar = format!("{STRING}|{NUMBER}|true|false|null");
    if depth == 0 {
        return format!("(?:{scalar})");
    }
    let inner = any(depth - 1);
    format!("(?:{scalar}|{}|{})", array(&inner), object(&inner))
}

fn literal(v: &Value) -> String {
    regex_syntax::escape(&v.to_string())
}

fn alt(parts: impl Iterator<Item = Result<String, GrammarError>>) -> Result<String, GrammarError> {
    let parts = parts.collect::<Result<Vec<_>, _>>()?;
    if parts.is_empty() {
        return Err(unsupported("an empty enum or alternative matches nothing"));
    }
    Ok(format!("(?:{})", parts.join("|")))
}

/// Whether `text` is an output `g` allows, read byte by byte: free text until the trigger,
/// then the pattern, re-armed after each completion; the text may end free or where the pattern
/// matches.
pub fn matches(g: &Grammar, text: &str) -> Result<bool, GrammarError> {
    let c = Compiled::new(&g.pattern, Arc::new([]))?;
    let Some(trigger) = &g.trigger else {
        return Ok(c
            .walk(c.start, text.as_bytes())
            .is_some_and(|s| c.accepts(s)));
    };
    if trigger.is_empty() {
        return Err(GrammarError::Empty);
    }
    let (mut tail, mut on) = (Vec::new(), None);
    for &b in text.as_bytes() {
        on = match on {
            None => {
                tail.push(b);
                tail.ends_with(trigger.as_bytes()).then_some(c.start)
            }
            Some(s) => match c.walk(s, &[b]) {
                None => return Ok(false),
                Some(n) if c.complete(n) => {
                    tail.clear();
                    None
                }
                n => n,
            },
        };
    }
    Ok(on.is_none_or(|s| c.accepts(s)))
}

/// A pattern compiled against a vocabulary, shared by every request that sends it.
pub struct Compiled {
    dfa: dense::DFA<Vec<u32>>,
    start: StateID,
    /// By state index: a match is still reachable. The DFA reports a match one byte late, so a
    /// byte that ends a match leads to a match state from which nothing continues, not to the
    /// dead state; only this tells such a byte apart from one that keeps the pattern alive.
    live: Vec<bool>,
    /// Each token's bytes as the output text shows them; empty (never allowed) for a token that
    /// shows none.
    vocab: Arc<[Vec<u8>]>,
    /// The allowed tokens of each state visited so far.
    // ponytail: grows with the states requests visit, bounded by the DFA's; evict when a long-lived
    // server's grammars show in its RSS.
    states: Mutex<HashMap<StateID, Arc<Allowed>>>,
}

impl fmt::Debug for Compiled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let states = self.states.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("Compiled")
            .field("vocab", &self.vocab.len())
            .field("states", &states.len())
            .finish_non_exhaustive()
    }
}

/// One state's allowed tokens.
struct Allowed {
    /// Bit `id` set: the token's bytes keep the DFA alive. Stop tokens are judged by `accept`.
    bits: Vec<u64>,
    /// The set ids in order, when at most MAX_SCORE_IDS.
    list: Option<Vec<u32>>,
    /// The output so far matches the pattern.
    accept: bool,
    /// It matches and no byte continues it.
    complete: bool,
}

impl Allowed {
    fn allows(&self, id: u32) -> bool {
        (self.bits.get(id as usize / 64)).is_some_and(|w| w >> (id % 64) & 1 == 1)
    }
}

impl Compiled {
    /// Compiles `pattern` for `vocab` (each token's visible bytes, by id); refuses an empty
    /// pattern, a non-regex and a DFA over the size limit.
    pub fn new(pattern: &str, vocab: Arc<[Vec<u8>]>) -> Result<Compiled, GrammarError> {
        if pattern.is_empty() {
            return Err(GrammarError::Empty);
        }
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Anchored)
                    // Every string the pattern matches, not just leftmost-first's preference.
                    .match_kind(MatchKind::All)
                    .dfa_size_limit(Some(DFA_LIMIT))
                    .determinize_size_limit(Some(DFA_LIMIT)),
            )
            .thompson(thompson::Config::new().nfa_size_limit(Some(DFA_LIMIT)))
            .build(pattern)
            .map_err(|e| {
                let nfa = std::error::Error::source(&e)
                    .and_then(|s| s.downcast_ref::<thompson::BuildError>());
                if e.is_size_limit_exceeded() || nfa.is_some_and(|n| n.size_limit().is_some()) {
                    GrammarError::TooLarge
                } else {
                    GrammarError::Pattern(nfa.map_or(e.to_string(), |n| {
                        match std::error::Error::source(n) {
                            Some(why) => why.to_string(),
                            None => n.to_string(),
                        }
                    }))
                }
            })?;
        let start = dfa
            .start_state(&start::Config::new().anchored(Anchored::Yes))
            .expect("an anchored start state without look-behind");
        // Every state reachable from the start, with its predecessors; then back from the
        // matching ones.
        let index = |s: StateID| s.as_usize() >> dfa.stride2();
        let mut preds: HashMap<StateID, Vec<StateID>> = HashMap::from([(start, Vec::new())]);
        let mut queue = VecDeque::from([start]);
        while let Some(x) = queue.pop_front() {
            for b in 0..=255 {
                let y = dfa.next_state(x, b);
                if dfa.is_dead_state(y) {
                    continue;
                }
                let seen = preds.contains_key(&y);
                preds.entry(y).or_default().push(x);
                if !seen {
                    queue.push_back(y);
                }
            }
        }
        let mut live = vec![false; preds.keys().map(|&s| index(s)).max().unwrap_or(0) + 1];
        let accepts = |s: StateID| dfa.is_match_state(dfa.next_eoi_state(s));
        let mut queue: VecDeque<StateID> = preds.keys().copied().filter(|&s| accepts(s)).collect();
        for &s in &queue {
            live[index(s)] = true;
        }
        while let Some(y) = queue.pop_front() {
            for &x in &preds[&y] {
                if !live[index(x)] {
                    live[index(x)] = true;
                    queue.push_back(x);
                }
            }
        }
        let c = Compiled {
            dfa,
            start,
            live,
            vocab,
            states: Mutex::new(HashMap::new()),
        };
        // On the caller's thread, so admission never walks the vocabulary under the engine's locks.
        c.allowed(start);
        Ok(c)
    }

    /// The vocabulary's size: ids past the model's would be gathered past its logits.
    pub(crate) fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// The state after `bytes` from `s`; None once dead.
    fn walk(&self, mut s: StateID, bytes: &[u8]) -> Option<StateID> {
        for &b in bytes {
            s = self.dfa.next_state(s, b);
            if !self
                .live
                .get(s.as_usize() >> self.dfa.stride2())
                .is_some_and(|&l| l)
            {
                return None;
            }
        }
        Some(s)
    }

    fn accepts(&self, s: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(s))
    }

    fn complete(&self, s: StateID) -> bool {
        self.accepts(s) && (0..=255).all(|b| self.walk(s, &[b]).is_none())
    }

    fn allowed(&self, s: StateID) -> Arc<Allowed> {
        let mut states = self.states.lock().unwrap_or_else(PoisonError::into_inner);
        let a = states.entry(s).or_insert_with(|| {
            // ponytail: every token walked on a state's first visit (vocab x piece bytes); a token
            // trie shares their prefixes, when first visits show in the step profile.
            let mut bits = vec![0u64; self.vocab.len().div_ceil(64)];
            let mut list = Some(Vec::new());
            for (id, piece) in self.vocab.iter().enumerate() {
                if !piece.is_empty() && self.walk(s, piece).is_some() {
                    bits[id / 64] |= 1 << (id % 64);
                    if let Some(l) = &mut list {
                        l.push(id as u32);
                        if l.len() > MAX_SCORE_IDS {
                            list = None;
                        }
                    }
                }
            }
            Arc::new(Allowed {
                bits,
                list,
                accept: self.accepts(s),
                complete: self.complete(s),
            })
        });
        Arc::clone(a)
    }

    /// The bytes of a shortest path from `s` to a match (empty when `s` matches).
    fn shortest(&self, s: StateID) -> Vec<u8> {
        let mut prev: HashMap<StateID, (StateID, u8)> = HashMap::new();
        let mut seen = HashSet::from([s]);
        let mut queue = VecDeque::from([s]);
        let mut end = None;
        while let Some(x) = queue.pop_front() {
            if self.accepts(x) {
                end = Some(x);
                break;
            }
            for b in 0..=255 {
                if let Some(y) = self.walk(x, &[b])
                    && seen.insert(y)
                {
                    prev.insert(y, (x, b));
                    queue.push_back(y);
                }
            }
        }
        let mut path = Vec::new();
        let mut at = end;
        while let Some((x, b)) = at.and_then(|y| prev.get(&y)) {
            path.push(*b);
            at = Some(*x);
        }
        path.reverse();
        path
    }
}

/// A request's grammar, on [`Request::grammar`](crate::Request::grammar).
#[derive(Clone, Debug)]
pub struct Constraint {
    /// As in [`Grammar`]; an empty trigger is refused (`invalid::GRAMMAR_PATTERN`).
    pub trigger: Option<String>,
    /// The compiled pattern, shared by the requests that send it.
    pub compiled: Arc<Compiled>,
    /// Output already generated under this grammar (a thinking-budget continuation's first
    /// part), fed through it before the first sample so the pattern resumes where it was.
    pub replay: Vec<u32>,
}

/// A request's grammar state, next to its rng.
pub(crate) struct Live {
    c: Constraint,
    mode: Mode,
    /// The ids the next step gathers exact logits for: a narrow state's allowed tokens, with the
    /// stops where it matches; empty otherwise.
    ids: Vec<u32>,
    /// Picks made with no allowed token among the 64 candidates.
    pub(crate) forced: u32,
}

enum Mode {
    /// The last trigger-length bytes of the free text.
    Free(Vec<u8>),
    On(StateID, Arc<Allowed>),
}

impl Live {
    pub(crate) fn new(mut c: Constraint, stops: &[u32]) -> Live {
        let start = c.compiled.start;
        let replay = std::mem::take(&mut c.replay);
        let mut live = Live {
            mode: Mode::Free(Vec::new()),
            ids: Vec::new(),
            forced: 0,
            c,
        };
        if live.c.trigger.is_none() {
            live.enter(start, stops);
        }
        for id in replay {
            live.advance(id, stops);
        }
        live
    }

    fn enter(&mut self, s: StateID, stops: &[u32]) {
        let a = self.c.compiled.allowed(s);
        self.ids.clear();
        if a.complete && self.c.trigger.is_some() {
            self.mode = Mode::Free(Vec::new());
            return;
        }
        if let Some(list) = &a.list {
            self.ids
                .extend(list.iter().filter(|id| !stops.contains(id)));
            if a.accept {
                self.ids.extend(stops);
            }
            if self.ids.len() > MAX_SCORE_IDS {
                self.ids.clear();
            }
        }
        self.mode = Mode::On(s, a);
    }

    /// Sampling is constrained this step (else the free path runs, unchanged).
    pub(crate) fn constrained(&self) -> bool {
        matches!(self.mode, Mode::On(..))
    }

    /// The ids to gather this step; empty means filter the candidates.
    pub(crate) fn gather(&self) -> &[u32] {
        &self.ids
    }

    /// Draws a constrained token: over the gathered logits of every allowed token when the state
    /// is narrow, else over the allowed candidates; temperature, top-p and top-k apply to the
    /// allowed tokens only, and logprobs stay the model's own. With none allowed among the
    /// candidates, the first token on a shortest path to a match is forced, and counted; its
    /// logprob is then the last candidate's, an upper bound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pick(
        &mut self,
        row: &Row,
        gathered: &[f32],
        p: &SamplingParams,
        logprobs: u8,
        recent: &[u32],
        stops: &[u32],
        rng: &mut ChaCha8Rng,
    ) -> SampledToken {
        let Mode::On(s, a) = &self.mode else {
            unreachable!("pick in free mode")
        };
        let mut pool: Vec<(f32, u32)> = if self.ids.is_empty() {
            let ok = |id: u32| {
                if stops.contains(&id) {
                    a.accept
                } else {
                    a.allows(id)
                }
            };
            (row.ids.iter().zip(row.vals))
                .take_while(|(id, _)| **id >= 0)
                .filter(|(id, _)| ok(**id as u32))
                .map(|(&id, &v)| (v, id as u32))
                .collect()
        } else {
            // The gathered logits are raw: the kernel's penalty, once per id in the window.
            let rp = p.repeat_penalty;
            let penalise = |v: f32, id: u32| match rp != 1.0 && recent.contains(&id) {
                true if v <= 0.0 => v * rp,
                true => v / rp,
                false => v,
            };
            let mut pool: Vec<_> = (self.ids.iter().zip(gathered))
                .map(|(&id, &v)| (penalise(v, id), id))
                .collect();
            // sample_prep's order: largest first, ties to the lower id.
            pool.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            pool
        };
        let top = top(row, usize::from(logprobs));
        if pool.is_empty() {
            self.forced += 1;
            let last = row.ids.iter().take_while(|&&id| id >= 0).count().max(1) - 1;
            return SampledToken {
                id: self.forced_pick(*s, a, stops, row),
                logprob: row.vals[last] - row.lse,
                top,
            };
        }
        let (vals, ids): (Vec<f32>, Vec<i32>) =
            pool.drain(..).map(|(v, id)| (v, id as i32)).unzip();
        let pool = Row {
            vals: &vals,
            ids: &ids,
            lse: row.lse,
        };
        SampledToken {
            top,
            ..sample(&pool, p, 0, &[], rng)
        }
    }

    fn forced_pick(&self, s: StateID, a: &Allowed, stops: &[u32], row: &Row) -> u32 {
        if a.accept
            && let Some(&stop) = stops.first()
        {
            return stop;
        }
        let compiled = &self.c.compiled;
        let path = compiled.shortest(s);
        let allowed =
            (0..compiled.vocab.len() as u32).filter(|&id| a.allows(id) && !stops.contains(&id));
        (allowed.clone())
            .filter(|&id| path.starts_with(&compiled.vocab[id as usize]))
            .max_by_key(|&id| (compiled.vocab[id as usize].len(), std::cmp::Reverse(id)))
            .or_else(|| allowed.clone().next())
            // ponytail: a state no token can leave ends the request on a stop, else takes the
            // model's pick; a typed error instead once benches show such dead ends in
            // grammar_forced.
            .or_else(|| stops.first().copied())
            .unwrap_or(row.ids[0] as u32)
    }

    /// Feeds the sampled token: free text watches for the trigger; a constrained state walks
    /// the token's bytes, then re-arms (triggered) or allows only the stops (untriggered) once
    /// the pattern is complete.
    pub(crate) fn advance(&mut self, id: u32, stops: &[u32]) {
        let compiled = Arc::clone(&self.c.compiled);
        let piece = compiled
            .vocab
            .get(id as usize)
            .map_or(&[][..], Vec::as_slice);
        match &mut self.mode {
            Mode::Free(tail) => {
                let trigger = self.c.trigger.as_deref().unwrap_or_default().as_bytes();
                tail.extend_from_slice(piece);
                // A token may carry bytes past the trigger (`<tool_call>\n` as one piece): they
                // are the pattern's first bytes.
                match tail.windows(trigger.len()).position(|w| w == trigger) {
                    Some(at) => {
                        let rest = tail.split_off(at + trigger.len());
                        let first = compiled.walk(compiled.start, &rest);
                        self.enter(first.unwrap_or(compiled.start), stops);
                    }
                    None => drop(tail.drain(..tail.len().saturating_sub(trigger.len()))),
                }
            }
            Mode::On(s, _) => {
                if stops.contains(&id) {
                    return;
                }
                // A forced pick past a dead end leaves the state as it was.
                if let Some(next) = compiled.walk(*s, piece) {
                    self.enter(next, stops);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_SCORE_IDS;
    use crate::sample::CANDIDATES;
    use rand_chacha::rand_core::SeedableRng;
    use serde_json::json;

    fn untriggered(pattern: &str) -> Grammar {
        Grammar {
            trigger: None,
            pattern: pattern.into(),
        }
    }

    fn schema_ok(schema: Value, texts: &[&str], refused: &[&str]) {
        let g = untriggered(&json_schema_regex(&schema).unwrap());
        for t in texts {
            assert!(matches(&g, t).unwrap(), "{schema} should match {t:?}");
        }
        for t in refused {
            assert!(!matches(&g, t).unwrap(), "{schema} should refuse {t:?}");
        }
    }

    #[test]
    fn scalar_types_and_type_lists() {
        schema_ok(
            json!({"type": "string"}),
            &[r#""""#, r#""a b""#, r#""\"\\\né""#, "\"é\""],
            &["a", r#""a"#, "\"\n\"", r#""\x""#],
        );
        schema_ok(
            json!({"type": "integer"}),
            &["0", "-12", "7"],
            &["01", "1.5", "+1", "", "1234567890123456789", "-0"],
        );
        schema_ok(
            json!({"type": "number"}),
            &["0", "-1.25", "3e10", "2.5E-3"],
            &["1.", ".5", "01", "2e810", "1.123456789012345678"],
        );
        schema_ok(json!({"type": "boolean"}), &["true", "false"], &["1"]);
        schema_ok(json!({"type": "null"}), &["null"], &["nil"]);
        schema_ok(
            json!({"type": ["integer", "null"]}),
            &["3", "null"],
            &["\"3\""],
        );
    }

    /// Properties in schema order, optional ones skippable, nothing else (additionalProperties
    /// true or absent is treated as false).
    #[test]
    fn properties_required_and_optional() {
        let props = json!({
            "a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": "boolean"}
        });
        schema_ok(
            json!({"type": "object", "properties": props, "required": ["a", "c"],
                   "additionalProperties": true}),
            &[r#"{"a": 1, "c": true}"#, r#"{"a":1,"b":"x","c":false}"#],
            &[
                r#"{"c": true, "a": 1}"#,
                r#"{"b": "x", "c": true}"#,
                r#"{"a": 1, "c": true, "d": 1}"#,
                r#"{"a": 1, "c": true,}"#,
            ],
        );
        schema_ok(
            json!({"properties": props}),
            &[
                "{}",
                r#"{"b": "x"}"#,
                r#"{"a": 1, "c": true}"#,
                r#"{"c": true}"#,
            ],
            &[r#"{, "c": true}"#, r#"{"b": 1}"#],
        );
    }

    #[test]
    fn enum_const_items_and_alternatives() {
        schema_ok(
            json!({"enum": ["get", "put", 3, null]}),
            &[r#""get""#, "3", "null"],
            &[r#""post""#],
        );
        schema_ok(json!({"const": {"k": [1]}}), &[r#"{"k":[1]}"#], &["{}"]);
        schema_ok(
            json!({"type": "array", "items": {"type": "integer"}}),
            &["[]", "[1, 2,3]"],
            &["[1,]", r#"["a"]"#],
        );
        for key in ["anyOf", "oneOf"] {
            schema_ok(
                json!({key: [{"type": "integer"}, {"type": "string"}]}),
                &["1", r#""x""#],
                &["true"],
            );
        }
    }

    #[test]
    fn local_refs_and_recursion() {
        schema_ok(
            json!({"$defs": {"n": {"type": "integer"}},
                   "type": "object", "properties": {"x": {"$ref": "#/$defs/n"}}, "required": ["x"]}),
            &[r#"{"x": 4}"#],
            &[r#"{"x": "4"}"#],
        );
        let recursive = json!({"$defs": {"t": {"type": "object",
            "properties": {"kids": {"type": "array", "items": {"$ref": "#/$defs/t"}}}}},
            "$ref": "#/$defs/t"});
        assert!(matches!(
            json_schema_regex(&recursive),
            Err(GrammarError::Unsupported(_))
        ));
        assert!(matches!(
            json_schema_regex(&json!({"$ref": "other.json#/x"})),
            Err(GrammarError::Unsupported(_))
        ));
    }

    /// Keywords that only narrow a value are ignored; the type still holds. A missing type is any
    /// JSON value (to a bounded depth).
    #[test]
    fn narrowing_keywords_are_ignored() {
        schema_ok(
            json!({"type": "string", "minLength": 5, "pattern": "^a", "format": "email"}),
            &[r#""b""#],
            &["1"],
        );
        schema_ok(json!({"type": "integer", "minimum": 10}), &["3"], &["3.5"]);
        schema_ok(
            json!({"description": "anything"}),
            &["1", r#"{"a": [true, null]}"#, r#""s""#],
            &["nope"],
        );
    }

    #[test]
    fn whitespace_is_bounded() {
        let pretty = "{\n  \"a\": 1,\n  \"b\": [\n    2\n  ]\n}";
        schema_ok(
            json!({"properties": {"a": {"type": "integer"},
                                  "b": {"type": "array", "items": {"type": "integer"}}}}),
            &[pretty, r#"{ "a" : 1 }"#],
            &[
                r#"{"a":  1}"#,
                "{\n\n\"a\": 1}",
                &format!("{{\n{}\"a\": 1}}", " ".repeat(17)),
                "{\t\"a\": 1}",
            ],
        );
    }

    #[test]
    fn refusals() {
        assert_eq!(
            matches(&untriggered(""), "").err(),
            Some(GrammarError::Empty)
        );
        let empty_trigger = Grammar {
            trigger: Some(String::new()),
            pattern: "a".into(),
        };
        assert_eq!(
            matches(&empty_trigger, "a").err(),
            Some(GrammarError::Empty)
        );
        assert!(matches!(
            matches(&untriggered("("), ""),
            Err(GrammarError::Pattern(_))
        ));
        // 2^21 states: over any sane size limit, refused rather than run unconstrained.
        assert_eq!(
            matches(&untriggered("[01]*1[01]{20}"), "1").err(),
            Some(GrammarError::TooLarge)
        );
        assert!(matches!(
            json_schema_regex(&json!({"allOf": [{"type": "integer"}]})),
            Err(GrammarError::Unsupported(_))
        ));
    }

    #[test]
    fn triggered_text_matches_every_call() {
        let g = Grammar {
            trigger: Some("<tool_call>".into()),
            pattern: r#"\{"a": [0-9]\}</tool_call>"#.into(),
        };
        let ok = |t: &str| matches(&g, t).unwrap();
        assert!(ok("free text"));
        assert!(ok(
            r#"hi <tool_call>{"a": 1}</tool_call> then <tool_call>{"a": 2}</tool_call>"#
        ));
        assert!(!ok(r#"<tool_call>{"a": 1}"#), "ends inside a call");
        assert!(!ok(r#"<tool_call>{"a": x}</tool_call>"#));
    }

    // A synthetic vocabulary: every printable ASCII byte, some multi-byte pieces, a hidden token
    // (no text), a stop whose text a JSON string could hold, and 400 words so a string state
    // allows more than MAX_SCORE_IDS tokens.
    const STOP: u32 = 0;
    const HIDDEN: u32 = 1;
    fn vocab() -> Vec<Vec<u8>> {
        let mut v = vec![b"<|im_end|>".to_vec(), Vec::new()];
        v.extend((0x20u8..0x7f).map(|b| vec![b]));
        for piece in ["{\"", "\":", "\": ", "true", "<tool_call>", "\n"] {
            v.push(piece.as_bytes().to_vec());
        }
        v.extend((0..400).map(|i| format!("w{i}").into_bytes()));
        v
    }
    fn id(v: &[Vec<u8>], piece: &str) -> u32 {
        v.iter().position(|p| p == piece.as_bytes()).unwrap() as u32
    }
    fn live(trigger: Option<&str>, pattern: &str) -> (Live, Vec<Vec<u8>>) {
        let v = vocab();
        let c = Constraint {
            trigger: trigger.map(Into::into),
            compiled: Arc::new(Compiled::new(pattern, v.clone().into()).unwrap()),
            replay: Vec::new(),
        };
        (Live::new(c, &[STOP]), v)
    }
    /// A row whose 64 candidates are `ids`, the first most likely.
    fn row(ids: &[u32]) -> (Vec<f32>, Vec<i32>) {
        let mut vals: Vec<f32> = (0..ids.len()).map(|i| -(i as f32)).collect();
        let mut out: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        vals.resize(CANDIDATES, f32::NEG_INFINITY);
        out.resize(CANDIDATES, -1);
        (vals, out)
    }
    fn params(temperature: f32) -> SamplingParams {
        SamplingParams {
            temperature,
            ..SamplingParams::GREEDY
        }
    }

    /// A narrow state packs every allowed token (and the stops in a match state) for the gather.
    #[test]
    fn narrow_states_take_the_gather_path() {
        let (mut g, v) = live(None, r#"\{"a": (?:1|true)\}"#);
        assert!(g.constrained());
        assert_eq!(g.gather(), [id(&v, "{"), id(&v, "{\"")]);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (vals, ids) = row(&[id(&v, "x")]);
        let r = Row {
            vals: &vals,
            ids: &ids,
            lse: 2.0,
        };
        // The draw is over the gathered logits: the second is larger.
        let t = g.pick(&r, &[0.0, 1.0], &params(0.0), 1, &[], &[STOP], &mut rng);
        assert_eq!(t.id, id(&v, "{\""));
        assert_eq!(t.logprob, 1.0 - 2.0);
        assert_eq!(t.top, [(id(&v, "x"), -2.0)], "the model's own top list");
        // A repetition penalty applies to gathered (raw) logits as the kernel applies it.
        let penalised = SamplingParams {
            repeat_penalty: 4.0,
            ..params(0.0)
        };
        let t = g.pick(
            &r,
            &[0.5, 1.0],
            &penalised,
            0,
            &[id(&v, "{\"")],
            &[STOP],
            &mut rng,
        );
        assert_eq!(t.id, id(&v, "{"));
        g.advance(id(&v, "{\""), &[STOP]);
        assert_eq!(g.gather(), [id(&v, "a")]);
        for piece in ["a", "\": ", "true", "}"] {
            g.advance(id(&v, piece), &[STOP]);
        }
        // Complete: only the stops are left.
        assert_eq!(g.gather(), [STOP]);
        assert!(g.constrained());
    }

    /// A string state allows more than MAX_SCORE_IDS tokens: the 64 candidates are filtered.
    #[test]
    fn wide_states_filter_the_candidates() {
        let (mut g, v) = live(None, r#""[a-z0-9]*""#);
        g.advance(id(&v, "\""), &[STOP]);
        assert!(g.gather().is_empty());
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (vals, ids) = row(&[id(&v, "{"), id(&v, "w7"), id(&v, "\""), STOP]);
        let r = Row {
            vals: &vals,
            ids: &ids,
            lse: 0.0,
        };
        let t = g.pick(&r, &[], &params(0.0), 0, &[], &[STOP], &mut rng);
        assert_eq!((t.id, t.logprob), (id(&v, "w7"), -1.0));
        // Sampling draws only allowed candidates; the stop's text fits a string, yet it is not
        // drawn before the string closes.
        for _ in 0..200 {
            let t = g.pick(&r, &[], &params(5.0), 0, &[], &[STOP], &mut rng);
            assert!(t.id == id(&v, "w7") || t.id == id(&v, "\""), "{}", t.id);
        }
        assert_eq!(g.forced, 0);
    }

    #[test]
    fn stop_tokens_only_in_match_states() {
        let (mut g, v) = live(None, r#""[^"]*"(?:,[0-9])?"#);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (vals, ids) = row(&[STOP, id(&v, "w1"), id(&v, "\"")]);
        let r = Row {
            vals: &vals,
            ids: &ids,
            lse: 0.0,
        };
        g.advance(id(&v, "\""), &[STOP]);
        assert_ne!(
            g.pick(&r, &[], &params(0.0), 0, &[], &[STOP], &mut rng).id,
            STOP
        );
        g.advance(id(&v, "\""), &[STOP]);
        // A match state that can still continue: the stop is allowed, and the most likely.
        assert!(g.gather().contains(&STOP));
        let gathered: Vec<f32> = g.gather().iter().map(|&t| f32::from(t == STOP)).collect();
        assert_eq!(
            g.pick(&r, &gathered, &params(0.0), 0, &[], &[STOP], &mut rng)
                .id,
            STOP
        );
        assert!(!g.gather().contains(&HIDDEN));
    }

    /// No allowed token among the 64 candidates, step after step: each pick is forced along a
    /// shortest path to a match, counted, and the text still matches.
    #[test]
    fn forced_picks_complete_the_pattern() {
        let pattern = r#"\{"k": "[a-z0-9]{2,}"\}"#;
        let (mut g, v) = live(None, pattern);
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let never: Vec<u32> = ["[", "]", "!", "?"].map(|p| id(&v, p)).to_vec();
        let (vals, ids) = row(&never);
        let r = Row {
            vals: &vals,
            ids: &ids,
            lse: 0.0,
        };
        let mut text = Vec::new();
        for _ in 0..64 {
            // The gather path draws from whatever the device returned for the packed ids.
            let gathered = vec![0.0; g.gather().len()];
            let t = g.pick(&r, &gathered, &params(1.0), 0, &[], &[STOP], &mut rng);
            if t.id == STOP {
                break;
            }
            text.extend_from_slice(&v[t.id as usize]);
            g.advance(t.id, &[STOP]);
        }
        let text = String::from_utf8(text).unwrap();
        assert!(matches(&untriggered(pattern), &text).unwrap(), "{text:?}");
        assert!(g.forced > 0);
    }

    #[test]
    fn complete_then_rearm() {
        let (mut g, v) = live(Some("<tool_call>"), r#"\{"a": 1\}"#);
        assert!(!g.constrained());
        assert!(g.gather().is_empty(), "free: nothing packed");
        for piece in ["x", "<", "tool", "_call>"] {
            if let Some(t) = v.iter().position(|p| p == piece.as_bytes()) {
                g.advance(t as u32, &[STOP]);
            } else {
                for b in piece.bytes() {
                    g.advance(id(&v, &(b as char).to_string()), &[STOP]);
                }
            }
        }
        assert!(g.constrained(), "the trigger arrived across tokens");
        for piece in ["{\"", "a", "\": ", "1", "}"] {
            g.advance(id(&v, piece), &[STOP]);
        }
        assert!(!g.constrained(), "complete: free again");
        g.advance(id(&v, "<tool_call>"), &[STOP]);
        assert!(g.constrained(), "re-armed by the next trigger");
        assert_eq!(g.gather(), [id(&v, "{"), id(&v, "{\"")]);
    }

    /// Bytes after the trigger inside its token feed the pattern; a replay resumes it.
    #[test]
    fn trigger_inside_a_token_and_replay() {
        let (mut g, v) = live(Some("<tool_call>"), r#"\n\{"a": 1\}"#);
        g.advance(id(&v, "x"), &[STOP]);
        let mut merged = v.clone();
        merged.push(b"<tool_call>\n".to_vec());
        let c = Constraint {
            trigger: Some("<tool_call>".into()),
            compiled: Arc::new(Compiled::new(r#"\n\{"a": 1\}"#, merged.clone().into()).unwrap()),
            replay: vec![
                id(&merged, "x"),
                merged.len() as u32 - 1,
                id(&merged, "{\""),
            ],
        };
        let resumed = Live::new(c, &[STOP]);
        assert!(resumed.constrained());
        assert_eq!(
            resumed.gather(),
            [id(&merged, "a")],
            "past the newline and the brace"
        );
        assert!(!g.constrained());
    }

    #[test]
    fn required_names_need_a_property() {
        assert!(matches!(
            json_schema_regex(&json!({"properties": {"a": {}}, "required": ["a", "b"]})),
            Err(GrammarError::Unsupported(_))
        ));
    }

    #[test]
    fn narrow_means_at_most_max_score_ids() {
        // 95 printable bytes + 400 words + multi-byte pieces: wide.
        let (g, _) = live(None, r"[ -~]+|w[0-9]+");
        assert!(g.gather().is_empty());
        let (g, _) = live(None, r"[a-z]");
        assert!(!g.gather().is_empty() && g.gather().len() <= MAX_SCORE_IDS);
    }

    /// The host cost a grammar adds, on a 151,936-token vocabulary: each state's first visit
    /// (walking every token) and a constrained token on cached states (pick and advance). No
    /// timing asserts; run with --release --ignored --nocapture for the numbers.
    #[test]
    #[ignore = "cost report: cargo test --release -p larch-engine --lib grammar_cost -- --ignored --nocapture"]
    fn grammar_cost() {
        use rand_chacha::rand_core::RngCore;
        use std::time::Instant;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut v: Vec<Vec<u8>> = vec![b"<|im_end|>".to_vec(), b"<tool_call>".to_vec()];
        v.extend((0..=255u8).map(|b| vec![b]));
        let alphabet = b" abcdefghijklmnopqrstuvwxyz\":{},0123456789";
        while v.len() < 151_936 {
            let n = 2 + rng.next_u32() as usize % 7;
            let piece = (0..n)
                .map(|_| alphabet[rng.next_u32() as usize % alphabet.len()])
                .collect();
            v.push(piece);
        }
        let schema = json!({"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"},
            "units": {"enum": ["metric", "imperial"]}}, "required": ["city", "days"]});
        let args = json_schema_regex(&schema).unwrap();
        let pattern = format!(
            "\n\\{{\"name\": \"(?:get_weather|search)\", \"arguments\": {args}\\}}\n</tool_call>"
        );
        let t = Instant::now();
        let compiled = Arc::new(Compiled::new(&pattern, v.clone().into()).unwrap());
        let compile = t.elapsed();
        let text = "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"oslo and bergen\", \
                    \"days\": 12, \"units\": \"metric\"}}\n</tool_call>";
        // Longest-match tokenization of the scripted call.
        let mut script = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let rest = &text.as_bytes()[at..];
            let id = (0..v.len())
                .filter(|&i| !v[i].is_empty() && rest.starts_with(&v[i]))
                .max_by_key(|&i| v[i].len())
                .unwrap();
            script.push(id as u32);
            at += v[id].len();
        }
        let run = |label: &str| {
            let c = Constraint {
                trigger: Some("<tool_call>".into()),
                compiled: compiled.clone(),
                replay: Vec::new(),
            };
            let mut g = Live::new(c, &[STOP]);
            g.advance(1, &[STOP]);
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            let (mut gather, mut filter) = (0, 0);
            let t = Instant::now();
            for &next in &script {
                let mut cands = vec![next];
                cands.extend((0..63).map(|i| 300 + i * 2000));
                let (vals, ids) = row(&cands);
                let r = Row {
                    vals: &vals,
                    ids: &ids,
                    lse: 1.0,
                };
                let gathered: Vec<f32> = (g.gather().iter())
                    .map(|&id| if id == next { 5.0 } else { 0.0 })
                    .collect();
                match g.gather().is_empty() {
                    true => filter += 1,
                    false => gather += 1,
                }
                let picked = g.pick(&r, &gathered, &params(0.0), 0, &[], &[STOP], &mut rng);
                assert_eq!(picked.id, next, "the scripted call stays allowed");
                g.advance(next, &[STOP]);
            }
            assert!(!g.constrained(), "complete: re-armed");
            let per = t.elapsed() / script.len() as u32;
            eprintln!(
                "grammar_cost {label}: {} tokens ({gather} gather, {filter} filter), {per:?} per token, forced {}",
                script.len(),
                g.forced
            );
        };
        run("first visit");
        let states = compiled.states.lock().unwrap().len();
        run("cached");
        eprintln!(
            "grammar_cost: vocab {}, compile {compile:?}, {states} states visited",
            v.len()
        );
    }
}
