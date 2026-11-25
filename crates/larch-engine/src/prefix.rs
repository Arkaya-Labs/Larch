//! Exact immutable block chains. Each entry holds one physical reference; sequence readers hold
//! their own. The key includes the canonical parent, so equal suffix blocks at different positions
//! or behind different histories cannot match. HashMap equality checks all sixteen token IDs.
//! On a model with recurrent layers an entry may also hold a checkpoint: the state after its
//! chain, in host memory, which goes with the entry.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::sched::{BLOCK_TOKENS, Blocks, Namespace};

/// Hard CPU index ceiling, in addition to the pool-relative physical-block cap. Two hash tables
/// with fixed-size keys/values, <= 8192 entries each; no full prompts or unbounded owner lists.
const MAX_ENTRIES: usize = 8192;

/// Host memory the checkpoints may hold, least recently used evicted first.
// ponytail: fixed 2 GiB (about 32 checkpoints of the hybrid MoE target), the figure a reload's
// carry holds; size it from free host memory if a bench shows checkpoints evicted under it.
const CHECKPOINT_BYTES: usize = 2 << 30;

/// Blocks of one turn's room ([`crate::TURN_ROOM`] tokens) pinned chains leave the pool.
pub(crate) const TURN_BLOCKS: usize = crate::TURN_ROOM as usize / BLOCK_TOKENS;

/// A prompt prefix whose chain the cache keeps whatever else it holds ([`crate::EngineHandle::pin`]).
struct Pin {
    id: u64,
    namespace: u64,
    tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Key {
    namespace: Namespace,
    parent: Option<u32>,
    tokens: [u32; BLOCK_TOKENS],
}

struct Entry {
    key: Key,
    source: u64,
    producer: u64,
    children: usize,
    touched: u64,
    /// The recurrent state after this chain, and when it was last saved or loaded.
    state: Option<(Arc<Vec<u8>>, u64)>,
    /// On a pinned chain: no eviction takes it, and its checkpoint stays.
    pinned: bool,
}

#[derive(Default)]
pub(crate) struct PrefixCache {
    index: HashMap<Key, u32>,
    entries: HashMap<u32, Entry>,
    capacity: usize,
    clock: u64,
    pub(crate) evictions: u64,
    /// Bytes the checkpoints may hold, and hold.
    pub(crate) state_budget: usize,
    state_bytes: usize,
    pins: Vec<Pin>,
    /// Entries on a pinned chain. They count against the pool, not against `capacity`.
    pinned: usize,
    /// Blocks pinned chains leave free for a turn ([`TURN_BLOCKS`]; a test sets fewer).
    pub(crate) turn_blocks: usize,
}

impl PrefixCache {
    pub(crate) fn configure(&mut self, blocks: &mut Blocks, capacity: usize) {
        self.clear(blocks);
        self.index = HashMap::new();
        self.entries = HashMap::new();
        // At most half the KV pool, which is evicted on demand: at a quarter a 4 GB GPU could not
        // keep a 3.2k-token tool prompt, so each MCP turn prefilled 38% of it again (first token
        // 806 ms, against 73 ms at a half, the same answers; toolcall-opt item 4).
        self.capacity = capacity.min(blocks.total() / 2).min(MAX_ENTRIES);
        self.state_budget = CHECKPOINT_BYTES;
        self.turn_blocks = TURN_BLOCKS;
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Every indexed block, pinned or not.
    pub(crate) fn blocks(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.keys().copied()
    }

    /// The blocks demand eviction may take: those off every pinned chain.
    pub(crate) fn unpinned(&self) -> impl Iterator<Item = u32> + '_ {
        (self.entries.iter())
            .filter(|(_, e)| !e.pinned)
            .map(|(&block, _)| block)
    }

    /// Entries on a pinned chain.
    pub(crate) fn pinned(&self) -> usize {
        self.pinned
    }

    /// Whether `tokens` are exactly a pin's prompt in `namespace`: a read of one.
    pub(crate) fn is_pin(&self, namespace: u64, tokens: &[u32]) -> bool {
        (self.pins.iter()).any(|p| p.namespace == namespace && p.tokens == tokens)
    }

    /// Keeps the chain of `tokens` in `namespace` against eviction, from now on and as it is
    /// published: the chain may pass `capacity` while the pool keeps `turn_blocks` free.
    pub(crate) fn pin(&mut self, id: u64, namespace: u64, tokens: Vec<u32>) {
        self.pins.push(Pin {
            id,
            namespace,
            tokens,
        });
        self.mark();
    }

    /// Ends pin `id`; a chain no other pin holds is evictable again.
    pub(crate) fn unpin(&mut self, id: u64) {
        self.pins.retain(|p| p.id != id);
        self.mark();
    }

    /// Evicts down to `capacity`, which a pin's end leaves its chain over.
    pub(crate) fn trim(&mut self, blocks: &mut Blocks) {
        while self.entries.len() - self.pinned > self.capacity
            && self.evict(blocks, &HashSet::new())
        {}
    }

    /// Forgets every pin: a reload onto another model leaves nothing they could hold.
    pub(crate) fn unpin_all(&mut self) {
        self.pins.clear();
        self.mark();
    }

    /// The pinned prompts as (id, namespace, tokens), for a reload's carry.
    pub(crate) fn pins(&self) -> impl Iterator<Item = (u64, u64, &[u32])> + '_ {
        (self.pins.iter()).map(|p| (p.id, p.namespace, &p.tokens[..]))
    }

    /// Whole blocks the pinned prompts hold between them, plus those of `extra` (namespace and
    /// tokens): a shared prefix counts once. Sorted, a set of prefixes is the sum of their
    /// lengths less each neighbour's common prefix.
    pub(crate) fn pinned_blocks_with(&self, extra: Option<(u64, &[u32])>) -> usize {
        let mut all: Vec<(u64, &[u32])> = (self.pins())
            .map(|(_, ns, t)| (ns, t))
            .chain(extra)
            .collect();
        all.sort();
        let blocks = |t: &[u32]| t.len() / BLOCK_TOKENS;
        let shared = |a: &[u32], b: &[u32]| {
            (a.iter().zip(b).take_while(|(x, y)| x == y).count() / BLOCK_TOKENS)
                .min(blocks(a))
                .min(blocks(b))
        };
        let mut total = 0;
        for (i, &(ns, t)) in all.iter().enumerate() {
            total += blocks(t);
            if let Some(&(pns, p)) = i.checked_sub(1).and_then(|j| all.get(j)) {
                total -= if pns == ns { shared(p, t) } else { 0 };
            }
        }
        total
    }

    /// The whole blocks of `tokens`' chain in `namespace`, as far as the index holds it.
    pub(crate) fn chain(&self, namespace: Namespace, tokens: &[u32]) -> Vec<u32> {
        let mut parent = None;
        let mut found = Vec::new();
        for chunk in tokens.chunks_exact(BLOCK_TOKENS) {
            let key = Key {
                namespace,
                parent,
                tokens: chunk.try_into().expect("full block"),
            };
            let Some(&block) = self.index.get(&key) else {
                break;
            };
            found.push(block);
            parent = Some(block);
        }
        found
    }

    /// Sets every entry's `pinned` from the pins: each pin's chain, under each version of its
    /// namespace the index holds a root for.
    fn mark(&mut self) {
        for e in self.entries.values_mut() {
            e.pinned = false;
        }
        let roots: HashSet<Namespace> = (self.entries.values())
            .filter(|e| e.key.parent.is_none())
            .map(|e| e.key.namespace)
            .collect();
        let mut held = HashSet::new();
        for p in &self.pins {
            for &ns in roots.iter().filter(|ns| ns.0 == p.namespace) {
                held.extend(self.chain(ns, &p.tokens));
            }
        }
        for block in &held {
            self.entries.get_mut(block).expect("indexed").pinned = true;
        }
        self.pinned = held.len();
    }

    pub(crate) fn clear(&mut self, blocks: &mut Blocks) {
        for block in self.entries.keys() {
            blocks.release(*block);
        }
        self.entries.clear();
        self.index.clear();
        self.state_bytes = 0;
        self.pinned = 0;
    }

    pub(crate) fn lookup(&self, namespace: Namespace, tokens: &[u32]) -> Vec<u32> {
        let mut parent = None;
        let mut found = Vec::new();
        // Keep final-prompt logits private, including exact block-aligned hits.
        let limit = tokens.len().saturating_sub(1) / BLOCK_TOKENS;
        for chunk in tokens.chunks_exact(BLOCK_TOKENS).take(limit) {
            let key = Key {
                namespace,
                parent,
                tokens: chunk.try_into().expect("full block"),
            };
            let Some(&block) = self.index.get(&key) else {
                break;
            };
            found.push(block);
            parent = Some(block);
        }
        found
    }

    /// Pinning a descendant also keeps the index ancestors needed to evict its chain safely.
    pub(crate) fn protect_ancestors(&self, protected: &mut HashSet<u32>) {
        let mut pending: Vec<_> = protected.iter().copied().collect();
        while let Some(block) = pending.pop() {
            if let Some(parent) = self.entries.get(&block).and_then(|e| e.key.parent)
                && protected.insert(parent)
            {
                pending.push(parent);
            }
        }
    }

    pub(crate) fn touch(&mut self, chain: &[u32]) {
        self.clock += 1;
        for block in chain {
            self.entries
                .get_mut(block)
                .expect("looked up block")
                .touched = self.clock;
        }
    }

    /// Drops every checkpoint, keeping the KV entries.
    pub(crate) fn drop_checkpoints(&mut self) {
        for e in self.entries.values_mut() {
            e.state = None;
        }
        self.state_bytes = 0;
    }

    /// The blocks of `chain` up to the last that holds a checkpoint; 0 when none does.
    pub(crate) fn checkpointed(&self, chain: &[u32]) -> usize {
        chain
            .iter()
            .rposition(|b| self.entries[b].state.is_some())
            .map_or(0, |i| i + 1)
    }

    /// `block`'s checkpoint, now the most recently used.
    pub(crate) fn checkpoint(&mut self, block: u32) -> Option<Arc<Vec<u8>>> {
        self.clock += 1;
        let (state, used) = self.entries.get_mut(&block)?.state.as_mut()?;
        *used = self.clock;
        Some(state.clone())
    }

    /// Whether `block` holds a checkpoint.
    pub(crate) fn has_checkpoint(&self, block: u32) -> bool {
        self.entries.get(&block).is_some_and(|e| e.state.is_some())
    }

    /// Drops `block`'s checkpoint, keeping its KV.
    pub(crate) fn drop_checkpoint(&mut self, block: u32) {
        if let Some(e) = self.entries.get_mut(&block)
            && let Some((state, _)) = e.state.take()
        {
            self.state_bytes -= state.len();
        }
    }

    /// Retires a pin read's last mark `old` once its next, `new`, is saved. A checkpoint another
    /// pin's chain ends on stays: that pin's carry and export stop at it, and without it find none.
    pub(crate) fn retire_checkpoint(&mut self, new: u32, old: u32) {
        let Some(ns) = self.entries.get(&old).map(|e| e.key.namespace) else {
            return;
        };
        if !self.has_checkpoint(new) {
            return;
        }
        let needed = (self.pins.iter())
            .any(|p| p.namespace == ns.0 && self.chain(ns, &p.tokens).last() == Some(&old));
        if !needed {
            self.drop_checkpoint(old);
        }
    }

    /// Whether `block` is an entry that holds no checkpoint yet.
    pub(crate) fn wants_checkpoint(&self, block: u32) -> bool {
        self.entries.get(&block).is_some_and(|e| e.state.is_none())
    }

    /// A zeroed buffer of `n` bytes for a checkpoint, then the least recently used ones that
    /// leave it no room under the budget go, a pinned chain's never; None (and nothing evicted)
    /// when the budget is smaller, the pinned ones leave it no room, or the host cannot allocate it.
    pub(crate) fn reserve(&mut self, n: usize) -> Option<Vec<u8>> {
        // Checked before anything is evicted: a pinned chain's bytes are never taken, and an
        // eviction that ends in None would have cost the others their checkpoints for nothing.
        let pinned = (self.entries.values())
            .filter(|e| e.pinned)
            .filter_map(|e| e.state.as_ref().map(|s| s.0.len()))
            .sum::<usize>();
        if pinned.saturating_add(n) > self.state_budget {
            return None;
        }
        let state = crate::engine::host_bytes(n)?;
        while self.state_bytes.saturating_add(n) > self.state_budget {
            let (_, e) = (self.entries.iter_mut())
                .filter(|(_, e)| e.state.is_some() && !e.pinned)
                .min_by_key(|(block, e)| (e.state.as_ref().map(|s| s.1), **block))?;
            self.state_bytes -= e.state.take().expect("filtered").0.len();
        }
        Some(state)
    }

    /// Keeps `state` as `block`'s checkpoint if the block is still an entry without one and the
    /// budget still has room; else drops it.
    pub(crate) fn keep(&mut self, block: u32, state: Vec<u8>) {
        self.clock += 1;
        let room = self.state_bytes.saturating_add(state.len()) <= self.state_budget;
        if let Some(e) = self.entries.get_mut(&block)
            && e.state.is_none()
            && room
        {
            self.state_bytes += state.len();
            e.state = Some((Arc::new(state), self.clock));
        }
    }

    // ponytail: bounded linear leaf selection (8192 entries); use a leaf LRU queue if measured
    // pressure-workload CPU time justifies the extra bookkeeping.
    /// Demand eviction only. A leaf with only the index reference has no reader to invalidate.
    pub(crate) fn evict(&mut self, blocks: &mut Blocks, excluded: &HashSet<u32>) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(block, e)| {
                e.children == 0
                    && !e.pinned
                    && blocks.refs[**block as usize] == 1
                    && !excluded.contains(block)
            })
            .min_by_key(|(block, e)| (e.touched, **block))
            .map(|(&block, _)| block);
        if let Some(block) = victim {
            self.remove(block, blocks);
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Puts an indexed entry on a pinned chain.
    fn hold(&mut self, block: u32) {
        let e = self.entries.get_mut(&block).expect("indexed");
        self.pinned += usize::from(!e.pinned);
        e.pinned = true;
    }

    fn remove(&mut self, block: u32, blocks: &mut Blocks) {
        let entry = self.entries.remove(&block).expect("cache entry");
        assert_eq!(entry.children, 0, "remove children before parent");
        self.index.remove(&entry.key);
        self.pinned -= usize::from(entry.pinned);
        self.state_bytes -= entry.state.map_or(0, |(s, _)| s.len());
        if let Some(parent) = entry.key.parent {
            self.entries
                .get_mut(&parent)
                .expect("parent retained")
                .children -= 1;
        }
        blocks.release(block);
    }

    /// Release/reset withdraws this session's entries and their descendants. Readers keep their
    /// pins; withdrawing eligibility never frees their physical KV or changes their tables.
    pub(crate) fn release_session(&mut self, session: u64, blocks: &mut Blocks) {
        self.withdraw(blocks, |entry| entry.source == session);
    }

    /// A chain a successful stateless request read is no longer one session's: that session's
    /// release leaves it, as it leaves stateless entries. A producer's cancel still withdraws it.
    /// The chain is the request's own block table (`held`) while each block is the indexed entry
    /// of `tokens`' next full block, under the previous one, in `namespace`, up to the last block
    /// before the prompt's final token, as `lookup` bounds it; it stops at the first that is not.
    /// No allocation and no index hashing: the step loop calls it as a request finishes.
    pub(crate) fn adopt(&mut self, namespace: Namespace, tokens: &[u32], held: &[u32]) {
        let limit = tokens.len().saturating_sub(1) / BLOCK_TOKENS;
        let mut parent = None;
        for (chunk, &block) in tokens.chunks_exact(BLOCK_TOKENS).take(limit).zip(held) {
            let Some(entry) = self.entries.get_mut(&block) else {
                break;
            };
            let key = &entry.key;
            if key.namespace != namespace || key.parent != parent || key.tokens[..] != *chunk {
                break;
            }
            entry.source = 0;
            parent = Some(block);
        }
    }

    /// Each entry's source session.
    #[cfg(test)]
    pub(crate) fn sources(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.values().map(|e| e.source)
    }

    /// The index's and the entries' allocated capacities.
    #[cfg(test)]
    pub(crate) fn heap(&self) -> (usize, usize) {
        (self.index.capacity(), self.entries.capacity())
    }

    pub(crate) fn cancel(&mut self, producer: u64, blocks: &mut Blocks) {
        self.withdraw(blocks, |entry| entry.producer == producer);
    }

    fn withdraw(&mut self, blocks: &mut Blocks, matches: impl Fn(&Entry) -> bool) {
        let mut removed: HashSet<_> = self
            .entries
            .iter()
            .filter(|(_, e)| matches(e))
            .map(|(&block, _)| block)
            .collect();
        if removed.is_empty() {
            return;
        }
        loop {
            let old = removed.len();
            for (&block, e) in &self.entries {
                if e.key.parent.is_some_and(|p| removed.contains(&p)) {
                    removed.insert(block);
                }
            }
            if old == removed.len() {
                break;
            }
        }
        while !removed.is_empty() {
            let leaves: Vec<_> = removed
                .iter()
                .copied()
                .filter(|b| self.entries[b].children == 0)
                .collect();
            assert!(!leaves.is_empty(), "acyclic prefix chain");
            for block in leaves {
                self.remove(block, blocks);
                removed.remove(&block);
            }
        }
    }

    /// Called after synchronized successful computation, never on planned or sampled-only KV.
    pub(crate) fn publish(
        &mut self,
        blocks: &mut Blocks,
        namespace: Namespace,
        source: u64,
        producer: u64,
        tokens: &[u32],
        physical: &mut [u32],
    ) {
        if self.capacity == 0 {
            return;
        }
        self.clock += 1;
        // The blocks of this prompt a pin holds: the common prefix with each pin of the namespace.
        let held = (self.pins.iter())
            .filter(|p| p.namespace == namespace.0)
            .map(|p| {
                p.tokens
                    .iter()
                    .zip(tokens)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .max()
            .map_or(0, |n| n / BLOCK_TOKENS);
        let mut parent = None;
        for (i, (chunk, physical_block)) in
            tokens.chunks_exact(BLOCK_TOKENS).zip(physical).enumerate()
        {
            let block = *physical_block;
            let key = Key {
                namespace,
                parent,
                tokens: chunk.try_into().expect("full block"),
            };
            if let Some(&canonical) = self.index.get(&key) {
                // Concurrent cold requests may compute the same full prefix in one step. Once
                // synchronized, adopt its canonical block instead of retaining a private copy
                // beside an idle index ancestor. Only this sequence's reference is released;
                // other readers keep their pins. No device copy or shared write is involved.
                if block != canonical {
                    blocks.retain(canonical);
                    blocks.release(block);
                    *physical_block = canonical;
                }
                if i < held {
                    self.hold(canonical);
                }
                parent = Some(canonical);
                continue;
            }
            // A physical block is immutable and has only one token history/namespace.
            if self.entries.contains_key(&block) {
                break;
            }
            // A pinned block passes `capacity` while the pool keeps a turn's room.
            let pinned = i < held && self.pinned + self.turn_blocks < blocks.total();
            let excluded: HashSet<_> = parent.into_iter().collect();
            if !pinned
                && self.entries.len() - self.pinned >= self.capacity
                && !self.evict(blocks, &excluded)
            {
                break;
            }
            self.pinned += usize::from(pinned);
            blocks.retain(block);
            if let Some(p) = parent {
                self.entries.get_mut(&p).expect("parent").children += 1;
            }
            self.index.insert(key, block);
            self.entries.insert(
                block,
                Entry {
                    key,
                    source,
                    producer,
                    children: 0,
                    touched: self.clock,
                    state: None,
                    pinned,
                },
            );
            parent = Some(block);
        }
    }
}
