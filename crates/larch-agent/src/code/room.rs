//! What the loaded deployment leaves code mode room for, decided once per run: how long the
//! planner and the writer think, how many writers run at once, which files are small enough
//! to rewrite whole, how much the repository map and a writer's brief may hold, and how many
//! attempts run side by side. Every choice is a share of the room, never a model's name or size, so a
//! bigger context or KV pool only ever makes the choice more generous.

use crate::{
    code::{
        r#loop::{BODY_TOKENS, BRIEF_BYTES, LIST_BYTES, MAP_TOKENS, MIN_ROOM, PROMPT_TOKENS},
        session,
    },
    config::{MAX_THINKING_BUDGET, Thinking},
    skills::read_papers::{self, TEMPLATE_TOKENS},
    tools::code::VIEW_TOKENS,
    workers::Workers,
};

/// Tokens of a code line, at the four bytes a token `tool::truncate` counts and a line of about
/// 40 bytes.
const LINE_TOKENS: u32 = 10;

/// The most attempts run side by side unasked: the plan's 2 to 4, past which each stream's share
/// of the batch slows every attempt more than another one helps.
// ponytail: from the plan, not measured; raise it when phase 5's Polyglot runs show a gain past 4
const MAX_ATTEMPTS: usize = 4;

/// The fewest tokens a run generates when the user sets none (`Room::tokens`): the fixed budget
/// every run had before the room sized it.
const MIN_TOKENS: u64 = 60_000;

/// The fewest tool calls an attempt makes when the user sets none (`Room::tool_calls`): the
/// fixed budget every attempt had before the room sized it.
const MIN_CALLS: u32 = 40;

/// What a run's deployment leaves room for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// A planner step's thinking and answer cap: config.toml's, its thinking at most an eighth
    /// of what the repository map leaves of the context, as the step's history and call need
    /// the rest.
    pub planner: Thinking,
    /// The writer's thinking and its file's cap (`body`): the file gets half of what the brief
    /// leaves, at least a small model's page (`BODY_TOKENS`); thinking gets a quarter of the
    /// rest, up to `MAX_THINKING_BUDGET`, and none when that is less than `MIN_ROOM`.
    pub writer: Thinking,
    /// The writers that run at once: as many as config.toml's workers (`Workers::at_once`) and
    /// the KV pool hold, each with its brief, thinking and a page's worth of file.
    pub writers: usize,
    /// The longest file rewritten whole rather than edited: its text takes at most half the
    /// writer's file cap.
    pub rewrite_max_lines: usize,
    /// The repository map's tokens: an eighth of the context, at least `MAP_TOKENS`.
    pub map_tokens: u32,
    /// The bytes of the project's other files a writer's brief carries: what the writer's file
    /// and thinking leave, up to a quarter of the context, at least the `BRIEF_BYTES` an 8k
    /// context fits beside a page.
    pub brief_bytes: usize,
    /// The attempts a task with tests runs side by side when the user sets none: as many as
    /// config.toml's workers allow and the KV pool holds, each its whole context and its
    /// writers, up to `MAX_ATTEMPTS`; 1 when the pool is not known.
    pub attempts: usize,
    /// The tokens a `view` or `lookup` result takes: three quarters of what a compaction keeps
    /// whole of a step past a first prompt with a full map (`session::kept`), so a view the
    /// model just asked for is not cut there, and the next one does not compact again; at most
    /// `VIEW_TOKENS`, at least `MIN_ROOM`.
    pub view_tokens: u32,
    /// The tokens of a change's lines in a write's result (`session::changed`): a quarter of a
    /// view's.
    pub changed_tokens: u32,
    /// The tokens a run generates when the user sets none (`BUDGET`): two contexts' worth, so a
    /// big context's writer, whose one file may take half of it, is never cut short by a fixed
    /// cap; at least `MIN_TOKENS`.
    // ponytail: two contexts by judgment; raise it when a big run stops on its tokens
    pub tokens: u64,
    /// The tokens of read results since the last change that no tool call is counted for
    /// (`BUDGET`): two contexts' worth, so a bigger context explores more before its reads
    /// count.
    // ponytail: two contexts by judgment; raise it when a big repository's run stops on its
    // calls before its first change
    pub read_tokens: u32,
    /// The tool calls an attempt makes when the user sets none (`BUDGET`): a call per 350 tokens
    /// of context, at least `MIN_CALLS`; the no-progress rule, not this, stops a loop.
    pub tool_calls: u32,
}

impl Room {
    /// The room a context of `ctx_max` tokens and a KV pool of `kv_pool_blocks` blocks (0 is
    /// not known, and caps nothing) leave, for `workers` and config.toml's `planner` thinking.
    pub fn new(ctx_max: u32, kv_pool_blocks: u32, workers: Workers, planner: Thinking) -> Room {
        let map_tokens = MAP_TOKENS.max(ctx_max / 8);
        let map = map_tokens + TEMPLATE_TOKENS as u32;
        let planner = Thinking {
            budget: planner.budget.min(ctx_max.saturating_sub(map) / 8),
            ..planner
        };
        // The file and the thinking are shared out beside a page-sized brief, so a bigger brief
        // never takes from them; the brief then gets what they leave, up to a quarter.
        let page = (BRIEF_BYTES + LIST_BYTES) as u32 / 4 + TEMPLATE_TOKENS as u32;
        let free = ctx_max.saturating_sub(page);
        let file = (free / 2).max(BODY_TOKENS);
        let think = (free.saturating_sub(file) / 4).min(MAX_THINKING_BUDGET);
        let writer = Thinking {
            budget: if think < MIN_ROOM { 0 } else { think },
            answer_max_tokens: file,
        };
        let left = ctx_max.saturating_sub(file + writer.budget + page - BRIEF_BYTES as u32 / 4);
        // A quarter of the context in tokens is its bytes.
        let brief_bytes = (left as usize * 4).min(ctx_max as usize).max(BRIEF_BYTES);
        // What a writer holds: a real brief and file are a page's worth each, and the engine
        // gives blocks as tokens come, so the caps would reserve many times what it uses.
        let each = u64::from(page + writer.budget + BODY_TOKENS);
        let pool = u64::from(kv_pool_blocks);
        let writers = read_papers::at_once(workers.at_once(), workers, pool, each);
        // An attempt holds its whole context, parked between steps, and its writers; a pool not
        // known holds one.
        let attempt = u64::from(ctx_max) + each * writers as u64;
        let attempts = match pool {
            0 => 1,
            _ => read_papers::at_once(MAX_ATTEMPTS, workers, pool, attempt).min(workers.at_once()),
        };
        let base = map + PROMPT_TOKENS;
        // Three quarters: code takes more than the four bytes a token the view counts.
        let view_tokens =
            ((session::kept(ctx_max, base) - base) / 4 * 3).clamp(MIN_ROOM, VIEW_TOKENS);
        Room {
            planner,
            writer,
            writers,
            rewrite_max_lines: (file / LINE_TOKENS / 2) as usize,
            map_tokens,
            brief_bytes,
            attempts,
            view_tokens,
            changed_tokens: view_tokens / 4,
            tokens: (2 * u64::from(ctx_max)).max(MIN_TOKENS),
            read_tokens: ctx_max.saturating_mul(2),
            // A bigger context holds the longer histories a big repository's task takes (a
            // Django fix took 64 steps unaided), where 40 calls ran out mid-test.
            tool_calls: (ctx_max / 350).max(MIN_CALLS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// config.toml's thinking when the user turns it up: the planner's ceiling.
    const FULL: Thinking = Thinking {
        budget: MAX_THINKING_BUDGET,
        answer_max_tokens: 0,
    };

    const WORKERS: Workers = Workers {
        max_workers: 8,
        prefix_cache: true,
    };

    /// An 8k context whose pool holds two of them, as a small model on a laptop.
    #[test]
    fn a_small_deployment_writes_at_once_with_no_thinking() {
        let room = Room::new(8192, 1024, WORKERS, FULL);
        let writer = Thinking {
            budget: 0,
            answer_max_tokens: BODY_TOKENS,
        };
        assert_eq!(room.writer, writer);
        assert!(room.planner.budget < FULL.budget, "{room:?}");
        assert_eq!(room.writers, 2);
        assert_eq!(room.rewrite_max_lines, 204);
    }

    /// At 8k a file has no room to think beside, whatever the model (web-cafe lost its
    /// index.html to a 1024-token think); at 16k the writer thinks.
    #[test]
    fn the_writer_thinks_from_16k_not_at_8k() {
        let at8 = Room::new(8192, 2048, WORKERS, FULL);
        assert_eq!(at8.writer.budget, 0);
        assert_eq!(at8.writer.answer_max_tokens, BODY_TOKENS);
        let at16 = Room::new(16_384, 2048, WORKERS, FULL);
        assert!(at16.writer.budget >= 1024, "{at16:?}");
        assert!(at16.writer.answer_max_tokens >= BODY_TOKENS, "{at16:?}");
        // The brief, the thinking and the file fit the context.
        let brief = (at16.brief_bytes + LIST_BYTES) as u32 / 4 + TEMPLATE_TOKENS as u32;
        assert!(brief + at16.writer.budget + at16.writer.answer_max_tokens <= 16_384);
        assert!(at16.planner.budget > at8.planner.budget);
        assert!(at16.rewrite_max_lines > at8.rewrite_max_lines);
    }

    /// An 8k context keeps the 40 calls it always had; 32k gets about twice as many.
    #[test]
    fn the_call_budget_grows_with_the_context() {
        assert_eq!(Room::new(8192, 1024, WORKERS, FULL).tool_calls, 40);
        assert_eq!(Room::new(32_768, 4096, WORKERS, FULL).tool_calls, 93);
    }

    /// The default planner thinking stays as config.toml sets it wherever it fits.
    #[test]
    fn the_planner_keeps_config_thinking_that_fits() {
        let config = Thinking::default();
        assert_eq!(Room::new(8192, 1024, WORKERS, config).planner, config);
    }

    /// Never worse on a big machine: a large model with a 128k context and a pool of two of them
    /// thinks in full, runs every writer config.toml allows, and still edits a file too long to
    /// rewrite.
    #[test]
    fn a_big_deployment_gets_every_generous_choice() {
        let small = Room::new(8192, 1024, WORKERS, FULL);
        let big = Room::new(131_072, 2 * 131_072 / 16, WORKERS, FULL);
        assert_eq!(big.planner, FULL);
        assert_eq!(big.writer.budget, MAX_THINKING_BUDGET);
        assert!(big.writer.answer_max_tokens > small.writer.answer_max_tokens);
        assert_eq!(big.writers, WORKERS.at_once());
        assert!(big.writers > small.writers);
        assert!(big.rewrite_max_lines > small.rewrite_max_lines);
        assert!(big.rewrite_max_lines < 10_000, "{big:?}");
        assert!(big.map_tokens > small.map_tokens && big.brief_bytes > small.brief_bytes);
        // A run may spend more than the fixed budget's tokens, more than one big file's.
        assert_eq!(small.tokens, MIN_TOKENS);
        assert!(big.read_tokens > small.read_tokens);
        assert!(big.tokens > small.tokens);
        assert!(big.tokens > u64::from(big.writer.answer_max_tokens) * 2);
    }

    /// An 8k context keeps the map and the brief it had before the room decided them.
    #[test]
    fn a_small_context_keeps_the_fixed_map_and_brief() {
        let room = Room::new(8192, 1024, WORKERS, FULL);
        assert_eq!(
            (room.map_tokens, room.brief_bytes),
            (MAP_TOKENS, BRIEF_BYTES)
        );
    }

    /// Attempts side by side: two on a small made-up deployment whose pool holds two
    /// contexts with their writers, the plan's most (4) on a big one, one where the pool holds
    /// one, is not known, or there is one worker.
    #[test]
    fn attempts_are_what_the_pool_holds_with_each_ones_writers() {
        let three = Workers {
            max_workers: 3,
            prefix_cache: true,
        };
        let small = Room::new(8192, 6 * 8192 / 16, three, FULL);
        assert_eq!((small.writers, small.attempts), (2, 2), "{small:?}");
        let big = Room::new(131_072, 16 * 131_072 / 16, WORKERS, FULL);
        assert_eq!(big.attempts, 4, "{big:?}");
        assert_eq!(Room::new(8192, 4 * 8192 / 16, three, FULL).attempts, 1);
        assert_eq!(Room::new(131_072, 0, WORKERS, FULL).attempts, 1);
        let one = Workers {
            max_workers: 1,
            prefix_cache: false,
        };
        assert_eq!(Room::new(131_072, 1 << 20, one, FULL).attempts, 1);
    }

    /// The bigger brief takes nothing from the writer: its thinking, file cap and rewrite size
    /// are what a page-sized brief left them before the brief grew, at every size.
    #[test]
    fn the_brief_grows_without_shrinking_the_writer() {
        for (ctx, think, file, lines) in [
            (16_384, 1598, 6394, 319),
            (32_768, 3646, 14_586, 729),
            (131_072, 4096, 63_738, 3186),
        ] {
            let room = Room::new(ctx, 8 * ctx / 16, WORKERS, FULL);
            let got = (room.writer.budget, room.writer.answer_max_tokens);
            assert_eq!(
                (got, room.rewrite_max_lines),
                ((think, file), lines),
                "{room:?}"
            );
            assert!(room.brief_bytes > BRIEF_BYTES, "{room:?}");
            let brief = (room.brief_bytes + LIST_BYTES) as u32 / 4 + TEMPLATE_TOKENS as u32;
            assert!(brief + think + file <= ctx, "{room:?}");
        }
    }

    /// A pool of one 128k context, the engine's least, still runs at least the four writers a
    /// reply took before the room decided.
    #[test]
    fn a_pool_of_one_long_context_runs_several_writers() {
        let room = Room::new(131_072, 131_072 / 16, WORKERS, FULL);
        assert!(room.writers >= WORKERS.at_once().min(4), "{room:?}");
    }

    /// A view fits what a compaction keeps whole past a first prompt with a full map at 8k, so
    /// the view that sets one off survives it; at 4k a full map leaves nothing, and a view is
    /// the least that shows lines. From 32k on a view and a change's lines are what they were
    /// before the room sized them.
    #[test]
    fn a_view_fits_what_a_compaction_keeps() {
        let at4 = Room::new(4096, 512, WORKERS, FULL);
        assert_eq!(
            (at4.view_tokens, at4.changed_tokens),
            (MIN_ROOM, MIN_ROOM / 4)
        );
        let at8 = Room::new(8192, 1024, WORKERS, FULL);
        let base = at8.map_tokens + PROMPT_TOKENS;
        assert_eq!(at8.map_tokens, MAP_TOKENS);
        assert!(at8.view_tokens > 800, "{at8:?}");
        assert!(
            base + at8.view_tokens <= session::kept(8192, base),
            "{at8:?}"
        );
        for ctx in [32_768, 131_072] {
            let big = Room::new(ctx, 2 * ctx / 16, WORKERS, FULL);
            assert_eq!((big.view_tokens, big.changed_tokens), (VIEW_TOKENS, 1000));
        }
    }

    /// One worker in config.toml is one writer, whatever the pool.
    #[test]
    fn writers_never_pass_config_workers() {
        let one = Workers {
            max_workers: 1,
            prefix_cache: false,
        };
        assert_eq!(Room::new(131_072, 1 << 20, one, FULL).writers, 1);
        // A pool not known caps nothing.
        assert_eq!(Room::new(8192, 0, WORKERS, FULL).writers, WORKERS.at_once());
    }
}
