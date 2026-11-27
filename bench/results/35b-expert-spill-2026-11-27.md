# m13: hot experts on the card, the cold tail read in place (expert spill), 27 Nov 2026

Test machine: RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, Windows 11 + WSL2 (8 logical CPUs in WSL). Model m13
(`bench/models.toml`). Raw results: [`35b-expert-spill-2026-11-27/`](35b-expert-spill-2026-11-27/).

## Summary

**Kept.** A static split's hot experts stay on the card and only the cold tail is read in place from page-locked host
memory, inside the decode graph (`mmvq_id_mixed` over a wrapped host range), with the split capped at the layers the
decode cost picks. Decode and check rows give the resident load's bits. Only `LARCH_EXPERT_CACHE=1` with
`bench/routes/m13.experts` beside the model takes it (the default is unchanged). B is `e3ffd4b`; every B number is
**with the shipped profile**.

| Load | larch B (with the shipped profile) | llama.cpp ncm 3 | ncm 4 | the fewest ncm that fits its free memory | larch A |
|---|---|---|---|---|---|
| W4@1 | 161.8 | 130.4: **+24.1 %** | 125.3: +29.1 % | ncm 4: +29.1 % | 136.0: +19.0 % |
| W4@4 | 298.2 | 262.4 (straight after B): **+13.6 %** | 232.1: +28.5 % | ncm 5, 224.6: +32.8 % | 245.8: +21.3 % |
| W4@8, 4 seats | 300.5 | 307.2: **−2.2 %, behind** | 288.8: +4.1 % | ncm 7, 239.2: +25.6 % | 245.3: +22.5 % |
| W4@8, 8 seats (measurement) | 335.8 | 307.2: **+9.3 %** | 288.8: +16.3 % | ncm 7: +40.4 % | — |
| W4@16 | 300.4 | 68.5 | 71.6 | ncm 9, 235.1: +27.8 % | 246.1: +22.1 % |

At W4@16 llama.cpp's faster settings past its free memory: ncm 8 243.5 (+23.4 %), ncm 7 266.3 (+12.8 %). One full run a
cell for B; the A B B A slices (W4@4 +22.0 %, W4@1 +20.0 %, spreads under 1 %) are the controlled comparison. llama.cpp
cells are last night's except W4@4's pair and today's L runs. The goal (ahead of ncm 3 at W4@4, no loss at W4@1) is met;
W4@8 at 4 seats is 2.2 % behind ncm 3 and ahead with 8 seats (the seats default is a separate decision).

## Plan

1. Setup: A is `ef28629` rebuilt (the night's `b3a53bf3…` binary is gone); cherry-pick the CUDA wrap of page-locked
   host memory (`2b90422`) and, in a commit of its own, drop its whole-layer mapped decode path.
2. L: llama.cpp load probes, then A, llama, llama, A at W4@4 and W4@8 and the W4@16 runs; the release table on A.
3. K1 (written during L, run after it): CUDA `mmvq_id_mixed` takes cold experts at the hot ones' dtype; a ctest over a
   wrapped page-locked range against `mmvq_id` on one tensor, bit for bit, at 1, 4 and 40 rows, red before green.
4. K3 (likewise): `expert_residency` caps the split layers (Lane E's rule) and picks the cap by the decode cost from
   `SPLIT_LAYER_US` and `COLD_PAIR_US`; tests on `bench/routes/m13.experts`; prints G2's table.
5. G: one GPU hold after L: G1 (a cold pair's cost in a captured graph), G2 (K3's table), G3 (`fused_bench`); GO if
   the cheapest cap's 4-row step is at most 12.0 ms with its 1-row step at most 6.2 ms, else stop and report.
6. K2 (GO only), test first: split loads page-locked; decode, check and a chunk's leading decode rows of split layers
   through `Experts::Mixed` (cold experts from `Buffer::view_host`), prompt rows on `split_ffn`; tests in `tiny.rs`.
7. SMOKE: one B run at W4@4 `--slice 1/10`: every decode bucket in one graph piece, replayed 4-row step at most 12.0 ms.
8. P: code review, over-engineering review, fmt, clippy, tests, ctest, then m13 parity (prefill, decode, decode -p 4).
9. M: A B B A at W4@4 and W4@1, full W4@4 and W4@8 on B, W4@8 with 8 seats, W4@16; histogram protocol on every run.
10. R: this report and, if kept, a draft PR to dev; parity `--mode all` on B after the hard end.

## L: the release table on A

A is `ef28629`, the kept stack of 27 Nov (whole host layers, the prefill pool, 4 state slots), rebuilt today. Its W4@4,
W4@8 and W4@16 runs ran today (12:34 to 13:25 CT), interleaved with llama.cpp: A, llama.cpp, llama.cpp, A at W4@4 and
W4@8, then A once and llama.cpp at `--n-cpu-moe` 8 and 7 at W4@16. W4@1 and the other llama.cpp cells are last night's
(27 Nov, UTC hour given), A's W4@1 there from the same commit's `b3a53bf3…` build. Every A run showed
`larch_host_experts_plan,4`. Memory: llama.cpp's CUDA0 buffers against the free memory it reported (15,009 MiB), from
its load lines; "fits" is the fewest `--n-cpu-moe` at or under it. Probes (`wsl5070-m13-llama-probes.txt`): `-np 4` fits at
ncm 5 (14,890 MiB); `-np 8` is 3 MiB past at ncm 6 (15,012) and fits at ncm 7 (14,682).

| Load | larch A (`ef28629`) tok/s | llama.cpp | tok/s, memory, date; larch's margin |
|---|---|---|---|
| W4@1 | 136.0 (136.0 / 136.1), 27 Nov 08Z | ncm 3 | 130.4 (129.9 / 130.9); +271 MiB past free, 27 Nov 08Z; larch +4.3 %, larch ahead |
|  |  | ncm 4 | 125.3; -93 MiB past free, 27 Nov 05Z; larch +8.6 %, larch ahead |
|  |  | fits: ncm 4 | 125.3; -93 MiB past free, 27 Nov 05Z; larch +8.6 %, larch ahead |
| W4@4 | 245.8 (245.8 / 245.7), today | ncm 3 | 260.9 (262.6 / 259.2); +609 MiB past free, 27 Nov 05Z; larch -5.8 %, larch behind |
|  |  | ncm 4 | 232.1; +245 MiB past free, 27 Nov 06Z; larch +5.9 %, larch ahead |
|  |  | fits: ncm 5 | 224.6 (223.8 / 225.5); -119 MiB past free, today; larch +9.4 %, larch ahead |
| W4@8 | 245.3 (245.5 / 245.2), today | ncm 3 | 307.2; +1061 MiB past free, 27 Nov 06Z; larch -20.1 %, larch behind |
|  |  | ncm 4 | 288.8; +697 MiB past free, 27 Nov 06Z; larch -15.0 %, larch behind |
|  |  | fits: ncm 7 | 239.2 (237.3 / 241.0); -327 MiB past free, today; larch +2.6 %, larch ahead |
| W4@16 | 246.1, today | ncm 3 | 68.5; +1963 MiB past free, 27 Nov 06Z; larch +259.4 %, larch ahead |
|  |  | ncm 4 | 71.6; +1599 MiB past free, 27 Nov 06Z; larch +243.8 %, larch ahead |
|  |  | fits: ncm 9 | 235.1; -119 MiB past free, 27 Nov 06Z; larch +4.7 %, larch ahead |
|  |  | ncm 8 | 243.5; +211 MiB past free, today; larch +1.1 %, noise |
|  |  | ncm 7 | 266.3; +575 MiB past free, today; larch -7.6 %, larch behind |

- **W4@4:** larch is behind ncm 3 (−5.8 %, which loads 609 MiB past its free memory) and ahead of ncm 4 and of the
  fitting ncm 5 (+9.4 %, spreads 0.0 % and 0.8 %).
- **W4@8:** behind ncm 3 and 4 (−20.1 %, −15.0 %); ahead of the fitting ncm 7 by 2.6 %, outside both spreads (0.1 %,
  1.5 %): a win by the rule, a thin one.
- **W4@16: llama.cpp's ncm 7 beats larch**, 266.3 against 246.1 (−7.6 %, one run a side), loading 575 MiB past its free
  memory. ncm 8 (211 MiB past) is level (+1.1 %, noise). Larch's +4.7 % over the fitting ncm 9 stands only against
  settings that fit; at 16 chats llama.cpp's faster settings run past its reported memory, as ncm 3 does at W4@4.
- **W4@1:** ahead of all three (last night's cells).

## G: the gates (GO)

One GPU hold after L (`g1-g3.txt`). **G1** (`g1.cu`): one layer at m13's expert shapes, 100 of 256 experts cold in a
wrapped page-locked range, the split layer's routed experts as the model runs them (`mmvq_id_mixed` gate and up,
`silu_mul`, `mmvq_id_mixed` down, weighted), captured in a graph over 512 rotating id sets, best of four replays; µs a
layer:

| rows | cold a row | resident (`mmvq_id`) | split | split gate+up | split down |
|---|---|---|---|---|---|
| 1 | 0 | 43.0 | 43.5 | 25.8 | 17.3 |
| 1 | 1 | 43.7 | 403.4 | 361.8 | 42.0 |
| 1 | 2 | 42.1 | 741.9 | 648.4 | 102.1 |
| 4 | 0 | 123.8 | 127.5 | 78.8 | 46.4 |
| 4 | 1 in the step | 124.7 | 496.9 | 404.3 | 101.3 |
| 4 | 1 | 125.7 | 1536.4 | 1233.8 | 292.9 |
| 8 | 0 | 234.2 | 239.6 | 149.4 | 86.3 |
| 8 | 1 in the step | 233.2 | 546.7 | 423.0 | 125.5 |
| 8 | 1 | 236.7 | 3083.2 | 2398.9 | 668.4 |

A cold pair costs **359.9, 369.4 and 307.1 µs** at 1, 4 and 8 rows (one in the step against none), and pairs add about
linearly (about 350 µs each). Cold gate and up dominate (Q3_K, 512 rows of 2,048 a matrix: about 3 GB/s over the bus);
cold down (Q4_K, 2,048 rows of 512) reads at about 24 GB/s. The brief's lever for a dominant cold down (the cold
tensor's own alignment in place of `a8 = false`) does not apply: down does not dominate, and Q3_K's decoder does not take
8-byte loads. The table lookup costs 0.5 to 5.4 µs a layer with no cold pair.

**G3** (`fused_bench --rows 1,4,8`, m13's q3_k/q4_k layer): unfused less fused gate_up and down a layer is 10.8, 14.4 and
17.7 µs; with G1's table lookup, `SPLIT_LAYER_US` = 11.3, 18.1, 23.1. Fused gate_up + down a layer: 36.6, 115.4 and
222.1 µs.

**G2** (K3's test, the card's figures: 718 experts' bytes on the host):

| cap | layers split | experts on the host | host share of routed pairs | cold pairs a step at 1, 4, 8 rows |
|---|---|---|---|---|
| 6 | 6 | 718 | 0.246 % | 0.79, 3.15, 6.30 |
| 8 | 8 | 718 | 0.124 % | 0.40, 1.58, 3.17 |
| 10 | 10 | 718 | 0.077 % | 0.25, 0.99, 1.97 |
| 12 | 12 | 718 | 0.054 % | 0.17, 0.69, 1.39 |
| 16 | 16 | 718 | 0.035 % | 0.11, 0.44, 0.89 |
| 20 | 20 | 718 | 0.026 % | 0.08, 0.34, 0.68 |
| uncapped | 37 | 718 | 0.018 % | 0.06, 0.23, 0.47 |

**GO**: step(rows) = base + 4 × fused + N × `SPLIT_LAYER_US` + cold pairs × `COLD_PAIR_US`, base = 5.11, 9.2 and 15.0 ms:

| cap | 1-row step | 4-row step | 8-row step |
|---|---|---|---|
| 6 | 5.608 | 10.934 | 17.962 |
| 8 | 5.490 | 10.392 | 17.046 |
| 12 | 5.455 | 10.135 | 16.591 |
| **15 (chosen)** | **5.470** | **10.113** | **16.535** |
| 16 | 5.478 | 10.115 | 16.530 |
| 20 | 5.513 | 10.149 | 16.558 |
| uncapped (37) | 5.697 | 10.417 | 16.886 |

The lowest 4-row step whose 1-row step is at most 6.2 ms is cap 15: **10.11 ms, under 12.0: GO**. The plan's own cost at
4 seats picks 15 too (11 at 1 row, 16 at 8). Against today's 6.81, 13.8 and 24.9 ms that predicts about +25 % at one
row and +36 % at four, if nothing else moves. Constants in `fit.rs`: `SPLIT_LAYER_US` [11.3, 18.1, 23.1],
`COLD_PAIR_US` [359.9, 369.4, 307.1].

## K: the code, test first

- **K1** (`eff11a0`). CUDA's `mmvq_id_mixed` takes cold experts at the hot experts' dtype as well as Q2_K (the approved
  `larch.h` comment; the call was already valid and only refused). ctest `ops`, new case `mixed wrapped`: the cold
  experts (Q3_K and Q4_K, a third of 16) in a wrapped range of page-locked host memory, `mmvq_id_mixed` against
  `mmvq_id` over one device tensor holding every expert, bit for bit, at 1, 4 and 40 rows, weighted and not. **Red**
  first (the 12 cases refused: "cold experts other than Q2_K"), then **green**: `ops` 1,347 cases, `wrap_host` passes.
  The grouped path from 33 rows (40 rows here) matches; it does not refuse.
- **K3** (`6ac556c`, constants `6f908df`). `expert_residency` splits at most N layers by Lane E's rule and picks N by the
  decode cost above; a histogram whose coldest experts hold as many routes a byte as whole layers keeps whole layers.
  Tests on `bench/routes/m13.experts`: a cap of 8 splits exactly 8 layers within the budget with 0.124 % of routed pairs
  on the host; an even histogram keeps whole layers; the cost picks a cap between 6 and 20 for the test's figures (60 µs
  a split layer, 150 µs a cold pair); a thin tail (3 experts' bytes past the room) splits one layer and keeps the other
  39 whole. 38 fit and plan tests pass. **Red, seen after the fact** (builds were barred during L, so the test first ran
  after its code): in a throwaway worktree, the test against `6ac556c^`'s planner does not compile (no `capped_split`,
  `cheapest_split`, `resident_budget`, `cold_pairs`); its cap assertion through the old planner's own API fails with
  "37 layers split". The branch's tree was not touched.
- **K2** (`1b29835`). A static split's host experts are page-locked on CUDA; the load wraps them as one device view
  (`Buffer::view_host`) with each layer's expert table. A split layer's decode and check rows, and the decode rows that
  lead a prompt chunk, run `mmvq_id_mixed` over its device slots and that view inside the step's graph; prompt rows keep
  `split_ffn` (now from a row offset), eagerly. Test `split_layers_decode_their_cold_experts_in_place` (`tiny.rs`, CUDA;
  layer 0 keeps 2 of 4 experts, layer 1 one, so every row routes a cold expert in layer 1): decode equals the resident
  load bit for bit at 1 and 4 rows; a check step's 3 rows equal their one-row decodes; a decode row beside a 7-row prompt
  chunk equals the row alone, and only the chunk's rows count host pairs (28, a cold one a row at least); 1- and 4-row
  decode steps capture in one graph piece and replay as they ran. **Red** first (1 row: `split_ffn`'s bits differ),
  then **green**; all of larch-model's tests pass on the 5070 Ti (CUDA backend).

## SMOKE (GO's check on the real model)

B = `larch 0.0.0 (1b29835)`, sha256 `b202456c…`, clean tree. One run, W4@4 `--slice 1/10`, histogram in,
`LARCH_EXPERT_CACHE=1`, `LARCH_PROFILE_STEPS=1 LARCH_GRAPH_DIAGNOSTICS=1` (`smoke-B-w4-4*`):

- **Plan.** The cost first picked 15 layers (9,502 of 10,240 experts on the card); the check after the load found the
  card short (279 against 452 MB) and the loader planned once more with the larger charge it already applies: **20
  layers**, 9,320 experts on the card, `larch_split_decode,mapped=1,layers=20`, 4 state slots. G's 15 assumed the first
  charge; every B run below loads the same 20.
- **Graphs.** Every decode bucket captured in **one piece** (1, 2 and 4 rows; 3 rows replay the 4-row bucket), none
  refused; all 1,975 decode steps replayed, none eager.
- **Steps.** Replayed **4-row step 10.94 ms** (A: 13.8; the gate 12.0), **1-row 5.72 ms** (A: 6.81), 2-row 7.56. Prompt
  chunks beside decode rows run eager: about 180 ms for 3 decode rows and a 512-row chunk (Lane E's 8-layer split: 127;
  A's pooled stream: about 150). Prompt rows' host pairs: 1,895,441 on the card, 1,039 on the host.
- **Result.** 275.9 tok/s, TTFT p50 171 ms and p99 708, TPOT p50 12.60 and p99 16.07 (A on this slice last night: 236.7
  to 239.4 tok/s). Measured against A in M.

## P: checks

- **Code review** (high, `origin/dev...HEAD`): 4 important, 3 nits. Fixed: a mixed step's prompt rows were gathered and
  scattered whenever decode rows led the step (about 6 MB of copies a split layer; `e3ffd4b`, with the test now checking
  the chunk's rows against the resident load); HIP answers UNSUPPORTED for the wrap, as `larch.h` promises it on CUDA
  only. Kept as debt rows below: split layers' decode rows go uncounted in the histogram; a captured step with decode
  and prompt rows on split layers is refused and runs eager; the cost assumes the in-place path where dynamic residency
  or a backend without the wrap pays a round trip; the shortfall replan (pre-existing: last night's split loads replanned
  from 577 to 809 MB of charge). Not changed: the wrap comment in `larch.h` is the change approved for this branch; freeing
  a wrap does not wait for queued reads (the Rust view holds the host buffer; a C caller is warned by the header's
  ordering rule).
- **Over-engineering review**: nothing to cut ("Lean already. Ship.").
- `cargo fmt --all --check`: clean. `cargo clippy --workspace --all-targets --all-features -- -D warnings` (CUDA backend):
  clean after two fixes (`3865686`).
- `cargo test --locked --workspace` (CUDA backend, GPU lock held): 3,178 passed, **1 failed**, 34 ignored. The failure is
  the known `larch-agent` `a_manifest_written_mid_run_makes_the_task_tested` (Windows' `npm` on `PATH`, no `node`).
- `ctest` (CUDA build, RelWithDebInfo): **20 of 20 passed**, none skipped (`ops` 1,347 cases with K1's 12, `wrap_host`).
- **Parity**, m13 on B (`e3ffd4b`, sha256 `582252d6…`), histogram in, `LARCH_EXPERT_CACHE=1`; every mode on a split load
  read in place (`larch_split_decode,mapped=1`; parity's loads plan at their own context and seats):

| mode | split layers | mean KL | p99 KL | top-1 | decode steps replayed / eager | bound | result |
|---|---|---|---|---|---|---|---|
| prefill | 14 | 0.00418 | 0.0185 | 0.9411 | 0 / 1 | 0.0354 | pass |
| decode | 10 | 0.00403 | 0.0170 | 0.9445 | 56,529 / 0 | 0.0354 | pass |
| decode `--parallel 4` | 17 | 0.00403 | 0.0170 | 0.9445 | 14,564 / 0 (96 mixed) | 0.0354 | pass |

  Decode at 1 and 4 seats gives the same KL to five figures: decode rows take the resident load's bits whatever the
  placement (Lane Z's resident-bit decode read 0.00405). `--mode all` runs after the report, as planned.

## M: measurements

Timed runs only, nothing else running, the histogram protocol on every run, `LARCH_PROFILE_STEPS=1
LARCH_GRAPH_DIAGNOSTICS=1` on both sides. A = `ef28629` (`c1ad854a…`), B = `e3ffd4b` (`582252d6…`).

**A B B A**, `--slice 1/10` (every B plan line: 20 split layers after the loader's replan, mapped=1):

| Load | A (A1 / A2) | B (B1 / B2) | tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| W4@4 | 237.3 (237.2 / 237.4) | 289.6 (289.8 / 289.4) | **+22.0 %** | −1.0 %, noise | −8.4 % | −19.0 % | −17.3 % |
| W4@1 | 135.4 (135.6 / 135.3) | 162.6 (163.1 / 162.1) | **+20.0 %** | −7.9 % | −13.9 % | −17.0 % | −17.5 % |

**Full runs on B** (one each; plan line N, the loader replanning from 15 to 20 where its check came out short):

| Run | plan | tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | replayed decode steps (step ms) | eager decode steps |
|---|---|---|---|---|---|---|
| W4@1 | 20, mapped=1 | 161.8 | 117 / 508 | 5.63 / 5.82 | 64,238 1-row (5.60) | 0 |
| W4@4 (r2) | 20, mapped=1 | 298.2 | 134 / 594 | 12.55 / 15.02 | 15,539 4-row (11.02), 136 1-row (5.84) | 0 |
| W4@4 (r1, **void**: planned 15, no replan) | 15, mapped=1 | 298.2 | 130 / 617 | 12.56 / 15.20 | — | — |
| W4@8, 4 seats | 20, mapped=1 | 300.5 | 4,456 / 6,146 | 12.48 / 15.03 | 15,612 4-row (11.00) | 0 |
| W4@8, `--set max_batch_seqs=8` | 29, mapped=1, 8 slots | 335.8 | 137 / 1,343 | 22.78 / 27.09 | 7,540 8-row (19.17), 200 4-row (11.55) | 0 |
| W4@16 | 20, mapped=1 | 300.4 | 12,752 / 16,054 | 12.56 / 14.94 | (4 seats) | 0 |

llama.cpp `--n-cpu-moe 3` straight after B's W4@4: 261.9 (after r1) and 262.4 (after r2), TPOT p99 17.14 and 17.06,
TTFT p99 1,170 and 1,478. Prompt chunks beside decode rows (3 decode rows and a 512-row chunk, eager on B): 133 ms on B
against 151 ms on A (A B B A W4@4, `-steps.txt`); SMOKE's 180 ms was before the review's gather fix.

- **KEEP**: B ahead of A at W4@4 outside noise (+22.0 %), not slower at W4@1 (+20.0 %), TPOT p99 better (−17 %), parity
  passed. **Kept.**
- **TTFT**: better than A at both loads; at W4@1 TTFT p99 is still behind llama.cpp ncm 3 (508 against 415 ms, last
  night's), as A's was (565).
- **The void run.** The protocol checks for the plan N the first B load showed (20); r1's load passed its check at 15 and
  did not replan. It was run again (r2, 20) and the pair with llama.cpp made again; r1 is reported, not counted.

## G against M

| Rows | today (A) | G's prediction (cap 15) | measured on B (cap 20) | G at cap 20 |
|---|---|---|---|---|
| 1 | 6.81 ms | 5.47 | 5.57 to 5.60 | 5.51 |
| 4 | 13.8 | 10.11 | 10.95 to 11.02 | 10.15 |
| 8 | 24.9 | 16.54 | 19.17 (29 layers, 8 seats) | — |

At one row the model is within 2 %; at four rows the measured step is 0.8 ms (8 %) over it: the load hosts 920 experts,
not the 718 G2 assumed (the replan's larger charge), and W4's cold share runs above the histogram's. The predicted gains
(+25 % at one row, +36 % at four) came out as +20 % and +22 % in tok/s (A B B A); prompt and mixed steps are part of it.

## Toward Basalt and Strata on a 16 GB card

This pass is their expert cache, exact for decode: the hot experts on the card, the cold tail in RAM read in place, with
the resident load's bits. On m13 with the shipped profile: 161.8 tok/s at one chat, 298.2 at four, 300.5 at eight on
four seats and 335.8 on eight. What remains, with this pass's numbers:

- **MTP drafting at one chat.** One row runs at 5.60 ms a step (178 steps a second), the card mostly idle; a drafter's
  check steps already keep decode's bits on split layers (K2's check-step test), so drafting adds no new path.
- **A low-bit cold tail.** A cold pair costs about 360 µs, gate and up read at about 3 GB/s (debt row): first a cold-path
  decode that keeps the bus busy, then Q2_K cold copies (a third fewer bytes) once the read is bandwidth-bound.
- **The head-to-head model that fits 32 GB of RAM.** Not run: this pass measured m13 only (920 experts, 1.4 GB, on the
  host). A model whose experts need most of 32 GB puts far more of its routes on the host, where the cold path's
  360 µs a pair decides.

## Next step

The cold-path read (debt row one): `g1.cu` shows a cold Q3_K gate or up matrix at about 3 GB/s over a bus that streams
53. A decode of the cold tensor with its loads in flight together, measured on `g1.cu` and the A B B A above, is the
lever both for larger caps and for the 32 GB model.

## Debt

Each row has a marker in the code (`ponytail:`) or is named here.

| Row | What | Numbers | When to revisit |
|---|---|---|---|
| Cold gate/up read slowly in place | `mmvq_id_mixed` reads a cold Q3_K gate or up expert over the bus at about 3 GB/s; cold Q4_K down at about 24 GB/s (G1). A cold pair costs 359.9, 369.4 and 307.1 µs a step at 1, 4 and 8 rows; gate+up is 336 of the 360 µs at one row. Not chased this run. | G1 table above; the bus does 53 GB/s for a streaming read (Lane Z's Z0) | Before a larger cap or a low-bit cold tail: a cold-path decode with loads in flight together (`id_dot`'s unroll, a wider read per lane), measured with `g1.cu` |
| Split layers' decode rows uncounted | Decode and check rows of split layers run on the device with no download, eager or replayed, so the histogram and `larch_expert_cache_hits` count only their prompt rows while resident layers count every eager row: the saved file makes split layers look colder. | the review's first finding | Before the default flip or before a load plans from a file it wrote itself |
| Fused gate_up/down over a mixed layer | Split layers run unfused. | 11.3, 18.1, 23.1 µs a split layer at 1, 4, 8 rows (`SPLIT_LAYER_US`) | If a profile shows split layers |
| Mixed steps eager | A step with decode and prompt rows on split layers is refused for capture and runs eagerly (its host section has no tag for rows past 0). | about 180 ms for 3 decode rows and a 512-row chunk on SMOKE's 20-layer split | If TTFT matters at many chats: a host tag that carries the row offset |
| Cost assumes the in-place path | `cheapest_split` charges no round trip; dynamic residency (`LARCH_EXPERT_CACHE=2`) and a backend without the wrap still pay one a split layer (`split_ffn`). Fewer layers than the old uncapped plan's 37. | Lane E: 8 round trips cost more than 1.7 ms at one row | When dynamic residency or another backend is used: a round-trip cost a split layer there |
| The cap under the first charge | The cost picks the cap under the planned charge; the check after the load is short (pre-existing, about 173 MB on m13) and the loader plans again with a larger charge, so 15 becomes 20. | SMOKE plan lines | Charge what the check finds missing before the first plan |
| Adaptive swaps | Dynamic residency keeps `split_ffn` for the whole layer; no table update between replays. | — | After the default flip |
| Low-bit cold experts | Cold experts keep their file dtype; a Q2_K cold copy would read a third fewer bytes. | — | With the cold-path decode row above |
| Hit counters for replayed split steps | (the brief's row) replayed steps of split layers count nothing; no download was added to keep them. | — | With the uncounted-rows row |
| The histogram at install | `bench/routes/m13.experts` is copied by hand; nothing ships it beside the model. | — | Before the default flip |
| The default flip (K4) | Only `LARCH_EXPERT_CACHE=1` takes the cache. A default split needs a measured win at 1, 4 and 8 chats, no recording of its own, and `replan_due` reloading only where seats change the plan. | M below | After the shipped profile |
| Prompt rows depend on placement | Decode and check rows give the resident load's bits whatever the placement; prompt rows of split layers still take `split_ffn`'s sums (host part in f32). | parity prefill mean KL 0.00418 | With a prompt path that keeps the resident bits |
| HIP | The wrap answers UNSUPPORTED on HIP until a HIP run checks it. | — | On HIP hardware |

## Method

- **Machine and order.** One session. Every GPU job (runs, ctest, benches, GPU tests, parity) is its own process in one
  hold of the GPU lock, with used VRAM back under 2.5 GB and steady for 5 s before it. Builds run under `nice -n 19`,
  and no build, test or review ran during a timed run.
- **llama.cpp.** b12723 through `llama-serve-bench` (the 27 Nov build), `-ngl 99 -fa on -ctk f16 -ctv f16 -t 8 -b 2048
  -ub 1024 -c <chats × 2560> --n-cpu-moe <n> -lv 4`. Its free-memory and CUDA0 buffer lines sit beside each result
  (`-load.txt`); a load probe is a short `fixed` run whose lines go to `wsl5070-m13-llama-probes.txt`.
- **Larch.** `larch bench --mode serve --set max_context=2560`, `LARCH_FIT_DIAGNOSTICS=1`; fit lines beside each result
  (`-fit.txt`). W4: 200 prompts, exact output lengths; `--slice K/10` is 20 of them.
- **Histogram protocol.** A and llama.cpp: any `<gguf>.experts` beside the model moved aside, `LARCH_EXPERT_CACHE`
  unset. B: `bench/routes/m13.experts` copied beside the model and `LARCH_EXPERT_CACHE=1` before every load (a split
  load rewrites the file; it is moved aside after). A run is void unless its fit lines show the plan its side needs:
  A `larch_host_experts_plan,4`; B `larch_expert_cache_plan,<N>` and `larch_split_decode,mapped=1`.
- **Statistics.** Each cell is the mean of its runs, spread = (max − min) / mean. A margin is noise when under 2 % or
  under either side's spread; a one-run cell has only the 2 % rule (the 26 Nov rule).

## Builds

| Build | sha256 | Commit | What it is |
|---|---|---|---|
| `larch 0.0.0 (ef28629)` | `c1ad854a…` | `ef28629` (#20's branch head, before its review fix) | A: rebuilt today, as the night's `b3a53bf3…` binary was gone; the same commit, not the same bytes |
| `larch 0.0.0 (1b29835)` | `b202456c…` | `1b29835`, this branch, clean tree | B: SMOKE |
| `larch 0.0.0 (e3ffd4b)` | `582252d6…` | `e3ffd4b`, this branch, clean tree | B: parity and M (the review fixes) |
| llama-serve-bench `3726f061…` | — | `9c4753d` against llama.cpp b12723 | the 27 Nov build, unchanged |
