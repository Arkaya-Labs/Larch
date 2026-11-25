# Tuned profiles milestone 0 · RTX 3050 Laptop 4 GB (sm_86, WSL2)

Does per-machine calibration of `prefill_chunk_tokens` / `schedule_policy` beat the fixed rule "always set
`prefill_chunk_tokens` to 256"? Design: *Larch Tuned Profiles Architecture* v4 (12 Nov 2026), milestone 0 only.
Raw per-block data: `tuned-profiles-m0-sm86/blocks.csv` (one row per block, 452 blocks) with the three workload files
the script wrote. The per-block bench JSONs (about 100 KB each) stay outside git; every value used here is in the CSV.

STATUS: complete for the one (machine, model) pair this laptop can run. Everything below was measured on 13 Nov 2026.
To run the same experiment on another machine and to read the result against the gate: `bench/tuned-profiles-m0.md`.

## Summary

| arm | four-chat stall cut | document stall cut | no-harm tests | verdict on this pair |
|---|---|---|---|---|
| Always-256 | 406 ms [404, 409], 41 % | 632 ms [629, 636], 44 % | **fails four-chat TTFT p50: +12.4 %, bound 13.6 % > 10 %** | fails no-harm |
| Calibrated (384 + `balanced`), confirmation | 141 ms [127, 152], 14 % (under the 20 % bar) | 321 ms [319, 327], 22 % | all pass | confirmed on doc, no harm anywhere |

On this pair the calibrated arm avoids a no-harm failure that always-256 commits and regresses on nothing, which is
one of the gate's two conditions. It also gives up most of the stall cut: a third of always-256's on four chats and
half on the document. One pair out of the nine the gate wants; the gate is not settled (see Recommendation).

## Setup

- Build: branch `a1/tuned-profiles-m0` at `dev` 08817a10 (5 commits past the reviewed 5986bee, none in bench,
  engine scheduling or kernels), release, CUDA sm_86, `--no-default-features` (no gait or strided linked; nothing
  here tunes live).
  `cargo build --release --locked -p larch --no-default-features` with `LARCH_BACKEND=CUDA LARCH_CUDA_ARCH=86`.
- Driver 610.91, CUDA 13.1 (V13.1.115), WSL2 kernel 6.18.58.1.
- GPU: RTX 3050 Laptop 4 GB, enforced power limit 40 W (default limit 35 W: Dynamic Boost raises it; it can't be
  changed without admin rights, so "default power limit" is recorded as the 40 W the driver enforces), AC power.
- Models: catalog = m1, m3, m5 (`bench/models.toml`, `use = "catalog"`). Only m3 fits 4 GB; m1 and m5 need
  `min_mem_gb` 7.

## Workload types

| type | brief | bench | why |
|---|---|---|---|
| short | one short chat | `--workload chat --concurrency 1 --prompts short.jsonl` | W1's first 6 turns (870 generated tokens), about 13 s |
| long | one long-context chat | `--workload agent --concurrency 1 --prompts long.jsonl` | W3's first loop, first 10 turns on its 4096-token prefix (1200 tokens), about 36 s |
| conc | four concurrent chats, prompts arriving during generation | `--workload chats --concurrency 4` (W2) | closed loop: each client's next turn arrives while the other three generate |
| doc | one long document arriving during generation | `--workload chats --concurrency 2 --prompts doc.jsonl` | built by the script from frozen prompts: W2 c0, and a conversation whose 32-token first turn (W2 c1's) is followed by W5's 4096-token document, so the document lands while c0 is generating |

The short and long blocks are trimmed because the stock ones (W1 whole, about 39 s; W3's first loop whole, about
71 s) make a 9-pair confirmation of all four types take about 67 min on this card, over the brief's 1-hour stop.
Trimmed, it takes about 47 min. Those two types only feed the no-harm checks. conc and doc stay whole: conc's
worst stall came 40 s into the block in the smoke run. The trimmed and composed files are written by the script
from the frozen prompts (`write_workloads`) and copied beside the data.

## Block properties

| property | enforced? | how |
|---|---|---|
| pre-tokenised prompts | yes | bench renders and tokenises before the clock |
| fixed output lengths | yes | scripted replay, EOS ignored: every answer is exactly `max_tokens` |
| fixed seed / greedy | yes | bench requests are greedy |
| fixed concurrency | yes | closed-loop clients, one value per block |
| cache cleared before each block | yes | one process per block: the engine and its prefix cache start empty (costs a model load per block) |
| discarded warm-up prefix | partly | bench's own warm-up is one 16-token request with 2 answer tokens; no longer prefill or batched decode runs before the clock (gap below) |
| identical work across arms | yes, checked | `gen_tokens` and `evaluated_tokens` per block are compared across arms in the data |

## Metrics used

- **Stall (primary):** `summary.token_gap_ms.max`, the longest receive gap between two tokens of any request in
  the block, measured by the client. It isn't gait's `ItlMax` (the longest decode-bearing engine step per window):
  bench's step profile sums durations per batch shape and keeps no per-step maximum. As a cross-check the data also
  holds `step_max_ms`, the slowest *mean* step among decode-bearing shapes.
- Throughput: `summary.agg_tok_s`. TTFT: `summary.ttft_ms.p50`. Single-chat token time: `summary.token_gap_ms.p50`.
- Failures: `summary.failures`, a non-zero exit, a timeout (900 s), or a fault (exit 3, which stops the run).

## Statistics (scripts/tuned_profiles_m0.py)

- Gain: exact Wilcoxon signed-rank (sign flips of the ranks) on std − candidate stall, zeros dropped, midranks for
  ties; looks at 9 pairs (p < 0.005) and 18 pairs (p < 0.045). Estimate: Hodges-Lehmann, bounds by inverting the
  same test (order statistics of the Walsh averages) at the look's alpha. Confirmed needs p under the look's alpha
  **and** the lower bound ≥ 100 ms and ≥ 20 % of the standard arm's median stall.
- No harm: the same inversion on the paired log ratios at one-sided 5 %, checked once at the last look.
- A candidate is confirmed on a type when the gain test passes there (concurrent types only), no no-harm bound
  fails on that type and the arm had no failure.
- Tests: `python3 -m unittest scripts/test_tuned_profiles_m0.py` (55 tests: brute-force enumeration of the exact
  distribution and of `w_crit`, bound inversion and coverage, sequential level by simulation, A/A never confirms,
  and the runner end to end against fake `nvidia-smi`, `powershell.exe` and `larch`: command lines, random but
  reproducible order, resume, hot start / battery / other-process voids repeated once, fault exit, failures and
  timeouts counted against the arm, analysis verdicts, single-look confirmation; for other machines: a host
  without nvidia-smi, AC detection on macOS and Linux, the GPU lock and `--wait`, several models in one CSV, and
  the hand-calibration and gate summary).

## Assumptions (no owner was asked; each is a choice the owners can change)

1. Primary metric = client-side max token gap (above). Gait's ItlMax isn't in bench's output.
2. "Passes every test" for a candidate = confirmed on at least one concurrent type and not harmful on any type.
3. Start-temperature band: at or below 70 °C, waiting up to 300 s for it, plus 5 s idle before every block. No lower bound:
   the GPU idles at 45-55 °C.
4. AC power read from Windows (`Win32_Battery.BatteryStatus` = 2); WSL has no power_supply class.
5. Another compute process at a block's start stops the runner (exit 3) and gives the GPU back instead of voiding
   pairs one after another; one appearing during a block voids the pair.
6. A compile or test running on the host (cargo, rustc, nvcc, ptxas, cicc, cc1plus) counts the same way. In the
   smoke run, other sessions' builds alone moved the short chat from 79 to 64 tok/s. Only the start and the end of
   a block are checked, so a compile that starts and ends inside one block goes unseen.
7. Short and long blocks are trimmed (above): a choice about block length, not a change to the stop rule. The
   untrimmed figure is given so the owners can judge.
8. A/A false-confirm rate: every 9-pair sliding window of the A/A pairs, both labellings, through the gain and
   no-harm tests.
9. One pair was voided by hand. WSL shut down at 11:55 during the confirmation, right after short pair 3 (attempt
   0) ran at 65 and 76 tok/s against about 80 in every other short block. The runner's checks (compile, other GPU
   process, temperature, power) can't see a host shutdown, so the pair was marked void with that reason in
   `blocks.csv` and repeated once, as section 5 asks. No other row was edited.
10. A pair lost after its one repeat leaves the type short of 9 pairs. In the confirmation, conc pair 2 was voided
    on both attempts (another session's compile started during the block), so one more conc pair (index 9) was
    run to reach the 9 pairs the brief asks for. Pairs are analysed in run order; no pair was dropped by its value.

## Results

### Noise (A/A), measured 13 Nov 2026

10 A/A pairs per type on m3 (standard against an identical second arm, `std2`, random order per pair), 04:26 to
07:07 in six GPU holds: the runner gave the GPU back five times when another session's compile was running at a
block's start. Five blocks were voided because another session's compile started during them and
were repeated (one short pair was lost on both attempts, so short has 9 pairs). Start temperature 40-60 °C, peak
56-69 °C, median clock 1380-1519 MHz, AC throughout.

| type | stall (max gap) median, ms | A/A sd of log ratio | largest A/A difference | tok/s A/A sd |
|---|---|---|---|---|
| conc | 996 | 0.2 % | 0.3 % | 0.1 % |
| doc | 1448 | 0.6 % | 1.8 % | 0.1 % |
| long | 17 | 9.8 % | 26.5 % | 0.2 % |
| short | 12 | 12.8 % | 20.0 % | 0.3 % |

- The stall on the concurrent types is almost deterministic: about 0.2-0.6 % sd, so a 100 ms (7-10 %) effect is
  15-50 sd. The single-chat max gap is noisy in relative terms (it is a 12-17 ms number), which doesn't matter: the
  single types are judged on throughput, TTFT and median gap, whose A/A sd is 0.2-0.6 %.
- TTFT on conc is the noisiest no-harm metric (sd 2.7 %, largest 7 %); its 10 % margin still holds with room.
- False confirms: 0 of 14 sliding 9-pair windows (both labellings) passed the gain and no-harm tests. All A/A
  no-harm bounds pass (tok/s lower bounds ≥ 0.997, TTFT and median-gap upper bounds ≤ 1.028).

### Order

No order effect of practical size. Of 12 permutation tests (4 types × stall, tok/s, TTFT), one had p < 0.05: long
tok/s, p = 0.017, with means of -0.2 % and +0.1 % by order, a 0.3 % difference that one test in twelve would show by
chance. Random order per pair stays in place for the comparisons.

### Hand calibration: chunk candidates, measured 13 Nov 2026

9 pairs per type for each candidate against standard (chunk 512, decode_first), in the brief's order, one GPU hold
each: 256 07:07-07:55, 128 07:55-08:43, 384 08:43-09:31. No voided blocks, start 52-55 °C, peak 59-63 °C, median
clock 1380-1519 MHz, AC. Every block of every arm generated and evaluated the same tokens (short 870 / 1934, long
1200 / 6601, conc 7171 / 12476, doc 2192 / 7960), with no failures, preemptions or non-zero exits.

Stall = longest token gap in the block, ms, medians over pairs; gain = Hodges-Lehmann of standard − candidate with
its 99.5 % interval (look 1). Ratios are candidate / standard, point estimate and the one-sided 95 % bound.

| arm | type | stall std → cand | gain [bounds] | gain test | tok/s (≥ 0.97) | TTFT p50 (≤ 1.10) | median gap (≤ 1.03) | verdict |
|---|---|---|---|---|---|---|---|---|
| 256 | conc | 992 → 587 | 406 [404, 409], 41 % | passed, p = 0.002 | 0.985, 0.985 | 1.124, **1.136 fail** | n/a | rejected |
| 256 | doc | 1443 → 811 | 632 [629, 636], 44 % | passed, p = 0.002 | 0.988, 0.987 | 1.056, 1.060 | n/a | confirmed |
| 256 | short | 12 → 12 | n/a | n/a | 0.991, 0.991 | 1.047, 1.051 | 0.999, 1.001 | no harm |
| 256 | long | 17 → 17 | n/a | n/a | 0.985, 0.984 | 1.045, 1.048 | 0.995, 0.996 | no harm |
| 128 | conc | 994 → 334 | 660 [658, 662], 66 % | passed, p = 0.002 | 0.961, **0.960 fail** | 1.288, **1.306 fail** | n/a | rejected |
| 128 | doc | 1450 → 430 | 1017 [1013, 1020], 70 % | passed, p = 0.002 | 0.968, **0.968 fail** | 1.128, **1.131 fail** | n/a | rejected |
| 128 | short | 12 → 12 | n/a | n/a | 0.975, 0.974 | 1.205, **1.210 fail** | 1.000, 1.002 | rejected |
| 128 | long | 18 → 17 | n/a | n/a | 0.965, **0.964 fail** | 1.078, 1.082 | 0.996, 0.998 | rejected |
| 384 | conc | 994 → 871 | 124 [123, 125], 12.5 % | p = 0.002, under 20 % | 0.994, 0.993 | 1.083, 1.093 | n/a | not confirmed |
| 384 | doc | 1449 → 1144 | 304 [298, 309], 21 % | passed, p = 0.002 | 0.998, 0.997 | 1.000, 1.002 | n/a | confirmed |
| 384 | short | 12 → 12 | n/a | n/a | 0.998, 0.997 | 1.076, 1.078 | 0.999, 1.001 | no harm |
| 384 | long | 18 → 18 | n/a | n/a | 0.999, 0.999 | 1.001, 1.002 | 1.000, 1.001 | no harm |

- Every candidate's stall cut is far outside the noise (all 9 pairs in the same direction, p = 1/512, the smallest a 9-pair
  exact test can give). The question is only the bars.
- **Always-256 fails a no-harm bar on this pair:** four-chat TTFT p50 rises 12.4 % (bound 13.6 %, margin 10 %).
  Every one of its 9 conc pairs was above +10 %. It passes everything else and cuts the stall the most of any
  arm that is close to passing.
- 128 cuts the stall most but fails throughput or TTFT on every type.
- 384 passes every no-harm bar on every type and is confirmed on doc (lower bound 298 ms against the 290 ms
  that 20 % of 1449 asks). On conc its gain is certain but small: 124 ms, 12.5 %, under the 20 % bar. The analysis
  prints "pending" there because look 2 (18 pairs) wasn't run; it wasn't, because the A/A sd of 0.2 % leaves no
  way for 9 more pairs to move a 123 ms lower bound to the 199 ms the bar needs.
- Best chunk (largest stall cut among the chunk candidates that pass every test, assumption 2): **384**.
- Order: of 36 order tests on the candidate pairs, 4 had p < 0.05 (c128 short TTFT p = 0.007, c128 conc TTFT
  p = 0.036, c256 long tok/s p = 0.044, c384 conc tok/s p = 0.025), each a difference of 0.1-2.2 points between
  orders on effects that are the same sign and nearly the same size either way. About 2 of 36 are expected by
  chance; none changes a verdict.

### Budget

Medians per block on this machine (all 301 measured blocks):

| type | cool-down + idle | process | run (`wall_s`) | load + warm-up + teardown |
|---|---|---|---|---|
| short | 5.1 s | 12.2 s | 10.7 s | 1.5 s |
| long | 5.1 s | 34.8 s | 33.3 s | 1.5 s |
| conc | 5.1 s | 50.3 s | 48.8 s | 1.5 s |
| doc | 5.1 s | 36.8 s | 35.4 s | 1.5 s |

- Load and warm-up can't be split: bench records neither (gap 3); together with teardown they are 1.5 s for this
  0.8 GB model with its file in the page cache. The temperature band never made the runner wait past the 5 s idle.
- One pair of all four types: 2 × 134 s + 8 × 5.1 s ≈ 309 s. A full confirmation (9 pairs, single look) is about
  46 min of GPU time; the measured stages each took 47-48 min wall clock, under the 1-hour stop.
- A real calibration by the hand procedure = 4 candidates + 1 confirmation = 5 stages ≈ 4 hours of exclusive GPU
  on this card for one model, plus the A/A stage (another 50 min) if noise must be measured per machine. Screening
  candidates on the two concurrent types only would cut a candidate to about 29 min (≈ 3 hours in all).

### Hand calibration: schedule_policy at the best chunk, measured 13 Nov 2026

One alternative policy at chunk 384, the best chunk that passed every test: `balanced` (the default is
decode_first). 9 pairs, 09:31-10:19, no voids, start 46-55 °C, peak 57-62 °C, median clock 1395-1508 MHz, AC, same
tokens as every other arm, no failures or preemptions. Both settings were applied at startup in every candidate
block (the bench JSON's `knob_timeline` shows `prefill_chunk_tokens` 384 and `schedule_policy` 1).

| arm | type | stall std → cand | gain [bounds] | gain test | tok/s (≥ 0.97) | TTFT p50 (≤ 1.10) | median gap (≤ 1.03) | verdict |
|---|---|---|---|---|---|---|---|---|
| 384 + balanced | conc | 994 → 856 | 139 [138, 141], 14 % | p = 0.002, under 20 % | 0.998, 0.997 | 1.048, 1.059 | n/a | not confirmed |
| 384 + balanced | doc | 1452 → 1133 | 319 [317, 320], 22 % | passed, p = 0.002 | 1.001, 1.000 | 0.999, 1.002 | n/a | confirmed |
| 384 + balanced | short | 12 → 13 | n/a | n/a | 0.997, 0.996 | 1.076, 1.081 | 1.000, 1.002 | no harm |
| 384 + balanced | long | 17 → 18 | n/a | n/a | 0.998, 0.998 | 1.001, 1.002 | 1.000, 1.001 | no harm |

- Against decode_first at the same chunk, `balanced` cut the stall slightly more (conc 139 against 124 ms, doc 319
  against 304 ms; the intervals don't overlap) and cost less: conc TTFT +4.8 % against +8.3 %, throughput 0.998
  against 0.994. The two were measured in different stages against their own standard blocks, so this is a
  between-stage comparison, not a paired one; the standard arm's stall moved by under 1 % between stages.
- Order: 2 of 12 order tests under 0.05 (long tok/s p = 0.023 and TTFT p = 0.042, each a 0.1-0.4 point difference).
- **Winner of the hand calibration: chunk 384 + `balanced`** (largest stall cut among the candidates that pass every
  test: 256 + decode_first fails conc TTFT, 128 fails throughput and TTFT).

### Confirmation of the winner

9 fresh pairs per type against standard, single look (p < 0.005 at 9 pairs, no second look), with the winner applied
at process start: `--set prefill_chunk_tokens=384 --set schedule_policy=balanced`. Comparison `std-vs-cal`, analysed
with `analyze --single-look std-vs-cal`. Ran 10:21-11:55 and 17:35-18:10: other sessions' compiles kept the runner
off the GPU in between, and WSL restarted at 11:55 (assumptions 9 and 10). 77 blocks, 5 voided (3 by compiles, 2 by
the restart), conc pair 2 lost on both attempts and replaced by pair 9. Start 49-69 °C, peak 59-79 °C, median clock
1312-1522 MHz, AC, 40 W; same tokens as every other arm, no failures, preemptions or non-zero exits.

| type | stall std → cand | gain [bounds] | gain test | tok/s (≥ 0.97) | TTFT p50 (≤ 1.10) | median gap (≤ 1.03) | verdict |
|---|---|---|---|---|---|---|---|
| conc | 1003 → 862 | 141 [127, 152], 14 % | p = 0.002, under 20 %: failed | 0.998, 0.997 | 1.002, 1.036 | n/a | not confirmed |
| doc | 1455 → 1135 | 321 [319, 327], 22 % | passed, p = 0.002 | 1.001, 0.999 | 0.995, 0.997 | n/a | **confirmed** |
| short | 12 → 12 | n/a | n/a | 0.995, 0.992 | 1.083, 1.092 | 1.002, 1.005 | no harm |
| long | 18 → 18 | n/a | n/a | 0.998, 0.995 | 0.998, 1.001 | 0.998, 1.004 | no harm |

- The confirmation reproduces the screening stage: conc 141 against 139 ms, doc 321 against 319 ms.
- The first two pairs ran on a hot, busy morning machine (start 66-69 °C, clocks down to 1312 MHz): both arms' stalls
  were higher (standard up to 1049 ms, candidate 888-890 ms) and the conc interval is wider than in the screening
  stage. Pairing absorbed it; the later pairs sit at 1000-1006 / 857-863 ms.
- Short-chat TTFT p50 is up 8.3 % (302 → 326 ms, bound 9.2 %), inside the 10 % margin but the closest call; every
  candidate at 384 or below showed it (+5 to +21 %).
- Order: 2 of 12 order tests under 0.05 (long stall p = 0.021, a 12 ms metric with 10 % A/A noise; doc tok/s
  p = 0.017, a 0.5-point difference). None changes a verdict.

### The comparison (section 4, item 4)

Standard = registry defaults (chunk 512, decode_first). Always-256 numbers are from the chunk-256 stage, calibrated
numbers from the confirmation; each is paired against its own standard blocks.

| arm | type | stall cut [99.5 % interval] | tok/s ratio, 95 % lower | TTFT p50 ratio, 95 % upper | median gap ratio, 95 % upper |
|---|---|---|---|---|---|
| Always-256 | conc | 406 ms [404, 409], 41 % | 0.985, 0.985 | 1.124, **1.136** (442 → 499 ms) | n/a |
| Always-256 | doc | 632 ms [629, 636], 44 % | 0.988, 0.987 | 1.056, 1.060 | n/a |
| Always-256 | short | n/a | 0.991, 0.991 | 1.047, 1.051 | 0.999, 1.001 |
| Always-256 | long | n/a | 0.985, 0.984 | 1.045, 1.048 | 0.995, 0.996 |
| Calibrated | conc | 141 ms [127, 152], 14 % | 0.998, 0.997 | 1.002, 1.036 | n/a |
| Calibrated | doc | 321 ms [319, 327], 22 % | 1.001, 0.999 | 0.995, 0.997 | n/a |
| Calibrated | short | n/a | 0.995, 0.992 | 1.083, 1.092 | 1.002, 1.005 |
| Calibrated | long | n/a | 0.998, 0.995 | 0.998, 1.001 | 0.998, 1.004 |

Failures: none in any arm (no failed request, fault, guardrail trip or timeout in 452 blocks).

The October reviews' 606 ms stall cut for 256 reproduces here as 632 ms on the document type and 406 ms on four
chats. Their unsettled throughput effect (+5.2 %, p ≈ 0.07, three whole-run pairs) doesn't: on interleaved blocks
256 costs 1.2-1.5 % tok/s on every type here, within the 3 % margin.

## Recommendation (section 7)

**What this pair shows.** Against always-256, the calibrated arm avoids a no-harm failure that always-256 commits
(four-chat TTFT p50 +12.4 %, bound 13.6 %, margin 10 %) and regresses beyond the margins on no type. By the gate's
wording this pair counts toward **build calibration**. It gives *less* stall reduction, not more: 141 against
406 ms on four chats, 321 against 632 ms on the document.

**What it can't show.** The gate needs the condition on at least 30 % of (machine, model) pairs. This is 1 pair of
the 9 wanted (3 machine classes × 3 catalog models); m1 and m5 don't fit 4 GB, and there is no 12-16 GB desktop or
Apple silicon run. The gate is not settled, and nothing here says anything about other hardware.

**Recommendation to the owners: don't start building calibration on this evidence.** Reasons, all from this pair:

1. Always-256's failure is one margin missed by 2.4 points (+57 ms median TTFT with four chats running), while it
   cuts the worst stall by 406-632 ms. Whether that is a failure worth a calibration system depends on the 10 %
   margin, which the brief marks as proposed.
2. The winner is a milder *fixed* setting (384 + `balanced`), not something only a per-machine search could find.
   If one fixed rule passes no-harm on the other machine classes, a calibration system buys nothing over it. The
   gate compares calibration only with always-256; the cheaper question is whether some fixed rule is safe
   everywhere.
3. Calibration by the hand procedure cost about 4 hours of exclusive GPU on this card for one 1B model (Budget).
   Sarathi-Serve's one-time profiling (largest chunk whose step stays under an ITL target) would be a few seconds at
   load; it isn't measured here.

**For this pair, by the brief's fallback:** always-256 fails a no-harm test, so "ship nothing" for always-256 on the
3050 with m3; 384 + `balanced` passes every test here if the owners want a safe fixed setting for this pair.

**Still missing before the gate can be decided:** the same runs on a 12-16 GB desktop and on Apple silicon with
three catalog models each, and m1/m5 on any machine that fits them; on each, always-256, the calibrated winner, and
(suggested) 384 + `balanced` as a candidate fixed rule.

## Not run, and why

- m1 and m5: need 7 GB (`min_mem_gb`); this card has 4 GB.
- Other machine classes: one machine available.
- Look 2 (18 pairs) for any candidate: every candidate's look-1 result was decisive either way (p = 1/512 on every
  concurrent type; the conc misses for 384 and 384 + `balanced` are on the 20 % bar, with intervals at 12-15 %
  that 9 more pairs at 0.2 % A/A noise can't move).
- `prefill_first`: the brief allows one alternative policy; `balanced` was picked because it sits between the
  default and prefill_first.
- Untrimmed W1/W3 blocks (assumption 7).

## Commands run

Every measured block was run by `scripts/tuned_profiles_m0.py run` (one fresh process per block) through the queue
below; the bench command per block was:

```
larch bench -m m3.gguf --workload chat   --concurrency 1 --machine sm86 --json <block>.json --prompts short.jsonl [--set k=v ...]
larch bench -m m3.gguf --workload agent  --concurrency 1 --machine sm86 --json <block>.json --prompts long.jsonl  [--set k=v ...]
larch bench -m m3.gguf --workload chats  --concurrency 4 --machine sm86 --json <block>.json                       [--set k=v ...]
larch bench -m m3.gguf --workload chats  --concurrency 2 --machine sm86 --json <block>.json --prompts doc.jsonl   [--set k=v ...]
```

Stages, each `flock ~/src/gpu.lock python3 scripts/tuned_profiles_m0.py run --bin target/release/larch --model
m3.gguf --out <dir> --hold-min 50 --b <arm> --pairs <n> [--types <t>]`:

| stage | `--b` | `--pairs` | when |
|---|---|---|---|
| A/A | `std2` | 10 | 04:26-07:07 |
| chunk 256 | `c256:prefill_chunk_tokens=256` | 9 | 07:07-07:55 |
| chunk 128 | `c128:prefill_chunk_tokens=128` | 9 | 07:55-08:43 |
| chunk 384 | `c384:prefill_chunk_tokens=384` | 9 | 08:43-09:31 |
| policy | `c384bal:prefill_chunk_tokens=384,schedule_policy=balanced` | 9 | 09:31-10:19 |
| confirmation | `cal:prefill_chunk_tokens=384,schedule_policy=balanced` | 9, then 10 with `--types conc` | 10:21-11:55, 17:35-18:10 |

Analysis: `python3 scripts/tuned_profiles_m0.py analyze --single-look std-vs-cal bench/results/tuned-profiles-m0-sm86/blocks.csv`.
Tests: `python3 -m unittest scripts/test_tuned_profiles_m0.py` (55 tests at the PR head; 44 when these runs were analysed).

## Per-pair primary metric

Longest token gap per block, ms, for the concurrent types (all other columns are in the CSV).

| comparison | type | pair: standard / candidate stall, ms (pairs in run order) |
|---|---|---|
| std-vs-std2 | conc | 0: 998 / 996, 1: 991 / 992, 2: 996 / 998, 3: 998 / 996, 4: 996 / 997, 5: 1000 / 997, 6: 999 / 998, 7: 996 / 999, 8: 994 / 994, 9: 994 / 993 |
| std-vs-std2 | doc | 0: 1450 / 1451, 1: 1436 / 1462, 2: 1447 / 1446, 3: 1447 / 1449, 4: 1450 / 1449, 5: 1451 / 1454, 6: 1448 / 1449, 7: 1453 / 1450, 8: 1443 / 1440, 9: 1442 / 1442 |
| std-vs-c256 | conc | 0: 991 / 587, 1: 992 / 588, 2: 992 / 586, 3: 992 / 586, 4: 991 / 587, 5: 997 / 587, 6: 995 / 587, 7: 993 / 587, 8: 994 / 588 |
| std-vs-c256 | doc | 0: 1441 / 812, 1: 1442 / 812, 2: 1446 / 809, 3: 1443 / 809, 4: 1444 / 810, 5: 1442 / 811, 6: 1443 / 812, 7: 1444 / 812, 8: 1442 / 810 |
| std-vs-c128 | conc | 0: 990 / 333, 1: 993 / 334, 2: 995 / 333, 3: 994 / 334, 4: 993 / 334, 5: 995 / 335, 6: 992 / 335, 7: 994 / 333, 8: 994 / 333 |
| std-vs-c128 | doc | 0: 1442 / 429, 1: 1450 / 430, 2: 1443 / 428, 3: 1443 / 430, 4: 1447 / 434, 5: 1450 / 430, 6: 1450 / 430, 7: 1452 / 435, 8: 1452 / 433 |
| std-vs-c384 | conc | 0: 994 / 869, 1: 995 / 872, 2: 995 / 871, 3: 993 / 871, 4: 994 / 871, 5: 997 / 873, 6: 994 / 869, 7: 994 / 870, 8: 996 / 870 |
| std-vs-c384 | doc | 0: 1450 / 1147, 1: 1453 / 1145, 2: 1446 / 1144, 3: 1451 / 1143, 4: 1447 / 1144, 5: 1448 / 1143, 6: 1449 / 1139, 7: 1441 / 1146, 8: 1453 / 1149 |
| std-vs-c384bal | conc | 0: 994 / 856, 1: 992 / 854, 2: 993 / 854, 3: 994 / 854, 4: 996 / 855, 5: 996 / 856, 6: 994 / 856, 7: 995 / 856, 8: 996 / 856 |
| std-vs-c384bal | doc | 0: 1454 / 1133, 1: 1448 / 1132, 2: 1452 / 1133, 3: 1451 / 1134, 4: 1454 / 1134, 5: 1453 / 1134, 6: 1453 / 1133, 7: 1451 / 1132, 8: 1452 / 1134 |
| std-vs-cal | conc | 0: 1003 / 888, 1: 1049 / 890, 3: 1006 / 862, 4: 1002 / 862, 5: 1001 / 862, 6: 1004 / 862, 7: 1002 / 857, 8: 1000 / 862, 9: 1004 / 863 |
| std-vs-cal | doc | 0: 1498 / 1175, 1: 1505 / 1185, 2: 1459 / 1129, 3: 1453 / 1133, 4: 1455 / 1136, 5: 1460 / 1136, 6: 1454 / 1135, 7: 1451 / 1133, 8: 1452 / 1133 |

## Bench-driver gaps found

1. **No per-step maximum.** The step profile keeps only sums per shape, so gait's ItlMax can't be read from the
   result. Smallest change: a `max_dur_ns` beside `dur_ns` in `StepTimes` (`bench.rs`), one `max()` in `take`.
2. **Weak warm-up.** `warm_up` sends one 16-token, 2-answer request. The first long prefill chunk and the first
   batched decode run inside the timed block. Smallest change: a `--warmup-tokens N` flag passed to the existing
   `warm_up` (it already clamps to the context).
3. **No model-load time.** A per-block budget can only take process time minus `wall_s`. Smallest change: record
   the load's duration in the result (the HTTP engines already have `http_load_ms`).
4. **No cache reset in-process.** A fresh process per block is the only way to start with an empty prefix cache, so
   each block pays a model load. Smallest change: a facade call that drops parked sequences between `--concurrency`
   runs.

## Other startup tunes from the literature (research, not measured here)

Literature review done for this milestone: a research agent read the sources, and its report marks which claims
it checked in each one. It's a reading list for later milestones, not evidence for this one.

| setting | what the sources say | larch knob |
|---|---|---|
| prefill token budget | Sarathi-Serve: small budgets cut decode stalls (TBT) but cost prefill efficiency (up to ~25 % at 512 on a 34B); tile quantisation makes 257 32 % slower than 256, so keep multiples of 64/128. vLLM: > 8192 for throughput on large GPUs. TetriInfer: the right chunk is the accelerator's saturation point, smaller on weaker GPUs. | `prefill_chunk_tokens` |
| decode-first scheduling | Sarathi's stall-free batching protects ITL; prefill-first protects TTFT. | `schedule_policy` |
| preemption headroom | SGLang: lower its admission conservativeness if KV use stays under 0.9 while requests queue, raise it if requests are often retracted. vLLM preempts by recompute. The 3050's own 0.85-watermark run (+4.7 % tok/s) wasn't interleaved. | `kv_preempt_watermark` |
| KV pool size | SGLang: as large as fits; smaller caps concurrency and peak throughput. | `kv_pool_fraction` (reload) |
| KV cache dtype q8 | Halves KV memory; perplexity +0.004 on a 7B at q8_0, q4 noticeably worse; llama.cpp speed impact minimal. The largest lever on a 4 GB card. | none (engine change) |
| CUDA-graph capture sizes | Graphs cost memory; fewer capture sizes leave more KV on small cards. | none (`decode_graphs` is excluded) |
| clocks / power | Locked clocks are steadier but slower on average; laptop Dynamic Boost shifts ~15 W to the GPU. Key a profile by power state rather than tune it. | none (profile metadata) |
| prefix caching, KV block size, host polling | Prefix caching cuts TTFT on multi-turn chats; block size 16-128 all fine; polling unmeasured. | n/a |

On calibration against a fixed rule, the sources split. Sarathi, TetriInfer, SCOOT and Vidur all find the best
budget or config depends on the GPU, the model and the workload (Vidur: reusing one trace's optimum cost up to 2×).
TensorRT-LLM measured < 0.6 % throughput change from 2048 to 16384 tokens on 4×H100, so throughput is often flat
and the stall is what a profile buys. Sarathi's suggested middle ground is "one-time profiling": set the budget to
the largest one whose step stays under an ITL target. On this engine that would be a cheap measured rule, not an
A/B search.

Sources: [Sarathi-Serve](https://arxiv.org/html/2403.02310), [DistServe](https://arxiv.org/html/2401.09670v3),
[TetriInfer](https://arxiv.org/html/2401.11181), [SCOOT](https://arxiv.org/html/2408.04323v2),
[Vidur](https://arxiv.org/html/2405.05465), [Etalon](https://arxiv.org/html/2407.07000),
[vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html),
[vLLM sweeps](https://docs.vllm.ai/en/latest/benchmarking/sweeps.html),
[SGLang tuning](https://docs.sglang.io/advanced_features/hyperparameter_tuning.html),
[TensorRT-LLM tuning](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html),
[KV quantisation](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/),
[TensorRT benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html),
[GPU variability, Sinha et al.](https://arxiv.org/abs/2208.11035).
