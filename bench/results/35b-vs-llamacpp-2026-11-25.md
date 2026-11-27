# m13 (35B MoE) on a 16 GB GPU: larch's faster copy and the original file, against llama.cpp on the original file, 25–26 Nov 2026

For the original file, [`35b-final-2026-11-26.md`](35b-final-2026-11-26.md) supersedes the rows here (PR #16, with
the host-expert, mixed-step and pinned-streaming changes). This report stays the evidence for the faster copy's
fraction (`COLD_FRACTION` in `crates/larch/src/faster.rs`).

Larch on m13 (`bench/models.toml`: a 35B-A3B hybrid MoE, Q3_K_M, 17.1 GB) against llama.cpp on the same machine and
the same downloaded file, with the faster copy that `larch faster` makes (PR #6, branch `feat/35b-repack` at
`b71282c`) and with the original file.

Test machine: RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, Windows 11 + WSL2 (NVIDIA driver 617.14).
llama.cpp b12723 (`abca692`), `-ngl 99 -fa on -ctk f16 -ctv f16`, `--n-cpu-moe 3 -t 4 -b 2048 -ub 1024` (searched
again in this session, below). Raw results are in [`35b-vs-llamacpp-2026-11-25/`](35b-vs-llamacpp-2026-11-25/).

## Summary

1. **The faster copy is a different file.** It stores the 35 % of each layer's experts that the model routes to least
   as Q2_K (15.7 GB instead of 17.1 GB), so on this card every expert layer fits on the GPU. It is compared here with
   llama.cpp on the original file, and both its quality and its speed are reported against that file.
2. **Larch with the faster copy vs llama.cpp with the original file (same session, 2 runs a side):** aggregate
   **+19.6 % at 1 chat** and **+24.9 % at 4 chats**. This is not a same-file baseline (D15): llama.cpp cannot load
   the copy (it refuses the cold-expert tensors' shape; checked 26 Nov), so it runs the original. TTFT p50 is 31 % and 44 % lower and TPOT p99 22 % and 20 % lower. The loss: TTFT p99 at 1 chat is
   20.8 % higher (484 vs 401 ms); at 4 chats it is 63 % lower.
3. **Quality of the faster copy:** mean KL against the llama.cpp CUDA reference dump of the original file is 8.8 %
   (prefill) and 7.9 % (decode) above the original file's under larch, inside the 10 % budget (an extra gate of ours on top of D12's bound;
   see Quality for what it measures and its noise). Decisive top-1 is 1.0
   and 0.99955 (original: 1.0 and 1.0).
4. **`compare` refuses all 8 faster-copy pairs** ("model differs: m13.gguf vs m13-cold35.gguf"): by the protocol, a
   different file is a different model. The faster-copy rows are therefore not compare-accepted pairs. All 8
   original-file pairs are accepted.
5. **The original file is still behind llama.cpp:** −10.7 % at 1 chat and −31.7 % at 4 chats, with TTFT p50 +5 % and
   +102 % and TPOT p99 +13 % (noise) and +74 %. Nothing in this run changes it. Where the 1-chat gap comes from is
   below; closing it is the next piece of work.
6. **The faster copy needs the GPU to itself.** With 499 MiB of VRAM held by another process it kept one expert layer
   in host memory and ran at 115.3 tok/s at 1 chat, below the original file (121.8, final rows, branch build). The app checks this after
   switching and goes back to the original.
7. **m1 and m5:** no regression from the branch. Same-session A/B against the dev build: m5 W4 at 1 143.4 vs 143.2,
   m1 W4 at 8 659.5 vs 643.9 (within the branch's run spread). The final-run rows read 2–4 % below the 25 Nov sweep on
   both builds; that is the session, not the change.

## Final rows (same session, V-L-L-V)

Order per load: faster copy, original file, llama.cpp, llama.cpp, original file, faster copy; every run its own
process and one exclusive GPU hold, used VRAM back under 2.5 GB and steady before each. W4: 200 prompts, exact output
lengths, `max_context` 2560. Per cell: mean of 2 runs (the runs), and for larch the change against llama.cpp. A
margin is noise when under 2 % or under either side's run spread (the sweep's rule).

### m13, W4 at 1

| Engine, file | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p99 ms |
|---|---|---|---|---|
| llama.cpp, original file | 136.4 (136.4 / 136.5) | 157 (157 / 157) | 401 (402 / 400) | 7.28 (7.28 / 7.27) |
| larch, faster copy | 163.2 (163.3 / 163.1); **+19.6 %** | 108 (106 / 110); **−31.2 %** | 484 (484 / 485); +20.8 %, worse | 5.71 (5.71 / 5.71); **−21.6 %** |
| larch, original file | 121.8 (122.9 / 120.8); −10.7 %, worse | 165 (167 / 164); +5.0 %, worse | 714 (711 / 716); +78.0 %, worse | 8.19 (7.50 / 8.89); +12.6 % (noise) |

### m13, W4 at 4

| Engine, file | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p99 ms |
|---|---|---|---|---|
| llama.cpp, original file | 254.9 (256.9 / 252.8) | 216 (208 / 225) | 1459 (1409 / 1510) | 17.74 (17.40 / 18.08) |
| larch, faster copy | 318.2 (317.8 / 318.6); **+24.9 %** | 122 (124 / 120); **−43.6 %** | 543 (553 / 533); **−62.8 %** | 14.25 (14.12 / 14.38); **−19.7 %** |
| larch, original file | 174.0 (173.8 / 174.2); −31.7 %, worse | 438 (449 / 428); +102.4 %, worse | 2362 (2535 / 2189); +61.9 %, worse | 30.89 (30.32 / 31.45); +74.1 %, worse |

`scripts/baseline.py compare`: 8 of 16 pairs accepted (every original-file pair); the 8 faster-copy pairs are
refused on the model file, as above.

### llama.cpp flags, searched again

The sweep's search (`--n-cpu-moe`: the fewest MoE layers on the CPU at which the driver runs W4's largest shape,
4 × 2560, with every CUDA0 buffer under the free VRAM; then `-t`/`-b`/`-ub` by max √(pp2048 · tg128)): 3 layers
again, `-t 4 -b 2048 -ub 1024` (the sweep had `-b 1024`). CUDA0 buffers 15,618 MiB of 15,728 MiB free. Larch's
memory use did not change for the original file, so no other count was tried. Log: `wsl5070-m13-llama-search.txt`.

## Quality

`larch parity` against the llama.cpp CUDA reference dump of the original file (100 prompts, `n_ubatch` 32), same
larch binary for every file (dev `488cd91`), graphs on in decode.

| File | Prefill mean KL | Decode mean KL | Within 10 % of the original? | Decisive top-1 (prefill / decode) |
|---|---|---|---|---|
| original | 0.0040895 | 0.0039561 | — | 1.0 / 1.0 |
| copy, 30 % cold | 0.0044458 (+8.7 %) | 0.0041208 (+4.2 %) | yes | 1.0 / 0.99955 |
| **copy, 35 % cold (shipped)** | 0.0044506 (+8.8 %) | 0.0042692 (+7.9 %) | yes | 1.0 / 0.99955 |
| copy, 40 % cold | 0.0045207 (+10.5 %) | 0.0042370 (+7.1 %) | no (prefill) | 1.0 / 1.0 |

Every run passed the parity bound (0.0354). 40 % was the first fraction tried; it is over budget in prefill, so the
fraction was lowered, never the budget.

**The 10 % rule.** D12's calibrated bound (0.0354) is the gate every load must pass, and every copy passes it. The
10 % rule is an extra margin we hold a lossy copy to: its mean KL over all positions (each position weighted
equally, as `larch parity` reports it) within 10 % of the original file's under the same binary. Decided 26 Nov 2026
(Aayush): keep 35 % on this rule, recorded here and beside D12 in `larch-architecture.md`.

Its noise, from the `per_prompt` entries of the parity files (bootstrap over the 100 prompts, 2000 resamples,
ratio of summed per-prompt means): weighted per prompt instead of per position, every copy is over 10 %, and the
ranges overlap, so one 100-prompt run does not separate 35 % from 40 %:

| Copy | Prefill, per prompt (95 % range) | Decode, per prompt (95 % range) |
|---|---|---|
| 30 % | +12.8 % (+5.2 to +20.0) | +16.6 % (+5.6 to +30.0) |
| 35 % (shipped) | +11.6 % (+3.8 to +20.3) | +15.1 % (+6.7 to +25.6) |
| 40 % | +17.5 % (+10.4 to +26.3) | +25.9 % (+12.4 to +44.0) |

Not checked: whether the parity prompts overlap the workload that recorded `bench/routes/m13.experts`; if they do,
the copy's KL reads optimistic. A second prompt set would settle both points.

## Steps: kept and dropped

All single runs of W4 at 1 and at 4 (each its own process), dev build unless noted; the final rows above supersede
them for the kept step.

| Step | What was measured | Before → after | Kept? |
|---|---|---|---|
| 1 Arms | A: original file, defaults | 122.8 / 174.0 | baseline |
| | B: original file, 4 seats planned at load, expert cache off | 122.5 / 174.5 | no change; no reload in the bench |
| | C: 40 % copy, defaults (0 host layers, 7 slots) | 162.9 / 318.4 | over the KL budget |
| | D: 40 % copy, 4 seats | 162.9 / 318.1 | same as C |
| | C3 / D3: 30 % copy, defaults / 4 seats | 75.2 / 158.0 and 119.7 / 180.8 | slower: no room for decode graphs at 1 chat, 3 slots at 4 |
| | C35 / D35: 35 % copy, defaults / 4 seats | 163.4 / 319.0 and 162.9 / 318.1 | the fraction shipped |
| 2 Seats (4 chats on a load left at 3 slots) | dev vs fix, same load | 239.6 vs 239.8 at 4 (identical load) | **dropped**: neutral where it can run; review found it would cost reloads (below) |
| 3 Faster copy in the app and CLI | final rows | 136.4 → 163.2 and 254.9 → 318.2 vs llama.cpp | **kept**, PR #6 |
| 4 Grouped MoE tile choice (sm_120) | not measured | — | held; the change is a prefill tile, not decode |
| 5 Fused entries 4, 5, 6, 8 by default | not measured | — | held |

Step 2's change made four chats ask for a fourth seat on a load whose check left three. Code review found that
the switch waits at most 5 s for started replies and is otherwise cancelled (so with three chats generating it
rarely runs), that a re-plan landing on three slots again would reload on every four-chat burst, and that a request
waiting for KV room, not a seat, would set it off. It was not shipped.

## Where the original file loses at 1 chat

From the fit diagnostics of arm A (`LARCH_FIT_DIAGNOSTICS=1`) and the loader's code; no change made yet.

- Larch keeps 4 expert layers in host memory, llama.cpp 3. At the one-slot plan that picks the count, 3 layers are
  58.7 MB short.
- The prefill streaming pool's reserve (318.8 MB) is sized from the file's largest expert tensor anywhere, which is
  the MTP block's (Q8_0, 285 MB); the host layers' own pool is 184.5 MB. That 134 MB over-reserve alone moves the
  count from 3 to 4.
- But the load also takes 132–199 MB the plan does not count (the post-load check passed by 2.4 MB, carried by that
  over-reserve). A plan that only shrinks the pool would load 3 layers, fail the check and reload at 4. Naming that
  memory comes first.
- Larch reads 821 MiB less free VRAM at start than the llama.cpp search's probe; whether the two readings are taken
  alike is to be checked.

## Memory margin (faster copy, W4 at 1)

| Condition | Host layers | Graphs | agg tok/s |
|---|---|---|---|
| Card to itself (final rows) | 0 | on | 163.2 |
| 499 MiB held by another CUDA process | 1 | on (63,983 of 64,564 steps replayed) | 115.3 |
| Original file, card to itself | 4 | on | 121.8 |

A copy short of room is slower than the original even with graphs on, so the app keeps the copy only when it loads
with every expert layer on the GPU and graphs on (PR #6).

## m1 and m5 (regression check)

| Model, load | Final run, branch (runs) | 25 Nov sweep, dev (runs) | Same-session A/B: dev vs branch (runs) |
|---|---|---|---|
| m1 W4 at 1 | 130.8 (132.6 / 128.9) | 134.0 (133.9 / 134.1) | — |
| m1 W4 at 8 | 637.3 (628.0 / 646.7) | 662.8 (663.1 / 662.6) | 659.5 (655.8 / 663.1) vs 643.9 (623.1 / 664.7) |
| m5 W4 at 1 | 139.5 (138.2 / 140.8) | 143.3 (143.3 / 143.2) | 143.4 (143.3 / 143.4) vs 143.2 (143.3 / 143.0) |
| m5 W4 at 8 | 717.4 (708.2 / 726.6) | 734.6 (734.4 / 734.7) | — |

The final-run rows ran last, after 1.5 h of m13 runs, and read 2.3–3.8 % below the sweep. The A/B, run straight
after in the same session (dev, branch, branch, dev), puts both builds back at the sweep's level; m1 at 8's
difference is inside the branch's run spread (6.5 %). The branch changes no decode path.

## Builds

| `build` hash in the results | Binary |
|---|---|
| `c352d7b1…` | larch, branch `feat/35b-repack` at `b71282c` (PR #6): final m13 rows, m1/m5 final rows, `regress-*-B*` |
| `6d683e09…` | larch, dev at `488cd91`: the arms (step 1), the margin run, `regress-*-A*` |
| `d28ab07f…` | llama.cpp's W4 driver at b12723 (`abca692`) |

The parity files carry no build field; they were made with the dev binary at `488cd91`.

## Protocol notes

- Not re-run: the `--n-cpu-moe 2` probe, which aborted at 15,581 MiB free while free memory was still settling
  (`wsl5070-m13-llama-search.txt`), and a further m1 W4 at 8 dev/branch pair. The 26 Nov report measures llama.cpp's
  fit with its own load log.

- llama.cpp's W4 driver needs `bench/manifest/m13.jsonl`, which was missing from the checkout; it was regenerated
  with the sweep's driver (`baseline.py --alias m13 manifest`, 338 lines, as in the sweep) and is added here.
- Result files keep every field except each request's `token_gaps_ms` list, as in earlier committed sweeps; the
  summaries keep the gap statistics. File names: `m13-{faster,orig}-r{1,2}-w4-{1,4}.json`,
  `m13-llama-r{1,2}-w4-{1,4}.json`, `{m1,m5}-larch-w4-{1,8}-r{1,2}.json`, `regress-*` (the A/B), `arm-*` (step 1;
  `C`/`D` 40 %, `C3`/`D3` 30 %, `C35`/`D35` 35 %), `margin-*`, `parity-*`.
- The bench's `--machine` tag for this computer is `wsl5070`.
