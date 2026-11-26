# m13 original file against llama.cpp on a 16 GB GPU, 26 Nov 2026

Larch on the original m13 file (`bench/models.toml`: a 35B-A3B hybrid MoE, Q3_K_M, 17.1 GB, the download as
published) against llama.cpp on the same machine and file, with every change of this run stacked, and the
faster copy (`larch faster`, PR #6) beside it.

Test machine: RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, Windows 11 + WSL2 (NVIDIA driver 617.14). llama.cpp b12723
(`abca692`), `-ngl 99 -fa on -ctk f16 -ctv f16 -t 4 -b 2048 -ub 1024` (searched again in this session).
larch: the stack of PR #12 (four chats) and PR #13 (graph floor, as pushed) on dev. Raw results:
[`35b-original-file-2026-11-26/`](35b-original-file-2026-11-26/).

## Summary

1. **The bar is missed on the original file.** Against llama.cpp at `--n-cpu-moe 4` (the setting that fits in
   the memory the GPU reports free, below), larch is **−3.2 % at 1 chat and −9.9 % at 4 chats**, both outside
   noise. Against `--n-cpu-moe 3`, which the protocol's search picks and which spills about 600 MiB into shared
   system memory, it is −9.6 % and −16.6 %.
2. **4 chats moved from 174 to 218 tok/s (+26 %)** with a mixed step's prefill rows now streamed (PR #12): a
   512-token chunk beside replying chats went from 577 ms to 198 ms. TTFT p99 at 4 chats fell from 2,347 to
   959–1,138 ms and is now 34 % below llama.cpp's; TPOT p99 fell from 30.4 to 20.4–23.0 ms, within noise of
   llama.cpp's.
3. **1 chat is unchanged** at 122 tok/s; its TTFT p99 (713 ms) stays 55 % above llama.cpp's and its TPOT p99 is
   3 % below. Matching llama.cpp's 3 CPU expert layers without spilling was tried and dropped (below).
4. **The faster copy** stays well ahead: +29.6 % at 1 chat and +31.3 % at 4 against `--n-cpu-moe 4`. It is a
   different file (35 % of each layer's experts as Q2_K, mean KL +8.8 % prefill / +7.9 % decode over the
   original, report PR #7), so `compare` refuses its pairs by design.
5. **llama.cpp's `--n-cpu-moe 3` fits only by spilling.** It allocates 15,619 MiB of CUDA0 buffers (model
   14,157.7, KV 200, recurrent state 251.3, compute 1,010) on a GPU whose free memory it reports as 15,009 MiB;
   the Windows driver backs the rest with shared system memory. At `--n-cpu-moe 4` it allocates 14,750 MiB and
   fits. Larch never plans past the reported free memory, so its comparison row is `--n-cpu-moe 4`; the
   spilling row is reported beside it.
6. **Where the 4-chat gap is now:** in decode. Of a 294 s run, 4-row decode steps take 239 s at 15.0 ms each
   (the faster copy, with no host layers: 10.5 ms), and the 323 mixed steps 54 s at 167 ms. TPOT p50 is 17.2 ms
   against llama.cpp's 15.5 ms. Next: the four host layers' CPU expert sections and their device round trip per
   layer in a 4-row decode step, then the mixed step's 512-token chunks.

## Final rows (same session)

Order per load: larch original, larch faster copy, llama.cpp `--n-cpu-moe 4`, `--n-cpu-moe 3`, `3`, `4`, faster
copy, original; every run its own process and one exclusive GPU hold, used VRAM back under 2.5 GB and steady
before each. W4: 200 prompts, exact output lengths, `max_context` 2560. Per cell: mean of 2 runs (the runs), and
the change against llama.cpp `--n-cpu-moe 4`. A margin is noise when under 2 % or under either side's run spread.

### m13, W4 at 1 (larch rows against llama.cpp --n-cpu-moe 4)

| Engine, file | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p99 ms |
|---|---|---|---|---|
| llama.cpp --n-cpu-moe 4, original file (fits) | 126.1 (126.5 / 125.7) | 177 (171 / 182) | 459 (449 / 468) | 7.83 (7.83 / 7.83) |
| llama.cpp --n-cpu-moe 3, original file (spills ~600 MiB to system memory) | 134.9 (134.5 / 135.2); **+7.0 %** | 169 (173 / 166); −4.0 % (noise) | 411 (423 / 400); **−10.4 %** | 7.30 (7.26 / 7.35); **−6.7 %** |
| larch, original file | 122.0 (122.2 / 121.7); −3.2 %, worse | 166 (166 / 166); −5.9 % (noise) | 713 (713 / 713); +55.4 %, worse | 7.59 (7.54 / 7.65); **−3.0 %** |
| larch, faster copy (0.35) | 163.4 (163.5 / 163.3); **+29.6 %** | 109 (112 / 106); **−38.1 %** | 488 (486 / 489); +6.3 %, worse | 5.71 (5.70 / 5.71); **−27.1 %** |

### m13, W4 at 4 (larch rows against llama.cpp --n-cpu-moe 4)

| Engine, file | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p99 ms |
|---|---|---|---|---|
| llama.cpp --n-cpu-moe 4, original file (fits) | 242.0 (243.7 / 240.2) | 232 (231 / 234) | 1584 (1670 / 1499) | 21.03 (18.35 / 23.71) |
| llama.cpp --n-cpu-moe 3, original file (spills ~600 MiB to system memory) | 261.5 (260.1 / 262.8); **+8.1 %** | 208 (212 / 204); **−10.6 %** | 1402 (1411 / 1393); **−11.5 %** | 17.18 (17.08 / 17.29); −18.3 % (noise) |
| larch, original file | 218.0 (219.2 / 216.9); −9.9 %, worse | 184 (185 / 182); **−20.9 %** | 1049 (959 / 1138); **−33.8 %** | 21.68 (20.37 / 22.99); +3.1 % (noise) |
| larch, faster copy (0.35) | 317.6 (317.5 / 317.6); **+31.3 %** | 120 (121 / 118); **−48.5 %** | 552 (552 / 552); **−65.2 %** | 14.20 (14.09 / 14.32); **−32.5 %** |

### Against today's larch (dev build, same day, A B B A)

| m13 original file | dev | this stack |
|---|---|---|
| W4@4 tok/s | 173.6 / 173.8 | 218.9 / 218.8 (A/B) and 219.2 / 216.9 (final rows) |
| W4@4 TTFT p99 | 2,347 / 2,189 ms | 801 / 806 ms (A/B), 959 / 1,138 ms (final rows) |
| W4@4 TPOT p99 | 30.4 / 30.7 ms | 20.4 / 20.5 ms (A/B), 20.4 / 23.0 ms (final rows) |
| W4@1 tok/s | 121.9 | 122.3 (A/B), 122.2 / 121.7 (final rows) |
| W4@1 TTFT p99 / TPOT p99 | 718 ms / 8.12 ms | 726 ms / 7.60 ms (A/B), 713 ms / 7.54–7.65 ms |

TTFT p99 and TPOT p99 are no worse than today's larch at either load.

## Steps: kept and dropped

| Step | Change | Target: before → after | Kept? |
|---|---|---|---|
| 1 Where the time goes | memory traces of one load; step profiles | — | findings below |
| 2 Tight fit at 1 chat | size reserves to reach llama.cpp's 3 CPU layers | — | **stopped**: needs the plan past the reported free memory (llama.cpp's 3 spills) |
| 2b One split layer | 3 whole host layers + 1 split by the routing histogram, static per-expert plan | W4@1 120.3 → 82.5 (−31 %) | **dropped** at measure-first (owner's gate +4 %) |
| 3a Four chats: mixed steps stream their prefill rows | PR #12 | W4@4 173.7 → 218.9 (+26 %); mixed step 577 → 198 ms | **kept** |
| 3b Four chats: four slots on a planned load | PR #12 | no cost where the check passes (172.4 vs 172.8); a short check moves a layer (134.5 vs ~113 on 3 seats, held-memory run, noisy) | **kept** |
| 3c Prefill chunk 1024 / 2048 at startup | knob only | W4@4 174.3 → 110.9 / 157.7 (cost seats or a layer) | not a lever |
| 4 Graph floor | PR #13 | cold 0.3 copy W4@1 70.6 → 163.0 (graphs off → one seat's capture) | **kept**; PR not for merge as pushed (review fixes committed, not pushed) |
| 5 Fused entries 4, 5, 6, 8 on mmvq_k | — | — | skipped (owner); next |
| Seats demand switch (earlier run) | PR #11 | neutral where measured | **not for merge** (record) |

### Step 1 findings

- CUDA reports 15,738 MB (15,009 MiB) free at device open in every run, whatever other programs use; llama.cpp
  reads the same figure. Larch's load then reads 107 MB less after engine start, and a fixed 116.6 MB more
  disappears when the first large host buffer is touched (the 117 MB the planner's headroom was measured from).
- The planner keeps 4 expert layers in host memory at 1 chat because its prefill-pool reserve is sized from the
  file's largest expert tensor (the MTP block's, 285 MB) where the host layers need 184.5 MB; but at 3 layers
  the post-load check would be about 65 MB short with every other reserve as measured, so 3 layers do not fit
  without exceeding the reported free memory.
- Prefill alone (one sequence): larch 2,665 tok/s at 512 tokens (llama.cpp 2,584) and 1,871 at 2048 (llama.cpp
  4,309, which streams per 1,024 tokens). At 4 chats nearly every chunk shares its step with decode rows, and
  those steps did not stream host experts at all (578 ms per 512 tokens): step 3a.

### Step 2b: why the split layer lost

The static per-expert path has no streaming pool (host experts prefill on the CPU: TTFT p50 651 against 185 ms)
and runs whole host layers through the split route, with a device round trip per layer (TPOT p50 8.8 against
7.4 ms). The loader cannot mix whole host layers with a split one, and a new placement path was ruled out.

## m1 and m5 (no regression)

### m1 and m5, W4: this build (B) against the dev build (A), same session, A B B A

| Model, load | B agg tok/s (runs) | A (runs) | change | TPOT p99 B / A |
|---|---|---|---|---|
| m1 W4 at 1 | 134.1 (134.0 / 134.1) | 134.0 (134.0 / 133.9) | +0.1 % (noise) | 7.52 / 7.53 |
| m1 W4 at 8 | 663.8 (663.5 / 664.2) | 664.6 (664.2 / 665.1) | −0.1 % (noise) | 14.24 / 14.25 |
| m5 W4 at 1 | 143.4 (143.5 / 143.4) | 143.4 (143.3 / 143.5) | +0.0 % (noise) | 7.00 / 7.00 |
| m5 W4 at 8 | 732.4 (731.7 / 733.2) | 733.4 (734.4 / 732.5) | −0.1 % (noise) | 12.75 / 12.65 |

Every m1 and m5 cell is within 0.1 % and inside the run spread: no regression. TPOT p99 moves by at most 0.10 ms (m5 at 8).

## Parity

`larch parity --mode all` with PR #12's build: m1 and m5 pass every mode. m13 passes every mode up to
`chunked`, which reports "not run: chunked: 933 chunk steps ran without a decode row", the same on the dev
build: the hybrid's chunked mode does not run on this card, before and after. PR #13 changes no kernel or model
path.

## Protocol notes

- llama.cpp's flags: the sweep's search (fewest MoE layers on the CPU at which the driver runs W4's largest
  shape with every CUDA0 buffer under the free VRAM) reads that free VRAM from `nvidia-smi` (15,728 MiB at the
  search), not the 15,009 MiB llama.cpp itself reports; by its own rule with llama.cpp's figure it gives
  `--n-cpu-moe 4`. Both rows are reported; no flag of the protocol was changed.
- `scripts/baseline.py compare` accepted all 16 original-file pairs (2 llama.cpp runs × 2 larch runs, both
  `--n-cpu-moe` rows, both loads); its log is `compare.log`. Faster-copy pairs are refused by design (another
  model file).
- `bench/manifest/m13.jsonl` (the llama.cpp driver's W4 manifest) is added by report PR #7.
- Result files keep every field except each request's `token_gaps_ms` list, as in earlier committed results.
- Files: `m13-{orig,faster,llama4,llama3}-r{1,2}-w4-{1,4}.json` are the final rows; `m{1,5}-{A,B}{1..4}-w4-{1,8}.json`
  the A B B A regression rows; `step-mix-*` (step 3a A/B), `step-clean4-*` (3b), `step-chunk-*` (3c) and
  `step-floor-*` (step 4) the step measurements; `parity-four-*.txt` and `parity-dev-m13-chunked.txt` the parity logs;
  `wsl5070-m13-llama-*` the llama.cpp flag search.
- The bench's `--machine` tag for this computer is `wsl5070`.
