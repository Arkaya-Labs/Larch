# Gap list: larch vs llama.cpp, m1 on wsl5070 (5070 Ti, WSL)

Where larch's time goes per decode step, against llama.cpp b11458 on the same GGUF, workload and prompts.

- **larch:** scratch build `55063bd` = dev `4b2e30d` + A1B3 + F4E3 `50685dd` (#61), CUDA release.
- **llama.cpp:** `llama-serve-bench` (tools/llama-serve-bench) with the flag search's best flags
  (`-ngl 99 -fa on -ctk f16 -ctv f16 -t 4 -b 2048 -ub 1024`), context 8 x 8192.
- **Capture:** `flock ~/src/gpu.lock nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -o <out> <cmd>`, then
  `nsys export --type sqlite`; per-kernel sums from `CUPTI_ACTIVITY_KIND_KERNEL`.
- **Steps:** RMS-norm launches / 65 (2 per layer x 32 + the final norm), so both engines are counted the same way.
  Host gaps = wall - GPU busy over the kernel span.
- nsys slows both engines a little. The unprofiled rows are this PR's `wsl5070-m1-*.json` and its scoreboard.
- **llama.cpp's driver numbers are withheld.** `llama-serve-bench` misses architecture 16's fixed-shape gate (3.2-6.7 %
  under `llama-batched-bench` at some shapes), so no number its host loop shapes is published: aggregate tok/s, TTFT,
  TPOT, wall, host gaps. What stays is trace evidence: llama.cpp's own GPU kernels, their time per call, per step and
  in total, for the same work and prompts. The gaps below are kernel time, not a baseline comparison.

## W2 at 8: 8 conversations x 8 turns, scripted, closed loop (decode-heavy)

Same work on both: 64 requests, 14,505 generated and 24,674 evaluated prompt tokens, every prompt hash equal.

| | larch | llama.cpp |
|---|---|---|
| aggregate tok/s (under nsys) | 49.95 | withheld |
| TTFT p50 / p99 ms | 494 / 1,276 | withheld |
| TPOT p50 / p99 ms | 144.6 / 164.8 | withheld |
| wall over the kernel span | 290.7 s | withheld |
| GPU busy (kernel time) | 286.6 s (99 %) | 29.2 s |
| steps | 2,040 | 2,550 |

Time per op per step (ms) and over the run (s):

| Op | larch kernels | larch ms/step | larch s | llama.cpp kernels | llama.cpp ms/step | llama.cpp s |
|---|---|---|---|---|---|---|
| mmvq (1-16 rows) | `mmvq_k` | **106.12** | 216.48 | `mul_mat_vec_q`, `quantize_q8_1` | 2.29 | 5.83 |
| dequant + gemm | `rows_k` (dequant) + cuBLAS `nvjet_*` | 3.95 | 8.07 | `mul_mat_q` (+ fixup, `quantize_mmq_q8_1`) | 6.31 | 16.10 |
| attention (+ KV write) | `attn_k`, `kv_write_k` | **29.23** | 59.62 | `flash_attn_ext_*`, `k_set_rows` | 2.38 | 6.08 |
| sample_prep | `select_k` | 0.27 | 0.54 | none (host sampling) | 0 | 0 |
| small ops | `rms_k`, `rope_k`, `silu_mul_k`, `add_k`, ... | 0.94 | 1.91 | `rms_norm_f32`, `k_bin_bcast`, `rope_norm`, ... | 0.47 | 1.19 |
| host gaps between steps | | 1.98 | 4.03 | | withheld | withheld |

llama.cpp runs its 8-row decode through `mul_mat_q` (quantized GEMM), larch through `mmvq_k`, so the fair line is
the quantized matmuls together: **larch 224.6 s vs llama.cpp 21.9 s.**

## The three biggest gaps (W2 at 8)

1. **Quantized matmul at decode: +202.6 s of the 257.4 s kernel-time difference (79 %). Owner: A2C2a (mmvq); the row-count
   routing between mmvq and GEMM is A3S1's.** `mmvq_k` takes 489 us per call (443,096 calls, about 217 per step)
   against 31.5 us for `mul_mat_vec_q` and 39.6 us for `mul_mat_q`. Its shape (`kernels/src/cuda/quant.cu`): one warp
   per output column, every weight decoded element by element through `elem<T>` with the lanes strided across k, and
   `x` read from global memory once per row per element. At 8 rows a step spends 106 ms in it, about 19x the 5.5 ms it takes to read the 4.9 GB of weights once at the card's 896 GB/s.
   llama.cpp quantizes the activations to q8_1 once and uses integer dot products over whole blocks.
2. **Attention: +53.5 s (21 %). Owner: A2C2b.** `attn_k` takes 881 us per layer-step (67,584 calls) against
   `flash_attn_ext_f16`'s 81 us at the same contexts (up to 3,624 tokens per conversation).
3. **Sampling and small ops: +1.3 s (0.5 %). Owner: A2C2c.** larch's `select_k` (0.27 ms/step) plus its small ops
   (0.94 ms/step) against llama.cpp's 0.47 ms/step of small ops (its greedy sampling is on the host).

**Not a gap here:** host overhead (F4E3). larch's host gaps are 2.0 ms/step because larch is GPU-bound: the GPU is
busy 99 % of the run. Once gaps 1 and 2 close, host overhead has to be measured
again: with no CUDA graph launch at batch 8 in this trace (0 `cudaGraphLaunch`) and about 530 kernel launches per
step, launch cost is the next thing to show.

## W4 at 16: mixed prefill and decode (200 single-turn prompts, no reuse, context 16 x 2560)

The spans differ: larch's capture is bounded (`--duration 120`, `-t cuda,nvtx`; 36 requests finished, 12,102
tokens generated), llama.cpp's is the whole run (200 requests, 64,437 tokens). So the comparison is per step and per
1,000 generated tokens, not totals. The first larch capture of the whole run (`-t cuda,nvtx,osrt`) was killed at
1,132 s by a segfault in nsys's injection thread inside the process (`[NSys]: potentially unexpected fatal signal
11`, a null dereference in the collector, not in larch code); the bounded rerun exited 0.

| | larch | llama.cpp |
|---|---|---|
| aggregate tok/s (under nsys) | 72.31 | withheld |
| TTFT p50 / p99 ms | 688 / 5,935 | withheld |
| TPOT p50 / p99 ms | 191.5 / 196.1 | withheld |
| wall over the kernel span / GPU busy | 167.6 s / 165.3 s (99 %) | withheld / 76.7 s |
| steps | 943 | 5,393 |

| Op | larch ms/step | larch s per 1k tokens | llama.cpp ms/step | llama.cpp s per 1k tokens |
|---|---|---|---|---|
| mmvq | **140.95** (`mmvq_k` 663 us/call) | 10.98 | 1.43 | 0.12 |
| dequant + gemm | 7.39 (`rows_k` 228 us/call, then cuBLAS) | 0.58 | 9.08 | 0.76 |
| attention (+ KV write) | **25.39** (`attn_k` 751 us/call) | 1.98 | 3.08 (`flash_attn_ext_f16` 92 us) | 0.26 |
| sample_prep | 0.45 | 0.03 | 0 | 0 |
| small ops | 1.13 | 0.09 | 0.64 | 0.05 |
| host gaps between steps | 2.46 | 0.19 | withheld | withheld |
| **wall** | | **13.85** | | withheld |

Same three gaps, same order: quantized matmul +10.68 s per 1k tokens (A2C2a, A3S1), attention +1.72 (A2C2b),
sampling and small ops +0.07 (A2C2c); larch's host gaps stay small while it is GPU-bound (F4E3).
At 16 rows `mmvq_k` costs 663 us per call against 489 us at 8: its time grows with the row count, where a kernel
bound by reading the weights would stay flat.
