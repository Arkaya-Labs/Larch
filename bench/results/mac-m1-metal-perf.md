# Metal performance: larch against llama.cpp, m1 on the M5 (mac)

The numbers that F1M2 reached (PR #89, on `dev` as `7b4b302`). Until this file they were only in that PR's
description, which also has the intermediate commits, every command and the experiments that gave no win.

- **Measured:** 9 and 10 Oct 2026. This file is from 11 Oct 2026. Nothing was measured again for it.
- **Machine:** one Apple M5 (10 GPU cores, 24 GB), macOS 27.0, Xcode 27.0. These are one machine's numbers, not
  targets. No other Mac is measured.
- **larch:** `e3a5c73` (`f1/metal-performance`), Metal, release build. No later commit changes a Metal kernel or the
  Metal route. The PR head `e33cf0a` adds a dev merge and one check in `device_open`, with the same metallib
  `bf42d38fe1fd`. Dev then got #92 and #105 (`4b41416`). #105 raises the mmvq row ceiling on CUDA only. The numbers
  are NOT RUN again at any of these commits.
- **dev before:** `origin/dev` at `4b4a7aa`, the Metal backend before this work.
- **llama.cpp:** b11458, Metal, `-ngl 99 -fa 1`.
- **Method:** one engine on the GPU at a time (`flock ~/src/gpu.lock`), and the binaries run in turn (dev, head,
  head, dev). Single runs differ by up to 5 % on this desktop, and the rates fall as the machine warms up.

## Summary

- **TTFT:** 114 ms against llama.cpp's 160 ms for a 52-token prompt (1.41x faster), and 2.04 s against 2.70 s for
  1594 tokens (1.32x faster). dev before: 836 ms and 16.75 s.
- **Decode, 1 conversation:** equal to llama.cpp at a short context (26.47 against 26.26 tok/s) and 2.5 % behind at
  1594 tokens (25.02 against 25.66). dev before: 20.90 and 7.38 tok/s.
- **Decode, 4, 8 and 16 conversations:** 82.19, 154.53 and 191.26 tok/s against llama.cpp's 85.41, 151.68 and 168.13
  (-3.8 %, +1.9 %, +13.8 %). dev before: 35.88, 45.43 and 43.31.
- **Where a token goes:** a 1-row decode token takes 37.1 ms. About 36 ms of it is mmvq at about 136 GB/s, which is
  the memory bandwidth. No measured win is left in the Metal kernels.

## One conversation

`larch chat -m <models>/m1.gguf --max-tokens 128 --temperature 0 --no-keep-kv --data-dir <new dir> < prompt`, a new
process and data directory per run, 2 short and 2 long runs per binary. The larch TTFT is the first request of a new
process. For llama.cpp, `llama-bench -m <m1> -p 52,512,1594 -n 128 -r 3` gave 324 tok/s at 52 tokens and 590 tok/s
at 1594 tokens. Its TTFT here is the prompt length over that rate. llama-bench warms up first, so its side is warm.

| Prompt | Metric | dev before | larch | llama.cpp |
|---|---|---|---|---|
| 52 tokens | TTFT | 836 ms | 114 ms | 160 ms |
| 52 tokens | decode tok/s | 20.90 | 26.59 | next table |
| 1594 tokens | TTFT | 16.75 s | 2.04 s | 2.70 s |
| 1594 tokens | decode tok/s | 7.38 | 25.14 | next table |

Decode rate in turn with llama.cpp in one session (larch, llama.cpp, llama.cpp, larch, twice per context).
llama.cpp: `llama-bench -p 0 -n 128 -d <context> -r 2`.

| Context at the first decoded token | larch tok/s | llama.cpp tok/s | larch against llama.cpp |
|---|---|---|---|
| 52 tokens (`-d 52`) | 26.47 | 26.26 | +0.8 % |
| 52 tokens for larch, empty for llama.cpp (`-d 0`) | 26.47 | 26.58 | -0.4 % |
| 1594 tokens (`-d 1594`) | 25.02 | 25.66 | -2.5 % |

## Concurrent conversations

W6 (`larch bench --mode batched`) stopped at its shape check at this commit (#64), so these rows put the W6 shape
through the serve driver: 16 single-turn conversations, a prompt of about 512 tokens (509 to 513, `prompt_tokens` in
the result files) and a 128-token answer each, c closed-loop clients, 16 / c waves. So the prompts are close to W6's
shape, not exactly it.

```
larch bench -m <models>/m1.gguf --mode serve --prompts fixed16.jsonl --concurrency <c> --machine mac --json <file>
llama-batched-bench -m <m1> -ngl 99 -fa 1 -c 16384 -b 2048 -ub 512 -npp 512 -ntg 128 -npl 1,4,8,16
```

`fixed16.jsonl` (sha256 `1f9b7f80bb05...`, in each result file) is not in the repo. Join all first messages of
`bench/prompts/chat.jsonl` with spaces. Conversation i is words 446 i to 446 (i + 1) of that text, so no two prompts
share a prefix:

```python
import json
pool = ' '.join(json.loads(l)['messages'][0]['content'] for l in open('bench/prompts/chat.jsonl')).split()
with open('fixed16.jsonl', 'w') as f:
    for i in range(16):
        f.write(json.dumps({"id": f"fixed-{i:02d}", "messages": [
            {"role": "user", "content": ' '.join(pool[i * 446:(i + 1) * 446])},
            {"role": "assistant", "max_tokens": 128}]}) + '\n')
```

Decode and prefill rate in tok/s, from the first wave (each client's first request). T1 is the last first token of
the wave and T2 its first finish. Only decode steps of all c sequences run from T1 to T2. S_TG is c x steps over
T2 - T1. The sequence admitted first decodes during each later prefill step, so steps is 127 less those prefill
steps. S_PP is the sum of the wave's `prompt_tokens` over T1. T1 also holds those decode rows, so the larch S_PP is
a lower bound. Each larch value is the mean of 2 runs.

| Conversations | S_TG dev before | S_TG larch | S_TG llama.cpp | larch against llama.cpp | S_PP dev before | S_PP larch | S_PP llama.cpp |
|---|---|---|---|---|---|---|---|
| 1 | 13.63 | 26.29 | 26.92 | -2.3 % | 124.2 | 822.9 | 639.3 |
| 4 | 35.88 | 82.19 | 85.41 | -3.8 % | 120.9 | 732.9 | 648.4 |
| 8 | 45.43 | 154.53 | 151.68 | +1.9 % | 119.5 | 703.8 | 639.3 |
| 16 | 43.31 | 191.26 | 168.13 | +13.8 % | 118.5 | 691.4 | 637.0 |

Generated tokens per second over the whole run (`agg_tok_s`). For llama.cpp it is c x 128 over T = T_PP + T_TG.
larch runs prefill and decode in the same steps, and llama-batched-bench runs them one after the other.

| Conversations | dev before | larch | llama.cpp | larch against llama.cpp |
|---|---|---|---|---|
| 1 | 9.51 | 23.40 | 23.04 | +1.6 % |
| 4 | 16.49 | 56.81 | 55.94 | +1.6 % |
| 8 | 18.26 | 82.88 | 77.82 | +6.5 % |
| 16 | 18.09 | 92.49 | 81.79 | +13.1 % |

The 8 larch runs are `mac-m1-larch-fixed16-c<c>-r<1|2>.json` here (schema `larch-bench/1`, 16 requests each, 0
failed). They are not a row of the scoreboard, and `scripts/scoreboard --machine mac` skips them.

## Kernels

`metal_perf` (built with `LARCH_BUILD_TESTS`), idle GPU, m1 shapes, µs per call. llama.cpp is
`test-backend-ops perf -b MTL0` (b11458) on the same shapes and weight formats. n/m = not measured. Peak:
`metal_perf copy --bytes 1073741824 --iterations 16` gives 131.7 GB/s read plus write (the Apple figure is 153 GB/s).

mmvq at 1 row (decode): `metal_perf mmvq --rows 1 --iterations 50`.

| Format, n x k | larch | llama.cpp | llama.cpp time / larch time |
|---|---|---|---|
| Q4_K 4096 x 4096 | 62 | 62 | 1.00x |
| Q4_K 1024 x 4096 | 20 | n/m | n/m |
| Q6_K 1024 x 4096 | 28 | n/m | n/m |
| Q4_K 14336 x 4096 | 232 | 255 | 1.10x |
| Q4_K 4096 x 14336 | 234 | 252 | 1.08x |
| Q6_K 4096 x 14336 | 331 | 368 | 1.11x |
| Q6_K 128256 x 4096 (output) | 3271 | n/m | n/m |

mmvq at 2 to 16 rows (concurrent decode): `metal_perf mmvq --rows <r> --iterations 50`. llama.cpp is `-o MUL_MAT` with
n = rows. From 3 rows, Q4_K and Q6_K run at llama.cpp's speed or faster. At 2 rows larch is up to 10 % slower.

| Format, n x k | Rows | larch | llama.cpp | llama.cpp time / larch time |
|---|---|---|---|---|
| Q4_K 14336 x 4096 | 2 | 270 | 258 | 0.96x |
| Q4_K 14336 x 4096 | 3 | 318 | 314 | 0.99x |
| Q4_K 14336 x 4096 | 4 | 320 | 315 | 0.98x |
| Q4_K 14336 x 4096 | 8 | 320 | 319 | 1.00x |
| Q4_K 14336 x 4096 | 16 | 534 | n/m | n/m |
| Q4_K 4096 x 14336 | 2 | 288 | 260 | 0.90x |
| Q4_K 4096 x 14336 | 4 | 326 | 331 | 1.02x |
| Q4_K 4096 x 14336 | 8 | 326 | 338 | 1.03x |
| Q4_K 4096 x 14336 | 16 | 542 | n/m | n/m |
| Q6_K 4096 x 14336 | 2 | 390 | 398 | 1.02x |
| Q6_K 4096 x 14336 | 4 | 392 | 408 | 1.04x |
| Q6_K 4096 x 14336 | 8 | 394 | 425 | 1.08x |
| Q6_K 4096 x 14336 | 16 | 572 | n/m | n/m |
| Q4_K 4096 x 4096 | 2 | 78 | 75 | 0.95x |
| Q4_K 4096 x 4096 | 4 | 94 | 92 | 0.97x |
| Q4_K 4096 x 4096 | 8 | 96 | 92 | 0.97x |
| Q4_K 4096 x 4096 | 16 | 160 | n/m | n/m |
| Q6_K 128256 x 4096 (output) | 2 | 3285 | n/m | n/m |
| Q6_K 128256 x 4096 (output) | 4 | 3432 | n/m | n/m |
| Q6_K 128256 x 4096 (output) | 8 | 3434 | n/m | n/m |
| Q6_K 128256 x 4096 (output) | 16 | 4949 | n/m | n/m |
| Q8_0 14336 x 4096 | 2 | 461 | n/m | n/m |
| Q8_0 14336 x 4096 | 4 | 460 | n/m | n/m |
| Q8_0 14336 x 4096 | 8 | 565 | n/m | n/m |
| Q8_0 14336 x 4096 | 16 | 769 | n/m | n/m |

Prefill linear at 512 rows (one prefill chunk). Before (`80d1636`): `dequant` into f16, then `gemm`. larch: the fused
`gemm_q` (`metal_perf linear`). llama.cpp: its fused `MUL_MAT`.

| Format, n x k | Before | larch | llama.cpp | llama.cpp time / larch time |
|---|---|---|---|---|
| Q4_K 4096 x 4096 | 1514 | 1336 | n/m | n/m |
| Q4_K 1024 x 4096 | 424 | 430 | n/m | n/m |
| Q4_K 14336 x 4096 | 5125 | 4458 | n/m | n/m |
| Q4_K 4096 x 14336 | 11560 | 5194 | 6539 | 1.26x |
| Q6_K 4096 x 14336 | 10974 | 5389 | 6630 | 1.23x |

Prefill attention (32 q heads, 8 KV heads, head dim 128): `metal_perf attn-prefill`, the last 512-row chunk of a
sequence. Before (`afca4ec`) is the scalar kernel. The tool's q, K and V are zeros, which is the best case of the
kernel's rescale skip. The TTFT above uses real data.

| Positions | Before | larch | Speed-up |
|---|---|---|---|
| 512 | 2182 | 588 | 3.7x |
| 1594 | 11071 | 2531 | 4.4x |
| 2048 | 14863 | 3421 | 4.3x |
| 4096 | 32696 | 7709 | 4.2x |

Decode attention, the same head layout, `--splits 0` (auto). Before is `4e5b8c0`. llama.cpp's `FLASH_ATTN_EXT` at 1
row takes 142 µs at 4096 tokens and 270 µs at 8192, so larch is 1.1x to 1.3x slower there.

| Rows | Tokens | Before | larch | Speed-up |
|---|---|---|---|---|
| 1 | 128 | 130 | 19 | 6.8x |
| 1 | 512 | 589 | 32 | 18x |
| 1 | 2048 | 3155 | 83 | 38x |
| 1 | 4096 | 7368 | 179 | 41x |
| 1 | 8192 | 14804 | 304 | 49x |
| 4 | 512 | 577 | 59 | 9.8x |
| 4 | 8192 | 14550 | 635 | 23x |
| 16 | 128 | 293 | 69 | 4.2x |
| 16 | 2048 | 5174 | 584 | 8.9x |
| 16 | 8192 | 23212 | 2253 | 10x |

`sample_prep`, top 64 without a penalty: `metal_perf sample --rows <r> --vocab <v> --top-k 0 --iterations 50`. Before
is `35d9fa5`. llama.cpp samples on the CPU, so it has no row.

| Vocabulary | Rows | Before | larch | Speed-up |
|---|---|---|---|---|
| 32000 | 1 | 413 | 46 | 9.0x |
| 128256 (m1) | 1 | 1042 | 76 | 13.6x |
| 128256 (m1) | 4 | 1074 | 214 | 5.0x |
| 128256 (m1) | 16 | 3288 | 771 | 4.3x |
| 151936 | 1 | 1189 | 84 | 14.2x |
| 151936 | 16 | 3922 | 901 | 4.4x |

Prefill GEMM on f16 weights (tensor path), a 512-token chunk. m1's prefill takes `gemm_q` above, so these rows matter
for f16 weights.

| m x n x k | Before | larch | larch TFLOP/s |
|---|---|---|---|
| 512 x 14336 x 4096 (ffn gate, up) | 11900 | 3937 | 15.3 |
| 512 x 4096 x 14336 (ffn down) | 15100 | 8600 to 10300 | 5.8 to 7.0 |
| 512 x 4096 x 4096 (q, o) | 2970 | 1132 | 15.2 |
| 512 x 1024 x 4096 (k, v) | 410 | 347 | 12.4 |
| 2048 x 4096 x 4096 | 6000 | 4508 | 15.2 |

## Where a decode token goes

From a scratch build that runs one op twice, chat at 1 row. At a 52-token context a token takes 37.1 ms: mmvq about 36
ms, and the small ops 1.0 ms (attention 0.34, rmsnorm 0.26, rope 0.17, add 0.11). At 1594 tokens attention takes 2.4
ms. Three items stay open. Fewer calls for the small ops give at most 1.3 % and need a contract change (#101). A
2-row mmvq kernel is not written. Attention past 4k tokens needs a split count that the device sizes.

## Parity at the measured commit

`larch parity --device 0 -m <models>/<alias>.gguf --ref <alias>-cublas-ub32.jsonl --bound <b> --margin <m>` at
`e3a5c73`, 100 prompts, every mode passed. Bounds: m8 0.0217, m1 0.0881. `--parallel 8` keeps 8 prompts in decode at
once, which is the only mode that reaches the 2 to 16 row mmvq kernels. Metal has no graph replay.

| Model | Mode | Decisive | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Mixed steps | Result |
|---|---|---|---|---|---|---|---|---|---|
| m8 | prefill | 27,636 of 57,803 | 1.0000 | 0.9902 | 8.827e-05 | 4.347e-04 | 1.793e-02 | 0 | pass |
| m8 | decode | 27,636 of 57,803 | 1.0000 | 0.9903 | 1.075e-04 | 3.836e-04 | 1.999e-02 | 0 | pass |
| m8 | decode `--parallel 8` | 27,636 of 57,803 | 1.0000 | 0.9902 | 8.877e-05 | 3.505e-04 | 1.888e-02 | 92 | pass |
| m8 | chunked | 28,130 of 58,856 | 1.0000 | 0.9902 | 8.824e-05 | 4.293e-04 | 1.640e-02 | 951 | pass |
| m8 | cached | 12,783 of 28,923 | 1.0000 | 0.9890 | 8.787e-05 | 3.454e-04 | 1.772e-02 | 0 | pass |
| m1 | prefill | 12,791 of 58,403 | 1.0000 | 0.9680 | 3.263e-04 | 1.611e-03 | 3.931e-03 | 0 | pass |
| m1 | decode | 12,791 of 58,403 | 1.0000 | 0.9681 | 3.239e-04 | 1.464e-03 | 3.940e-03 | 0 | pass |
| m1 | decode `--parallel 8` | 12,791 of 58,403 | 1.0000 | 0.9679 | 3.283e-04 | 1.658e-03 | 3.864e-03 | 92 | pass |
| m1 | chunked | 12,994 of 59,464 | 1.0000 | 0.9678 | 3.268e-04 | 1.611e-03 | 3.931e-03 | 959 | pass |
| m1 | cached | 4,986 of 29,223 | 1.0000 | 0.9624 | 3.596e-04 | 1.440e-03 | 3.931e-03 | 0 | pass |

## NOT RUN

| What | State |
|---|---|
| Any timing or parity at dev's head | NOT RUN after `e3a5c73`. If a Metal regression shows on dev, run parity at dev first |
| W6, `larch bench --mode batched` | NOT RUN. It was blocked on #64 at `e3a5c73`. #79 and #102 are on dev now |
| The mac rows of `scripts/baseline.py` (W1, W2, W3, W5, W6) | No `mac-*` baseline result is in the repo |
| `scripts/bench-merge --machine mac`, `history.csv` | BLOCKED: it refuses, because mac has no llama.cpp W4 at 8 row |
| Timing of another model | NOT RUN. m8 has parity only |
| Another Mac, and the fallbacks for a GPU without the Metal 4 tensor API | NOT RUN. Only this M5 is measured |
