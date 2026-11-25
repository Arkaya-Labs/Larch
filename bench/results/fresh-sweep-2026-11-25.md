# Fresh sweep against llama.cpp, Ollama and vLLM (CUDA, wsl5070), 25 Nov 2026

dev `488cd91` on the RTX 5070 Ti, against llama.cpp b12723 (`abca692`), Ollama 0.40.18 and vLLM 0.34.0, all the
newest releases when checked at 06:20Z. Models: m1 and m5 at Q4_K_M (all engines), and m13, the 35B MoE (larch vs
llama.cpp on the same file). One unattended run, 06:29Z to 12:18Z. Every GPU job was one exclusive
`flock ~/src/gpu.lock` hold: 109 holds, 103 with exit 0. The 6 failures were 3 holds and their reruns, each caused by a
script defect that was found, fixed and rerun (see FAILED), and every hold is in `fresh-sweep-2026-11-25/holds.log`.
Every result that carries a `failures` field reports 0, 123 of 123. Raw results are in `fresh-sweep-2026-11-25/`, and
the GPU log is `fresh-sweep-2026-11-25-gpu-log.csv`.

## Summary

1. Parity passes on m1 and m5 in all four modes (`--mode all`). The llama.cpp driver gate passes on both models: the
   driver's pooled S is within −1.4 to +1.2 % of llama-batched-bench at every npl, over 9 alternated runs a side.
   m13 has no calibrated bound, so its parity is NOT RUN.
2. **vs llama.cpp, W4** (in process, exact lengths; `compare` accepts 40 of 40 pairs): larch's aggregate is higher at 4,
   8 and 16 concurrent: +6.5 / +17.2 / +31.3 % on m1 and +4.7 / +17.1 / +35.7 % on m5. At 1 it is within noise (−1.3 %,
   −0.3 %). TPOT p99 is 57–64 % lower at 8 and 16.
3. TTFT p99 is still the weak spot vs llama.cpp: +33–35 % at 1, +17–19 % at 8 and +7.2 % on m1 at 16, all outside noise.
   m5 at 16 is within noise. At 4 it is −24.4 % on m1 (better) and within noise on m5. TTFT p50 is 21–45 % lower at
   4–16 and 25 % higher at 1.
4. **W6** fixed pp512/tg128 (pooled S, 9 runs a side): within noise at npl 1 and 4, and 10–16 % below the driver at
   npl 8 and 16. larch's prefill S_pp is flat at 6.0–6.4k tok/s while the driver reaches 7.6–8.3k at npl ≥ 4.
5. **vs Ollama** (both over localhost HTTP, natural EOS): aggregate +46 to +76 % at 8 and 16 concurrent on both
   models, outside noise; TPOT p99 36–68 % lower where outside noise. W1 chat at 1 is within noise on aggregate (+5.9 %,
   +11.4 %), and larch's TPOT is 8–15 % lower there. TTFT p99 is higher on m5's W4 at 16 (+38.4 %); every other Ollama
   TTFT p99 is within noise.
6. **vs vLLM** (m1; vLLM runs AWQ INT4 weights, *different weights* from the GGUF): on W2 chats larch is −6.2 % (at 4)
   and −10.6 % (at 8) in aggregate, with TPOT p99 6–11 % higher. W1 at 1 is within noise on aggregate (−3.1 %).
   TTFT p99 is higher in all three rows (+6 % to +269 %; the W1 figure is one 859 ms cold first turn in larch r1).
   vLLM's own GGUF path is 1.4–4.8× below larch in aggregate on every row. The W4 vLLM rows ran, but `compare` refuses them
   (`n_ctx_seq` 8192 vs 2560, as in October), so they are shown as detail only, not as results.
7. **m13 vs llama.cpp** (same file; llama.cpp with `--n-cpu-moe 3`, searched; larch with dev defaults): a clear loss.
   Aggregate −9.9 % at 1 and −31.6 % at 4; TPOT p99 +2.9 % and +71.1 %; TTFT p99 +68.5 % and +93.9 %. All outside noise.
8. Thermals: max 71 °C; no thermal or HW-slowdown reason (0x20, 0x40, 0x8, 0x80) in any of the 729 samples; no pause.

## For the website (five sentences)

> On one RTX 5070 Ti (16 GB, WSL2) on 25 November 2026, we compared larch (dev `488cd91`) with llama.cpp b12723,
> Ollama 0.40.18 and vLLM 0.34.0, using the same GGUF files for llama.cpp and Ollama and AWQ INT4 weights for vLLM.
> On 200 independent prompts with exact output lengths, larch's aggregate throughput on two 7–8B models was 5–7 %
> above llama.cpp's at 4 concurrent requests, 17 % above at 8 and 31–36 % above at 16, within noise at 1, while its
> p99 time to first token was 7–35 % higher in five of the eight rows. In llama.cpp's fixed pp512/tg128 shape larch
> was within noise at 1 and 4 sequences and 10–16 % slower at 8 and 16. Over localhost HTTP with natural end of
> sequence, larch's aggregate was 46–76 % above Ollama's at 8 and 16 concurrent requests and within noise on a single
> 12-turn chat. Against vLLM with AWQ weights larch was 6–11 % lower in aggregate on multi-turn chats at 4 and 8
> concurrent requests, and on the 35B mixture-of-experts model m13 it was 10 % and 32 % below llama.cpp at 1 and 4.

## Headline (medians; % = larch vs the baseline; "noise" = under 2 % or under either engine's run-to-run spread)

| Row | Baseline | m1 agg tok/s (V vs B) | m1 TTFT p99 | m1 TPOT p99 | m5 agg tok/s (V vs B) | m5 TTFT p99 | m5 TPOT p99 |
|---|---|---|---|---|---|---|---|
| W4 at 1 | llama.cpp | 134.0 vs 135.8, −1.3 % noise | **+33.2 %** | +0.7 % noise | 143.3 vs 143.7, −0.3 % noise | **+35.4 %** | +0.3 % noise |
| W4 at 4 | llama.cpp | 430.5 vs 404.0, **+6.5 %** | **−24.4 %** | **−8.2 %** | 462.4 vs 441.8, **+4.7 %** | −9.8 % noise | **−5.5 %** |
| W4 at 8 | llama.cpp | 662.8 vs 565.7, **+17.2 %** | **+16.8 %** | **−57.3 %** | 734.6 vs 627.2, **+17.1 %** | **+19.4 %** | **−63.0 %** |
| W4 at 16 | llama.cpp | 943.0 vs 718.0, **+31.3 %** | **+7.2 %** | **−57.9 %** | 1057.9 vs 779.8, **+35.7 %** | +3.5 % noise | **−63.9 %** |
| W6 S, npl 1 / 4 / 8 / 16 | llama.cpp driver | +3.0 % noise / −0.2 % noise / **−10.1 %** / **−13.5 %** | | | +4.1 % noise / −2.4 % noise / **−12.8 %** / **−15.9 %** | | |
| W1 chat at 1 | Ollama | 125.1 vs 118.2, +5.9 % noise | +182.2 % noise | **−7.6 %** | 135.8 vs 121.9, +11.4 % noise | +179.7 % noise | **−14.6 %** |
| W2 chats at 8 | Ollama | 547.0 vs 311.7, **+75.5 %** | −81.6 % noise | −56.5 % noise | 642.9 vs 421.9, **+52.4 %** | +16.3 % noise | **−68.0 %** |
| W4 at 8 (ctx 2560) | Ollama | 649.4 vs 444.6, **+46.1 %** | −24.4 % noise | **−47.9 %** | 578.0 vs 396.8, **+45.7 %** | +5.1 % noise | −40.5 % noise |
| W4 at 16 (ctx 2560) | Ollama | 924.2 vs 536.7, **+72.2 %** | −2.1 % noise | **−58.9 %** | 747.2 vs 446.9, **+67.2 %** | **+38.4 %** | **−36.3 %** |

For agg tok/s a positive % is a larch win; for TTFT and TPOT a negative % is. Bold marks a margin outside noise. The
llama.cpp rows are in process on both sides (W4: `larch bench --mode serve` vs `tools/llama-serve-bench`; W6:
`--mode batched` vs the driver's `fixed`). The Ollama rows are HTTP on both sides: larch through the benchmark-only
adapter `examples/serve_bench.rs` with `--engine larch-http --natural-eos`, Ollama through its own server. With natural
EOS the answer lengths differ per engine (generated tokens are in the detail tables), and HTTP rows report chunk gaps,
not token gaps.

m13 (35B MoE, same GGUF) and vLLM (m1; larch in process with dev defaults vs vLLM's OpenAI server, `ignore_eos`):

| Row | Baseline | agg tok/s (V vs B) | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|
| m13 W4 at 1 | llama.cpp (same file, `--n-cpu-moe 3`) | 122.0 vs 135.3, **−9.9 %** | −0.5 % noise | **+68.5 %** | **+9.7 %** | **+2.9 %** |
| m13 W4 at 4 | llama.cpp (same file, `--n-cpu-moe 3`) | 174.6 vs 255.4, **−31.6 %** | **+92.4 %** | **+93.9 %** | **+36.3 %** | **+71.1 %** |
| m1 W1 chat at 1 | vLLM AWQ (different weights) | 124.9 vs 128.8, −3.1 % noise | **+8.4 %** | **+268.9 %** | +0.7 % noise | +0.1 % noise |
| m1 W2 chats at 4 | vLLM AWQ (different weights) | 364.5 vs 388.6, **−6.2 %** | **−2.1 %** | **+111.1 %** | **+6.2 %** | **+6.3 %** |
| m1 W2 chats at 8 | vLLM AWQ (different weights) | 562.5 vs 629.0, **−10.6 %** | **−13.2 %** | **+6.1 %** | **+12.7 %** | **+10.8 %** |

The vLLM rows compare **different weights**: vLLM runs m1's base model in AWQ INT4 (Marlin), larch runs the Q4_K_M
GGUF. vLLM on the same GGUF (plugin path) is in the detail tables; it is 1.4–4.8× below larch in aggregate on every row. Only W1 and
W2 count as results, because `compare` accepted those 9 of 9 pairs. vLLM's W4 rows are shown in the detail only.

**TTFT p99 is the known weak spot** vs llama.cpp and vLLM. In process, larch's TTFT p50 is the lower one at 4–16
concurrent requests, but its tail is longer: a prompt that arrives while others are prefilling waits for the shared
prefill budget. The W6 prefill column shows the same limit: S_pp is flat across npl. In-process token-gap p99 is 56–57 ms
at 4, 80–90 ms at 8 and 84–95 ms at 16, while TPOT p99 is 9.7–19 ms. Counted from the full per-token gaps of the
in-process W4 runs (64,237 per run, before the arrays were stripped), the share of gaps above 40 ms is 1.2–1.3 % at 4,
2.8–3.1 % at 8, 6.1–6.5 % at 16 and 0.00 % at 1 on m1 and m5. Above 100 ms it is 0.00 % everywhere except m1 at 16
(0.05 %). On m13 at 4, 1.47–1.48 % of gaps are above 40 ms, and nearly all of them (1.43 %) are also above 100 ms.

**Host-gap caveat, llama.cpp driver at npl 16.** As in October, `llama-serve-bench` reads every logit row on the host
for its greedy scan, which llama-batched-bench never does. In this run's gate the driver's pooled S is −0.4 % (m1) and
−1.4 % (m5) of llama-batched-bench at npl 16. Read the driver's npl-16 numbers (W6, and W4 at 16) as carrying about
1 % of host overhead that llama.cpp's own loop does not have. Every larch margin outside noise is larger than that.

## Ledger rows

Each row is one model, workload row and baseline, on a headline metric (agg tok/s, TTFT p99, TPOT p99; for W6, S).
Only margins outside noise and at least 2 % are listed. Ties and noise are left out. llama.cpp rows are included
because the driver gate passed (m1 and m5, the same driver binary that ran m13). The gate was not run on m13: its
llama.cpp rows rely on the m1/m5 gate of the same build. vLLM rows come from this sweep only, W1 and W2 only, and use
different weights (AWQ).

| Model | Row | Baseline | Metric | larch vs baseline | Verdict |
|---|---|---|---|---|---|
| m1 | W4 at 1 | llama.cpp | TTFT p99 ms | 345.5 vs 259.4 (+33.2 %) | LOSS |
| m1 | W4 at 4 | llama.cpp | agg tok/s | 430.5 vs 404.0 (+6.5 %) | WIN |
| m1 | W4 at 4 | llama.cpp | TTFT p99 ms | 459.3 vs 607.6 (−24.4 %) | WIN |
| m1 | W4 at 4 | llama.cpp | TPOT p99 ms | 10.5 vs 11.5 (−8.2 %) | WIN |
| m1 | W4 at 8 | llama.cpp | agg tok/s | 662.8 vs 565.7 (+17.2 %) | WIN |
| m1 | W4 at 8 | llama.cpp | TTFT p99 ms | 1039.6 vs 889.8 (+16.8 %) | LOSS |
| m1 | W4 at 8 | llama.cpp | TPOT p99 ms | 14.3 vs 33.4 (−57.3 %) | WIN |
| m1 | W4 at 16 | llama.cpp | agg tok/s | 943.0 vs 718.0 (+31.3 %) | WIN |
| m1 | W4 at 16 | llama.cpp | TTFT p99 ms | 1769.7 vs 1650.8 (+7.2 %) | LOSS |
| m1 | W4 at 16 | llama.cpp | TPOT p99 ms | 19.0 vs 45.2 (−57.9 %) | WIN |
| m1 | W6 fixed pp512/tg128 | llama.cpp driver | S, npl 8 | 2774.7 vs 3086.2 (−10.1 %) | LOSS |
| m1 | W6 fixed pp512/tg128 | llama.cpp driver | S, npl 16 | 3678.7 vs 4250.4 (−13.5 %) | LOSS |
| m1 | W1 chat at 1 | Ollama | TPOT p99 ms | 7.7 vs 8.4 (−7.6 %) | WIN |
| m1 | W2 chats at 8 | Ollama | agg tok/s | 547.0 vs 311.7 (+75.5 %) | WIN |
| m1 | W4 at 8 (ctx 2560) | Ollama | agg tok/s | 649.4 vs 444.6 (+46.1 %) | WIN |
| m1 | W4 at 8 (ctx 2560) | Ollama | TPOT p99 ms | 14.7 vs 28.1 (−47.9 %) | WIN |
| m1 | W4 at 16 (ctx 2560) | Ollama | agg tok/s | 924.2 vs 536.7 (+72.2 %) | WIN |
| m1 | W4 at 16 (ctx 2560) | Ollama | TPOT p99 ms | 20.1 vs 48.8 (−58.9 %) | WIN |
| m5 | W4 at 1 | llama.cpp | TTFT p99 ms | 322.5 vs 238.3 (+35.4 %) | LOSS |
| m5 | W4 at 4 | llama.cpp | agg tok/s | 462.4 vs 441.8 (+4.7 %) | WIN |
| m5 | W4 at 4 | llama.cpp | TPOT p99 ms | 9.7 vs 10.2 (−5.5 %) | WIN |
| m5 | W4 at 8 | llama.cpp | agg tok/s | 734.6 vs 627.2 (+17.1 %) | WIN |
| m5 | W4 at 8 | llama.cpp | TTFT p99 ms | 998.9 vs 836.4 (+19.4 %) | LOSS |
| m5 | W4 at 8 | llama.cpp | TPOT p99 ms | 12.7 vs 34.3 (−63.0 %) | WIN |
| m5 | W4 at 16 | llama.cpp | agg tok/s | 1057.9 vs 779.8 (+35.7 %) | WIN |
| m5 | W4 at 16 | llama.cpp | TPOT p99 ms | 16.9 vs 46.6 (−63.9 %) | WIN |
| m5 | W6 fixed pp512/tg128 | llama.cpp driver | S, npl 8 | 3000.6 vs 3441.5 (−12.8 %) | LOSS |
| m5 | W6 fixed pp512/tg128 | llama.cpp driver | S, npl 16 | 4024.8 vs 4786.7 (−15.9 %) | LOSS |
| m5 | W1 chat at 1 | Ollama | TPOT p99 ms | 6.9 vs 8.1 (−14.6 %) | WIN |
| m5 | W2 chats at 8 | Ollama | agg tok/s | 642.9 vs 421.9 (+52.4 %) | WIN |
| m5 | W2 chats at 8 | Ollama | TPOT p99 ms | 12.6 vs 39.4 (−68.0 %) | WIN |
| m5 | W4 at 8 (ctx 2560) | Ollama | agg tok/s | 578.0 vs 396.8 (+45.7 %) | WIN |
| m5 | W4 at 16 (ctx 2560) | Ollama | agg tok/s | 747.2 vs 446.9 (+67.2 %) | WIN |
| m5 | W4 at 16 (ctx 2560) | Ollama | TTFT p99 ms | 1820.5 vs 1315.5 (+38.4 %) | LOSS |
| m5 | W4 at 16 (ctx 2560) | Ollama | TPOT p99 ms | 28.5 vs 44.7 (−36.3 %) | WIN |
| m13 | W4 at 1 | llama.cpp | agg tok/s | 122.0 vs 135.3 (−9.9 %) | LOSS |
| m13 | W4 at 1 | llama.cpp | TTFT p99 ms | 722.3 vs 428.6 (+68.5 %) | LOSS |
| m13 | W4 at 1 | llama.cpp | TPOT p99 ms | 7.6 vs 7.4 (+2.9 %) | LOSS |
| m13 | W4 at 4 | llama.cpp | agg tok/s | 174.6 vs 255.4 (−31.6 %) | LOSS |
| m13 | W4 at 4 | llama.cpp | TTFT p99 ms | 2262.6 vs 1166.9 (+93.9 %) | LOSS |
| m13 | W4 at 4 | llama.cpp | TPOT p99 ms | 30.6 vs 17.9 (+71.1 %) | LOSS |
| m1 | W1 at 1 | vLLM AWQ | TTFT p99 ms | 532.0 vs 144.2 (+268.9 %) | LOSS |
| m1 | W2 at 4 | vLLM AWQ | agg tok/s | 364.5 vs 388.6 (−6.2 %) | LOSS |
| m1 | W2 at 4 | vLLM AWQ | TTFT p99 ms | 341.4 vs 161.7 (+111.1 %) | LOSS |
| m1 | W2 at 4 | vLLM AWQ | TPOT p99 ms | 10.6 vs 10.0 (+6.3 %) | LOSS |
| m1 | W2 at 8 | vLLM AWQ | agg tok/s | 562.5 vs 629.0 (−10.6 %) | LOSS |
| m1 | W2 at 8 | vLLM AWQ | TTFT p99 ms | 372.2 vs 350.9 (+6.1 %) | LOSS |
| m1 | W2 at 8 | vLLM AWQ | TPOT p99 ms | 15.2 vs 13.7 (+10.8 %) | LOSS |

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (16303 MiB), sm_120, VBIOS 98.03.58.40.8b, power limit 300 W, max SM clock 3090 MHz |
| Driver | 617.14 (Windows host; CUDA UMD 13.4) |
| CPU, RAM | AMD Ryzen 7 9800X3D, 8 logical CPUs and 23 GiB visible to WSL |
| OS | Ubuntu 26.04.1 LTS on WSL2, kernel 6.18.33.2-microsoft-standard-WSL2 |
| larch | dev `488cd912432011773a432e706cee5223397fe402`. Release build `cargo build --release --locked -p larch --no-default-features --bin larch --example serve_bench`: CUDA backend, the default kernel list, rustc 1.99.0, nvcc 13.3.73, g++ 14.3.0. `larch` sha256 `6d683e09aea3…`, equal to `build` in all 121 JSONs it wrote (in process, plus the bench client of every HTTP run); `serve_bench` sha256 `75488ee2d587…` |
| Engine settings | dev defaults (no `LARCH_*` variable set). m13: dev's own MoE placement (`larch bench` does not log the split it chose) |
| llama.cpp | b12723 (`abca6920035e`, clean checkout of the tag), `-DGGML_CUDA=ON`, GGML_NATIVE (120a-real), via `baseline.py --llama-src`; `llama-serve-bench` build `d28ab07fd41b…` (this branch's driver fixes, see below). Searched flags (`baseline.py flags`): m1 `-t 4 -b 2048 -ub 1024`, m5 `-t 8 -b 2048 -ub 1024`, m13 `-t 4 -b 1024 -ub 1024 --n-cpu-moe 3`, all with `-ngl 99 -fa on -ctk f16 -ctv f16` |
| Ollama | 0.40.18 (its bundled llama-server, CUDA backend), its own model store; `OLLAMA_NUM_PARALLEL=N`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_KEEP_ALIVE=-1` (`baseline.py ollama-serve`), on port 11435 |
| vLLM | 0.34.0, torch 2.14.0, transformers 5.22.2, vllm-gguf-plugin 0.0.5 with gguf 0.19.0, Python 3.12.14, uv 0.12.10. AWQ: m1's base weights in AWQ INT4 from the hugging-quants organization, revision `db1f81ad4b8c7e39777509fac66c652eb0a52f91` (October's), MarlinLinearKernel, FLASH_ATTN. GGUF: `m1.gguf` with `--tokenizer <awq>`, the plugin's GGUFModelLoader, FLASH_ATTN |
| Models | m1, m5, m13 (`bench/models.toml`); sha256 checked against `models.toml` by `baseline.py models` |
| Parity references | b11458 cuBLAS ub32 dumps, sha256 `f0e5ce16…` (m1) and `7c2f92f4…` (m5), equal to `bench/oracle-bounds.json` (see the header note under Parity) |

vLLM, from its logs: free memory was 14.56 of 15.89 GiB at all 12 starts, and `--gpu-memory-utilization 0.90` was
accepted every time. KV cache: AWQ 57,344 tokens (53,760 on the first start); GGUF 41,984. Weights in memory: AWQ
5.39 GiB, GGUF 7.19 GiB. The one ERROR line (`AsyncLLM output_handler failed`, in the GGUF W4 server's log) comes at
shutdown, after the engine had stopped, from the window's kill after the last W4 row, which had completed with
0 failures.

## Protocol

- **Parity** (once per model): `larch parity --mode all` with the bound and margin from `bench/oracle-bounds.json`.
- **Driver gate**: `baseline.py run --row fixed`, which alternates llama-batched-bench and the driver 9 times in one
  hold. `baseline.check_fixed` compares pooled S (total tokens over total time) and must be within 3 %.
- **W4 vs llama.cpp**: chat.jsonl, 200 prompts, `ignore_eos`, per-sequence context 2560 (`--set max_context=2560`;
  the driver's `n_ctx_seq`). Each of c = 1, 4, 8 and 16 ran V-L-L-V, one hold per run. `compare` accepted all 4 pairs
  of every row (32 of 32 on m1/m5, 8 of 8 on m13).
- **W6**: per model, one hold alternating `larch bench --mode batched` and the driver's `fixed` 9 times. Pooled S per
  npl.
- **vs Ollama**: W1 chat at 1 (context 8192), W2 chats at 8 (8192), and W4 at 8 and at 16 (`--set max_context=2560`
  on both engines). Each ran V-O-O-V-V-O, one hold per run, with a fresh server every run on both sides. `serve_bench`
  got `--context` equal to the row's, `--sequences N` and `--requests` = turns + 1 warm-up (13, 65, 201), and exited by
  itself after its last request in all 24 runs. `compare` refuses these pairs by design: natural-EOS lengths are off
  the manifest, and both engines report `cache_prompt: true` on W4 (checked on m1 W1 r1 and W4 at 8 r1; the output is
  quoted in NOT RUN). The Ollama rows are therefore governed by the brief's HTTP protocol, as in October.
  Cached prompt tokens on W1/W2 match to within 7 tokens (m1 W1 22,072 larch vs 22,073 Ollama; m1 W2 81,785 vs
  81,792; m5 equal). On W4, Ollama served 5,541–5,778 of 112,624 prompt tokens from its cache on m1 and 4,427–4,624 on
  m5, while larch-http served 0 (its prefix cache is off by default). This favours Ollama; no verdict changes.
- **vs vLLM** (m1): windows in October's order, AWQ r1, larch r1, AWQ r2, larch r2, GGUF r1, one hold each, with no
  vLLM process alive during a larch window. W1 and W2 at 4 and 8 run on a fresh server each with prefix caching on; W4
  at 1, 4, 8 and 16 run on one server with `--disable-prefix-caching`. Prompts go as token ids, and every turn checks
  `usage.prompt_tokens` against the manifest. `compare` accepted 9 of 9 W1/W2 pairs (AWQ r1, AWQ r2 and GGUF r1 against
  larch r1, r2 and r1). It refused every W4 pair tried (c1 and c8 of each engine run): `n_ctx_seq differs: 8192 vs 2560`.
- **m13 llama.cpp search** (one hold, `d-search.sh`): a binary search for the fewest MoE layers on the CPU
  (`LLAMA_ARG_N_CPU_MOE`, llama.cpp's env binding of `--n-cpu-moe`) at which the driver runs W4 at 4's largest shape
  (4 × (2048 + 128) in `-c 10240`) **and** the CUDA0 buffers llama.cpp reports sum below the VRAM free before the probe.
  The second test is needed because the Windows driver's system-memory fallback can let an oversized model "run". Then
  baseline.py's `-t/-b/-ub` rule (max √(pp2048·tg128) over `-t 4,8 -b 512,1024,2048 -ub ≤ b`) at that count, and a
  re-probe with the chosen flags. Result: n_cpu_moe = 3 (2 aborted with exit 134; 1 "ran" with 15,841 MiB of CUDA0
  buffers against 15,267 MiB free, so it did not fit). With `-t 4 -b 1024 -ub 1024` the re-probe fit at 15,618 of
  15,713 MiB. The whole search is in `wsl5070-m13-llama-search.txt`.
- **Statistics**: per row and engine, the median [min–max] over runs of each run's summary value. Spread =
  (max − min) / median. A margin is "within noise" when it is under 2 % or under either engine's spread. With 2 runs
  the median is their mean. October's vLLM report used a flat 2 % tie rule instead; this sweep uses the spread rule
  everywhere.

### Differences from October's protocol

1. Newer engines everywhere: llama.cpp b11458 → b12723, Ollama 0.40.1 → 0.40.18, vLLM 0.32.0 → 0.34.0. The machine's
   system Ollama service (0.40.16, port 11434) could not be stopped without sudo. It stayed idle with no model loaded
   (`/api/ps` was logged before every Ollama run), and ours served on port 11435 from its own model store.
2. W4 at 4 vs llama.cpp is added (October ran 1, 8 and 16).
3. m13 (Phase D) is new.
4. No cache-on rows (not in this sweep's brief).
5. vLLM's prefix-cache flag is `--disable-prefix-caching` in 0.34 (0.32's `--no-enable-prefix-caching` was removed).
   The client reads `enable_prefix_caching` from the server, so each result's `cache_prompt` shows what the server
   did: false on every W4 run, true on W1/W2.
6. vLLM's vocabulary check (`POST /tokenize` on 20 prompts) was not repeated. The prompts go as token ids, every turn
   checks the prompt-token count, and the tokenizer files are October's revision, which passed 20 of 20.
7. Parity read header-renamed copies of the reference dumps (see Parity).
8. The hold log and the nvidia-smi throttle counters are committed this time.

## Parity (larch `6d683e09`, CUDA device 0, graphs on, 100 prompts)

| Model | Mode | Decisive / positions | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Bound | Result |
|---|---|---|---|---|---|---|---|---|---|
| m1 | prefill | 12791/58403 | 0.9999 | 0.9557 | 1.617e-03 | 3.045e-02 | 0.2584 | 0.0881 | pass |
| m1 | decode | 12791/58403 | 1.0000 | 0.9678 | 3.248e-04 | 1.583e-03 | 0.003947 | 0.0881 | pass |
| m1 | chunked | 12994/59464 | 0.9923 | 0.9484 | 1.829e-03 | 2.878e-02 | 0.2595 | 0.0881 | pass |
| m1 | cached | 4986/29223 | 1.0000 | 0.9487 | 6.911e-04 | 2.539e-03 | 0.02925 | 0.0881 | pass |
| m5 | prefill | 9735/57803 | 0.9999 | 0.9353 | 2.121e-03 | 2.390e-02 | 0.4326 | 0.0653 | pass |
| m5 | decode | 9735/57803 | 1.0000 | 0.9644 | 5.489e-04 | 2.648e-03 | 0.01694 | 0.0653 | pass |
| m5 | chunked | 9937/58855 | 1.0000 | 0.9262 | 3.079e-03 | 2.762e-02 | 0.5406 | 0.0653 | pass |
| m5 | cached | 4529/28923 | 0.9998 | 0.9378 | 1.621e-03 | 6.449e-03 | 0.9956 | 0.0653 | pass |

Runs took 8 min 8 s (m1) and 7 min 35 s (m5); the raw output is in `wsl5070-<alias>-larch-parity.txt`. Compared with
October, m1 prefill/chunked decisive top-1 rose (0.9990 → 0.9999, 0.9922 → 0.9923), and both models' cached-mode
max KL moved: m1 0.076 → 0.029, m5 0.302 → 0.996. m5 cached's max KL is now near 1. Its mean (1.55e-3 → 1.62e-3)
and p99 (6.52e-3 → 6.45e-3) are about where they were in October, and the mode passes; the max is one tail value,
and which position it comes from was not examined. Engine and parity bounds are
unchanged.

**Reference header.** The dumps in the oracle directory still begin with the pre-rename key `{"vivace_oracle":1,`, and
`crates/larch-model/src/oracle.rs` accepts only `larch_oracle` (the same format, version 1). As a result,
`larch parity` exited 2 ("not run") twice. The sweep read copies in which only line 1's key was renamed
(`sed '1s/^{"vivace_oracle":1,/{"larch_oracle":1,/'`); `cmp` confirms every other byte is identical. The sources'
sha256 equal `oracle-bounds.json`, and the copies hash `994f3df8…` (m1) and `85b54ccc…` (m5), which is the
`dump_sha256` each parity output records. No engine code and no file in the oracle directory changed. Whether to
rename the dumps' key or accept both keys is the maintainer's call.

## Driver gate (llama-batched-bench vs llama-serve-bench fixed, S over 9 alternated runs a side)

| Model | npl | llama-batched-bench S | driver S | driver vs batched-bench | gate (3 %) |
|---|---|---|---|---|---|
| m1 | 1 | 619.96 (n=9) | 627.22 (n=9) | +1.2 % | pass |
| m1 | 4 | 1915.36 (n=9) | 1937.37 (n=9) | +1.1 % | pass |
| m1 | 8 | 3099.78 (n=9) | 3087.32 (n=9) | −0.4 % | pass |
| m1 | 16 | 4277.07 (n=9) | 4258.90 (n=9) | −0.4 % | pass |
| m5 | 1 | 662.58 (n=9) | 663.09 (n=9) | +0.1 % | pass |
| m5 | 4 | 2123.36 (n=9) | 2121.20 (n=9) | −0.1 % | pass |
| m5 | 8 | 3450.72 (n=9) | 3441.66 (n=9) | −0.3 % | pass |
| m5 | 16 | 4887.22 (n=9) | 4818.71 (n=9) | −1.4 % | pass |

## Temperature and clocks (`fresh-sweep-2026-11-25-gpu-log.csv`, 729 samples every 30 s, 06:17Z–12:21Z, local time UTC−6)

- Max core temperature: 71 °C at 06:49:50Z, during the m5 llama-bench flag search. Phase maxima: gate/flags 71,
  A 69, B 67, C 70, D 56 °C (D1 search 53). No sample reached 80 °C, and the 85 °C pause never triggered.
- **Reliability (0x400)** was active in 677 of 729 samples, including idle ones. It is the voltage-reliability clock
  limiter, not a thermal limiter. SW thermal (0x20), HW thermal (0x40), HW slowdown (0x8) and power brake (0x80) never
  appeared.
- **SW power cap (0x4)** was active in 52 samples. They fall in larch W4 at 4 (m1 10, m5 6), three Phase C
  windows (vLLM GGUF r1 11, larch r1 5, larch r2 5), llama W4 at 1 (m1 3, m5 1), the flag searches and gates
  (6), m5 W6 (2), m5 llama W4 at 8 (1) and m5 Ollama W4 at 16 (2). Max board power was 281.8 W of 300.
- nvidia-smi's cumulative counters (`throttle-counters-{start,end}.txt`) are identical at the start and the end:
  SW power capping 219,118 µs, all thermal counters 0. They did not advance under WSL even while 52 samples showed
  0x4, so this report relies on the per-sample masks.
- **No row was thermally throttled.**

## FAILED

None. Three holds failed on their first run and on their rerun. Each failure was a defect in this sweep's own
scripts, root-caused with systematic-debugging, fixed in the script, and rerun from scratch. No engine, driver or
bench rule was changed for them.

| Hold | Symptom | Root cause | Fix |
|---|---|---|---|
| H-parity-m1 (06:29Z) | `larch parity` exit 2, "not run" | the reference dumps' header key is the pre-rename `vivace_oracle` | header-renamed copies (see Parity) |
| B-m1-w1-V-r1 (08:46Z) | 1 of 12 requests "Connection refused"; serve_bench "request: Broken pipe" | the window script's TCP readiness probe was accepted by `serve_bench` and counted toward `--requests`, so it exited one request early | wait for serve_bench's own `ready` line. Phase B was restarted; the in-flight Ollama r1 hold was interrupted and discarded (no result written), then rerun |
| D1-m13-search (11:13Z) | probe at n_cpu_moe = 41 ran (S 384.55) but read "CUDA0 buffers 0 MiB" | b12723 maps library INFO messages to verbosity 4 (TRACE) above the default threshold 3, so the `buffer size` lines the fit check sums are no longer printed | the probe passes `-lv 4` (logging only). Checked with one probe under the lock (2,358 MiB summed), then rerun |

## Driver and script changes on this branch

1. `tools/llama-serve-bench/CMakeLists.txt` always recorded `llama_cpp.tag = b11458`, even when built from another
   checkout with `--llama-src`. A local checkout must now sit on a tag; that tag is recorded (`-dirty` for local
   edits), and the build reconfigures when the checkout's HEAD moves. This sweep's results record `b12723`, `abca692`.
2. `tools/llama-serve-bench/main.cpp`: in b12723, `common_chat_params::prompt` is a `common_chat_prompt` with `str()`,
   where b11458 has a std::string. A two-overload `prompt_text()` builds against both. The bench rule is unchanged: the
   whole rendered string is tokenised with parse_special. The manifests regenerated with the driver built against
   b11458 and against b12723 are byte-identical to `bench/manifest/` for m1, m5 and m13 (6 of 6).
3. `scripts/baseline.py`: the `--llama-src` help and docstring say any tag is built and recorded.
4. `bench/manifest/m13.jsonl` (new, 338 lines): `baseline.py --alias m13 manifest`. m13's W4 per-sequence context is
   2560 (peak 2358), the same as m1/m5.

## NOT RUN

| What | Why |
|---|---|
| m13 parity | `bench/oracle-bounds.json` has no m13 entry (no calibrated bound or margin) |
| Driver gate on m13 | not in the brief; m13's llama.cpp rows rely on the m1/m5 gate of the same driver build |
| vLLM W4 as results | they ran (detail below), but `compare` refuses every W4 pair: `n_ctx_seq differs: 8192 vs 2560` |
| `compare` on the Ollama rows | refused by design: m1 W1 r1 "hash or length off the manifest" (natural EOS); m1 W4 at 8 r1 "not one of wsl5070's rows (chat.jsonl at 8, cache_prompt True)" |
| vLLM `/tokenize` vocabulary check | replaced by token-id prompts plus the per-turn prompt-token check (see protocol differences) |
| m13 larch placement record | `larch bench` does not print the MoE split it chose; finding it would need an engine log flag, out of scope |
| W2 at 4 / W5 / W7 vs llama.cpp or Ollama, Ollama or vLLM on m5/m13, cache-on rows | not part of this sweep's brief |
| `baseline.py check` row completeness | it expects the machine's full row set under baseline.py's own file names; this sweep runs the brief's subset with October's names. Its gate part passes on the committed files (the 8 lines above) |
| `cargo fmt`, `clippy`, `cargo test` | no Rust changed |
| `scripts/bench-merge` | not a merge; it would append to `history.csv` |
| The `tune` feature in the measured binary | built with `--no-default-features`, as in October |

## Changed since October

Both sides changed between the sweeps (larch dev `4a540ae` → `488cd91`; each baseline's version, above), so a shifted
margin is not attributed to one side. In absolute numbers, from the two reports:

| Row | larch Oct → now | baseline Oct → now | margin Oct → now |
|---|---|---|---|
| m1 W4 at 1 vs llama.cpp | 123.9 → 134.0 | 129.0 → 135.8 | −4.0 % → −1.3 % noise |
| m1 W4 at 8 vs llama.cpp | 572.2 → 662.8 | 507.9 → 565.7 | +12.7 % → +17.2 % |
| m1 W4 at 16 vs llama.cpp | 777.8 → 943.0 | 676.0 → 718.0 | +15.1 % → +31.3 % |
| m5 W4 at 1 vs llama.cpp | 135.2 → 143.3 | 137.9 → 143.7 | −2.0 % noise → −0.3 % noise |
| m5 W4 at 8 vs llama.cpp | 621.0 → 734.6 | 558.6 → 627.2 | +11.2 % → +17.1 % |
| m5 W4 at 16 vs llama.cpp | 860.7 → 1057.9 | 771.1 → 779.8 | +11.6 % → +35.7 % |
| W4 TTFT p99 vs llama.cpp, m1 at 1 / 8 / 16 | | | +125 / +46 / +40 % → +33 / +17 / +7 % |
| W4 TTFT p99 vs llama.cpp, m5 at 1 / 8 / 16 | | | +108 / +39 / +16 % → +35 / +19 / +3.5 % noise |
| W6 S vs driver, m1 npl 4 / 8 / 16 | | | −13.8 / −10.4 / −16.4 % → −0.2 % noise / −10.1 / −13.5 % |
| W6 S vs driver, m5 npl 4 / 8 / 16 | | | −13.5 / −12.9 / −19.9 % → −2.4 % noise / −12.8 / −15.9 % |
| W6 larch prefill S_pp | 4.9–5.3k | | 6.0–6.4k |
| m1 W1 vs Ollama | 103.6 → 125.1 | 116.0 → 118.2 | −10.8 % → +5.9 % noise; TTFT p50 +147 % → +12 % |
| m5 W1 vs Ollama | 117.6 → 135.8 | 124.4 → 121.9 | −5.4 % noise → +11.4 % noise; TTFT p50 +104 % → +1.6 % noise |
| m1 W2 at 8 vs Ollama | 392.6 → 547.0 | 337.4 → 311.7 | +16.4 % → +75.5 % |
| m5 W2 at 8 vs Ollama | 446.2 → 642.9 | 407.5 → 421.9 | +9.5 % noise → +52.4 % |
| m1 W4 at 8 / 16 vs Ollama | 558.3 / 756.9 → 649.4 / 924.2 | 426.2 / 509.5 → 444.6 / 536.7 | +31.0 / +48.6 % → +46.1 / +72.2 % |
| m5 W4 at 8 / 16 vs Ollama | 463.2 / 584.1 → 578.0 / 747.2 | 385.3 / 409.8 → 396.8 / 446.9 | +20.2 / +42.5 % → +45.7 / +67.2 % |
| m1 W1 vs vLLM AWQ | 120.1 → 124.9 | 120.2 → 128.8 | −0.1 % → −3.1 % noise |
| m1 W2 at 4 vs vLLM AWQ | 320.9 → 364.5 | 384.1 → 388.6 | −16.5 % → −6.2 % |
| m1 W2 at 8 vs vLLM AWQ | 556.3 → 562.5 | 598.4 → 629.0 | −7.0 % → −10.6 % |

The W1 cold-first-turn caveat: the first run of each model per engine has one slow turn 0 (Ollama W1 r1 6,764 ms
on m1 and 6,935 ms on m5; Ollama W2 at 8 r1 3,167 / 5,864 ms; larch-http m1 W1 r1 3,333 ms; larch-v W1 r1 859 ms).
Every later run's turn 0 takes 100–420 ms. The likely cause is the first touch of the weights after the warm-up. The
outliers are kept, per protocol (no selective reruns), and they set the W1 spreads; that is why the W1 aggregates and
TTFT p99 vs Ollama are within noise.

## Exact commands

`T=~/src/target/sweep-nov25`, `M=<models>` (the `<alias>.gguf` directory), `B=<llama-build>`, `R=<raw>`,
`BL="python3 scripts/baseline.py --machine wsl5070 --models $M --build $B --out $R"`. Every GPU job ran as
`timeout <wait+job> flock -w 14400 ~/src/gpu.lock timeout <job> <command>`, one hold per line below unless noted.

```
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv -l 30 > gpu-log.csv
nvidia-smi -q -d PERFORMANCE > throttle-counters-{start,end}.txt
CARGO_TARGET_DIR=$T CARGO_BUILD_JOBS=8 nice -n 19 cargo build --release --locked -p larch --no-default-features --bin larch --example serve_bench
nice -n 19 $BL --llama-src <llama.cpp-b12723 checkout> build
$BL models; $BL --alias m13 models; $BL --alias m13 manifest

$T/release/larch parity -m $M/m1.gguf --ref <m1-cublas-ub32.jsonl> --bound 0.0880593441491973 --margin 0.703125 --mode all
$T/release/larch parity -m $M/m5.gguf --ref <m5-cublas-ub32.jsonl> --bound 0.06534985856703847 --margin 1.234375 --mode all

$BL --alias <m> flags
$BL --alias <m> --row fixed run    # the gate

# W4 vs llama.cpp, c in 1 4 8 16, V-L-L-V
$T/release/larch bench -m $M/<m>.gguf --mode serve --concurrency <c> --set max_context=2560 --machine wsl5070 --json $R/wsl5070-<m>-larch-w4-<c>-r<k>.json
$BL --alias <m> --row w4-<c> run    # renamed to wsl5070-<m>-llama-w4-<c>-r<k>.json
python3 scripts/baseline.py --machine wsl5070 compare $R/wsl5070-<m>-llama-w4-<c>-r<i>.json $R/wsl5070-<m>-larch-w4-<c>-r<j>.json    # i, j in 1 2

# W6, one hold per model: 9 x (larch, driver)
$T/release/larch bench -m $M/<m>.gguf --mode batched --pp 512 --tg 128 --npl 1,4,8,16 --machine wsl5070 --json $R/wsl5070-<m>-larch-w6-r<i>.json
$B/bin/llama-serve-bench fixed -m $M/<m>.gguf -ngl 99 -fa on -ctk f16 -ctv f16 -t <t> -b <b> -ub <ub> -npp 512 -ntg 128 -npl 1,4,8,16 -c 16384 --machine wsl5070 --json $R/wsl5070-<m>-llama-w6-r<i>.json

# Ollama models, one hold
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=<store> PATH=<ollama-0.40.18>/bin:$PATH $BL --alias m1 --alias m5 ollama

# vs Ollama, per row V-O-O-V-V-O; each run one hold with its own server
$T/release/examples/serve_bench --model $M/<m>.gguf --context <ctx> --sequences <N> --requests <turns+1> --port 11436 &   # wait for its "ready" line
$T/release/larch bench -m $M/<m>.gguf --engine larch-http --url http://127.0.0.1:11436 --natural-eos <row> --concurrency <N> --machine wsl5070 --json $R/wsl5070-<m>-larch-http-<row>-r<k>.json
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=<store> PATH=<ollama-0.40.18>/bin:$PATH python3 scripts/baseline.py --machine wsl5070 --parallel <N> ollama-serve &
$T/release/larch bench -m $M/<m>.gguf --engine ollama --url http://127.0.0.1:11435 <row> --concurrency <N> --machine wsl5070 --json $R/wsl5070-<m>-ollama-<row>-r<k>.json
#   w1:    --workload chat                        N 1,  ctx 8192, 13 requests
#   w2-8:  --workload chats                       N 8,  ctx 8192, 65 requests
#   w4-8:  --mode serve --set max_context=2560    N 8,  ctx 2560, 201 requests
#   w4-16: --mode serve --set max_context=2560    N 16, ctx 2560, 201 requests

# vs vLLM (m1): one hold per window, AWQ r1, larch r1, AWQ r2, larch r2, GGUF r1
# environment: HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TORCHINDUCTOR_COMPILE_THREADS=4
vllm serve <awq> --port 8000 --max-model-len 8192 --seed 0 --enable-prompt-tokens-details --gpu-memory-utilization 0.90 [--disable-prefix-caching]
vllm serve $M/m1.gguf --tokenizer <awq> --port 8000 --max-model-len 8192 --seed 0 --enable-prompt-tokens-details --gpu-memory-utilization 0.90 [--disable-prefix-caching]
$T/release/larch bench -m $M/m1.gguf --engine openai --url http://127.0.0.1:8000 --machine wsl5070 --workload chat --concurrency 1 --json <out>     # own server
$T/release/larch bench -m $M/m1.gguf --engine openai --url http://127.0.0.1:8000 --machine wsl5070 --workload chats --concurrency 4 --json <out>    # then 8, own server
$T/release/larch bench -m $M/m1.gguf --engine openai --url http://127.0.0.1:8000 --machine wsl5070 --mode serve --concurrency 1,4,8,16 --json <out>   # --disable-prefix-caching server
$T/release/larch bench -m $M/m1.gguf --machine wsl5070 --workload chat --concurrency 1 --json $R/wsl5070-m1-larch-v-w1-r<k>.json
$T/release/larch bench -m $M/m1.gguf --machine wsl5070 --workload chats --concurrency 4,8 --json $R/wsl5070-m1-larch-v-w2-r<k>.json
$T/release/larch bench -m $M/m1.gguf --machine wsl5070 --mode serve --concurrency 1,4,8,16 --set max_context=2560 --json $R/wsl5070-m1-larch-v-w4-r<k>.json
python3 scripts/baseline.py --machine wsl5070 compare <vllm.json> <larch.json>

# m13 (Phase D): one search hold, then W4 at 1 and 4, V-L-L-V
LLAMA_ARG_N_CPU_MOE=<n> $B/bin/llama-serve-bench fixed -lv 4 -m $M/m13.gguf -ngl 99 -fa on -ctk f16 -ctv f16 [-t T -b B -ub UB] -npp 2048 -ntg 128 -npl 4 -c 10240 --machine wsl5070 --json <probe>
$B/bin/llama-bench -m $M/m13.gguf -ngl 99 -ncmoe 3 -fa on -ctk f16 -ctv f16 -t 4,8 -b <b> -ub <ub ≤ b> -p 2048 -n 128 -o csv    # b in 512 1024 2048
$T/release/larch bench -m $M/m13.gguf --mode serve --concurrency <c> --set max_context=2560 --machine wsl5070 --json $R/wsl5070-m13-larch-w4-<c>-r<k>.json
LLAMA_ARG_N_CPU_MOE=3 $BL --alias m13 --row w4-<c> run    # renamed to wsl5070-m13-llama-w4-<c>-r<k>.json
```

The committed JSONs have the per-request gap arrays (`token_gaps_ms`, `http_chunk_gaps_ms`) removed (116 MB → 13 MB).
Every summary, including the token-gap and chunk-gap percentiles, is unchanged. In the committed copies, the local
model paths were reduced to `<alias>.gguf`, and vLLM's AWQ directory (its served model id) to `<awq>`. No other file
was edited after its run. `compare` accepts the stripped, redacted copies: 40 of 40 W4 llama.cpp pairs and 9 of 9
vLLM W1/W2 pairs. `compare` prints only the first 10 characters of each file stem, so m13 rows show as
`wsl5070-m1`: a display truncation, not a mix-up.

`started_at` in the llama.cpp JSONs is the run's end time, not its start, as in October.

## Files (`fresh-sweep-2026-11-25/`)

`wsl5070-<alias>-<engine>-<row>-r<k>.json`, with engine `larch` (in process), `llama` (llama-serve-bench),
`larch-http`, `ollama`, `vllm-awq`, `vllm-gguf` or `larch-v` (larch in process in the vLLM windows), and row `w4-<c>`,
`w6`, `w1`, `w2-8` (vLLM rows: `w1`, `w2-4`, `w2-8`, `w4-r<k>-c<c>`; larch-v: `w1`, `w2-r<k>-c<c>`, `w4-r<k>-c<c>`). The
gate is `wsl5070-<alias>-batched.jsonl` and `-fixed.json`, the flag search is `wsl5070-<alias>-llama-flags.{csv,json}`
(m13: `-llama-flags.json` and `-llama-search.txt`), and parity is `wsl5070-<alias>-larch-parity.txt`. `holds.log` lists
every hold with its UTC start, end and exit code. 161 result JSONs in total.

## Detail (generated from the raw JSON; spread V / B = larch / baseline)

### m1 detail

#### m1 W4 at 1: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 134.0 [133.9-134.1] | 135.8 [135.5-136.1] | -1.3% (spread V 0.1%, B 0.4%): within noise |
| TTFT p50 ms | 69.0 [68.9-69.1] | 55.3 [55.2-55.3] | +24.8% (spread V 0.3%, B 0.2%): larch worse |
| TTFT p99 ms | 345.5 [345.5-345.5] | 259.4 [259.3-259.5] | +33.2% (spread V 0.0%, B 0.1%): larch worse |
| TPOT p50 ms | 7.08 [7.08-7.09] | 7.10 [7.09-7.11] | -0.3% (spread V 0.1%, B 0.2%): within noise |
| TPOT p99 ms | 7.52 [7.52-7.52] | 7.47 [7.44-7.50] | +0.7% (spread V 0.0%, B 0.8%): within noise |
| token-gap p99 ms | 7.61 [7.61-7.61] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m1 W4 at 4: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 430.5 [430.4-430.6] | 404.0 [399.1-409.0] | +6.5% (spread V 0.1%, B 2.5%): larch better |
| TTFT p50 ms | 76.4 [74.8-78.1] | 96.5 [89.8-103.2] | -20.8% (spread V 4.4%, B 13.9%): larch better |
| TTFT p99 ms | 459.3 [430.0-488.7] | 607.6 [594.2-620.9] | -24.4% (spread V 12.8%, B 4.4%): larch better |
| TPOT p50 ms | 8.76 [8.76-8.76] | 9.41 [9.31-9.51] | -6.9% (spread V 0.1%, B 2.2%): larch better |
| TPOT p99 ms | 10.53 [10.52-10.54] | 11.47 [11.33-11.61] | -8.2% (spread V 0.2%, B 2.4%): larch better |
| token-gap p99 ms | 56.08 [55.59-56.57] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m1 W4 at 8: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 662.8 [662.6-663.1] | 565.7 [561.0-570.5] | +17.2% (spread V 0.1%, B 1.7%): larch better |
| TTFT p50 ms | 74.1 [73.8-74.4] | 113.1 [112.5-113.7] | -34.5% (spread V 0.9%, B 1.0%): larch better |
| TTFT p99 ms | 1039.6 [1024.4-1054.7] | 889.8 [839.5-940.2] | +16.8% (spread V 2.9%, B 11.3%): larch worse |
| TPOT p50 ms | 11.47 [11.45-11.50] | 12.20 [12.08-12.31] | -5.9% (spread V 0.4%, B 1.9%): larch better |
| TPOT p99 ms | 14.29 [14.24-14.34] | 33.44 [31.79-35.10] | -57.3% (spread V 0.7%, B 9.9%): larch better |
| token-gap p99 ms | 90.19 [90.11-90.26] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m1 W4 at 16: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 943.0 [941.8-944.2] | 718.0 [708.4-727.5] | +31.3% (spread V 0.3%, B 2.7%): larch better |
| TTFT p50 ms | 75.9 [75.9-76.0] | 118.7 [113.6-123.9] | -36.1% (spread V 0.1%, B 8.6%): larch better |
| TTFT p99 ms | 1769.7 [1748.1-1791.4] | 1650.8 [1645.0-1656.6] | +7.2% (spread V 2.4%, B 0.7%): larch worse |
| TPOT p50 ms | 15.95 [15.94-15.97] | 17.81 [17.73-17.89] | -10.4% (spread V 0.1%, B 0.9%): larch better |
| TPOT p99 ms | 19.04 [18.97-19.10] | 45.20 [42.34-48.07] | -57.9% (spread V 0.7%, B 12.7%): larch better |
| token-gap p99 ms | 94.67 [94.65-94.68] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m1 W6 fixed pp512/tg128: S pooled over larch and driver runs, alternated in one hold

| npl | larch S | driver S | larch vs driver | larch S_tg | driver S_tg | larch S_pp | driver S_pp |
|---|---|---|---|---|---|---|---|
| 1 | 644.0 [642.1-646.9] (n=9) | 625.0 [603.8-644.4] (n=9) | +3.0% (spread V 0.7%, D 6.5%): within noise | 140.8 | 139.9 | 6052 | 4685 |
| 4 | 1929.6 [1925.0-1936.4] (n=9) | 1933.8 [1890.9-1978.7] (n=9) | -0.2% (spread V 0.6%, D 4.6%): within noise | 518.7 | 486.3 | 6031 | 7559 |
| 8 | 2774.7 [2768.2-2782.1] (n=9) | 3086.2 [3035.4-3127.1] (n=9) | -10.1% (spread V 0.5%, D 3.0%): larch worse | 879.8 | 913.0 | 6011 | 7623 |
| 16 | 3678.7 [3675.0-3684.9] (n=9) | 4250.4 [4188.2-4282.3] (n=9) | -13.5% (spread V 0.3%, D 2.2%): larch worse | 1441.5 | 1533.6 | 6011 | 7629 |

#### m1 W1 chat at 1: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 125.1 [105.4-125.3] | 118.2 [85.2-118.7] | +5.9% (spread V 15.9%, B 28.3%): within noise |
| TTFT p50 ms | 77.6 [77.1-77.7] | 69.3 [69.2-75.6] | +11.9% (spread V 0.7%, B 9.2%): larch worse |
| TTFT p99 ms | 318.7 [289.3-3332.7] | 112.9 [103.4-6763.6] | +182.2% (spread V 954.9%, B 5897.4%): within noise |
| TPOT p50 ms | 7.30 [7.29-7.34] | 7.97 [7.97-8.02] | -8.5% (spread V 0.7%, B 0.6%): larch better |
| TPOT p99 ms | 7.74 [7.70-7.75] | 8.38 [8.30-8.53] | -7.6% (spread V 0.7%, B 2.8%): larch better |
| generated tokens | 2021 [2021-2021] | 2043 [2043-2043] |  |
| cached prompt tokens | 22072 [22072-22072] | 22073 [22073-22073] |  |

failures over all 6 runs: 0

#### m1 W2 chats at 8: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 547.0 [546.1-562.7] | 311.7 [289.7-409.8] | +75.5% (spread V 3.0%, B 38.5%): larch better |
| TTFT p50 ms | 103.8 [103.7-104.4] | 150.7 [130.9-150.7] | -31.1% (spread V 0.7%, B 13.2%): larch better |
| TTFT p99 ms | 582.3 [511.9-591.5] | 3167.2 [789.0-5082.9] | -81.6% (spread V 13.7%, B 135.6%): within noise |
| TPOT p50 ms | 12.73 [12.70-13.09] | 16.69 [16.16-17.07] | -23.7% (spread V 3.1%, B 5.4%): larch better |
| TPOT p99 ms | 15.53 [15.52-16.44] | 35.69 [24.29-72.87] | -56.5% (spread V 5.9%, B 136.1%): within noise |
| generated tokens | 12168 [12168-12978] | 12766 [12608-12911] |  |
| cached prompt tokens | 81785 [81785-81785] | 81792 [81792-81792] |  |

failures over all 6 runs: 0

#### m1 W4 at 8 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 649.4 [649.2-650.4] | 444.6 [434.7-447.9] | +46.1% (spread V 0.2%, B 2.9%): larch better |
| TTFT p50 ms | 82.7 [82.2-83.4] | 200.2 [196.8-208.5] | -58.7% (spread V 1.4%, B 5.9%): larch better |
| TTFT p99 ms | 645.5 [572.2-789.1] | 853.4 [648.4-928.7] | -24.4% (spread V 33.6%, B 32.8%): within noise |
| TPOT p50 ms | 11.67 [11.67-11.70] | 15.86 [15.83-16.07] | -26.4% (spread V 0.3%, B 1.5%): larch better |
| TPOT p99 ms | 14.66 [14.23-14.68] | 28.12 [28.04-37.60] | -47.9% (spread V 3.1%, B 34.0%): larch better |
| generated tokens | 59421 [59106-59480] | 58977 [58932-59594] |  |
| cached prompt tokens | 0 [0-0] | 5772 [5771-5778] |  |

failures over all 6 runs: 0

#### m1 W4 at 16 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 924.2 [921.9-924.2] | 536.7 [507.7-546.4] | +72.2% (spread V 0.2%, B 7.2%): larch better |
| TTFT p50 ms | 98.0 [97.8-99.3] | 229.0 [224.6-231.4] | -57.2% (spread V 1.5%, B 2.9%): larch better |
| TTFT p99 ms | 1695.4 [1684.5-1758.5] | 1732.2 [1563.5-1751.2] | -2.1% (spread V 4.4%, B 10.8%): within noise |
| TPOT p50 ms | 16.17 [16.07-16.27] | 24.72 [24.70-25.25] | -34.6% (spread V 1.2%, B 2.2%): larch better |
| TPOT p99 ms | 20.07 [19.49-20.55] | 48.84 [48.10-57.52] | -58.9% (spread V 5.3%, B 19.3%): larch better |
| generated tokens | 59208 [59057-59507] | 59171 [58824-59622] |  |
| cached prompt tokens | 0 [0-0] | 5541 [5540-5544] |  |

failures over all 6 runs: 0

### m5 detail

#### m5 W4 at 1: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 143.3 [143.2-143.3] | 143.7 [143.7-143.7] | -0.3% (spread V 0.1%, B 0.0%): within noise |
| TTFT p50 ms | 66.1 [66.1-66.1] | 52.9 [51.9-54.0] | +25.0% (spread V 0.1%, B 4.0%): larch worse |
| TTFT p99 ms | 322.5 [322.3-322.8] | 238.3 [238.2-238.4] | +35.4% (spread V 0.1%, B 0.1%): larch worse |
| TPOT p50 ms | 6.62 [6.62-6.63] | 6.72 [6.72-6.72] | -1.5% (spread V 0.1%, B 0.0%): within noise |
| TPOT p99 ms | 7.00 [7.00-7.01] | 6.99 [6.97-7.00] | +0.3% (spread V 0.2%, B 0.4%): within noise |
| token-gap p99 ms | 7.09 [7.09-7.09] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m5 W4 at 4: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 462.4 [462.4-462.4] | 441.8 [439.6-444.0] | +4.7% (spread V 0.0%, B 1.0%): larch better |
| TTFT p50 ms | 71.7 [69.7-73.6] | 101.7 [97.0-106.3] | -29.5% (spread V 5.5%, B 9.2%): larch better |
| TTFT p99 ms | 520.7 [388.5-652.8] | 577.2 [575.1-579.3] | -9.8% (spread V 50.8%, B 0.7%): within noise |
| TPOT p50 ms | 8.16 [8.15-8.17] | 8.59 [8.58-8.60] | -5.0% (spread V 0.2%, B 0.3%): larch better |
| TPOT p99 ms | 9.67 [9.58-9.75] | 10.23 [10.15-10.31] | -5.5% (spread V 1.7%, B 1.6%): larch better |
| token-gap p99 ms | 56.81 [56.03-57.59] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m5 W4 at 8: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 734.6 [734.4-734.7] | 627.2 [615.0-639.4] | +17.1% (spread V 0.0%, B 3.9%): larch better |
| TTFT p50 ms | 68.8 [68.3-69.2] | 117.3 [108.6-126.1] | -41.4% (spread V 1.4%, B 15.0%): larch better |
| TTFT p99 ms | 998.9 [944.6-1053.2] | 836.4 [814.5-858.4] | +19.4% (spread V 10.9%, B 5.2%): larch worse |
| TPOT p50 ms | 10.36 [10.32-10.40] | 10.69 [10.61-10.78] | -3.1% (spread V 0.7%, B 1.6%): larch better |
| TPOT p99 ms | 12.68 [12.62-12.74] | 34.25 [32.37-36.14] | -63.0% (spread V 1.0%, B 11.0%): larch better |
| token-gap p99 ms | 80.15 [80.10-80.21] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m5 W4 at 16: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 1057.9 [1053.7-1062.0] | 779.8 [758.9-800.6] | +35.7% (spread V 0.8%, B 5.3%): larch better |
| TTFT p50 ms | 70.4 [70.4-70.5] | 127.2 [125.9-128.4] | -44.6% (spread V 0.2%, B 1.9%): larch better |
| TTFT p99 ms | 1664.4 [1619.2-1709.5] | 1608.1 [1600.4-1615.9] | +3.5% (spread V 5.4%, B 1.0%): within noise |
| TPOT p50 ms | 14.21 [14.18-14.25] | 15.73 [15.71-15.74] | -9.6% (spread V 0.5%, B 0.2%): larch better |
| TPOT p99 ms | 16.85 [16.71-16.99] | 46.61 [41.54-51.69] | -63.9% (spread V 1.6%, B 21.8%): larch better |
| token-gap p99 ms | 83.91 [83.87-83.94] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m5 W6 fixed pp512/tg128: S pooled over larch and driver runs, alternated in one hold

| npl | larch S | driver S | larch vs driver | larch S_tg | driver S_tg | larch S_pp | driver S_pp |
|---|---|---|---|---|---|---|---|
| 1 | 692.2 [689.2-695.3] (n=9) | 665.2 [641.7-677.4] (n=9) | +4.1% (spread V 0.9%, D 5.3%): within noise | 151.5 | 149.3 | 6420 | 4890 |
| 4 | 2065.8 [2061.4-2069.7] (n=9) | 2116.6 [2063.2-2179.1] (n=9) | -2.4% (spread V 0.4%, D 5.5%): within noise | 556.9 | 531.2 | 6402 | 8337 |
| 8 | 3000.6 [2990.8-3007.4] (n=9) | 3441.5 [3366.8-3496.8] (n=9) | -12.8% (spread V 0.6%, D 3.7%): larch worse | 962.3 | 1029.6 | 6378 | 8306 |
| 16 | 4024.8 [4019.0-4032.0] (n=9) | 4786.7 [4734.4-4857.8] (n=9) | -15.9% (spread V 0.3%, D 2.6%): larch worse | 1622.5 | 1769.7 | 6390 | 8342 |

#### m5 W1 chat at 1: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 135.8 [135.0-136.1] | 121.9 [84.2-122.1] | +11.4% (spread V 0.8%, B 31.1%): within noise |
| TTFT p50 ms | 73.7 [73.6-73.8] | 72.5 [68.8-73.2] | +1.6% (spread V 0.4%, B 6.1%): within noise |
| TTFT p99 ms | 324.0 [273.5-414.0] | 115.9 [106.8-6935.3] | +179.7% (spread V 43.3%, B 5894.2%): within noise |
| TPOT p50 ms | 6.70 [6.68-6.71] | 7.76 [7.73-7.83] | -13.6% (spread V 0.4%, B 1.4%): larch better |
| TPOT p99 ms | 6.93 [6.91-6.94] | 8.12 [7.92-8.13] | -14.6% (spread V 0.5%, B 2.6%): larch better |
| generated tokens | 1890 [1890-1890] | 1861 [1861-1861] |  |
| cached prompt tokens | 21829 [21829-21829] | 21829 [21829-21829] |  |

failures over all 6 runs: 0

#### m5 W2 chats at 8: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 642.9 [630.2-649.0] | 421.9 [347.7-422.0] | +52.4% (spread V 2.9%, B 17.6%): larch better |
| TTFT p50 ms | 94.3 [92.2-95.2] | 135.6 [122.3-136.2] | -30.4% (spread V 3.2%, B 10.2%): larch better |
| TTFT p99 ms | 637.0 [569.0-854.1] | 547.8 [449.8-5864.2] | +16.3% (spread V 44.8%, B 988.3%): within noise |
| TPOT p50 ms | 10.93 [10.88-10.93] | 14.07 [13.62-14.35] | -22.3% (spread V 0.5%, B 5.2%): larch better |
| TPOT p99 ms | 12.61 [12.58-13.16] | 39.43 [37.47-50.74] | -68.0% (spread V 4.6%, B 33.6%): larch better |
| generated tokens | 13972 [13917-14033] | 13561 [13286-13567] |  |
| cached prompt tokens | 80557 [80557-80557] | 80557 [80557-80557] |  |

failures over all 6 runs: 0

#### m5 W4 at 8 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 578.0 [572.1-581.7] | 396.8 [387.2-413.9] | +45.7% (spread V 1.7%, B 6.7%): larch better |
| TTFT p50 ms | 77.8 [77.6-81.6] | 166.0 [155.0-174.6] | -53.1% (spread V 5.1%, B 11.8%): larch better |
| TTFT p99 ms | 847.4 [738.4-983.6] | 806.5 [676.2-882.0] | +5.1% (spread V 28.9%, B 25.5%): within noise |
| TPOT p50 ms | 12.62 [12.46-12.66] | 17.07 [16.92-17.26] | -26.1% (spread V 1.5%, B 2.0%): larch better |
| TPOT p99 ms | 18.07 [17.63-18.46] | 30.35 [27.37-75.68] | -40.5% (spread V 4.6%, B 159.2%): within noise |
| generated tokens | 28539 [27740-28823] | 29417 [28323-29458] |  |
| cached prompt tokens | 0 [0-0] | 4618 [4617-4624] |  |

failures over all 6 runs: 0

#### m5 W4 at 16 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 747.2 [746.8-762.3] | 446.9 [438.5-465.0] | +67.2% (spread V 2.1%, B 5.9%): larch better |
| TTFT p50 ms | 114.7 [107.3-116.5] | 195.6 [184.5-198.1] | -41.4% (spread V 8.0%, B 7.0%): larch better |
| TTFT p99 ms | 1820.5 [1658.7-1842.4] | 1315.5 [1290.1-1475.9] | +38.4% (spread V 10.1%, B 14.1%): larch worse |
| TPOT p50 ms | 18.60 [18.54-18.66] | 28.06 [27.60-29.37] | -33.7% (spread V 0.7%, B 6.3%): larch better |
| TPOT p99 ms | 28.47 [27.45-29.25] | 44.71 [38.50-48.11] | -36.3% (spread V 6.3%, B 21.5%): larch better |
| generated tokens | 28903 [28106-28972] | 29591 [28902-29905] |  |
| cached prompt tokens | 0 [0-0] | 4430 [4427-4430] |  |

failures over all 6 runs: 0

### m13 detail

#### m13 W4 at 1: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 122.0 [121.3-122.6] | 135.3 [134.9-135.8] | -9.9% (spread V 1.0%, B 0.6%): larch worse |
| TTFT p50 ms | 168.9 [168.3-169.5] | 169.7 [162.6-176.9] | -0.5% (spread V 0.7%, B 8.5%): within noise |
| TTFT p99 ms | 722.3 [719.0-725.6] | 428.6 [415.0-442.2] | +68.5% (spread V 0.9%, B 6.4%): larch worse |
| TPOT p50 ms | 7.40 [7.36-7.44] | 6.74 [6.74-6.74] | +9.7% (spread V 1.0%, B 0.0%): larch worse |
| TPOT p99 ms | 7.60 [7.52-7.68] | 7.39 [7.37-7.41] | +2.9% (spread V 2.0%, B 0.5%): larch worse |
| token-gap p99 ms | 8.71 [8.55-8.87] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

#### m13 W4 at 4: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 174.6 [174.6-174.7] | 255.4 [255.0-255.8] | -31.6% (spread V 0.0%, B 0.3%): larch worse |
| TTFT p50 ms | 431.3 [424.6-438.1] | 224.2 [222.5-225.9] | +92.4% (spread V 3.1%, B 1.5%): larch worse |
| TTFT p99 ms | 2262.6 [2221.5-2303.8] | 1166.9 [1121.3-1212.5] | +93.9% (spread V 3.6%, B 7.8%): larch worse |
| TPOT p50 ms | 20.09 [20.08-20.10] | 14.74 [14.72-14.75] | +36.3% (spread V 0.1%, B 0.2%): larch worse |
| TPOT p99 ms | 30.57 [30.53-30.61] | 17.87 [17.84-17.89] | +71.1% (spread V 0.3%, B 0.3%): larch worse |
| token-gap p99 ms | 285.67 [284.89-286.45] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |
| cached prompt tokens | 0 [0-0] | 0 [0-0] |  |

failures over all 4 runs: 0

### vLLM detail (m1; W4 rows are not results: `compare` refuses them)

#### m1 W1 at 1: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 128.9 | 70.9 | 149.1 | 7.27 | 7.75 |
| vLLM AWQ | r2 | 128.8 | 70.8 | 139.3 | 7.29 | 7.74 |
| vLLM AWQ | mean | 128.8 | 70.8 | 144.2 | 7.28 | 7.75 |
| vLLM GGUF | r1 | 88.1 | 589.4 | 1072.7 | 7.93 | 8.42 |
| larch | r1 | 122.8 | 76.6 | 859.0 | 7.31 | 7.77 |
| larch | r2 | 126.9 | 76.9 | 205.0 | 7.35 | 7.73 |
| larch | mean | 124.9 | 76.8 | 532.0 | 7.33 | 7.75 |
| larch vs vLLM AWQ | mean Δ | -3.1% noise (spread L 3.3%, A 0.1%) | +8.4% worse (spread L 0.4%, A 0.1%) | +268.9% worse (spread L 122.9%, A 6.8%) | +0.7% noise (spread L 0.6%, A 0.2%) | +0.1% noise (spread L 0.5%, A 0.1%) |

failures over all 5 runs: 0

#### m1 W2 at 4: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 388.6 | 102.5 | 159.3 | 8.92 | 9.98 |
| vLLM AWQ | r2 | 388.6 | 101.5 | 164.0 | 8.91 | 9.98 |
| vLLM AWQ | mean | 388.6 | 102.0 | 161.7 | 8.91 | 9.98 |
| vLLM GGUF | r1 | 151.7 | 786.1 | 1413.3 | 21.31 | 30.75 |
| larch | r1 | 364.0 | 100.3 | 352.4 | 9.47 | 10.62 |
| larch | r2 | 365.1 | 99.4 | 330.4 | 9.46 | 10.59 |
| larch | mean | 364.5 | 99.8 | 341.4 | 9.46 | 10.61 |
| larch vs vLLM AWQ | mean Δ | -6.2% worse (spread L 0.3%, A 0.0%) | -2.1% better (spread L 0.9%, A 1.0%) | +111.1% worse (spread L 6.4%, A 2.9%) | +6.2% worse (spread L 0.1%, A 0.2%) | +6.3% worse (spread L 0.3%, A 0.1%) |

failures over all 5 runs: 0

#### m1 W2 at 8: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 629.1 | 107.6 | 349.0 | 11.32 | 13.75 |
| vLLM AWQ | r2 | 628.9 | 107.6 | 352.7 | 11.32 | 13.75 |
| vLLM AWQ | mean | 629.0 | 107.6 | 350.9 | 11.32 | 13.75 |
| vLLM GGUF | r1 | 175.7 | 806.4 | 3270.0 | 39.71 | 61.11 |
| larch | r1 | 562.7 | 94.3 | 369.2 | 12.77 | 15.21 |
| larch | r2 | 562.3 | 92.3 | 375.2 | 12.73 | 15.26 |
| larch | mean | 562.5 | 93.3 | 372.2 | 12.75 | 15.23 |
| larch vs vLLM AWQ | mean Δ | -10.6% worse (spread L 0.1%, A 0.0%) | -13.2% better (spread L 2.1%, A 0.0%) | +6.1% worse (spread L 1.6%, A 1.1%) | +12.7% worse (spread L 0.3%, A 0.0%) | +10.8% worse (spread L 0.3%, A 0.0%) |

failures over all 5 runs: 0

#### m1 W4 at 1: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 136.7 | 64.5 | 324.9 | 6.96 | 7.31 |
| vLLM AWQ | r2 | 136.7 | 65.8 | 319.6 | 6.97 | 7.31 |
| vLLM AWQ | mean | 136.7 | 65.2 | 322.2 | 6.97 | 7.31 |
| vLLM GGUF | r1 | 95.4 | 586.9 | 2997.8 | 7.65 | 8.00 |
| larch | r1 | 134.0 | 68.9 | 343.7 | 7.09 | 7.54 |
| larch | r2 | 134.0 | 68.8 | 341.2 | 7.08 | 7.53 |
| larch | mean | 134.0 | 68.8 | 342.4 | 7.09 | 7.53 |
| larch vs vLLM AWQ | mean Δ | -2.0% noise (spread L 0.0%, A 0.0%) | +5.6% worse (spread L 0.1%, A 2.0%) | +6.3% worse (spread L 0.7%, A 1.7%) | +1.7% noise (spread L 0.1%, A 0.0%) | +3.1% worse (spread L 0.1%, A 0.1%) |

failures over all 5 runs: 0

#### m1 W4 at 4: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 463.6 | 77.5 | 427.9 | 8.12 | 10.06 |
| vLLM AWQ | r2 | 463.5 | 77.9 | 429.3 | 8.12 | 10.08 |
| vLLM AWQ | mean | 463.6 | 77.7 | 428.6 | 8.12 | 10.07 |
| vLLM GGUF | r1 | 167.4 | 610.3 | 4084.9 | 19.83 | 35.35 |
| larch | r1 | 430.0 | 74.8 | 387.5 | 8.77 | 10.61 |
| larch | r2 | 429.7 | 74.1 | 387.3 | 8.80 | 10.68 |
| larch | mean | 429.8 | 74.5 | 387.4 | 8.78 | 10.64 |
| larch vs vLLM AWQ | mean Δ | -7.3% worse (spread L 0.1%, A 0.0%) | -4.2% better (spread L 1.0%, A 0.5%) | -9.6% better (spread L 0.1%, A 0.3%) | +8.1% worse (spread L 0.3%, A 0.0%) | +5.7% worse (spread L 0.7%, A 0.1%) |

failures over all 5 runs: 0

#### m1 W4 at 8: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 765.3 | 92.7 | 769.1 | 9.88 | 12.51 |
| vLLM AWQ | r2 | 765.2 | 93.4 | 767.8 | 9.88 | 12.51 |
| vLLM AWQ | mean | 765.2 | 93.0 | 768.4 | 9.88 | 12.51 |
| vLLM GGUF | r1 | 190.7 | 753.0 | 7327.9 | 37.73 | 60.56 |
| larch | r1 | 663.9 | 73.5 | 819.4 | 11.49 | 14.21 |
| larch | r2 | 666.3 | 73.7 | 819.6 | 11.46 | 14.19 |
| larch | mean | 665.1 | 73.6 | 819.5 | 11.48 | 14.20 |
| larch vs vLLM AWQ | mean Δ | -13.1% worse (spread L 0.4%, A 0.0%) | -20.9% better (spread L 0.2%, A 0.7%) | +6.6% worse (spread L 0.0%, A 0.2%) | +16.2% worse (spread L 0.2%, A 0.0%) | +13.5% worse (spread L 0.2%, A 0.0%) |

failures over all 5 runs: 0

#### m1 W4 at 16: larch in process vs vLLM

| Engine | Run | agg tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms |
|---|---|---|---|---|---|---|
| vLLM AWQ | r1 | 1034.0 | 102.5 | 1442.9 | 14.61 | 17.88 |
| vLLM AWQ | r2 | 1034.1 | 104.4 | 1440.9 | 14.61 | 17.98 |
| vLLM AWQ | mean | 1034.1 | 103.4 | 1441.9 | 14.61 | 17.93 |
| vLLM GGUF | r1 | 198.7 | 793.3 | 13804.9 | 74.76 | 105.38 |
| larch | r1 | 947.1 | 75.9 | 1556.8 | 15.98 | 19.05 |
| larch | r2 | 945.9 | 75.3 | 1551.1 | 16.03 | 19.13 |
| larch | mean | 946.5 | 75.6 | 1554.0 | 16.00 | 19.09 |
| larch vs vLLM AWQ | mean Δ | -8.5% worse (spread L 0.1%, A 0.0%) | -26.9% better (spread L 0.7%, A 1.9%) | +7.8% worse (spread L 0.4%, A 0.1%) | +9.5% worse (spread L 0.3%, A 0.0%) | +6.4% worse (spread L 0.4%, A 0.6%) |

failures over all 5 runs: 0

#### Prompt tokens served from cache (cached / prompt)

| Row | vLLM AWQ | vLLM GGUF | larch |
|---|---|---|---|
| W1 at 1 | 21,760 / 26,625 | 21,760 / 26,625 | 22,072 / 26,625 |
| W2 at 4 | 39,680 / 53,170 | 39,680 / 53,170 | 40,694 / 53,170 |
| W2 at 8 | 79,936 / 106,460 | 79,872 / 106,460 | 81,785 / 106,460 |
| W4 at 1 | 0 / 112,624 | 0 / 112,624 | 0 / 112,624 |
| W4 at 4 | 0 / 112,624 | 0 / 112,624 | 0 / 112,624 |
| W4 at 8 | 0 / 112,624 | 0 / 112,624 | 0 / 112,624 |
| W4 at 16 | 0 / 112,624 | 0 / 112,624 | 0 / 112,624 |
