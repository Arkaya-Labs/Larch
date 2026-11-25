# Final sweep after the perf stack (CUDA, wsl5070), 17 Oct 2026

dev `4a540ae` on the RTX 5070 Ti, m1 and m5 at Q4_K_M, against llama.cpp b11458 (`c479922ac`) and Ollama 0.40.1 on
the same GGUF files. One unattended run, 05:48Z to 08:54Z, every GPU job one exclusive `flock ~/src/gpu.lock` hold
(the hold count and exit codes were not recorded in a committed log; every result JSON that carries a `failures` field reports 0, 74 of 74, so no row was rerun). Raw results are in `final-sweep/`, the GPU log is
`final-sweep-gpu-log.csv`.

## Summary

1. Parity passes on m1 and m5 in all four modes (`--mode all`). The llama.cpp driver gate passes on both: the driver's S is within 0.2–1.5 % of llama-batched-bench at every npl (pooled over 9 alternated runs a side).
2. vs llama.cpp on W4 (in process, exact lengths; `compare` accepts all 24 pairs), larch's aggregate is higher at 8 and 16 concurrent: +12.7 % and +15.1 % on m1, +11.2 % and +11.6 % on m5. TPOT p99 is 42–50 % lower.
3. TTFT p99 is the loss at every concurrency: +40–46 % on m1 and +16–39 % on m5 at 8/16, +108–125 % at 1. TTFT p50 is 21–26 % lower at 8/16.
4. W4 at 1: aggregate −4.0 % on m1 (outside noise) and −2.0 % on m5 (within noise); TPOT p99 +3.5 % and +4.5 %.
5. W6 fixed shape (pooled S, 9 runs a side): within noise at npl 1 (+0.1 %, +1.6 %), but 10–20 % below llama.cpp's driver at npl 4–16. Decode S_tg is −9 % to +3 % of the driver's; prefill S_pp stays at 4.9–5.3k tok/s at every npl while the driver reaches 7.1–7.6k.
6. vs Ollama (both over localhost HTTP, natural EOS): W4 aggregate +31 % and +49 % on m1, +20 % and +43 % on m5 at 8/16. W2 at 8: +16.4 % on m1, +9.5 % on m5 (within noise).
7. W1 chat at 1 is a clear loss to Ollama: TTFT p50 2.0–2.5× and p99 4.0–4.5× Ollama's, aggregate −10.8 % on m1 and −5.4 % on m5 (within noise). Both engines reuse the same cached prefix, so the gap is larch's prefill of each new turn.
8. TTFT p99 vs Ollama is also higher on W4 at 16 (+22 %, +28 %) and on m5's W2 at 8 (+61 %). TPOT p99 is lower or within noise in every Ollama row.
9. Cache-on W4 at 8 (`LARCH_PREFIX_CACHE_BLOCKS=512`, not the default): 3,184 prompt tokens served from the cache; every metric within noise of the default.
10. Thermals: max 70 °C; no thermal throttle reason (0x20, 0x40, 0x8, 0x80) in any of the 371 samples of `final-sweep-gpu-log.csv`. Only the reliability limiter (active from idle, 364 samples) and SW power capping (7 samples).

## For the docs

> On one RTX 5070 Ti (16 GB, sm_120, WSL2, driver 617.14), we compared larch (dev `4a540ae`) with llama.cpp b11458 and
> Ollama 0.40.1 on the same two 7–8B instruct models at Q4_K_M (m1 and m5 in `bench/models.toml`). On 200 independent
> prompts with exact output lengths (W4, 2 runs per engine, alternated), larch's aggregate throughput was 11–15 % above
> llama.cpp's server loop at 8 and 16 concurrent requests, with 42–50 % lower p99 time per output token. Its p99 time to
> first token was 16–46 % higher at the same loads, and single-stream throughput was 4 % behind on m1 and within noise on
> m5. In llama-batched-bench's fixed pp512/tg128 shape (9 runs a side), larch matched llama.cpp at one sequence and was
> 10–20 % slower at 4–16 sequences, limited mainly by prefill. Over localhost HTTP with natural end of sequence (3 runs per engine,
> alternated), larch's W4 aggregate was 20–49 % above Ollama's at 8 and 16 concurrent, but on a 12-turn single chat (W1)
> its time to first token was 2.0–2.5× Ollama's at p50.

## Headline (medians; % = larch vs the baseline; "noise" = under 2 % or under either engine's run-to-run spread)

| Row | Baseline | m1 agg tok/s (V vs B) | m1 TTFT p99 | m1 TPOT p99 | m5 agg tok/s (V vs B) | m5 TTFT p99 | m5 TPOT p99 |
|---|---|---|---|---|---|---|---|
| W4 at 1 | llama.cpp | 123.9 vs 129.0, **−4.0 %** | **+125 %** | **+3.5 %** | 135.2 vs 137.9, −2.0 % noise | **+108 %** | **+4.5 %** |
| W4 at 8 | llama.cpp | 572.2 vs 507.9, **+12.7 %** | **+45.7 %** | **−47.1 %** | 621.0 vs 558.6, **+11.2 %** | **+38.7 %** | **−50.3 %** |
| W4 at 16 | llama.cpp | 777.8 vs 676.0, **+15.1 %** | **+39.9 %** | **−42.5 %** | 860.7 vs 771.1, **+11.6 %** | **+16.2 %** | **−44.6 %** |
| W6 S, npl 1 / 4 / 8 / 16 | llama.cpp driver | +0.1 % noise / **−13.8 %** / **−10.4 %** / **−16.4 %** | | | +1.6 % noise / **−13.5 %** / **−12.9 %** / **−19.9 %** | | |
| W1 chat at 1 | Ollama | 103.6 vs 116.0, **−10.8 %** | **+351 %** | +3.7 % noise | 117.6 vs 124.4, −5.4 % noise | **+298 %** | −4.5 % noise |
| W2 chats at 8 | Ollama | 392.6 vs 337.4, **+16.4 %** | −8.4 % noise | **−38.7 %** | 446.2 vs 407.5, +9.5 % noise | **+61.4 %** | −40.8 % noise |
| W4 at 8 (ctx 2560) | Ollama | 558.3 vs 426.2, **+31.0 %** | +38.6 % noise | **−43.8 %** | 463.2 vs 385.3, **+20.2 %** | −4.6 % noise | +2.7 % noise |
| W4 at 16 (ctx 2560) | Ollama | 756.9 vs 509.5, **+48.6 %** | **+22.4 %** | **−56.8 %** | 584.1 vs 409.8, **+42.5 %** | **+28.3 %** | −7.1 % noise |

For agg tok/s a positive % is a larch win; for TTFT and TPOT a negative % is. Bold marks a margin outside noise. The
llama.cpp rows are in process on both sides (W4: `larch bench --mode serve` vs `tools/llama-serve-bench`; W6: `--mode
batched` vs the driver's `fixed`). The Ollama rows are HTTP on both sides: larch through the benchmark-only adapter
`examples/serve_bench.rs` and `--engine larch-http --natural-eos`, Ollama through its own server. With natural EOS the
answer lengths differ per engine (generated tokens are in the detail tables), and HTTP rows have chunk gaps, not token
gaps.

**TTFT p99 is the known weak spot.** It is higher than llama.cpp's in all six W4 rows. Outside noise it is higher than Ollama's
in five of eight HTTP rows; the other three are within noise. At 8 and 16 concurrent, larch's TTFT p50 is the lower one, but its tail is longer: a
prompt that arrives while others are prefilling waits for the shared 512-token prefill budget. The W6 prefill column shows
the same limit: S_pp is flat across npl. In-process token-gap p99 is 98–154 ms at 8/16 while TPOT p99 is 16–25 ms. Counted from the
per-token gaps of the in-process W4 runs (64,237 gaps per run), the share of gaps above 40 ms (about 2× TPOT p99 at
8) is 3.0–3.1 % at 8 concurrent and 6.5–6.7 % at 16 on both models, and 0.0–0.01 % at 1. Above 100 ms: 1.8 % (m1)
and 0.8 % (m5) at 8, 3.7–3.8 % (m1) and 2.0 % (m5) at 16. The driver records no per-token gaps, so there is no llama.cpp column for it.

**Host-gap caveat, llama.cpp driver at npl 16.** `llama-serve-bench` reads every logit row on the host for its greedy scan,
which llama-batched-bench never does. Its source puts this at about 165 µs per 16-row step (its comment estimates 2 % of a
decode step). In this run's gate the driver's pooled S is −0.6 % (m1) and +0.5 % (m5) of llama-batched-bench at npl 16.
Read the driver's npl-16 numbers (W6, and W4 at 16) as carrying about 1 % of host overhead that llama.cpp's own loop
does not have. That is smaller than every larch margin outside noise.

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (16303 MiB), sm_120, VBIOS 98.03.58.40.8b, power limit 300 W, max SM clock 3090 MHz |
| Driver | 617.14 (Windows host; CUDA UMD 13.4) |
| CPU, RAM | AMD Ryzen 7 9800X3D, 8 logical CPUs and 15 GiB visible to WSL |
| OS | Ubuntu 26.04.1 LTS on WSL2, kernel 6.18.33.2-microsoft-standard-WSL2 |
| larch | dev `4a540ae2143a1550b3503d40fb7dce59e7927451`. Release build `cargo build --release --locked -p larch --no-default-features --bin larch --example serve_bench`: CUDA backend, kernels for `86-real;89-real;120-real;120-virtual` (the default list), rustc 1.97.1, nvcc 13.3.73, runtime 13.3. `larch` sha256 `0d7480186d2f…`, equal to `build` in all 80 JSONs it wrote (32 in process, plus the bench client of 24 larch-http and 24 Ollama runs); `serve_bench` sha256 `213077465449…` |
| Engine settings | dev defaults (INT8 on for sm_120, prefix cache off); no `LARCH_*` variable set, except the one cache-on row |
| llama.cpp | b11458 (`c479922ac`), `-DGGML_CUDA=ON`, GGML_NATIVE (sm_120), CUDAHOSTCXX g++ 14.3; `llama-serve-bench` build `d151f1ff29a7…`. Searched flags (`baseline.py flags`): m1 `-t 4 -b 2048 -ub 1024`, m5 `-t 4 -b 1024 -ub 1024`, both with `-ngl 99 -fa on -ctk f16 -ctv f16` |
| Ollama | 0.40.1 (its bundled llama-server, CUDA v13 backend); `OLLAMA_NUM_PARALLEL=N`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_KEEP_ALIVE=-1` (`baseline.py ollama-serve`) |
| Models | m1, m5 (`bench/models.toml`), Q4_K_M; sha256 `7b064f58…3557c` and `65b8fcd9…1423`, both equal to `models.toml` |
| Parity references | b11458 cuBLAS ub32 dumps, sha256 `f0e5ce16…` (m1) and `7c2f92f4…` (m5), equal to `bench/oracle-bounds.json` |

## Protocol

- **Parity** (once per model): `larch parity --mode all` with the bound and margin from `bench/oracle-bounds.json`, the same as `parity-wsl5070.md`.
- **Driver gate** (D13): `baseline.py run --row fixed`, which alternates llama-batched-bench and the driver 9 times in one hold; `baseline.check_fixed` compares pooled S. Pooled S is total tokens over total time, i.e. the harmonic mean of the runs' S.
- **W4 vs llama.cpp**: chat.jsonl, 200 prompts, `ignore_eos`, per-sequence context 2560 (the driver's `n_ctx_seq`: the longest prompt plus its output, rounded up to 256). Each of c = 1, 8 and 16 ran V-L-L-V, one hold per run. W4 at 1 ran whole (about 8 min a run), with no `--slice`. `baseline.py compare` accepted all 4 pairs of every row (24 of 24).
- **W6**: per model, one hold alternating `larch bench --mode batched` and the driver's `fixed` 9 times (V-D × 9). Pooled S per npl, the same procedure as the gate.
- **vs Ollama**: W1 chat at 1 (context 8192), W2 chats at 8 (8192), and W4 at 8 and at 16 (`--set max_context=2560` on both engines). Each ran V-O-O-V-V-O, one hold per run, with a fresh server every run on both sides. `serve_bench` got `--context` equal to the row's, `--sequences N` and `--requests` equal to turns + 1 warm-up (13, 65, 201), and exited by itself after its last request in all 24 runs. Before each Ollama start, any other ollama process was quit. `compare` does not apply here: it refuses natural-EOS lengths, and both engines report `cache_prompt: true` on W4. Cached prompt tokens match on W1 and W2 to within 7 tokens (m1 W1 22,072 larch vs 22,073 Ollama; W2 81,786 vs 81,793; m5 equal). On W4 they do not: Ollama served 5,541–5,776 of 112,624 prompt tokens from its cache on m1 (about 5 %) and 4,423–4,622 of 111,424 on m5 (about 4 %), while larch-http served 0 (its prefix cache is off by default). This favours Ollama; no verdict changes.
- **Cache-on**: one extra in-process W4 at 8 per model with `LARCH_PREFIX_CACHE_BLOCKS=512`, labelled cache-on, never mixed into the default rows.
- **Statistics**: per row and engine, the median [min-max] over runs of each run's summary value. Spread = (max − min) / median. A margin is "within noise" when it is under 2 % or under either engine's spread. With 2 runs, the median is their mean.

## Parity (larch `0d748018`, CUDA device 0, graphs on, 100 prompts)

| Model | Mode | Decisive / positions | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Bound | Replayed | Mixed | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m1 | prefill | 12791/58403 | 0.9990 | 0.9527 | 1.557e-03 | 3.390e-02 | 4.278e-01 | 0.0881 | 0 | 0 | pass |
| m1 | decode | 12791/58403 | 1.0000 | 0.9678 | 3.238e-04 | 1.669e-03 | 3.908e-03 | 0.0881 | 58303 | 0 | pass |
| m1 | chunked | 12994/59464 | 0.9922 | 0.9524 | 1.594e-03 | 3.126e-02 | 4.285e-01 | 0.0881 | 101 | 959 | pass |
| m1 | cached | 4986/29223 | 0.9998 | 0.9491 | 6.843e-04 | 2.465e-03 | 7.628e-02 | 0.0881 | 0 | 0 | pass |
| m5 | prefill | 9735/57803 | 1.0000 | 0.9417 | 2.090e-03 | 2.237e-02 | 2.370e-01 | 0.0653 | 0 | 0 | pass |
| m5 | decode | 9735/57803 | 1.0000 | 0.9644 | 5.510e-04 | 2.504e-03 | 1.998e-02 | 0.0653 | 57703 | 0 | pass |
| m5 | chunked | 9937/58856 | 0.9999 | 0.9419 | 2.288e-03 | 2.599e-02 | 5.097e-01 | 0.0653 | 101 | 951 | pass |
| m5 | cached | 4529/28923 | 1.0000 | 0.9380 | 1.549e-03 | 6.520e-03 | 3.016e-01 | 0.0653 | 0 | 0 | pass |

Runs took 557 s (m1) and 518 s (m5); raw output is in `final-sweep/wsl5070-<alias>-larch-parity.txt`. Every mode passes,
but compared with the older `parity-wsl5070.md` rows every mode except decode drifted, and the tail moved more than the mean:

| Model, mode | Decisive top-1 (old → now) | Mean KL | p99 KL | Max KL |
|---|---|---|---|---|
| m1 prefill | 1.0000 → 0.9990 | 3.24e-4 → 1.56e-3 (4.8×) | 1.57e-3 → 3.39e-2 (22×) | 3.84e-3 → 0.428 (111×) |
| m1 chunked | 1.0000 → **0.9922** (gate 0.99) | 3.24e-4 → 1.59e-3 (4.9×) | 1.56e-3 → 3.13e-2 (20×) | 3.90e-3 → 0.429 (110×) |
| m1 cached | 1.0000 → 0.9998 | 3.60e-4 → 6.84e-4 (1.9×) | 1.44e-3 → 2.47e-3 | 3.84e-3 → 7.63e-2 (20×) |
| m5 prefill | 1.0000 → 1.0000 | 5.48e-4 → 2.09e-3 (3.8×) | 2.60e-3 → 2.24e-2 | 1.59e-2 → 0.237 |
| m5 chunked | 1.0000 → 0.9999 | 5.48e-4 → 2.29e-3 (4.2×) | 2.52e-3 → 2.60e-2 | 1.80e-2 → 0.510 |
| m5 cached | 1.0000 → 1.0000 | 5.60e-4 → 1.55e-3 (2.8×) | 2.50e-3 → 6.52e-3 | 1.32e-2 → 0.302 |

Raw top-1 is also lower (m1 prefill 0.953 vs 0.968). Decode is unchanged. Every mean KL is inside its bound, but m1
chunked's decisive top-1 is only 0.0022 above the 0.99 gate. The cause is unconfirmed: the older rows predate dev's sm_120
INT8 default (#109), but #109 and #152 also changed projection routes, and no `LARCH_QUANT_I8=0` control was run. The
control (`LARCH_QUANT_I8=0 larch parity --mode chunked` on m1) is NOT RUN; see below.

## Temperature and clocks (`final-sweep-gpu-log.csv`, 371 samples every 30 s, local time UTC−5)

- Max core temperature: 70 °C at 08:06:34Z, during `wsl5070-m1-larch-http-w2-8-r2`. In-process and driver rows peaked at 58–69 °C.
- Reasons other than idle or power: **reliability (0x400)** in 364 of 371 samples, in every row and in the idle samples before the first job. It is the voltage-reliability clock limiter, not a thermal one. SW thermal (0x20), HW thermal (0x40), HW slowdown (0x8) and power brake (0x80) never appeared. nvidia-smi's cumulative throttle counters were read before and after the run, but that output was not committed, so this report relies only on the per-sample reason masks in the CSV.
- SW power cap (0x4) appeared in 7 samples: the m5 gate (2), the m5 llama.cpp W4 at 1 r1 and r2 (1 each), the W6 hold of each model (1 each), and the m1 Ollama W4 at 16 r1 (1).
- **No row was thermally throttled.**

## NOT RUN

| What | Why |
|---|---|
| Metal | out of scope for this sweep |
| larch-agent, `bench/agent`, W3 agent workload | out of scope (inference engine only) |
| Rows of architecture 16 outside this protocol: W2 at 4, W4 at 4, W5, W7, llama.cpp's W1/W2 replay, `baseline.py single` (llama-bench), `baseline.py server`, `--replay generated` W1–W3, parity `--parallel 8` and `--graphs off` | not part of this sweep's protocol |
| `baseline.py compare` on the Ollama rows | not applicable: it refuses natural-EOS lengths and `cache_prompt: true` on W4 by design (bench/README) |
| m1 parity control with `LARCH_QUANT_I8=0` (chunked mode) | needs a GPU run; this report was finalised from the existing data only. The parity drift's cause stays unconfirmed until it runs |
| `scripts/bench-merge` | not a merge; it would append to `history.csv` |
| `cargo clippy`, `cargo test`, `ctest` | no code changed (measurement only); the default `tune` feature also needs the strided and gait repositories, which this run did not fetch |
| The `tune` feature in the measured binary | built with `--no-default-features`: the strided and gait dependencies were never fetched on this machine, and fetching them is a credential-bearing command outside this run's rules. `larch bench` without `--tune` never starts the tuner, and the engine crates do not depend on the feature |

## Protocol differences from architecture 16 (for the maintainer to amend section 16, or reject)

1. The driver gate and the W6 comparison use S pooled over 9 alternated runs a side in one hold. Section 16 still says one `--fixed-shape` run within 3 %. The pooled procedure was approved on 10 Oct (#97).
2. The Ollama column is HTTP against HTTP: larch runs behind the benchmark-only `serve_bench` adapter with `--natural-eos`, and both engines stop at EOS. Section 16 sets in-process larch (ignore_eos, exact lengths) beside Ollama "over its streaming chat API". bench/README already says raw `/api/generate`.
3. W4 runs at per-sequence context 2560 (the driver's `n_ctx_seq`) on all engines, Ollama included. Section 16's W4 row says 8192.
4. Run counts and order: W4 2 per engine (V-L-L-V); Ollama rows 3 per engine (V-O-O-V-V-O), with a fresh server for every run on both sides. Section 16 names no run counts.
5. Ollama is 0.40.1, the build installed on this machine, which this run did not check against Ollama's latest release.

## Exact commands

`T=~/src/target/final-sweep`, `M=<models>` (the `<alias>.gguf` directory), `B=$T/llama-serve-bench-wsl5070`,
`R=bench/results/final-sweep`. Every GPU job ran as `timeout <wait+job> flock -w 14400 ~/src/gpu.lock timeout <job> <command>`, one hold per line below unless noted.

```
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active --format=csv -l 30 > bench/results/final-sweep-gpu-log.csv
CARGO_TARGET_DIR=$T nice -n 19 cargo build --release --locked -p larch --no-default-features --bin larch --example serve_bench
nice -n 19 python3 scripts/baseline.py --machine wsl5070 --build $B build

$T/release/larch parity -m $M/m1.gguf --ref <oracle>/7b064f58…/cublas-ub32.jsonl --bound 0.0880593441491973 --margin 0.703125 --mode all
$T/release/larch parity -m $M/m5.gguf --ref <oracle>/65b8fcd9…/cublas-ub32.jsonl --bound 0.06534985856703847 --margin 1.234375 --mode all

python3 scripts/baseline.py --machine wsl5070 --models $M --build $B --out $R --alias <m> flags
python3 scripts/baseline.py --machine wsl5070 --models $M --build $B --out $R --alias <m> --row fixed run    # the gate

# W4 vs llama.cpp, c in 1 8 16, V-L-L-V
$T/release/larch bench -m $M/<m>.gguf --mode serve --concurrency <c> --set max_context=2560 --machine wsl5070 --json $R/wsl5070-<m>-larch-w4-<c>-r<k>.json
python3 scripts/baseline.py --machine wsl5070 --models $M --build $B --out $R --alias <m> --row w4-<c> run    # renamed to wsl5070-<m>-llama-w4-<c>-r<k>.json
python3 scripts/baseline.py --machine wsl5070 compare $R/wsl5070-<m>-llama-w4-<c>-r<i>.json $R/wsl5070-<m>-larch-w4-<c>-r<j>.json    # i, j in 1 2

# W6, one hold per model: 9 x (larch, driver)
$T/release/larch bench -m $M/<m>.gguf --mode batched --pp 512 --tg 128 --npl 1,4,8,16 --machine wsl5070 --json $R/wsl5070-<m>-larch-w6-r<i>.json
$B/bin/llama-serve-bench fixed -m $M/<m>.gguf -ngl 99 -fa on -ctk f16 -ctv f16 -t <t> -b <b> -ub <ub> -npp 512 -ntg 128 -npl 1,4,8,16 -c 16384 --machine wsl5070 --json $R/wsl5070-<m>-llama-w6-r<i>.json

# Ollama models, one hold: a server, then
PATH=<ollama-0.40.1>/bin:$PATH python3 scripts/baseline.py --machine wsl5070 --models $M --build $B --out $R --alias m1 --alias m5 ollama

# vs Ollama, per row V-O-O-V-V-O; each run one hold with its own server
$T/release/examples/serve_bench --model $M/<m>.gguf --context <ctx> --sequences <N> --requests <turns+1> --port 11436 &
$T/release/larch bench -m $M/<m>.gguf --engine larch-http --url http://127.0.0.1:11436 --natural-eos <row> --concurrency <N> --machine wsl5070 --json $R/wsl5070-<m>-larch-http-<row>-r<k>.json
PATH=<ollama-0.40.1>/bin:$PATH python3 scripts/baseline.py --machine wsl5070 --parallel <N> ollama-serve &
$T/release/larch bench -m $M/<m>.gguf --engine ollama --url http://127.0.0.1:11434 <row> --concurrency <N> --machine wsl5070 --json $R/wsl5070-<m>-ollama-<row>-r<k>.json
#   w1:    --workload chat                        N 1,  ctx 8192, 13 requests
#   w2-8:  --workload chats                       N 8,  ctx 8192, 65 requests
#   w4-8:  --mode serve --set max_context=2560    N 8,  ctx 2560, 201 requests
#   w4-16: --mode serve --set max_context=2560    N 16, ctx 2560, 201 requests

# cache-on, one hold per model
LARCH_PREFIX_CACHE_BLOCKS=512 $T/release/larch bench -m $M/<m>.gguf --mode serve --concurrency 8 --set max_context=2560 --machine wsl5070 --json $R/wsl5070-<m>-larch-cacheon-w4-8-r1.json
```

The flags CSVs had llama-bench's local model path reduced to `<alias>.gguf`, as `baseline.py` does for its llama-bench
JSON; no other file was edited after its run.

## Files (`final-sweep/`)

`wsl5070-<alias>-<engine>-<row>-r<k>.json`, with engine `larch` (in process), `llama` (llama-serve-bench),
`larch-http`, `ollama` or `larch-cacheon`, and row `w4-1`, `w4-8`, `w4-16`, `w6`, `w1`, `w2-8`. The gate is
`wsl5070-<alias>-batched.jsonl` and `-fixed.json` (9 runs each, `baseline.py`'s names), the flag search is
`wsl5070-<alias>-llama-flags.{csv,json}`, and parity is `wsl5070-<alias>-larch-parity.txt`. 112 run results in total.

The committed JSONs have the per-request gap arrays (`token_gaps_ms`, `http_chunk_gaps_ms`) removed to keep the repo
lean (65 MB → about 8 MB); every summary, including the token-gap and chunk-gap percentiles, is unchanged, and
`baseline.py compare` still accepts the stripped W4 pairs. The stall fractions above were counted from the full arrays
before stripping; the full set is kept outside git.

`started_at` in the llama.cpp JSONs (`wsl5070-<alias>-llama-*.json`) is the run's end time, not its start:
`tools/llama-serve-bench/main.cpp` writes the header after `srv.run()`. The runs were not concurrent with larch's.

## Detail (generated from the raw JSON; spread V / B = larch / baseline)

### m1 detail

#### m1 driver gate (llama-batched-bench vs llama-serve-bench fixed, S over 9 alternated runs a side)

| npl | llama-batched-bench S | driver S | driver vs batched-bench | gate (3 %) |
|---|---|---|---|---|
| 1 | 604.8 (n=9) | 605.9 (n=9) | +0.18% | pass |
| 4 | 1762.6 (n=9) | 1765.8 (n=9) | +0.18% | pass |
| 8 | 2710.4 (n=9) | 2669.9 (n=9) | -1.49% | pass |
| 16 | 3689.9 (n=9) | 3667.2 (n=9) | -0.62% | pass |

#### m1 W4 at 1: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 123.9 [122.2-125.6] | 129.0 [128.7-129.3] | -4.0 % (spread V 2.7 %, B 0.5 %): larch worse |
| TTFT p50 ms | 80.1 [78.6-81.6] | 57.3 [56.9-57.8] | +39.7 % (spread V 3.7 %, B 1.6 %): larch worse |
| TTFT p99 ms | 624.5 [606.8-642.1] | 277.5 [276.5-278.6] | +125.0 % (spread V 5.6 %, B 0.8 %): larch worse |
| TPOT p50 ms | 7.52 [7.38-7.67] | 7.46 [7.43-7.50] | +0.8 % (spread V 3.9 %, B 0.9 %): within noise |
| TPOT p99 ms | 8.46 [8.36-8.57] | 8.18 [8.08-8.28] | +3.5 % (spread V 2.5 %, B 2.5 %): larch worse |
| token-gap p99 ms | 8.74 [8.46-9.02] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m1 W4 at 8: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 572.2 [571.5-572.9] | 507.9 [501.8-514.0] | +12.7 % (spread V 0.2 %, B 2.4 %): larch better |
| TTFT p50 ms | 85.0 [84.9-85.1] | 109.5 [108.1-111.0] | -22.4 % (spread V 0.2 %, B 2.7 %): larch better |
| TTFT p99 ms | 1397.4 [1353.4-1441.4] | 958.9 [958.3-959.6] | +45.7 % (spread V 6.3 %, B 0.1 %): larch worse |
| TPOT p50 ms | 13.08 [13.08-13.08] | 13.78 [13.69-13.87] | -5.1 % (spread V 0.0 %, B 1.3 %): larch better |
| TPOT p99 ms | 18.09 [18.02-18.17] | 34.18 [32.59-35.77] | -47.1 % (spread V 0.8 %, B 9.3 %): larch better |
| token-gap p99 ms | 110.70 [110.68-110.72] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m1 W4 at 16: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 777.8 [777.7-777.9] | 676.0 [666.0-686.0] | +15.1 % (spread V 0.0 %, B 3.0 %): larch better |
| TTFT p50 ms | 91.3 [91.1-91.5] | 116.2 [114.0-118.5] | -21.5 % (spread V 0.5 %, B 3.9 %): larch better |
| TTFT p99 ms | 2329.0 [2328.6-2329.3] | 1664.9 [1642.8-1687.0] | +39.9 % (spread V 0.0 %, B 2.7 %): larch worse |
| TPOT p50 ms | 19.31 [19.28-19.34] | 19.48 [19.21-19.76] | -0.9 % (spread V 0.3 %, B 2.8 %): within noise |
| TPOT p99 ms | 24.81 [24.79-24.83] | 43.14 [41.89-44.38] | -42.5 % (spread V 0.2 %, B 5.8 %): larch better |
| token-gap p99 ms | 153.58 [153.56-153.59] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m1 W6 fixed pp512/tg128: S pooled over 9 larch and 9 driver runs, alternated in one hold

| npl | larch S | driver S | larch vs driver | larch S_tg | driver S_tg | larch S_pp | driver S_pp |
|---|---|---|---|---|---|---|---|
| 1 | 618.2 [615.7-621.2] | 617.8 [607.5-625.9] | +0.1 % (spread V 0.9 %, D 3.0 %): within noise | 137.5 | 135.2 | 4919 | 5721 |
| 4 | 1554.2 [1537.6-1561.9] | 1802.0 [1746.1-1836.8] | -13.8 % (spread V 1.6 %, D 5.0 %): larch worse | 416.6 | 451.5 | 4899 | 7146 |
| 8 | 2434.0 [2428.3-2439.2] | 2716.0 [2663.0-2751.9] | -10.4 % (spread V 0.4 %, D 3.3 %): larch worse | 807.6 | 782.8 | 4902 | 7099 |
| 16 | 3149.0 [3142.1-3155.3] | 3766.6 [3710.5-3806.2] | -16.4 % (spread V 0.4 %, D 2.5 %): larch worse | 1296.1 | 1302.4 | 4900 | 7148 |

#### m1 W1 chat at 1: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 103.6 [103.5-106.1] | 116.0 [113.0-119.2] | -10.8 % (spread V 2.5 %, B 5.3 %): larch worse |
| TTFT p50 ms | 180.8 [180.3-188.1] | 73.2 [71.0-76.7] | +146.8 % (spread V 4.4 %, B 7.8 %): larch worse |
| TTFT p99 ms | 549.0 [518.1-564.4] | 121.7 [109.9-127.5] | +351.1 % (spread V 8.4 %, B 14.5 %): larch worse |
| TPOT p50 ms | 8.09 [8.06-8.23] | 8.17 [7.85-8.38] | -1.0 % (spread V 2.1 %, B 6.6 %): within noise |
| TPOT p99 ms | 8.90 [8.34-9.10] | 8.59 [8.44-8.90] | +3.7 % (spread V 8.5 %, B 5.3 %): within noise |
| generated tokens | 2017 [2017-2017] | 2187 [2187-2187] |  |

#### m1 W2 chats at 8: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 392.6 [391.4-398.1] | 337.4 [330.2-338.2] | +16.4 % (spread V 1.7 %, B 2.4 %): larch better |
| TTFT p50 ms | 202.4 [194.3-217.0] | 153.3 [141.9-154.6] | +32.0 % (spread V 11.2 %, B 8.3 %): larch worse |
| TTFT p99 ms | 618.2 [595.3-693.7] | 675.1 [622.0-709.5] | -8.4 % (spread V 15.9 %, B 13.0 %): within noise |
| TPOT p50 ms | 18.19 [17.70-18.26] | 17.73 [17.25-17.74] | +2.6 % (spread V 3.1 %, B 2.8 %): within noise |
| TPOT p99 ms | 24.42 [22.37-25.50] | 39.86 [37.33-40.43] | -38.7 % (spread V 12.8 %, B 7.8 %): larch better |
| generated tokens | 13181 [12951-13203] | 12399 [12151-12881] |  |

#### m1 W4 at 8 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 558.3 [557.4-558.4] | 426.2 [423.9-433.4] | +31.0 % (spread V 0.2 %, B 2.2 %): larch better |
| TTFT p50 ms | 97.6 [97.4-97.8] | 198.0 [184.7-202.0] | -50.7 % (spread V 0.4 %, B 8.7 %): larch better |
| TTFT p99 ms | 1102.0 [723.8-1293.7] | 795.2 [792.3-866.1] | +38.6 % (spread V 51.7 %, B 9.3 %): within noise |
| TPOT p50 ms | 13.47 [13.44-13.53] | 16.65 [16.38-16.94] | -19.1 % (spread V 0.6 %, B 3.4 %): larch better |
| TPOT p99 ms | 18.08 [18.04-18.65] | 32.15 [25.95-32.21] | -43.8 % (spread V 3.3 %, B 19.5 %): larch better |
| generated tokens | 59256 [59190-59569] | 58950 [58838-59541] |  |

#### m1 W4 at 16 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 756.9 [756.3-759.2] | 509.5 [506.1-514.9] | +48.6 % (spread V 0.4 %, B 1.7 %): larch better |
| TTFT p50 ms | 114.8 [112.2-118.4] | 219.9 [208.2-229.3] | -47.8 % (spread V 5.4 %, B 9.6 %): larch better |
| TTFT p99 ms | 2342.9 [2327.4-2366.7] | 1913.6 [1729.0-2023.6] | +22.4 % (spread V 1.7 %, B 15.4 %): larch worse |
| TPOT p50 ms | 19.61 [19.52-19.72] | 25.81 [25.34-26.34] | -24.0 % (spread V 1.0 %, B 3.9 %): larch better |
| TPOT p99 ms | 25.24 [25.09-25.60] | 58.45 [46.79-61.93] | -56.8 % (spread V 2.0 %, B 25.9 %): larch better |
| generated tokens | 59564 [59069-59904] | 59356 [58532-60015] |  |

#### m1 W4 at 8, cache-on (LARCH_PREFIX_CACHE_BLOCKS=512, not default) vs default

| metric | cache-on (n=1) | default (n=2) | larch vs default |
|---|---|---|---|
| agg tok/s | 573.9 | 572.2 [571.5-572.9] | +0.3 % (spread V 0.0 %, B 0.2 %): within noise |
| TTFT p50 ms | 83.3 | 85.0 [84.9-85.1] | -1.9 % (spread V 0.0 %, B 0.2 %): within noise |
| TTFT p99 ms | 1362.5 | 1397.4 [1353.4-1441.4] | -2.5 % (spread V 0.0 %, B 6.3 %): within noise |
| TPOT p50 ms | 13.05 | 13.08 [13.08-13.08] | -0.2 % (spread V 0.0 %, B 0.0 %): within noise |
| TPOT p99 ms | 18.05 | 18.09 [18.02-18.17] | -0.2 % (spread V 0.0 %, B 0.8 %): within noise |
| token-gap p99 ms | 112.04 | 110.70 [110.68-110.72] | +1.2 % (spread V 0.0 %, B 0.0 %): within noise |
| generated tokens | 64437 | 64437 [64437-64437] |  |

### m5 detail

#### m5 driver gate (llama-batched-bench vs llama-serve-bench fixed, S over 9 alternated runs a side)

| npl | llama-batched-bench S | driver S | driver vs batched-bench | gate (3 %) |
|---|---|---|---|---|
| 1 | 626.5 (n=9) | 629.1 (n=9) | +0.41% | pass |
| 4 | 1828.0 (n=9) | 1837.9 (n=9) | +0.54% | pass |
| 8 | 2905.6 (n=9) | 2944.1 (n=9) | +1.33% | pass |
| 16 | 4135.5 (n=9) | 4158.0 (n=9) | +0.54% | pass |

#### m5 W4 at 1: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 135.2 [135.1-135.3] | 137.9 [137.8-137.9] | -2.0 % (spread V 0.2 %, B 0.1 %): within noise |
| TTFT p50 ms | 73.9 [73.4-74.3] | 55.5 [55.1-55.9] | +33.2 % (spread V 1.2 %, B 1.5 %): larch worse |
| TTFT p99 ms | 537.6 [537.4-537.8] | 258.2 [257.5-258.9] | +108.2 % (spread V 0.1 %, B 0.5 %): larch worse |
| TPOT p50 ms | 6.89 [6.89-6.89] | 7.00 [6.99-7.01] | -1.6 % (spread V 0.1 %, B 0.2 %): within noise |
| TPOT p99 ms | 7.62 [7.62-7.62] | 7.29 [7.26-7.32] | +4.5 % (spread V 0.0 %, B 0.8 %): larch worse |
| token-gap p99 ms | 7.70 [7.69-7.70] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m5 W4 at 8: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 621.0 [620.9-621.0] | 558.6 [545.2-572.0] | +11.2 % (spread V 0.0 %, B 4.8 %): larch better |
| TTFT p50 ms | 80.0 [78.6-81.5] | 107.5 [102.3-112.6] | -25.5 % (spread V 3.6 %, B 9.6 %): larch better |
| TTFT p99 ms | 1256.5 [1250.6-1262.4] | 906.1 [854.7-957.6] | +38.7 % (spread V 0.9 %, B 11.3 %): larch worse |
| TPOT p50 ms | 12.11 [12.11-12.11] | 12.38 [12.13-12.64] | -2.2 % (spread V 0.1 %, B 4.1 %): within noise |
| TPOT p99 ms | 16.35 [16.32-16.39] | 32.92 [31.15-34.68] | -50.3 % (spread V 0.4 %, B 10.7 %): larch better |
| token-gap p99 ms | 97.70 [97.66-97.74] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m5 W4 at 16: larch in process vs llama-serve-bench (ignore_eos, exact lengths)

| metric | larch (n=2) | llama.cpp (n=2) | larch vs llama.cpp |
|---|---|---|---|
| agg tok/s | 860.7 [859.9-861.5] | 771.1 [742.3-799.8] | +11.6 % (spread V 0.2 %, B 7.5 %): larch better |
| TTFT p50 ms | 82.8 [82.4-83.2] | 109.6 [104.7-114.5] | -24.4 % (spread V 1.0 %, B 9.0 %): larch better |
| TTFT p99 ms | 2123.9 [2109.5-2138.2] | 1828.6 [1775.9-1881.2] | +16.2 % (spread V 1.4 %, B 5.8 %): larch worse |
| TPOT p50 ms | 17.43 [17.41-17.45] | 16.82 [16.34-17.30] | +3.6 % (spread V 0.2 %, B 5.7 %): within noise |
| TPOT p99 ms | 22.22 [22.19-22.24] | 40.07 [36.68-43.46] | -44.6 % (spread V 0.2 %, B 16.9 %): larch better |
| token-gap p99 ms | 133.48 [133.44-133.52] | n/a | n/a |
| generated tokens | 64437 [64437-64437] | 64437 [64437-64437] |  |

#### m5 W6 fixed pp512/tg128: S pooled over 9 larch and 9 driver runs, alternated in one hold

| npl | larch S | driver S | larch vs driver | larch S_tg | driver S_tg | larch S_pp | driver S_pp |
|---|---|---|---|---|---|---|---|
| 1 | 661.6 [658.5-664.3] | 650.9 [643.7-660.3] | +1.6 % (spread V 0.9 %, D 2.6 %): within noise | 146.9 | 142.8 | 5334 | 5912 |
| 4 | 1633.3 [1626.8-1638.1] | 1888.9 [1838.4-1938.5] | -13.5 % (spread V 0.7 %, D 5.2 %): larch worse | 433.5 | 470.8 | 5303 | 7647 |
| 8 | 2606.0 [2598.4-2613.9] | 2993.4 [2911.4-3032.5] | -12.9 % (spread V 0.6 %, D 4.0 %): larch worse | 858.6 | 871.6 | 5305 | 7647 |
| 16 | 3397.3 [3390.7-3407.8] | 4242.2 [4162.5-4278.8] | -19.9 % (spread V 0.5 %, D 2.7 %): larch worse | 1393.8 | 1525.4 | 5303 | 7647 |

#### m5 W1 chat at 1: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 117.6 [117.1-117.7] | 124.4 [122.5-129.5] | -5.4 % (spread V 0.5 %, B 5.6 %): within noise |
| TTFT p50 ms | 148.6 [148.1-149.6] | 72.8 [71.5-72.9] | +104.0 % (spread V 1.0 %, B 1.9 %): larch worse |
| TTFT p99 ms | 447.5 [447.2-452.0] | 112.4 [109.0-122.3] | +298.0 % (spread V 1.1 %, B 11.8 %): larch worse |
| TPOT p50 ms | 7.09 [7.09-7.11] | 7.55 [7.25-7.65] | -6.1 % (spread V 0.3 %, B 5.3 %): larch better |
| TPOT p99 ms | 7.59 [7.57-7.63] | 7.95 [7.53-8.24] | -4.5 % (spread V 0.7 %, B 8.9 %): within noise |
| generated tokens | 1860 [1860-1860] | 1854 [1854-1854] |  |

#### m5 W2 chats at 8: larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 446.2 [437.8-450.1] | 407.5 [372.5-447.8] | +9.5 % (spread V 2.7 %, B 18.5 %): within noise |
| TTFT p50 ms | 170.4 [168.5-175.0] | 131.3 [128.7-140.9] | +29.8 % (spread V 3.8 %, B 9.3 %): larch worse |
| TTFT p99 ms | 603.1 [591.8-673.2] | 373.8 [356.8-383.3] | +61.4 % (spread V 13.5 %, B 7.1 %): larch worse |
| TPOT p50 ms | 15.53 [15.21-15.59] | 14.49 [14.49-14.73] | +7.2 % (spread V 2.4 %, B 1.7 %): larch worse |
| TPOT p99 ms | 19.83 [19.67-20.55] | 33.52 [22.68-37.92] | -40.8 % (spread V 4.4 %, B 45.5 %): within noise |
| generated tokens | 13545 [13508-13755] | 13527 [13289-13550] |  |

#### m5 W4 at 8 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 463.2 [459.1-469.3] | 385.3 [375.1-408.6] | +20.2 % (spread V 2.2 %, B 8.7 %): larch better |
| TTFT p50 ms | 94.6 [91.1-99.1] | 151.7 [143.7-162.5] | -37.6 % (spread V 8.5 %, B 12.3 %): larch better |
| TTFT p99 ms | 787.9 [700.1-801.1] | 825.7 [785.5-847.2] | -4.6 % (spread V 12.8 %, B 7.5 %): within noise |
| TPOT p50 ms | 15.23 [14.96-15.30] | 17.59 [17.34-18.39] | -13.4 % (spread V 2.3 %, B 5.9 %): larch better |
| TPOT p99 ms | 25.41 [23.86-26.20] | 24.73 [24.51-26.38] | +2.7 % (spread V 9.2 %, B 7.6 %): within noise |
| generated tokens | 28272 [27440-29129] | 28292 [27162-30226] |  |

#### m5 W4 at 16 (max_context 2560): larch-http vs Ollama (localhost HTTP, natural EOS)

| metric | larch-http (n=3) | Ollama (n=3) | larch vs Ollama |
|---|---|---|---|
| agg tok/s | 584.1 [583.9-600.0] | 409.8 [404.9-447.5] | +42.5 % (spread V 2.8 %, B 10.4 %): larch better |
| TTFT p50 ms | 119.2 [111.0-145.8] | 197.1 [187.9-214.2] | -39.5 % (spread V 29.2 %, B 13.3 %): larch better |
| TTFT p99 ms | 2108.0 [2048.5-2162.5] | 1643.6 [1591.8-1730.5] | +28.3 % (spread V 5.4 %, B 8.4 %): larch worse |
| TPOT p50 ms | 24.04 [23.80-24.05] | 29.93 [29.54-30.86] | -19.7 % (spread V 1.1 %, B 4.4 %): larch better |
| TPOT p99 ms | 41.50 [37.20-42.36] | 44.67 [43.42-55.49] | -7.1 % (spread V 12.4 %, B 27.0 %): within noise |
| generated tokens | 28154 [27999-28806] | 28744 [28425-28838] |  |

#### m5 W4 at 8, cache-on (LARCH_PREFIX_CACHE_BLOCKS=512, not default) vs default

| metric | cache-on (n=1) | default (n=2) | larch vs default |
|---|---|---|---|
| agg tok/s | 622.5 | 621.0 [620.9-621.0] | +0.3 % (spread V 0.0 %, B 0.0 %): within noise |
| TTFT p50 ms | 77.9 | 80.0 [78.6-81.5] | -2.7 % (spread V 0.0 %, B 3.6 %): within noise |
| TTFT p99 ms | 1265.1 | 1256.5 [1250.6-1262.4] | +0.7 % (spread V 0.0 %, B 0.9 %): within noise |
| TPOT p50 ms | 12.06 | 12.11 [12.11-12.11] | -0.4 % (spread V 0.0 %, B 0.1 %): within noise |
| TPOT p99 ms | 16.30 | 16.35 [16.32-16.39] | -0.3 % (spread V 0.0 %, B 0.4 %): within noise |
| token-gap p99 ms | 98.99 | 97.70 [97.66-97.74] | +1.3 % (spread V 0.0 %, B 0.1 %): within noise |
| generated tokens | 64437 | 64437 [64437-64437] |  |

