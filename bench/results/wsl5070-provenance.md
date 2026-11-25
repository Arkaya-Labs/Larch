# wsl5070 rows: what produced each file

**llama.cpp rows are not here.** `baseline.py --machine wsl5070 check` fails the fixed-shape honesty gate (the driver's
mirror is 3.2 to 6.7 % under `llama-batched-bench` at some shapes on the final driver; architecture 16: no driver number
is published until it is within 3 %), so they stay out of the tree until it passes. `server` (a real llama-server
against the driver, per turn) passes.

Every larch and Ollama row below ran under `flock ~/src/gpu.lock` (exclusive), one invocation per lock.
The larch rows are the "before" rows for the speed PRs (scratch build `55063bd`, never pushed). Its W4 run was stopped
on 9 Oct under the re-plan; W4 and m5 follow in A1B4 on a later build. Its own parity gate did not run (see
`parity-wsl5070.md`).

| File | Engine and build | Command | Exit | Wall incl. lock wait |
|---|---|---|---|---|
| `wsl5070-m1-ollama-w1.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 1] larch bench -m <models>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload chat --concurrency 1 --machine wsl5070 --json bench/results/wsl5070-m1-ollama-w1.json` | 0 | 62 s |
| `wsl5070-m1-ollama-w2-4.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 4] larch bench -m <models>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload chats --concurrency 4 --machine wsl5070 --json bench/results/wsl5070-m1-ollama-w2-4.json` | 0 | 27 s |
| `wsl5070-m1-ollama-w2-8.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 8] larch bench -m <models>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload chats --concurrency 8 --machine wsl5070 --json bench/results/wsl5070-m1-ollama-w2-8.json` | 0 | 48 s |
| `wsl5070-m1-ollama-w3-1.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 1] larch bench -m <models>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload agent --concurrency 1 --machine wsl5070 --json bench/results/wsl5070-m1-ollama-w3-1.json` | 0 | 27 s |
| `wsl5070-m1-ollama-w3-3.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 3] larch bench -m <models>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload agent --concurrency 3 --machine wsl5070 --json bench/results/wsl5070-m1-ollama-w3-3.json` | 0 | 38 s |
| `wsl5070-m1-larch-gen-w1.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chat --replay generated --concurrency 1 --json bench/results/wsl5070-m1-larch-gen-w1.json` | 0 | 241 s |
| `wsl5070-m1-larch-gen-w2-4.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chats --replay generated --concurrency 4 --json bench/results/wsl5070-m1-larch-gen-w2-4.json` | 0 | 233 s |
| `wsl5070-m1-larch-gen-w2-8.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chats --replay generated --concurrency 8 --json bench/results/wsl5070-m1-larch-gen-w2-8.json` | 0 | 281 s |
| `wsl5070-m1-larch-gen-w3-1.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload agent --replay generated --concurrency 1 --json bench/results/wsl5070-m1-larch-gen-w3-1.json` | 0 | 363 s |
| `wsl5070-m1-larch-gen-w3-3.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload agent --replay generated --concurrency 3 --json bench/results/wsl5070-m1-larch-gen-w3-3.json` | 0 | 398 s |
| `wsl5070-m1-larch-w1.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chat --concurrency 1 --json bench/results/wsl5070-m1-larch-w1.json` | 0 | 258 s |
| `wsl5070-m1-larch-w2-4.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chats --concurrency 4 --json bench/results/wsl5070-m1-larch-w2-4.json` | 0 | 236 s |
| `wsl5070-m1-larch-w2-8.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload chats --concurrency 8 --json bench/results/wsl5070-m1-larch-w2-8.json` | 0 | 291 s |
| `wsl5070-m1-larch-w3-1.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload agent --concurrency 1 --json bench/results/wsl5070-m1-larch-w3-1.json` | 0 | 371 s |
| `wsl5070-m1-larch-w3-3.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload agent --concurrency 3 --json bench/results/wsl5070-m1-larch-w3-3.json` | 0 | 425 s |
| `wsl5070-m1-larch-w5.json` | larch `55063bd` (dev `4b2e30d` + A1B3 + F4E3 `50685dd`, #61), CUDA release | `larch bench -m <models>/m1.gguf --machine wsl5070 --workload longprompt --concurrency 1 --json bench/results/wsl5070-m1-larch-w5.json` | 0 | 37 s |
| `wsl5070-m5-ollama-w1.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 1] larch bench -m <models>/m5.gguf --engine ollama --url http://127.0.0.1:11434 --workload chat --concurrency 1 --machine wsl5070 --json bench/results/wsl5070-m5-ollama-w1.json` | 0 | 32 s |
| `wsl5070-m5-ollama-w2-4.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 4] larch bench -m <models>/m5.gguf --engine ollama --url http://127.0.0.1:11434 --workload chats --concurrency 4 --machine wsl5070 --json bench/results/wsl5070-m5-ollama-w2-4.json` | 0 | 29 s |
| `wsl5070-m5-ollama-w2-8.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 8] larch bench -m <models>/m5.gguf --engine ollama --url http://127.0.0.1:11434 --workload chats --concurrency 8 --machine wsl5070 --json bench/results/wsl5070-m5-ollama-w2-8.json` | 0 | 43 s |
| `wsl5070-m5-ollama-w3-1.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 1] larch bench -m <models>/m5.gguf --engine ollama --url http://127.0.0.1:11434 --workload agent --concurrency 1 --machine wsl5070 --json bench/results/wsl5070-m5-ollama-w3-1.json` | 0 | 26 s |
| `wsl5070-m5-ollama-w3-3.json` | Ollama 0.40.1 (`ollama serve`, D15 env), driven by larch bench at `55063bd` | `[ollama serve --parallel 3] larch bench -m <models>/m5.gguf --engine ollama --url http://127.0.0.1:11434 --workload agent --concurrency 3 --machine wsl5070 --json bench/results/wsl5070-m5-ollama-w3-3.json` | 0 | 39 s |

## A1B4: larch W4 (`--mode serve`, chat.jsonl, `--set max_context=2560`)

Build `426c5bf`: dev `2ea8a24` (no #87, no F4E3) + A1B3 + `--slice`; CUDA release. Each `--slice K/N` part is one
exclusive `flock ~/src/gpu.lock` hold under the 20-minute cap; `python3 scripts/bench-join` merges the parts into the
row (wall time = the parts' sum). Parts are kept beside the row; the scoreboard reads only the joined row.
Each part ramps up and drains on its own, so a joined row's tok/s is a little below one unbroken run's (the
more parts, the more tails). Hold times below are each part's run; the runner log's times also count the wait for the
lock.

| Row | Parts | Joined |
|---|---|---|
| `wsl5070-m1-larch-w4-16.json` | 2 (383 s, 400 s holds), 0 failed | 200 requests, 0 failed, 84.84 tok/s over 759.5 s |
| `wsl5070-m1-larch-w4-8.json` | 2 (about 630 s and 565 s holds), 0 failed | 200 requests, 0 failed, 57.03 tok/s over 1129.9 s |
| `wsl5070-m1-larch-w4-4.json` | 3 (holds of about 600 s each), 0 failed | 200 requests, 0 failed, 36.14 tok/s over 1783.2 s |
| `wsl5070-m5-larch-w4-16.json` | 2 (369 s, 371 s holds), 0 failed | 200 requests, 0 failed, 88.44 tok/s over 728.6 s |
| `wsl5070-m5-larch-w4-8.json` | 2 (541 s, 534 s holds), 0 failed | 200 requests, 0 failed, 60.12 tok/s over 1071.8 s |
| `wsl5070-m1-larch-w4-1.json` | 8 (runs of 665-788 s), 0 failed | 200 requests, 0 failed, 11.20 tok/s over 5751.0 s |
| `wsl5070-m5-larch-w4-4.json` | 3 (runs of 547-575 s), 0 failed | 200 requests, 0 failed, 38.10 tok/s over 1691.3 s |
| `wsl5070-m5-larch-w4-1.json` | 8 (runs of 648-743 s), 0 failed | 200 requests, 0 failed, 11.61 tok/s over 5550.8 s |
