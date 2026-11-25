# Tuned profiles milestone 0: runbook for other machines

Milestone 0 asks one question: is per-machine calibration of `prefill_chunk_tokens` / `schedule_policy` worth
building, or does a fixed rule ("always set `prefill_chunk_tokens` to 256") do as well? The gate wants an answer on
three machine classes (a 4 GB laptop, a 12-16 GB desktop, Apple silicon) with three catalog models each. The 4 GB
laptop is done for the one model that fits (`bench/results/tuned-profiles-m0-sm86.md`). This file is how to run
the same experiment on the 5070 Ti desktop and the Mac, and how to read the result.

Everything runs through `scripts/tuned_profiles_m0.py` (Python 3.11+, stdlib only; `run` and `analyze`). Its tests:
`python3 -m unittest scripts/test_tuned_profiles_m0.py` (about 80 s, no GPU needed).

## 1. What to run where

| machine | `--machine` | GPU memory | catalog models that fit | backend |
|---|---|---|---|---|
| RTX 3050 Laptop 4 GB, WSL2 | `sm86` | 4 GB | m3 (done) | CUDA, arch 86 |
| RTX 5070 Ti desktop, WSL2 | `wsl5070` | 16 GB | m1, m3, m5 | CUDA, arch 120 |
| Apple M5, 24 GB | `mac` | unified | m1, m3, m5 | Metal |

Catalog = `use` contains `"catalog"` in `bench/models.toml`. m1 and m5 need 7 GB, m3 2 GB. Run m3 first on each
machine (fast, and it is the one model all three machines share), then m1, then m5.

Native Windows isn't supported by the runner (it uses `fcntl`, `pgrep` and POSIX paths): on the 5070 Ti machine
use WSL2.

## 2. Setup (once per machine)

```
git switch dev
export LARCH_MODELS=~/larch-models                         # any directory; files are <alias>.gguf
python3 scripts/baseline.py --machine <machine> --models $LARCH_MODELS --alias m3 --alias m1 --alias m5 models
```

`baseline.py models` downloads each GGUF and checks its sha256 against `bench/models.toml`.

Build the engine without gait or strided linked (nothing tunes live during the experiment):

```
# 5070 Ti (WSL2). CUDAHOSTCXX as in scripts/baseline.py's wsl5070 entry.
LARCH_BACKEND=CUDA LARCH_CUDA_ARCH=120 CUDAHOSTCXX=/usr/bin/g++-14 \
  cargo build --release --locked -p larch --no-default-features

# Mac
LARCH_BACKEND=METAL cargo build --release --locked -p larch --no-default-features
```

Check that the binary runs the bench and that a `--set` takes effect before spending GPU hours:

```
target/release/larch bench -m $LARCH_MODELS/m3.gguf --workload chat --concurrency 1 --machine <machine> \
  --set prefill_chunk_tokens=256 --json /tmp/smoke.json
python3 -c "import json; print(json.load(open('/tmp/smoke.json'))['knob_timeline'])"
```

The `knob_timeline` must list `prefill_chunk_tokens` 256 with reply `startup`.

## 3. Conditions while measuring

The runner checks what it can and voids a pair (repeated once) when something interferes. It can't see everything,
so before a stage:

- **AC power.** WSL: read from Windows. Mac: `pmset -g batt`. A battery reading voids the pair.
- **Nothing else on the GPU.**
  - NVIDIA: `nvidia-smi` compute processes. Under WSL that list doesn't show *Windows* programs, so close games,
    video, and GPU-heavy browser tabs on the Windows side yourself.
  - Mac: no tool lists GPU clients without root. The runner stops for these process names instead: `larch`,
    `larch-agent`, `llama-server`, `llama-cli`, `llama-bench`, `ollama`, `LM Studio`, `mlx_lm`. Quit the Larch
    app and Ollama (it keeps a server running) before a stage.
- **No compiles.** `cargo`, `rustc`, `nvcc`, `clang`, … on the host make the runner give the GPU back (exit 3, or
  wait with `--wait`). On the 3050 one build moved single-chat throughput 79 → 64 tok/s.
- **Temperature.** NVIDIA: each block starts at or below `--temp-max` (default 70 °C), waiting up to `--cool-max`
  (300 s); the CSV logs start/peak temperature, clocks and the enforced power limit. Mac: there is no reading without
  root, so the columns stay empty and no band applies. Use `--idle 15` there (15 s rest before every block instead
  of 5) and keep the machine awake: prefix the command with `caffeinate -dims`.
- **Same power settings for the whole run.** Mac: Low Power Mode off. Laptops: the same power mode throughout.

## 4. The procedure for one model

Each stage is one command. It resumes from `blocks.csv` when rerun, takes the GPU lock (`--lock`), holds it at most
`--hold-min` minutes (default 50), and with `--wait` gives it back for `--retry-s` (300 s) whenever the host gets busy
or the hold runs out, then carries on. One `--out` directory per machine; several models can share it.

```
BIN=target/release/larch; MACHINE=wsl5070            # or mac
M=$LARCH_MODELS/m3.gguf                               # then m1, then m5
OUT=~/tp-m0/$MACHINE                                   # per-block JSONs and blocks.csv
RUN="python3 scripts/tuned_profiles_m0.py run --bin $BIN --model $M --machine $MACHINE --out $OUT --lock ~/src/gpu.lock --wait"
# Mac: RUN="caffeinate -dims $RUN --idle 15"

$RUN --b std2 --pairs 10                               # 1. A/A noise (standard against standard)
$RUN --b c256:prefill_chunk_tokens=256 --pairs 9       # 2. always-256 (also the first candidate)
$RUN --b c128:prefill_chunk_tokens=128 --pairs 9       # 3. candidate
$RUN --b c384:prefill_chunk_tokens=384 --pairs 9       # 4. candidate
python3 scripts/tuned_profiles_m0.py analyze $OUT/blocks.csv > $OUT/analysis.md
```

Read `## Hand calibration: <model>` in `analysis.md`. Take the chunk among c256 / c128 / c384 that passes every test
with the largest stall cut (call it B; if none passes, stop here for this model: see section 6). Then one alternative
policy at B. On the 3050, `balanced` was used because it sits between the default (`decode_first`) and
`prefill_first`; use the same one so the machines compare:

```
$RUN --b cB-bal:prefill_chunk_tokens=B,schedule_policy=balanced --pairs 9    # 5. policy at the best chunk
python3 scripts/tuned_profiles_m0.py analyze $OUT/blocks.csv > $OUT/analysis.md
```

The analysis now names the winner. Confirm it on fresh blocks, with exactly the winner's `--set` values and the
arm name `cal` (the analysis treats `std-vs-cal` as the single-look confirmation and the gate reads it by that name):

```
$RUN --b cal:prefill_chunk_tokens=B,schedule_policy=balanced --pairs 9      # 6. confirmation (winner's values)
python3 scripts/tuned_profiles_m0.py analyze $OUT/blocks.csv > $OUT/analysis.md
```

If the winner is plain c256 or c384, the confirmation is `--b cal:prefill_chunk_tokens=256` (or 384).

Special cases:

- **A pair lost twice** (voided, then voided again on its repeat) leaves that type short of 9. Run one more pair for
  that type only: `$RUN --b <same arm> --pairs 10 --types conc`.
- **"pending" on a candidate's concurrent type** means look 1 (9 pairs, p < 0.005) wasn't passed. If the p value is
  small and the interval is near the 20 % / 100 ms bar, run look 2: the same stage with `--pairs 18`. If the
  interval is far from the bar (as on the 3050, where the A/A noise was 0.2 %), 9 more pairs can't move it; say so.
- **Exit 4** is a GPU fault: stop and report (brief section 8). A fault, failed request or timeout in a candidate arm
  is never voided; it rejects that arm.
- A stage stopped by hand (Ctrl-C, reboot) resumes with the same command. If the host went down *during* a pair,
  void that pair by hand in `blocks.csv` (put the reason in its `void` column) and rerun: it is repeated once.

## 5. Time budget and the one-hour stop

`analyze` prints `## Time per block`. A full confirmation is 9 × (2 × the four types' process times + 8 × the idle
rest). The brief says to stop and report if a confirmation would take more than an hour on the machine; work it out
from the A/A stage before starting the candidates.

| machine, model | blocks per pair (s) | 9-pair stage |
|---|---|---|
| 3050, m3 (measured) | 2 × 134 + 8 × 5 ≈ 309 | about 47 min |
| 5070 Ti, m1 / m3 / m5 | not measured | expect shorter than the 3050's m3 |
| Mac, m1 / m5 | not measured | likely over an hour: decode at one chat is about 26 tok/s for m1 (`mac-m1-metal-perf.md`) |

If a model goes over the hour, that is a result for the gate in itself (calibration would cost more than an hour of
exclusive GPU per stage there). Report it. Run on only if the owners lift the stop; screening candidates on
`--types conc,doc` halves a candidate stage, but the confirmation needs all four types.

## 6. How to decide whether the feature is worth building

### Per (machine, model) pair: the analysis does it

`analyze` prints, per model, the candidate table, the winner, and a gate line. On the 3050 with m3:

```
| std-vs-c256    | 406 ms, 41 % | 632 ms, 44 % | no: harm or failure on conc |
| std-vs-c384bal | 139 ms, 14 % | 319 ms, 22 % | yes |
Gate (m3): calibrated passes; always-256 harms conc; calibrated cut ≥ 1.10 × always-256's on: none;
avoids an always-256 no-harm failure: yes.
This pair counts toward **build calibration**; fallback for this pair: **ship nothing**.
```

The rules behind it (brief sections 6 and 7; margins are proposals the owners may change):

- **An arm passes every test** when it ran all four types, broke no no-harm bar on any of them (throughput ≥ 0.97 ×
  standard, TTFT p50 ≤ 1.10 ×, single-chat median token gap ≤ 1.03 ×, all as one-sided 95 % bounds), had no failure,
  and is confirmed on at least one concurrent type: the longest token gap fell by ≥ 20 % and ≥ 100 ms at the
  interval's conservative end, with the exact test passing.
- **A pair counts toward "build calibration"** when the confirmed calibrated arm passes every test, regresses on
  nothing, and either cuts the stall at least 10 % more than always-256 (calibrated cut ≥ 1.10 × always-256's on a
  concurrent type), or avoids a no-harm failure that always-256 commits.
- **Fallback for a pair** that doesn't count: ship always-256 there if it passes no-harm, ship nothing if it fails.

### Across pairs: the gate

Collect one row per (machine, model):

| machine | model | always-256 passes no-harm? | calibrated winner | counts toward build? | stall cut 256 / calibrated (conc, doc) | 9-pair stage time |
|---|---|---|---|---|---|---|
| 3050 | m3 | no (conc TTFT +12.4 %) | 384 + balanced | yes (avoids the failure) | 406 / 141 ms, 632 / 321 ms | 47 min |
| 5070 Ti | m1 | | | | | |
| 5070 Ti | m3 | | | | | |
| 5070 Ti | m5 | | | | | |
| Mac | m1 | | | | | |
| Mac | m3 | | | | | |
| Mac | m5 | | | | | |

**Build calibration** only if at least 30 % of the pairs count toward it and the calibrated arm regresses beyond the
margins on none. Otherwise ship always-256 where it passes no-harm, nothing where it doesn't.

### Read past the verdict

The gate is mechanical. Before recommending, check what it hides:

1. **Is the evidence sound?** A/A false confirms must be 0, and the A/A spread of the stall small (0.2-0.6 % on the
   3050; if it is above about 5 %, 9 pairs are weak and the verdicts need look 2). Every arm must show the same
   `gen_tokens` and `evaluated_tokens` per type (identical work). No order effect bigger than the noise.
2. **Is the winner the same everywhere?** This is the question that decides whether calibration is worth *building*,
   as opposed to choosing a better fixed rule. If every pair's winner is the same setting (on the 3050 it was
   384 + `balanced`), a fixed rule with that setting gets the same result with no profile store, no calibration
   time and no stale profiles: ship the rule. Calibration only earns its cost if the winners *differ* between
   machines or models **and** no single setting passes no-harm on all of them.
3. **How big is the trade?** On the 3050, always-256 fails TTFT by 2.4 points (+57 ms with four chats) while cutting
   the worst stall by 406-632 ms, and the safe winner gives up two thirds of that cut. A gate verdict that rests on
   one margin narrowly missed is worth reporting with the raw numbers, so the owners can decide whether the margin
   is right.
4. **What does calibration cost on that machine?** On the 3050 the hand procedure was five 47-minute stages
   (about 4 hours of exclusive GPU) per model. A feature that needs hours per model per machine needs a much larger
   benefit than one that needs seconds. The cheaper alternative in the literature is Sarathi-Serve's one-time
   profiling: at load, time a few prefill chunks and take the largest whose step stays under a token-gap target.
   It isn't measured here; if winners differ between machines, it is the next thing to try before a full
   calibration system.

Summing up, the possible outcomes:

| what the pairs show | recommendation |
|---|---|
| always-256 passes no-harm on (nearly) every pair | ship always-256; don't build calibration |
| always-256 fails somewhere, one other fixed setting passes everywhere | ship that fixed setting; don't build calibration |
| winners differ, no fixed setting passes everywhere, ≥ 30 % of pairs count | calibration is worth building; consider the load-time rule first |
| no candidate passes on a pair | ship nothing there; the knob doesn't help that machine |

## 7. Reporting

Write `bench/results/tuned-profiles-m0-<machine>.md` in the shape of the sm86 file: setup (commit, driver, GPU,
power state), the commands, the five answers (noise, order, budget, comparison, recommendation), what wasn't run
and why. Copy `blocks.csv` and the three generated workload files (`short.jsonl`, `long.jsonl`, `doc.jsonl`) into
`bench/results/tuned-profiles-m0-<machine>/`; the per-block bench JSONs are large and can stay out of git. Use
model aliases only (m1, m3, m5), never model names. Fill your rows into the table in section 6 and push to your own
branch.
