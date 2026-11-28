# m13: the expert spill by default, 27 Nov 2026 (evening)

Test machine: RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, Windows 11 + WSL2 (8 logical CPUs in WSL). Model m13
(`bench/models.toml`). Raw results: [`35b-spill-default-2026-11-27/`](35b-spill-default-2026-11-27/). Follows
[`35b-expert-spill-2026-11-27.md`](35b-expert-spill-2026-11-27.md) (#21), whose head this branch starts from.

## Summary

**Kept.** A plain load of m13 on a CUDA card (no `LARCH_EXPERT_CACHE`, no file beside the model) now takes #21's
capped split from a profile that ships in the binary, plans once (20 split layers, one `larch_expert_cache_plan` line),
writes no histogram, and runs #21's opt-in numbers: **161.6 tok/s at W4@1** (#21: 161.8, −0.1 %) and **300.6 at W4@4**
(#21: 298.2, +0.8 %). Against A (#21's head at default: whole layers and the prefill pool) the A B B A slices give
**+19.4 % at W4@1** and **+21.9 % at W4@4**, TPOT p99 −16.7 % and −18.2 %. `LARCH_EXPERT_CACHE=0` gives A's plan back
(237.4 against A's 237.7 tok/s, the same plan line); m1's and m5's fit lines are A's, line for line. Parity `--mode all`
on the default load passes every mode. The seats default stays at 4: the 8-seat plan lost 3.0 % at W4@1 and 4.6 % at
W4@4 (D5, the table below is the decision's). #21 is not merged yet, so there is no pull request: the branch is pushed
and the commands that open it once #21 is in close this report.

| Load | larch B, default (`e7c402e`) | llama.cpp ncm 3 | ncm 4 | the fewest ncm that fits its free memory | larch A |
|---|---|---|---|---|---|
| W4@1 | 161.6 | 130.4: **+23.9 %** | 125.3: +29.0 % | ncm 4: +29.0 % | 135.4 (slices): **+19.4 %** |
| W4@4 | 300.6 | 262.4: **+14.6 %** | 232.1: +29.5 % | ncm 5, 224.6: +33.8 % | 237.7 (slices): **+21.9 %** |
| W4@8 | 300.4 | 302.7 (today, straight after B): −0.8 %, noise | 288.3 (today): +4.2 % | ncm 7, 239.2: +25.6 % | 245.3 (#21's A, `ef28629`): +22.5 % |
| W4@16 | 297.2 / 301.1 (two runs) | 68.5 | 71.6 | ncm 9, 235.1: +27.2 % | 246.1 (#21's A, `ef28629`): +21.6 % |

B's W4@16 is the mean of its two runs (299.2). llama.cpp cells are #21's (27 Nov) except W4@8's ncm 3 and ncm 4, run
again tonight because B landed within 10 % of them (their CUDA0 buffers 1,061 and 697 MiB past the 15,009 MiB free
that llama.cpp reported, `m13-llama*-load.txt`); past its free memory at W4@16, ncm 8 ran 243.5 and ncm 7 266.3
(B +22.9 % and +12.3 %). A at W4@1 and W4@4 is tonight's slices of `6e011de`'s code; at W4@8 and W4@16 it is
#21's full runs of `ef28629` (whole layers, the same `larch_host_experts_plan,4`). One full run a cell for B; the A B B A slices are the controlled comparison with A.

## Plan

1. D1: a fake-CUDA loader test that a split load leaves the histogram beside the model unchanged (red), then only
   dynamic residency records (green).
2. D2: embed `bench/routes/m13.experts` with m13's file length and FNV-1a of its GGUF header; `profile()` takes the
   histogram beside the model, else the shipped one; tests: the match, one byte longer, a file beside wins.
3. D3: unset, a hybrid MoE on CUDA with a usable profile tries the static split (`=0` turns it off; the split attempt
   plans without the pool, a fallback has it back); tests: the default plan, `=0`, profile-less and other files as A.
4. D4: find the 173 MB the post-load check missed, charge it before the first plan; 5 loads' plan lines, load times.
5. Build B; A is #21's head binary.
6. D5: X Y Y X slices of the 8-seat plan against the 4-seat one at W4@1 and W4@4, full W4@16 on 8 seats; the rule.
7. P: fmt, clippy, workspace tests, ctest; `/code-review high`, `/ponytail-review` until Ship.
8. M (fresh-install protocol): A B B A at W4@4 and W4@1, full W4@1/4/8/16 on B, `=0` at W4@4, m1 and m5 fit lines.
9. m13 parity `--mode all` on the default load.
10. R: this report; the dev check and the PR, or the commands for it.

## What changed (`crates/larch-engine`, one message in `crates/larch-model`)

- **The default (D3).** With `LARCH_EXPERT_CACHE` unset, a hybrid MoE load on CUDA (the backend that reads host experts
  in place, where `cheapest_split`'s costs were measured) plans the static split from its profile; `cheapest_split`
  still decides, and a split that does not fit or that it prices over whole layers loads whole layers. HIP, Metal and
  the CPU keep whole layers. `=0` (or any other value) keeps whole layers and the prefill pool; `=1` and `=2` mean what
  they did (a shipped profile counts for them too). The split attempt plans without the prefill pool, as `=1` did; the
  whole-layer fallback gets the pool back where the cache was not asked for, so its plan is A's.
- **The profile (D2).** `profile()` takes the histogram beside the model when it has counts, else a profile shipped in
  the binary for that very file: `PROFILES` holds m13's (`include_bytes!` of `bench/routes/m13.experts`, 81,944 B) with
  the file's length (17,117,951,808 B) and FNV-1a 64 of its GGUF header (the 10,990,912 bytes before the data section,
  `0x6a18da79eac085e8`). The header is read again only for a file of a listed length (never the weights); `Histogram::parse`
  checks the profile's own length, layer and expert counts as for the file beside the model.
- **No skewed histogram (D1).** A static split load records no routes: its split layers' decode and check rows run in
  place inside the graph and go uncounted, so the file it wrote drifted colder with use. Dynamic residency (`=2`) has no
  in-place path (its slots move between steps), so its split layers count every row from their own download: it keeps
  recording, and only it is charged the route log. A plain whole-layer load (no profile, or a split that did not fit)
  records nothing either (the review's finding below); `=1` and `=2` record whole layers as before.
- **One plan (D4).** The first per-expert plan reserved no free swing while the check after the load demands it
  (68 MiB, `FREE_SWING`), and the load takes about 102 MB past its plan's terms (A's whole-layer loads miss as much past
  their ledger, inside a layer's rounding). The first plan now keeps the swing in its reserve and charges `first_charge`:
  98 MiB on CUDA and the re-plan's margin (one expert and the swing, so a low reading at the check still passes). The
  re-plan stays for a card where the figure is short.
- **`replan_due`** is unchanged: a split load's split layers are its host layers, so seats re-plan it as whole layers'
  plan; the test now says so for the split. The loader's comment and ponytail marker say what is true now.
- **Not changed:** `kernels/include/larch.h`, `crates/larch-records`, the seats default (D5), any kernel.

## D1: a split load writes no skewed histogram

`a_static_split_load_writes_no_histogram` (fake CUDA, `=1`, a hot histogram beside the tiny hybrid MoE): the file's
bytes and inode before and after a split load and its drop are equal. Red before the change ("the file beside the
model was rewritten": the drop's save renames a new file over it), green after. The test checks the load writes
nothing at all rather than stepping: the fake card's split layers run on the CPU device, where every row is counted, so
only the in-place path of a real card skews the counts; a load that writes nothing cannot write skewed counts.
Dynamic residency (`=2`) has no in-place rows (its slots move between steps, so `mixed_rows` never runs), its split
layers count from their own download, and its route log counts the resident layers: the skew does not apply, so it keeps
recording. On the card, every M run found no `.experts` file beside any model before or after it.

The review found that a plain load that kept whole layers recorded too (with the variable unset, every hybrid MoE on CUDA
now builds a profile): a model with no profile wrote a histogram on its first run and split from it on the next.
`a_plain_whole_layer_load_writes_no_histogram` was red ("a plain load wrote a histogram"), then whole layers record only
where `=1` or `=2` asks (green; `=1` still writes one, as on A).

## D2: the profile ships with larch

`PROFILES` in `crates/larch-engine/src/lib.rs`: (m13's length, FNV-1a 64 of its header, `include_bytes!` of
`bench/routes/m13.experts`). `profile()` runs after `Histogram::for_load` (which checks the expert tensors' shapes
and bytes against the file, as before): a histogram beside the model with counts wins, else the shipped profile whose
length and header id match the file, parsed by `Histogram::parse` against the file's length, layers and experts.
Tests: `a_plain_load_splits_from_its_shipped_profile` (fake CUDA): the tiny model's own profile (its length and header
id) splits; the same profile for a file one byte longer plans what `=0` plans; a histogram beside the model wins over
the shipped one (an even file beside gives its own plan with or without the shipped profile, and not the shipped
one's). `the_shipped_profiles_are_their_models_histograms`: every shipped profile parses as a 40-layer, 256-expert
histogram with counts, and `header_id` is FNV-1a 64 (cross-checked against Python on a fixed input) and refuses a file
shorter than the header. On the card the plain load of m13 picked it: `larch_expert_cache_plan,20,9319` with no file
beside the model and the variable unset, on all 19 of tonight's 4-seat serve loads of B. A file of another length (another copy or quant of the model) matches
no profile by construction; no such load was measured.

## D3: the default

`expert_cache(v, default)`: unset gives the static split where `default` holds (a hybrid MoE on CUDA); `"1"`, `"2"` and
every other value mean what they did. `expert_cache_is_static_by_default_where_it_reads_in_place` covers both. The
fake-CUDA test above covers the plan lines: the default with the shipped profile splits (2 split layers, more weights
on the device than whole layers' 651,744 B), `=0` and no profile give the whole-layer plan with the pool, `(2 host
layers, 651,744 B on the device, 4 slots)`, byte for byte, and `=1` gives the default's split. On the card: the `=0` slice
shows A's `larch_host_experts_plan,4,…,648016256,…,318767104,true` and `larch_stream_pool,184549376,…` exactly; m1's
and m5's four fit lines each (`dense-*-fit.txt`) are identical between A and B. `only_a_differing_plan_replans` passes,
extended with the split (its split layers are its host layers: seats past its slots re-plan it, slots it holds do not,
a split that kept every expert never does).

## D4: plan once

Cause, from #21's fit lines: the first plan came 173,162,418 B short on 9 of the 10 B loads at 4 slots (the tenth read a
swing quantum lower and came 240,553,906 B short) and 173,470,819 B on the one at 8. Of that, 71,303,168 B (`FREE_SWING`) is the check's free
swing, which the whole-layer plan keeps in its reserve and the per-expert plan did not; the rest, 101.9 to 102.2 MB, is
what the load takes past the plan's terms. A's whole-layer loads miss about as much past their ledger (100,085,632 B at
4 host layers), hidden inside a layer's rounding (381 MB). The first plan now reserves the swing and charges 98 MiB and
the re-plan's margin (one expert and the swing), the reserve the re-plan reached. `a_split_load_plans_once` (fake CUDA, a
card that reads as a real one after the load and takes half the charge past the plan's own slack): one per-expert plan
and a split load; red with the charge off (the check came short and the re-plan no longer fit a split).

| Loads (1/50 slice at W4@1) | plan lines | load s |
|---|---|---|
| B default, 5 in a row | `larch_expert_cache_plan,20,9319` once each, all five | 10.06 (cold page cache, after the m1/m5 loads), 4.96, 4.34, 4.15, 4.98 |
| #21 opt-in (A, `=1`, the shipped histogram beside the model), 2 | `…plan,15,9502` then `…plan,19,9330` each | 6.95, 8.02 |
| A default (whole layers), 1 | `larch_host_experts_plan,4` | 4.15 |

Before, a load was two loads (the first upload thrown away) and its second plan depended on the first's reading: 20
layers in #21's runs, 21 in one, 19 tonight. After, the first plan is the final one: 20 split layers on every one of
the 19 4-seat serve loads of m13 tonight (5 above, the smoke load, D5's 4 and M's 9), the 8-seat plan 28 on all of
its 5; parity's loads, at their own context and rows, each planned once too.

## D5: seats, by measurement

The 8-seat plan (`--set max_batch_seqs=8`: 28 split layers, one plan, 8 slots) against the 4-seat default (20 split
layers, 4 slots), X Y Y X slices on `dae3678`:

| Load | 4 seats (X1 / X2) | 8 seats (Y1 / Y2) | tok/s | TTFT p99 | TPOT p99 |
|---|---|---|---|---|---|
| W4@1 | 162.2 (162.4 / 162.0) | 157.3 (157.2 / 157.3) | **−3.0 %** (spreads 0.2 %, 0.0 %) | +4.1 % | +2.6 % |
| W4@4 | 289.9 (289.3 / 290.6) | 276.6 (277.8 / 275.4) | **−4.6 %** (0.4 %, 0.9 %) | +7.6 % | +3.9 % |
| W4@8 (#21, full runs) | 300.5 | 335.8 | +11.7 % | TTFT p50 4,456 → 137 ms | 15.0 → 27.1 ms |
| W4@16 (full runs) | 299.2 (297.2 / 301.1) | 336.5 | +12.5 % | 16,213 / 15,805 → 10,895 ms | 15.2 → 26.8 ms |

Rule: within 3 % at W4@1 and at W4@4, else keep 4 seats. The 8-seat plan loses 3.0 % at W4@1 (157.26 against 162.20,
−3.05 %) and 4.6 % at W4@4: **4 seats stay, nothing changed.** The cost of each choice: 4 seats keep one and four chats
3 to 5 % faster and pay 11 to 13 % of throughput at eight or more chats, with an eight-chat TTFT p50 of 4.5 s
(requests wait for a seat) against 0.14 s; 8 seats are the reverse, and their TPOT p99 at many chats is 27 ms against
15 (more rows a step). A demand switch (a reload between the two plans, as the whole-layer plan has between 1 and 4
seats) would take both, at a reload's cost; the brief added none. The maintainer's decision.

## P

- `cargo fmt --all --check`: clean. `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean.
- `cargo test --locked --workspace --no-fail-fast`: 3,183 passed, 34 ignored, 1 failed: larch-agent's
  `a_manifest_written_mid_run_makes_the_task_tested`, the known failure (npm on the path without node).
- `ctest` (CUDA kernels, `flock`): 20 of 20 passed (no kernel changed).
- `/code-review high` (`6e011de...HEAD`): one finding, from both bug passes: a plain whole-layer load recorded routes
  (D1's second test, fixed in `0bdc9be`). The two CLAUDE.md passes found no violation.
- `/ponytail-review`: three findings (one first-charge function, shared fake-card test setup, a shorter `log`), applied
  in `e7c402e` (−46 lines); the second pass: Lean already. Ship.
- m13 parity `--mode all` on the default load (`e7c402e`, the variable unset, no file beside the model): every mode
  passed, each mode's load planning once (`parity-m13-all.log`):

| Mode | split layers | mean KL | p99 KL | top-1 | bound |
|---|---|---|---|---|---|
| prefill | 14 | 0.00425 | 0.0188 | 0.943 | 0.0354 |
| decode | 11 | 0.00403 | 0.0170 | 0.9445 | 0.0354 |
| chunked (933 mixed steps) | 19 | 0.00417 | 0.0180 | 0.942 | 0.0354 |
| cached | 14 | 0.00490 | 0.0229 | 0.9411 | 0.0354 |

## M (fresh-install protocol)

1. A B B A slices (`--slice 1/10`), A `6e011de`'s code at default, B `e7c402e` at default:

| Load | A (A1 / A2) | B (B1 / B2) | tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| W4@4 | 237.7 (237.7 / 237.7) | 289.8 (289.1 / 290.5) | **+21.9 %** | −12.6 % | −14.9 % | −18.9 % | −18.2 % |
| W4@1 | 135.4 (135.2 / 135.6) | 161.6 (161.8 / 161.5) | **+19.4 %** | −12.0 % | −11.6 % | −16.9 % | −16.7 % |

2. Full runs on B's default (4 seats), against #21's opt-in (`=1`, the histogram beside the model):

| Load | plan | tok/s | #21 | TTFT p50 / p99 ms | TPOT p50 / p99 ms | #21 TPOT p99 |
|---|---|---|---|---|---|---|
| W4@1 | 20, mapped=1, once | 161.6 | 161.8 (−0.1 %) | 119 / 502 | 5.64 / 5.83 | 5.82 |
| W4@4 | 20, once | 300.6 | 298.2 (+0.8 %) | 133 / 572 | 12.49 / 15.04 | 15.03 |
| W4@8 | 20, once | 300.4 | 300.5 (0.0 %) | 4,418 / 6,268 | 12.47 / 14.97 | 15.03 |
| W4@16, run 1 | 20, once | 297.2 | 300.4 (−1.1 %) | 12,778 / 16,213 | 12.49 / **19.62** | 14.94 |
| W4@16, run 2 | 20, once | 301.1 | (+0.2 %) | 12,755 / 15,805 | 12.51 / 15.19 | |

W4@16's first run: the first 4-row capture fell short of the graph quota (`larch_graph_eager,decode=4,reason=quota`)
and 254 decode steps ran eager at 20.4 ms until `larch_graph_retry` captured it 267 steps later; #21's W4@16 hit the
same quota on its 2-row bucket, where it cost less. The second run captured at once and matched #21. Both runs count;
the quota race is not this change's (it is the graph fit's, before and after).

3. B with `LARCH_EXPERT_CACHE=0`, W4@4 slice: 237.4 tok/s against A's 237.7 (−0.1 %, noise), A's plan and pool lines.
4. m1 and m5 (dense): A's and B's fit lines identical (`dense-m1-*-fit.txt`, `dense-m5-*-fit.txt`).

**KEEP:** B's default is within noise of #21's opt-in (−0.1 % at W4@1, +0.8 % at W4@4) and ahead of A outside noise
at both (+19.4 %, +21.9 %); TPOT p99 within 3 % of #21's (+0.2 %, +0.1 %) and 17 to 18 % under A's; parity passed;
items 3 and 4 hold. Kept.

## Void runs

None tonight: every B run showed exactly one `larch_expert_cache_plan` and `larch_split_decode,mapped=1`; every A run
and the `=0` slice `larch_host_experts_plan,4` and no per-expert plan; no `.experts` file was beside a model before or
after any run. One smoke load (`load-smoke-1`) first failed to start (exit 127, the binary named without its path) and
ran again; it is not a measurement.

## Debt

| What stays | Why | Trigger to revisit |
|---|---|---|
| Profiles for other models | One shipped profile, m13's, for its one file; any other file (another quant, another model) keeps whole layers unless a histogram is beside it. | A second hybrid MoE target, or m13 in another quant. |
| Recording a profile on first use | A static split's in-place rows go uncounted, so a default load records nothing; a first-use profile needs those rows counted (a count inside `mmvq_id_mixed`, or the route log for split layers). | Before a second profile is recorded by hand. |
| Adaptive swaps | Dynamic residency (`=2`) is opt-in and has no in-place path; the static split never moves an expert after the load. | A workload whose hot set differs from the profile's (hit rate low in `larch_expert_cache_hits`). |
| The cold gate/up read | #21's G1: cold gate and up read at about 3 GB/s while down reads at about 24 GB/s; a cold pair costs about 360 µs. | Unchanged; the split's cost model assumes it. |
| HIP | HIP's wrap answers UNSUPPORTED, so HIP keeps whole layers by default (and Metal, the CPU). | A HIP wrap of page-locked memory and measured `SPLIT_LAYER_US`/`COLD_PAIR_US` there. |
| 98 MiB allowance | Measured on one card (5070 Ti, WSL2), charged on CUDA only; A's loads miss as much inside a layer's rounding. | A card whose first plan re-plans (`larch_expert_cache_plan` twice). |
| Seats (D5) | 4 seats: −11 to −13 % at 8+ chats against 8 seats; 8 seats: −3 to −5 % at 1 and 4. | The maintainer's decision; or a demand switch between the two plans. |
| Expert adapters on m13 | A split load refuses expert adapters (`larch_expert_cache_no_learning`); m13 now splits by default, so they need `LARCH_EXPERT_CACHE=0` (the message says so). Suffix training is Qwen3 MoE only and unaffected. | Adapters on the hybrid MoE target. |
| Eager mixed steps, uncounted histogram rows | #21's rows (prompt chunks beside decode rows run eager; the static split's rows go uncounted, now moot for the file). | As #21. |

## Next step: the pull request

#21 is open and `origin/dev` does not hold its squash commit (fetched 21:00 CT), so no pull request is opened (its diff
would carry all of #21). Once #21 is merged:

```
git fetch origin dev
git switch perf/35b-spill-default
git merge origin/dev          # a merge, never a rebase
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked -p larch-engine -p larch-model
git push origin perf/35b-spill-default
gh pr create --draft --base dev --head perf/35b-spill-default \
  --title "Expert spill by default on CUDA: the shipped m13 profile, no skewed histogram, one plan" \
  --body-file <this report's summary, P, the debt table, net:, and "Assisted by Claude Code">
```

## Method

- **A** is #21's head at default settings (whole layers, `larch_host_experts_plan,4`): the `e3ffd4b` release binary,
  sha256 `582252d62c48f68a2991d4496217032dc8d0dcf3c3b2b5328f99d478a550d538`; `git diff e3ffd4b 6e011de` touches no
  crate, kernel or Cargo file (the four commits after it are bench results), so it is #21's head's code.
- **B** is this branch's default: `e7c402e`, sha256 `eca197d769471f6f30f092ecf57a4dcb256beb37240545c0c94f54e250f5e3ec`. D5 ran on `dae3678` (sha256 `8030bf3f…`), whose m13
  split plan is `e7c402e`'s (same plan lines; the later commits gate whole-layer recording and move test code).
- **Fresh-install protocol** on every larch run: no `LARCH_EXPERT_CACHE` in the environment (the `=0` slice sets it),
  every `.experts` beside the models moved aside before the run and counted after it (0 after every run). A B run
  counts only if its fit lines show exactly one `larch_expert_cache_plan` and `larch_split_decode,mapped=1`; A and the
  `=0` slice need `larch_host_experts_plan,4` and no per-expert plan.
- Each GPU job in one `flock` hold; before each: no build process, `tmux ls` (only this session), used VRAM under
  2.5 GB (400 MiB), the core under 85 C. Builds under `nice -n 19`; no build, test or review during a timed run.
- Serve runs: `larch bench -m m13.gguf --mode serve --concurrency <c> --set max_context=2560 --machine wsl5070
  [--slice 1/10] [--set max_batch_seqs=8]`, `LARCH_FIT_DIAGNOSTICS=1 LARCH_PROFILE_STEPS=1 LARCH_GRAPH_DIAGNOSTICS=1`.
  Slices are the A B B A comparison (noise: under 2 % or under either side's spread). Load time: process start to the
  first `larch_context_fit` line, on a 1/50 slice at W4@1.
- Parity: `larch parity -m m13.gguf --ref <cuBLAS oracle> --bound 0.035366004871120384 --margin 3.3277282305061817
  --mode all`, `LARCH_EXPERT_CACHE` unset, no file beside the model.
- llama.cpp: `llama-serve-bench run --workload chat.jsonl --concurrency <c> -m m13.gguf -ngl 99 -fa on -ctk f16 -ctv f16
  -t 8 -b 2048 -ub 1024 -c <c x 2560> --n-cpu-moe <n>`, as #21.
