# m13: hot experts on the card, the cold tail read in place (expert spill), 27 Nov 2026

Test machine: RTX 5070 Ti 16 GB, Ryzen 7 9800X3D, Windows 11 + WSL2 (8 logical CPUs in WSL). Model m13
(`bench/models.toml`). Raw results: [`35b-expert-spill-2026-11-27/`](35b-expert-spill-2026-11-27/).

## Plan

1. Setup: A is `ef28629` rebuilt (the night's `b3a53bf3…` binary is gone); cherry-pick the CUDA wrap of page-locked
   host memory (`2b90422`) and, in a commit of its own, drop its whole-layer mapped decode path.
2. L: llama.cpp load probes, then A, llama, llama, A at W4@4 and W4@8 and the W4@16 runs; the release table on A.
3. K1 (written during L, run after it): CUDA `mmvq_id_mixed` takes cold experts at the hot ones' dtype; a ctest over a
   wrapped page-locked range against `mmvq_id` on one tensor, bit for bit, at 1, 4 and 40 rows, red before green.
4. K3 (likewise): `expert_residency` caps the split layers (Lane E's rule) and picks the cap by the decode cost from
   `SPLIT_LAYER_US` and `COLD_PAIR_US`; tests on `bench/routes/m13.experts`; prints G2's table.
5. G: one GPU hold after L: G1 (a cold pair's cost in a captured graph), G2 (K3's table), G3 (`fused_bench`); GO if
   the cheapest cap's 4-row step is at most 12.0 ms with its 1-row step at most 6.2 ms, else stop and report.
6. K2 (GO only), test first: split loads page-locked; decode, check and a chunk's leading decode rows of split layers
   through `Experts::Mixed` (cold experts from `Buffer::view_host`), prompt rows on `split_ffn`; tests in `tiny.rs`.
7. SMOKE: one B run at W4@4 `--slice 1/10`: every decode bucket in one graph piece, replayed 4-row step at most 12.0 ms.
8. P: code review, over-engineering review, fmt, clippy, tests, ctest, then m13 parity (prefill, decode, decode -p 4).
9. M: A B B A at W4@4 and W4@1, full W4@4 and W4@8 on B, W4@8 with 8 seats, W4@16; histogram protocol on every run.
10. R: this report and, if kept, a draft PR to dev; parity `--mode all` on B after the hard end.

## L: the release table on A

A is `ef28629`, the kept stack of 27 Nov (whole host layers, the prefill pool, 4 state slots), rebuilt today. Its W4@4,
W4@8 and W4@16 runs ran today (12:34 to 13:25 CT), interleaved with llama.cpp: A, llama.cpp, llama.cpp, A at W4@4 and
W4@8, then A once and llama.cpp at `--n-cpu-moe` 8 and 7 at W4@16. W4@1 and the other llama.cpp cells are last night's
(27 Nov, UTC hour given), A's W4@1 there from the same commit's `b3a53bf3…` build. Every A run showed
`larch_host_experts_plan,4`. Memory: llama.cpp's CUDA0 buffers against the free memory it reported (15,009 MiB), from
its load lines; "fits" is the fewest `--n-cpu-moe` at or under it. Probes (`wsl5070-m13-llama-probes.txt`): `-np 4` fits at
ncm 5 (14,890 MiB); `-np 8` is 3 MiB past at ncm 6 (15,012) and fits at ncm 7 (14,682).

| Load | larch A (`ef28629`) tok/s | llama.cpp | tok/s, memory, date; larch's margin |
|---|---|---|---|
| W4@1 | 136.0 (136.0 / 136.1), 27 Nov 08Z | ncm 3 | 130.4 (129.9 / 130.9); +271 MiB past free, 27 Nov 08Z; larch +4.3 %, larch ahead |
|  |  | ncm 4 | 125.3; -93 MiB past free, 27 Nov 05Z; larch +8.6 %, larch ahead |
|  |  | fits: ncm 4 | 125.3; -93 MiB past free, 27 Nov 05Z; larch +8.6 %, larch ahead |
| W4@4 | 245.8 (245.8 / 245.7), today | ncm 3 | 260.9 (262.6 / 259.2); +609 MiB past free, 27 Nov 05Z; larch -5.8 %, larch behind |
|  |  | ncm 4 | 232.1; +245 MiB past free, 27 Nov 06Z; larch +5.9 %, larch ahead |
|  |  | fits: ncm 5 | 224.6 (223.8 / 225.5); -119 MiB past free, today; larch +9.4 %, larch ahead |
| W4@8 | 245.3 (245.5 / 245.2), today | ncm 3 | 307.2; +1061 MiB past free, 27 Nov 06Z; larch -20.1 %, larch behind |
|  |  | ncm 4 | 288.8; +697 MiB past free, 27 Nov 06Z; larch -15.0 %, larch behind |
|  |  | fits: ncm 7 | 239.2 (237.3 / 241.0); -327 MiB past free, today; larch +2.6 %, larch ahead |
| W4@16 | 246.1, today | ncm 3 | 68.5; +1963 MiB past free, 27 Nov 06Z; larch +259.4 %, larch ahead |
|  |  | ncm 4 | 71.6; +1599 MiB past free, 27 Nov 06Z; larch +243.8 %, larch ahead |
|  |  | fits: ncm 9 | 235.1; -119 MiB past free, 27 Nov 06Z; larch +4.7 %, larch ahead |
|  |  | ncm 8 | 243.5; +211 MiB past free, today; larch +1.1 %, noise |
|  |  | ncm 7 | 266.3; +575 MiB past free, today; larch -7.6 %, larch behind |

- **W4@4:** larch is behind ncm 3 (−5.8 %, which loads 609 MiB past its free memory) and ahead of ncm 4 and of the
  fitting ncm 5 (+9.4 %, spreads 0.0 % and 0.8 %).
- **W4@8:** behind ncm 3 and 4 (−20.1 %, −15.0 %); ahead of the fitting ncm 7 by 2.6 %, outside both spreads (0.1 %,
  1.5 %): a win by the rule, a thin one.
- **W4@16: llama.cpp's ncm 7 beats larch**, 266.3 against 246.1 (−7.6 %, one run a side), loading 575 MiB past its free
  memory. ncm 8 (211 MiB past) is level (+1.1 %, noise). Larch's +4.7 % over the fitting ncm 9 stands only against
  settings that fit; at 16 chats llama.cpp's faster settings run past its reported memory, as ncm 3 does at W4@4.
- **W4@1:** ahead of all three (last night's cells).

## Method

- **Machine and order.** One session. Every GPU job (runs, ctest, benches, GPU tests, parity) is its own process in one
  hold of the GPU lock, with used VRAM back under 2.5 GB and steady for 5 s before it. Builds run under `nice -n 19`,
  and no build, test or review ran during a timed run.
- **llama.cpp.** b12723 through `llama-serve-bench` (the 27 Nov build), `-ngl 99 -fa on -ctk f16 -ctv f16 -t 8 -b 2048
  -ub 1024 -c <chats × 2560> --n-cpu-moe <n> -lv 4`. Its free-memory and CUDA0 buffer lines sit beside each result
  (`-load.txt`); a load probe is a short `fixed` run whose lines go to `wsl5070-m13-llama-probes.txt`.
- **Larch.** `larch bench --mode serve --set max_context=2560`, `LARCH_FIT_DIAGNOSTICS=1`; fit lines beside each result
  (`-fit.txt`). W4: 200 prompts, exact output lengths; `--slice K/10` is 20 of them.
- **Histogram protocol.** A and llama.cpp: any `<gguf>.experts` beside the model moved aside, `LARCH_EXPERT_CACHE`
  unset. B: `bench/routes/m13.experts` copied beside the model and `LARCH_EXPERT_CACHE=1` before every load (a split
  load rewrites the file; it is moved aside after). A run is void unless its fit lines show the plan its side needs:
  A `larch_host_experts_plan,4`; B `larch_expert_cache_plan,<N>` and `larch_split_decode,mapped=1`.
- **Statistics.** Each cell is the mean of its runs, spread = (max − min) / mean. A margin is noise when under 2 % or
  under either side's spread; a one-run cell has only the 2 % rule (the 26 Nov rule).

## Builds

| Build | sha256 | Commit | What it is |
|---|---|---|---|
| `larch 0.0.0 (ef28629)` | `c1ad854a…` | `ef28629` (#20's branch head, before its review fix) | A: rebuilt today, as the night's `b3a53bf3…` binary was gone; the same commit, not the same bytes |
