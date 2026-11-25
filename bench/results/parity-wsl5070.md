# Parity on wsl5070 (5070 Ti, CUDA device 0)

`larch parity` against each model's llama.cpp b11458 cuBLAS ub32 reference, with the bound and margin of
`bench/oracle-bounds.json`. Copied from the A3O2 hand-offs: m8 from #60 (at `b7a22fd`), m1, m2, m5 and m7 from #70
(the #63 matrix). The Metal pair is NOT RUN (F2's M5).

**What these rows do not cover.** The m1, m2, m5 and m7 tables ran on #63's head with `graph_replay` false, so their
`decode` rows are the direct decode path, not the captured-graph replay that dev now ships (F4E2b); m8's table does
cover replay. Graph-replay parity and the `--graphs off` control for m1, m2, m5 and m7 belong to A1R1 (pass 2). A1B3's own
gate on its scratch bench build (`55063bd`) failed to run (out of memory under a shared GPU hold) and was dropped from
this PR when parity moved to A1R1, so no row here vouches for that build.

## m8 (100 prompts on CUDA; 2 on the CPU reference)

| Backend | Mode | Decisive top-1 | Raw top-1 | Mean KL | Replayed steps | Mixed steps | Exit |
|---|---|---|---|---|---|---|---|
| CUDA dev 0, 100 prompts | prefill | 1.0000 (27,636/57,803) | 0.9903 | 8.91e-5 | 0 | 0 | 0 (all modes 650 s) |
| CUDA | decode, graphs on | 1.0000 | 0.9902 | 9.26e-5 | **57,703** | 0 | 0 |
| CUDA | chunked | 1.0000 (28,130/58,856) | 0.9904 | 8.65e-5 | 101 | **951** | 0 |
| CUDA | cached | 1.0000 (12,783/28,923) | 0.9888 | 8.81e-5 | 0 | 0 | 0 |
| CUDA | decode `--parallel 8`, graphs on | 1.0000 | 0.9902 | 9.26e-5 | 7,644 | 92 | 0 (123 s) |
| CUDA | decode `--graphs off` | 1.0000 | 0.9902 | 9.26e-5 | 0 (`graph_replay: true`, `decode_graphs: false`) | 0 | 0 (824 s) |
| CPU reference, 2 prompts | prefill / decode / chunked / cached | 1.0000 each | 0.985 / 0.982 / 0.985 / 0.975 | 9.6e-5 / 1.06e-4 / 9.7e-5 / 9.1e-5 | 0 | 13 (chunked) | 0 (853 s) |

## m1 (100 prompts)

| Mode | Decisive | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Result |
|---|---|---|---|---|---|---|---|
| prefill | 12791/58403 | 1.0000 | 0.9680 | 3.239e-04 | 1.570e-03 | 3.841e-03 | pass |
| decode | 12791/58403 | 1.0000 | 0.9680 | 3.225e-04 | 1.597e-03 | 3.893e-03 | pass |
| chunked | 12791/58403 | 1.0000 | 0.9679 | 3.241e-04 | 1.555e-03 | 3.901e-03 | pass |
| cached | 4986/29223 | 1.0000 | 0.9626 | 3.596e-04 | 1.441e-03 | 3.841e-03 | pass |
| decode --parallel 8 | 12791/58403 | 1.0000 | 0.9680 | 3.225e-04 | 1.597e-03 | 3.893e-03 | pass |
| decode --graphs off | | | | | | | NOT RUN: same path as decode until F4E2 (`graph_replay` false); #63 fork 7 |

## m2 (100 prompts)

| Mode | Decisive | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Result |
|---|---|---|---|---|---|---|---|
| prefill | 17492/55703 | 1.0000 | 0.9717 | 5.133e-04 | 1.987e-03 | 1.827e-02 | pass |
| decode | 17492/55703 | 1.0000 | 0.9717 | 5.132e-04 | 1.986e-03 | 1.755e-02 | pass |
| chunked | 17492/55703 | 1.0000 | 0.9717 | 5.134e-04 | 1.986e-03 | 1.888e-02 | pass |
| cached | 8499/27880 | 1.0000 | 0.9699 | 5.397e-04 | 2.008e-03 | 1.868e-02 | pass |
| decode --parallel 8 | 17492/55703 | 1.0000 | 0.9717 | 5.132e-04 | 1.986e-03 | 1.755e-02 | pass |
| decode --graphs off | | | | | | | NOT RUN: same path as decode until F4E2 (`graph_replay` false); #63 fork 7 |

## m5 (100 prompts)

| Mode | Decisive | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Result |
|---|---|---|---|---|---|---|---|
| prefill | 9735/57803 | 1.0000 | 0.9644 | 5.484e-04 | 2.599e-03 | 1.593e-02 | pass |
| decode | 9735/57803 | 1.0000 | 0.9645 | 5.497e-04 | 2.677e-03 | 2.070e-02 | pass |
| chunked | 9735/57803 | 1.0000 | 0.9645 | 5.483e-04 | 2.521e-03 | 1.801e-02 | pass |
| cached | 4529/28923 | 1.0000 | 0.9646 | 5.601e-04 | 2.496e-03 | 1.321e-02 | pass |
| decode --parallel 8 | 9735/57803 | 1.0000 | 0.9645 | 5.497e-04 | 2.677e-03 | 2.070e-02 | pass |
| decode --graphs off | | | | | | | NOT RUN: same path as decode until F4E2 (`graph_replay` false); #63 fork 7 |

## m7 (100 prompts)

| Mode | Decisive | Decisive top-1 | Raw top-1 | Mean KL | p99 KL | Max KL | Result |
|---|---|---|---|---|---|---|---|
| prefill | 26521/57836 | 1.0000 | 0.9761 | 2.107e-04 | 7.828e-04 | 7.639e-03 | pass |
| decode | 26521/57836 | 1.0000 | 0.9758 | 2.127e-04 | 8.205e-04 | 9.264e-03 | pass |
| chunked | 26521/57836 | 1.0000 | 0.9761 | 2.104e-04 | 7.839e-04 | 8.048e-03 | pass |
| cached | 12425/28944 | 1.0000 | 0.9745 | 2.198e-04 | 7.953e-04 | 2.734e-03 | pass |
| decode --parallel 8 | 26521/57836 | 1.0000 | 0.9758 | 2.127e-04 | 8.205e-04 | 9.264e-03 | pass |
| decode --graphs off | | | | | | | NOT RUN: same path as decode until F4E2 (`graph_replay` false); #63 fork 7 |
