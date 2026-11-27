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
