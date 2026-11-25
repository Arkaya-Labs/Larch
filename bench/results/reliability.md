# Reliability matrix: A1 rows

Architecture §15. Raw logs stay outside git.

## Fuzz (row R9 and §15's fuzz row)

Harness: `fuzz/` at `5d7b574`, CPU backend, libFuzzer through cargo-fuzz on nightly.
The seed corpus is from `cargo +nightly run --bin seed`: the tiny model GGUFs, the tiny tokenizer and the test prompts.
Each target runs for 2 h; the three run in parallel on one 8-core WSL2 host. 02:36 to 04:37 UTC on 2026-10-08.

```
cd fuzz && LARCH_BACKEND=CPU cargo +nightly fuzz run <target> -- -max_total_time=7200 -rss_limit_mb=2048 -print_final_stats=1
```

| Target | What it feeds | Wall | Execs (avg/s) | Edges · corpus | Crashes / OOM / timeouts | Status |
|---|---|---|---|---|---|---|
| `gguf` (row R9) | `Gguf::parse`, then `ModelConfig::from_gguf` and `Tokenizer::from_gguf` | 7201 s | 58,682,890 (8,149) | 1615 · 861 inputs, 15.8 MB | 0 / 0 / 0 (peak RSS 577 MB) | PASS |
| `tokenizer` | `encode`, every id within the vocab, `decode_piece`, `Detokenizer` on arbitrary text | 7201 s | 27,591,047 (3,831) | 2447 · 1373 inputs, 292 KB | 0 / 0 / 0 (peak RSS 584 MB) | PASS |
| `knob_request` | `larch_records::validate` on arbitrary requests; an accepted value must be finite and within its bounds | 7201 s | 8,720,373,400 (1,210,994) | 83 · 22 inputs (flat from the start) | 0 / 0 / 0 (peak RSS 506 MB) | PASS |

`knob_request` stayed at 83 edges because `validate` is small and was covered within seconds. `cargo +nightly fuzz coverage knob_request` over the 22-input corpus reaches every arm of `validate`, `Ok` included. The only line it misses is `KnobType::Unknown`, which no registry knob has.

## Recovery (rows R4, R5, R10)

Scripts and checklist: `recovery-plan.md`; details: `recovery-wsl.md`, `recovery-windows.md`. F4E2c (#69) is on `dev`; its `device_lost` and `hang` are synthetic host faults until #66 adds a debug device op.

| Row | Proof | WSL | Windows native |
|---|---|---|---|
| R4 | injected `device_lost` and `panic`, plus a plain kill, under `larch supervise`: exit 3 ≤ 100 ms after `detected_ns`, start→Ready < 2 s | kill: PASS; panic, device_lost: **FAIL (no fault line, #74)**: exit 3 and Ready < 2 s, but the 100 ms gap cannot be taken | kill (`resume`): PASS on run 2; panic, device_lost: **FAIL (no fault line, #74)**: exit 3 and Ready < 2 s, but the 100 ms gap cannot be taken. Run unattended through WSL interop |
| R5 | 20 kills: no surviving processes, VRAM released | PASS (and `parent`: PASS, 39 ms) | run 1 **FAIL** (cold start 2271 ms, first read of the copied GGUF), run 2 PASS; `parent`: PASS, 83 ms. Run unattended through WSL interop |
| R10 | `LARCH_FAULT=hang@step:N`: the watchdog exits 3 (`fault=Timeout`, callers see `DeviceLost`), restart to Ready < 2 s | **FAIL (no fault line, #74)**: exit 3 and Ready < 2 s, but the 100 ms gap cannot be taken | the same as WSL. Run unattended through WSL interop |
| R10 | a real TDR (Windows) or GPU hang (Linux) | BLOCKED (#66) | BLOCKED (#66) |
