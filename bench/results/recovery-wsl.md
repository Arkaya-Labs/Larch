# Recovery runs: WSL2 (rows R4, R5, R10)

Rig: WSL2 on Windows, RTX 5070 Ti (16 GB, display GPU). Binary: `larch` CUDA release with
`--features larch-engine/fault-inject`, built from PR #50's `2f18002` with dev `106f2a3` (F4E2c #69)
merged. Model m1, plus m8 for the runtime OOM. Scripts: `recovery-plan.md`, as committed with this file.
2026-10-08, 23:26 to 23:33 UTC (the run-2 OOM rows: 2026-10-09 00:47 UTC, on `727ab08`), on an idle GPU (no other larch process; 1221-1237 MiB used before each run).
The startup-OOM, hang and parity runs held the shared GPU lock. The first six ran before the lock file
existed, with nothing else on the GPU. Raw logs stay outside git.

## Results

| Run | Result | What it showed |
|---|---|---|
| `kill20` | PASS | 20 death and 20 ready lines, restart 1..20; every `exit` null; `recovery_ms − backoff_ms` 1078-1788 ms; death-line `ready_ms` 1061-1784 ms (the cold start 1537); no zombie, no leftover; VRAM 1235 → 1237 MiB; the child was listed in compute apps and gone after |
| `parent` | PASS | child gone 39 ms after `kill -9` of the supervisor; VRAM 1235 → 1229 MiB; gone from compute apps |
| `resume` | PASS | kill -9 mid-reply: `recovery_ms` 1035 (backoff 0); killed run's `ready_ms` 1102; `resumed delivered history` once; turn 1 in the transcript; VRAM +6 MiB |
| `fault panic` (`panic@step:8`) | FAIL (1 item) | exit 3; `recovery_ms` 1053; faulted run's `ready_ms` 1264; resumed once; VRAM +0. **FAIL: no fault line, so `exited_ns − detected_ns` cannot be taken (#74)** |
| `fault device_lost` (`device_lost@step:8`, synthetic) | FAIL (1 item) | exit 3; `recovery_ms` 1120; `ready_ms` 1121; resumed once; VRAM +0; `stderr_tail` `device lost; the process must restart`. **FAIL: no fault line (#74)** |
| `fault hang` (`hang@step:8`, host-thread park) | FAIL (1 item) | the watchdog ended it: exit 3; `recovery_ms` 1061; `ready_ms` 1117; resumed once; VRAM +2 MiB; callers see `device lost` (#67). **FAIL: no fault line (#74)** |
| `fault oom`, `OOM_ALLOC=1..5` (startup), run 1 | INCOMPLETE (1 item each) | all five: exit 1, `DoesNotFit` in `stderr_tail`, no fault line, `ready_ms` null, restart to Ready in 1138-1143 ms, resume n/a, VRAM back. INCOMPLETE only because the script never probed the restarted child (review 5464504355, fixed in `727ab08`) |
| `fault oom`, `OOM_ALLOC=1..5`, run 2 (`727ab08`, `--features fault-inject`) | PASS ×5 | the same, plus the restarted child is listed in compute apps and gone after; restart to Ready in 874-1002 ms |
| parity m8, `oom@alloc:6`, `:7`, `:8` (runtime), run 1 | FAIL (1 item each) | each exit 3 (parity exits 3 only under `OnFault::Exit`; the same command without `LARCH_FAULT` exits 0, every mode passed). **FAIL: no fault line (#74)**. Run 1 recorded no exit time |
| `parity-oom 6`, `7`, `8`, run 2 (`727ab08`) | FAIL (1 item each) | each exit 3, with `exited_ns` recorded (taken after the reap, an upper bound). **FAIL: no fault line (#74)**, so the gap cannot be taken |

Native Windows: `recovery-windows.md`.

## Findings

- **#74, the fault line never prints.** `main.rs` hands chat and parity `io::stderr().lock()` for the whole
  command. The engine's `larch-fault-flush` thread blocks on that lock, its 20 ms deadline passes, and
  `exit(3)` runs without the `larch fault=… detected_ns=…` line. So "exit 3 ≤ 100 ms after detection"
  is **NOT PROVEN** on any fault. The check stays as written; it needs the fix, then a rerun.
- Start → Ready < 2 s holds on every run: the largest is restart 4's 1788 ms (`recovery_ms − backoff_ms`),
  and the largest death-line `ready_ms` is 1784 ms.
- `device_lost` and `hang` are synthetic (F4E2c's host faults). A real sticky CUDA error, a GPU hang and a
  TDR stay BLOCKED on #66.
- Journal flush: NOT RUN. The third contributor's hook isn't on `dev`.
