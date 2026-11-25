# Recovery runs: native Windows (rows R4, R5, R10)

Every row here was **run unattended through WSL interop**: `powershell.exe` was called from WSL into the
Windows clone, checked out detached at `b80095f`. Its code is the same as `dev`'s `5013a64`;
only the plan differs.
- Rig: Windows 11, RTX 5070 Ti (16 GB, display GPU).
- Binary: `larch.exe`, CUDA release with `--features larch-engine/fault-inject`, cuBLAS 13 dynamic. Built in the
  VS 2022 Build Tools x64 dev shell with CUDA 13.4.2 and the Ninja generator, through `cmd /c`.
- Models: m1, plus m8 for the runtime OOM. The GGUFs were on a local Windows disk.
- When: 2026-10-08, 23:45 to 23:56 UTC (run 2 of the OOM rows: 2026-10-09 00:49 UTC, rebuilt at `727ab08` with `--features fault-inject`), holding the shared GPU lock, on an idle GPU (1221-1242 MiB before
  each run, no other larch process).
- Scripts: `recovery.ps1` from `2890249`. The `resume` and second `kill20` runs used `f6845b1`, which reads
  the transcript after `/quit`. During the second `kill20`, WSL was running a CPU test build.
- Raw logs stay outside git.

## Results

| Run (all: run unattended through WSL interop) | Result | What it showed |
|---|---|---|
| `kill20`, run 1 | **FAIL** (1 item) | 20 death and 20 ready lines, restart 1..20; every `exit` −1; `recovery_ms − backoff_ms` 1017-1175 ms; no leftover; VRAM 1221 → 1237 MiB; gone from compute apps. **FAIL: the cold start's `ready_ms` was 2271 ms** (bound 2000), the first load of m1 after it was copied to the Windows disk. Restarts 2-20 were 1017-1175 ms |
| `kill20`, run 2 | PASS | the same checks: cold start 1188 ms, death-line `ready_ms` 1012-1188, `recovery_ms − backoff_ms` 1011-1118 ms, VRAM +0 MiB |
| `parent` | PASS | child gone 83 ms after `Stop-Process -Force` of the supervisor; VRAM back; gone from compute apps |
| `resume`, run 1 | INCOMPLETE (script defect) | the restart, the ready line and `resumed delivered history` once all passed. Then `Select-String` could not read the transcript, because the live chat holds a byte-range lock on it, and the script stopped before its verdict. Fixed in `f6845b1`: the check runs after `/quit` |
| `resume`, run 2 | PASS | `recovery_ms` 1022 (backoff 0); killed run's `ready_ms` 1137; resumed once; turn 1 in the transcript; VRAM +0 MiB |
| `fault panic` (`panic@step:8`) | FAIL (1 item) | exit 3; `recovery_ms` 1118; `ready_ms` 1133; resumed once; VRAM back; `stderr_tail` `internal error 0`. **FAIL: no fault line (#74)** |
| `fault device_lost` (synthetic) | FAIL (1 item) | exit 3; `recovery_ms` 1107; `ready_ms` 1171; resumed once; `stderr_tail` `device lost; the process must restart`. **FAIL: no fault line (#74)** |
| `fault hang` (host-thread park) | FAIL (1 item) | the watchdog ended it: exit 3; `recovery_ms` 1125; `ready_ms` 1176; resumed once; callers see `device lost` (#67); VRAM +16 MiB. **FAIL: no fault line (#74)** |
| `fault oom`, `OOM_ALLOC=1..5` (startup), run 1 | INCOMPLETE (1 item each) | all five: exit 1, `DoesNotFit`, no fault line, `ready_ms` null, restart to Ready in 1131-1148 ms, resume n/a, VRAM back. INCOMPLETE only because the script never probed the restarted child (review 5464504355, fixed in `727ab08`) |
| `fault oom`, `OOM_ALLOC=1..5`, run 2 (`727ab08`) | PASS ×5 | the same, plus the restarted child is listed in compute apps and gone after; restart to Ready in 1071-1097 ms |
| parity m8, `oom@alloc:6`, `:7`, `:8` (runtime), run 1 | FAIL (1 item each) | each exit 3, with nothing on stderr. The same command without `LARCH_FAULT` exits 0 (`parity: every mode passed`). **FAIL: no fault line (#74)**. Run 1 recorded no exit time |
| `parity-oom 6`, `7`, `8`, run 2 (`727ab08`) | FAIL (1 item each) | each exit 3, with `exited_ns` from the OS (`Process.ExitTime`). **FAIL: no fault line (#74)**, so the gap cannot be taken. The script now takes `exited_ns` from the precise clock after the reap, as `recovery.sh` does (review 5464934913). No gap was taken, so the verdict stands |

## Findings

- **#74 holds on Windows too.** No fault line on any fault, so "exit 3 ≤ 100 ms after detection" is **NOT PROVEN**.
- **Start → Ready < 2 s:** every restart and every warm start passed (≤ 1188 ms). **One cold start failed at 2271 ms**:
  the first process to read the freshly copied 4.9 GB m1 from disk. The rerun's cold start took 1188 ms. A first
  start from a cold disk cache is a real case, so this stays a FAIL until a cold-cache run passes.
- A real TDR and a real device loss stay BLOCKED on #66.
