# Recovery plan: rows R4, R5, R10 (architecture 9.11, 9.12, 10.3, 14, 15)

## Status

- F4E2c (#69) and F4U1 (#27) are on `dev`. What the scripts use from them:
  - `LARCH_FAULT=panic@step:N | device_lost@step:N | hang@step:N | oom@alloc:N`, built in with `--features fault-inject` (forwarded to `larch-engine`, #65; any debug build has it too).
  - The fault line `larch fault=<code> detected_ns=<unix ns>` on stderr; `exited_ns − detected_ns` is the detection → exit gap.
  - `oom@alloc:N` hits chat's startup allocations (K/V and the serving buffers): a `DoesNotFit` load error and exit 1, not a fault. Exit 1 takes §9.11's backoff (0 ms on restart 1), and with no load there is no transcript, so the resume check is skipped.
  - `device_lost` and `hang` are synthetic host faults until #66 adds a debug device op: no sticky CUDA error, no GPU hang, no TDR.
- Base: `54391fb` (head `1aa2298`, a clippy-only change on top). The supervisor log these scripts
  parse is two JSON lines per restart n, paired by `restart`: a death line with `exit` and the dead
  run's spawn → Ready `ready_ms` (null if it never got Ready), written before the backoff, and
  `{"recovery_ms": N, "restart": n}` when restart n's child first prints Ready (exit → Ready, backoff
  included). The cold start gets no ready line; its Ready is the child's own `larch: ready`.
  start→Ready < 2 s: `ready_ms` on death lines, and `recovery_ms − backoff_ms` on each restart; raw
  `recovery_ms` includes the backoff §9.11 mandates for non-fault deaths. Head for
  the attended runs: `HEAD_SHA`, the PR head with `dev` merged after F4E2c, filled in below on the day.
- Phase 1 dry run (dry run, fake child, not evidence), WSL, `LARCH=target/a1-dry/debug/larch`
  (CPU debug build of `1aa2298`), GGUF `/dev/null`, each run inside `unshare -rpf --mount-proc` because
  another lane's `larch parity` jobs were running (the scripts' `pgrep -x larch` guard and leftover
  checks, unchanged, then see only the run's own namespace). Each prints `DRY RUN (CHILD_CMD): not
  evidence`, so none can read PASS. `fault` and `resume` could not be dry-run then (no `LARCH_FAULT` or `chat` on
  `dev`). The fakes use no GPU. VRAM readings taken while another lane's GPU parity jobs were running:
  not a noise-floor measurement.
  - `kill20`, fake `sh -c 'echo "fault=${LARCH_FAULT-unset}" >&2; echo "larch: ready" >&2; while read -r l;
    do [ "$l" = /quit ] && exit 0; done'`, run with `LARCH_FAULT=hang@step:1` in the environment: all 21
    runs saw it unset. No zombie under the live supervisor after 20 kills, `/quit` rc 0, 20 death and
    20 ready lines paired 1..20, every `exit` null, death-line `ready_ms` 1-8, no leftovers. Raw
    `recovery_ms` 9 257 507 1002, then 2005-2013 for restarts 5..20, against `backoff_ms` 0 250 500 1000
    then 2000: `recovery_ms − backoff_ms` 2-13, PASS. VRAM MiB: baseline pair 14826 then 14811, at Ready
    14811, after 14862 (+51 from the second baseline). RESULT INCOMPLETE (dry run, compute apps NOT
    OBSERVABLE).
  - `parent`, fake `"$LARCH" supervise -- sh -c 'echo "larch: ready" >&2; exec sleep 600'`, `TMPDIR` a
    symlink: child gone 15 ms after `kill -9` of the supervisor. VRAM MiB: baseline pair 14846 then
    14840, at Ready 14840, after 14840 (0). One FAIL, from the fake: its `sleep 600` leaked (a SIGKILLed
    nested supervisor cannot kill its child's process group; chat starts no processes), found under
    the canonical path by the run's inherited stdin/stdout (the namespace's end reaped it). The log
    holds the child's one `larch: ready` and no ready line (a cold start only).
  - The zombie check itself, offline: `sh -c 'sleep 0 & exec sleep 5'` as the supervisor gives
    `FAIL no zombie under the supervisor: Z 286744`; a plain `sleep 5` gives PASS.

## Preconditions (every run)

- [ ] A maintainer is watching only for a real GPU hang or TDR (#66); F4E2c's synthetic faults run unattended.
- [ ] GPU apps closed (browsers, games, other inference servers); `pgrep -x larch` / `Get-Process larch` empty.
- [ ] The model is `m1` from `bench/models.toml`, as `<models>/m1.gguf` in the driver's model dir
  (`scripts/baseline.py --models <dir> models` fetched and verified it).
- [ ] A release build of `HEAD_SHA` with `--features fault-inject` (`larch` forwards F4E2c's `larch-engine/fault-inject`, #65).
- [ ] `nvidia-smi` baseline taken (the scripts print `VRAM MiB before` themselves; also save one full
  `nvidia-smi` output next to the logs).
- [ ] Native Windows and WSL are recorded separately: same commands, separate result files.

Run order on each OS: `kill20`, `parent`, `resume`, `fault panic`, `fault device_lost`, `fault oom`,
and `fault hang` last (a host-thread park until #66: no TDR, so no maintainer needed; a real GPU hang after #66 is attended).
Each script ends with `RESULT: PASS` (exit 0), `RESULT: FAIL` (exit 1) or `RESULT: INCOMPLETE` (exit 4:
some item BLOCKED or NOT OBSERVABLE; never a pass).

## Linux and WSL: `recovery.sh`

```
HEAD_SHA=<sha>; git fetch origin && git checkout --detach "$HEAD_SHA"   # from the repo root
LARCH_BACKEND=CUDA CUDAHOSTCXX=/usr/bin/g++-14 CARGO_TARGET_DIR=target/cuda cargo build --release -p larch --features fault-inject
sed -n '/^```bash recovery.sh$/,/^```$/{//!p}' bench/results/recovery-plan.md > /tmp/recovery.sh
MODELS=<models dir>; V=target/cuda/release/larch; M=$MODELS/m1.gguf
bash /tmp/recovery.sh kill20 "$V" "$M" 2>&1 | tee /tmp/recovery-kill20.log
bash /tmp/recovery.sh parent "$V" "$M" 2>&1 | tee /tmp/recovery-parent.log
bash /tmp/recovery.sh resume "$V" "$M" 2>&1 | tee /tmp/recovery-resume.log
for k in panic device_lost oom; do bash /tmp/recovery.sh fault $k "$V" "$M" 2>&1 | tee /tmp/recovery-$k.log; done
M8=$MODELS/m8.gguf; R8=<m8 cublas-ub32 dump>   # the runtime OOM: forced requests grow the observation buffers
for n in 6 7 8; do bash /tmp/recovery.sh parity-oom $n "$V" "$M8" "$R8" 2>&1 | tee /tmp/recovery-parity-oom$n.log; done
bash /tmp/recovery.sh fault hang "$V" "$M" 2>&1 | tee /tmp/recovery-hang.log
```

```bash recovery.sh
#!/usr/bin/env bash
# R4/R5/R10 runs under `larch supervise` (bench/results/recovery-plan.md). Linux and WSL.
# usage: recovery.sh kill20|parent|resume LARCH GGUF
#        recovery.sh fault panic|device_lost|oom|hang LARCH GGUF
#        recovery.sh parity-oom N LARCH M8_GGUF M8_REF   (runtime OOM, oom@alloc:N, N = 6..8)
# env: CHILD_CMD replaces `larch chat ...` (dry run); FAULT_STEP (8); OOM_ALLOC (1); GPU (0);
#      PROMPT, PROMPT2. `fault hang` is F4E2c's host-thread park (no GPU hang); a real GPU hang (#66) needs a maintainer watching.
set -euo pipefail
unset LARCH_FAULT # an inherited one would bypass the hang guard and the first-run-only logic

# The supervisor's two log lines, as stamped below (serde_json sorts the keys), paired by `restart`:
# restart n's death line, and the Ready of the run restart n started. The cold start has no ready
# line, so its Ready is the child's own line, teed.
DEATH='^[0-9]+ \{"backoff_ms":[0-9]+,"exit":(null|-?[0-9]+),"exited_ns":'
READY='^[0-9]+ \{"recovery_ms":[0-9]+,"restart":[0-9]+\}$' COLD='^[0-9]+ larch: ready$'
PROMPT=${PROMPT:-Count from 1 to 200, one number per line.}
PROMPT2=${PROMPT2:-Now count back from 200 to 1, one number per line.}
FAILS=0 INCOMPLETE=0 SUP='' TEE='' SEEN_ON_GPU=0 BASE_VRAM='' W='' LOG='' OUT='' ID=''

usage() {
  echo "usage: $0 kill20|parent|resume LARCH GGUF | fault panic|device_lost|oom|hang LARCH GGUF | parity-oom N LARCH M8_GGUF M8_REF" >&2
  exit 2
}
die() { echo "ERROR $*" >&2; echo "RESULT: FAIL (aborted)" >&2; exit 1; }
check() { # check MESSAGE TEST...: PASS or FAIL
  local msg=$1; shift
  if "$@"; then echo "PASS $msg"; else echo "FAIL $msg"; FAILS=$((FAILS + 1)); fi
}
incomplete() { echo "$1 $2"; INCOMPLETE=$((INCOMPLETE + 1)); }
le() { [[ $1 =~ ^[0-9]+$ ]] && [ "$1" -le "$2" ]; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) && [[ $s != Z* ]]; }
# Every supervisor stderr line (the child's, teed, and the supervisor's own), prefixed with unix ns.
stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%s%N)" "$l"; done; }
count() { grep -cE -- "$1" "$LOG" || true; }
lines() { grep -E -- "$1" "$LOG" | cut -d' ' -f2- || true; } # lines PATTERN: matching lines, unstamped
field() { sed -n "s/.*\"$1\": *\(null\|-\?[0-9]\+\).*/\1/p"; }
vram() { nvidia-smi -i "${GPU:-0}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo none; }
compute_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true; }
size() { stat -c %s "$OUT"; }

cleanup() {
  if [ -n "$SUP" ] && alive "$SUP"; then kill "$SUP"; fi
  if [ -s "${W:-/nonexistent}/pids" ]; then
    # Only a pid still named larch: after 20 kills a recorded pid may be an unrelated process now.
    while read -r p; do if alive "$p" && [ "$(ps -o comm= -p "$p")" = larch ]; then kill -9 "$p"; fi; done <"$W/pids"
  fi
}
trap cleanup EXIT

start() { # start LARCH GGUF [FAULT_SPEC]: supervised chat, stdin from a FIFO this script holds open
  [ -z "$(pgrep -x larch || true)" ] || die "a larch process already runs: close it first"
  W=$(mktemp -d "${TMPDIR:-/tmp}/recovery.XXXXXX") ID="r5-$(date +%s)"
  W=$(cd -P "$W" && pwd) # /proc/*/fd links are canonical paths
  LOG=$W/stderr.log OUT=$W/stdout.log
  echo "raw logs: $W"
  local -a child=("$1" chat -m "$2" --resume "$ID")
  if [ -n "${CHILD_CMD:-}" ]; then
    child=(sh -c "exec $CHILD_CMD")
    incomplete "DRY RUN (CHILD_CMD):" "not evidence"
  fi
  if [ -n "${3:-}" ]; then
    # First run only. A marker file, not LARCH_RESTARTED_AT: a fault that exits non-3 must not re-arm.
    export FAULT_MARK=$W/fault-fired FAULT_SPEC=$3
    # shellcheck disable=SC2016 # expanded by the child's sh, not here
    child=(sh -c '[ -e "$FAULT_MARK" ] || { : >"$FAULT_MARK"; export LARCH_FAULT="$FAULT_SPEC"; }; exec "$@"' sh "${child[@]}")
  fi
  local v0
  v0=$(vram); sleep 3; BASE_VRAM=$(vram)
  echo "VRAM MiB baseline: $v0, then $BASE_VRAM 3 s later (the difference is the noise floor)"
  mkfifo "$W/in" "$W/err"
  exec 3<>"$W/in"
  : >"$LOG"
  stamp <"$W/err" | tee -a "$LOG" >&2 &
  TEE=$!
  # stdout to a file: with no terminal on stdin or stdout the supervisor gives the child its own group.
  "$1" supervise -- "${child[@]}" <&3 3<&- >"$OUT" 2>"$W/err" &
  SUP=$!
  echo "supervisor $SUP, transcript id $ID"
}

wait_for() { # wait_for PATTERN N SECONDS: until LOG has N lines matching PATTERN
  local i
  for ((i = 0; i < $3 * 10; i++)); do
    [ "$(count "$1")" -ge "$2" ] && return 0
    alive "$SUP" || die "supervisor gone before $2 x '$1'"
    sleep 0.1
  done
  die "no $2 x '$1' after $3 s"
}

child_pid() { # the supervisor's one live child
  local p
  p=$(pgrep -P "$SUP") || die "supervisor $SUP has no child"
  [ "$(wc -l <<<"$p")" -eq 1 ] || die "supervisor $SUP has several children: $p"
  echo "$p" >>"$W/pids"
  echo "$p"
}

gpu_seen() { # gpu_seen PID, at the first Ready: the child's VRAM footprint, and whether compute apps lists it
  echo "VRAM MiB at Ready: $(vram)"
  if [ -n "$(compute_pids | grep -Fx "$1" || true)" ]; then SEEN_ON_GPU=1; fi
}
no_zombie() { # no_zombie WHEN: no unreaped child under the live supervisor
  local z
  z=$(ps -o stat=,pid= --ppid "$SUP" | awk '$1 ~ /^Z/' || true)
  check "no zombie under the supervisor $1${z:+: $z}" [ -z "$z" ]
}
# Resume lines on stderr stamped after NS (the faulted run's exited_ns), plus any on stdout.
resumed_after() { { awk -v t="$1" '$1 > t' "$LOG"; cat "$OUT"; } | grep -c 'resumed delivered history' || true; }

wait_grow() { # wait_grow SIZE: chat stdout grows past SIZE (a reply is streaming)
  local i
  for ((i = 0; i < 600; i++)); do [ "$(size)" -gt "$1" ] && return 0; sleep 0.1; done
  die "no reply on stdout after 60 s"
}
wait_idle() { local s=-1; while [ "$(size)" != "$s" ]; do s=$(size); sleep 3; done; }

quit() { # /quit; the supervisor must exit 0
  printf '/quit\n' >&3
  local i rc=0
  for ((i = 0; i < 300; i++)); do alive "$SUP" || break; sleep 0.1; done
  if alive "$SUP"; then kill "$SUP"; fi
  wait "$SUP" || rc=$?
  SUP=''
  check "supervisor exited 0 after /quit (rc $rc)" [ "$rc" -eq 0 ]
}

pairs() { # pairs PATTERN FIELD: "restart value" per matching line, sorted for join
  lines "$1" | while IFS= read -r l; do echo "$(field restart <<<"$l") $(field "$2" <<<"$l")"; done | sort -k1,1
}
check_lines() { # check_lines N [fault]: restarts 1..N have one death and one ready line each, every
  # recovery_ms - backoff_ms < 2000, every death line's ready_ms < 2000 (null allowed only for `fault`)
  local want d r ms slow tbl
  want=$(seq "$1")
  d=$(lines "$DEATH" | field restart)
  check "$1 death lines, restart 1..$1 (got: $(tr '\n' ' ' <<<"$d"))" [ "$d" = "$want" ]
  r=$(lines "$READY" | field restart)
  check "$1 ready lines, restart 1..$1, one each (got: $(tr '\n' ' ' <<<"$r"))" [ "$r" = "$want" ]
  # recovery_ms (exit -> Ready) includes the backoff 9.11 mandates; the bound is on the rest.
  tbl=$(join <(pairs "$READY" recovery_ms) <(pairs "$DEATH" backoff_ms) | sort -n | awk '{print $1, $2, $3, $2 - $3}')
  echo "restart:recovery_ms-backoff_ms=rest: $(awk '{printf "%s:%s-%s=%s ", $1, $2, $3, $4}' <<<"$tbl")"
  slow=$(awk '$4 >= 2000 {printf "%s ", $1}' <<<"$tbl")
  [ "$(cut -d' ' -f1 <<<"$tbl")" = "$want" ] || slow+="(not every restart 1..$1 paired) "
  check "every restart's recovery_ms - backoff_ms < 2000${slow:+ (not restart: $slow)}" [ -z "$slow" ]
  ms=$(lines "$DEATH" | field ready_ms)
  echo "ready_ms (start -> Ready of the run that died): $(tr '\n' ' ' <<<"$ms")"
  slow=$(grep -vxE "[0-9]{1,3}|1[0-9]{3}${2:+|null}" <<<"$ms" | tr '\n' ' ' || true)
  check "every death line's ready_ms < 2000${2:+ or null}${slow:+ (not: $slow)}" [ -z "$slow" ]
}

finish() { # leftovers, VRAM, compute apps, then the verdict
  local left after stale
  exec 3>&-
  wait "$TEE" || true
  sleep 2 # settle before the VRAM reading
  left=$(pgrep -a -x larch || true)
  check "no larch process left${left:+: $left}" [ -z "$left" ]
  # Every process of the tree inherits the run's stdin FIFO and stdout file, whatever its name or group.
  left=$({ find /proc/[0-9]*/fd -lname "$W/*" 2>/dev/null || true; } | cut -d/ -f3 | sort -u | tr '\n' ' ')
  check "no process left holding the run's stdin or stdout${left:+: $left}" [ -z "$left" ]
  after=$(vram)
  echo "VRAM MiB before: $BASE_VRAM after: $after"
  if [[ $BASE_VRAM =~ ^[0-9]+$ && $after =~ ^[0-9]+$ ]]; then
    check "VRAM back to baseline within 64 MiB ($((after - BASE_VRAM)) MiB)" [ $((after - BASE_VRAM)) -le 64 ]
  else
    incomplete "NOT OBSERVABLE" "VRAM: nvidia-smi gave no number"
  fi
  if [ "$SEEN_ON_GPU" = 1 ]; then
    stale=$(compute_pids | grep -Fx -f "$W/pids" || true)
    check "no larch pid in nvidia-smi compute apps${stale:+: $stale}" [ -z "$stale" ]
  else
    incomplete "NOT OBSERVABLE" "compute apps: nvidia-smi never listed the child (none at all in WSL)"
  fi
  verdict
}
verdict() {
  if [ "$FAILS" -gt 0 ]; then echo "RESULT: FAIL ($FAILS failed, $INCOMPLETE incomplete)"; exit 1; fi
  if [ "$INCOMPLETE" -gt 0 ]; then echo "RESULT: INCOMPLETE ($INCOMPLETE not proven), not a pass"; exit 4; fi
  echo "RESULT: PASS"
}

kill20() { # R5: 20 kill -9 in a row
  local k pid exits
  start "$1" "$2"
  for ((k = 1; k <= 20; k++)); do
    if [ "$k" -eq 1 ]; then wait_for "$COLD" 1 120; else wait_for "$READY" $((k - 1)) 120; fi
    pid=$(child_pid)
    if [ "$k" -eq 1 ]; then gpu_seen "$pid"; fi
    kill -9 "$pid"
  done
  wait_for "$READY" 20 120
  no_zombie "after 20 kills"
  quit
  check_lines 20
  exits=$(lines "$DEATH" | field exit | grep -vx null || true)
  check "every exit null (killed by a signal)${exits:+ (not: $exits)}" [ -z "$exits" ]
  finish
}

parent() { # the child dies with its supervisor
  local pid t0 ms=never
  start "$1" "$2"
  wait_for "$COLD" 1 120
  pid=$(child_pid)
  gpu_seen "$pid"
  t0=$(date +%s%N)
  kill -9 "$SUP"
  wait "$SUP" || true
  SUP=''
  while alive "$pid" && [ $(($(date +%s%N) - t0)) -lt 3000000000 ]; do sleep 0.01; done
  if ! alive "$pid"; then ms=$((($(date +%s%N) - t0) / 1000000)); fi
  check "child $pid gone $ms ms after kill -9 of the supervisor (limit 1000)" le "$ms" 1000
  finish
}

resume() { # R4: kill -9 mid-reply; the transcript survives and resumes once
  local s pid transcript n
  start "$1" "$2"
  wait_for "$COLD" 1 120
  s=$(size); printf '%s\n' "$PROMPT" >&3; wait_grow "$s"; wait_idle
  s=$(size); printf '%s\n' "$PROMPT2" >&3; wait_grow "$s"
  pid=$(child_pid)
  gpu_seen "$pid"
  kill -9 "$pid"
  wait_for "$DEATH" 1 60
  wait_for "$READY" 1 120
  sleep 1 # the restarted chat prints its resume line
  no_zombie "after the restart"
  lines "$DEATH"
  check_lines 1
  n=$(resumed_after "$(lines "$DEATH" | head -n1 | field exited_ns)")
  check "resumed delivered history after the restart, once ($n)" [ "$n" -eq 1 ]
  s=$(size); printf '%s\n' "$PROMPT" >&3; wait_grow "$s"; wait_idle
  echo "MANUAL: the reply after the restart continues the conversation, re-prefilled once (stdout tail):"
  tail -n 5 "$OUT"
  quit
  # After /quit: on Windows the live chat holds a byte-range lock on its transcript.
  transcript=${XDG_DATA_HOME:-$HOME/.local/share}/larch/chat/$ID.jsonl
  check "transcript $transcript holds turn 1" grep -qF "$PROMPT" "$transcript"
  finish
}

fault() { # R4/R10: an injected fault in the first run only
  local kind=$1 spec line code pid n det gap
  case $kind in
    panic | device_lost | hang) spec=$kind@step:${FAULT_STEP:-8} ;;
    oom) spec=oom@alloc:${OOM_ALLOC:-1} ;;
    *) usage ;;
  esac
  start "$2" "$3" "$spec"
  wait_for "$COLD|$DEATH" 1 120
  # Prompt run 1 if it printed Ready before dying; its death line's ready_ms says so for the record.
  if grep -m1 -E "$COLD|$DEATH" "$LOG" | grep -qE "$COLD"; then
    pid=$(child_pid)
    gpu_seen "$pid"
    printf '%s\n' "$PROMPT" >&3
  fi
  wait_for "$DEATH" 1 60
  wait_for "$READY" 1 120
  # A run that died before Ready (a startup OOM) was never probed: probe the restarted child.
  if [ -z "${pid:-}" ]; then gpu_seen "$(child_pid)"; fi
  sleep 1 # the restarted chat prints its resume line
  line=$(lines "$DEATH" | head -n1)
  echo "death line: $line"
  echo "faulted run's ready_ms: $(field ready_ms <<<"$line") (null: the fault fired before Ready)"
  code=$(field exit <<<"$line")
  det=$(lines 'larch fault=[A-Za-z]+ detected_ns=[0-9]+$' | head -n1)
  if [ "$kind" = oom ]; then
    # F4E2c counts chat's allocations at startup only: LoadError::DoesNotFit, a failed start, not a fault.
    check "exit 1, a failed start ($code)" [ "$code" = 1 ]
    check "no fault line (startup OOM is a load error)${det:+: $det}" [ -z "$det" ]
  else
    check "exit 3 ($code)" [ "$code" = 3 ]
    echo "fault line: ${det:-none}"
    gap=none
    if [ -n "$det" ]; then gap=$(($(field exited_ns <<<"$line") - ${det##*detected_ns=})); fi
    check "exited_ns - detected_ns = $gap ns, within 0..100 ms" le "$gap" 100000000
  fi
  check_lines 1 fault
  if [ "$(field ready_ms <<<"$line")" = null ]; then
    # chat creates the transcript only after a load, so a run that never loaded leaves none to resume.
    echo "resume: n/a (the faulted run never got Ready, so the restart starts a fresh transcript)"
  else
    n=$(resumed_after "$(field exited_ns <<<"$line")")
    check "resumed delivered history after the restart, once ($n)" [ "$n" -eq 1 ]
  fi
  echo "MANUAL caller-visible error: stderr_tail in the death line above; chat stdout tail:"
  tail -n 3 "$OUT"
  quit
  finish
}

parity_oom() { # runtime OOM: oom@alloc:N hits the observation buffers, which only forced requests grow
  local n=$1 rc=0 t det gap=none
  [ -z "$(pgrep -x larch || true)" ] || die "a larch process already runs: close it first"
  W=$(mktemp -d "${TMPDIR:-/tmp}/recovery.XXXXXX")
  echo "raw logs: $W"
  # m8's calibrated bound and margin; no supervisor, so the exit time is taken here.
  LARCH_FAULT=oom@alloc:$n "$2" parity --device "${GPU:-0}" -m "$3" --ref "$4" --bound 0.021716577412409198 \
    --margin 0.42968761920928955 --mode prefill --prompts 1 >"$W/parity.out" 2>"$W/parity.err" || rc=$?
  t=$(date +%s%N) # after the reap: an upper bound on the exit time, so the gap can only read long
  det=$(grep -m1 -oE 'larch fault=[A-Za-z]+ detected_ns=[0-9]+' "$W/parity.err" || true)
  echo "exit $rc, exited_ns=$t, fault line: ${det:-none}"
  if [ "$rc" = 0 ]; then
    incomplete "NOT REACHED" "oom@alloc:$n: parity passed, nothing fired"
  else
    check "exit 3, the fault path ($rc)" [ "$rc" = 3 ]
    if [ -n "$det" ]; then gap=$((t - ${det##*detected_ns=})); fi
    check "exited_ns - detected_ns = $gap ns, within 0..100 ms" le "$gap" 100000000
  fi
  verdict
}

case ${1:-} in
  kill20 | parent | resume) [ $# -eq 3 ] || usage; "$@" ;;
  fault) [ $# -eq 4 ] || usage; "$@" ;;
  parity-oom) [ $# -eq 5 ] || usage; shift; parity_oom "$@" ;;
  *) usage ;;
esac
```

## Native Windows: `recovery.ps1`

Run unattended from WSL through Windows interop (`powershell.exe`), in the Windows clone
`%USERPROFILE%\src\larch`, checked out detached at `HEAD_SHA` so no branch is checked out in two clones.
The 64-bit dev shell comes from `Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64`, with CUDA 13.4.2 at
`C:\cuda\13.4.2`. `CARGO_TARGET_DIR=C:\vt`, because a space in the clone's path breaks nvcc.
`LARCH_CMAKE_GENERATOR=Ninja`, because a zip-installed CUDA has no Visual Studio integration, so the
default generator fails with "No CUDA toolset found". Cargo goes through `cmd /c` with its output
redirected. Copy the GGUFs to a local Windows disk first: over `\\wsl.localhost` start→Ready is
I/O-bound. Zombies do not exist on Windows; the process check covers leftovers. A killed process shows
`exit: -1` there (TerminateProcess), not `null`. Hold the shared GPU lock (`flock`, from WSL) around the runs.

`devshell.ps1`, next to `recovery.ps1` (outside the clone):

```
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$env:CUDA_PATH = 'C:\cuda\13.4.2'; $env:CUDACXX = 'C:\cuda\13.4.2\bin\nvcc.exe'
$env:PATH = "C:\cuda\13.4.2\bin;C:\cuda\13.4.2\bin\x64;$env:PATH"
$env:CARGO_TARGET_DIR = 'C:\vt'; $env:LARCH_BACKEND = 'CUDA'; $env:LARCH_CMAKE_GENERATOR = 'Ninja'
Remove-Item Env:LARCH_FAULT -ErrorAction SilentlyContinue
Set-Location "$env:USERPROFILE\src\larch"
```

From WSL (`D` = that folder, as a Windows path; `MODELS` = the local model dir on Windows; `GPU_LOCK` = the shared GPU lock file):

```
cd "$(wslpath -u "$D")"
sed -n '/^```powershell recovery.ps1$/,/^```$/{//!p}' <repo>/bench/results/recovery-plan.md > recovery.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". .\devshell.ps1; git fetch origin; git checkout --detach $HEAD_SHA; cmd /c 'cargo build --release -p larch --features fault-inject > build.log 2>&1'"
P() { flock "${GPU_LOCK:?the shared GPU lock file}" powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '$D\devshell.ps1'; & '$D\recovery.ps1' $*"; }
P kill20 'C:\vt\release\larch.exe' "'$MODELS\m1.gguf'" > kill20.log 2>&1   # likewise parent, resume, fault panic|device_lost|oom|hang
```

Set `OOM_ALLOC`, `FAULT_STEP`, `GPU`, `PROMPT` or `PROMPT2` inside the `-Command` string (`$env:OOM_ALLOC='2';`): they reach `powershell.exe` from WSL only that way, or through `WSLENV`. (`fault-inject` forwards F4E2c's `larch-engine/fault-inject`, #65. Leave `TdrDelay` at its default ≈ 2 s. The logs stay outside git.)

```powershell recovery.ps1
# R4/R5/R10 runs under `larch supervise` (bench/results/recovery-plan.md). Native Windows.
# usage: recovery.ps1 kill20|parent|resume LARCH GGUF
#        recovery.ps1 fault panic|device_lost|oom|hang LARCH GGUF
#        recovery.ps1 parity-oom N LARCH M8_GGUF M8_REF   (runtime OOM, oom@alloc:N, N = 6..8)
# env: FAULT_STEP (8); OOM_ALLOC (1); GPU (0); PROMPT, PROMPT2. `fault hang` is a host-thread park; a real GPU hang (#66) needs a maintainer watching.
param([string]$Cmd, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$ErrorActionPreference = 'Stop'
Remove-Item Env:LARCH_FAULT -ErrorAction SilentlyContinue # would bypass the hang guard and the first-run-only logic

# The supervisor's two log lines, as stamped by Pump (serde_json sorts the keys), paired by `restart`:
# restart n's death line, and the Ready of the run restart n started. The cold start has no ready
# line, so its Ready is the child's own line, teed.
$Death = '^\d+ \{"backoff_ms":\d+,"exit":(null|-?\d+),"exited_ns":'
$Ready = '^\d+ \{"recovery_ms":\d+,"restart":\d+\}$'; $Cold = '^\d+ larch: ready$'
$Prompt = if ($env:PROMPT) { $env:PROMPT } else { 'Count from 1 to 200, one number per line.' }
$Prompt2 = if ($env:PROMPT2) { $env:PROMPT2 } else { 'Now count back from 200 to 1, one number per line.' }
$Gpu = if ($env:GPU) { $env:GPU } else { '0' }
$script:Fails = 0; $script:Incomplete = 0; $script:Sup = $null; $script:SeenOnGpu = $false
$script:Pids = @(); $script:Lines = New-Object Collections.Generic.List[string]; $script:OutLines = 0

function Usage { [Console]::Error.WriteLine('usage: recovery.ps1 kill20|parent|resume LARCH GGUF | fault panic|device_lost|oom|hang LARCH GGUF | parity-oom N LARCH M8_GGUF M8_REF'); exit 2 }
function Die($m) { [Console]::Error.WriteLine("ERROR $m"); Write-Host 'RESULT: FAIL (aborted)'; exit 1 }
function Check($msg, [bool]$ok) { if ($ok) { Write-Host "PASS $msg" } else { Write-Host "FAIL $msg"; $script:Fails++ } }
function Incomplete($tag, $msg) { Write-Host "$tag $msg"; $script:Incomplete++ }
# The clock Rust's SystemTime reads (detected_ns, exited_ns); PS 5.1's DateTime.UtcNow follows the timer tick.
Add-Type -Namespace Recovery -Name Clock -MemberDefinition '[DllImport("kernel32.dll")] public static extern void GetSystemTimePreciseAsFileTime(out long t);'
function NowNs { $t = 0L; [Recovery.Clock]::GetSystemTimePreciseAsFileTime([ref]$t); ($t - 116444736000000000) * 100 }
function Q($s) { '"' + $s + '"' }
function Alive($p) { [bool](Get-Process -Id $p -ErrorAction SilentlyContinue) }
function Seen($pattern) { @($script:Lines | Where-Object { $_ -match $pattern }).Count }
function Json($pattern) { @($script:Lines | Where-Object { $_ -match $pattern } | ForEach-Object { $_.Substring($_.IndexOf(' ') + 1) | ConvertFrom-Json }) }
function Vram { try { "$(& nvidia-smi -i $Gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)".Trim() } catch { 'none' } }
function ComputePids { try { @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null | ForEach-Object { "$_".Trim() }) } catch { @() } }

function Pump { # drain both pipes: an undrained pipe blocks the supervisor and the child behind it
  while ($script:ErrTask -and $script:ErrTask.IsCompleted) {
    $l = $script:ErrTask.Result
    if ($null -eq $l) { $script:ErrTask = $null; break }
    $s = "$(NowNs) $l"
    $script:Lines.Add($s); [IO.File]::AppendAllText($script:Log, $s + "`n"); [Console]::Error.WriteLine($s)
    $script:ErrTask = $script:Sup.StandardError.ReadLineAsync()
  }
  while ($script:OutTask -and $script:OutTask.IsCompleted) {
    $l = $script:OutTask.Result
    if ($null -eq $l) { $script:OutTask = $null; break }
    [IO.File]::AppendAllText($script:Out, $l + "`n"); $script:OutLines++
    $script:OutTask = $script:Sup.StandardOutput.ReadLineAsync()
  }
}
function Nap($ms) { Pump; Start-Sleep -Milliseconds $ms; Pump }
function Drain { $end = (Get-Date).AddSeconds(2); while (($script:ErrTask -or $script:OutTask) -and (Get-Date) -lt $end) { Nap 50 } }
function Send($text) { $b = [Text.Encoding]::UTF8.GetBytes($text + "`n"); $s = $script:Sup.StandardInput.BaseStream; $s.Write($b, 0, $b.Length); $s.Flush() }

function Start-Sup($larch, $gguf, $spec) {
  if (Get-Process larch -ErrorAction SilentlyContinue) { Die 'a larch process already runs: close it first' }
  $larch = (Resolve-Path $larch).Path; $gguf = (Resolve-Path $gguf).Path
  $script:W = Join-Path $env:TEMP ('recovery-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
  New-Item -ItemType Directory $script:W | Out-Null
  $script:Log = Join-Path $script:W 'stderr.log'; $script:Out = Join-Path $script:W 'stdout.log'
  New-Item -ItemType File $script:Log, $script:Out | Out-Null
  $script:Id = 'r5-' + [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  Write-Host "raw logs: $script:W"
  $child = (Q $larch) + ' chat -m ' + (Q $gguf) + ' --resume ' + $script:Id
  if ($spec) {
    # First run only, through a marker file. A .cmd because supervise quotes its arguments for the C
    # runtime, which `cmd /c "..."` does not parse.
    # The paths go through the environment (UTF-16), not the file: -Encoding ASCII corrupts non-ASCII ones.
    $f = Join-Path $script:W 'fault.cmd'
    Set-Content -Encoding ASCII $f @('@echo off', 'if not exist "%~dp0fired" (', '  type nul > "%~dp0fired"', "  set `"LARCH_FAULT=$spec`"", ')', '"%VV_EXE%" chat -m "%VV_GGUF%" --resume %VV_ID%')
    $child = Q $f
    $fault = $true
  }
  $v0 = Vram; Start-Sleep 3; $script:BaseVram = Vram
  Write-Host "VRAM MiB baseline: $v0, then $script:BaseVram 3 s later (the difference is the noise floor)"
  $psi = New-Object Diagnostics.ProcessStartInfo -ArgumentList $larch, ('supervise -- ' + $child)
  if ($fault) { $psi.EnvironmentVariables['VV_EXE'] = $larch; $psi.EnvironmentVariables['VV_GGUF'] = $gguf; $psi.EnvironmentVariables['VV_ID'] = $script:Id }
  $psi.UseShellExecute = $false; $psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
  $script:Sup = [Diagnostics.Process]::Start($psi)
  $script:ErrTask = $script:Sup.StandardError.ReadLineAsync(); $script:OutTask = $script:Sup.StandardOutput.ReadLineAsync()
  Write-Host "supervisor $($script:Sup.Id), transcript id $script:Id"
}

function WaitFor($pattern, $n, $secs) {
  $end = (Get-Date).AddSeconds($secs)
  while ((Seen $pattern) -lt $n) {
    if ($script:Sup.HasExited) { Drain; if ((Seen $pattern) -ge $n) { return }; Die "supervisor gone before $n x '$pattern'" }
    if ((Get-Date) -gt $end) { Die "no $n x '$pattern' after $secs s" }
    Nap 100
  }
}
function ChildPid { # the supervisor's one larch child
  $c = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($script:Sup.Id) AND Name='larch.exe'")
  if ($c.Count -ne 1) { Die "supervisor $($script:Sup.Id) has $($c.Count) larch children" }
  [int]$c[0].ProcessId
}
function GpuProbe { # at the first Ready: the VRAM footprint; every larch but the supervisor, and whether compute apps lists one
  Write-Host "VRAM MiB at Ready: $(Vram)"
  $ids = @(Get-Process larch -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $script:Sup.Id } | ForEach-Object { $_.Id })
  $script:Pids += $ids
  $apps = ComputePids
  foreach ($i in $ids) { if ($apps -contains "$i") { $script:SeenOnGpu = $true } }
}
# Resume lines on stderr stamped after NS (the faulted run's exited_ns), plus any on stdout.
function ResumedAfter($ns) {
  @(@($script:Lines | Where-Object { [decimal]$_.Substring(0, $_.IndexOf(' ')) -gt [decimal]$ns }) + @(Get-Content $script:Out) | Where-Object { $_ -match 'resumed delivered history' }).Count
}
function WaitGrow($n) { $end = (Get-Date).AddSeconds(60); while ($script:OutLines -le $n) { if ((Get-Date) -gt $end) { Die 'no reply on stdout after 60 s' }; Nap 100 } }
function WaitIdle { $n = -1; while ($script:OutLines -ne $n) { $n = $script:OutLines; for ($i = 0; $i -lt 30; $i++) { Nap 100 } } }

function Quit {
  Send '/quit'
  $end = (Get-Date).AddSeconds(30)
  while (-not $script:Sup.HasExited -and (Get-Date) -lt $end) { Nap 100 }
  if (-not $script:Sup.HasExited) { Stop-Process -Id $script:Sup.Id -Force; $script:Sup.WaitForExit() }
  Drain
  Check "supervisor exited 0 after /quit (rc $($script:Sup.ExitCode))" ($script:Sup.ExitCode -eq 0)
}
function Nul($v) { if ($null -eq $v) { 'null' } else { $v } }
# Restarts 1..n have one death and one ready line each, every recovery_ms - backoff_ms < 2000, every
# death line's ready_ms < 2000 (null allowed only after a fault).
function CheckLines($n, [bool]$fault) {
  $d = @(Json $Death); $r = @(Json $Ready); $want = (1..$n) -join ' '
  $got = ($d | ForEach-Object { $_.restart }) -join ' '
  Check "$n death lines, restart 1..$n (got: $got)" ($got -eq $want)
  $got = ($r | ForEach-Object { $_.restart }) -join ' '
  Check "$n ready lines, restart 1..$n, one each (got: $got)" ($got -eq $want)
  # recovery_ms (exit -> Ready) includes the backoff 9.11 mandates; the bound is on the rest.
  $bk = @{}; foreach ($x in $d) { $bk["$($x.restart)"] = $x.backoff_ms }
  $rows = @($r | ForEach-Object { $b = $bk["$($_.restart)"]; [pscustomobject]@{ n = $_.restart; rec = $_.recovery_ms; bk = $b; rest = $(if ($null -ne $b) { $_.recovery_ms - $b }) } })
  Write-Host "restart:recovery_ms-backoff_ms=rest: $(($rows | ForEach-Object { "$($_.n):$($_.rec)-$(Nul $_.bk)=$(Nul $_.rest)" }) -join ' ')"
  Write-Host "ready_ms (start -> Ready of the run that died): $(($d | ForEach-Object { Nul $_.ready_ms }) -join ' ')"
  Write-Host "exit: $(($d | ForEach-Object { Nul $_.exit }) -join ' ')"
  $slow = @($rows | Where-Object { $null -eq $_.rest -or $_.rest -ge 2000 })
  Check "every restart's recovery_ms - backoff_ms < 2000 ($($slow.Count) not)" ($slow.Count -eq 0 -and $rows.Count -eq $n)
  $slow = @($d | Where-Object { if ($null -eq $_.ready_ms) { -not $fault } else { $_.ready_ms -ge 2000 } })
  Check "every death line's ready_ms < 2000$(if ($fault) { ' or null' }) ($($slow.Count) not)" ($slow.Count -eq 0)
}
function Finish {
  Drain; Start-Sleep 2
  $left = @(Get-Process larch -ErrorAction SilentlyContinue)
  Check "no larch process left $(($left | ForEach-Object { $_.Id }) -join ' ')" ($left.Count -eq 0)
  $after = Vram; Write-Host "VRAM MiB before: $script:BaseVram after: $after"
  if ("$script:BaseVram" -match '^\d+$' -and "$after" -match '^\d+$') {
    $d = [int]$after - [int]$script:BaseVram; Check "VRAM back to baseline within 64 MiB ($d MiB)" ($d -le 64)
  } else { Incomplete 'NOT OBSERVABLE' 'VRAM: nvidia-smi gave no number' }
  if ($script:SeenOnGpu) {
    $stale = @(ComputePids | Where-Object { $script:Pids -contains [int]$_ })
    Check "no larch pid in nvidia-smi compute apps $($stale -join ' ')" ($stale.Count -eq 0)
  } else { Incomplete 'NOT OBSERVABLE' 'compute apps: nvidia-smi never listed the child' }
  Verdict
}
function Verdict {
  if ($script:Fails -gt 0) { Write-Host "RESULT: FAIL ($script:Fails failed, $script:Incomplete incomplete)"; exit 1 }
  if ($script:Incomplete -gt 0) { Write-Host "RESULT: INCOMPLETE ($script:Incomplete not proven), not a pass"; exit 4 }
  Write-Host 'RESULT: PASS'; exit 0
}

function kill20($larch, $gguf) { # R5: 20 forced kills in a row
  Start-Sup $larch $gguf $null
  for ($k = 1; $k -le 20; $k++) {
    if ($k -eq 1) { WaitFor $Cold 1 120; GpuProbe } else { WaitFor $Ready ($k - 1) 120 }
    $p = ChildPid; $script:Pids += $p; Stop-Process -Id $p -Force
  }
  WaitFor $Ready 20 120
  Quit; CheckLines 20 $false
  # Stop-Process -Force is TerminateProcess with exit code -1; null is a signal death, as on Unix.
  $exits = @(Json $Death | Where-Object { $null -ne $_.exit -and $_.exit -ne -1 } | ForEach-Object { $_.exit })
  Check "every exit -1 or null (killed)$(if ($exits) { " (not: $($exits -join ' '))" })" ($exits.Count -eq 0)
  Finish
}
function parent($larch, $gguf) { # the child dies with its supervisor (Job Object kill-on-close)
  Start-Sup $larch $gguf $null
  WaitFor $Cold 1 120
  GpuProbe; $p = ChildPid
  $t0 = Get-Date; Stop-Process -Id $script:Sup.Id -Force
  while ((Alive $p) -and ((Get-Date) - $t0).TotalMilliseconds -lt 3000) { Start-Sleep -Milliseconds 10 }
  $ms = [int]((Get-Date) - $t0).TotalMilliseconds
  Check "child $p gone $ms ms after Stop-Process -Force of the supervisor (limit 1000)" ((-not (Alive $p)) -and $ms -le 1000)
  Finish
}
function resume($larch, $gguf) { # R4: forced kill mid-reply; the transcript survives and resumes once
  Start-Sup $larch $gguf $null
  WaitFor $Cold 1 120
  $n = $script:OutLines; Send $Prompt; WaitGrow $n; WaitIdle
  $n = $script:OutLines; Send $Prompt2; WaitGrow $n
  GpuProbe; Stop-Process -Id (ChildPid) -Force
  WaitFor $Death 1 60
  WaitFor $Ready 1 120
  Nap 1000 # the restarted chat prints its resume line
  CheckLines 1 $false
  $resumed = ResumedAfter (@(Json $Death)[0].exited_ns)
  Check "resumed delivered history after the restart, once ($resumed)" ($resumed -eq 1)
  $n = $script:OutLines; Send $Prompt; WaitGrow $n; WaitIdle
  Write-Host 'MANUAL: the reply after the restart continues the conversation, re-prefilled once (stdout tail):'
  Get-Content $script:Out -Tail 5
  Quit
  # After /quit: the live chat holds a byte-range lock on its transcript, which Select-String cannot read.
  $t = Join-Path $env:LOCALAPPDATA "larch\chat\$script:Id.jsonl"
  Check "transcript $t holds turn 1" ((Test-Path $t) -and (Select-String -SimpleMatch -Quiet -Path $t -Pattern $Prompt))
  Finish
}
function fault($kind, $larch, $gguf) { # R4/R10: an injected fault in the first run only
  $step = if ($env:FAULT_STEP) { $env:FAULT_STEP } else { '8' }
  $alloc = if ($env:OOM_ALLOC) { $env:OOM_ALLOC } else { '1' }
  switch ($kind) {
    { $_ -in 'panic', 'device_lost', 'hang' } { $spec = "$kind@step:$step" }
    'oom' { $spec = "oom@alloc:$alloc" }
    default { Usage }
  }
  Start-Sup $larch $gguf $spec
  WaitFor "$Cold|$Death" 1 120
  # Prompt run 1 if it printed Ready before dying; its death line's ready_ms says so for the record.
  $probed = $false
  if (@($script:Lines | Where-Object { $_ -match $Cold -or $_ -match $Death })[0] -match $Cold) { GpuProbe; $probed = $true; Send $Prompt }
  WaitFor $Death 1 60
  WaitFor $Ready 1 120
  # A run that died before Ready (a startup OOM) was never probed: probe the restarted child.
  if (-not $probed) { GpuProbe }
  Nap 1000 # the restarted chat prints its resume line
  $r = @(Json $Death)[0]
  Write-Host "death line: $($r | ConvertTo-Json -Compress)"
  Write-Host "faulted run's ready_ms: $(Nul $r.ready_ms) (null: the fault fired before Ready)"
  $det = $null; foreach ($l in $script:Lines) { if ($l -match 'larch fault=\w+ detected_ns=(\d+)$') { $det = $l; $ns = [decimal]$Matches[1]; break } }
  # F4E2c counts chat's allocations at startup only: LoadError::DoesNotFit, a failed start, not a fault.
  if ($kind -eq 'oom') { Check "exit 1, a failed start ($($r.exit))" ($r.exit -eq 1); Check "no fault line (startup OOM is a load error) $det" ($null -eq $det) }
  else {
    Check "exit 3 ($($r.exit))" ($r.exit -eq 3)
    Write-Host "fault line: $(Nul $det)"
    $gap = if ($det) { [decimal]$r.exited_ns - $ns } else { 'none' }
    Check "exited_ns - detected_ns = $gap ns, within 0..100 ms" ($det -and $gap -ge 0 -and $gap -le 100000000)
  }
  CheckLines 1 $true
  # chat creates the transcript only after a load, so a run that never loaded leaves none to resume.
  if ($null -eq $r.ready_ms) { Write-Host 'resume: n/a (the faulted run never got Ready, so the restart starts a fresh transcript)' }
  else {
    $resumed = ResumedAfter $r.exited_ns
    Check "resumed delivered history after the restart, once ($resumed)" ($resumed -eq 1)
  }
  Write-Host 'MANUAL caller-visible error: stderr_tail in the death line above; chat stdout tail:'
  Get-Content $script:Out -Tail 3
  Quit; Finish
}

function ParityOom($n, $larch, $gguf, $ref) { # runtime OOM: oom@alloc:N hits the observation buffers, which only forced requests grow
  if (Get-Process larch -ErrorAction SilentlyContinue) { Die 'a larch process already runs: close it first' }
  $w = Join-Path ([IO.Path]::GetTempPath()) ('recovery.' + [IO.Path]::GetRandomFileName())
  New-Item -ItemType Directory $w | Out-Null; Write-Host "raw logs: $w"
  # m8's calibrated bound and margin; no supervisor, so this script takes the exit time.
  $a = "parity --device $Gpu -m $(Q (Resolve-Path $gguf).Path) --ref $(Q (Resolve-Path $ref).Path) --bound 0.021716577412409198 --margin 0.42968761920928955 --mode prefill --prompts 1"
  $env:LARCH_FAULT = "oom@alloc:$n"
  $p = Start-Process -FilePath (Resolve-Path $larch).Path -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$w\parity.out" -RedirectStandardError "$w\parity.err"
  $null = $p.Handle # keeps the handle, so ExitCode stays readable after the exit
  Remove-Item Env:LARCH_FAULT
  $p.WaitForExit()
  $t = NowNs # after the reap: an upper bound on the exit time, so the gap can only read long (as recovery.sh)
  $det = Select-String -Path "$w\parity.err" -Pattern 'larch fault=\w+ detected_ns=(\d+)' | Select-Object -First 1
  Write-Host "exit $($p.ExitCode), exited_ns=$t, fault line: $(if ($det) { $det.Matches[0].Value } else { 'none' })"
  if ($p.ExitCode -eq 0) { Incomplete 'NOT REACHED' "oom@alloc:${n}: parity passed, nothing fired" }
  else {
    Check "exit 3, the fault path ($($p.ExitCode))" ($p.ExitCode -eq 3)
    $gap = if ($det) { [decimal]$t - [decimal]$det.Matches[0].Groups[1].Value } else { 'none' }
    Check "exited_ns - detected_ns = $gap ns, within 0..100 ms" ($det -and $gap -ge 0 -and $gap -le 100000000)
  }
  Verdict
}

try {
  if ($Cmd -in 'kill20', 'parent', 'resume') { if ($Rest.Count -ne 2) { Usage }; & $Cmd $Rest[0] $Rest[1] }
  elseif ($Cmd -eq 'fault') { if ($Rest.Count -ne 3) { Usage }; fault $Rest[0] $Rest[1] $Rest[2] }
  elseif ($Cmd -eq 'parity-oom') { if ($Rest.Count -ne 4) { Usage }; ParityOom $Rest[0] $Rest[1] $Rest[2] $Rest[3] }
  else { Usage }
} finally {
  if ($script:Sup -and -not $script:Sup.HasExited) { Stop-Process -Id $script:Sup.Id -Force }
  # Only a pid still named larch: Windows reuses pids fast, so after 20 kills one may be another process.
  foreach ($p in $script:Pids) { if ((Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName -eq 'larch') { Stop-Process -Id $p -Force } }
}
```

## Checklist

`sh` = `bash /tmp/recovery.sh`, `ps1` = `P` above. Every Native Windows result was run unattended through WSL interop. Pass conditions are what the script prints as PASS; MANUAL ones
are the operator's reading of what it prints (the death line's `stderr_tail`, chat's stdout tail).

| Proof | Command | Pass condition | WSL | Native Windows |
|---|---|---|---|---|
| Injected panic | `sh`/`ps1 fault panic` | exit 3, a fault line, restart 1's ready line, `recovery_ms − backoff_ms` < 2000; MANUAL: callers see `Internal` | FAIL: exit 3, Ready < 2 s, but no fault line (#74) | FAIL: exit 3, Ready < 2 s, no fault line (#74) |
| Injected device_lost (synthetic until #66) | `fault device_lost` | exit 3, a fault line, restart 1's ready line, `recovery_ms − backoff_ms` < 2000; MANUAL: callers see `DeviceLost` | FAIL: exit 3, Ready < 2 s, but no fault line (#74) | FAIL: exit 3, Ready < 2 s, no fault line (#74) |
| Hang → watchdog (a host-thread park until #66) | `fault hang` (unattended: it never touches the GPU) | watchdog (≥ 5 s) exits 3, fault line `fault=Timeout`; MANUAL: callers see `DeviceLost` (no `Timeout` variant until #67) | FAIL: exit 3, Ready < 2 s, but no fault line (#74) | FAIL: exit 3, Ready < 2 s, no fault line (#74) |
| Injected oom | `fault oom` (`OOM_ALLOC=1..5`) | exit 1 (startup `DoesNotFit`), no fault line, restart 1's ready line, `recovery_ms − backoff_ms` < 2000, the restarted child seen and gone in compute apps | PASS ×5 | PASS ×5 |
| Runtime oom (observation buffers) | `parity-oom 6` (7, 8) `LARCH M8_GGUF M8_REF` | exit 3, the fault line, `0 ≤ exited_ns − detected_ns ≤ 100 ms`; NOT REACHED if parity passes | FAIL ×3: exit 3, no fault line (#74) | FAIL ×3: exit 3, no fault line (#74) |
| Kill | `resume` (kill -9 / Stop-Process mid-reply) | restart 1's death and ready line, `recovery_ms − backoff_ms` < 2000, the killed run's `ready_ms` < 2000 | PASS | PASS (run 2; run 1's script stopped on the transcript lock, fixed) |
| Real Windows TDR | a GPU hang (#66's debug device op), attended | TDR → exit 3 → restart 1's `recovery_ms − backoff_ms` < 2000 (backoff 0 after exit 3); MANUAL: callers see `DeviceLost` | BLOCKED (#66: the injected hang is no GPU hang) | BLOCKED (#66) |
| Exit 3 ≤ 100 ms after detection | `fault panic`, `fault device_lost`, `fault hang` | `0 ≤ exited_ns − detected_ns ≤ 100 ms` (`detected_ns` from the fault line) | NOT PROVEN: no fault line (#74) | NOT PROVEN: no fault line (#74) |
| Start → Ready < 2 s | `kill20`, `resume`, `fault` | `ready_ms` on death lines (start → Ready of the run that died, the cold start's included) < 2000, and `recovery_ms − backoff_ms` on each restart < 2000; raw `recovery_ms` includes the backoff §9.11 mandates for non-fault deaths, and both are printed per restart. `fault` allows a null `ready_ms` (the fault fired before Ready) | PASS (max 1788 ms) | FAIL in kill20 run 1 (cold start 2271 ms); PASS in run 2 and every other run |
| Transcript resume | `resume` | `resumed delivered history` once, turn 1 in the transcript; MANUAL: re-prefilled once, the next reply continues | PASS | PASS (run 2) |
| Parent death | `parent` | child gone ≤ 1 s after the supervisor is killed | PASS (39 ms) | PASS (83 ms) |
| 20-kill loop | `kill20` | 20 death and 20 ready lines, restart 1..20, one ready line each; every `recovery_ms − backoff_ms` < 2000, every death line's `ready_ms` non-null and < 2000, every `exit` null (`ps1`: −1 or null) | PASS | FAIL run 1 (cold start), PASS run 2 |
| No surviving child processes | `kill20`, `resume`, `parent` | no `larch` process or process holding the run's stdin/stdout; no zombie under the live supervisor (`kill20`, `resume`) | PASS | PASS |
| Released VRAM | `kill20`, `parent` | `memory.used` within 64 MiB of baseline after 2 s; no larch pid in compute apps (not observable in WSL) | PASS (+2 MiB; compute apps observed) | PASS (≤ +16 MiB; compute apps observed) |

NOT RUN rows need GPU time only; BLOCKED rows need #66, and a real GPU hang then needs a maintainer watching.

## Where results go

- `bench/results/recovery-wsl.md` and `bench/results/recovery-windows.md`: the script's PASS/FAIL/BLOCKED
  lines, VRAM numbers, `recovery_ms`, `backoff_ms` and `ready_ms` values and the caller-visible error text, per subcommand, with
  `HEAD_SHA`, driver version and the `nvidia-smi` baseline.
- `bench/results/tdr.md`: the Windows `fault hang` run (and the WSL one), with what the screen did.
- Rows R4, R5 and R10 of `bench/results/reliability.md`: one line each pointing at the files above.
- Raw logs (`/tmp/recovery-*.log`, the `raw logs:` directory, the clone's `recovery-*.log`) stay out of git.
