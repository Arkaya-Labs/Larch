#!/bin/bash
# Runs GPU jobs from a queue file one at a time, sharing the GPU fairly with another user of the
# GPU: each job holds ~/src/gpu.lock for its whole run (each job is
# kept under ~40 min), starts only with under 300 MiB used and no compute process, and after
# each job it waits for the other side to start a run (or 25 min) before queueing for the lock
# again, so turns alternate.
#   gpu-queue.sh <queue file> <log dir>
# Each queue line: <name> <command...>. Finished names go to <queue>.done; a restart skips them.
Q=$1; L=$2; mkdir -p "$L"; DONE="$Q.done"; touch "$DONE"
RUN=${RUN:-$(dirname "$0")/logrun.sh}
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
log() { echo "$(date '+%F %T') $*" | tee -a "$L/queue.log"; }
first=${FIRST_FREE:-0}  # 1 only when no turn of ours just ended
while read -r name cmd; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  grep -qx "$name" "$DONE" && continue
  # A turn holds the lock for several jobs while they fit in TURN_S (default 40 min, the most the
  # two sessions agreed); a job starts in the same turn only with at least 12 min of it left.
  if [ "${held:-0}" = 1 ]; then
    if [ $(( $(date +%s) - turn_start )) -lt $(( ${TURN_S:-2400} - 720 )) ]; then
      log "start $name (same turn)"
      $RUN "$L/$name.log" bash -c "$cmd" < /dev/null
      rc=$?
      log "end $name rc=$rc"
      [ $rc = 0 ] && echo "$name" >> "$DONE"
      continue
    fi
    flock -u 9; exec 9>&-; held=0; log "turn over"
    first=0
  fi
  if [ $first = 0 ]; then
    # Fair turn: give the other session a chance to start its next run (FAIR_WAIT_MIN, default 25 min).
    for i in $(seq 1 ${FAIR_WAIT_MIN:-25}); do
      [ "$(used)" -gt 300 ] && { log "other side started; queueing behind it"; break; }
      sleep 60
    done
  fi
  first=0
  while :; do
    exec 9>~/src/gpu.lock
    flock 9
    if [ "$(used)" -lt 300 ] && [ "$(procs)" = 0 ]; then break; fi
    flock -u 9; exec 9>&-
    log "GPU busy ($(used) MiB) though the lock was free; checking again in 20 min"
    sleep 1200
  done
  held=1; turn_start=$(date +%s)
  log "start $name"
  $RUN "$L/$name.log" bash -c "$cmd" < /dev/null
  rc=$?
  log "end $name rc=$rc"
  [ $rc = 0 ] && echo "$name" >> "$DONE"
  python3 -I ~/larch-image-paste-bench/tools/evict.py ~/larch-models >/dev/null 2>&1
done < "$Q"
[ "${held:-0}" = 1 ] && { flock -u 9; exec 9>&-; }
log "queue finished"
