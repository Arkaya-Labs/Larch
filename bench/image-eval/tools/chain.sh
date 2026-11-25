#!/bin/bash
# Runs queue files one after another through gpu-queue.sh (each waits for the previous to finish).
cd ~/larch-vision-run/eval-v2
while pgrep -f "gpu-queue.sh queue-baseline" >/dev/null; do sleep 60; done
for q in "$@"; do CAP_HIGH=4G CAP_MAX=5G ./gpu-queue.sh "$q" runs/logs; done
