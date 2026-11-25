#!/bin/bash
# logrun.sh <log> <command...>: runs the command with its output appended to <log>. The default
# $RUN of gpu-queue.sh and qwen35.sh; set RUN to your own wrapper (one that caps memory, say) with
# the same arguments to use that instead.
log=$1; shift
"$@" >>"$log" 2>&1
