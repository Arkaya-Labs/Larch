#!/bin/bash
# First-image timings after a 3 s pause at load (a person pasting), with the old one-row warm-up
# (LARCH_EXP_WARM_SIDE=1) and the new 256-row one; three loads each, both models.
P='{"thinking":false,"temperature":0,"ctx":8192,"max":{"qa":384,"absent":384},"pause_ms":3000}'
B=$HOME/larch-imgopt-cuda/release/examples/image_eval
for m in m14 m15; do
  for w in old:1 new:16; do
    for r in 1 2 3; do
      LARCH_EXP_WARM_SIDE=${w#*:} $B $HOME/larch-models/$m.gguf $HOME/larch-models/mmproj-$m.gguf \
        $HOME/larch-vision-run/eval-v2/items.jsonl $HOME/larch-vision-run/eval-v2/runs/warm3-$m-${w%%:*}-$r.jsonl "$P" \
        $HOME/larch-vision-run/eval-v2/first5.txt
    done
  done
done
