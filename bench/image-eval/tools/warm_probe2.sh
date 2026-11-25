#!/bin/bash
# First-image timings, three loads each, with the old warm-up (the pre-switch build, a 1-row blank)
# and the new one (a 256-row blank), no warm turn from the harness. Both models.
P='{"thinking":false,"temperature":0,"ctx":8192,"max":{"qa":384,"absent":384}}'
for m in m14 m15; do
  for b in old:$HOME/larch-imgref-cuda new:$HOME/larch-imgopt-cuda; do
    for r in 1 2 3; do
      ${b#*:}/release/examples/image_eval $HOME/larch-models/$m.gguf $HOME/larch-models/mmproj-$m.gguf \
        $HOME/larch-vision-run/eval-v2/items.jsonl $HOME/larch-vision-run/eval-v2/runs/warm2-$m-${b%%:*}-$r.jsonl "$P" \
        $HOME/larch-vision-run/eval-v2/first5.txt
    done
  done
done
