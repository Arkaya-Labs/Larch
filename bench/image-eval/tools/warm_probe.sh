#!/bin/bash
# First-item timings with no warm-up, a text turn first, and an image turn first (both models).
for m in m14 m15; do
  for w in none text image; do
    case $w in none) P='{"thinking":false,"temperature":0,"ctx":8192,"max":{"qa":384,"absent":384}}';;
      text) P='{"thinking":false,"temperature":0,"ctx":8192,"max":{"qa":384,"absent":384},"warm":"text"}';;
      image) P='{"thinking":false,"temperature":0,"ctx":8192,"max":{"qa":384,"absent":384},"warm":"image:'"$HOME"'/larch-image-paste-bench/images/terminal.png"}';; esac
    $HOME/larch-imgopt-cuda/release/examples/image_eval $HOME/larch-models/$m.gguf $HOME/larch-models/mmproj-$m.gguf $HOME/larch-vision-run/eval-v2/items.jsonl $HOME/larch-vision-run/eval-v2/runs/warm-$m-$w.jsonl "$P" $HOME/larch-vision-run/eval-v2/first5.txt
  done
done
