#!/bin/bash
# Times the Qwen encoder under each experiment switch and dumps its rows (one GPU job, ~2 min).
X=$HOME/larch-imgopt-cuda/release/examples/diag_encode; M=$HOME/larch-models
O=~/larch-vision-run/eval-v2/prof/ab2; mkdir -p $O
IMGS="$HOME/larch-image-paste-bench/images/terminal.png $HOME/larch-vision-run/gate35b/wide.png"
for cfg in base lt nolo lt+nolo; do
  envs=""
  case $cfg in *lt*) envs="$envs LARCH_EXP_LT_BIAS=1";; esac
  case $cfg in *nolo*) envs="$envs LARCH_EXP_ATTN_NOLO=1";; esac
  case $cfg in *g16*) envs="$envs LARCH_EXP_GEMM16=1";; esac
  case $cfg in *bias*) envs="$envs LARCH_EXP_BIAS_INPLACE=1";; esac
  for mm in m15:2048 m14:1024; do
    m=${mm%%:*}; d=${mm#*:}
    for img in $IMGS; do
      n=$(basename $img .png)
      echo "== $cfg $m $n: $(env $envs DIAG_DUMP=$O/$cfg-$m-$n.f32 $X $M/mmproj-$m.gguf $d $img 0 7)"
    done
  done
done
