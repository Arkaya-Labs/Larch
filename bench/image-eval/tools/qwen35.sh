#!/bin/bash
# Try Qwen 3.5 image input from branch wip/qwen35-vision (worktree ~/larch-qwen35).
#   qwen35.sh ask <image> "<question>" [2b|0.8b] [max tokens]   one question about one image, on the GPU
#   qwen35.sh app      the native desktop app (GPUI, PR #474) with both Qwen 3.5 models (builds it the
#                      first time, 20-40 min)
#   qwen35.sh build    only (re)build the desktop app
#   qwen35.sh demo     the app's sample data with no engine (no GPU)
#   qwen35.sh test     the Qwen 3.5 CPU tests (no GPU, no model files, ~1 min)
#   qwen35.sh test all the engine, model, facade and app suites (~780 tests)
#   qwen35.sh gate     the full check against llama.cpp on the GPU: tools/q35-gpu.sh build dumps gate accuracy clean
set -u
W=$HOME/larch-qwen35; B=$HOME/larch-image-paste-bench; M=$HOME/larch-models
RUN=${RUN:-$(dirname "$0")/logrun.sh}; LOGS=$B/results
cuda_env() {
  export PATH=/usr/local/cuda/bin:$PATH LARCH_BACKEND=CUDA LARCH_CUDA_ARCH=86 LARCH_CMAKE_GENERATOR=Ninja \
         CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=$HOME/larch-opt-cuda CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) CUDAHOSTCXX=/usr/bin/g++-14
}
# gpui's X11 client needs libxkbcommon-x11 and libxcb-xkb; without sudo they are unpacked in
# ~/.local/xkb-libs. Its Wayland client panics under WSLg, so the app runs on X11.
XKB=$HOME/.local/xkb-libs/root/usr/lib/x86_64-linux-gnu
xkb_env() { export PKG_CONFIG_PATH=$XKB/pkgconfig LIBRARY_PATH=$XKB; }
run_env() {
  export LD_LIBRARY_PATH=$XKB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; unset WAYLAND_DISPLAY
  # WSLg's X root has no cursor and gpui never sets the plain arrow: give the root one.
  [ -n "${WSL_DISTRO_NAME:-}" ] && python3 -I $B/tools/x-root-cursor.py 2>/dev/null || true
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
model() { case ${1:-2b} in 2b|2B) echo m15 ;; 0.8b|0.8B) echo m14 ;; *) echo "pick 2b or 0.8b" >&2; exit 2 ;; esac; }
summary() { grep -E '^test result' "$1" | awk '{p+=$4; f+=$6} END {print p " passed, " f " failed"}'; grep -E -- '--- FAILED|^error' "$1" | head; }

cmd=${1:-}; shift || true
case $cmd in
ask)
  [ $# -ge 2 ] || { sed -n 3p "$0"; exit 2; }
  m=$(model "${3:-2b}"); used=$(gpu_used)
  [ "${used:-9999}" -lt 300 ] || { echo "The GPU is in use (${used} MiB), likely by another session. Try again later." >&2; exit 1; }
  exec flock -w 600 $HOME/src/gpu.lock $HOME/larch-opt-cuda/release/examples/image_ask \
    $M/$m.gguf $M/mmproj-$m.gguf "$1" "$2" 0 "${4:-300}" ;;
build|app)
  APP=$HOME/larch-opt-cuda/release/larch-desktop
  if [ ! -x "$APP" ] || [ $cmd = build ]; then
    echo "Building the desktop app (20-40 min the first time). Log: $LOGS/qwen35-app-build.log"
    (cuda_env; xkb_env; cd $W/apps/desktop && CAP_HIGH=10G CAP_MAX=11G \
      $RUN $LOGS/qwen35-app-build.log cargo build -j $(nproc) --release) \
      || { echo "The build failed; see the log." >&2; exit 1; }
  fi
  [ $cmd = build ] && { echo "Built: $APP"; exit 0; }
  # Its own data folder: the everyday (gait) app can't read chats with images yet. Hard links, no extra disk.
  T=$HOME/.local/share/larch-images-trial; D=$T/larch/models; mkdir -p $D
  for p in m14.gguf:Qwen3.5-0.8B-Q8_0.gguf mmproj-m14.gguf:mmproj-Qwen3.5-0.8B-F16.gguf \
           m15.gguf:Qwen3.5-2B-Q4_K_M.gguf mmproj-m15.gguf:mmproj-Qwen3.5-2B-F16.gguf; do
    [ -e "$D/${p#*:}" ] || ln "$M/${p%%:*}" "$D/${p#*:}"
  done
  used=$(gpu_used); [ "${used:-9999}" -lt 300 ] || echo "Note: the GPU already has ${used} MiB in use; the 2B may not fit." >&2
  run_env; XDG_DATA_HOME=$T exec "$APP" ;;
demo)
  run_env; exec $HOME/larch-opt-cuda/release/larch-desktop --demo ;;
test)
  export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:) LARCH_BACKEND=CPU \
         CARGO_TARGET_DIR=$HOME/larch-opt-cpu CAP_HIGH=10G CAP_MAX=11G
  cd $W || exit 1; L=$LOGS/qwen35-tests.log; : > $L
  if [ "${1:-}" = all ]; then
    $RUN $L cargo test -j $(nproc) --release --no-fail-fast -p larch-engine -p larch-model -p larch -p larch-app
  else
    $RUN $L cargo test -j $(nproc) --release --no-fail-fast -p larch-model --test qwen35 --test qwen_vision
    $RUN $L cargo test -j $(nproc) --release --no-fail-fast -p larch-engine --test mrope_images --test images
  fi
  summary $L; echo "Log: $L" ;;
gate)
  exec $B/tools/q35-gpu.sh build dumps gate accuracy clean ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
