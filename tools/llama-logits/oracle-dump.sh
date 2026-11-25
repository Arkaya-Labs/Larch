#!/bin/sh
# The model oracle's reference (architecture 15): llama-logits on one pinned GGUF, written outside git to
# ${LARCH_ORACLE:-~/larch-oracle}/<model sha256>/<label>.jsonl.zst. <label> names the run (cpu, cuda,
# metal); the dump's header records what actually ran. Arguments after <label> go to llama-logits
# (--first 100, --ngl 99 for a GPU backend); CMAKE_ARGS=-DGGML_CUDA=ON builds CUDA. The GGUF must be one
# models.sha256 pins. Calibrate with the example in larch-model, on files (it hashes each dump, then
# reads it, so a pipe would be empty by then; its dump_sha256 is of the decompressed file):
#   tools/llama-logits/oracle-dump.sh <gguf> <prompts.jsonl> cpu --first 100
#   tools/llama-logits/oracle-dump.sh <gguf> <prompts.jsonl> cuda --first 100 --ngl 99
#   zstd -dc <dir>/cpu.jsonl.zst > cpu.jsonl && zstd -dc <dir>/cuda.jsonl.zst > cuda.jsonl
#   cargo run --release -p larch-model --example oracle-calibrate -- <sha256> cpu.jsonl cuda.jsonl
set -eu
G=$(cd "$(dirname "${1:?usage: oracle-dump.sh <gguf> <prompts.jsonl> <label> [llama-logits args]}")" && pwd)/$(basename "$1")
P=$(cd "$(dirname "${2:?prompts}")" && pwd)/$(basename "$2")
L=${3:?label}
shift 3
O=${LARCH_ORACLE:-$HOME/larch-oracle}
mkdir -p "$O"
O=$(cd "$O" && pwd)
cd "$(dirname "$0")/../.."
if command -v sha256sum > /dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
S=$($SHA "$G" | cut -c1-64)
if ! grep -q "^$S " crates/larch-model/tests/data/models.sha256; then
    echo "$G ($S) is not a GGUF models.sha256 pins" >&2
    exit 1
fi
B=target/llama-tools-$L
cmake -S tools/llama-logits -B "$B" -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-}
cmake --build "$B" --target llama-logits -j "$(getconf _NPROCESSORS_ONLN)"
O=$O/$S
mkdir -p "$O"
# Through a file, so a failed run cannot leave a truncated .zst behind.
"$B/llama-logits" "$G" "$P" "$@" > "$O/$L.jsonl"
zstd -q -f --rm "$O/$L.jsonl"
echo "$O/$L.jsonl.zst"
