#!/bin/sh
# The tokenizer's merge gate (architecture 15): llama.cpp's tokens and chat renders for the four
# requirement-R4 GGUFs, its tool renders for m1 and m2, and m2's renders with `enable_thinking` true
# and false, then larch against them. m7 is pinned
# but not rendered: its template has no tool support, so llama.cpp raises on the tool conversation and
# gguf_templates_render_tools checks that larch's render fails too. <dir> holds them as <id>.gguf, by
# the opaque fixture IDs below (MODELS in crates/larch-model/tests/tokenizer.rs; links work); the
# ID-to-file mapping stays outside the repo and models.sha256 pins their bytes. That file also pins
# the oracle's GGUFs (oracle-dump.sh refuses any other), which this keeps. Commit what it writes to
# tests/data; the tool and thinking renders stay in <dir>, where gguf_templates_render_tools and
# gguf_template_reads_thinking read them:
#   tools/llama-logits/model-dumps.sh <dir>
set -eu
# Absolute, because cargo runs the test from the crate's directory.
M=$(cd "${1:?usage: model-dumps.sh <dir of the GGUFs>}" && pwd)
cd "$(dirname "$0")/../.."
D=crates/larch-model/tests/data
B=target/llama-tools
cmake -S tools/llama-logits -B $B -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
# A bare -j lets make run unlimited jobs; one per core is enough.
cmake --build $B --target tokdump -j "$(getconf _NPROCESSORS_ONLN)"
if command -v sha256sum > /dev/null; then SHA=sha256sum; else SHA="shasum -a 256"; fi
# Only its own five pins are rewritten: the others are the oracle's, which it cannot rehash. The new
# list replaces models.sha256 only once every pin is written, so a failure leaves it as it was.
: > $B/models.sha256
for n in m1 m2 m3 m4; do
    (cd "$M" && $SHA "$n.gguf") >> $B/models.sha256
    $B/tokdump tokens "$M/$n.gguf" $D/prompts.jsonl > $D/$n.tokens.jsonl
    $B/tokdump chat "$M/$n.gguf" - $D/chat.json "26 Jul 2024" > $D/$n.chat.jsonl
done
(cd "$M" && $SHA m7.gguf) >> $B/models.sha256
for n in m1 m2; do
    $B/tokdump chat "$M/$n.gguf" - $D/chat-tools.json "26 Jul 2024" > "$M/$n.tools.jsonl"
done
$B/tokdump chat "$M/m2.gguf" - $D/chat.json "26 Jul 2024" true > "$M/m2.thinking.chat.jsonl"
$B/tokdump chat "$M/m2.gguf" - $D/chat.json "26 Jul 2024" false >> "$M/m2.thinking.chat.jsonl"
$B/tokdump chat "$M/m2.gguf" - $D/chat-tools.json "26 Jul 2024" true > "$M/m2.thinking.tools.jsonl"
$B/tokdump chat "$M/m2.gguf" - $D/chat-tools.json "26 Jul 2024" false >> "$M/m2.thinking.tools.jsonl"
grep -v ' m[1-47]\.gguf$' $D/models.sha256 >> $B/models.sha256 || [ $? -eq 1 ]
mv $B/models.sha256 $D/models.sha256
LARCH_MODELS=$M cargo test -p larch-model --test tokenizer -- --ignored models
LARCH_MODELS=$M cargo test -p larch-model --test tokenizer -- --ignored gguf_templates_render_tools
LARCH_MODELS=$M cargo test -p larch-model --test tokenizer -- --ignored gguf_template_reads_thinking
