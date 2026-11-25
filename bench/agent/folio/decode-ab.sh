#!/usr/bin/env bash
# Decode with the documents held, LARCH_DECODE_TC=1 against =0, alternated, one process a sample: each sample is
# `folio-bench --reopen` on the chat a KEEP_DATA=1 gpu-run.sh step kept (its pin imported, as the run's restart does),
# then the run's decode_documents measure. The switch is read at library load, hence a process each.
# usage: decode-ab.sh <gpu-run results dir> [pairs, default 9] [context, default 131072]
# Needs the dir's ctx-<context>.datadir and .json, and the binary and model gpu-run.sh used (BIN, MODEL override).
set -u -o pipefail
OUT=${1:?usage: decode-ab.sh <results dir> [pairs] [context]}
PAIRS=${2:-9}
C=${3:-131072}
BIN=${BIN:-$HOME/src/target/folio/release/larch-agent}
MODEL=${MODEL:-$HOME/models/$(sed -n 's/^model alias: \([^,]*\),.*/\1/p' "$OUT/alias.txt").gguf}
D=$(cat "$OUT/ctx-$C.datadir") || exit 1
CHAT=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chat"])' "$OUT/ctx-$C.json") || exit 1
Q=${Q:-$HOME/src/wt/folio/bench/agent/folio/questions.jsonl}
AB=$OUT/decode-ab; mkdir -p "$AB"
locked() { if command -v flock >/dev/null; then flock "$HOME/src/gpu.lock" "$@"; else "$@"; fi; }
sample() {  # <pair> <tc>; a failed sample leaves no JSON, so no earlier run's is read in its place
  rm -f "$AB/$1-tc$2.json"
  LARCH_DECODE_TC=$2 "$BIN" folio-bench -m "$MODEL" --data-dir "$D" --questions "$Q" --context "$C" \
    --reopen "$CHAT" --json "$AB/$1-tc$2.json" --docs "$OUT"/pool/* > "$AB/$1-tc$2.stdout" 2> "$AB/$1-tc$2.stderr"
  local code=$?
  [ "$code" = 0 ] || rm -f "$AB/$1-tc$2.json"
  echo "pair $1 tc=$2 exit $code"
}
export -f sample
export BIN MODEL C Q CHAT D AB OUT
# One lock for the whole A/B: no other GPU job between its samples.
locked bash -c "for i in \$(seq 1 $PAIRS); do sample \$i 1; sample \$i 0; done"
python3 -I - "$AB" "$PAIRS" <<'PY'
import json, os, statistics, sys
ab, pairs = sys.argv[1], int(sys.argv[2])
def med(i, tc):
    p = os.path.join(ab, f"{i}-tc{tc}.json")
    return json.load(open(p))["decode_documents"]["median"] if os.path.exists(p) else None
rows = [(i, med(i, 1), med(i, 0)) for i in range(1, pairs + 1)]
ok = [(a, b) for _, a, b in rows if a and b]
print("pair  tc=1 tok/s  tc=0 tok/s  ratio 1/0")
for i, a, b in rows:
    print(f"{i}  {a and round(a, 2)}  {b and round(b, 2)}  {a and b and round(a / b, 4)}")
if ok:
    one, zero = [a for a, _ in ok], [b for _, b in ok]
    ratios = [a / b for a, b in ok]
    print(f"pooled median tc=1 {statistics.median(one):.2f}, tc=0 {statistics.median(zero):.2f}; "
          f"median pair ratio {statistics.median(ratios):.4f} (min {min(ratios):.4f}, max {max(ratios):.4f}); "
          f"spread tc=1 {min(one):.1f}-{max(one):.1f}, tc=0 {min(zero):.1f}-{max(zero):.1f}; {len(ok)} pairs")
PY
