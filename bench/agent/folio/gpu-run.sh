#!/usr/bin/env bash
# The Folio GPU run: 32k, 64k and 128k of real documents on the large hybrid model, one table at the end.
# Run from anywhere in WSL or on macOS: bash gpu-run.sh. ALIAS picks the model in MODELS (default ~/models),
# REPO is the larch clone (default ~/src/larch), LARCH_BACKEND the build (default METAL on macOS, else CUDA).
# REMOTE is the remote it fetches from (default origin) and REF the commit it builds (default $REMOTE/agent/folio; any
# commit the clone has, such as a local merge, also works).
set -u -o pipefail
DATE=$(date +%F)
OUT=$HOME/folio-gpu-results/$DATE
mkdir -p "$OUT" "$HOME/folio-docs/filler" "$HOME/src" || exit 1
MODELS=${MODELS:-$HOME/models}
REPO=${REPO:-$HOME/src/larch}
BACKEND=${LARCH_BACKEND:-$([ "$(uname)" = Darwin ] && echo METAL || echo CUDA)}
REMOTE=${REMOTE:-origin}
REF=${REF:-$REMOTE/agent/folio}
# One GPU job at a time where flock exists (Linux); macOS has none, so run nothing else meanwhile.
locked() { if command -v flock >/dev/null; then flock "$HOME/src/gpu.lock" "$@"; else "$@"; fi; }

# --- model: ALIAS or the largest gguf ---
ALIAS=${ALIAS:-$(basename "$(ls -S "$MODELS"/*.gguf | head -1)" .gguf)}
MODEL=$MODELS/$ALIAS.gguf
[ -f "$MODEL" ] || { echo "no model file $MODEL"; exit 1; }
echo "model alias: $ALIAS, backend $BACKEND, $(uname -sm)" | tee "$OUT/alias.txt"

# --- code: a fresh worktree of REF ---
cd "$REPO" || exit 1
# A REF on REMOTE is fetched, so it is never a stale tracking ref.
case $REF in "$REMOTE"/*) git fetch "$REMOTE" "${REF#"$REMOTE"/}" || exit 1 ;; esac
if [ -e "$HOME/src/wt/folio" ]; then git worktree remove --force "$HOME/src/wt/folio" || exit 1; fi
git worktree add --detach "$HOME/src/wt/folio" "$REF" || exit 1
git -C "$HOME/src/wt/folio" rev-parse HEAD | tee "$OUT/commit.txt"

# --- build (release, on the device's GPU backend) ---
cd "$HOME/src/wt/folio" || exit 1
CARGO_TARGET_DIR=$HOME/src/target/folio LARCH_BACKEND=$BACKEND \
  nice -n 19 cargo build --release --locked -p larch-agent </dev/null || { echo "build failed"; exit 1; }
BIN=$HOME/src/target/folio/release/larch-agent
Q=$HOME/src/wt/folio/bench/agent/folio/questions.jsonl

# --- documents: the docfollow bench's fetcher, from the tree being built, in a temp dir ---
T=$(mktemp -d)
git -C "$HOME/src/wt/folio" archive HEAD bench/agent/docfollow | tar -x -C "$T" || exit 1
python3 -I "$T/bench/agent/docfollow/run.py" fetch </dev/null | tee "$OUT/fetch.txt"
# lm317, nist600 and nvda10k are reported missing or blocked there: they are not used.
# wchlink is AES-256 (R6) encrypted, which the agent's PDF reader refuses: it is not used either.
C=$T/bench/agent/docfollow/data/cache
P=$OUT/pool; rm -rf "$P"; mkdir -p "$P"
# Name order is letter order; the last name (zz-v32.pdf) is the document the questions ask about.
n=0
for f in kimi.pdf r1.pdf ocr.pdf math2.pdf ghtos.md mcptasks.md tsc.md nda.md v32.pdf; do
  [ -f "$C/$f" ] || { echo "missing $f (see fetch.txt)"; exit 1; }
done
cp "$C/kimi.pdf" "$P/01-kimi.pdf"; cp "$C/r1.pdf" "$P/03-r1.pdf"; cp "$C/ocr.pdf" "$P/04-ocr.pdf"
cp "$C/math2.pdf" "$P/05-math2.pdf"; cp "$C/ghtos.md" "$P/06-ghtos.md"; cp "$C/mcptasks.md" "$P/07-mcptasks.md"
cp "$C/tsc.md" "$P/09-tsc.md"; cp "$C/nda.md" "$P/10-nda.md"
cp "$C/v32.pdf" "$P/zz-v32.pdf"
for f in "$HOME"/folio-docs/filler/*; do [ -f "$f" ] && cp "$f" "$P/02-$(basename "$f")"; done
ls -l "$P" | tee "$OUT/pool.txt"

# --- one GPU-timed step, run under the lock ---
cat > "$OUT/step.sh" <<'STEP'
#!/usr/bin/env bash
# usage: step.sh <context> [--plain]; needs BIN MODEL Q P OUT in the environment
# --plain is the baseline: the same documents as plain text, no pin, the same memory plan.
# KEEP_DATA=1 keeps the data dir (its chat and .pin) and writes its path to <step>.datadir, for decode-ab.sh.
C=$1
MODE=${2:-}
R=$OUT/ctx-$C${MODE:+-plain}
D=$(mktemp -d); mkdir -p "$D/agent"
printf 'thinking_budget = 0\n' > "$D/agent/config.toml"   # fresh data dir: no kept pin to import
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
if [ -x "$SMI" ]; then
  "$SMI" --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$R.vram" 2>/dev/null &
  SAMPLER=$!
fi
"$BIN" folio-bench -m "$MODEL" --data-dir "$D" --questions "$Q" --context "$C" --json "$R.json" $MODE \
  --docs "$P"/* > "$R.stdout" 2> "$R.stderr"
echo $? > "$R.exit"
[ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null
if [ "${KEEP_DATA:-0}" = 1 ]; then echo "$D" > "$R.datadir"; else rm -rf "$D"; fi
STEP
export BIN MODEL Q P OUT
# CONTEXTS="131072" PLAIN=0 reruns one context pinned only; the table then lists the rest as missing.
for CTX in ${CONTEXTS:-32768 65536 131072}; do
  echo "=== context $CTX ($(date +%T)) ==="
  locked bash "$OUT/step.sh" "$CTX" </dev/null
  echo "exit $(cat "$OUT/ctx-$CTX.exit" 2>/dev/null)"
  [ "${PLAIN:-1}" = 0 ] && continue
  locked bash "$OUT/step.sh" "$CTX" --plain </dev/null
  echo "plain exit $(cat "$OUT/ctx-$CTX-plain.exit" 2>/dev/null)"
done

# --- the table ---
cat > "$OUT/table.py" <<'PY'
import json, os, sys
out = sys.argv[1]
rows, fails, notes = [], [], []
def f(x, d=1): return "-" if x is None else f"{x:.{d}f}"
for ctx in (32768, 65536, 131072):
    base = os.path.join(out, f"ctx-{ctx}")
    code = open(base + ".exit").read().strip() if os.path.exists(base + ".exit") else "none"
    cold = sum("larch_pin_cold" in l for l in open(base + ".stderr", errors="replace")) if os.path.exists(base + ".stderr") else -1
    # larch_pin_stall,<op>,<ms>: a pin copy that held the engine thread past 50 ms.
    stalls = [int(l.rsplit(",", 1)[1]) for l in open(base + ".stderr", errors="replace")
              if l.startswith("larch_pin_stall,") and l.strip().rsplit(",", 1)[1].isdigit()] if os.path.exists(base + ".stderr") else []
    vram = "-"
    if os.path.exists(base + ".vram"):
        v = [int(x) for x in open(base + ".vram").read().split() if x.isdigit()]
        vram = f"{max(v)/1024:.1f}" if v else "-"
    if code != "0" or not os.path.exists(base + ".json"):
        rows.append((ctx, "FAILED exit " + code)); fails.append(f"{ctx}: run failed (see {base}.stderr)"); continue
    r = json.load(open(base + ".json")); s = r["summary"]
    d0, d1 = r["decode_default"]["median"], r["decode_documents"]["median"]
    note = "" if r["plan_context"] == ctx else f" UNDERFILLED(plan {r['plan_context']})"
    rows.append((ctx, f"{len(r['documents'])} docs {r['tokens']} tok{note}", r["fits"], r.get("seats", "-"), r["host_layers"],
                 f(r["first_read_ms"] and r["first_read_ms"]/1000), f"{s['quote_ok']}/{s['asked']}", f"{s['value_ok']}/{s['asked']}",
                 f(s["later_ttft_max_ms"] and s["later_ttft_max_ms"]/1000, 2), f(s["later_first_text_max_ms"] and s["later_first_text_max_ms"]/1000, 2),
                 f"{d0:.1f}/{d1:.1f}", cold, f"{len(stalls)}/{max(stalls, default=0)}", vram))
    slow = max(r["questions"][1:], key=lambda a: a["ttft_ms"], default=None)
    if slow: notes.append(f"{ctx}: slowest later question {slow['id']} {slow['ttft_ms']/1000:.2f} s, calls {slow.get('calls', [])}, cached {slow['cached_tokens']}")
    r["_q"], r["_cold"], r["_ctx"] = s["quote_ok"], cold, ctx
    if s["later_ttft_max_ms"] is None or s["later_ttft_max_ms"] >= 3000: fails.append(f"{ctx}: second answer first token {f(s['later_ttft_max_ms'])} ms, not under 3000")
    if cold != 0: fails.append(f"{ctx}: {cold} cold pins")
    rows[-1] += (r,)
hdr = ("context", "documents", "fits", "seats", "host_layers", "read_s", "quote_ok", "value_ok", "2nd+ ttft_s", "2nd+ 1st text_s", "decode tok/s default/docs", "cold", "stalls n/max ms", "peak VRAM GiB")
lines = ["  ".join(hdr)]
good = {}
for row in rows:
    if len(row) == 2: lines.append(f"{row[0]}  {row[1]}"); continue
    lines.append("  ".join(str(x) for x in row[:-1])); good[row[0]] = row[-1]
if 32768 in good and 131072 in good and good[131072]["_q"] < good[32768]["_q"] - 1:
    fails.append(f"last-page accuracy at 128k ({good[131072]['_q']}) is more than one below 32k ({good[32768]['_q']})")
# The A/B is a measurement, not a rule: pinned against plain text on the same plan, per context.
def ab(r, key):
    a = r.get(key)
    return a and a["ttft_ms"] / 1000
def load(name):
    base = os.path.join(out, name)
    ok = os.path.exists(base + ".exit") and open(base + ".exit").read().strip() == "0" and os.path.exists(base + ".json")
    return json.load(open(base + ".json")) if ok else None
lines += ["", "pinned vs plain (same plan; seconds unless noted)",
          "  ".join(("context", "mode", "2nd+ ttft_s", "after switch ttft_s", "after switch total_s", "after restart open-to-answer_s", "restart load_s", "other chat tok/s"))]
for ctx in (32768, 65536, 131072):
    for mode, name in (("pinned", f"ctx-{ctx}"), ("plain", f"ctx-{ctx}-plain")):
        r = load(name)
        if r is None:
            lines.append(f"{ctx}  {mode}  FAILED or missing (see {name}.stderr)"); continue
        sw = r.get("after_switch")
        lines.append("  ".join(str(x) for x in (ctx, mode,
            f(r["summary"]["later_ttft_max_ms"] and r["summary"]["later_ttft_max_ms"]/1000, 2),
            f(ab(r, "after_switch"), 2), f(sw and sw["total_ms"]/1000, 2),
            f(r.get("restart_open_ms") and r["restart_open_ms"]/1000, 2),
            f(r.get("restart_load_ms") and r["restart_load_ms"]/1000, 2),
            f(r.get("other_chat_tok_s"), 1))))
# The restart's open, step by step (bench restart_stamps), and the engine's import phases (larch_pin_import, from
# f4/pin-import on: an older engine prints none).
for ctx, r in sorted(good.items()):
    steps = ", ".join(f"{st['event'].split(' ')[0]} {st['ms']/1000:.2f}" for st in r.get("restart_stamps", []))
    if steps: notes.append(f"{ctx}: restart open steps (s): {steps}")
    base = os.path.join(out, f"ctx-{ctx}.stderr")
    phases = [l.strip().split(",", 1)[1] for l in open(base, errors="replace") if l.startswith("larch_pin_import,")]
    if phases: notes.append(f"{ctx}: pin import ms: {'; '.join(phases)}")
lines += [""] + notes
lines.append("")
lines.append("PASS" if not fails and len(good) == 3 else "FAIL: " + "; ".join(fails or ["a context did not run"]))
text = "\n".join(lines)
open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
print(text)
PY
python3 -I "$OUT/table.py" "$OUT"
echo "results: $OUT"
