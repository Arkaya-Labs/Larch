#!/usr/bin/env python3
"""Loop stop by periodicity: stop when the stream's tail is one unit of 1-64 characters repeated so
that the repeated span is at least `span` characters (and at least `minrep` copies). Simulated on the
finished text: the first prefix length at which the rule fires, checked every 16 characters.
  python3 -I periodic_sim.py items.jsonl run.jsonl [span=160] [minrep=6]"""
import json, sys, importlib.util, os
spec = importlib.util.spec_from_file_location("se", os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_eval.py"))
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)

def fires(t, span, minrep):
    for p in range(1, 65):
        reps = max(minrep, -(-span // p))
        if len(t) < p * reps:
            continue
        unit = t[-p:]
        if t[-p * reps:] == unit * reps:
            return True
    return False

def first_fire(text, span, minrep):
    for n in range(16, len(text) + 1, 16):
        if fires(text[:n], span, minrep):
            return n
    return None

items = {}
for l in open(sys.argv[1], encoding="utf-8"):
    if l.strip():
        it = json.loads(l); items[it["id"]] = it
span = int(sys.argv[3]) if len(sys.argv) > 3 else 160
minrep = int(sys.argv[4]) if len(sys.argv) > 4 else 6
run = se.load(sys.argv[2])
hits = []; saved = 0.0
for i, r in run.items():
    if i not in items: continue
    t = se.clean(r.get("text", ""))
    c = first_fire(t, span, minrep)
    if c is None: continue
    ms = (r.get("completion_tokens") or 0) / max(r.get("tok_s") or 1, 1e-6) * 1000
    saved += (1 - c / len(t)) * ms
    g = se.grade(items[i], r)
    extra = ""
    if items[i]["kind"] == "transcribe":
        extra = f" CER {g['cer']:.3f} -> {se.cer(t[:c], items[i].get('text','')):.3f}"
    hits.append(f"  {i} ({items[i]['kind']}) cut at {c} of {len(t)} chars{extra}; tail {t[max(0,c-40):c]!r}")
print(f"span={span} minrep={minrep}: fires on {len(hits)} answers, decode saved {saved/1000:.1f} s")
print("\n".join(hits))
