#!/usr/bin/env python3
"""Simulates a stream-level loop stop on finished image_eval runs.

  python3 -I loopstop_sim.py items.jsonl run.jsonl [n=8] [repeats=3]

The stop fires at the first point where some n-word sequence has appeared `repeats` times, or a line
of 8+ characters has appeared `repeats` times (the third copy). The answer is cut there (keeping
the text up to the end of the second copy). Reports: how many answers would be cut, how many of
those the scorer called loops, how many non-loop answers would be cut (false stops), the decode
time saved (cut fraction x completion tokens / tok/s), and the score change for transcriptions
(CER on the cut text).
"""
import json, sys, importlib.util, os
spec = importlib.util.spec_from_file_location("score_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_eval.py"))
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)

def cut_point(text, n, reps):
    words = []
    pos = 0
    for w in text.split():
        i = text.find(w, pos); words.append((w, i + len(w))); pos = i + len(w)
    seen = {}
    for k in range(len(words) - n + 1):
        g = " ".join(w for w, _ in words[k:k + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return words[k][1] - len(words[k][0])  # start of the third copy
    lines = {}
    off = 0
    for ln in text.splitlines(True):
        s = ln.strip()
        if len(s) >= 8:
            lines[s] = lines.get(s, 0) + 1
            if lines[s] >= reps:
                return off
        off += len(ln)
    return None

def main():
    items = {}
    for l in open(sys.argv[1], encoding="utf-8"):
        if l.strip():
            it = json.loads(l); items[it["id"]] = it
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    run = se.load(sys.argv[2])
    cut = loops_cut = false = 0
    saved_ms = 0.0
    total_ms = 0.0
    dcer = []
    for i, r in run.items():
        it = items.get(i)
        if not it:
            continue
        text = se.clean(r.get("text", ""))
        g = se.grade(it, r)
        ms = (r.get("completion_tokens") or 0) / max(r.get("tok_s") or 1, 1e-6) * 1000
        total_ms += ms
        c = cut_point(text, n, reps)
        if c is None:
            continue
        cut += 1
        if g["loop"]:
            loops_cut += 1
        else:
            false += 1
        frac = 1 - c / max(len(text), 1)
        saved_ms += frac * ms
        if it["kind"] == "transcribe":
            dcer.append((i, g.get("cer"), se.cer(text[:c], it.get("text", ""))))
    nloops = sum(se.grade(items[i], r)["loop"] for i, r in run.items() if i in items)
    print(f"n={n} repeats={reps}: cut {cut} answers ({loops_cut} of {nloops} scored loops, {false} non-loops); "
          f"decode time saved {saved_ms/1000:.1f} s of {total_ms/1000:.1f} s")
    for i, a, b in dcer:
        print(f"  transcribe {i}: CER {a:.3f} -> {b:.3f}")

if __name__ == "__main__":
    main()
