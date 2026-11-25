#!/usr/bin/env python3
"""Scores image_eval runs against items.jsonl.

  python3 -I score_eval.py items.jsonl run.jsonl                 # one run: tables by kind/category/source
  python3 -I score_eval.py items.jsonl base.jsonl new.jsonl      # paired comparison of two runs

Metrics: qa accuracy (an answer contained in the normalized output, or ANLS >= 0.5 for docvqa),
absent accuracy (says "not shown"/"no" and nothing else assertive), transcription CER, loop rate
(an 8-word n-gram 4+ times, a line repeated 3+ times, or the token limit hit on describe/transcribe),
latency (ttft ms on first visits, tok/s).
"""
import json, re, sys, math, unicodedata
from collections import defaultdict

def load(p):
    rows = {}
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("meta"):
            continue
        rows[r["id"]] = r
    return rows

THINK = re.compile(r"<think>.*?</think>", re.S)

def clean(t):
    return THINK.sub("", t or "").strip()

def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[\s]+", " ", t)
    return re.sub(r"[^\w%$.,:/@#+\- ]", "", t).strip(" .")

def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def cer(pred, gold):
    p = re.sub(r"\s+", " ", pred).strip()
    g = re.sub(r"\s+", " ", gold).strip()
    return lev(p, g) / max(1, len(g))

def anls(pred, answers):
    best = 0.0
    for a in answers:
        pa, aa = norm(pred), norm(a)
        if not aa:
            continue
        d = lev(pa, aa) / max(len(pa), len(aa), 1)
        best = max(best, 1 - d if d < 0.5 else 0.0)
    return best

def looped(text, hit_max, kind):
    w = text.split()
    grams = defaultdict(int)
    for i in range(len(w) - 7):
        grams[" ".join(w[i:i + 8])] += 1
    if grams and max(grams.values()) >= 4:
        return True
    lines = defaultdict(int)
    for ln in text.splitlines():
        ln = ln.strip()
        if len(ln) > 8:
            lines[ln] += 1
    if lines and max(lines.values()) >= 3 and kind == "describe":
        return True
    return bool(hit_max) and kind in ("describe", "transcribe")

NEG = ("not shown", "not visible", "no ", "no.", "there is no", "does not", "doesn't", "isn't", "not present", "not mentioned", "cannot find", "can't find", "not in the image", "none")

def grade(item, r):
    kind, text = item["kind"], clean(r.get("text", ""))
    g = {"kind": kind, "cat": item.get("category", "?"), "src": item.get("source", "?"),
         "err": bool(r.get("err")), "loop": looped(text, r.get("hit_max"), kind)}
    if kind == "qa":
        n = norm(text)
        ok = any(norm(a) and norm(a) in n for a in item.get("answers", []))
        if item.get("source") == "docvqa" and not ok:
            ok = anls(text, item.get("answers", [])) >= 0.5
        g["score"] = 1.0 if ok else 0.0
    elif kind == "absent":
        n = " " + norm(text) + " "
        g["score"] = 1.0 if any(k in n for k in NEG) else 0.0
    elif kind == "transcribe":
        g["cer"] = cer(text, item.get("text", ""))
        g["score"] = max(0.0, 1 - g["cer"])
    else:
        g["score"] = 0.0 if g["loop"] else 1.0
    g["ttft"] = r.get("ttft_ms") if r.get("first_visit") else None
    g["tok_s"] = r.get("tok_s")
    g["wall"] = r.get("wall_ms")
    return g

def table(items, run):
    gs = {i: grade(items[i], run[i]) for i in run if i in items}
    def agg(key):
        b = defaultdict(list)
        for g in gs.values():
            b[g[key]].append(g)
        return b
    out = []
    for key in ("kind", "cat", "src"):
        out.append(f"\n| {key} | n | score | CER | loops | errors |\n|---|---|---|---|---|---|")
        for k, v in sorted(agg(key).items()):
            cers = [g["cer"] for g in v if "cer" in g]
            out.append(f"| {k} | {len(v)} | {sum(g['score'] for g in v)/len(v):.3f} | "
                       f"{(sum(cers)/len(cers)) if cers else float('nan'):.3f} | {sum(g['loop'] for g in v)} | {sum(g['err'] for g in v)} |")
    tt = sorted(g["ttft"] for g in gs.values() if g["ttft"])
    ts = [g["tok_s"] for g in gs.values() if g["tok_s"]]
    if tt:
        out.append(f"\nTTFT first visit: median {tt[len(tt)//2]:.0f} ms, p90 {tt[int(len(tt)*0.9)]:.0f} ms (n={len(tt)}); "
                   f"decode median {sorted(ts)[len(ts)//2]:.1f} tok/s")
    allg = list(gs.values())
    out.insert(0, f"overall score {sum(g['score'] for g in allg)/max(1,len(allg)):.4f} over {len(allg)} items; "
                  f"loops {sum(g['loop'] for g in allg)}; errors {sum(g['err'] for g in allg)}")
    return "\n".join(out), gs

def compare(items, a, b):
    _, ga = table(items, a)
    _, gb = table(items, b)
    ids = sorted(set(ga) & set(gb))
    wins = sum(gb[i]["score"] > ga[i]["score"] + 1e-9 for i in ids)
    losses = sum(gb[i]["score"] < ga[i]["score"] - 1e-9 for i in ids)
    n = wins + losses
    # Two-sided sign test.
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n)
    d = sum(gb[i]["score"] - ga[i]["score"] for i in ids) / max(1, len(ids))
    lines = [f"paired over {len(ids)} items: mean score delta {d:+.4f}; better {wins}, worse {losses}, sign-test p={p:.3g}"]
    by = defaultdict(list)
    for i in ids:
        by[ga[i]["cat"]].append(gb[i]["score"] - ga[i]["score"])
    lines.append("| category | n | delta |\n|---|---|---|")
    for k, v in sorted(by.items()):
        lines.append(f"| {k} | {len(v)} | {sum(v)/len(v):+.3f} |")
    def med(g, key):
        v = sorted(x[key] for x in g.values() if x.get(key))
        return v[len(v)//2] if v else float("nan")
    lines.append(f"TTFT median {med(ga,'ttft'):.0f} -> {med(gb,'ttft'):.0f} ms; decode {med(ga,'tok_s'):.1f} -> {med(gb,'tok_s'):.1f} tok/s; "
                 f"loops {sum(g['loop'] for g in ga.values())} -> {sum(g['loop'] for g in gb.values())}")
    worse = [i for i in ids if gb[i]["score"] < ga[i]["score"] - 1e-9][:15]
    lines.append("worse ids: " + ", ".join(worse))
    return "\n".join(lines)

if __name__ == "__main__":
    items = {}
    for l in open(sys.argv[1], encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            items[it["id"]] = it
    if len(sys.argv) == 3:
        print(table(items, load(sys.argv[2]))[0])
    else:
        print(compare(items, load(sys.argv[2]), load(sys.argv[3])))
