#!/usr/bin/env python3
"""Guess decisions from larch serve logs (LARCH_VERIFY_TIME=1), per arm and per prompt kind.

  steps.py ARM=SERVE_LOG,ANSWERS_JSONL ...

Each decode step prints larch_verify_time (rows, guessed, ...) and, when it could guess, larch_guesses
(cumulative rounds, tried, kept; this step's draft_us). A step's tokens are 1 + the guesses it kept.
Prompts ran one at a time in ANSWERS_JSONL's order, each emitting its usage tokens (the first from
the prompt's own step, which prints no verify line), so steps are cut into prompts by those counts.
Per kind: decode steps, share that checked guesses, guesses per check, kept per check, steps that
drafted but checked nothing (the draft's confidence trimmed every guess), and steps that could guess
but did not draft (the first pick, on acceptance alone, said no).
"""
import json
import re
import sys
from collections import defaultdict

KV = re.compile(r"(\w+)=(-?\d+)")


def steps_of(path):
    out, pending = [], None
    for line in open(path, errors="replace"):
        if line.startswith("larch_verify_time,"):
            if pending:
                out.append(pending)
            v = dict((k, int(x)) for k, x in KV.findall(line))
            pending = {"rows": v["rows"], "guessed": v["guessed"], "wall": v["wall_us"], "could": False,
                       "draft": 0, "kept": None}
        elif line.startswith("larch_guesses,") and pending:
            v = dict((k, int(x)) for k, x in KV.findall(line))
            pending.update(could=True, draft=v["draft_us"], cum_kept=v["kept"])
            out.append(pending)
            pending = None
    if pending:
        out.append(pending)
    last = 0
    for s in out:
        if "cum_kept" in s:
            s["kept"] = s["cum_kept"] - last
            last = s["cum_kept"]
        else:
            s["kept"] = 0
    return out


def main(args):
    speed = {}
    print("| arm | kind | decode steps | checked | guesses a check | kept a check | drafted, none checked | could guess, no draft | p50 wall plain / check ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for arg in args:
        arm, rest = arg.split("=", 1)
        log, answers = rest.split(",", 1)
        steps = steps_of(log)
        rows = [json.loads(l) for l in open(answers) if l.strip()]
        by = defaultdict(list)
        i = 0
        for r in rows:
            need, got = max((r["tokens"] or 1) - 1, 0), 0
            while got < need and i < len(steps):
                s = steps[i]
                i += 1
                if s["rows"] != 1 + s["guessed"]:
                    continue  # a step beside another request's rows (none expected at 1 chat)
                by[r["kind"]].append(s)
                got += 1 + s["kept"]
        by["all"] = [s for k in list(by) for s in by[k]]
        for kind in sorted(by):
            ss = by[kind]
            checks = [s for s in ss if s["guessed"] > 0]
            plain = sorted(s["wall"] for s in ss if s["guessed"] == 0)
            chk = sorted(s["wall"] for s in checks)
            med = lambda v: v[len(v) // 2] / 1000 if v else 0
            n = len(ss) or 1
            print(f"| {arm} | {kind} | {len(ss)} | {len(checks) / n:.2f} | "
                  f"{sum(s['guessed'] for s in checks) / max(len(checks), 1):.2f} | "
                  f"{sum(s['kept'] for s in checks) / max(len(checks), 1):.2f} | "
                  f"{sum(1 for s in ss if s['draft'] > 0 and s['guessed'] == 0)} | "
                  f"{sum(1 for s in ss if s['could'] and s['draft'] == 0 and s['guessed'] == 0)} | "
                  f"{med(plain):.1f} / {med(chk):.1f} |")
        print(f"| {arm} | (steps left uncut) | {len(steps) - i} | | | | | | |")
        speed[arm] = (by, rows)
    print()
    print("Step-implied speed: tokens over the sum of step wall and draft time, against the measured decode")
    print("| arm | kind | tokens | wall ms | draft ms | step tok/s (wall) | step tok/s (wall + draft) | measured decode tok/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for arm, (by, rows) in speed.items():
        for kind in sorted(by):
            ss = by[kind]
            tok = sum(1 + s["kept"] for s in ss)
            wall = sum(s["wall"] for s in ss) / 1000
            draft = sum(s["draft"] for s in ss) / 1000
            meas = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s") and (kind == "all" or r["kind"] == kind)]
            m = sum(meas) / len(meas) if meas else 0
            print(f"| {arm} | {kind} | {tok} | {wall:.0f} | {draft:.0f} | {tok / max(wall, 1e-9) * 1000:.1f} | "
                  f"{tok / max(wall + draft, 1e-9) * 1000:.1f} | {m:.1f} |")
    print()
    print("Kept by guesses checked (all kinds): share of checks keeping at least j guesses")
    print("| arm | guessed | checks | kept mean | p50 wall ms | at least 1 .. guessed |")
    print("|---|---:|---:|---:|---:|---|")
    for arm, (by, rows) in speed.items():
        ss = by["all"]
        for g in sorted({s["guessed"] for s in ss if s["guessed"] > 0}):
            cs = [s for s in ss if s["guessed"] == g]
            walls = sorted(s["wall"] for s in cs)
            surv = " ".join(f"{sum(1 for s in cs if s['kept'] >= j) / len(cs):.2f}" for j in range(1, g + 1))
            print(f"| {arm} | {g} | {len(cs)} | {sum(s['kept'] for s in cs) / len(cs):.2f} | "
                  f"{walls[len(walls) // 2] / 1000:.1f} | {surv} |")


if __name__ == "__main__":
    main(sys.argv[1:])
