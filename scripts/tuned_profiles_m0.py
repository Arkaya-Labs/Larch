#!/usr/bin/env python3
"""Tuned profiles milestone 0: runs paired bench blocks and analyses them. Python 3.11+, stdlib only.

    python3 scripts/tuned_profiles_m0.py run --bin <larch> --model <dir>/<alias>.gguf --machine <name> \
        --b c256:prefill_chunk_tokens=256 --pairs 9 --out <dir> --lock ~/src/gpu.lock --wait
    python3 scripts/tuned_profiles_m0.py analyze <dir>/blocks.csv

A block is one `larch bench` process running one workload type once (scripted, greedy, EOS ignored, so every arm
does the same work); a pair is one block per arm, in a random order. `run` appends every block to `blocks.csv` beside
its result JSON and resumes where a stopped run left off. Exit 3: another GPU process or a compile appeared, or
`--hold-min` ran out (release the lock, rerun later; `--wait` does that itself); exit 4: a GPU fault (stop and
report). bench/tuned-profiles-m0.md is the runbook for other machines.

Hosts: NVIDIA (Linux, WSL) reads temperature, clocks, power limit and other compute processes from nvidia-smi;
without nvidia-smi (macOS) those columns stay empty, no temperature band applies, and other inference engines are
found by process name. AC power: Windows' battery status on WSL, `pmset -g batt` on macOS, /sys/class/power_supply
elsewhere (no battery = AC).
"""

import argparse
import bisect
import csv
import fcntl
import json
import math
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Workload types (the brief's 1-4) on existing bench workloads: W1's first 6 turns at 1, W3's first loop's first 10
# turns at 1 (both trimmed so a 9-pair confirmation fits an hour), W2 at 4, and a W2 chat at 2 whose second client
# sends W5's 4096-token document after one short turn, so it arrives while the first is generating.
TYPES = {
    "short": ("chat", 1, "short.jsonl"),
    "long": ("agent", 1, "long.jsonl"),
    "conc": ("chats", 4, None),
    "doc": ("chats", 2, "doc.jsonl"),
}
SHORT_TURNS, LONG_TURNS = 6, 10
CONCURRENT = ("conc", "doc")
# The engine's fault path exits the process with 3 (larch-engine lifecycle.rs, OnFault::Exit) and writes no result.
FAULT_EXIT = 3
PERMS = 5000  # random relabellings in the order test
SINGLE = ("short", "long")
# Proposed bars (brief section 6); report raw numbers too.
MIN_GAIN_MS, MIN_GAIN_FRAC = 100.0, 0.20
LOOKS = ((9, 0.005), (18, 0.045))
# metric, direction (+1: higher is harm), margin on the ratio, types it applies to
NO_HARM = (("tok_s", -1, 0.97, tuple(TYPES)), ("ttft_p50", +1, 1.10, tuple(TYPES)), ("gap_p50", +1, 1.03, SINGLE))
COLS = ["comp", "model", "type", "pair", "attempt", "pos", "arm", "sets", "started", "cool_s", "proc_s", "wall_s",
        "temp_start", "temp_max", "clock_start", "clock_min", "clock_med", "power_limit", "on_ac", "gap_max", "gap_max_t", "gap_p99",
        "gap_p50", "tok_s", "ttft_p50", "ttft_p99", "step_max_ms", "failures", "gen_tokens", "evaluated_tokens",
        "preemptions", "exit", "void", "note", "json"]


# ---------------------------------------------------------------------------------------------------------- stats

def signed_rank_null(ranks):
    """Exact null distribution of W+ for the given (doubled, so integer) ranks: {sum: count} over all 2^n sign
    flips."""
    dist = {0: 1}
    for r in ranks:
        nxt = dict(dist)
        for s, c in dist.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        dist = nxt
    return dist


def gain_p(d):
    """One-sided exact sign-flip (Wilcoxon signed-rank) p that d is centred above 0. Zero differences are dropped,
    tied magnitudes get midranks. Returns (p, n non-zero)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n  # doubled midranks
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2
        i = j + 1
    w = sum(r for r, x in zip(ranks2, d) if x > 0)
    dist = signed_rank_null(ranks2)
    return sum(c for s, c in dist.items() if s >= w) / 2 ** n, n


def w_crit(n, alpha):
    """Smallest w with P(W+ >= w) <= alpha for n untied pairs, or None when no w is that rare."""
    dist = signed_rank_null([2 * (i + 1) for i in range(n)])
    tail = 0
    for s in sorted(dist, reverse=True):
        if (tail + dist[s]) / 2 ** n > alpha:
            return None if tail == 0 else (s // 2) + 1
        tail += dist[s]
    return 0


def walsh(x):
    return sorted((x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x)))


def hl(x):
    """Hodges-Lehmann estimate: the median of the Walsh averages."""
    return statistics.median(walsh(x))


def hl_bounds(x, alpha):
    """(lower, upper) bounds that invert the signed-rank test at one-sided alpha: `#Walsh averages > delta` is W+ of
    x - delta, so each bound is an order statistic of the Walsh averages; infinite when n is too small for alpha."""
    a = walsh(x)
    w = w_crit(len(x), alpha) if x else None
    if not w:
        return -math.inf, math.inf
    return a[len(a) - w], a[w - 1]


SINGLE_LOOK = LOOKS[:1]  # the confirmation: 9 fresh pairs, one look, p < 0.005


def gain_verdict(std, cand, looks=LOOKS):
    """The sequential gain test on paired primary values (std and candidate per pair, in run order). Returns a dict:
    look used, p, HL and bounds of std - cand, the min-improvement check, and passed / failed / pending /
    insufficient. With looks=SINGLE_LOOK a miss at 9 pairs is final (failed), never pending."""
    d_all = [s - c for s, c in zip(std, cand)]
    out = {"pairs": len(d_all)}
    for k, (n_look, alpha) in enumerate(looks):
        d = d_all[:n_look]
        p, nz = gain_p(d)
        if len(d) < n_look:
            if k == 0:
                out.update(look=0, p=p, nz=nz, verdict="insufficient")
                return out
            out["verdict"] = "pending"  # look 1 failed and look 2 is not yet reached
            return out
        lo, hi = hl_bounds([x for x in d if x != 0], alpha)
        base = statistics.median(std[:n_look])
        out.update(look=k + 1, p=p, nz=nz, alpha=alpha, hl=hl(d), lo=lo, hi=hi, base=base)
        if nz < LOOKS[0][0]:
            out["verdict"] = "insufficient"
            return out
        big = lo >= MIN_GAIN_MS and lo >= MIN_GAIN_FRAC * base
        if p < alpha and big:
            out["verdict"] = "passed"
            return out
        out["test_passed"] = p < alpha
    out["verdict"] = "failed"
    return out


def no_harm(std, cand, direction, margin, alpha=0.05):
    """One-sided 95 % bound on the paired log ratio cand/std, against the margin. Returns (ratio HL, bound, ok)."""
    x = [math.log(c / s) for s, c in zip(std, cand) if s > 0 and c > 0]
    if len(x) < 2:
        return None, None, None
    lo, hi = hl_bounds(x, alpha)
    est = math.exp(hl(x))
    if direction > 0:
        b = math.exp(hi) if hi != math.inf else math.inf
        return est, b, b <= margin
    b = math.exp(lo) if lo != -math.inf else 0.0
    return est, b, b >= margin


# ------------------------------------------------------------------------------------------------------------ run

def nvidia():
    """True when nvidia-smi is on PATH: GPU telemetry and the compute-process list come from it."""
    return shutil.which("nvidia-smi") is not None


def smi(query):
    """nvidia-smi's values for the query, or empty strings without nvidia-smi."""
    if not nvidia():
        return [""] * len(query.split(","))
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [v.strip() for v in out.splitlines()[0].split(",")]


# Without nvidia-smi (Metal) no tool lists GPU clients without root, so other inference engines are found by name.
# Our own bench process has exited whenever this is checked.
ENGINE_PROCS = ("larch", "larch-agent", "llama-server", "llama-cli", "llama-bench", "ollama", "LM Studio", "mlx_lm")


def ere(names):
    """`pgrep` pattern matching exactly these process names: it reads an extended regex, where clang++'s `+` is
    a repeat."""
    return "|".join("".join("\\" + ch if ch in ".^$*+?()[]{}|\\" else ch for ch in n) for n in names)


def other_gpu_procs():
    if nvidia():
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [p.strip() for p in out.splitlines() if p.strip()]
    out = subprocess.run(["pgrep", "-x", ere(ENGINE_PROCS)], capture_output=True, text=True).stdout
    return out.split()


# Compilers and test runners on the same host: one overlapping a block skewed decode 67 -> 36 tok/s in an earlier
# session and 79 -> 64 tok/s in this one's smoke run, so they count as environmental interruptions.
BUILD_PROCS = ("cargo", "rustc", "clippy-driver", "nvcc", "ptxas", "cicc", "cc1plus", "clang", "clang++")


def builds():
    """PIDs of compile or test processes running on the host."""
    out = subprocess.run(["pgrep", "-x", ere(BUILD_PROCS)], capture_output=True, text=True).stdout
    return out.split()


def pmset_on_ac(text):
    """macOS `pmset -g batt`: its first line names the source ("Now drawing from 'AC Power'")."""
    first = text.splitlines()[0] if text.strip() else ""
    return "Battery Power" not in first


def sysfs_on_ac(root="/sys/class/power_supply"):
    """Linux: on AC when a mains supply is online, or when there is no battery at all (a desktop)."""
    supplies = [pathlib.Path(p) for p in sorted(pathlib.Path(root).glob("*"))] if pathlib.Path(root).exists() else []
    read = lambda d, f: (d / f).read_text().strip() if (d / f).exists() else ""  # noqa: E731
    mains = [d for d in supplies if read(d, "type") == "Mains"]
    if mains:
        return any(read(d, "online") == "1" for d in mains)
    return not any(read(d, "type") == "Battery" and read(d, "status") == "Discharging" for d in supplies)


def on_ac():
    """AC power: Windows' battery status on WSL (2 = AC), pmset on macOS, sysfs elsewhere. True when the check
    itself fails, so a desktop is never voided by a missing tool."""
    try:
        if sys.platform == "darwin":
            return pmset_on_ac(subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                                              timeout=20).stdout)
        if shutil.which("powershell.exe"):
            out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Battery).BatteryStatus"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            return out in ("", "2") or "2" in out.split()
        return sysfs_on_ac()
    except (OSError, subprocess.TimeoutExpired):
        return True


def frozen(name):
    return [json.loads(line) for line in (ROOT / "bench/prompts" / name).read_text().splitlines()]


def first_turns(conv, n):
    """The conversation up to and including its n-th assistant turn."""
    k = [i for i, m in enumerate(conv["messages"]) if m["role"] == "assistant"][n - 1]
    return {"id": f"{conv['id']}-t{n}", "messages": conv["messages"][:k + 1]}


def write_workloads(out):
    """The trimmed and composed workload files the types name, written beside the results."""
    (out / "short.jsonl").write_text(json.dumps(first_turns(frozen("w1-chat.jsonl")[0], SHORT_TURNS)) + "\n")
    (out / "long.jsonl").write_text(json.dumps(first_turns(frozen("w3-agent.jsonl")[0], LONG_TURNS)) + "\n")
    doc_workload(out / "doc.jsonl")


def doc_workload(out):
    """The doc type's workload: W2's first conversation, and a second whose short first turn (W2 c1's) is followed by
    W5's 4096-token document, so the document arrives while conversation 0 generates."""
    w2, w5 = frozen("w2-chats.jsonl"), frozen("w5-longprompt.jsonl")
    doc = next(c for c in w5 if c["id"] == "w5-4096")["messages"]
    second = {"id": "doc-c1", "messages": w2[1]["messages"][:3] + doc}
    second["messages"][2] = dict(second["messages"][2], max_tokens=32)
    out.write_text("".join(json.dumps(c) + "\n" for c in (w2[0], second)))


def block_metrics(path):
    v = json.loads(path.read_text())
    s = v["summary"]
    ok = [r for r in v["requests"] if r["error"] is None]
    best = (0.0, None)
    for r in ok:
        t = r["t_first_ms"]
        for g in r.get("token_gaps_ms") or []:
            t += g
            if g > best[0]:
                best = (g, t / 1e3)
    tel = v.get("telemetry") or {}
    shapes = (tel.get("step_profile") or {}).get("shapes") or []
    dec = [sh["dur_ns"] / sh["steps"] / 1e6 for sh in shapes if sh["decode_sequences"] > 0 and sh["steps"]]
    g = s.get("token_gap_ms") or {}
    return {"wall_s": s["wall_s"], "gap_max": g.get("max"), "gap_max_t": best[1], "gap_p99": g.get("p99"),
            "gap_p50": g.get("p50"), "tok_s": s["agg_tok_s"], "ttft_p50": (s.get("ttft_ms") or {}).get("p50"),
            "ttft_p99": (s.get("ttft_ms") or {}).get("p99"), "step_max_ms": max(dec, default=None),
            "failures": s["failures"], "gen_tokens": s["gen_tokens"], "evaluated_tokens": s["evaluated_tokens"],
            "preemptions": tel.get("engine_total_preemptions")}


def run_block(a, out, typ, arm, sets, tag):
    """One measured block. Returns a CSV row dict; row['void'] is set for an environmental interruption."""
    workload, conc, prompts = TYPES[typ]
    row = {"arm": arm, "sets": " ".join(sets), "void": "", "note": ""}
    if other_gpu_procs() or builds():
        sys.exit(3)
    t0 = time.time()
    hot = lambda t: t != "" and int(t) > a.temp_max  # noqa: E731  (no reading without nvidia-smi: no band)
    while hot(smi("temperature.gpu")[0]) and time.time() - t0 < a.cool_max:
        time.sleep(2)
    time.sleep(a.idle)
    row["cool_s"] = round(time.time() - t0, 1)
    temp, clock, plimit = smi("temperature.gpu,clocks.sm,enforced.power.limit")
    row.update(temp_start=temp, clock_start=clock, power_limit=plimit, on_ac=on_ac())
    if hot(temp):
        row["void"] = f"start {temp} C above {a.temp_max}"
    if not row["on_ac"]:
        row["void"] = "on battery"
    if row["void"]:
        return row
    js = out / f"{tag}.json"
    cmd = [a.bin, "bench", "-m", a.model, "--workload", workload, "--concurrency", str(conc), "--machine", a.machine,
           "--json", str(js)]
    if prompts:
        cmd += ["--prompts", str(out / prompts)]
    for s in sets:
        cmd += ["--set", s]
    samp = subprocess.Popen(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits",
                             "-lms", "500"], stdout=subprocess.PIPE, text=True) if nvidia() else None
    row["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    t1 = time.time()
    with (out / f"{tag}.log").open("w") as log:
        try:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
            row["exit"] = p.returncode
        except subprocess.TimeoutExpired:
            row["exit"], row["note"] = "timeout", f"killed after {a.timeout} s"
    row["proc_s"] = round(time.time() - t1, 2)
    if samp:
        samp.terminate()
        # The 0.5 s trace plus one reading at the end, so a block shorter than the sampler's first line has one too.
        trace = [ln.split(",") for ln in samp.communicate()[0].splitlines() if ln.count(",") == 1]
        trace.append(smi("temperature.gpu,clocks.sm"))
        row["temp_max"] = max(int(t) for t, _ in trace)
        row["clock_min"] = min(int(c) for _, c in trace)
        row["clock_med"] = statistics.median(int(c) for _, c in trace)
    if row["exit"] == FAULT_EXIT:
        row["note"] = (row["note"] + " fault").strip()
    if js.exists():
        row.update(block_metrics(js))
        row["json"] = js.name
    if other_gpu_procs() or not on_ac():
        row["void"] = "another GPU process or battery during the block"
    elif builds():
        row["void"] = "a compile or test started during the block"
    return row


def done_pairs(rows, comp, typ, model):
    """Pair numbers with two blocks in one attempt and no void among them."""
    by = {}
    for r in rows:
        if r["comp"] == comp and r["type"] == typ and r["model"] == model:
            by.setdefault((r["pair"], r["attempt"]), []).append(r)
    return sorted(int(p) for (p, _), v in by.items() if len(v) == 2 and not any(r["void"] for r in v))


def run(a):
    """Runs the pairs under the GPU lock (`--lock`). With `--wait`, exit 3 (busy host or hold over) releases the lock,
    sleeps `--retry-s` and goes on from blocks.csv, so one command finishes the stage."""
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_workloads(out)
    csv_path = out / "blocks.csv"
    while True:
        lock = open(os.path.expanduser(a.lock), "a") if a.lock else None
        try:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            rows = list(csv.DictReader(csv_path.read_text().splitlines())) if csv_path.exists() else []
            with csv_path.open("a", newline="") as f:
                try:
                    return run_pairs(a, out, rows, f)
                except SystemExit as e:
                    if e.code != 3 or not a.wait:
                        raise
        finally:
            if lock:
                lock.close()  # closing releases the flock
        print(f"busy or hold over: lock released, retrying in {a.retry_s:.0f} s", flush=True)
        time.sleep(a.retry_s)


def run_pairs(a, out, rows, f):
    w = csv.DictWriter(f, COLS)
    if f.tell() == 0:  # a new file (a run that stopped before its first block left only a header otherwise)
        w.writeheader()
    arms = [(x.split(":", 1)[0], [s for s in x.split(":", 1)[1].split(",") if s] if ":" in x else []) for x in (a.a, a.b)]
    comp = f"{arms[0][0]}-vs-{arms[1][0]}"
    model = pathlib.Path(a.model).stem
    rng = random.Random(f"{a.seed}-{comp}-{model}")
    deadline = time.time() + a.hold_min * 60
    for pair in range(a.pairs):
        for typ in a.types.split(","):
            order = rng.sample([0, 1], 2)  # drawn even for finished pairs, so a resumed run keeps the order
            while True:
                have = [r for r in rows if r["comp"] == comp and r["type"] == typ and r["pair"] == str(pair)
                        and r["model"] == model]
                attempt = len({r["attempt"] for r in have})
                # A voided pair is repeated once (brief section 5); a second void leaves the pair missing.
                if pair in done_pairs(rows, comp, typ, model) or attempt >= 2:
                    break
                if time.time() > deadline:
                    sys.exit(3)
                for pos, i in enumerate(order):
                    name, sets = arms[i]
                    tag = f"{model}-{typ}-{comp}-p{pair}a{attempt}-{name}"
                    r = run_block(a, out, typ, name, sets, tag)
                    r.update(comp=comp, model=model, type=typ, pair=pair, attempt=attempt, pos=pos)
                    w.writerow(r)
                    f.flush()
                    rows.append({k: str(r.get(k, "")) for k in COLS})
                    print(f"{tag} pos {pos}: exit {r.get('exit')} gap_max {r.get('gap_max')} tok/s {r.get('tok_s')} "
                          f"void '{r['void']}' {r['note']}", flush=True)
                    if "fault" in r["note"]:
                        sys.exit(4)
                    if r["void"]:
                        # The pair's other block, if run, is voided with it.
                        for x in rows:
                            if x["comp"] == comp and x["type"] == typ and x["pair"] == str(pair) \
                                    and x["attempt"] == str(attempt) and x["model"] == model:
                                x["void"] = x["void"] or "pair voided"
                        break
    return 0


# -------------------------------------------------------------------------------------------------------- analyse

def num(v):
    return float(v) if v not in ("", None, "None") else None


def pairs_of(rows, comp, typ, model=None):
    """[(std row, cand row)] for complete, unvoided pairs in pair order; std is the comparison's first arm."""
    by = {}
    for r in rows:
        if r["comp"] == comp and r["type"] == typ and model in (None, r["model"]):
            by.setdefault((int(r["pair"]), int(r["attempt"])), {})[r["arm"]] = r
    std_name = comp.split("-vs-")[0]
    return [(v[std_name], next(x for k, x in v.items() if k != std_name)) for _, v in sorted(by.items())
            if len(v) == 2 and not any(x["void"] for x in v.values())]


def cand_failures(rows, comp, typ, model=None):
    """Failed candidate blocks, paired or not: a failure whose reference never ran (an interrupted, resumed run)
    still counts, so a rerun cannot retry it away. Voided blocks (environment, not the arm) don't count."""
    std_name = comp.split("-vs-")[0]
    return sum(int(num(r["failures"]) or 0) + (r["exit"] not in ("0", "") and not num(r["failures"]))
               for r in rows if r["comp"] == comp and r["type"] == typ and model in (None, r["model"])
               and r["arm"] != std_name and not r["void"])


def fmt(x, n=1):
    return "—" if x is None else ("∞" if x == math.inf else f"{x:.{n}f}")


def analyze(a):
    rows = list(csv.DictReader(pathlib.Path(a.csv).read_text().splitlines()))
    combos = sorted({(r["model"], r["comp"], r["type"]) for r in rows}, key=lambda t: (t[0], t[1], list(TYPES).index(t[2])))
    print("## Comparisons\n")
    print("| model | comparison | type | pairs | std gap max p50 ms | cand gap max p50 ms | HL std−cand ms [bounds] | "
          "p (look) | gain | tok/s ratio, 95 % lower | TTFT p50 ratio, 95 % upper | gap p50 ratio, 95 % upper | "
          "failures cand | verdict |")
    print("|" + "---|" * 14)
    verdicts, info = {}, {}
    single = a.single_look if a.single_look else [CALIBRATED]
    for model, comp, typ in combos:
        pr = pairs_of(rows, comp, typ, model)
        if not pr:
            continue
        # A block without the metric (a timeout or a failed run) counts as a failure below, not as a pair here.
        both = [(num(s["gap_max"]), num(c["gap_max"])) for s, c in pr]
        both = [(x, y) for x, y in both if x is not None and y is not None]
        sv, cv = [x for x, _ in both], [y for _, y in both]
        fails = cand_failures(rows, comp, typ, model)
        looks = SINGLE_LOOK if comp in single else LOOKS
        g = gain_verdict(sv, cv, looks) if typ in CONCURRENT and sv else None
        # Too few usable pairs to bound a metric shows no harm no more than it shows harm.
        harm, unknown, cells = False, False, []
        for metric, direction, margin, types in NO_HARM:
            if typ not in types:
                cells.append("n/a")
                continue
            vals = [(num(s[metric]), num(c[metric])) for s, c in pr]
            vals = [(x, y) for x, y in vals if x is not None and y is not None]
            est, b, ok = no_harm([x for x, _ in vals], [y for _, y in vals], direction, margin)
            harm |= ok is False
            unknown |= ok is None
            cells.append(f"{fmt(est, 3)}, {fmt(b, 3)} {'ok' if ok else 'FAIL' if ok is False else '?'}")
        if fails or harm:
            v = "rejected"
        elif unknown:
            v = "insufficient"
        elif g is None:
            v = "no harm" if typ not in CONCURRENT else "insufficient"
        else:
            v = {"passed": "confirmed", "failed": "rejected"}.get(g["verdict"], g["verdict"])
        verdicts[(model, comp, typ)] = v
        info[(model, comp, typ)] = {"verdict": v, "harm": bool(fails or harm), "unknown": unknown, "gain": g}
        gain = "n/a" if g is None else g["verdict"]
        hlc = "n/a" if g is None or "hl" not in g else f"{fmt(g['hl'])} [{fmt(g['lo'])}, {fmt(g['hi'])}]"
        pc = "n/a" if g is None else f"{g['p']:.4f} ({g.get('look', 0)})"
        med = lambda v: statistics.median(v) if v else None  # noqa: E731
        print(f"| {model} | {comp} | {typ} | {len(pr)} | {fmt(med(sv))} | {fmt(med(cv))} | "
              f"{hlc} | {pc} | {gain} | {cells[0]} | {cells[1]} | {cells[2]} | {fails} | {v} |")
    calibration(info)
    aa(rows)
    order(rows)
    budget(rows)


ALWAYS, CALIBRATED = "std-vs-c256", "std-vs-cal"  # the brief's always-256 arm and the confirmation of the winner


def arm_summary(info, model, comp):
    """One arm on one model: per concurrent type the stall cut (ms, % of the standard median) and whether the arm
    passes every test (assumption 2: all four types run, no harm or failure on any, confirmed on a concurrent type)."""
    got = {t: info.get((model, comp, t)) for t in TYPES}
    cuts = {t: (got[t]["gain"]["hl"], 100 * got[t]["gain"]["hl"] / got[t]["gain"]["base"])
            for t in CONCURRENT if got[t] and got[t]["gain"] and "hl" in got[t]["gain"]}
    missing = [t for t, x in got.items() if not x]
    harm = [t for t, x in got.items() if x and x["harm"]]
    unknown = [t for t, x in got.items() if x and x.get("unknown")]
    confirmed = [t for t in CONCURRENT if got[t] and got[t]["verdict"] == "confirmed"]
    why = ("missing types " + ",".join(missing) if missing else "harm or failure on " + ",".join(harm) if harm
           else "too few pairs to show no harm on " + ",".join(unknown) if unknown
           else "not confirmed on a concurrent type" if not confirmed else "")
    # Safe to ship: shown, on every type, to do no harm (a gain is not needed for that).
    safe = not (missing or harm or unknown)
    return {"cuts": cuts, "harm": harm, "safe": safe, "passes": not why, "why": why,
            "mean_pct": statistics.mean(c[1] for c in cuts.values()) if cuts else None}


def calibration(info):
    """The hand calibration's winner among the candidates, and this (machine, model) pair against the gate."""
    models = sorted({m for m, _, _ in info})
    for model in models:
        comps = sorted({c for m, c, _ in info if m == model and not c.startswith("std-vs-std") and c != CALIBRATED})
        if not comps:
            continue
        print(f"\n## Hand calibration: {model}\n")
        print("| candidate | conc stall cut | doc stall cut | passes every test |")
        print("|---|---|---|---|")
        best = None
        for comp in comps:
            sm = arm_summary(info, model, comp)
            cell = lambda t: (f"{sm['cuts'][t][0]:.0f} ms, {sm['cuts'][t][1]:.0f} %"  # noqa: E731
                              if t in sm["cuts"] else "—")
            print(f"| {comp} | {cell('conc')} | {cell('doc')} | {'yes' if sm['passes'] else 'no: ' + sm['why']} |")
            if sm["passes"] and (best is None or sm["mean_pct"] > best[1]):
                best = (comp, sm["mean_pct"])
        print("\nWinner (largest mean % stall cut over conc and doc among candidates that pass every test): "
              + (f"**{best[0]}**. Confirm it as `{CALIBRATED}` (arm name `cal`, same --set values, 9 fresh pairs)."
                 if best else "none: no candidate passes every test, so there is nothing to calibrate to."))
        if not any(c == ALWAYS for c in comps) or (model, CALIBRATED, "conc") not in info:
            print(f"\nGate: needs both {ALWAYS} and {CALIBRATED} for this model.")
            continue
        al, ca = arm_summary(info, model, ALWAYS), arm_summary(info, model, CALIBRATED)
        more = [t for t in CONCURRENT if t in al["cuts"] and t in ca["cuts"] and al["cuts"][t][0] > 0
                and ca["cuts"][t][0] >= 1.10 * al["cuts"][t][0]]
        avoids = bool(al["harm"]) and not ca["harm"]
        counts = ca["passes"] and not ca["harm"] and (bool(more) or avoids)
        print(f"\nGate ({model}): calibrated {'passes' if ca['passes'] else 'does not pass (' + ca['why'] + ')'}; "
              f"always-256 {'harms ' + ','.join(al['harm']) if al['harm'] else 'passes no-harm' if al['safe'] else 'is not shown to do no harm on every type'}; "
              f"calibrated cut ≥ 1.10 × always-256's on: {','.join(more) or 'none'}; "
              f"avoids an always-256 no-harm failure: {'yes' if avoids else 'no'}.")
        print(f"This pair counts toward **{'build calibration' if counts else 'not building calibration'}**; "
              f"fallback for this pair: **{'ship always-256' if al['safe'] else 'ship nothing'}**.")


def aa(rows):
    """A/A spread and the false-confirm rate: every 9-pair window of each A/A comparison, both labellings, through
    the gain test and the no-harm tests."""
    comps = sorted({(r["model"], r["comp"], r["type"]) for r in rows if r["comp"].startswith("std-vs-std")})
    if not comps:
        return
    print("\n## A/A spread (log ratio second-label / first-label, as %)\n")
    print("| model | type | pairs | metric | sd % | max abs % | CV of all standard blocks % |")
    print("|---|---|---|---|---|---|---|")
    confirms = windows = failed = 0
    for model, comp, typ in comps:
        pr = pairs_of(rows, comp, typ, model)
        tested = ["gap_max"] + [m for m, _, _, types in NO_HARM if typ in types]
        for metric in ("gap_max", "gap_p99", "gap_p50", "tok_s", "ttft_p50", "step_max_ms"):
            xs = [(num(s[metric]), num(c[metric])) for s, c in pr]
            xs = [(s, c) for s, c in xs if s and c]
            if len(xs) < 2:
                continue
            lr = [100 * math.log(c / s) for s, c in xs]
            allv = [v for p in xs for v in p]
            print(f"| {model} | {typ} | {len(xs)} | {metric} | {statistics.stdev(lr):.1f} | {max(map(abs, lr)):.1f} | "
                  f"{100 * statistics.stdev(allv) / statistics.mean(allv):.1f} |")
        for i in range(max(0, len(pr) - 8)):
            for flip in (False, True):
                win = pr[i:i + 9]
                if flip:
                    win = [(c, s) for s, c in win]
                # A block that failed (a timeout: no metrics) fails the window; it is never a confirm.
                if any(num(x[m]) is None for pair in win for x in pair for m in tested):
                    failed += 1
                    continue
                windows += 1
                g = gain_verdict([num(s["gap_max"]) for s, _ in win], [num(c["gap_max"]) for _, c in win])
                ok = all(no_harm([num(s[m]) for s, _ in win], [num(c[m]) for _, c in win], d, mg)[2] is not False
                         for m, d, mg, t in NO_HARM if typ in t)
                confirms += typ in CONCURRENT and g["verdict"] == "passed" and ok
    print(f"\nA/A false confirms: {confirms} of {windows} 9-pair windows (sliding, both labellings)"
          + (f"; {failed} more held a failed block and were not tested." if failed else "."))


def order(rows):
    """Order and carryover: the paired difference (second label minus first) when the first label ran first against
    when it ran second, and for A/A the position effect (second block minus first)."""
    print("\n## Order\n")
    print("| model | comparison | type | metric | n first-label-first | mean Δ % | n second-first | mean Δ % | "
          "perm p (two-sided) |")
    print("|---|---|---|---|---|---|---|---|---|")
    rng = random.Random(0)
    for model, comp, typ in sorted({(r["model"], r["comp"], r["type"]) for r in rows}):
        pr = pairs_of(rows, comp, typ, model)
        for metric in ("gap_max", "tok_s", "ttft_p50"):
            g1, g2 = [], []
            for s, c in pr:
                if num(s[metric]) and num(c[metric]):
                    (g1 if s["pos"] == "0" else g2).append(100 * math.log(num(c[metric]) / num(s[metric])))
            if len(g1) < 2 or len(g2) < 2:
                continue
            n1, total = len(g1), sum(g1) + sum(g2)
            n2 = len(g2)
            obs = abs(sum(g1) / n1 - sum(g2) / n2)
            allg, hits = g1 + g2, 0
            for _ in range(PERMS):
                rng.shuffle(allg)
                s1 = sum(allg[:n1])
                hits += abs(s1 / n1 - (total - s1) / n2) >= obs - 1e-9
            print(f"| {model} | {comp} | {typ} | {metric} | {len(g1)} | {statistics.mean(g1):+.1f} | {len(g2)} | "
                  f"{statistics.mean(g2):+.1f} | {hits / PERMS:.3f} |")


def budget(rows):
    print("\n## Time per block (s, medians)\n")
    print("| model | type | blocks | cool-down | process | run (wall_s) | load + warm-up + teardown |")
    print("|---|---|---|---|---|---|---|")
    for model, typ in sorted({(r["model"], r["type"]) for r in rows}):
        b = [r for r in rows if r["model"] == model and r["type"] == typ and num(r["proc_s"]) and num(r["wall_s"])]
        if b:
            med = lambda k: statistics.median(num(r[k]) for r in b)  # noqa: E731
            print(f"| {model} | {typ} | {len(b)} | {med('cool_s'):.1f} | {med('proc_s'):.1f} | {med('wall_s'):.1f} | "
                  f"{statistics.median(num(r['proc_s']) - num(r['wall_s']) for r in b):.1f} |")
    voids = [r for r in rows if r["void"]]
    print(f"\nVoided blocks: {len(voids)}" + "".join(f"\n- {r['comp']} {r['type']} pair {r['pair']}: {r['void']}" for r in voids))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bin", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--types", default="short,long,conc,doc")
    r.add_argument("--a", default="std", help="name[:knob=v,...], the reference arm")
    r.add_argument("--b", required=True, help="name[:knob=v,...]")
    r.add_argument("--pairs", type=int, default=9)
    r.add_argument("--out", required=True)
    r.add_argument("--machine", default="sm86")
    r.add_argument("--seed", default="m0")
    r.add_argument("--temp-max", type=int, default=70, help="start-temperature band's upper end, C")
    r.add_argument("--cool-max", type=float, default=300, help="longest wait for the band, s")
    r.add_argument("--idle", type=float, default=5, help="idle seconds before every block")
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--hold-min", type=float, default=50, help="stop (exit 3) after this many minutes")
    r.add_argument("--lock", help="GPU lock file to hold while running (fcntl; e.g. ~/src/gpu.lock)")
    r.add_argument("--wait", action="store_true", help="on exit 3, release the lock, sleep --retry-s and go on")
    r.add_argument("--retry-s", type=float, default=300)
    z = sub.add_parser("analyze")
    z.add_argument("csv")
    z.add_argument("--single-look", action="append", default=[], metavar="COMP",
                   help=f"comparison analysed with one look at 9 pairs; repeatable (default: {CALIBRATED})")
    a = p.parse_args()
    sys.exit(run(a) if a.cmd == "run" else analyze(a))


if __name__ == "__main__":
    main()
