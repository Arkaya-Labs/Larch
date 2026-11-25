"""Tests for scripts/tuned_profiles_m0.py: python3 -m unittest scripts/test_tuned_profiles_m0.py -v

The statistics are checked against brute-force enumeration and simulation; the runner runs end to end against fake
`nvidia-smi`, `powershell.exe` and `larch` executables on PATH, so no GPU is needed.
"""

import contextlib
import csv
import fcntl
import io
import itertools
import json
import math
import os
import pathlib
import random
import statistics
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tuned_profiles_m0 as tp  # noqa: E402


# ------------------------------------------------------------------------------------------------ brute force

def brute_p(d):
    """P(W+ >= observed) by enumerating every sign flip, midranks for ties, zeros dropped."""
    d = [x for x in d if x != 0]
    mags = sorted(abs(x) for x in d)
    rank = {m: statistics.mean(i + 1 for i, v in enumerate(mags) if v == m) for m in set(mags)}
    obs = sum(rank[abs(x)] for x in d if x > 0)
    hits = sum(sum(rank[abs(x)] for x, s in zip(d, signs) if s) >= obs - 1e-9
               for signs in itertools.product((0, 1), repeat=len(d)))
    return hits / 2 ** len(d)


def w_plus(x):
    xs = sorted(x, key=abs)
    return sum(i + 1 for i, v in enumerate(xs) if v > 0)


class Stats(unittest.TestCase):
    def test_known_vector(self):
        # Wilcoxon, n = 9: P(W+ = 45) = 1/512, P(W+ >= 44) = 2/512, P(W+ >= 43) = 3/512.
        self.assertEqual(tp.gain_p(list(range(1, 10))), (1 / 512, 9))
        self.assertEqual(tp.gain_p([-1] + list(range(2, 10))), (2 / 512, 9))
        self.assertEqual(tp.gain_p([-2, 1] + list(range(3, 10))), (3 / 512, 9))
        self.assertEqual(tp.gain_p([-x for x in range(1, 10)]), (1.0, 9))
        self.assertEqual(tp.gain_p([0, 0] + list(range(1, 10)))[1], 9)  # zeros dropped
        self.assertEqual(tp.gain_p([0, 0]), (1.0, 0))
        self.assertEqual(tp.gain_p([]), (1.0, 0))

    def test_gain_p_matches_enumeration(self):
        rng = random.Random(3)
        for _ in range(60):
            # integer values, so ties and zeros occur
            d = [rng.randint(-4, 6) for _ in range(rng.randint(1, 11))]
            self.assertAlmostEqual(tp.gain_p(d)[0], brute_p(d), places=12, msg=d)

    def test_w_crit_matches_enumeration(self):
        for n in range(1, 15):
            counts = {}
            for signs in itertools.product((0, 1), repeat=n):
                w = sum(i + 1 for i, s in enumerate(signs) if s)
                counts[w] = counts.get(w, 0) + 1
            for alpha in (0.005, 0.025, 0.045, 0.05):
                want = None
                for w in range(n * (n + 1) // 2, -1, -1):
                    if sum(c for s, c in counts.items() if s >= w) / 2 ** n <= alpha:
                        want = w
                self.assertEqual(tp.w_crit(n, alpha), want, (n, alpha))
        self.assertEqual(tp.w_crit(9, 0.005), 44)
        self.assertIsNone(tp.w_crit(5, 0.005))  # 1/32 is the rarest outcome of 5 pairs

    def test_walsh_identity(self):
        # The bounds rest on #(Walsh averages > delta) == W+(x - delta) for untied data.
        rng = random.Random(5)
        for _ in range(100):
            x = [rng.gauss(0, 1) for _ in range(rng.randint(2, 12))]
            delta = rng.gauss(0, 1)
            self.assertEqual(sum(a > delta for a in tp.walsh(x)), w_plus([v - delta for v in x]))

    def test_bounds(self):
        self.assertEqual(tp.hl_bounds([3.0] * 9, 0.005), (3.0, 3.0))
        self.assertEqual(tp.hl_bounds(list(range(1, 10)), 0.005), (1.5, 8.5))  # 2nd smallest, 2nd largest
        self.assertEqual(tp.hl(list(range(1, 10))), 5)
        self.assertEqual(tp.hl_bounds([1.0, 2.0], 0.05), (-math.inf, math.inf))  # too few pairs for 5 %
        self.assertEqual(tp.hl_bounds([], 0.05), (-math.inf, math.inf))

    def test_bound_inverts_the_test(self):
        # Just below the lower bound the shifted data still rejects at alpha; just above it, it does not.
        rng = random.Random(11)
        for _ in range(40):
            x = [rng.gauss(1, 1) for _ in range(rng.choice((9, 12, 18)))]
            for alpha in (0.005, 0.045, 0.05):
                lo, hi = tp.hl_bounds(x, alpha)
                self.assertLessEqual(tp.gain_p([v - (lo - 1e-9) for v in x])[0], alpha)
                self.assertGreater(tp.gain_p([v - (lo + 1e-9) for v in x])[0], alpha)
                self.assertLessEqual(lo, tp.hl(x))
                self.assertLessEqual(tp.hl(x), hi)

    def test_hl_equivariant(self):
        rng = random.Random(2)
        x = [rng.gauss(0, 1) for _ in range(9)]
        self.assertAlmostEqual(tp.hl([v + 7 for v in x]), tp.hl(x) + 7)
        self.assertAlmostEqual(tp.hl([3 * v for v in x]), 3 * tp.hl(x))

    def test_lower_bound_coverage(self):
        # A one-sided 95 % bound covers the true shift at least 95 % of the time (exact test: conservative).
        rng = random.Random(8)
        miss = sum(tp.hl_bounds([0.3 + rng.gauss(0, 1) for _ in range(12)], 0.05)[0] > 0.3 for _ in range(2000))
        self.assertLessEqual(miss / 2000, 0.065)


class Gain(unittest.TestCase):
    def test_real_gain_confirms_at_look_1(self):
        rng = random.Random(1)
        std = [1200 + rng.uniform(-50, 50) for _ in range(9)]
        cand = [s - 600 + rng.uniform(-50, 50) for s in std]
        g = tp.gain_verdict(std, cand)
        self.assertEqual((g["look"], g["verdict"]), (1, "passed"))
        self.assertGreater(g["lo"], tp.MIN_GAIN_MS)

    def test_confirms_at_look_2(self):
        # Look 1: two losses among 9 (p > 0.005); look 2: the 18 pairs pass at 0.045.
        d = [-30, 400, 380, -20, 420, 390, 410, 395, 405] + [400 + 5 * i for i in range(9)]
        std = [1500.0] * 18
        cand = [s - x for s, x in zip(std, d)]
        g = tp.gain_verdict(std, cand)
        self.assertEqual((g["look"], g["verdict"]), (2, "passed"))
        self.assertEqual(tp.gain_verdict(std[:12], cand[:12])["verdict"], "pending")

    def test_single_look_is_final(self):
        # The confirmation has one look: the same 12 pairs that are pending sequentially are a failure there,
        # and a clean 9-pair gain passes at look 1.
        d = [-30, 400, 380, -20, 420, 390, 410, 395, 405, 400, 405, 410]
        std = [1500.0] * 12
        cand = [s - x for s, x in zip(std, d)]
        g = tp.gain_verdict(std, cand, tp.SINGLE_LOOK)
        self.assertEqual((g["look"], g["verdict"]), (1, "failed"))
        g = tp.gain_verdict(std[:9], [1100.0 + i for i in range(9)], tp.SINGLE_LOOK)
        self.assertEqual((g["look"], g["verdict"]), (1, "passed"))

    def test_small_gain_fails_min_improvement(self):
        std = [1000.0 + i for i in range(18)]
        cand = [s - 20 - (i % 3) for i, s in enumerate(std)]  # always lower, by about 20 ms only
        g = tp.gain_verdict(std, cand)
        self.assertEqual(g["verdict"], "failed")
        self.assertTrue(g["test_passed"])  # significant, but below the bar

    def test_relative_bar_binds(self):
        # 150 ms off a 1500 ms stall is 10 %: over 100 ms, under 20 %.
        std = [1500.0 + i for i in range(18)]
        cand = [s - 150 for s in std]
        self.assertEqual(tp.gain_verdict(std, cand)["verdict"], "failed")

    def test_insufficient(self):
        self.assertEqual(tp.gain_verdict([800.0] * 5, [700.0] * 5)["verdict"], "insufficient")
        same = [800.0] * 9
        self.assertEqual(tp.gain_verdict(same, same)["verdict"], "insufficient")
        mostly_same = [800.0] * 18
        self.assertEqual(tp.gain_verdict(mostly_same, mostly_same[:10] + [500.0] * 8)["verdict"], "insufficient")

    def test_aa_does_not_confirm(self):
        rng = random.Random(7)
        for _ in range(300):
            base = [rng.uniform(500, 1500) for _ in range(18)]
            a = [b * rng.lognormvariate(0, 0.2) for b in base]
            b = [x * rng.lognormvariate(0, 0.2) for x in base]
            self.assertNotEqual(tp.gain_verdict(a, b)["verdict"], "passed")

    def test_sequential_test_level(self):
        # Without the minimum-improvement bar the two looks together reject a null at most about 5 % of the time.
        rng = random.Random(9)
        hits = 0
        for _ in range(3000):
            d = [rng.gauss(0, 1) for _ in range(18)]
            hits += tp.gain_p(d[:9])[0] < 0.005 or tp.gain_p(d)[0] < 0.045
        self.assertLess(hits / 3000, 0.06)


class NoHarm(unittest.TestCase):
    def test_margins(self):
        std = [100.0] * 9
        self.assertTrue(tp.no_harm(std, [99.0] * 9, -1, 0.97)[2])
        self.assertFalse(tp.no_harm(std, [95.0] * 9, -1, 0.97)[2])
        self.assertTrue(tp.no_harm(std, [109.0] * 9, +1, 1.10)[2])
        self.assertFalse(tp.no_harm(std, [112.0] * 9, +1, 1.10)[2])
        est, bound, _ = tp.no_harm(std, [105.0] * 9, +1, 1.10)
        self.assertAlmostEqual(est, 1.05)
        self.assertAlmostEqual(bound, 1.05)

    def test_noise_widens_the_bound(self):
        rng = random.Random(4)
        noisy = [100.0 * rng.lognormvariate(0, 0.15) for _ in range(9)]
        est, bound, _ = tp.no_harm([100.0] * 9, noisy, +1, 1.10)
        self.assertGreater(bound, est)

    def test_too_few_or_bad_values(self):
        self.assertEqual(tp.no_harm([1.0], [1.0], +1, 1.1), (None, None, None))
        self.assertEqual(tp.no_harm([0.0, -1.0], [1.0, 1.0], +1, 1.1), (None, None, None))
        # 4 pairs cannot reach 5 %: the bound is infinite, so the metric fails rather than passes.
        self.assertFalse(tp.no_harm([100.0] * 4, [100.0] * 4, +1, 1.10)[2])
        self.assertFalse(tp.no_harm([100.0] * 4, [100.0] * 4, -1, 0.97)[2])


# ------------------------------------------------------------------------------------------------ fakes

FAKE_SMI = r'''#!/usr/bin/env python3
import json, os, sys, time
st = json.load(open(os.environ["FAKE_STATE"]))
args = " ".join(sys.argv[1:])
if "--query-compute-apps" in args:
    if st.get("procs_for"):  # a process that goes away after this many checks
        st["procs_for"] -= 1
        json.dump(st, open(os.environ["FAKE_STATE"], "w"))
        print("4242")
    print("\n".join(st.get("procs", [])))
elif "-lms" in args:
    for _ in range(3):
        print(f"{st['temp']}, 1700", flush=True)
    time.sleep(60)
else:
    fields = args.split("--query-gpu=")[1].split()[0].split(",")
    temps = st.get("temps")
    if temps:  # a falling temperature: one value per query
        st["temp"] = temps.pop(0)
        json.dump(st, open(os.environ["FAKE_STATE"], "w"))
    vals = {"temperature.gpu": st["temp"], "clocks.sm": 1700, "enforced.power.limit": "40.00"}
    print(", ".join(str(vals[f]) for f in fields))
'''

FAKE_PGREP = r'''#!/usr/bin/env python3
import json, os, sys
st = json.load(open(os.environ["FAKE_STATE"]))
key = "engines" if "llama-server" in sys.argv[-1] else "builds"  # the engine-name list or the compile list
if st.get(key):
    print("\n".join(st[key]))
sys.exit(0 if st.get(key) else 1)
'''

FAKE_PS = r'''#!/usr/bin/env python3
import json, os
print(json.load(open(os.environ["FAKE_STATE"])).get("ac", "2"))
'''

# The stall scales with prefill_chunk_tokens / 512; tok/s takes an optional penalty per chunk value.
FAKE_LARCH = r'''#!/usr/bin/env python3
import hashlib, json, os, random, sys, time
argv = sys.argv[1:]
st = json.load(open(os.environ["FAKE_STATE"]))
with open(os.environ["FAKE_ARGV"], "a") as f:
    f.write(json.dumps(argv) + "\n")
sets = dict(argv[i + 1].split("=") for i, a in enumerate(argv) if a == "--set")
out = argv[argv.index("--json") + 1]
mode = st.get("mode", {}).get(sets.get("prefill_chunk_tokens", "512"), "ok")
if st.get("procs_during"):
    st["procs"] = ["999"]
    json.dump(st, open(os.environ["FAKE_STATE"], "w"))
if st.get("build_during"):
    st["builds"] = ["777"]
    json.dump(st, open(os.environ["FAKE_STATE"], "w"))
if mode == "hang":
    time.sleep(60)
if mode == "fault":
    sys.exit(3)
rng = random.Random(hashlib.sha256(out.encode()).hexdigest())
chunk = int(sets.get("prefill_chunk_tokens", "512"))
noise = st.get("noise", 0.05)
stall = st.get("stall", 1000.0) * (chunk / 512 if st.get("effect", True) else 1.0)
gap_max = 60 + stall * rng.lognormvariate(0, noise)
tok_s = 100.0 * st.get("tok_penalty", {}).get(str(chunk), 1.0) * rng.lognormvariate(0, noise / 5)
failures = 1 if mode == "fail" else 0
reqs = [{"error": None, "t_first_ms": 100.0, "token_gaps_ms": [50.0] * 20 + [gap_max]}]
if failures:
    reqs.append({"error": "Timeout", "t_first_ms": None, "token_gaps_ms": None})
v = {"summary": {"wall_s": 2.0,
                 "token_gap_ms": {"max": gap_max, "p99": gap_max * 0.8, "p50": 50.0 * rng.lognormvariate(0, noise / 5)},
                 "agg_tok_s": tok_s, "ttft_ms": {"p50": 300.0 * rng.lognormvariate(0, noise / 5), "p99": 900.0},
                 "failures": failures, "gen_tokens": 2387, "evaluated_tokens": 4553},
     "requests": reqs,
     "telemetry": {"engine_total_preemptions": 0, "step_profile": {"shapes": [
         {"decode_sequences": 1, "prefill_sequences": 1, "steps": 2, "dur_ns": int(2 * gap_max * 1e6)},
         {"decode_sequences": 0, "prefill_sequences": 1, "steps": 1, "dur_ns": int(5000e6)},
         {"decode_sequences": 1, "prefill_sequences": 0, "steps": 0, "dur_ns": 0}]}}}
json.dump(v, open(out, "w"))
sys.exit(1 if failures else 0)
'''


class Fakes:
    """A directory with the three fakes on PATH and a state file they read."""

    def __init__(self, root, **state):
        self.dir = pathlib.Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        for name, src in (("nvidia-smi", FAKE_SMI), ("powershell.exe", FAKE_PS), ("larch", FAKE_LARCH),
                          ("pgrep", FAKE_PGREP)):
            p = self.bin / name
            p.write_text(src)
            p.chmod(0o755)
        self.state = self.dir / "state.json"
        self.argv = self.dir / "argv.jsonl"
        self.state.write_text(json.dumps({"temp": 50, "procs": [], "ac": "2", **state}))
        self.out = self.dir / "out"

    def calls(self):
        return [json.loads(line) for line in self.argv.read_text().splitlines()] if self.argv.exists() else []

    def rows(self):
        return list(csv.DictReader((self.out / "blocks.csv").read_text().splitlines()))

    def run(self, *extra, b="c256:prefill_chunk_tokens=256", pairs=2, types="conc", model="m3"):
        argv = ["run", "--bin", str(self.bin / "larch"), "--model", str(self.dir / f"{model}.gguf"), "--types", types,
                "--b", b, "--pairs", str(pairs), "--out", str(self.out), "--idle", "0", "--cool-max", "0",
                "--timeout", "5", *extra]
        env = {k: os.environ.get(k) for k in ("PATH", "FAKE_STATE", "FAKE_ARGV")}
        old_argv = sys.argv
        sys.argv = ["tuned_profiles_m0.py", *argv]
        os.environ.update(PATH=f"{self.bin}:{env['PATH']}", FAKE_STATE=str(self.state), FAKE_ARGV=str(self.argv))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tp.main()
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = old_argv
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def analysis(csv_path, single_look=()):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tp.analyze(type("A", (), {"csv": str(csv_path), "single_look": list(single_look)})())
    return buf.getvalue()


def verdict_of(text, comp, typ):
    """The verdict cell of a comparison row in analyze's main table."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 4 and cells[2] == comp and cells[3] == typ:
            return cells[-2]
    return None


def order_key(rows):
    return [(r["pair"], r["pos"], r["arm"]) for r in rows]


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp))

    def test_command_lines(self):
        fk = Fakes(self.tmp)
        self.assertEqual(fk.run(types="short,long,conc,doc", pairs=1), 0)
        calls = fk.calls()
        self.assertEqual(len(calls), 8)
        for c in calls:
            self.assertEqual(c[0], "bench")
            self.assertEqual(c[c.index("--machine") + 1], "sm86")
            w, n = c[c.index("--workload") + 1], c[c.index("--concurrency") + 1]
            self.assertIn((w, n), {("chat", "1"), ("agent", "1"), ("chats", "4"), ("chats", "2")})
            self.assertEqual("--prompts" in c, (w, n) != ("chats", "4"))
            if "--prompts" in c:
                self.assertTrue(pathlib.Path(c[c.index("--prompts") + 1]).exists())
        sets = [c[c.index("--set") + 1] for c in calls if "--set" in c]
        self.assertEqual(sets, ["prefill_chunk_tokens=256"] * 4)
        doc = [c for c in calls if "--prompts" in c][0]
        self.assertTrue(pathlib.Path(doc[doc.index("--prompts") + 1]).exists())

    def test_several_sets_per_arm(self):
        fk = Fakes(self.tmp)
        fk.run(b="c256bal:prefill_chunk_tokens=256,schedule_policy=balanced", pairs=1)
        cand = [c for c in fk.calls() if "--set" in c][0]
        self.assertEqual([cand[i + 1] for i, a in enumerate(cand) if a == "--set"],
                         ["prefill_chunk_tokens=256", "schedule_policy=balanced"])

    def test_rows_and_metrics(self):
        fk = Fakes(self.tmp)
        fk.run(pairs=1)
        rows = fk.rows()
        self.assertEqual(sorted(r["arm"] for r in rows), ["c256", "std"])
        self.assertEqual(sorted(r["pos"] for r in rows), ["0", "1"])
        for r in rows:
            self.assertEqual((r["void"], r["exit"], r["temp_start"], r["temp_max"]), ("", "0", "50", "50"))
            self.assertEqual((r["power_limit"], r["on_ac"], r["clock_min"], r["clock_med"]), ("40.00", "True", "1700", r["clock_med"]))
            self.assertEqual(float(r["clock_med"]), 1700)
            self.assertAlmostEqual(float(r["gap_max_t"]), (100 + 20 * 50 + float(r["gap_max"])) / 1e3)
            self.assertAlmostEqual(float(r["step_max_ms"]), float(r["gap_max"]), places=3)  # decode-bearing only
            self.assertTrue((fk.out / r["json"]).exists())
            self.assertTrue(r["started"])

    def test_order_random_both_ways_and_reproducible(self):
        a = Fakes(self.tmp / "a")
        a.run(pairs=12)
        firsts = [r["arm"] for r in a.rows() if r["pos"] == "0"]
        self.assertEqual(len(firsts), 12)
        self.assertGreater(firsts.count("std"), 0)
        self.assertGreater(firsts.count("c256"), 0)
        b = Fakes(self.tmp / "b")
        b.run(pairs=12)
        self.assertEqual(firsts, [r["arm"] for r in b.rows() if r["pos"] == "0"])
        c = Fakes(self.tmp / "c")
        c.run("--seed", "other", pairs=12)
        self.assertNotEqual(firsts, [r["arm"] for r in c.rows() if r["pos"] == "0"])

    def test_resume_keeps_pairs_and_order(self):
        fk = Fakes(self.tmp / "a")
        fk.run(pairs=2)
        first = order_key(fk.rows())
        fk.run(pairs=4)
        rows = fk.rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(order_key(rows)[:4], first)
        self.assertEqual(len(fk.calls()), 8)
        fk.run(pairs=4)  # nothing left to do
        self.assertEqual(len(fk.calls()), 8)
        fresh = Fakes(self.tmp / "b")
        fresh.run(pairs=4)
        self.assertEqual(order_key(fresh.rows()), order_key(rows))

    def test_hot_start_voids_and_repeats_once(self):
        fk = Fakes(self.tmp, temp=90)
        self.assertEqual(fk.run(pairs=1), 0)
        rows = fk.rows()
        self.assertEqual([r["attempt"] for r in rows], ["0", "1"])  # first block of each attempt, both void
        self.assertTrue(all("above" in r["void"] for r in rows))
        self.assertEqual(fk.calls(), [])  # a voided start never runs bench
        fk.run(pairs=1)  # a resumed run does not try a third time
        self.assertEqual(len(fk.rows()), 2)

    def test_cooldown_then_runs(self):
        fk = Fakes(self.tmp, temp=80, temps=[80, 75, 69] + [60] * 20)
        fk.run("--cool-max", "30", pairs=1)
        rows = fk.rows()
        self.assertEqual([r["void"] for r in rows], ["", ""])
        self.assertGreater(float(rows[0]["cool_s"]), 1)

    def test_battery_voids(self):
        fk = Fakes(self.tmp, ac="1")
        fk.run(pairs=1)
        self.assertTrue(all(r["void"] == "on battery" for r in fk.rows()))

    def test_other_process_at_start_exits_3(self):
        fk = Fakes(self.tmp, procs=["4242"])
        self.assertEqual(fk.run(pairs=1), 3)
        self.assertEqual(fk.calls(), [])

    def test_other_process_during_block_voids_pair(self):
        fk = Fakes(self.tmp, procs_during=True)
        code = fk.run(pairs=1)
        rows = fk.rows()
        self.assertEqual(rows[0]["void"], "another GPU process or battery during the block")
        self.assertEqual(code, 3)  # the repeat finds the process still there and gives the GPU back
        self.assertEqual(tp.done_pairs(rows, "std-vs-c256", "conc", "m3"), [])

    def test_void_second_block_voids_the_pair(self):
        fk = Fakes(self.tmp)
        fk.run(pairs=1)
        rows = fk.rows()
        rows[1]["void"] = "x"
        self.assertEqual(tp.done_pairs(rows, "std-vs-c256", "conc", "m3"), [])
        self.assertEqual(tp.pairs_of(rows, "std-vs-c256", "conc"), [])

    def test_compile_at_start_exits_3(self):
        fk = Fakes(self.tmp, builds=["31337"])
        self.assertEqual(fk.run(pairs=1), 3)
        self.assertEqual(fk.calls(), [])

    def test_compile_during_block_voids_pair(self):
        fk = Fakes(self.tmp, build_during=True)
        self.assertEqual(fk.run(pairs=1), 3)  # voided, then the repeat sees the compile and gives the GPU back
        rows = fk.rows()
        self.assertEqual(rows[0]["void"], "a compile or test started during the block")
        self.assertEqual(tp.done_pairs(rows, "std-vs-c256", "conc", "m3"), [])

    def test_fault_exits_4(self):
        fk = Fakes(self.tmp, mode={"256": "fault"})
        self.assertEqual(fk.run(pairs=3), 4)
        bad = [r for r in fk.rows() if r["arm"] == "c256"]
        self.assertEqual((bad[-1]["exit"], bad[-1]["note"], bad[-1]["void"]), ("3", "fault", ""))

    def test_candidate_failure_is_not_voided(self):
        fk = Fakes(self.tmp, mode={"256": "fail"})
        self.assertEqual(fk.run(pairs=1), 0)
        bad = [r for r in fk.rows() if r["arm"] == "c256"][0]
        self.assertEqual((bad["exit"], bad["failures"], bad["void"]), ("1", "1", ""))
        self.assertEqual(verdict_of(analysis(fk.out / "blocks.csv"), "std-vs-c256", "conc"), "rejected")

    def test_timeout_is_a_failure(self):
        fk = Fakes(self.tmp, mode={"256": "hang"})
        fk.run("--timeout", "1", pairs=1)
        bad = [r for r in fk.rows() if r["arm"] == "c256"][0]
        self.assertEqual((bad["exit"], bad["void"]), ("timeout", ""))
        self.assertIn("killed", bad["note"])
        text = analysis(fk.out / "blocks.csv")
        self.assertEqual(verdict_of(text, "std-vs-c256", "conc"), "rejected")

    def test_hold_deadline_exits_3(self):
        fk = Fakes(self.tmp)
        self.assertEqual(fk.run("--hold-min", "0", pairs=2), 3)
        self.assertEqual(fk.calls(), [])

    def test_wait_retries_until_the_gpu_is_free(self):
        fk = Fakes(self.tmp, procs_for=2)  # busy for two checks, then free
        self.assertEqual(fk.run("--wait", "--retry-s", "0", pairs=1), 0)
        self.assertEqual(len(fk.rows()), 2)
        self.assertEqual(fk.run(pairs=1), 0)  # nothing left to do
        self.assertEqual(len(fk.calls()), 2)

    def test_lock_waits_for_the_holder(self):
        fk = Fakes(self.tmp)
        path = self.tmp / "gpu.lock"
        holder = open(path, "a")
        fcntl.flock(holder, fcntl.LOCK_EX)
        threading.Timer(1.0, holder.close).start()  # closing releases it
        t0 = time.time()
        self.assertEqual(fk.run("--lock", str(path), pairs=1), 0)
        self.assertGreaterEqual(time.time() - t0, 0.9)
        self.assertEqual(len(fk.rows()), 2)

    def test_two_models_share_one_csv(self):
        fk = Fakes(self.tmp)
        fk.run(pairs=2, model="m3")
        fk.run(pairs=2, model="m5")
        rows = fk.rows()
        self.assertEqual(len(rows), 8)
        for m in ("m3", "m5"):
            self.assertEqual(tp.done_pairs(rows, "std-vs-c256", "conc", m), [0, 1])
            self.assertEqual(len(tp.pairs_of(rows, "std-vs-c256", "conc", m)), 2)
        text = analysis(fk.out / "blocks.csv")
        self.assertIn("| m5 | std-vs-c256 | conc | 2 |", text)

    def test_without_nvidia_smi(self):
        # A Metal host: no telemetry columns, no temperature band, other engines found by name.
        fk = Fakes(self.tmp, temp=95)
        with mock.patch.object(tp, "nvidia", return_value=False):
            self.assertEqual(fk.run(pairs=1), 0)
            rows = fk.rows()
            self.assertEqual([(r["void"], r["temp_start"], r["temp_max"], r["clock_med"]) for r in rows],
                             [("", "", "", "")] * 2)
            fk2 = Fakes(self.tmp / "b", engines=["555"])
            self.assertEqual(fk2.run(pairs=1), 3)
            self.assertEqual(fk2.calls(), [])
        fk3 = Fakes(self.tmp / "c", engines=["555"])  # with nvidia-smi its process list decides, not names
        self.assertEqual(fk3.run(pairs=1), 0)


class Patterns(unittest.TestCase):
    def test_process_names_match_whole_and_literal(self):
        import re
        pat = re.compile(tp.ere(tp.BUILD_PROCS))
        self.assertTrue(pat.fullmatch("clang++"))
        self.assertTrue(pat.fullmatch("cc1plus"))
        self.assertFalse(pat.fullmatch("clangg"))
        self.assertTrue(re.compile(tp.ere(tp.ENGINE_PROCS)).fullmatch("LM Studio"))


class Power(unittest.TestCase):
    def test_pmset(self):
        self.assertTrue(tp.pmset_on_ac("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged;"))
        self.assertFalse(tp.pmset_on_ac("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%;"))
        self.assertTrue(tp.pmset_on_ac(""))  # a Mac without a battery prints nothing useful

    def test_sysfs(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t)
            self.assertTrue(tp.sysfs_on_ac(root / "missing"))
            self.assertTrue(tp.sysfs_on_ac(root))  # no supplies: a desktop
            bat = root / "BAT0"
            bat.mkdir()
            (bat / "type").write_text("Battery\n")
            (bat / "status").write_text("Discharging\n")
            self.assertFalse(tp.sysfs_on_ac(root))
            ac = root / "AC"
            ac.mkdir()
            (ac / "type").write_text("Mains\n")
            (ac / "online").write_text("1\n")
            self.assertTrue(tp.sysfs_on_ac(root))
            (ac / "online").write_text("0\n")
            self.assertFalse(tp.sysfs_on_ac(root))


def arm(model, comp, verdicts, harm=(), cuts=None):
    """Synthetic analysis info for one arm: verdict per type, the types with harm, and stall cuts {type: (ms, base)}."""
    cuts = cuts or {}
    out = {}
    for t in tp.TYPES:
        g = {"hl": cuts[t][0], "base": cuts[t][1]} if t in cuts else None
        out[(model, comp, t)] = {"verdict": verdicts.get(t, "no harm"), "harm": t in harm, "gain": g}
    return out


class Gate(unittest.TestCase):
    def summary(self, info):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tp.calibration(info)
        return buf.getvalue()

    def test_this_pair_shape(self):
        # Always-256 cuts more but harms conc; the winner is milder and safe: counts toward build, fallback nothing.
        info = {**arm("m3", "std-vs-c256", {"conc": "rejected", "doc": "confirmed"}, ("conc",),
                      {"conc": (406, 992), "doc": (632, 1443)}),
                **arm("m3", "std-vs-c384", {"conc": "pending", "doc": "confirmed"}, (),
                      {"conc": (124, 994), "doc": (304, 1449)}),
                **arm("m3", "std-vs-cal", {"conc": "rejected", "doc": "confirmed"}, (),
                      {"conc": (141, 1003), "doc": (321, 1455)})}
        text = self.summary(info)
        self.assertIn("Winner (largest mean % stall cut over conc and doc among candidates that pass every test): "
                      "**std-vs-c384**", text)
        self.assertIn("avoids an always-256 no-harm failure: yes", text)
        self.assertIn("counts toward **build calibration**; fallback for this pair: **ship nothing**", text)

    def test_unproven_256_is_no_fallback(self):
        # Always-256 has too few pairs to show no harm on short: it is not the fallback to ship.
        al = arm("m1", "std-vs-c256", {"conc": "confirmed", "short": "insufficient"}, (),
                 {"conc": (400, 1000), "doc": (600, 1400)})
        al[("m1", "std-vs-c256", "short")]["unknown"] = True
        info = {**al, **arm("m1", "std-vs-cal", {"conc": "confirmed"}, (), {"conc": (420, 1000), "doc": (610, 1400)})}
        text = self.summary(info)
        self.assertIn("always-256 is not shown to do no harm on every type", text)
        self.assertIn("fallback for this pair: **ship nothing**", text)

    def test_safe_256_wins_the_pair(self):
        # Always-256 passes and cuts more: calibration buys nothing here, fallback ship always-256.
        info = {**arm("m1", "std-vs-c256", {"conc": "confirmed"}, (), {"conc": (400, 1000), "doc": (600, 1400)}),
                **arm("m1", "std-vs-cal", {"conc": "confirmed"}, (), {"conc": (420, 1000), "doc": (610, 1400)})}
        text = self.summary(info)
        self.assertIn("calibrated cut ≥ 1.10 × always-256's on: none", text)
        self.assertIn("counts toward **not building calibration**; fallback for this pair: **ship always-256**", text)

    def test_bigger_cut_counts(self):
        info = {**arm("m5", "std-vs-c256", {"conc": "confirmed"}, (), {"conc": (200, 1000), "doc": (300, 1400)}),
                **arm("m5", "std-vs-cal", {"conc": "confirmed"}, (), {"conc": (300, 1000), "doc": (310, 1400)})}
        text = self.summary(info)
        self.assertIn("calibrated cut ≥ 1.10 × always-256's on: conc", text)
        self.assertIn("**build calibration**", text)

    def test_harmful_calibrated_never_counts(self):
        info = {**arm("m5", "std-vs-c256", {"conc": "rejected"}, ("conc",), {"conc": (200, 1000)}),
                **arm("m5", "std-vs-cal", {"conc": "confirmed"}, ("short",), {"conc": (400, 1000)})}
        self.assertIn("**not building calibration**", self.summary(info))

    def test_no_candidate_passes(self):
        info = arm("m3", "std-vs-c128", {"conc": "rejected", "doc": "rejected"}, ("conc", "doc"),
                   {"conc": (660, 994), "doc": (1017, 1450)})
        text = self.summary(info)
        self.assertIn("none: no candidate passes every test", text)
        self.assertIn("Gate: needs both", text)


class Analysis(unittest.TestCase):
    """Whole experiments on the fakes, through `analyze`."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp))

    def test_real_effect_confirms_and_aa_does_not(self):
        fk = Fakes(self.tmp)
        fk.run(types="short,conc,doc", pairs=9)
        fk.run(types="short,conc,doc", pairs=10, b="std2")
        text = analysis(fk.out / "blocks.csv")
        self.assertEqual(verdict_of(text, "std-vs-c256", "conc"), "confirmed")
        self.assertEqual(verdict_of(text, "std-vs-c256", "doc"), "confirmed")
        self.assertEqual(verdict_of(text, "std-vs-c256", "short"), "no harm")
        self.assertIn(verdict_of(text, "std-vs-std2", "conc"), ("rejected", "insufficient", "pending"))
        self.assertIn("A/A false confirms: 0 of", text)
        self.assertIn("## A/A spread", text)
        self.assertIn("## Order", text)
        self.assertIn("## Time per block", text)

    def test_throughput_harm_rejects(self):
        fk = Fakes(self.tmp, tok_penalty={"256": 0.9})
        fk.run(types="conc,short", pairs=9)
        text = analysis(fk.out / "blocks.csv")
        self.assertEqual(verdict_of(text, "std-vs-c256", "conc"), "rejected")
        self.assertEqual(verdict_of(text, "std-vs-c256", "short"), "rejected")
        self.assertIn("FAIL", text)

    def test_no_effect_is_not_confirmed(self):
        fk = Fakes(self.tmp, effect=False, noise=0.15)
        fk.run(types="conc", pairs=18)
        self.assertEqual(verdict_of(analysis(fk.out / "blocks.csv"), "std-vs-c256", "conc"), "rejected")

    def test_voided_pairs_are_left_out(self):
        fk = Fakes(self.tmp)
        fk.run(types="conc", pairs=9)
        rows = fk.rows()
        rows[0]["void"] = "test"
        with (fk.out / "edited.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, tp.COLS)
            w.writeheader()
            w.writerows(rows)
        self.assertEqual(len(tp.pairs_of(rows, "std-vs-c256", "conc")), 8)
        self.assertEqual(verdict_of(analysis(fk.out / "edited.csv"), "std-vs-c256", "conc"), "insufficient")

    def test_an_unpaired_candidate_failure_still_rejects(self):
        # A candidate that failed before its reference ran (the run stopped, then resumed): its pair never
        # completes, but the failure stays against the arm.
        fk = Fakes(self.tmp)
        fk.run(types="conc", pairs=9)
        rows = fk.rows()
        cand = next(r for r in rows if r["arm"] != "std")
        lone = dict(cand, pair="99", attempt="0", failures="1", exit="1")
        self.assertEqual(tp.cand_failures(rows + [lone], "std-vs-c256", "conc"), 1)
        with (fk.out / "edited.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, tp.COLS)
            w.writeheader()
            w.writerows(rows + [lone])
        self.assertEqual(verdict_of(analysis(fk.out / "edited.csv"), "std-vs-c256", "conc"), "rejected")

    def edited(self, fk, change):
        """The fakes' blocks with `change` applied to each row, as a CSV for `analysis`."""
        rows = fk.rows()
        for r in rows:
            change(r)
        with (fk.out / "edited.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, tp.COLS)
            w.writeheader()
            w.writerows(rows)
        return fk.out / "edited.csv"

    def test_too_few_pairs_for_no_harm_choose_no_winner(self):
        # One usable short pair and one long: no bound, so neither type shows no harm, and nothing passes.
        fk = Fakes(self.tmp)
        fk.run(types="short,long,conc,doc", pairs=9)
        first = min(int(r["pair"]) for r in fk.rows())
        text = analysis(self.edited(fk, lambda r: r.update(
            void="test" if r["type"] in tp.SINGLE and int(r["pair"]) != first else r["void"])))
        self.assertEqual(verdict_of(text, "std-vs-c256", "short"), "insufficient")
        self.assertEqual(verdict_of(text, "std-vs-c256", "long"), "insufficient")
        self.assertIn("no: too few pairs to show no harm on short,long", text)
        self.assertIn("none: no candidate passes every test", text)

    def test_a_failed_aa_block_is_counted_not_a_crash(self):
        # The last A/A pair's second block timed out: no metrics, not voided.
        fk = Fakes(self.tmp)
        fk.run(types="conc", pairs=9, b="std2")
        last = max(int(r["pair"]) for r in fk.rows())
        metrics = ("gap_max", "gap_max_t", "gap_p99", "gap_p50", "tok_s", "ttft_p50", "ttft_p99", "step_max_ms")

        def timeout(r):
            if r["arm"] == "std2" and int(r["pair"]) == last:
                r.update({m: "" for m in metrics}, exit="124", failures="1")

        text = analysis(self.edited(fk, timeout))
        self.assertIn("A/A false confirms: 0 of 0 9-pair windows (sliding, both labellings); 2 more held a failed "
                      "block and were not tested.", text)


class Inputs(unittest.TestCase):
    def test_doc_workload(self):
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "doc.jsonl"
            tp.doc_workload(p)
            convs = [json.loads(line) for line in p.read_text().splitlines()]
        w2 = [json.loads(line) for line in (tp.ROOT / "bench/prompts/w2-chats.jsonl").read_text().splitlines()]
        w5 = {d["id"]: d for d in map(json.loads, (tp.ROOT / "bench/prompts/w5-longprompt.jsonl").read_text().splitlines())}
        self.assertEqual(convs[0], w2[0])
        doc = convs[1]["messages"]
        self.assertEqual([m["role"] for m in doc], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(doc[2]["max_tokens"], 32)
        self.assertEqual(doc[2]["content"], w2[1]["messages"][2]["content"])  # history keeps the scripted answer
        self.assertEqual(doc[3:], w5["w5-4096"]["messages"])
        # The document arrives after a short turn, while conversation 0's first answer (hundreds of tokens) runs.
        self.assertLess(doc[2]["max_tokens"], convs[0]["messages"][2]["max_tokens"])

    def test_trimmed_workloads(self):
        with tempfile.TemporaryDirectory() as t:
            tp.write_workloads(pathlib.Path(t))
            short, long_ = ([json.loads(x) for x in (pathlib.Path(t) / f).read_text().splitlines()]
                            for f in ("short.jsonl", "long.jsonl"))
        for got, src, n in ((short, "w1-chat.jsonl", tp.SHORT_TURNS), (long_, "w3-agent.jsonl", tp.LONG_TURNS)):
            self.assertEqual(len(got), 1)
            msgs, whole = got[0]["messages"], tp.frozen(src)[0]["messages"]
            self.assertEqual(msgs, whole[:len(msgs)])  # a prefix of the frozen conversation
            self.assertEqual(sum(m["role"] == "assistant" for m in msgs), n)
            self.assertEqual(msgs[-1]["role"], "assistant")
        self.assertEqual(long_[0]["messages"][0], tp.frozen("w3-agent.jsonl")[0]["messages"][0])  # the long prefix

    def test_block_metrics_without_gaps(self):
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "b.json"
            p.write_text(json.dumps({"summary": {"wall_s": 1.0, "agg_tok_s": 9.0, "ttft_ms": None, "failures": 0,
                                                 "gen_tokens": 1, "evaluated_tokens": 1},
                                     "requests": [{"error": None, "t_first_ms": 1.0}]}))
            m = tp.block_metrics(p)
        self.assertEqual((m["gap_max"], m["gap_max_t"], m["ttft_p50"], m["step_max_ms"]), (None, None, None, None))

    def test_every_type_maps_to_a_bench_workload(self):
        self.assertEqual(set(tp.TYPES), {"short", "long", "conc", "doc"})
        self.assertEqual(set(tp.CONCURRENT) | set(tp.SINGLE), set(tp.TYPES))
        src = (tp.ROOT / "crates/larch/src/bench.rs").read_text()
        for workload, _, _ in tp.TYPES.values():
            self.assertIn("    " + workload.capitalize() + ",", src)


if __name__ == "__main__":
    unittest.main()
