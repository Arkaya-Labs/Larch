"""Tests for scripts/bench-merge's pure logic: python3 -m unittest scripts/test_bench_merge.py -v"""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
# bench-merge has no .py extension, so the default finders would not recognise it.
loader = importlib.machinery.SourceFileLoader("bench_merge", str(HERE / "bench-merge"))
bm = importlib.util.module_from_spec(importlib.util.spec_from_loader("bench_merge", loader))
loader.exec_module(bm)

RUN = {"summary": {"agg_tok_s": 100.0, "ttft_ms": {"p99": 200.0}, "tpot_ms": {"p99": 20.0}}}
FIXED = {"fixed": [{"pl": 4, "speed_tg": 90.0, "speed": 300.0}, {"pl": 1, "speed_tg": 50.0, "speed": 60.0},
                   {"pl": 8, "speed_tg": 95.0, "speed": 400.0}]}
# agg_tok_s_8 is the fixed shape's aggregate at 8 (architecture 16); serve_tok_s_8 the W4 serve one.
BASE = {"commit": "a" * 40, "machine": "win5070", "model": "m1", "agg_tok_s_8": 400.0, "ttft_p99_ms": 200.0,
        "tpot_p99_ms": 20.0, "single_tok_s": 50.0, "serve_tok_s_8": 100.0}
METRICS = (("agg_tok_s_8", 0.97, 0.969), ("single_tok_s", 0.97, 0.969), ("serve_tok_s_8", 0.97, 0.969),
           ("ttft_p99_ms", 1.03, 1.031), ("tpot_p99_ms", 1.03, 1.031))


def with_(**kw):
    return {**BASE, **kw}


class Row(unittest.TestCase):
    def test_extracts_the_five_numbers(self):
        self.assertEqual(bm.row("a" * 40, "win5070", "m1", RUN, FIXED), BASE)

    def test_missing_pl1_row_refuses(self):
        with self.assertRaises(SystemExit):
            bm.row("c", "win5070", "m1", RUN, {"fixed": [r for r in FIXED["fixed"] if r["pl"] != 1]})

    def test_missing_pl8_row_refuses(self):
        with self.assertRaises(SystemExit):
            bm.row("c", "win5070", "m1", RUN, {"fixed": [r for r in FIXED["fixed"] if r["pl"] != 8]})

    def test_null_or_missing_number_refuses(self):
        for run in ({"summary": {**RUN["summary"], "ttft_ms": None}},
                    {"summary": {**RUN["summary"], "agg_tok_s": None}},
                    {"summary": {"agg_tok_s": 1.0, "ttft_ms": {"p99": 1.0}}},
                    {}):
            with self.assertRaises(SystemExit):
                bm.row("c", "win5070", "m1", run, FIXED)
        with self.assertRaises(SystemExit):
            bm.row("c", "win5070", "m1", RUN, {"fixed": [{**r, "speed_tg": None} for r in FIXED["fixed"]]})

    def test_a_row_with_an_eager_step_is_invalid_not_compared(self):
        # A graph-capable load with any decode step run eager on a refused capture, however many replayed.
        for pl in (1, 8):
            with self.subTest(pl=pl):
                eager = {"fixed": [{**r, "graph_replays": 400, "eager_decode_steps": 1} if r["pl"] == pl else r
                                   for r in FIXED["fixed"]]}
                with self.assertRaises(SystemExit) as e:
                    bm.row("c", "win5070", "m1", RUN, eager)
                self.assertIn(f"invalid: fixed pl={pl}", str(e.exception.code))
        # Every step replayed, and the CPU's none of either, stay valid.
        ok = {"fixed": [{**r, "graph_replays": 4, "eager_decode_steps": 0} for r in FIXED["fixed"]]}
        self.assertEqual(bm.row("a" * 40, "win5070", "m1", RUN, ok), BASE)
        cpu = {"fixed": [{**r, "graph_replays": 0, "eager_decode_steps": 0} for r in FIXED["fixed"]]}
        self.assertEqual(bm.row("a" * 40, "win5070", "m1", RUN, cpu), BASE)


class Losses(unittest.TestCase):
    def test_each_metric_at_3_percent_is_not_a_loss_and_at_3_1_is(self):
        for key, worse, _ in METRICS:
            with self.subTest(key=key):
                self.assertEqual(bm.losses(BASE, with_(**{key: BASE[key] * worse})), [])
        for key, _, worse in METRICS:
            with self.subTest(key=key):
                got = bm.losses(BASE, with_(**{key: BASE[key] * worse}))
                self.assertEqual(len(got), 1)
                self.assertIn(key, got[0])

    def test_a_batched_aggregate_regression_is_a_loss(self):
        # The review's reproduction: W6 at 8 down 99 %, everything else unchanged.
        slow = {"fixed": [{**r, "speed": r["speed"] / 100} if r["pl"] == 8 else r for r in FIXED["fixed"]]}
        got = bm.losses(BASE, bm.row("b" * 40, "win5070", "m1", RUN, slow))
        self.assertEqual(len(got), 1)
        self.assertIn("agg_tok_s_8", got[0])

    def test_gains_are_never_losses(self):
        self.assertEqual(bm.losses(BASE, with_(agg_tok_s_8=800.0, single_tok_s=90.0, ttft_p99_ms=10.0, tpot_p99_ms=1.0, serve_tok_s_8=200.0)), [])

    def test_prev_read_back_from_csv_strings(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "history.csv"
            bm.append(path, BASE)
            self.assertEqual(bm.losses(bm.last(path, "win5070", "m1", "b" * 40), with_(agg_tok_s_8=387.6)),
                             ["agg_tok_s_8 400.0 -> 387.6 (-3.1%)"])


class History(unittest.TestCase):
    def test_append_writes_the_header_once_and_never_rewrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "history.csv"
            bm.append(path, BASE)
            first = path.read_text()
            bm.append(path, with_(commit="b" * 40))
            text = path.read_text()
            self.assertTrue(text.startswith(first))
            lines = text.splitlines()
            self.assertEqual(lines[0], "commit,machine,model,agg_tok_s_8,ttft_p99_ms,tpot_p99_ms,single_tok_s,serve_tok_s_8")
            self.assertEqual(len(lines), 3)
            self.assertEqual(sum(l.startswith("commit,") for l in lines), 1)

    def test_other_machines_and_models_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "history.csv"
            self.assertIsNone(bm.last(path, "win5070", "m1", "9"))  # no file yet
            bm.append(path, with_(commit="1"))
            bm.append(path, with_(commit="2", machine="win4060"))
            bm.append(path, with_(commit="3", model="m5"))
            self.assertEqual(bm.last(path, "win5070", "m1", "9")["commit"], "1")
            self.assertIsNone(bm.last(path, "wsl5070", "m1", "9"))
            bm.append(path, with_(commit="4"))
            self.assertEqual(bm.last(path, "win5070", "m1", "9")["commit"], "4")

    def test_one_line_per_merge(self):
        # A rerun of a merged commit would add a second line and move the baseline the next merge is compared with.
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "history.csv"
            bm.append(path, with_(commit="1"))
            with self.assertRaises(SystemExit) as e:
                bm.last(path, "win5070", "m1", "1")
            self.assertIn("already", str(e.exception))
            # The same commit on another machine or model is its own line.
            self.assertIsNone(bm.last(path, "wsl5070", "m1", "1"))
            self.assertIsNone(bm.last(path, "win5070", "m5", "1"))


class Built(unittest.TestCase):
    @staticmethod
    def msg(name, kind, exe, reason="compiler-artifact"):
        return json.dumps({"reason": reason, "target": {"name": name, "kind": kind}, "executable": exe})

    def test_the_binary_cargo_reports_not_target_release(self):
        # CARGO_TARGET_DIR elsewhere: target/release/larch may exist, but it is not this build.
        out = "\n".join([self.msg("larch", ["lib"], None), self.msg("larch_engine", ["lib"], None),
                         self.msg("supervise", ["bin"], "/elsewhere/release/supervise"),
                         self.msg("larch", ["bin"], "/elsewhere/release/larch"),
                         json.dumps({"reason": "build-finished", "success": True})])
        self.assertEqual(bm.built(out), pathlib.Path("/elsewhere/release/larch"))

    def test_no_reported_binary_refuses(self):
        for out in ("", self.msg("larch", ["lib"], None)):
            with self.assertRaises(SystemExit):
                bm.built(out)


class Outputs(unittest.TestCase):
    def test_merge_results_never_replace_published_rows(self):
        # The attended run publishes larch-w4-8 / larch-fixed, and llama.cpp's row is w4-8.
        for name in (bm.SERVED, bm.FIXED):
            self.assertFalse(name.startswith("larch-") or name == bm.W4_8[0], name)
        self.assertNotEqual(bm.SERVED, bm.FIXED)


class Machines(unittest.TestCase):
    def test_only_machines_with_a_llama_w4_8_row_take_history(self):
        for m in ("win5070", "wsl5070", "win4060"):
            bm.check_machine(m)
        for m in ("mac", "sm86", "cpu"):
            with self.assertRaises(SystemExit):
                bm.check_machine(m)


REF = b'{"larch_oracle":1}\n'
ENTRY = {"bound": 0.05, "margin": 0.7, "reference": {"dump_sha256": hashlib.sha256(REF).hexdigest()}}


class Bounds(unittest.TestCase):
    def bounds(self, alias, doc, ref=REF):
        with tempfile.TemporaryDirectory() as d:
            path, dump = pathlib.Path(d) / "oracle-bounds.json", pathlib.Path(d) / "ref.jsonl"
            path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
            dump.write_bytes(ref)
            return bm.bounds(alias, dump, path)

    def test_bounds_read_for_alias(self):
        # larch parity needs both; bench-merge takes them from the calibration entry, never a default.
        self.assertEqual(self.bounds("m1", {"m1": ENTRY}), (0.05, 0.7))

    def test_a_ref_other_than_the_calibrated_reference_refuses(self):
        # A CPU or ub512 dump of the same model would gate against a bound calibrated for another reference.
        with self.assertRaises(SystemExit) as e:
            self.bounds("m1", {"m1": ENTRY}, ref=b'{"larch_oracle":1,"backend":"CPU"}\n')
        self.assertIn("reference", str(e.exception))

    def test_missing_or_malformed_entry_refuses(self):
        bad_numbers = [{**ENTRY, k: v} for k, v in (("bound", float("nan")), ("bound", 0), ("margin", -1.0), ("bound", True),
                                                    ("margin", "0.7"), ("bound", float("inf")))]
        bad_numbers += [{**ENTRY, "reference": {"dump_sha256": v}} for v in (None, 1, "")]
        for doc in (*({"m1": e} for e in bad_numbers), {"m2": ENTRY}, {"m1": {"bound": 0.05, "margin": 0.7}}, {"m1": {"pairs": []}}, {"m1": [1]}, [1], "not json"):
            with self.assertRaises(SystemExit, msg=doc) as e:
                self.bounds("m1", doc)
            self.assertIn("oracle-bounds.json", str(e.exception))

    def test_the_repo_file_has_every_bench_merge_model(self):
        # Every model of a machine bench-merge runs on (one with llama.cpp's W4-at-8 row).
        aliases = {m for spec in bm.baseline.MACHINES.values() if bm.W4_8 in spec["rows"] for m in spec["models"]}
        self.assertTrue(aliases)
        for alias in sorted(aliases):
            entry = json.loads((HERE.parent / "bench" / "oracle-bounds.json").read_text())[alias]
            self.assertGreater(entry["bound"], 0)
            self.assertGreaterEqual(entry["margin"], 0)
            self.assertRegex(entry["reference"]["dump_sha256"], "^[0-9a-f]{64}$")


class Tree(unittest.TestCase):
    def test_only_results_and_bytecode_may_be_untracked(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            # No global config: a signing key or hook of the developer's must not decide this test.
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
            git = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True, env=env)  # noqa: E731
            git("init", "-q")
            (root / "a.txt").write_text("a")
            git("add", "a.txt")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
            for f in ("scripts/__pycache__/baseline.cpython-314.pyc", "bench/results/x.json"):
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_text("x")
            self.assertFalse(bm.dirty(root))
            (root / "scripts" / "new.py").write_text("x")
            self.assertTrue(bm.dirty(root))

    def test_committed_results_are_never_edited(self):
        # Untracked output is the run's own; an edit to a committed result (history, a baseline row) is not.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
            git = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True, env=env)  # noqa: E731
            git("init", "-q")
            (root / "bench" / "results").mkdir(parents=True)
            (root / "bench" / "results" / "history.csv").write_text("commit\n")
            git("add", ".")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
            (root / "bench" / "results" / "new.json").write_text("x")
            self.assertFalse(bm.dirty(root))
            (root / "bench" / "results" / "history.csv").write_text("commit\nedited\n")
            self.assertTrue(bm.dirty(root))
            git("checkout", "--", "bench/results/history.csv")
            (root / "bench" / "results" / "history.csv").unlink()
            self.assertTrue(bm.dirty(root))

if __name__ == "__main__":
    unittest.main()
