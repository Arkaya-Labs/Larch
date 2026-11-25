"""Tests for scripts/scoreboard: python3 -m unittest scripts/test_scoreboard.py -v"""

import importlib.machinery
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
# scoreboard has no .py extension, so the default finders would not recognise it.
loader = importlib.machinery.SourceFileLoader("scoreboard", str(HERE / "scoreboard"))
sb = importlib.util.module_from_spec(importlib.util.spec_from_loader("scoreboard", loader))
loader.exec_module(sb)

STATS = {"p50": 10.0, "p99": 50.0, "mean": 20.0, "max": 60.0}


REUSE = {"w1-chat.jsonl", "w2-chats.jsonl", "w3-agent.jsonl"}


def sha(model, wl):
    """The workload's bytes as the model's manifest records them, when the repo has that manifest."""
    path = sb.baseline.BENCH / "manifest" / f"{model.removesuffix('.gguf')}.jsonl"
    if not path.exists():
        return "x"
    return next(r["workload_sha256"] for r in sb.baseline.manifest(model.removesuffix(".gguf")) if r["workload"] == wl)


def run(engine, wl, n, agg, model="m1.gguf", failures=0, cache=None):
    """A run of one of wsl5070's rows; `cache` other than the row's makes it none of them. larch's records its binary."""
    return {"schema": "larch-bench/1", "engine": engine, "model": model, "mode": "run", "concurrency": n, "machine": "wsl5070",
            **({"build": "a" * 64} if engine == "larch" else {}),
            "workload": {"file": wl, "sha256": sha(model, wl)}, "cache_prompt": wl in REUSE if cache is None else cache,
            "params": {"n_ctx_seq": 2560}, "requests": [],
            "summary": {"agg_tok_s": agg, "failures": failures, "requests": 10, "ttft_ms": STATS, "tpot_ms": STATS}}


def fixed(engine, pp, rows):
    return {"schema": "larch-bench/1", "engine": engine, "model": "m2.gguf", "mode": "fixed", "machine": "wsl5070",
            "params": {"n_ctx_seq": 1024}, "fixed": [{"pp": pp, "tg": 128, "pl": pl, "speed": s} for pl, s in rows]}


def full(engine, agg):
    """A W4 at 8 run of m1 with every request the manifest expects, at its exact length: passes its own checks."""
    m = [r for r in sb.baseline.manifest("m1") if r["workload"] == "chat.jsonl"]
    reqs = [{"conv": r["conv"], "turn": r["turn"], "prompt_sha256": r["prompt_sha256"], "max_tokens": r["max_tokens"],
             "gen_tokens": r["max_tokens"], "error": None} for r in m]
    d = run(engine, "chat.jsonl", 8, agg)
    return {**d, "workload": {"file": "chat.jsonl", "sha256": m[0]["workload_sha256"]}, "requests": reqs,
            "summary": {**d["summary"], "requests": len(reqs)}}


def gen(wl, n, failures=0):
    """larch's --replay generated run of m1's row: the session gate scripted W1 to W3 need first. Every request the
    manifest expects, each finding exactly the KV it was promised; `failures` of them fail."""
    row = next(r for r in sb.baseline.W1 + sb.baseline.W2 + sb.baseline.W3 if r[1:3] == (wl, n))
    reqs = [{"conv": c, "turn": t, "error": None, "cached_tokens": 7 * t, "kv_tokens": 7 * t, "gen_tokens": 1}
            for c, t in sb.baseline.expected_requests("m1", row)]
    for r in reqs[:failures]:
        r["error"] = "Internal(1)"
    d = run("larch", wl, n, 1.0, failures=failures)
    return {**d, "replay": "generated", "requests": reqs, "summary": {**d["summary"], "requests": len(reqs)},
            "params": {"n_ctx_seq": sb.baseline.row_ctx(row, sb.baseline.peaks("m1"))}}


def batched(rows):
    """llama-batched-bench's JSONL, the W6 baseline, as baseline.py run writes it."""
    return "\n".join(json.dumps({"pp": 512, "tg": 128, "pl": pl, "speed": s}) for pl, s in rows) + "\n"


class Scoreboard(unittest.TestCase):
    def setUp(self):
        # Like-for-like and the driver's fixed-shape gate are baseline.py's calls (compare_problems, check_fixed),
        # tested there; these fixtures have no manifest or flag search.
        for name, fake in (("compare_problems", lambda files, runs: []), ("check_fixed", lambda a, alias: [])):
            patch = mock.patch.object(sb.baseline, name, fake)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(sb, "server_problems", lambda machine, out, alias: [])
        patch.start()
        self.addCleanup(patch.stop)

    def report(self, files):
        """Writes the named JSON values (a str is written as is) into a temp dir; the report over all of them."""
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, doc in files.items():
                p = pathlib.Path(d) / name
                p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
                if name.endswith(".json"):
                    paths.append(p)
            return sb.report(paths)

    def test_all_three_engines(self):
        out = self.report({"a.json": run("larch", "w2-chats.jsonl", 8, 110.0),
                           "b.json": run("llama.cpp", "w2-chats.jsonl", 8, 100.0),
                           "c.json": run("ollama", "w2-chats.jsonl", 8, 125.0),
                           "g.json": gen("w2-chats.jsonl", 8)})
        self.assertIn("m1", out)
        row = next(l for l in out.splitlines() if l.startswith("W2"))
        self.assertIn("+10.0%", row)
        self.assertIn("-12.0%", row)

    def test_missing_engine_column(self):
        out = self.report({"a.json": run("larch", "chat.jsonl", 4, 90.0),
                           "b.json": run("llama.cpp", "chat.jsonl", 4, 100.0)})
        row = next(l for l in out.splitlines() if l.startswith("W4"))
        self.assertIn("-10.0%", row)
        self.assertEqual(row.split()[-1], "-")

    def test_failed_run_is_flagged(self):
        out = self.report({"a.json": run("larch", "chat.jsonl", 1, 90.0, failures=3)})
        self.assertIn("90.0!", out)
        self.assertIn("! a.json: 3 failed requests", out)

    def test_foreign_json_skipped(self):
        out = self.report({"flags.json": {"ngl": 99}, "list.json": [1, 2],
                           "a.json": run("larch", "w1-chat.jsonl", 1, 90.0)})
        self.assertIn("W1", out)

    def test_fixed_mode_rows(self):
        rows = [l for l in self.report({"f.json": fixed("larch", 512, [(1, 60.0), (8, 400.0)])}).splitlines()
                if l.startswith("W6")]
        self.assertEqual(len(rows), 2)
        self.assertIn("400.0", rows[1])

    def test_only_w6_shapes_on_fixed_machines_are_w6_rows(self):
        # W6 is pp512/tg128 at 1, 4, 8, 16 on a machine that runs the fixed shape; nothing else is labelled W6.
        out = self.report({"a.json": fixed("larch", 2048, [(8, 300.0)]), "wsl5070-m2-fixed.json": fixed("llama.cpp", 512, []),
                           "wsl5070-m2-batched.jsonl": batched([(8, 400.0)]),
                           "s.json": {**fixed("larch", 512, [(4, 222.2)]), "machine": "sm86"},
                           "n.json": fixed("larch", 512, [(2, 333.3)])})
        self.assertNotIn("%", out)
        self.assertNotIn("W6 pp2048", out)
        self.assertIn("W6 pp512", out)
        for n in ("300.0", "222.2", "333.3"):
            self.assertNotIn(n, out)
        # A context the shape does not fit is withheld.
        out = self.report({"v.json": {**fixed("larch", 512, [(8, 300.0)]), "params": {"n_ctx_seq": 512}}})
        self.assertNotIn("300.0", out)
        self.assertIn("n_ctx_seq", out)

    def test_unlike_runs_get_no_percent(self):
        with mock.patch.object(sb.baseline, "compare_problems", lambda files, runs: ["cache_prompt differs: True vs False"]):
            out = self.report({"a.json": run("larch", "chat.jsonl", 8, 110.0), "b.json": run("llama.cpp", "chat.jsonl", 8, 100.0)})
        self.assertNotIn("%", out)
        self.assertIn("cache_prompt differs", out)

    def test_duplicate_refused(self):
        with self.assertRaises(SystemExit) as e:
            self.report({"a.json": run("larch", "w1-chat.jsonl", 1, 90.0), "b.json": run("larch", "w1-chat.jsonl", 1, 91.0)})
        self.assertIn("a.json", str(e.exception))
        self.assertIn("b.json", str(e.exception))

    def test_generated_replay_skipped(self):
        # --replay generated is larch's session gate, not a comparison: never beside the scripted row.
        g = gen("w1-chat.jsonl", 1)
        gen_ = {**g, "summary": {**g["summary"], "agg_tok_s": 77.7}}
        out = self.report({"g.json": gen_, "a.json": {**run("larch", "w1-chat.jsonl", 1, 90.0), "replay": "scripted"}})
        self.assertIn("90.0", out)
        self.assertNotIn("77.7", out)

    def test_slice_parts_skipped(self):
        # A --slice part is a piece of a row; only scripts/bench-join's whole row is shown.
        out = self.report({"a.json": run("larch", "chat.jsonl", 8, 90.0),
                           "p.json": {**run("larch", "chat.jsonl", 8, 77.7), "slice": {"k": 0, "n": 2, "first": 0, "end": 100}}})
        self.assertIn("90.0", out)
        self.assertNotIn("77.7", out)

    def test_soak_and_cancel_runs_skipped(self):
        # R2/R12 soaks (--duration) and R3 (--cancel-rate) reuse the workload's key but are not its row.
        out = self.report({"a.json": run("larch", "chat.jsonl", 8, 90.0),
                           "s.json": {**run("larch", "chat.jsonl", 8, 77.7), "duration": 600.0},
                           "c.json": {**run("larch", "chat.jsonl", 8, 66.6), "cancel_rate": 0.02}})
        self.assertIn("90.0", out)
        self.assertNotIn("77.7", out)
        self.assertNotIn("66.6", out)

    def test_fixed_rows_from_two_machines_get_no_percent(self):
        out = self.report({"a.json": fixed("larch", 512, [(8, 300.0)]),
                           "win4060-m2-fixed.json": {**fixed("llama.cpp", 512, []), "machine": "win4060"},
                           "win4060-m2-batched.jsonl": batched([(8, 400.0)])})
        self.assertNotIn("%", out)
        # One table per machine and model: the two machines' rows never share a line.
        self.assertIn("m2 (wsl5070)", out)
        self.assertIn("m2 (win4060)", out)

    def test_default_files_leave_out_merge_rows(self):
        # bench-merge's merge-* rows repeat the attended larch W4 and W6 rows under the same key.
        with tempfile.TemporaryDirectory() as d:
            # And the server check's driver run, a consistency check taken under a shared GPU hold.
            for name in ("wsl5070-m1-w4-8.json", "wsl5070-m1-merge-w4-8.json", "win5070-m1-w4-8.json",
                         "wsl5070-m1-server-check-driver.json"):
                (pathlib.Path(d) / name).write_text("{}")
            self.assertEqual([p.name for p in sb.results(pathlib.Path(d), "wsl5070")], ["wsl5070-m1-w4-8.json"])

    def test_ollama_w4_refused_by_the_real_compare(self):
        # Ollama always reuses prefixes (cache_prompt true); W4 reuses nothing, so Ollama's W4 is no row of the machine.
        mock.patch.stopall()
        out = self.report({"a.json": full("larch", 110.0), "o.json": {**full("ollama", 100.0), "cache_prompt": True}})
        self.assertNotIn("%", out)
        self.assertNotIn("100.0", out)
        self.assertEqual(next(l for l in out.splitlines() if l.startswith("W4")).split()[4], "-")
        self.assertIn("110.0", out)
        # Notes name files, never local paths (the table goes into PR bodies), and stay short.
        self.assertNotIn(tempfile.gettempdir(), out)
        self.assertTrue(all(len(line) < 400 for line in out.splitlines()), out)

    def test_scripted_w1_to_w3_need_a_clean_generated_run(self):
        # architecture 16: --replay generated runs W1 to W3 clean before a scripted number is published.
        rows = {"a.json": run("larch", "w2-chats.jsonl", 8, 110.0), "b.json": run("llama.cpp", "w2-chats.jsonl", 8, 100.0)}
        for extra, ok in (({}, False), ({"g.json": gen("w2-chats.jsonl", 8, failures=1)}, False),
                          ({"g.json": gen("w2-chats.jsonl", 4)}, False), ({"g.json": gen("w2-chats.jsonl", 8)}, True)):
            out = self.report({**rows, **extra})
            self.assertEqual("+10.0%" in out, ok, (extra, out))
            # Withheld means the scripted number itself, not only its %.
            self.assertEqual("110.0" in out, ok, (extra, out))
            if not ok:
                self.assertIn("--replay generated", out)
        # Any failed generated run of the row withholds it, whichever file sorts last; a soak never counts.
        out = self.report({**rows, "g1.json": gen("w2-chats.jsonl", 8), "g2.json": gen("w2-chats.jsonl", 8, failures=2)})
        self.assertNotIn("110.0", out)
        out = self.report({**rows, "g.json": {**gen("w2-chats.jsonl", 8), "duration": 60.0}})
        self.assertNotIn("110.0", out)
        # A generated run of other workload bytes is not this row's gate.
        out = self.report({**rows, "g.json": {**gen("w2-chats.jsonl", 8), "workload": {"file": "w2-chats.jsonl", "sha256": "old"}}})
        self.assertNotIn("110.0", out)

    def test_w6_baseline_is_llama_batched_bench(self):
        # The driver's own fixed file is the mirror the gate checks, never the column.
        out = self.report({"wsl5070-m2-fixed.json": fixed("llama.cpp", 512, [(8, 999.0)]),
                           "wsl5070-m2-batched.jsonl": batched([(8, 400.0)]),
                           "v.json": fixed("larch", 512, [(8, 300.0)])})
        row = next(l for l in out.splitlines() if l.startswith("W6"))
        self.assertIn("400.0", row)
        self.assertNotIn("999.0", out)
        self.assertIn("-25.0%", row)

    def test_w6_cells_are_s_over_all_runs(self):
        # baseline.py run alternates llama-batched-bench with the driver FIXED_REPS times. A cell is its runs' tokens over
        # their time: runs at 300 and 600 t/s are 400, not the mean of the speeds (450); larch's 200 and 600 are 300.
        out = self.report({"wsl5070-m2-fixed.json": fixed("llama.cpp", 512, [(8, 999.0)]),
                           "wsl5070-m2-batched.jsonl": batched([(8, s) for s in (300.0, 600.0, 300.0, 600.0)]),
                           "v.json": fixed("larch", 512, [(8, 200.0), (8, 600.0)])})
        row = next(l for l in out.splitlines() if l.startswith("W6"))
        self.assertIn("400.0", row)
        self.assertNotIn("450.0", out)
        self.assertIn("-25.0%", row)

    def test_a_failed_mirror_gate_withholds_every_llama_percent(self):
        files = {"wsl5070-m2-fixed.json": fixed("llama.cpp", 512, [(8, 380.0)]),
                 "wsl5070-m2-batched.jsonl": batched([(8, 400.0)]), "v.json": fixed("larch", 512, [(8, 300.0)]),
                 "a.json": run("larch", "chat.jsonl", 8, 110.0, model="m2.gguf"),
                 "b.json": run("llama.cpp", "chat.jsonl", 8, 100.0, model="m2.gguf")}
        with mock.patch.object(sb.baseline, "check_fixed", lambda a, alias: ["fixed npl=8 off by -5.0%"]):
            out = self.report(files)
        self.assertNotIn("-25.0%", out)
        self.assertNotIn("+10.0%", out)
        # No number from the driver or llama-batched-bench is published, not only no %.
        self.assertNotIn("400.0", out)
        self.assertNotIn("100.0", out)
        self.assertIn("fixed npl=8 off by -5.0%", out)
        # Said once for the model, not once per row.
        self.assertEqual(out.count("fixed npl=8 off by -5.0%"), 1, out)

    def test_a_machine_without_the_fixed_shape_is_not_gated_on_it(self):
        # baseline.py's own policy: sm86 (the 3050) runs no fixed shape, and its check passes without one.
        mock.patch.stopall()
        mock.patch.object(sb.baseline, "compare_problems", lambda files, runs: []).start()
        mock.patch.object(sb, "server_problems", lambda machine, out, alias: []).start()
        self.addCleanup(mock.patch.stopall)
        out = self.report({"a.json": {**run("larch", "w5-longprompt.jsonl", 1, 110.0), "machine": "sm86"},
                           "sm86-m1-w5.json": {**run("llama.cpp", "w5-longprompt.jsonl", 1, 100.0), "machine": "sm86"}})
        self.assertIn("+10.0%", out)

    def test_an_unchecked_driver_publishes_nothing(self):
        # The real check_fixed over a directory without the fixed-shape files: a gate problem, never a pass or an exit.
        mock.patch.stopall()
        mock.patch.object(sb.baseline, "compare_problems", lambda files, runs: []).start()
        self.addCleanup(mock.patch.stopall)
        out = self.report({"a.json": run("larch", "chat.jsonl", 8, 110.0), "wsl5070-m1-w4-8.json": run("llama.cpp", "chat.jsonl", 8, 100.0)})
        self.assertNotIn("100.0", out)
        self.assertNotIn("%", out.replace("vs llama.cpp", ""))
        self.assertIn("driver gate", out)

    def test_a_run_that_fails_its_manifest_check_publishes_nothing(self):
        # Ollama ends answers at EOS: short answers are less work, so its numbers are withheld, not only its %.
        def problems(files, runs):
            return [f"{files[1]}: w1 turn 5: hash or length off the manifest"] if runs[1]["engine"] == "ollama" else []
        with mock.patch.object(sb.baseline, "compare_problems", problems):
            out = self.report({"a.json": run("larch", "chat.jsonl", 8, 110.0), "o.json": run("ollama", "chat.jsonl", 8, 295.6)})
        self.assertNotIn("295.6", out)
        self.assertIn("o.json: w1 turn 5", out)

    def test_a_withheld_run_is_named_once_in_its_note(self):
        def problems(files, runs):
            return [f"{files[1]}: w1 turn {t}: hash or length off the manifest" for t in (5, 6)] if runs[1]["engine"] == "ollama" else []
        with mock.patch.object(sb.baseline, "compare_problems", problems):
            out = self.report({"a.json": run("larch", "chat.jsonl", 8, 110.0), "o.json": run("ollama", "chat.jsonl", 8, 295.6)})
        self.assertIn("x o.json: w1 turn 5: hash or length off the manifest; w1 turn 6:", out)
        self.assertEqual(out.count("o.json"), 1)

    def test_a_generated_run_clears_its_row_only_when_every_turn_checks_out(self):
        # architecture 16: the gate is the requests themselves (cached_tokens == the KV the session found, per turn),
        # never a summary a foreign or empty file can claim.
        rows = {"a.json": run("larch", "w2-chats.jsonl", 8, 110.0)}
        good = gen("w2-chats.jsonl", 8)
        kv = [dict(r) for r in good["requests"]]
        kv[3]["kv_tokens"] += 1
        no_kv = [{k: v for k, v in r.items() if k != "kv_tokens"} for r in good["requests"]]
        for name, g in (("empty", {**good, "requests": []}), ("ollama", {**good, "engine": "ollama"}),
                        ("kv", {**good, "requests": kv}), ("no kv", {**good, "requests": no_kv}),
                        ("short", {**good, "requests": good["requests"][1:]}),
                        ("twice", {**good, "requests": good["requests"] + good["requests"][:1]}),
                        ("error", {**good, "requests": [{**good["requests"][0], "error": "Internal(1)"}] + good["requests"][1:]}),
                        ("bytes", {**good, "workload": {**good["workload"], "sha256": "x"}})):
            out = self.report({**rows, "g.json": g})
            self.assertNotIn("110.0", out, name)
        self.assertIn("110.0", self.report({**rows, "g.json": good}))

    def test_a_generated_run_clears_only_its_rows_context_and_identity(self):
        # The gate is the row as run: its context, and the same build, device and startup settings as the scripted run.
        rows = {"a.json": {**run("larch", "w2-chats.jsonl", 8, 110.0), "system_info": "Cuda dev0"}}
        good = {**gen("w2-chats.jsonl", 8), "system_info": "Cuda dev0"}
        knob = [{"at_s": None, "knob": "prefill_chunk_tokens", "reply": "startup", "value": 256.0}]
        for name, g in (("context", {**good, "params": {"n_ctx_seq": 16384}}), ("no context", {**good, "params": {}}),
                        ("device", {**good, "system_info": "Cuda dev1"}), ("build", {**good, "build": "b" * 64}),
                        ("settings", {**good, "knob_timeline": knob})):
            self.assertNotIn("110.0", self.report({**rows, "g.json": g}), name)
        self.assertIn("110.0", self.report({**rows, "g.json": good}))
        # The same startup settings in another order are the same settings.
        two = knob + [{"at_s": None, "knob": "max_batch_seqs", "reply": "startup", "value": 8.0}]
        out = self.report({"a.json": {**rows["a.json"], "knob_timeline": two}, "g.json": {**good, "knob_timeline": two[::-1]}})
        self.assertIn("110.0", out)
        # The engine applies startup values in order, so the last assignment of a knob is the setting in force.
        k = lambda v: {"at_s": None, "knob": "prefill_chunk_tokens", "reply": "startup", "value": v}  # noqa: E731
        out = self.report({"a.json": {**rows["a.json"], "knob_timeline": [k(256.0), k(512.0)]},
                           "g.json": {**good, "knob_timeline": [k(512.0), k(256.0)]}})
        self.assertNotIn("110.0", out)
        out = self.report({"a.json": {**rows["a.json"], "knob_timeline": [k(256.0), k(512.0)]},
                           "g.json": {**good, "knob_timeline": [k(512.0)]}})
        self.assertIn("110.0", out)

    def test_results_without_a_build_clear_no_scripted_row(self):
        # Results from before `build` cannot show that one binary ran both: the gate stays shut, never null == null.
        old = {k: v for k, v in run("larch", "w2-chats.jsonl", 8, 110.0).items() if k != "build"}
        g = {k: v for k, v in gen("w2-chats.jsonl", 8).items() if k != "build"}
        out = self.report({"a.json": old, "g.json": g})
        self.assertNotIn("110.0", out)
        self.assertIn("records no build", out)

    def test_only_the_machines_rows_are_rows(self):
        # A run's label comes from its machine's table: W4 at 3, or W1 without reuse, is no row of wsl5070.
        out = self.report({"a.json": run("larch", "chat.jsonl", 3, 77.7), "b.json": run("larch", "w1-chat.jsonl", 1, 66.6, cache=False),
                           "c.json": {**run("larch", "chat.jsonl", 8, 55.5), "machine": "nowhere"},
                           "d.json": run("larch", "chat.jsonl", 8, 90.0)})
        self.assertIn("90.0", out)
        for n in ("77.7", "66.6", "55.5"):
            self.assertNotIn(n, out)

    def test_malformed_results_are_skipped_not_rendered(self):
        bad = {"agg.json": run("larch", "chat.jsonl", 4, "fast"), "n.json": run("larch", "chat.jsonl", "8", 1.0),
               "mode.json": {**run("larch", "chat.jsonl", 1, 1.0), "mode": "other"},
               "eng.json": run("other", "chat.jsonl", 16, 1.0),
               "lat.json": {**run("larch", "w5-longprompt.jsonl", 1, 1.0),
                            "summary": {**run("larch", "chat.jsonl", 1, 1.0)["summary"], "ttft_ms": {"p50": "x"}}},
               "fixed.json": {**fixed("larch", 512, [(8, None)]), "model": "m1.gguf"},
               "fixed2.json": {**fixed("larch", 512, [(8, 1.0)]), "fixed": [{"pp": 512}]},
               "machine.json": {**run("larch", "w1-chat.jsonl", 1, 1.0), "machine": ["wsl5070"]},
               "nan.json": run("larch", "chat.jsonl", 16, float("nan"))}
        out = self.report({**bad, "ok.json": run("larch", "chat.jsonl", 1, 90.0),
                           "wsl5070-m1-fixed.json": fixed("llama.cpp", 512, []), "wsl5070-m1-batched.jsonl": batched([(8, "x")])})
        self.assertIn("90.0", out)
        self.assertEqual([l for l in out.splitlines() if l[:2] in ("W4", "W5", "W6")], [next(l for l in out.splitlines() if l.startswith("W4"))])

    def test_a_fixed_result_with_malformed_params_is_skipped_and_the_rest_prints(self):
        for params in ([1], 3, "x"):
            out = self.report({"bad.json": {**fixed("larch", 512, [(8, 777.7)]), "params": params},
                               "ok.json": run("larch", "chat.jsonl", 8, 90.0)})
            self.assertIn("90.0", out, params)
            self.assertNotIn("777.7", out, params)

    def test_notes_carry_no_windows_path(self):
        self.assertEqual(sb.names(r"C:\Users\someone\src\larch\bench\results\wsl5070-m1-llama-flags.json is missing"),
                         "wsl5070-m1-llama-flags.json is missing")
        self.assertEqual(sb.names(r"see D:/models/m1.gguf and \\host\share\x\m5.gguf"), "see m1.gguf and m5.gguf")
        self.assertEqual(sb.names("'/home/someone/src/bench/manifest/m1.jsonl' is missing"), "'m1.jsonl' is missing")
        # A repo-relative path names nobody and stays whole.
        self.assertEqual(sb.names("bench/manifest/m1.jsonl is missing"), "bench/manifest/m1.jsonl is missing")

    def test_live_knob_runs_skipped(self):
        # W7 (W2 at 8 with gait on) or any live --set change is not the plain row.
        live = {**run("larch", "chat.jsonl", 8, 77.7), "knob_timeline": [{"at_s": 30.0, "knob": "k", "value": 1}]}
        out = self.report({"a.json": run("larch", "chat.jsonl", 8, 90.0), "l.json": live})
        self.assertIn("90.0", out)
        self.assertNotIn("77.7", out)
        # A joined --slice row keeps each part's timeline under slices.
        joined = {**run("larch", "chat.jsonl", 16, 55.5), "slices": [{"k": 0, "knob_timeline": []},
                                                                       {"k": 1, "knob_timeline": live["knob_timeline"]}]}
        self.assertNotIn("55.5", self.report({"j.json": joined}))

    def test_gate_notes_carry_no_local_path(self):
        with mock.patch.object(sb.baseline, "check_fixed",
                               lambda a, alias: sb.baseline.die("/home/someone/src/bench/results/wsl5070-m1-llama-flags.json is missing")):
            out = self.report({"wsl5070-m1-w4-8.json": run("llama.cpp", "chat.jsonl", 8, 100.0)})
        self.assertNotIn("/home/", out)
        self.assertIn("wsl5070-m1-llama-flags.json is missing", out)

    def test_malformed_result_skipped(self):
        broken = run("larch", "chat.jsonl", 8, 77.7)
        del broken["summary"]
        out = self.report({"a.json": run("larch", "chat.jsonl", 4, 90.0), "x.json": broken})
        self.assertIn("90.0", out)

    def test_zero_baseline_has_no_percent(self):
        out = self.report({"a.json": run("larch", "w1-chat.jsonl", 1, 90.0), "b.json": run("llama.cpp", "w1-chat.jsonl", 1, 0.0)})
        self.assertNotIn("%", out)



class CompareProblems(unittest.TestCase):
    """baseline.compare_problems, shared by `baseline.py compare` and the scoreboard: problems, never an exit."""

    def problems(self, x, y):
        return sb.baseline.compare_problems(["x.json", "y.json"], [x, y])

    def test_a_model_without_a_manifest_is_a_problem_not_an_exit(self):
        # A bench dir with the models but no manifests, so the test holds whichever manifests the repo gains.
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "models.toml").write_text((sb.baseline.BENCH / "models.toml").read_text())
            with mock.patch.object(sb.baseline, "BENCH", pathlib.Path(d)):
                self._missing_manifest()

    def _missing_manifest(self):
        bad = self.problems(run("larch", "w1-chat.jsonl", 1, 1.0, model="m8.gguf"), run("llama.cpp", "w1-chat.jsonl", 1, 1.0, model="m8.gguf"))
        self.assertTrue(any("manifest/m8.jsonl is missing" in b for b in bad), bad)

    def test_malformed_and_foreign_results_are_problems(self):
        no_cache = run("larch", "w1-chat.jsonl", 1, 1.0)
        del no_cache["cache_prompt"]
        for x in (no_cache, run("larch", "w1-chat.jsonl", 1, 1.0, model="other.gguf"), [1], {"schema": "x"}):
            self.assertTrue(self.problems(x, run("llama.cpp", "w1-chat.jsonl", 1, 1.0)), x)

    def test_a_truncated_manifest_is_a_problem_not_an_exit(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "models.toml").write_text((sb.baseline.BENCH / "models.toml").read_text())
            (pathlib.Path(d) / "manifest").mkdir()
            (pathlib.Path(d) / "manifest" / "m1.jsonl").write_text('{"workload": "w1-chat.jsonl", "conv"')
            x, y = run("larch", "w1-chat.jsonl", 1, 1.0), run("llama.cpp", "w1-chat.jsonl", 1, 1.0)
            with mock.patch.object(sb.baseline, "BENCH", pathlib.Path(d)):
                bad = self.problems(x, y)
        self.assertTrue(any("malformed" in b for b in bad), bad)

    def test_mixed_key_types_are_a_problem_not_an_exit(self):
        x, y = run("larch", "w1-chat.jsonl", 1, 1.0), run("llama.cpp", "w1-chat.jsonl", 1, 1.0)
        x["requests"] = [{"conv": 1, "turn": 0, "prompt_sha256": "a", "max_tokens": 1}]
        y["requests"] = [{"conv": "c", "turn": 0, "prompt_sha256": "b", "max_tokens": 1}]
        self.assertTrue(self.problems(x, y))

    def test_machines_must_match(self):
        bad = self.problems(run("larch", "w1-chat.jsonl", 1, 1.0), {**run("llama.cpp", "w1-chat.jsonl", 1, 1.0), "machine": "win4060"})
        self.assertIn("machine differs: 'wsl5070' vs 'win4060'", bad)


    def test_a_run_is_checked_against_its_machines_row_not_its_own_fields(self):
        x = full("larch", 1.0)
        # Concurrency 3 with these 200 requests would pass a check derived from the file's own fields.
        bad = self.problems({**x, "concurrency": 3}, full("llama.cpp", 1.0))
        self.assertTrue(any("not one of wsl5070's rows" in b for b in bad), bad)
        bad = self.problems({**x, "params": {"n_ctx_seq": 4096}}, {**x, "engine": "ollama", "params": {"n_ctx_seq": 4096}})
        self.assertTrue(any("n_ctx_seq = 4096, not the row's 2560" in b for b in bad), bad)

    def test_a_driver_run_must_use_the_searched_flags(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "wsl5070-m1-llama-flags.json").write_text(json.dumps(FLAGS))
            llama = pathlib.Path(d) / "wsl5070-m1-w4-8.json"
            good = {**full("llama.cpp", 1.0), "params": {**PARAMS, "n_ctx_seq": 2560}}
            ok = sb.baseline.compare_problems([str(llama), "v.json"], [good, full("larch", 1.0)])
            self.assertEqual(ok, [])
            bad = sb.baseline.compare_problems([str(llama), "v.json"], [{**good, "params": {**good["params"], "n_threads": 8}},
                                                                        full("larch", 1.0)])
            self.assertTrue(any("params.n_threads = 8, not 4" in b for b in bad), bad)
            # No flag search beside the result is a problem, never a pass or an exit.
            (pathlib.Path(d) / "wsl5070-m1-llama-flags.json").unlink()
            bad = sb.baseline.compare_problems([str(llama), "v.json"], [good, full("larch", 1.0)])
            self.assertTrue(any("llama-flags.json" in b for b in bad), bad)


FLAGS = {"t": 4, "b": 2048, "ub": 1024}
BB_FLAGS = {"is_pp_shared": 0, "flash_attn": 1, "n_batch": 2048, "n_ubatch": 1024, "n_threads": 4, "n_gpu_layers": 99}
PARAMS = {"n_gpu_layers": 99, "flash_attn": "enabled", "type_k": "f16", "type_v": "f16", "kv_unified": False,
          "n_batch": 2048, "n_ubatch": 1024, "n_threads": 4}


class ServerCheck(unittest.TestCase):
    """baseline.py server's proof, per alias, as the scoreboard's driver gate reads it."""

    def check(self, d, files):
        for name, doc in files.items():
            (pathlib.Path(d) / name).write_text(json.dumps(doc))
        return sb.server_problems("wsl5070", pathlib.Path(d), "m5")

    def test_only_the_aliass_own_current_check_counts(self):
        sc = {"alias": "m5", "machine": "wsl5070", "params": PARAMS, "llama_cpp": {"build": 1}, "ok": True}
        flags = {"wsl5070-m5-llama-flags.json": FLAGS}
        for name, files in (("another alias's", {"wsl5070-m1-server-check.json": {**sc, "alias": "m1"}}),
                            ("failed", {"wsl5070-m5-server-check.json": {**sc, "ok": False}}),
                            ("old format", {"wsl5070-m5-server-check.json": {"ok": True}}),
                            ("other flags", {"wsl5070-m5-server-check.json": {**sc, "params": {**PARAMS, "n_batch": 512}}}),
                            ("mislabelled", {"wsl5070-m5-server-check.json": {**sc, "alias": "m1"}})):
            with tempfile.TemporaryDirectory() as d:
                self.assertTrue(self.check(d, {**flags, **files}), name)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.check(d, {**flags, "wsl5070-m5-server-check.json": sc}), [])

    def test_a_row_from_another_llama_build_than_the_check_is_withheld(self):
        sc = {"alias": "m1", "machine": "wsl5070", "params": PARAMS, "llama_cpp": {"build": 1}, "ok": True}
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "wsl5070-m1-server-check.json").write_text(json.dumps(sc))
            path = pathlib.Path(d) / "wsl5070-m1-w4-8.json"
            for doc, want in (({"llama_cpp": {"build": 1}}, []), ({"llama_cpp": {"build": 2}}, ["another llama.cpp or driver build"])):
                got = sb.build_problems(sb.Cell(1.0, 0, {}, {}, path, {**run("llama.cpp", "chat.jsonl", 8, 1.0), **doc}))
                self.assertEqual([w for w in want if not any(w in g for g in got)], [], got)
                self.assertEqual(bool(got), bool(want), got)

    def test_a_failed_rerun_leaves_no_earlier_pass(self):
        with tempfile.TemporaryDirectory() as d:
            old = pathlib.Path(d) / "wsl5070-m5-server-check.json"
            old.write_text(json.dumps({"alias": "m5", "ok": True}))
            a = sb.argparse.Namespace(machine="wsl5070", out=pathlib.Path(d), models=pathlib.Path(d), build=pathlib.Path(d),
                                      alias=["m5"])
            with self.assertRaises(SystemExit):  # no flag search here: it dies before llama-server starts
                sb.baseline.step_server(a)
            self.assertFalse(old.exists())



class DriverIdentity(unittest.TestCase):
    """The real gates, unmocked, over a directory of llama.cpp evidence: the fixed-shape mirror, llama-batched-bench's
    rows, the flag search and the server check must name one llama.cpp and driver build."""

    def evidence(self, d, mirror_build, check=None):
        d = pathlib.Path(d)
        (d / "wsl5070-m1-llama-flags.json").write_text(json.dumps(FLAGS))
        shapes = [(pl, 100.0 * pl) for pl in (1, 4, 8, 16)] * sb.baseline.FIXED_REPS
        (d / "wsl5070-m1-batched.jsonl").write_text("\n".join(json.dumps(
            {"pp": 512, "tg": 128, "pl": pl, "speed": v, **BB_FLAGS}) for pl, v in shapes) + "\n")
        mirror = {**fixed("llama.cpp", 512, shapes), "model": "m1.gguf", "params": {**PARAMS, "n_ctx_seq": 1024},
                  "llama_cpp": {"tag": "b1"}, "build": mirror_build}
        (d / "wsl5070-m1-fixed.json").write_text(json.dumps(mirror))
        sc = {"alias": "m1", "machine": "wsl5070", "params": PARAMS, "llama_cpp": {"tag": "b1"}, "build": "new", "ok": True}
        (d / "wsl5070-m1-server-check.json").write_text(check if isinstance(check, str) else json.dumps(check or sc))
        (d / "v.json").write_text(json.dumps({**fixed("larch", 512, [(8, 500.0)]), "model": "m1.gguf"}))
        return sb.report(sorted(d.glob("*.json")))

    def test_a_mirror_from_another_build_than_the_server_check_publishes_no_w6(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.evidence(d, "new")
        self.assertIn("800.0", out)  # the same build: llama-batched-bench's W6 row is published
        with tempfile.TemporaryDirectory() as d:
            out = self.evidence(d, "old")
        self.assertNotIn("800.0", out)
        self.assertNotIn("%", out.replace("vs llama.cpp", ""))
        self.assertIn("another llama.cpp or driver build", out)

    def test_a_malformed_server_check_is_a_gate_problem_not_a_crash(self):
        for check in ("[1]", "3", "null", json.dumps({"alias": "m1", "machine": "wsl5070", "ok": True, "params": [1]})):
            with tempfile.TemporaryDirectory() as d:
                out = self.evidence(d, "new", check)
                (pathlib.Path(d) / "wsl5070-m1-w4-8.json").write_text(json.dumps({**full("llama.cpp", 100.0), "llama_cpp": {"tag": "b1"}}))
                out = sb.report(sorted(pathlib.Path(d).glob("*.json")))
            self.assertNotIn("800.0", out, check)
            self.assertNotIn("100.0", out, check)
            self.assertIn("driver gate", out, check)



class FixedGate(unittest.TestCase):
    """baseline.check_fixed over FIXED_REPS alternated runs a side: per shape, S over all runs within 3 %."""

    def check(self, bb, drv):
        """`bb` and `drv`: one {pl: speed} per run. The gate's problems for m1 on wsl5070."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wsl5070-m1-llama-flags.json").write_text(json.dumps(FLAGS))
            (d / "wsl5070-m1-batched.jsonl").write_text("\n".join(
                json.dumps({"pp": 512, "tg": 128, "pl": pl, "speed": v, **BB_FLAGS}) for run in bb for pl, v in run.items()))
            rows = [(pl, v) for run in drv for pl, v in run.items()]
            (d / "wsl5070-m1-fixed.json").write_text(json.dumps({**fixed("llama.cpp", 512, rows), "model": "m1.gguf",
                                                                 "params": {**PARAMS, "n_ctx_seq": 1024}}))
            with mock.patch("sys.stdout"):
                return sb.baseline.check_fixed(sb.argparse.Namespace(machine="wsl5070", out=d), "m1")

    def runs(self, scale=1.0, last=None):
        """FIXED_REPS runs at 100 t/s per sequence times `scale`, the last one at `last` times that when given."""
        n = sb.baseline.FIXED_REPS
        return [{pl: 100.0 * pl * scale * (last if last and i == n - 1 else 1) for pl in (1, 4, 8, 16)} for i in range(n)]

    def test_one_outlier_run_a_side_passes(self):
        # Single runs 7 % and 8 % off (llama-batched-bench's own spread on wsl5070 reaches 8.4 %): S over all runs agrees.
        self.assertEqual(self.check(self.runs(last=0.93), self.runs(last=1.08)), [])

    def test_a_steady_gap_past_3_percent_fails(self):
        bad = self.check(self.runs(), self.runs(scale=0.96))
        self.assertEqual(bad, [f"fixed npl={pl} off by -4.0%" for pl in (1, 4, 8, 16)])

    def test_a_run_at_zero_is_a_gate_problem_not_a_crash(self):
        bb = self.runs()
        bb[0][8] = 0.0
        self.assertEqual(self.check(bb, self.runs()), ["fixed npl=8: S 800.0 driver, 0 llama-batched-bench, not positive"])

    def test_a_shape_short_of_its_runs_fails(self):
        drv = self.runs()
        del drv[0][8]
        bad = self.check(self.runs(), drv)
        n = sb.baseline.FIXED_REPS
        self.assertEqual(bad, [f"fixed npl=8: {n - 1} driver and {n} llama-batched-bench runs, not {n} each"])


FAKE = """#!/usr/bin/env python3
import json, os, sys
log = os.environ["FAKE_LOG"]
tool = os.path.basename(sys.argv[0])
with open(log, "a") as f:
    f.write(tool + "\\n")
calls = open(log).read().split().count(tool)
if tool == "llama-serve-bench" and calls == int(os.environ.get("FAKE_FAIL_AT", 0)):
    sys.exit(3)
if tool == "llama-serve-bench" and calls == int(os.environ.get("FAKE_BAD_AT", 0)):
    open(sys.argv[sys.argv.index("--json") + 1], "w").write('{"fixed": [')
    sys.exit(0)
rows = [{"pp": 512, "tg": 128, "pl": pl, "speed": 100.0 * pl + calls} for pl in (1, 4, 8, 16)]
if tool == "llama-batched-bench":
    print("llama-batched-bench log line")
    print("\\n".join(json.dumps(r) for r in rows))
else:
    json.dump({"mode": "fixed", "fixed": rows}, open(sys.argv[sys.argv.index("--json") + 1], "w"))
"""


@unittest.skipIf(sb.baseline.os.name == "nt", "the fake tools are POSIX scripts")
class FixedRun(unittest.TestCase):
    """baseline.run_fixed: llama-batched-bench and the driver alternate FIXED_REPS times in one hold, every row kept."""

    def run_fixed(self, fail_at=0, bad_at=0):
        d = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (d / "bin").mkdir()
        for name in ("llama-batched-bench", "llama-serve-bench"):
            (d / "bin" / name).write_text(FAKE)
            (d / "bin" / name).chmod(0o755)
        a = sb.argparse.Namespace(machine="wsl5070", build=d, out=d / "out")
        a.out.mkdir()
        env = {"FAKE_LOG": str(d / "log"), "FAKE_FAIL_AT": str(fail_at), "FAKE_BAD_AT": str(bad_at)}
        with mock.patch.dict(sb.baseline.os.environ, env), mock.patch("sys.stdout"):
            ok = sb.baseline.run_fixed(a, "m1", ["-m", "m1.gguf"])
        return ok, (d / "log").read_text().split(), a.out

    def test_alternates_and_keeps_every_row(self):
        ok, log, out = self.run_fixed()
        n = sb.baseline.FIXED_REPS
        self.assertTrue(ok)
        self.assertEqual(log, ["llama-batched-bench", "llama-serve-bench"] * n)
        bb = sb.baseline.batched_rows(out / "wsl5070-m1-batched.jsonl")
        fx = json.loads((out / "wsl5070-m1-fixed.json").read_text())["fixed"]
        self.assertEqual(sorted(r["speed"] for r in bb if r["pl"] == 8), [801.0 + i for i in range(n)])
        self.assertEqual(sorted(r["speed"] for r in fx if r["pl"] == 8), [801.0 + i for i in range(n)])

    def test_a_failed_run_leaves_no_result(self):
        ok, log, out = self.run_fixed(fail_at=2)
        self.assertFalse(ok)
        self.assertEqual(log, ["llama-batched-bench", "llama-serve-bench"] * 2)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["wsl5070-m1-fixed.failed"])

    def test_exit_0_without_a_fixed_shape_result_is_a_failed_run(self):
        ok, log, out = self.run_fixed(bad_at=2)
        self.assertFalse(ok)
        self.assertEqual(log, ["llama-batched-bench", "llama-serve-bench"] * 2)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["wsl5070-m1-fixed.failed"])
        self.assertIn("not a fixed-shape result", (out / "wsl5070-m1-fixed.failed").read_text())


if __name__ == "__main__":
    unittest.main()
