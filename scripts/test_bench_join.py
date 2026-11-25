"""Tests for scripts/bench-join: python3 -m unittest scripts/test_bench_join.py -v"""

import importlib.machinery
import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
# bench-join has no .py extension, so the default finders would not recognise it.
loader = importlib.machinery.SourceFileLoader("bench_join", str(HERE / "bench-join"))
bj = importlib.util.module_from_spec(importlib.util.spec_from_loader("bench_join", loader))
loader.exec_module(bj)


def req(conv, ttft, tpot, gen=4, prompt=10, error=None, t=0.0):
    """A request record as bench.rs serializes it: no cancel flag, so a cancellation is told apart only by its run's
    summary."""
    return {"conv": conv, "turn": 0, "prompt_tokens": prompt, "cached_tokens": 0, "gen_tokens": gen, "ttft_ms": ttft,
            "tpot_ms": tpot, "e2e_ms": ttft + 3 * tpot, "error": error, "t_submit_ms": t,
            "t_first_ms": None if error else t + ttft, "t_done_ms": t + ttft + 3 * tpot}


def part(k, n, first, end, reqs, wall, cancelled=0):
    failed = sum(r["error"] is not None for r in reqs) - cancelled
    return {"schema": "larch-bench/1", "engine": "larch", "mode": "run", "model": "m1.gguf", "machine": "wsl5070",
            "build": "b" * 64, "system_info": "Cuda dev0", "llama_cpp": None,
            "concurrency": 1, "replay": "scripted", "cache_prompt": False, "cancel_rate": 0.0, "params": {"n_ctx_seq": 2560},
            "workload": {"file": "chat.jsonl", "sha256": "s", "conversations": end - first}, "started_at": f"t{k}",
            "slice": {"k": k, "n": n, "first": first, "end": end}, "telemetry": {"engine_samples": k}, "knob_timeline": [],
            "summary": {"wall_s": wall, "requests": len(reqs), "failures": failed, "cancelled": cancelled}, "requests": reqs}


A = part(0, 2, 0, 2, [req("c0", 10.0, 2.0), req("c1", 30.0, 4.0, t=5.0)], 10.0)
B = part(1, 2, 2, 3, [req("c2", 20.0, 3.0, error="Internal(2)")], 5.0)


class Join(unittest.TestCase):
    def test_summary_is_recomputed_over_every_request(self):
        j = bj.join([B, A])
        s = j["summary"]
        self.assertEqual([r["conv"] for r in j["requests"]], ["c0", "c1", "c2"])
        self.assertEqual((s["requests"], s["failures"], s["cancelled"]), (3, 1, 0))
        # Only successful requests count tokens and latencies, as bench.rs's summary().
        self.assertEqual((s["prompt_tokens"], s["gen_tokens"], s["cached_tokens"], s["evaluated_tokens"]), (20, 8, 0, 20))
        self.assertEqual(s["wall_s"], 15.0)
        self.assertAlmostEqual(s["agg_tok_s"], 8 / 15)
        self.assertEqual(s["ttft_ms"], {"p50": 10.0, "p99": 30.0, "mean": 20.0, "max": 30.0})
        self.assertNotIn("slice", j)
        self.assertEqual(j["workload"]["conversations"], 3)
        self.assertEqual([x["k"] for x in j["slices"]], [0, 1])
        self.assertEqual(j["slices"][1]["telemetry"], {"engine_samples": 1})

    def test_requests_are_time_ordered_like_one_run(self):
        # Each part's clock starts at 0; a later part starts where the earlier ones ended (the gaps between holds
        # never count, as in the summed wall time).
        r = bj.join([A, B])["requests"]
        self.assertEqual([(x["t_submit_ms"], x["t_first_ms"], x["t_done_ms"]) for x in r],
                         [(0.0, 10.0, 16.0), (5.0, 35.0, 47.0), (47.0, None, 76.0)])
        self.assertEqual(A["requests"][0]["t_submit_ms"], 0.0)  # the parts are not changed

    def test_mean_sums_in_order_as_bench_rs(self):
        # Rust's iter().sum() adds left to right; Python 3.12+ sum() is compensated and would give 1/3 here.
        self.assertEqual(bj.stats([1e16, 1.0, -1e16])["mean"], 0.0)

    def test_the_same_startup_settings_in_another_order_join(self):
        k = [{"at_s": None, "knob": "prefill_chunk_tokens", "reply": "startup", "value": 256.0},
             {"at_s": None, "knob": "max_batch_seqs", "reply": "startup", "value": 8.0}]
        self.assertEqual(bj.join([{**A, "knob_timeline": k}, {**B, "knob_timeline": k[::-1]}])["summary"]["requests"], 3)
        # Repeated assignments: the engine applies them in order, so the last one is the setting in force.
        dup = [k[0], {**k[0], "value": 512.0}]
        with self.assertRaises(SystemExit):
            bj.join([{**A, "knob_timeline": dup}, {**B, "knob_timeline": dup[::-1]}])
        self.assertEqual(bj.join([{**A, "knob_timeline": dup}, {**B, "knob_timeline": dup[1:]}])["summary"]["requests"], 3)

    def test_cancellations_are_counted_from_the_parts_summaries(self):
        # bench.rs serializes no cancel flag: a Cancelled error is a cancellation only when its run counted it as one.
        c = part(1, 2, 2, 3, [req("c2", 20.0, 3.0, error="Cancelled")], 5.0, cancelled=1)
        s = bj.join([A, c])["summary"]
        self.assertEqual((s["requests"], s["failures"], s["cancelled"]), (3, 0, 1))
        s = bj.join([A, {**c, "summary": {**c["summary"], "cancelled": 0, "failures": 1}}])["summary"]
        self.assertEqual((s["failures"], s["cancelled"]), (1, 0))

    def test_nearest_rank_matches_bench_rs(self):
        # 100 values: p50 is the 50th, p99 the 99th (ceil(q * n), 1-based).
        self.assertEqual(bj.stats([float(x) for x in range(100, 0, -1)]),
                         {"p50": 50.0, "p99": 99.0, "mean": 50.5, "max": 100.0})
        self.assertIsNone(bj.stats([]))

    def test_token_gaps_preserve_observations_without_slice_boundaries(self):
        a = part(0, 2, 0, 1, [{**req("c0", 10.0, 2.0), "token_gaps_ms": [1.0, 3.0, 2.0]}], 10.0)
        b = part(1, 2, 1, 2, [{**req("c1", 20.0, 4.0), "token_gaps_ms": [4.0, 6.0, 2.0]}], 20.0)
        j = bj.join([a, b])
        self.assertEqual(j["summary"]["token_gap_ms"], bj.stats([1.0, 3.0, 2.0, 4.0, 6.0, 2.0]))
        self.assertEqual(j["requests"][1]["token_gaps_ms"], [4.0, 6.0, 2.0])
        self.assertNotIn("token_gap_ms", bj.join([a, {**b, "requests": [req("c1", 20.0, 4.0)]}])["summary"])

    def test_incomplete_or_mixed_parts_refuse(self):
        live = [{"at_s": 30.0, "knob": "max_batch_seqs", "value": 2}]
        for parts in ([A], [A, A], [A, {**B, "model": "m5.gguf"}], [A, {**B, "slice": {**B["slice"], "first": 3}}],
                      [A, {**B, "concurrency": 4}], [{**A, "slice": None}, B], [A, {**B, "requests": []}],
                      # Another binary or device, or a live knob change that each part would replay from its start.
                      [A, {**B, "build": "c" * 64}], [A, {**B, "system_info": "Cuda dev1"}], [A, {**B, "knob_timeline": live}],
                      # Other startup settings: the same binary and device, configured differently.
                      [A, {**B, "knob_timeline": [{"at_s": None, "knob": "prefill_chunk_tokens", "reply": "startup", "value": 1024.0}]}],
                      # A summary that does not count the part's own requests.
                      [A, {**B, "summary": {**B["summary"], "requests": 2}}]):
            with self.assertRaises(SystemExit, msg=parts):
                bj.join(parts)

    def test_main_writes_the_joined_file(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, doc in (("a.json", A), ("b.json", B)):
                (pathlib.Path(d) / name).write_text(json.dumps(doc))
                paths.append(str(pathlib.Path(d) / name))
            out = pathlib.Path(d) / "w4-1.json"
            bj.main([str(out), *paths])
            self.assertEqual(json.loads(out.read_text())["summary"]["requests"], 3)


if __name__ == "__main__":
    unittest.main()
