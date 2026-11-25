"""Offline checks of the public-bench driver: When2Call against a fake server, the BFCL subset and scores, tau2's
rewards and the gate. BFCL and tau2 themselves need their own packages and run on the Mac."""

import contextlib
import hashlib
import http.server
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402

PLAY = {
    "name": "Music.PlayMedia",
    "description": "Plays a track.",
    "parameters": {
        "type": "dict",
        "required": ["track"],
        "properties": {
            "track": {"type": "string", "description": "The track."},
            "volume": {"type": "float", "description": "0 to 1."},
            "rooms": {"type": "array", "items": {"type": "dict", "properties": {"id": {"type": "any"}}}},
        },
    },
}


def item(uuid, label, tools, question="q"):
    target = json.dumps(PLAY) if label == "tool_call" else None
    return {"uuid": uuid, "source": "s", "correct_answer": label, "question": question, "target_tool": target,
            "tools": [json.dumps(t) for t in tools]}


def write_scores(root, accuracy):
    """BFCL's score files under `root` for each category's `accuracy`."""
    for cat, a in accuracy.items():
        f = root / "score" / "larch-FC" / "x" / f"BFCL_v4_{cat}_score.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"accuracy": a}) + "\n" + json.dumps({"id": "row"}) + "\n")


def bfcl_data_dir(d):
    """A bfcl-eval data folder under `d` with one id in each category of the subset."""
    data = d / "bfcl-data"
    data.mkdir()
    for cat in run.BFCL_SUBSET:
        (data / f"BFCL_v4_{cat}.json").write_text(json.dumps({"id": f"{cat}_0"}) + "\n")
    return data


class Fake(http.server.BaseHTTPRequestHandler):
    """An OpenAI chat endpoint whose reply to a question is `replies[question]`: a tool name to call, or text."""

    replies = {}
    seen = []

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(req)
        reply = Fake.replies[req["messages"][-1]["content"]]
        if reply == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"grammar too large")
            return
        message = {"role": "assistant", "content": reply}
        if reply.startswith("call:"):
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": reply[5:], "arguments": "{}"}}]}
        body = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


@contextlib.contextmanager
def server(replies):
    Fake.replies, Fake.seen = replies, []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


class PublicBench(unittest.TestCase):
    def test_a_bfcl_function_becomes_an_openai_tool(self):
        tool = run.openai_tool(json.dumps(PLAY))
        self.assertEqual(tool["function"]["name"], "Music_PlayMedia")
        params = tool["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"]["volume"]["type"], "number")
        self.assertEqual(params["properties"]["rooms"]["items"]["type"], "object")
        self.assertEqual(params["properties"]["rooms"]["items"]["properties"]["id"]["type"], "string")

    def test_when2call_scores_the_call_decision(self):
        items = [
            item("a", "tool_call", [PLAY], "play it"),
            item("b", "tool_call", [PLAY], "play the other"),
            item("c", "cannot_answer", [], "weather?"),
            item("d", "cannot_answer", [PLAY], "book a flight"),
            item("e", "request_for_info", [PLAY], "play something"),
            item("f", "request_for_info", [PLAY], "too big"),
        ]
        replies = {"play it": "call:Music_PlayMedia", "play the other": "Which room?", "weather?": "I can't.",
                   "book a flight": "call:Music_PlayMedia", "play something": "Which track?", "too big": "500"}
        with server(replies) as url:
            rows = [run.when2call_row(url, "m", i, 0.7, 2) for i in items]
        self.assertEqual([r["right"] for r in rows], [True, False, True, False, True, False])
        self.assertEqual(rows[3]["called"], ["Music_PlayMedia"])
        self.assertIn("500", rows[5]["error"])
        # No tools, no `tools` key; the run's seed goes with every request.
        self.assertNotIn("tools", Fake.seen[2])
        self.assertEqual({r["seed"] for r in Fake.seen}, {2})
        scores = run.when2call_scores(rows)
        self.assertEqual(scores["accuracy"], 0.5)
        self.assertEqual(scores["tool_call"], 0.5)
        self.assertEqual(scores["false_calls"], 1)
        self.assertEqual(scores["errors"], 1)

    def test_a_when2call_run_writes_each_runs_scores_and_the_reference(self):
        items = [item("a", "tool_call", [PLAY], "play it"), item("c", "cannot_answer", [], "weather?"),
                 item("f", "request_for_info", [PLAY], "too big")]
        body = "".join(json.dumps(i) + "\n" for i in items).encode()
        replies = {"play it": "call:Music_PlayMedia", "weather?": "No.", "too big": "500"}
        with tempfile.TemporaryDirectory() as d, server(replies) as url:
            (pathlib.Path(d) / "data").mkdir()
            (pathlib.Path(d) / "data" / "when2call_test_llm_judge.jsonl").write_bytes(body)
            out = pathlib.Path(d) / "r.json"
            with unittest.mock.patch.object(run, "HERE", pathlib.Path(d)), \
                    unittest.mock.patch.object(run, "WHEN2CALL_SHA256", hashlib.sha256(body).hexdigest()), \
                    contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
                run.main(["when2call", "--url", url, "--model", "m", "--runs", "2", "--out", str(out)])
            report = json.loads(out.read_text())
        self.assertEqual(report["served"], ["fake"])
        # The server's 500 reaches the report as a wrong, failed request.
        self.assertEqual([(r["accuracy"], r["errors"]) for r in report["when2call"]["runs"]], [(2 / 3, 1)] * 2)
        self.assertEqual([len(rows) for rows in report["when2call"]["rows"]], [3, 3])
        self.assertEqual(report["reference"], run.REFERENCE["when2call"])
        self.assertIn("when2call accuracy: 0.667", printed.getvalue())
        self.assertEqual(sorted(r["seed"] for r in Fake.seen), [0, 0, 0, 1, 1, 1])

    def test_a_when2call_file_that_no_longer_matches_the_pin_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            data = pathlib.Path(d)
            (data / "when2call_test_llm_judge.jsonl").write_text("{}\n")
            with self.assertRaises(SystemExit) as e:
                run.when2call_items(data)
        self.assertIn("does not match", str(e.exception.code))

    def test_a_server_that_does_not_answer_stops_the_run_with_its_address(self):
        with tempfile.TemporaryDirectory() as d, server({}) as url:
            pass
        with self.assertRaises(SystemExit) as e:
            run.main(["when2call", "--url", url, "--model", "m", "--out", f"{d}/r.json"])
        self.assertIn(f"cannot reach {url}/models", str(e.exception.code))

    def test_a_call_of_another_tool_is_wrong(self):
        other = dict(PLAY, name="Music_Stop")
        with server({"play it": "call:Music_Stop"}) as url:
            row = run.when2call_row(url, "m", item("a", "tool_call", [PLAY, other], "play it"), 0.7, 0)
        self.assertFalse(row["right"])

    def test_the_bfcl_subset_is_spread_over_each_category(self):
        with tempfile.TemporaryDirectory() as d:
            data = pathlib.Path(d)
            (data / "BFCL_v4_simple_python.json").write_text(
                "".join(json.dumps({"id": f"simple_python_{i}"}) + "\n" for i in range(10)))
            ids = run.bfcl_subset(data, {"simple_python": 4})
        self.assertEqual(ids, {"simple_python": ["simple_python_0", "simple_python_2", "simple_python_5",
                                                 "simple_python_7"]})

    def test_bfcl_scores_and_groups_as_bfcl_takes_them(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            write_scores(root, {"simple_python": 0.9, "simple_java": 0.6, "simple_javascript": 0.3, "multiple": 0.8,
                                "parallel": 1.0, "parallel_multiple": 0.5, "live_simple": 0.75,
                                "multi_turn_base": 0.25, "multi_turn_miss_func": 0.1})
            scores = run.bfcl_scores(root / "score" / "larch-FC")
        self.assertEqual(scores["simple_python"], 0.9)
        # BFCL's non-live AST: the simple languages' mean, then the mean with the other three.
        self.assertAlmostEqual(scores["simple"], 0.6)
        self.assertAlmostEqual(scores["non_live_ast"], (0.6 + 0.8 + 1.0 + 0.5) / 4)
        # Two of multi-turn's four categories ran: no group.
        self.assertNotIn("multi_turn", scores)
        self.assertNotIn("live_multiple", scores)

    def test_bfcl_runs_each_in_a_process_and_folder_of_its_own(self):
        roots = []

        def child(cmd, env, check):
            # The child's folder is its BFCL_PROJECT_ROOT; it scores the ids the parent wrote there.
            root = pathlib.Path(env["BFCL_PROJECT_ROOT"])
            roots.append(root)
            self.assertEqual(cmd[2:4], ["bfcl-child", "m"])
            ids = json.loads((root / "test_case_ids_to_generate.json").read_text())
            write_scores(root, {c: 0.5 + 0.25 * (len(roots) - 1) for c in ids})

        with tempfile.TemporaryDirectory() as d, server({}) as url:
            data = bfcl_data_dir(pathlib.Path(d))
            out = pathlib.Path(d) / "b.json"
            with unittest.mock.patch.object(run, "bfcl_data", lambda: data), \
                    unittest.mock.patch.object(run.subprocess, "run", child), \
                    contextlib.redirect_stdout(io.StringIO()):
                run.main(["bfcl", "--url", url, "--model", "m", "--runs", "2", "--limit", "1", "--out", str(out)])
            report = json.loads(out.read_text())
        self.assertEqual(roots, [pathlib.Path(d) / "b-bfcl-0", pathlib.Path(d) / "b-bfcl-1"])
        self.assertEqual([r["non_live_ast"] for r in report["bfcl"]["runs"]], [0.5, 0.75])
        self.assertEqual(report["limit"], 1)

    def test_a_bfcl_run_that_scores_nothing_stops(self):
        with tempfile.TemporaryDirectory() as d, server({}) as url:
            data = bfcl_data_dir(pathlib.Path(d))
            with unittest.mock.patch.object(run, "bfcl_data", lambda: data), \
                    unittest.mock.patch.object(run.subprocess, "run", lambda *a, **k: None), \
                    self.assertRaises(SystemExit) as e:
                run.main(["bfcl", "--url", url, "--model", "m", "--limit", "1", "--out", f"{d}/b.json"])
        self.assertIn("scored no simple_python", str(e.exception.code))

    def test_a_bfcl_folder_left_from_a_crash_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "b-bfcl-0").mkdir()
            with contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit):
                run.main(["bfcl", "--url", "http://x/v1", "--model", "m", "--out", f"{d}/b.json"])
        self.assertIn("left from an earlier run", said.getvalue())

    def test_tau2_runs_save_under_new_names_with_their_seed(self):
        seen = []
        with tempfile.TemporaryDirectory() as d, server({}) as url:
            tau2 = pathlib.Path(d) / "tau2"

            def tau2_cli(cmd, check, stdin):
                seen.append((cmd, stdin))
                name = cmd[cmd.index("--save-to") + 1]
                f = tau2 / "data" / "simulations" / name / "results.json"
                f.parent.mkdir(parents=True)
                f.write_text(json.dumps({"simulations": [{"reward_info": {"reward": 1.0}}]}))

            out = pathlib.Path(d) / "t.json"
            with unittest.mock.patch.object(run.subprocess, "run", tau2_cli), \
                    contextlib.redirect_stdout(io.StringIO()):
                run.main(["tau2", "--url", url, "--model", "m", "--tau2-dir", str(tau2), "--user-llm", "u",
                          "--runs", "2", "--out", str(out)])
            report = json.loads(out.read_text())
        names = report["tau2"]["saved"]
        self.assertEqual(len(set(names)), 2)
        self.assertTrue(all(n.startswith("larch-t-") for n in names), names)
        agent = [json.loads(cmd[cmd.index("--agent-llm-args") + 1]) for cmd, _ in seen]
        self.assertEqual([a["seed"] for a in agent], [0, 1])
        self.assertEqual({stdin for _, stdin in seen}, {run.subprocess.DEVNULL})

    def test_tau2_pass1_leaves_errors_out_and_its_strict_twin_counts_them_as_fails(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "results.json"
            f.write_text(json.dumps({"simulations": [
                {"task_id": "1", "reward_info": {"reward": 1.0}}, {"task_id": "2", "reward_info": {"reward": 0.0}},
                {"task_id": "3", "reward_info": None}, {"task_id": "4", "reward_info": {"reward": 1.0}}]}))
            self.assertEqual(run.tau2_scores(f), {"pass^1": 2 / 3, "pass^1_errors_fail": 0.5, "errors": 1,
                                                  "tasks": 4})

    def test_tau2_needs_a_user_model(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
            run.main(["tau2", "--url", "http://x/v1", "--model", "m", "--out", "/nonexistent/r.json",
                      "--tau2-dir", "/x"])
        self.assertEqual(e.exception.code, 2)

    def test_the_gate_fails_a_drop_past_the_baselines_spread(self):
        base = {"when2call": {"runs": [{"accuracy": 0.80}, {"accuracy": 0.84}, {"accuracy": 0.82}]}}
        held = {"when2call": {"runs": [{"accuracy": 0.79}]}}
        dropped = {"when2call": {"runs": [{"accuracy": 0.77}]}}
        self.assertEqual(run.gate(base, held), [])
        self.assertEqual(len(run.gate(base, dropped)), 1)
        self.assertIn("when2call accuracy", run.gate(base, dropped)[0])

    def test_the_gate_fails_a_bench_the_report_lacks(self):
        base = {"when2call": {"runs": [{"accuracy": 0.8}]}}
        self.assertEqual(run.gate(base, {"bfcl": {"runs": [{"simple": 0.9}]}}), ["when2call: missing"])

    def test_the_gate_fails_reports_run_another_way(self):
        base = {"temperature": 0.7, "limit": None, "when2call": {"sha256": "a", "rows": [], "runs": [{"accuracy": 0.8}]}}
        new = {"temperature": 0.7, "limit": 5, "when2call": {"sha256": "b", "rows": [1], "runs": [{"accuracy": 0.8}]}}
        self.assertEqual(run.gate(base, new), ["limit: None in the baseline, 5 here",
                                               "when2call sha256: differs from the baseline"])

    def test_the_gate_skips_counts(self):
        base = {"when2call": {"runs": [{"accuracy": 0.8, "false_calls": 1, "errors": 0}]}}
        new = {"when2call": {"runs": [{"accuracy": 0.8, "false_calls": 0, "errors": 0}]}}
        self.assertEqual(run.gate(base, new), [])


if __name__ == "__main__":
    unittest.main()
