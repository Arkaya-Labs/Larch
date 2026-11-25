"""Offline checks of the research bench core: corpus, graders, citations, scorer."""

import base64
import contextlib
import gzip
import hashlib
import http.client
import http.server
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch  # noqa: E402
import run  # noqa: E402


class StubModel:
    """Answers each chat with the next canned reply and records the messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages, tools=None):
        self.sent.append(messages)
        return {"message": {"role": "assistant", "content": self.replies.pop(0)}, "usage": {}}


class GzipReply:
    """A urlopen response whose body is `body`, sent as Content-Encoding: gzip."""

    headers = {"Content-Encoding": "gzip"}

    def __init__(self, body, url="http://x"):
        self.body = body
        self.url = url
        self.asked = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        self.asked = max(self.asked or 0, n)
        out, self.body = (self.body, b"") if n < 0 else (self.body[:n], self.body[n:])
        return out

    def geturl(self):
        return self.url


class CorpusTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)  # runs after the corpus closes (LIFO)
        self.dir = td.name
        self.docs = os.path.join(self.dir, "docs.jsonl")
        self.index = os.path.join(self.dir, "index.db")

    def write(self, docs):
        with open(self.docs, "w", encoding="utf-8") as f:
            for i, (title, text) in enumerate(docs):
                row = {"docid": "d%d" % i, "url": "http://x/%d" % i, "title": title, "text": text}
                f.write(json.dumps(row) + "\n")

    def open(self):
        corpus = run.Corpus(self.docs, self.index)
        self.addCleanup(corpus.close)
        return corpus

    def test_search_ranks_the_matching_doc_first(self):
        self.write([
            ("Paris", "The Eiffel Tower is in Paris."),
            ("Rust", "Rust is a systems programming language."),
            ("Bread", "Sourdough needs flour, water and salt."),
        ])
        hits = self.open().search("sourdough flour")
        self.assertEqual(hits[0]["docid"], "d2")
        self.assertEqual(set(hits[0]), {"docid", "title", "url", "snippet"})
        self.assertIn("flour", hits[0]["snippet"])

    def test_search_returns_five_hits_by_default(self):
        self.write([("Alpha %d" % i, "alpha beta") for i in range(7)])
        corpus = self.open()
        self.assertEqual(len(corpus.search("alpha")), 5)
        self.assertEqual(len(json.loads(run.call_tool(corpus, "search", '{"query": "alpha"}')[0])), 5)

    def test_search_survives_fts_syntax(self):
        self.write([("vLLM", "vLLM serves models."), ("Other", "nothing here")])
        corpus = self.open()
        self.assertIsInstance(corpus.search('"vLLM" AND -(x:y) NEAR* '), list)
        self.assertEqual(run.fts_query(""), "")
        self.assertEqual(corpus.search("!!"), [])

    def test_fetch_cuts_long_text(self):
        self.write([("Long", "x" * 13000)])
        corpus = self.open()
        got = corpus.fetch("d0")
        marker = "\n[cut at 12000 of 13000 characters]"
        self.assertEqual(got["text"], "x" * 12000 + marker)
        self.assertEqual(len(corpus.text("d0")), 13000)
        self.assertIsNone(corpus.fetch("nope"))
        self.assertIsNone(corpus.text("nope"))

    def test_index_rebuilds_when_docs_change(self):
        self.write([("Old", "alpha beta")])
        corpus = run.Corpus(self.docs, self.index)
        self.assertEqual(corpus.search("alpha")[0]["docid"], "d0")
        corpus.close()
        self.write([("New", "gamma delta epsilon"), ("More", "zeta")])
        corpus = self.open()
        self.assertEqual(corpus.search("alpha"), [])
        self.assertEqual(corpus.search("gamma")[0]["docid"], "d0")


class GraderTest(unittest.TestCase):
    def test_normalize_and_graders(self):
        self.assertTrue(run.grade_exact("x\nAnswer: The Eiffel Tower.", "eiffel tower"))
        self.assertFalse(run.grade_exact("Answer: Eiffel Towers", "eiffel tower"))
        self.assertTrue(run.grade_contains("Answer: about 42 people", "42"))
        self.assertFalse(run.grade_contains("Answer: 420", "42"))
        self.assertFalse(run.grade_contains("Answer: anything", "the"))
        self.assertEqual(run.normalize("Zoë"), run.normalize("Zoe"))
        self.assertFalse(run.grade_contains("Answer: 3.5", "3"))  # a number stays whole
        self.assertFalse(run.grade_contains("Answer: version 2.5", "2"))
        self.assertTrue(run.grade_exact("Answer: 1000", "1,000"))
        self.assertTrue(run.grade_contains("Answer: Tarn 0.9", "0.9"))
        self.assertTrue(run.grade_contains("Answer: Paris, France", "Paris"))
        self.assertTrue(run.grade_contains("Answer: 0.9.", "0.9"))
        self.assertEqual(run.answer_line("Answer: a\nanswer : b"), "b")

    def test_answer_line_is_the_one_line_after_the_marker(self):
        self.assertTrue(run.grade_exact("Answer: 7421\n\nI hope this helps.", "7421"))
        self.assertFalse(run.grade_contains("Answer: 41\nOne older page said 42", "42"))
        self.assertEqual(run.answer_line("Answer:\n\n **7421** \nmore"), "7421")
        self.assertEqual(run.answer_line("Answer:  \n"), "")

    def test_a_citation_on_the_answer_line_is_not_part_of_the_answer(self):
        self.assertTrue(run.grade_exact("Answer: 7421 [doc:tarn-server-guide]", "7421"))
        self.assertFalse(run.grade_contains("Answer: 1987 [doc:1999]", "1999"))
        self.assertEqual(run.answer_line("Answer: 7421 [doc:a] [doc:b]"), "7421")
        self.assertEqual(run.answer_line("no marker, last line 7421 [doc:a]"), "no marker, last line 7421")

    def test_answer_line_reads_decorated_markers(self):
        for response in ("**Answer:** 0.8", "## Answer: 0.8", "> __Answer__: 0.8", "**Answer: 0.8**", "*Answer*: _0.8_", "**Final Answer:** 0.8"):
            with self.subTest(response):
                self.assertEqual(run.answer_line("reasoning\n" + response), "0.8")
        self.assertTrue(run.grade_exact("**Answer:** 0.8", "0.8"))
        self.assertEqual(run.answer_line("**Final Answer:** 42\n\nHope this helps."), "42")

    def test_answer_line_without_a_marker_is_the_last_line(self):
        self.assertEqual(run.answer_line("the gold is 42 in the reasoning\nfinal words\n\n"), "final words")
        self.assertFalse(run.grade_contains("the gold is 42 in the reasoning\nso it is large", "42"))
        self.assertEqual(run.answer_line(""), "")

    def test_parse_verdict(self):
        self.assertIs(run.parse_verdict("VERDICT: no\nwait\nverdict:  YES"), True)
        self.assertIs(run.parse_verdict("VERDICT: yes\nVERDICT: no"), False)
        self.assertIsNone(run.parse_verdict("looks fine to me"))

    def test_judge_prompts_are_pinned(self):
        digest = hashlib.sha256((run.JUDGE_PROMPT + run.RUBRIC_PROMPT).encode()).hexdigest()
        self.assertEqual(digest, run.JUDGE_SHA256)

    def test_judge_asks_the_model_and_needs_a_verdict(self):
        model = StubModel("fine.\nVERDICT: yes", "VERDICT: no", "no idea")
        judge = run.Judge(model)
        self.assertTrue(judge.answer("Q?", "gold", "resp"))
        self.assertEqual([m["role"] for m in model.sent[0]], ["user"])
        self.assertIn("gold", model.sent[0][0]["content"])
        self.assertFalse(judge.point("Q?", "it says X", "resp"))
        with self.assertRaisesRegex(RuntimeError, "judge gave no verdict"):
            judge.answer("Q?", "gold", "resp")

    def test_judge_treats_empty_content_as_no_verdict(self):
        with self.assertRaisesRegex(RuntimeError, "judge gave no verdict"):
            run.Judge(StubModel(None)).answer("Q?", "gold", "resp")

    def test_rubric_needs_a_judge(self):
        task = {"id": "r", "question": "Q?", "rubric": ["a", "b"]}
        with self.assertRaises(ValueError):
            run.grade(task, "resp", "judge", None)
        judge = run.Judge(StubModel("VERDICT: yes", "VERDICT: no"))
        self.assertEqual(run.grade(task, "resp", "judge", judge), {"passed": False, "points": [True, False]})
        # no answer: all points fail and the judge is never asked
        self.assertEqual(run.grade(task, None, "judge", judge), {"passed": False, "points": [False, False]})

    def test_grade_answer_tasks(self):
        task = {"id": "a", "question": "Q?", "answer": "Paris"}
        self.assertEqual(run.grade(task, "Answer: paris", "exact", None), {"passed": True, "points": None})
        self.assertEqual(run.grade(task, "Answer: Paris, France", "contains", None)["passed"], True)
        self.assertEqual(run.grade(task, None, "exact", None), {"passed": False, "points": None})
        with self.assertRaises(ValueError):
            run.grade(task, "Answer: Paris", "judge", None)
        with self.assertRaises(ValueError):
            run.grade(task, "Answer: Paris", "fuzzy", None)
        judge = run.Judge(StubModel("VERDICT: yes"))
        self.assertTrue(run.grade(task, "Answer: Paris", "judge", judge)["passed"])


class CitationTest(unittest.TestCase):
    def test_quotes_and_citations(self):
        with tempfile.TemporaryDirectory() as d:
            docs = os.path.join(d, "docs.jsonl")
            with open(docs, "w", encoding="utf-8") as f:
                f.write(json.dumps({"docid": "d1", "url": "u", "title": "T",
                                    "text": "It was the best   of times,\nit was the worst of times."}) + "\n")
            corpus = run.Corpus(docs, os.path.join(d, "index.db"))
            try:
                response = (
                    'It said "the best of times" [doc:d1] and “the worst of times,” [doc:d2], '
                    'also "an uncited phrase here" but "two words" [doc:d1].\nAnswer: x'
                )
                self.assertEqual(run.quotes(response), [("the best of times", "d1"), ("the worst of times,", "d2")])
                self.assertEqual(run.citations(response, corpus), (1, 2))
                # an unmatched quote mark (an inch mark) must not shift the pairing of the quotes after it
                inch = 'a 27" monitor; it says "it was the worst of times" [doc:d1]'
                self.assertEqual(run.quotes(inch), [("it was the worst of times", "d1")])
                self.assertEqual(run.citations(inch, corpus), (1, 1))
                # a span never runs over a line break
                self.assertEqual(run.quotes('he said "it was the\nbest of times" [doc:d1]'), [])
                self.assertEqual(run.citations('"not in the text" [doc:d1]', corpus), (0, 1))
                curly = 'He wrote “it was the worst of times.” [doc:d1]'
                self.assertEqual(run.citations(curly, corpus), (1, 1))
                self.assertEqual(run.citations("no quotes", corpus), (0, 0))
            finally:
                corpus.close()


class SummarizeTest(unittest.TestCase):
    def row(self, task, passed, **kw):
        base = {"task": task, "passed": passed, "tool_calls": 2, "pages": 1, "wall_s": 3.0,
                "prompt_tokens": 100, "completion_tokens": 10, "quotes_found": 1, "quotes_total": 2,
                "error": None}
        base.update(kw)
        return base

    def test_summarize_pass_metrics(self):
        rows = [self.row("A", True) for _ in range(4)]
        rows += [self.row("B", True)] * 3 + [self.row("B", False, error="boom")]
        s = run.summarize(rows)
        self.assertEqual((s["tasks"], s["runs"]), (2, 8))
        self.assertEqual(s["pass@1"], 0.875)
        self.assertEqual(s["pass^4"], 0.5)
        self.assertEqual(s["citation_accuracy"], 0.5)
        self.assertEqual(s["tool_calls"], {"total": 16, "mean": 2})
        self.assertEqual(s["tokens"], {"prompt": 800, "completion": 80})
        self.assertEqual(s["errors"], 1)
        short = run.summarize(rows[:3] + [self.row("B", True)] * 3)
        self.assertIsNone(short["pass^4"])
        self.assertIsNone(run.summarize([self.row("A", True, quotes_total=0, quotes_found=0)])["citation_accuracy"])


@unittest.skipIf(os.name == "nt", "the stand-in agent is a #! script")
class AgentTest(unittest.TestCase):
    """--agent: larch's own research skill, as a stand-in binary that prints what `run research --json` prints."""

    URL, MADE_UP = "https://a.example/tower", "https://made.up/x"

    FAKE = r"""
import hashlib, json, os, sys
args = sys.argv[1:]
data, question = args[args.index("--data-dir") + 1], args[args.index("--question") + 1]
effort = args[args.index("--effort") + 1]
if question == "fail":
    print("research failed; its steps are in the log", file=sys.stderr)
    sys.exit(1)
url, sha = "https://a.example/tower", "ab" * 32
folder = os.path.join(data, "agent", "web", hashlib.sha256(url.encode()).hexdigest()[:16])
os.makedirs(folder, exist_ok=True)
with open(os.path.join(folder, sha + ".json"), "w", encoding="utf-8") as f:
    json.dump({"url": url, "text": "# Tower\n\nIt is 212 metres tall and has 48 floors.\n"}, f)
print("loaded the model in 3 s", file=sys.stderr)
for step in [{"step": "load", "wall_ms": 1},
             {"step": "search", "detail": {"queries": ["tower height", "tower floors"], "hits": 4, "failed": []}},
             {"step": "fetch", "detail": {"urls": [url], "pages": 1}},
             {"kind": "turn", "step": "plan", "prompt_tokens": 100, "generated_tokens": 20}]:
    print(json.dumps(step), file=sys.stderr)
source = lambda n, u, s, q: {"n": n, "url": u, "final_url": u, "title": "", "fetched_at": "", "sha256": s,
                             "cited": {"file": u, "line": 1, "claim": "", "quote": q}}
report = {"type": "report", "unverified": [], "stopped": "the planner stopped: done", "pages": 1, "fixed": 1,
          "removed": 2, "sections": [{"title": "Tower", "text": "It is 212 metres tall [1], with 48 floors [1, 3]."}],
          "sources": [source(1, url, sha, "It is 212 metres tall and has 48 floors."),
                      source(2, "https://made.up/x", "cd" * 32, "A quote from nowhere.")]}
print(json.dumps({"markdown": "# %s (%s)" % (question, effort), "report": report}))
"""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.data = os.path.join(td.name, "data")
        script = os.path.join(td.name, "larch-agent")
        with open(script, "w", encoding="utf-8") as f:
            f.write("#!%s -I\n%s" % (sys.executable, self.FAKE))
        os.chmod(script, 0o755)
        self.agent = [script]

    def test_a_report_is_scored_with_its_audit_steps_and_code_checks(self):
        task = {"question": "How tall is the tower?", "effort": "deep"}
        row = run.run_agent_task(self.agent, "m", self.data, task)
        self.assertIsNone(row["error"])
        self.assertTrue(row["answer"].startswith("# How tall is the tower? (deep)"))
        # the report's own verification counts: sentences without a source its writer fixed, and that Rust cut
        self.assertEqual((row["effort"], row["unsourced_fixed"], row["unsourced_removed"]), ("deep", 1, 2))
        self.assertEqual((row["tool_calls"], row["pages"], row["prompt_tokens"], row["completion_tokens"]),
                         (2, 1, 100, 20))
        self.assertEqual(row["stopped"], "the planner stopped: done")
        # the made-up source was never fetched and has no saved page; [3] names no source
        self.assertEqual(row["checks"], {"invented_urls": [self.MADE_UP], "quotes_missing": 1, "dangling_markers": [3]})
        self.assertEqual((row["quotes_found"], row["quotes_total"]), (1, 2))

    def test_a_failed_run_is_an_error_row_with_its_last_line(self):
        row = run.run_agent_task(self.agent, "m", self.data, {"question": "fail"})
        self.assertEqual(row["error"], "exit 1: research failed; its steps are in the log")
        self.assertIsNone(row["answer"])
        self.assertEqual(row["effort"], "standard")

    def test_the_summary_gives_partial_credit_the_checks_and_the_stops(self):
        base = {"task": "A", "passed": False, "tool_calls": 2, "pages": 1, "wall_s": 3.0, "prompt_tokens": 1,
                "completion_tokens": 1, "quotes_found": 1, "quotes_total": 2, "error": None}
        clean = {"invented_urls": [], "quotes_missing": 0, "dangling_markers": []}
        rows = [dict(base, points=[True, False, True, True], stopped="a round found no new fact", checks=clean,
                     unsourced_fixed=2, unsourced_removed=0),
                dict(base, points=[False, False], stopped="a round found no new fact", unsourced_fixed=1,
                     unsourced_removed=1, checks={"invented_urls": ["u"], "quotes_missing": 1, "dangling_markers": [3, 4]})]
        s = run.summarize(rows)
        self.assertEqual(s["unsourced"], {"fixed": 3, "removed": 1})
        self.assertEqual(s["points_share"], 0.375)
        self.assertEqual(s["checks"], {"invented_urls": 1, "quotes_missing": 1, "dangling_markers": 2})
        self.assertEqual(s["stopped"], {"a round found no new fact": 2})

    def test_a_run_of_the_skill_is_graded_and_fails_its_gate_on_an_invented_url(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "VERDICT: yes"}}], "usage": {}}
        with tempfile.TemporaryDirectory() as d, chat_server(reply) as (url, _):
            out = os.path.join(d, "report.json")
            argv = ["--set", "iris", "--agent", self.agent[0], "--model", "m", "--agent-data", self.data,
                    "--runs", "1", "--limit", "1", "--judge-url", url, "--judge-model", "j", "--out", out]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(run.main(argv), 1)
            self.assertIn("code checks FAILED", err.getvalue())
            with open(out, encoding="utf-8") as f:
                report = json.load(f)
        row = report["rows"][0]
        self.assertEqual((row["passed"], row["quotes_found"], row["quotes_total"]), (True, 1, 2))
        self.assertEqual((report["agent"], report["docs_sha256"], report["temperature"]), (self.agent[0], None, None))
        self.assertEqual(report["summary"]["points_share"], 1.0)

    def test_a_report_counts_the_facts_it_kept(self):
        row = run.run_agent_task(self.agent, "m", self.data, {"question": "How tall is the tower?"})
        # the made-up source's quote is on no saved page: it is not kept
        self.assertEqual(row["facts_kept"], 1)
        # the model's load is its own step, left out of the run's time
        self.assertEqual(row["load_s"], 0.001)

    def test_a_live_run_passes_only_inside_its_time_and_prints_its_wall_clock(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "VERDICT: yes"}}], "usage": {}}
        with tempfile.TemporaryDirectory() as d, chat_server(reply) as (url, _):
            out = os.path.join(d, "report.json")
            argv = ["--set", "iris", "--agent", self.agent[0], "--model", "m", "--agent-data", self.data,
                    "--runs", "1", "--limit", "1", "--judge-url", url, "--judge-model", "j", "--out", out,
                    "--live", "--budget", "0"]
            with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
                run.main(argv)
            with open(out, encoding="utf-8") as f:
                report = json.load(f)
        row = report["rows"][0]
        # the judge said yes, but the report came after the budget: facts right in time is the score
        self.assertEqual((row["passed"], row["in_time"], row["facts_kept"]), (False, False, 1))
        self.assertEqual(report["live_budget_s"], 0)
        self.assertEqual(report["summary"]["in_time"], 0)
        self.assertEqual(report["summary"]["facts_kept"]["total"], 1)
        self.assertIn("wall_s", printed.getvalue())
        self.assertIn("facts_kept", printed.getvalue())

    def test_live_needs_the_agent(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run.main(["--set", "iris", "--live", "--url", "u", "--model", "m"]), 2)
        self.assertIn("--live runs larch's own research", err.getvalue())

    def test_agent_runs_need_the_iris_set_a_model_and_its_data_folder(self):
        for argv in (["--set", "iris", "--agent", "x", "--model", "m"],
                     ["--set", "frames", "--agent", "x", "--model", "m", "--agent-data", self.data]):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(run.main(argv), 2)
            self.assertIn("--agent runs the iris set", err.getvalue())


class CallToolTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        docs = os.path.join(td.name, "docs.jsonl")
        with open(docs, "w", encoding="utf-8") as f:
            f.write(json.dumps({"docid": "d1", "url": "u", "title": "T", "text": "alpha beta gamma"}) + "\n")
        self.corpus = run.Corpus(docs, os.path.join(td.name, "index.db"))
        self.addCleanup(self.corpus.close)

    def test_call_tool_errors(self):
        for name, args in [("search", "{not json"), ("fetch", "{}"), ("search", "[]"), ("nope", "{}"),
                           ("fetch", '{"doc_id": "zzz"}')]:
            result, fetched = run.call_tool(self.corpus, name, args)
            self.assertIn("error", json.loads(result), (name, args))
            self.assertIsNone(fetched)

    def test_call_tool_results(self):
        result, fetched = run.call_tool(self.corpus, "search", '{"query": "alpha"}')
        self.assertEqual(json.loads(result)[0]["docid"], "d1")
        self.assertIsNone(fetched)
        result, fetched = run.call_tool(self.corpus, "fetch", '{"doc_id": "d1"}')
        self.assertEqual(json.loads(result)["text"], "alpha beta gamma")
        self.assertEqual(fetched, "d1")


class RunTaskTest(unittest.TestCase):
    def corpus(self, td):
        docs = os.path.join(td, "docs.jsonl")
        with open(docs, "w", encoding="utf-8") as f:
            f.write(json.dumps({"docid": "d1", "url": "u", "title": "T", "text": "alpha"}) + "\n")
        return run.Corpus(docs, os.path.join(td, "index.db"))

    def call(self, name, args, i=0):
        return {"id": "c%d" % i, "type": "function", "function": {"name": name, "arguments": args}}

    def test_run_task_stops_after_max_turns(self):
        call = self.call

        class Looper:
            def chat(self, messages, tools=None):
                msg = {"role": "assistant", "content": None, "tool_calls": [call("search", '{"query": "x"}')]}
                return {"message": msg, "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                out = run.run_task(Looper(), corpus, {"id": "t", "question": "Q?"})
            finally:
                corpus.close()
        self.assertEqual(out["error"], "no answer after 24 turns")
        self.assertEqual((out["answer"], out["tool_calls"], out["pages"]), (None, 24, 0))
        self.assertEqual((out["prompt_tokens"], out["completion_tokens"]), (120, 24))

    def test_the_system_prompt_is_sent_and_carries_the_protocol(self):
        model = StubModel("Answer: alpha")
        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                run.run_task(model, corpus, {"id": "t", "question": "Q?"})
            finally:
                corpus.close()
        self.assertEqual(model.sent[0][0], {"role": "system", "content": run.SYSTEM})
        self.assertIn("[doc:", run.SYSTEM)
        self.assertIn("Answer:", run.SYSTEM)

    def test_a_run_past_the_time_cap_stops_before_the_next_model_call(self):
        now, asked = [0.0], []
        call = self.call

        class Slow:
            def chat(self, messages, tools=None):
                asked.append(now[0])
                now[0] += 1000
                msg = {"role": "assistant", "content": None, "tool_calls": [call("search", '{"query": "x"}')]}
                return {"message": msg, "usage": {}}

        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                with unittest.mock.patch.object(run, "time", unittest.mock.Mock(monotonic=lambda: now[0])):
                    out = run.run_task(Slow(), corpus, {"id": "t", "question": "Q?"})
            finally:
                corpus.close()
        self.assertEqual(run.RUN_SECONDS, 1800)
        self.assertEqual(asked, [0.0, 1000.0])  # at 2000 s the next call is not made
        self.assertEqual((out["error"], out["answer"], out["wall_s"]), ("over 1800 s", None, 2000.0))

    def test_run_task_records_a_model_error(self):
        class Down:
            def chat(self, messages, tools=None):
                raise OSError("connection refused")

        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                out = run.run_task(Down(), corpus, {"id": "t", "question": "Q?"})
            finally:
                corpus.close()
        self.assertEqual((out["error"], out["answer"], out["tool_calls"]), ("connection refused", None, 0))

    def test_run_task_counts_a_failed_call_and_keeps_the_history(self):
        calls = [self.call("fetch", "{}", 1), self.call("fetch", '{"doc_id": "d1"}', 2)]
        out, model = self.answer_after({"role": "assistant", "content": None, "tool_calls": calls})
        self.assertEqual((out["answer"], out["tool_calls"], out["pages"], out["error"]), ("Answer: alpha", 2, 1, None))
        roles = [m["role"] for m in model.sent[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(model.sent[1][3]["tool_call_id"], "c1")

    def answer_after(self, first):
        replies = [first, {"role": "assistant", "content": "Answer: alpha"}]

        class Scripted:
            sent = []

            def chat(self, messages, tools=None):
                self.sent.append(list(messages))
                return {"message": replies[len(self.sent) - 1], "usage": {}}

        model = Scripted()
        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                return run.run_task(model, corpus, {"id": "t", "question": "Q?"}), model
            finally:
                corpus.close()

    def test_odd_tool_call_shapes_get_an_error_reply_and_the_run_goes_on(self):
        odd = [{}, {"id": "a", "function": None}, {"id": "b", "function": {"name": "fetch"}},
               {"id": "c", "function": {"arguments": "{}"}}, "junk"]
        out, model = self.answer_after({"role": "assistant", "content": None, "tool_calls": odd})
        self.assertEqual((out["answer"], out["tool_calls"], out["error"]), ("Answer: alpha", 5, None))
        replies = model.sent[1][3:]
        self.assertEqual([m["tool_call_id"] for m in replies], ["", "a", "b", "c", ""])
        for m in replies:
            self.assertIn("error", json.loads(m["content"]))

    def test_tool_calls_that_are_not_a_list_fail_the_run(self):
        out, _ = self.answer_after({"role": "assistant", "content": None, "tool_calls": "oops"})
        self.assertEqual((out["answer"], out["tool_calls"]), (None, 0))
        self.assertIn("tool_calls", out["error"])

    def test_a_disconnect_mid_body_fails_the_run(self):
        class Cut:
            def chat(self, messages, tools=None):
                raise http.client.IncompleteRead(b"")

        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(td)
            try:
                out = run.run_task(Cut(), corpus, {"id": "t", "question": "Q?"})
            finally:
                corpus.close()
        self.assertIn("IncompleteRead", out["error"])


@contextlib.contextmanager
def chat_server(reply):
    """Answers every POST with `reply` (a dict, or raw bytes); yields (url, seen) with seen = [(path, body, user agent)]."""
    seen = []
    payload = reply if isinstance(reply, bytes) else json.dumps(reply).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers["User-Agent"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port, seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class RequestTest(unittest.TestCase):
    def test_openai_model_posts_a_chat_request(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        with chat_server(reply) as (url, seen):
            got = run.OpenAIModel(url, "m").chat([{"role": "user", "content": "q"}], tools=run.TOOLS)
        path, body, agent = seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["tools"], run.TOOLS)
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("seed", body)
        self.assertEqual(got, {"message": {"role": "assistant", "content": "hi"},
                               "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
        self.assertEqual(agent, fetch.UA)  # one user agent for every request the bench makes
        self.assertIs(run.UA, fetch.UA)

    def test_openai_model_sends_its_temperature_and_seed(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        with chat_server(reply) as (url, seen):
            run.OpenAIModel(url, "m", temperature=0.7, seed=0).chat([{"role": "user", "content": "q"}])
        self.assertEqual((seen[0][1]["temperature"], seen[0][1]["seed"]), (0.7, 0))  # seed 0 is a seed

    def test_openai_model_fails_a_bad_reply_with_a_runtime_error(self):
        # the run records RuntimeError as "the server failed"; a ValueError would escape as a bug
        for reply in ({"choices": [{"message": None}]}, {"choices": []}, b"<html>502</html>", b"\xff\xfe"):
            with self.subTest(reply), chat_server(reply) as (url, _):
                with self.assertRaises(RuntimeError):
                    run.OpenAIModel(url, "m").chat([{"role": "user", "content": "q"}])

    def test_fake_model_fails_with_a_runtime_error_when_the_ranking_is_off(self):
        model = run.FakeModel({"fake": {"search": "x", "answer": "Answer [doc:b]"}})
        model.chat([])
        model.chat([{"content": json.dumps([{"docid": "a"}])}])
        with self.assertRaisesRegex(RuntimeError, "ranked 'a' first"):
            model.chat([])


class CliTest(unittest.TestCase):
    def write_set(self, d, *tasks):
        folder = os.path.join(d, "frames")
        os.mkdir(folder)
        with open(os.path.join(folder, "tasks.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(t) + "\n" for t in tasks))
        with open(os.path.join(folder, "docs.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"docid": "d", "url": "u", "title": "T", "text": "alpha"}) + "\n")

    def real_run(self, d, url, *extra):
        out = os.path.join(d, "report.json")
        argv = ["--set", "frames", "--url", url, "--model", "m", "--data", d, "--out", out] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run.main(argv)
        with open(out, encoding="utf-8") as f:
            return code, json.load(f)

    def test_the_report_names_what_it_scored(self):
        here = os.path.dirname(os.path.abspath(__file__))

        def sha(*path):
            with open(os.path.join(here, *path), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = os.path.join(d, "report.json")
            self.assertEqual(run.main(["--fake", "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["tasks_sha256"], sha("fixtures", "tasks.jsonl"))
        self.assertEqual(report["docs_sha256"], sha("fixtures", "docs.jsonl"))
        self.assertEqual(report["system_prompt_sha256"], hashlib.sha256(run.SYSTEM.encode("utf-8")).hexdigest())
        self.assertEqual((report["temperature"], report["seeds"]), (0.7, "run index"))
        graders = {r["task"]: r["grader"] for r in report["rows"]}
        self.assertEqual(graders, {"fake-exact": "exact", "fake-contains": "contains", "fake-rubric": "rubric"})

    def test_sha256_file_reads_in_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big")
            blob = os.urandom(1 << 20) + b"tail"
            with open(path, "wb") as f:
                f.write(blob)
            self.assertEqual(run.sha256_file(path), hashlib.sha256(blob).hexdigest())

    def test_a_real_run_sends_the_temperature_and_the_run_index_as_seed(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "Answer: 42\nVERDICT: yes"}}], "usage": {}}
        for extra, temperature in (([], 0.7), (["--temperature", "0.2"], 0.2)):
            with self.subTest(extra), tempfile.TemporaryDirectory() as d, chat_server(reply) as (url, seen):
                self.write_set(d, {"id": "t", "question": "Q?", "answer": "42"})
                code, report = self.real_run(d, url, "--runs", "2", "--grader", "judge",
                                             "--judge-url", url, "--judge-model", "j", *extra)
                bodies = [b for _, b, _ in seen]
                agent = [b for b in bodies if "tools" in b]
                judge = [b for b in bodies if "tools" not in b]
                self.assertEqual(code, 0)
                self.assertEqual([(b["temperature"], b["seed"]) for b in agent], [(temperature, 0), (temperature, 1)])
                self.assertEqual([(b["temperature"], "seed" in b) for b in judge], [(0, False)] * 2)
                self.assertEqual((report["temperature"], report["seeds"]), (temperature, "run index"))

    def test_a_judge_that_fails_or_gives_no_verdict_fails_the_run(self):
        class Silent:
            def chat(self, messages, tools=None):
                return {"message": {"role": "assistant", "content": "maybe"}, "usage": {}}

        class Down:
            def chat(self, messages, tools=None):
                raise RuntimeError("judge down")

        for judge in (Silent, Down):
            with self.subTest(judge.__name__), tempfile.TemporaryDirectory() as d, \
                    unittest.mock.patch.object(run, "FakeJudge", judge), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                out = os.path.join(d, "report.json")
                run.main(["--fake", "--runs", "1", "--out", out])
                with open(out, encoding="utf-8") as f:
                    row = [r for r in json.load(f)["rows"] if r["task"] == "fake-rubric"][0]
                self.assertIs(row["passed"], False)
                self.assertTrue(row["error"])

    def test_the_exit_code_is_1_when_every_run_errored(self):
        class Down:
            def __init__(self, url, name, timeout=600, temperature=0, seed=None):
                self.seed = seed

            def chat(self, messages, tools=None):
                if self.seed == 0 or self.always:
                    raise OSError("connection refused")
                return {"message": {"role": "assistant", "content": "Answer: 42"}, "usage": {}}

        for always, runs, code in ((True, "2", 1), (False, "2", 0)):  # one good run keeps the exit code at 0
            with self.subTest(always=always), tempfile.TemporaryDirectory() as d:
                self.write_set(d, {"id": "t", "question": "Q?", "answer": "42"})
                model = type("Model", (Down,), {"always": always})
                with unittest.mock.patch.object(run, "OpenAIModel", model):
                    got, report = self.real_run(d, "http://x", "--runs", runs)
                self.assertEqual((got, report["summary"]["errors"]), (code, 2 if always else 1))

    def test_a_task_grader_wins_and_needs_its_judge(self):
        task = {"question": "Q?", "answer": "7421", "grader": "exact"}
        self.assertFalse(run.grade(task, "Answer: port 7421", "contains", None)["passed"])
        self.assertTrue(run.grade(dict(task, grader="contains"), "Answer: port 7421", "exact", None)["passed"])
        self.assertTrue(run.grade({k: v for k, v in task.items() if k != "grader"}, "Answer: port 7421",
                                  "contains", None)["passed"])
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            self.write_set(d, dict(task, grader="judge"))
            self.assertEqual(run.main(["--set", "frames", "--url", "http://x", "--model", "m", "--data", d]), 2)
            self.assertIn("--judge-url", shown.getvalue())

    def test_run_py_starts_under_isolated_mode(self):
        # -I drops the script's directory from sys.path; run.py must put it back for `from fetch import UA`.
        # -P (3.11+) drops only that: -I also ignores PYTHONHOME, which some installs (the Windows runner's) need to start
        safe_path = "-P" if sys.version_info >= (3, 11) else "-I"
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
            done = subprocess.run([sys.executable, safe_path, script, "--fake"], cwd=d, capture_output=True, text=True)
        # stderr is not asserted empty: some interpreters warn at start ("Could not find platform independent libraries")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("fake: ok", done.stdout)

    def test_a_real_run_needs_its_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run.main(["--set", "frames", "--data", d]), 2)
            self.assertEqual(run.main(["--set", "iris", "--url", "http://x", "--model", "m", "--data", d]), 2)

    def test_iris_asks_for_the_judge_before_the_data(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            self.assertEqual(run.main(["--set", "iris", "--url", "http://x", "--model", "m", "--data", d]), 2)
            self.assertIn("--judge-url", shown.getvalue())
            self.assertEqual(run.main(["--set", "frames", "--url", "http://x", "--model", "m", "--data", d]), 2)
            self.assertIn("missing " + os.path.join(d, "frames", "tasks.jsonl"), shown.getvalue())

    def test_the_report_is_saved_after_each_task(self):
        class Interrupted(run.FakeModel):
            def chat(self, messages, tools=None):
                if self.script["search"].startswith("tarn paged"):
                    raise KeyboardInterrupt
                return run.FakeModel.chat(self, messages, tools)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "report.json")
            with unittest.mock.patch.object(run, "FakeModel", Interrupted), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    run.main(["--fake", "--runs", "1", "--out", out])
            with open(out, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual([r["task"] for r in report["rows"]], ["fake-exact"])
            self.assertEqual(report["summary"]["runs"], 1)
            self.assertFalse(os.path.exists(out + ".part"))


    def test_an_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            out = os.path.join(d, "report.json")
            with open(out, "w", encoding="utf-8") as f:
                f.write("saved")
            with unittest.mock.patch.object(run, "FakeModel", side_effect=AssertionError("model called")):
                self.assertEqual(run.main(["--fake", "--out", out]), 2)
            self.assertIn(out + " exists", shown.getvalue())
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "saved")

    def test_a_config_error_in_grading_is_not_a_failed_run(self):
        with unittest.mock.patch.object(run, "grade", side_effect=ValueError("unknown grader 'x'")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "unknown grader"):
                run.main(["--fake", "--runs", "1"])

    def test_a_judge_that_fails_fails_that_run_only(self):
        class Down:
            def chat(self, messages, tools=None):
                raise RuntimeError("judge gave no verdict")

        with unittest.mock.patch.object(run, "FakeJudge", Down), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(["--fake", "--runs", "1"]), 1)  # the rubric run fails, the others still ran

    def test_a_real_run_uses_no_temp_dir_and_summarizes_once_per_save(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "It is.\nAnswer: 42"}}], "usage": {}}
        with tempfile.TemporaryDirectory() as d, chat_server(reply) as (url, _):
            folder = os.path.join(d, "frames")
            os.mkdir(folder)
            with open(os.path.join(folder, "tasks.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "t", "question": "Q?", "answer": "42"}) + "\n")
            with open(os.path.join(folder, "docs.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"docid": "d", "url": "u", "title": "T", "text": "alpha"}) + "\n")
            out = os.path.join(d, "report.json")
            argv = ["--set", "frames", "--url", url, "--model", "m", "--runs", "1", "--data", d, "--out", out]
            with unittest.mock.patch.object(tempfile, "TemporaryDirectory", side_effect=AssertionError("temp dir")), \
                    unittest.mock.patch.object(run, "summarize", wraps=run.summarize) as summarize, \
                    contextlib.redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(run.main(argv), 0)
            self.assertEqual(summarize.call_count, 2)  # after the one task, and the final save
            self.assertEqual(json.loads(shown.getvalue())["pass@1"], 1.0)


class IrisTest(unittest.TestCase):
    def test_iris_set_shape(self):
        kinds = ["competitor", "survey", "changes", "compare", "howto"]
        tasks = run._read_jsonl(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris.jsonl"))
        self.assertEqual([t["id"] for t in tasks], ["iris-%03d" % i for i in range(1, 41)])
        self.assertEqual([t["kind"] for t in tasks], [k for k in kinds for _ in range(8)])
        for t in tasks:
            with self.subTest(t["id"]):
                self.assertTrue(t["question"].strip())
                self.assertEqual(t["as_of"], "2026-10-25")
                self.assertTrue(2 <= len(t["sources"]) <= 5)
                self.assertTrue(all(u.startswith(("http://", "https://")) for u in t["sources"]))
                self.assertTrue(3 <= len(t["rubric"]) <= 6)
                self.assertTrue(all(isinstance(p, str) and p.strip() for p in t["rubric"]))
                self.assertNotIn("@", json.dumps(t))


def _encrypt(text):
    """The inverse of fetch.decrypt, written out so the test does not call what it checks."""
    raw = text.encode("utf-8")
    key = hashlib.sha256(fetch.CANARY.encode("utf-8")).digest()
    key = (key * (len(raw) // len(key) + 1))[:len(raw)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode("ascii")


class FetchTest(unittest.TestCase):
    def test_decrypt_roundtrip(self):
        for text in ("a", "caf\u00e9 " * 40, "x" * 33):
            self.assertEqual(fetch.decrypt(_encrypt(text)), text)

    def test_bcp_task_decrypts_the_query_and_the_answer_only(self):
        row = {"query_id": "7", "query": _encrypt("a question?"), "answer": _encrypt("an answer"), "docs": "left alone"}
        self.assertEqual(fetch._bcp_task(row), {"id": "bcp-7", "question": "a question?", "answer": "an answer"})

    def test_doc_title(self):
        self.assertEqual(fetch.doc_title("---\ntitle:  A: B \ndate: 2020\n---\nbody"), "A: B")
        self.assertEqual(fetch.doc_title("---\ndate: 2020\n---\ntitle: no"), "")
        self.assertEqual(fetch.doc_title("title: x\nbody"), "")

    def test_wiki_title(self):
        self.assertEqual(fetch.wiki_title("https://en.m.wikipedia.org/wiki/Coal"), "Coal")
        self.assertEqual(fetch.wiki_title("en.wikipedia.org/wiki/James_A._Garfield"), "James_A._Garfield")
        self.assertEqual(fetch.wiki_title("https://en.wikipedia.org/wiki/Caf%C3%A9#History"), "Caf\u00e9")
        self.assertEqual(fetch.wiki_title("https://en.wikipedia.org/wiki/Paul%20Revere"), "Paul_Revere")  # a doc id has no space
        self.assertIsNone(fetch.wiki_title("https://fr.wikipedia.org/wiki/Coal"))
        self.assertIsNone(fetch.wiki_title("https://example.com/wiki/Coal"))

    def test_html_text(self):
        html = ("<html><head><title> T </title><script>var x = 1;</script></head><body>"
                "<p>Hello<sup class='reference'>[1]</sup> world</p>"
                "<div class='navbox'><div>inner</div>junk</div><span class='mw-editsection'>edit</span>"
                "<table><tr><td>a</td><td> b </td></tr></table><ul><li>one</li><li>two</li></ul>after</body></html>")
        self.assertEqual(fetch.html_text(html), ("T", "Hello world\na | b\none\ntwo\nafter"))

    def test_only_the_first_title_outside_svg_is_the_page_title(self):
        svg = "<svg><title>Chart tooltip</title><g><title>inner</title></g></svg>"
        self.assertEqual(fetch.html_text("<title>Page</title><body>" + svg + "<p>text</p></body>"), ("Page", "text"))
        self.assertEqual(fetch.html_text("<body>" + svg + "<title>Late</title><title>Other</title><p>text</p></body>"),
                         ("Late", "text"))
        self.assertEqual(fetch.html_text("<body>" + svg + "<p>text</p></body>"), ("", "text"))

    def test_github_releases_become_one_doc_each(self):
        releases = json.dumps([
            {"tag_name": "v2.0", "published_at": "2024-02-01T00:00:00Z", "body": "two", "html_url": "https://g/o/r/v2.0"},
            {"tag_name": "v1", "published_at": "2024-01-01T00:00:00Z", "body": None, "html_url": "https://g/o/r/v1"},
        ]).encode()
        taken = {"api-github-com-repos-o-r-releases-v1"}
        docs = fetch.release_docs("https://api.github.com/repos/o/r/releases?per_page=100", releases, taken)
        self.assertEqual(docs, [
            {"docid": "api-github-com-repos-o-r-releases-v2-0", "url": "https://g/o/r/v2.0",
             "title": "o/r v2.0 (2024-02-01)", "text": "## v2.0 (2024-02-01T00:00:00Z)\ntwo"},
            {"docid": "api-github-com-repos-o-r-releases-v1-2", "url": "https://g/o/r/v1",
             "title": "o/r v1 (2024-01-01)", "text": "## v1 (2024-01-01T00:00:00Z)\n"},
        ])

    def test_the_pins_are_enforced(self):
        with unittest.mock.patch.object(fetch, "hf_revision", return_value="x"), \
                self.assertRaisesRegex(SystemExit, "is at revision x, pinned"):
            fetch._browsecomp(tempfile.gettempdir(), None)
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(fetch, "get", return_value=b"other"), \
                self.assertRaisesRegex(SystemExit, "sha256 is .*, pinned " + fetch.FRAMES[1]):
            fetch.main(["frames", "--data", d])
        body = json.dumps({"num_rows_total": 5, "rows": []}).encode()
        with unittest.mock.patch.object(fetch, "get", return_value=body), \
                self.assertRaisesRegex(SystemExit, "has 5 rows, pinned 830"):
            next(fetch.rows(fetch.QUERIES, "test", 0, 1, ()))

    def test_web_text_formats(self):
        for name in ("README.md", "faq.mdx", "notes.txt"):
            self.assertEqual(fetch.web_text("https://x/" + name, b"---\ntitle: T\n---\n<b>as is</b>")[1],
                             "---\ntitle: T\n---\n<b>as is</b>")
        self.assertEqual(fetch.web_text("https://x/y", b'{"a": 1}'), ("", '{"a": 1}'))
        self.assertEqual(fetch.web_text("https://x/y", b"<title>T</title><p>hi</p>"), ("T", "hi"))

    def test_slug_deduplicates(self):
        taken = set()
        self.assertEqual(fetch.slug("https://Example.com/a/b.html?x=1", taken), "Example-com-a-b-html")
        self.assertEqual(fetch.slug("https://Example.com/a/b.html?x=2", taken), "Example-com-a-b-html-2")
        self.assertEqual(len(fetch.slug("https://e.com/" + "z" * 200, set())), 80)

    def test_simpleqa_slice(self):
        def row(i, *urls):
            return {"metadata": repr({"topic": "t", "urls": list(urls)}), "problem": "q%d" % i, "answer": "a%d" % i}

        wiki = "https://en.wikipedia.org/wiki/"
        table = [row(0, wiki + "A", "https://b.com/x"), row(1, "https://b.com/x"), row(2, "https://en.m.wikipedia.org/wiki/C"),
                 row(3, wiki + "D"), row(4, wiki + "E")]
        self.assertEqual(fetch.simpleqa_tasks(table, 2), [
            {"id": "simpleqa-0", "question": "q0", "answer": "a0", "pages": ["A"]},
            {"id": "simpleqa-2", "question": "q2", "answer": "a2", "pages": ["C"]},
        ])

    def test_frames_tasks(self):
        raw = ('Prompt\tAnswer\twiki_links\treasoning_types\n'
               '"two\nlines"\t42\t[\'https://en.wikipedia.org/wiki/A_b\', \'https://x.org/\']\tNumerical\n').encode()
        self.assertEqual(fetch.frames_tasks(fetch.parse_table(raw, "\t"), 5), [
            {"id": "frames-0", "question": "two\nlines", "answer": "42", "reasoning_types": "Numerical", "pages": ["A_b"]}])

    def test_append_rows_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "docs.jsonl")
            with open(path + ".part", "w", encoding="utf-8") as f:
                f.write('{"n": 0}\n{"n": 1}\n{"n": 2}\n{"n"')  # the last line was cut by the interruption
            offsets = []

            def page(offset):
                offsets.append(offset)
                return [{"n": n} for n in range(offset, offset + 2)]

            fetch.append_rows(path, 6, page)
            self.assertEqual(offsets, [3, 5])
            with open(path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["n"] for line in f], [0, 1, 2, 3, 4, 5])
            self.assertFalse(os.path.exists(path + ".part"))

    def test_rows_guards_only_the_cells_it_keeps(self):
        def page(*truncated):
            body = {"num_rows_total": 830, "rows": [{"row_idx": 0, "row": {"query": "q"}, "truncated_cells": list(truncated)}]}
            return unittest.mock.patch.object(fetch, "get", return_value=json.dumps(body).encode())

        keep = ("query_id", "query", "answer")
        with page("docs"):  # an embedded document cell is thrown away, so its truncation is harmless
            self.assertEqual(next(fetch.rows(fetch.QUERIES, "test", 0, 1, keep)), {"query": "q"})
        with page("docs", "answer"), self.assertRaisesRegex(SystemExit, "truncated in .*answer"):
            next(fetch.rows(fetch.QUERIES, "test", 0, 1, keep))

    def test_rows_follows_short_pages_and_checks_the_row_index(self):
        asked = []

        def get(url, shift=0):
            offset = int(url.split("offset=")[1].split("&")[0])
            asked.append(offset)
            got = [{"row_idx": offset + i + shift, "row": {"n": offset + i}, "truncated_cells": []} for i in range(2)]
            return json.dumps({"num_rows_total": 830, "rows": got}).encode()  # 2 rows per page of 3

        with unittest.mock.patch.object(fetch, "get", get):
            got = list(itertools.islice(fetch.rows(fetch.QUERIES, "test", 0, 3, ()), 5))
        self.assertEqual([r["n"] for r in got], [0, 1, 2, 3, 4])
        self.assertEqual(asked, [0, 2, 4])
        with unittest.mock.patch.object(fetch, "get", lambda url: get(url, shift=1)), \
                self.assertRaisesRegex(SystemExit, "row 1 at offset 0"):
            next(fetch.rows(fetch.QUERIES, "test", 0, 3, ()))

    def test_append_rows_leaves_a_finished_file_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tasks.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"n": 0}\n')
            page = unittest.mock.Mock(side_effect=AssertionError("downloaded again"))
            fetch.append_rows(path, 1, page)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"n": 0}\n')
            self.assertFalse(os.path.exists(path + ".part"))

    def test_append_rows_resumes_from_a_short_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tasks.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"n": 0}\n{"n": 1}\n')  # an earlier --limit run
            offsets = []

            def page(offset):
                offsets.append(offset)
                return [{"n": n} for n in range(offset, offset + 2)]

            fetch.append_rows(path, 5, page)
            self.assertEqual(offsets, [2, 4])
            with open(path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["n"] for line in f], [0, 1, 2, 3, 4])
            self.assertFalse(os.path.exists(path + ".part"))

    def test_get_retries_a_truncated_or_corrupt_gzip_body(self):
        good = gzip.compress(b"ok")
        for bad in (good[:-6], good[:10] + b"\xff" * 8):  # cut short (EOFError), bad deflate stream (zlib.error)
            replies = [GzipReply(bad), GzipReply(good)]
            with unittest.mock.patch.object(urllib.request, "urlopen", side_effect=replies), \
                    unittest.mock.patch.object(fetch.time, "sleep") as sleep:
                self.assertEqual(fetch.get("http://x"), b"ok")
                self.assertEqual(sleep.call_count, 1)

    def test_get_retries_with_backoff(self):
        calls = []

        def urlopen(req, timeout=None):
            calls.append(req.get_header("User-agent"))
            if len(calls) <= 2:
                raise urllib.error.URLError("down")
            return GzipReply(gzip.compress(b"ok"))

        with unittest.mock.patch.object(urllib.request, "urlopen", urlopen), \
                unittest.mock.patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(fetch.get("http://x"), b"ok")
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4])
            self.assertEqual(calls, ["larch-research-evals/1"] * 3)
            sleep.reset_mock()  # now every attempt fails
            urlopen_down = unittest.mock.Mock(side_effect=urllib.error.URLError("down"))
            with unittest.mock.patch.object(urllib.request, "urlopen", urlopen_down):
                with self.assertRaises(urllib.error.URLError):
                    fetch.get("http://x")
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8, 16])
            self.assertEqual(urlopen_down.call_count, 5)

    def test_wiki_docs_resume_after_a_torn_tail(self):
        whole = json.dumps({"docid": "A", "text": "caf\u00e9"}, ensure_ascii=False).encode() + b"\n"
        for tail in (b'{"docid": "B", "ur', '{"docid": "B", "t": "\u00e9"'.encode()[:-2]):  # cut in ASCII, cut mid-character
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "docs.jsonl")
                with open(path + ".part", "wb") as f:
                    f.write(whole + tail)
                asked = []

                def doc(title):
                    asked.append(title)
                    return {"docid": title, "text": "t"}

                with unittest.mock.patch.object(fetch, "wiki_doc", doc):
                    self.assertFalse(fetch._wiki_docs(path, ["A", "B", "C"]))
                self.assertEqual(asked, ["B", "C"])
                with open(path, encoding="utf-8") as f:
                    self.assertEqual([json.loads(line)["docid"] for line in f], ["A", "B", "C"])

    def test_wiki_docs_keeps_the_finished_docs_when_a_limited_run_follows(self):
        def doc(title):
            return {"docid": title, "text": "t"}

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "docs.jsonl")
            with unittest.mock.patch.object(fetch, "wiki_doc", doc):
                self.assertFalse(fetch._wiki_docs(path, ["A", "B", "C"]))
                self.assertFalse(fetch._wiki_docs(path, ["A"]))  # --limit 1 after the full download
            with open(path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["docid"] for line in f], ["A", "B", "C"])

    def test_a_limited_wiki_set_keeps_the_full_task_list(self):
        tasks = [{"id": "t%d" % i, "pages": ["P%d" % i]} for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            with unittest.mock.patch.object(fetch, "wiki_doc", lambda t: {"docid": t, "text": "t"}):
                self.assertEqual(fetch._wiki_set(d, tasks), 0)
                self.assertEqual(fetch._wiki_set(d, tasks[:1]), 0)  # --limit 1 after the full fetch
            with open(os.path.join(d, "tasks.jsonl"), encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["id"] for line in f], ["t0", "t1", "t2"])
            with open(os.path.join(d, "docs.jsonl"), encoding="utf-8") as f:
                self.assertEqual(len(f.readlines()), 3)

    def test_wiki_docs_errors_keep_the_part_and_abort_early(self):
        def flaky(title):
            if title in ("B", "C"):
                raise urllib.error.URLError("down")
            return None if title == "M" else {"docid": title, "text": "t"}

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            path = os.path.join(d, "docs.jsonl")
            with unittest.mock.patch.object(fetch, "wiki_doc", flaky):
                self.assertTrue(fetch._wiki_docs(path, ["A", "B", "M", "C", "D"]))
                self.assertIn("2 errors", shown.getvalue())
                self.assertFalse(os.path.exists(path))  # the rerun fetches only what is left
                with open(path + ".part", encoding="utf-8") as f:
                    self.assertEqual([json.loads(line)["docid"] for line in f], ["A", "D"])
            with unittest.mock.patch.object(fetch, "wiki_doc", lambda t: {"docid": t, "text": "t"}):
                self.assertFalse(fetch._wiki_docs(path, ["A", "B", "M", "C", "D"]))
            with open(path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["docid"] for line in f], ["A", "D", "B", "M", "C"])
            down = unittest.mock.Mock(side_effect=urllib.error.URLError("down"))
            with unittest.mock.patch.object(fetch, "wiki_doc", down):
                self.assertTrue(fetch._wiki_docs(path + "2", list("ABCDEFGH")))
            self.assertEqual(down.call_count, 5)  # a blocked host is not retried title by title for hours

    def test_wiki_docs_counts_a_truncated_body_as_an_error(self):
        def doc(title):
            if title == "B":
                raise EOFError("Compressed file ended before the end-of-stream marker was reached")
            return {"docid": title, "text": "t"}

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            path = os.path.join(d, "docs.jsonl")
            with unittest.mock.patch.object(fetch, "wiki_doc", doc):
                self.assertTrue(fetch._wiki_docs(path, ["A", "B", "C"]))
            self.assertIn("1 errors", shown.getvalue())
            with open(path + ".part", encoding="utf-8") as f:
                self.assertEqual([json.loads(line)["docid"] for line in f], ["A", "C"])

    def test_wiki_doc_api_errors(self):
        def stub(*replies):
            return unittest.mock.patch.object(fetch, "get", side_effect=[json.dumps(r).encode() for r in replies])

        with stub({"error": {"code": "ratelimited"}}), self.assertRaises(ValueError):
            fetch.wiki_doc("A")
        with stub({"batchcomplete": True, "query": {"interwiki": [{"title": "fr:A"}]}}):
            self.assertIsNone(fetch.wiki_doc("fr:A"))
        page = {"query": {"pages": [{"title": "A", "revisions": [{"revid": 7}]}]}}
        with stub(page, {"error": {"code": "missingtitle"}}), self.assertRaises(ValueError):
            fetch.wiki_doc("A")
        with stub(page, {"parse": {"text": "<p>hi</p>"}}):
            self.assertEqual(fetch.wiki_doc("A")["url"], "https://en.wikipedia.org/w/index.php?oldid=7")

    def test_iris_reports_bad_tasks_and_sources(self):
        def docs(url, taken):
            if url.endswith("/bad"):
                raise TypeError("odd json")
            if url.endswith("/cut"):
                raise http.client.IncompleteRead(b"")
            return [{"docid": fetch.slug(url, taken), "url": url, "title": url, "text": "t"} for _ in range(2)]

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            source = os.path.join(d, "iris.jsonl")
            with open(source, "w", encoding="utf-8") as f:
                f.write('{"sources": ["https://x/ok", "https://x/bad", "https://x/cut"]}\n{"question": "no sources"}\n')
            with unittest.mock.patch.object(fetch, "web_docs", docs), unittest.mock.patch.object(fetch.time, "sleep"):
                self.assertEqual(fetch._iris(d, source), 1)
            self.assertIn("3 failed", shown.getvalue())
            with open(os.path.join(d, "docs.jsonl"), encoding="utf-8") as f:
                got = [json.loads(line)["docid"] for line in f]
            self.assertEqual(got, ["x-ok", "x-ok-2"])  # one source, several docs, docids still unique

    def test_iris_keeps_the_saved_snapshot_when_a_rerun_fails(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            source = os.path.join(d, "iris.jsonl")
            with open(source, "w", encoding="utf-8") as f:
                f.write('{"sources": ["https://x/a", "https://x/b"]}\n')
            saved = json.dumps({"docid": "x-a", "url": "https://x/a", "title": "A", "text": "good"}) + "\n"
            with open(os.path.join(d, "docs.jsonl"), "w", encoding="utf-8") as f:
                f.write(saved)  # sources change: this snapshot cannot be taken again
            offline = unittest.mock.Mock(side_effect=OSError("offline"))
            with unittest.mock.patch.object(fetch, "web_docs", offline), unittest.mock.patch.object(fetch.time, "sleep"):
                self.assertEqual(fetch._iris(d, source), 1)
            with open(os.path.join(d, "docs.jsonl"), encoding="utf-8") as f:
                self.assertEqual(f.read(), saved)

    def test_iris_merges_a_refetch_into_the_saved_snapshot(self):
        def docs(url, taken):
            if url.endswith(("/a", "/c")):
                raise OSError("down")
            return [{"docid": fetch.slug(url, taken), "url": url, "title": "B", "text": "new"}]

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown:
            source = os.path.join(d, "iris.jsonl")
            with open(source, "w", encoding="utf-8") as f:
                f.write('{"sources": ["https://x/a", "https://x/b", "https://x/c"]}\n')
            path = os.path.join(d, "docs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"docid": "x-a", "url": "https://x/a", "title": "A", "text": "saved"}) + "\n")
            with unittest.mock.patch.object(fetch, "web_docs", docs), unittest.mock.patch.object(fetch.time, "sleep"):
                self.assertEqual(fetch._iris(d, source), 1)
            self.assertIn("kept 1 saved docs for failed sources", shown.getvalue())
            with open(path, encoding="utf-8") as f:
                self.assertEqual([(r["docid"], r["text"]) for r in map(json.loads, f)], [("x-b", "new"), ("x-a", "saved")])

    def test_get_refuses_a_redirect_out_of_https(self):
        reply = GzipReply(gzip.compress(b"ok"), url="http://elsewhere/x")
        with unittest.mock.patch.object(urllib.request, "urlopen", return_value=reply) as opened, \
                unittest.mock.patch.object(fetch.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, r"https://a/x.*http://elsewhere/x"):
                fetch.get("https://a/x")
        self.assertEqual((opened.call_count, sleep.call_count), (1, 0))  # a refusal is not retried
        same = GzipReply(gzip.compress(b"ok"), url="https://b/y")  # a redirect within https is fine
        with unittest.mock.patch.object(urllib.request, "urlopen", return_value=same):
            self.assertEqual(fetch.get("https://a/x"), b"ok")

    def test_get_caps_a_response(self):
        with unittest.mock.patch.object(fetch, "MAX_BYTES", 10):
            for size, fits in ((10, True), (11, False)):
                reply = GzipReply(b"x" * size)
                reply.headers = {}
                with unittest.mock.patch.object(urllib.request, "urlopen", return_value=reply):
                    if fits:
                        self.assertEqual(fetch.get("http://x"), b"x" * size)
                    else:
                        with self.assertRaisesRegex(ValueError, "over 10 bytes"):
                            fetch.get("http://x")
        with unittest.mock.patch.object(fetch, "MAX_BYTES", 4 << 20):
            # read in chunks: a cap-sized read() reserves the whole cap when the server sends no length
            reply = GzipReply(b"x" * (2 << 20))
            reply.headers = {}
            with unittest.mock.patch.object(urllib.request, "urlopen", return_value=reply):
                self.assertEqual(len(fetch.get("http://x")), 2 << 20)
            self.assertLessEqual(reply.asked, 1 << 20)
        with unittest.mock.patch.object(fetch, "MAX_BYTES", 1000):
            # a small gzip body that inflates past the cap is refused too
            with unittest.mock.patch.object(urllib.request, "urlopen", return_value=GzipReply(gzip.compress(b"x" * 1001))):
                with self.assertRaisesRegex(ValueError, "over 1000 bytes"):
                    fetch.get("http://x")
        self.assertEqual(fetch.MAX_BYTES, 1 << 30)  # 1 GiB

    def test_iris_docs_carry_when_and_what_was_fetched(self):
        stamp = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
        releases = json.dumps([{"tag_name": "v1", "html_url": "h1", "published_at": "2026-01-01T00:00:00Z", "body": "a"},
                               {"tag_name": "v2", "html_url": "h2", "published_at": "2026-02-01T00:00:00Z", "body": "b"}]).encode()
        for url, raw, count in (("https://x/a", b"<p>hi</p>", 1), ("https://api.github.com/repos/o/r/releases", releases, 2)):
            with self.subTest(url), unittest.mock.patch.object(fetch, "get", return_value=raw):
                docs = fetch.web_docs(url, set())
            self.assertEqual(len(docs), count)
            for doc in docs:
                self.assertEqual(doc["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertRegex(doc["fetched_at"], stamp)

    def test_a_limited_browsecomp_fetch_warns_that_it_is_not_scoreable(self):
        for limit, warned in ((2, True), (None, False)):
            with self.subTest(limit), tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as shown, \
                    unittest.mock.patch.object(fetch, "hf_revision", side_effect=lambda ds: fetch.PINS[ds][0]), \
                    unittest.mock.patch.object(fetch, "append_rows"):
                fetch._browsecomp(d, limit)
            self.assertEqual("smoke test" in shown.getvalue(), warned)
            self.assertEqual("never be scored" in shown.getvalue(), warned)

    def test_get_does_not_retry_a_404(self):
        down = unittest.mock.Mock(side_effect=urllib.error.HTTPError("http://x", 404, "nf", {}, None))
        with unittest.mock.patch.object(urllib.request, "urlopen", down), \
                unittest.mock.patch.object(fetch.time, "sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                fetch.get("http://x")
        self.assertEqual((down.call_count, sleep.call_count), (1, 0))


if __name__ == "__main__":
    unittest.main()
