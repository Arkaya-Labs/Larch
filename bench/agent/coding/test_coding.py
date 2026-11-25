"""Unit tests for the coding bench: id files, the instance filter, every failure class, pass^k and the CLI's guards."""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


def lines(*path):
    return run.read_ids(os.path.join(HERE, *path))


class IdFileTest(unittest.TestCase):
    def test_id_files_are_well_formed_and_unique(self):
        for path, n, form in [(("swe-mini.txt",), 50, r"[\w.-]+__[\w.-]+-\d+"),
                              (("fixtures", "swe-bench-verified-ids.txt"), 500, r"[\w.-]+__[\w.-]+-\d+"),
                              (("tb2-slice.txt",), 20, r"[a-z0-9][a-z0-9.-]*"),
                              (("fixtures", "tb2-tasks.txt"), 89, r"[a-z0-9][a-z0-9.-]*")]:
            ids = lines(*path)
            self.assertEqual((len(ids), len(set(ids))), (n, n), path)
            self.assertEqual([i for i in ids if not re.fullmatch(form, i)], [], path)

    def test_every_swe_mini_id_is_in_swe_bench_verified(self):
        self.assertLessEqual(set(lines("swe-mini.txt")), set(lines("fixtures", "swe-bench-verified-ids.txt")))

    def test_the_filter_matches_exactly_the_50_ids(self):
        mini, verified = lines("swe-mini.txt"), lines("fixtures", "swe-bench-verified-ids.txt")
        # mini-swe-agent applies the filter with re.match to every instance id of the dataset
        self.assertEqual([v for v in verified if re.match(run.swe_filter(mini), v)], sorted(mini))
        for iid in mini:  # one task's filter
            self.assertEqual([v for v in verified if re.match(run.swe_filter([iid]), v)], [iid])
        # no swe-mini id prefixes another Verified id today; the anchors and escapes keep it so for a new id
        self.assertIsNone(re.match(run.swe_filter(["django__django-1179"]), "django__django-11790"))
        self.assertIsNone(re.match(run.swe_filter(["a__b.c-1"]), "a__bxc-1"))

    def test_the_tb2_slice_follows_the_readme_rule(self):
        self.assertEqual(run.tb2_slice(lines("fixtures", "tb2-tasks.txt")), lines("tb2-slice.txt"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class RowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def swe(self, iid, edit=None):
        """The row of a copy of SWE fixture iid, after edit(task_dir) changed it."""
        d = os.path.join(tempfile.mkdtemp(dir=self.tmp), iid)
        shutil.copytree(os.path.join(FIX, "swe-mini", "run0", iid), d)
        if edit:
            edit(d)
        return run.swe_row(d, iid, 0, 1.5)

    def tb2(self, task, edit_result=None, edit_traj=None):
        d = os.path.join(tempfile.mkdtemp(dir=self.tmp), task)
        shutil.copytree(os.path.join(FIX, "tb2-slice", "run0", task), d)
        trial = os.path.join(d, os.listdir(d)[0])
        for edit, path in [(edit_result, os.path.join(trial, "result.json")),
                           (edit_traj, os.path.join(trial, "agent", "mini-swe-agent.trajectory.json"))]:
            if edit:
                data = load(path)
                edit(data)
                dump(path, data)
        return run.tb2_row(d, task, 0)

    def test_a_resolved_swe_row_counts_what_the_tools_wrote(self):
        self.assertEqual(self.swe("fake__repo-1"), {
            "task": "fake__repo-1", "run": 0, "resolved": True, "failure": None, "steps": 2, "wall_s": 1.5,
            "prompt_tokens": 2001, "completion_tokens": 41, "patch_bytes": 123})

    def test_every_swe_failure_class(self):
        def set_patch(patch):
            def edit(d):
                preds = load(os.path.join(d, "preds.json"))
                preds[os.path.basename(d)]["model_patch"] = patch
                dump(os.path.join(d, "preds.json"), preds)
            return edit

        def set_exit(status):
            def edit(d):
                path = os.path.join(d, os.path.basename(d), os.path.basename(d) + ".traj.json")
                traj = load(path)
                traj["info"]["exit_status"] = status
                dump(path, traj)
            return edit

        def set_log(text):
            def edit(d):
                log = os.path.join(d, "logs", "run_evaluation", run.RUN_ID + "-run0", "hosted_vllm__m5", os.path.basename(d), "run_instance.log")
                with open(log, "w", encoding="utf-8") as f:
                    f.write(text)
            return edit

        cases = [
            (self.swe("fake__repo-2"), "tests_fail"),
            (self.swe("fake__repo-3"), "patch_does_not_apply"),
            (self.swe("fake__repo-4"), "step_limit"),  # its patch is empty too: the limit is why
            (self.swe("fake__repo-1", set_patch("")), "no_patch"),
            (self.swe("fake__repo-1", lambda d: (set_exit("RepeatedFormatError")(d), set_patch("")(d))), "no_patch"),
            (self.swe("fake__repo-4", set_exit("APIConnectionError")), "server_error"),
            (self.swe("fake__repo-4", set_exit("BadRequestError")), "server_error"),  # the server refused an overlong history
            (self.swe("fake__repo-4", set_exit("Timeout")), "timeout"),
            (self.swe("fake__repo-3", set_log("EvaluationError: Test timed out after 1800 seconds.\n")), "timeout"),
        ]
        for row, failure in cases:
            self.assertEqual((row["failure"], row["resolved"]), (failure, False), row)
        self.assertEqual(self.swe("fake__repo-4")["patch_bytes"], 0)

    def test_a_grading_error_with_no_class_stops_the_bench(self):
        def infra(d):
            log = os.path.join(d, "logs", "run_evaluation", run.RUN_ID + "-run0", "hosted_vllm__m5", "fake__repo-3", "run_instance.log")
            with open(log, "w", encoding="utf-8") as f:
                f.write("Error in evaluating model for fake__repo-3: 500 Server Error (docker)\n")
        with self.assertRaisesRegex(RuntimeError, "grading fake__repo-3 failed"):
            self.swe("fake__repo-3", infra)

    def test_an_unresolved_instance_the_harness_blames_on_the_host_stops_the_bench(self):
        def oom(d):
            path = os.path.join(d, "hosted_vllm__m5.%s-run0.json" % run.RUN_ID)
            report = load(path)
            report["infra_failure_ids"], report["failure_reasons"] = ["fake__repo-2"], {"fake__repo-2": "oom_killed"}
            dump(path, report)
        with self.assertRaisesRegex(RuntimeError, "grading fake__repo-2 failed.*oom_killed"):
            self.swe("fake__repo-2", oom)

    def test_a_task_the_scaffold_never_started_stops_the_bench(self):
        with self.assertRaisesRegex(RuntimeError, "no trajectory"):
            self.swe("fake__repo-1", lambda d: shutil.rmtree(os.path.join(d, "fake__repo-1")))

    def test_every_tb2_failure_class(self):
        def fail(data):
            data["verifier_result"]["rewards"]["reward"] = 0.0

        def crash(data):
            fail(data)
            data["exception_info"] = dict(load(os.path.join(
                FIX, "tb2-slice", "run0", "fake-timeout", "fake-timeout__Qw7rT2y", "result.json"))["exception_info"],
                exception_type="RuntimeError")

        def raised(name):
            def edit(data):
                crash(data)
                data["exception_info"]["exception_type"] = name
            return edit

        for name in ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError"):
            with self.assertRaisesRegex(RuntimeError, name):
                self.tb2("fake-hello", raised(name))

        def limit(traj):
            traj["info"]["exit_status"] = "LimitsExceeded"

        self.assertEqual(self.tb2("fake-hello"), {
            "task": "fake-hello", "run": 0, "resolved": True, "failure": None, "steps": 2, "wall_s": 61.5,
            "prompt_tokens": 2001, "completion_tokens": 41, "patch_bytes": None})
        for row, failure in [(self.tb2("fake-timeout"), "timeout"), (self.tb2("fake-hello", fail), "tests_fail"),
                             (self.tb2("fake-hello", crash), "server_error"),
                             (self.tb2("fake-hello", raised("ApiConnectionError")), "server_error"),
                             (self.tb2("fake-hello", raised("VerifierTimeoutError")), "timeout"),
                             (self.tb2("fake-hello", fail, limit), "step_limit")]:
            self.assertEqual((row["failure"], row["resolved"]), (failure, False), row)


class SummaryTest(unittest.TestCase):
    def row(self, task, resolved, failure=None):
        return {"task": task, "run": 0, "resolved": resolved, "failure": failure, "steps": 10, "wall_s": 2.5,
                "prompt_tokens": 100, "completion_tokens": 10, "patch_bytes": 50}

    def test_pass_k_and_failures_by_class(self):
        rows = [self.row("A", True)] * 4 + [self.row("B", True)] * 3 + [self.row("B", False, "tests_fail")]
        s = run.summarize(rows)
        self.assertEqual((s["tasks"], s["runs"], s["pass@1"], s["pass^4"]), (2, 8, 0.875, 0.5))
        self.assertEqual(s["steps"], {"total": 80, "mean": 10})
        self.assertEqual(s["wall_s"], {"total": 20.0, "mean": 2.5})
        self.assertEqual(s["tokens"], {"prompt": 800, "completion": 80})
        self.assertEqual(s["failures"], dict.fromkeys(run.FAILURES, 0) | {"tests_fail": 1})
        self.assertIsNone(run.summarize(rows[:3] + rows[4:7])["pass^4"])  # 3 runs a task: no pass^4
        self.assertIsNone(run.summarize([])["pass@1"])


class CliTest(unittest.TestCase):
    def test_the_fake_gate_passes_and_fails_on_other_numbers(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run.main(["--fake"]), 0)
        self.assertIn("fake: ok", out.getvalue())
        real = run.summarize
        with unittest.mock.patch.object(run, "summarize", lambda rows, k=4: dict(real(rows, k), steps=None)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run.main(["--fake"]), 1)
        self.assertIn("fake: FAILED", err.getvalue())

    def test_run_py_starts_under_isolated_mode(self):
        # as research's: -P drops the script's directory from sys.path, run.py must put it back
        safe_path = "-P" if sys.version_info >= (3, 11) else "-I"
        with tempfile.TemporaryDirectory() as d:
            done = subprocess.run([sys.executable, safe_path, os.path.join(HERE, "run.py"), "--fake"],
                                  cwd=d, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_an_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as err:
            out = os.path.join(d, "report.json")
            with open(out, "w", encoding="utf-8") as f:
                f.write("saved")
            with unittest.mock.patch.object(run, "preflight", side_effect=AssertionError("checked")):
                self.assertEqual(run.main(["--set", "swe-mini", "--url", "http://x", "--model", "m", "--out", out]), 2)
            self.assertIn(out + " exists", err.getvalue())
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "saved")

    def test_a_used_work_dir_is_not_reused(self):
        # the tools skip what their folder already holds, so a rerun there would report old results
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as err:
            os.makedirs(os.path.join(d, "runs", "report"))
            with unittest.mock.patch.object(run, "preflight", return_value=None):
                self.assertEqual(run.main(["--set", "tb2-slice", "--url", "http://x", "--model", "m", "--data", d,
                                           "--out", os.path.join(d, "report.json")]), 2)
            self.assertIn(os.path.join(d, "runs", "report") + " exists", err.getvalue())

    def test_docker_then_the_venv_are_checked_first(self):
        argv = ["--set", "tb2-slice", "--url", "http://x", "--model", "m", "--out", "never.json"]
        with unittest.mock.patch.object(run.subprocess, "run", side_effect=FileNotFoundError("docker")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run.main(argv), 2)
        self.assertIn("docker info", err.getvalue())
        docker_up = subprocess.CompletedProcess(["docker", "info"], 0)
        with tempfile.TemporaryDirectory() as d, \
                unittest.mock.patch.object(run.subprocess, "run", return_value=docker_up), \
                unittest.mock.patch.object(run, "VENV", os.path.join(d, ".venv")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run.main(argv), 2)
        self.assertIn("pip install -r", err.getvalue())
        self.assertFalse(os.path.exists("never.json"))

    def test_a_venv_off_the_pins_is_refused(self):
        stale = subprocess.CompletedProcess([], 0, stdout="2.4.5 5.0.2 0.24.0 1.104.0\n")
        with unittest.mock.patch.object(run.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0), stale]), \
                unittest.mock.patch.object(run.os.path, "exists", return_value=True):
            self.assertIn("mini-swe-agent 2.4.5, not 2.4.6", run.preflight())

    def test_the_report_is_saved_after_each_task(self):
        def task(args, work, iid, n):
            if iid == run.read_ids(os.path.join(HERE, "swe-mini.txt"))[1]:
                raise KeyboardInterrupt
            return {"task": iid, "run": n, "resolved": True, "failure": None, "steps": 3, "wall_s": 1.0,
                    "prompt_tokens": 5, "completion_tokens": 1, "patch_bytes": 9}

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "report.json")
            with unittest.mock.patch.object(run, "preflight", return_value=None), \
                    unittest.mock.patch.object(run, "swe_dataset", return_value=d), \
                    unittest.mock.patch.dict(run.TASKS, {"swe-mini": task}):
                with self.assertRaises(KeyboardInterrupt):
                    run.main(["--set", "swe-mini", "--url", "http://x", "--model", "m", "--limit", "2",
                              "--data", d, "--out", out])
            report = load(out)
            self.assertEqual([r["task"] for r in report["rows"]], run.read_ids(os.path.join(HERE, "swe-mini.txt"))[:1])
            self.assertEqual((report["schema"], report["set"], report["summary"]["runs"]), ("larch-coding/1", "swe-mini", 1))
            self.assertEqual((report["scaffold"], report["scaffold_version"]), (run.SCAFFOLD["swe-mini"], "2.4.6"))
            self.assertFalse(os.path.exists(out + ".part"))

    def test_a_task_that_raises_keeps_the_rows_of_tasks_running_beside_it(self):
        first, second = run.read_ids(os.path.join(HERE, "tb2-slice.txt"))[:2]

        # The first raises only once the second runs: a pool may reuse the first's thread, and the
        # second then waits in the queue that shutdown cancels.
        running = threading.Event()

        def task(args, work, name, n):
            if name == first:
                running.wait(5)
                raise RuntimeError("Docker broke")
            running.set()
            time.sleep(0.3)
            return {"task": name, "run": n, "resolved": False, "failure": "tests_fail", "steps": 3, "wall_s": 1.0,
                    "prompt_tokens": 5, "completion_tokens": 1, "patch_bytes": None}

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "report.json")
            with unittest.mock.patch.object(run, "preflight", return_value=None), \
                    unittest.mock.patch.dict(run.TASKS, {"tb2-slice": task}):
                with self.assertRaisesRegex(RuntimeError, "Docker broke"):
                    run.main(["--set", "tb2-slice", "--url", "http://x", "--model", "m", "--limit", "2", "--workers", "2",
                              "--data", d, "--out", out])
            self.assertEqual([r["task"] for r in load(out)["rows"]], [second])

    def test_a_fixture_folder_ignores_stray_files(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(FIX, os.path.join(d, "fixtures"))
            for name in run.SETS:
                open(os.path.join(d, "fixtures", name, "run0", ".DS_Store"), "wb").close()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.fake(os.path.join(d, "fixtures")), 0)


if __name__ == "__main__":
    unittest.main()
