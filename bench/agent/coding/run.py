"""Coding-agent bench: mini-swe-agent (bash only) against an OpenAI-compatible server, graded by the official harnesses.

swe-mini runs `mini-extra swebench` and grades with `swebench.harness.run_evaluation` on local Docker; tb2-slice runs
mini-swe-agent through Harbor, which grades each Terminal-Bench 2.0 task in its container. Each task is one scaffold
run plus its grading; its row is built from the files the tools wrote, so --fake feeds committed tool outputs through
the same row, summary and report code.
"""

import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# -P/-I drop the script's directory from sys.path; bench/agent holds the research bench whose report code this reuses
sys.path.insert(0, os.path.dirname(HERE))
from research import fetch  # noqa: E402
from research import run as research  # noqa: E402

VENV = os.path.join(HERE, ".venv")
SETS = {"swe-mini": "swe-mini.txt", "tb2-slice": "tb2-slice.txt"}
VERIFIED = ("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/"
            "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet",
            "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd")
HARBOR_REGISTRY = "laude-institute/harbor@872da9479f4a2078142d22c49c9ffae0f09e2e63"
# Harbor installs mini-swe-agent in each task container with an unpinned litellm; litellm 1.105.0 (11 Oct 2026) and
# later import the uninstalled `mcp` package on every call that sends tools, so resolve as of the Harbor pin
EXCLUDE_NEWER = "2026-10-06T00:00:00Z"
RUN_ID = "larch-coding"
APPLY_FAIL = ">>>>> Patch Apply Failed"
# Harbor 0.24.0's exceptions for a sandbox that never got to run the agent or grade it: no class fits them
HARBOR_INFRA = ("EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "SandboxBuildFailedError", "HealthcheckError",
                "AddTestsDirError", "DownloadVerifierDirError", "RewardFileNotFoundError", "RewardFileEmptyError",
                "VerifierOutputParseError")
FAILURES = ("no_patch", "patch_does_not_apply", "tests_fail", "step_limit", "timeout", "server_error")


def read_ids(path):
    """The non-empty lines of an id file."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pins():
    """requirements.txt as {package: version}."""
    return dict(line.split("==") for line in read_ids(os.path.join(HERE, "requirements.txt")) if not line.startswith("#"))


SCAFFOLD = {"swe-mini": "mini-swe-agent, swebench_backticks.yaml",
            "tb2-slice": "mini-swe-agent, mini.yaml, through harbor==%s" % pins()["harbor"]}


def swe_filter(ids):
    """mini-extra's --filter for exactly these instance ids (it applies re.match to each id)."""
    return "^(%s)$" % "|".join(map(re.escape, ids))


def tb2_slice(names, n=20):
    """The README's slice rule: n tasks evenly spaced over the sorted names, from the first."""
    names = sorted(names)
    return [names[i * len(names) // n] for i in range(n)]


def agent_failure(status):
    """The failure class a mini-swe-agent exit status decides, or None when grading decides it."""
    if status in ("Submitted", "RepeatedFormatError", ""):  # "" is a run Harbor stopped: its exception says why
        return None
    if status == "LimitsExceeded":  # the step limit: the cost limit never trips at zero cost
        return "step_limit"
    if status in ("TimeExceeded", "Timeout"):  # the agent's wall limit, LiteLLM's request timeout
        return "timeout"
    return "server_error"  # an exception ended the run: the server's after LiteLLM's retries, or the sandbox's


def usage(traj):
    """(model calls, prompt tokens, completion tokens) of a mini-swe-agent trajectory."""
    prompt = completion = 0
    for m in traj["messages"]:
        response = (m.get("extra") or {}).get("response")
        u = (response.get("usage") if isinstance(response, dict) else None) or {}
        prompt += u.get("prompt_tokens") or 0
        completion += u.get("completion_tokens") or 0
    return traj["info"]["model_stats"]["api_calls"], prompt, completion


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _one(pattern):
    found = glob.glob(pattern)
    if len(found) != 1:
        raise RuntimeError("expected one file at %s, found %d" % (pattern, len(found)))
    return found[0]


def run_id(n):
    """The harness run id of run n: two runs grading one instance at once must not share a container name."""
    return "%s-run%d" % (RUN_ID, n)


def graded(d, iid, n):
    """The failure class the swebench harness report of run n in task dir d gives iid, None when resolved."""
    report = _load(_one(os.path.join(d, "*.%s.json" % run_id(n))))
    if iid in report["resolved_ids"]:
        return None
    if iid in report.get("infra_failure_ids", []):  # the harness blames the host (OOM kill, Docker error)
        raise RuntimeError("grading %s failed: %s" % (iid, report["failure_reasons"].get(iid)))
    if iid in report["unresolved_ids"]:
        return "tests_fail"
    logs = glob.glob(os.path.join(d, "logs", "run_evaluation", run_id(n), "*", iid, "run_instance.log"))
    text = ""
    if logs:
        with open(logs[0], encoding="utf-8", errors="replace") as f:
            text = f.read()
    if APPLY_FAIL in text:
        return "patch_does_not_apply"
    if "Test timed out" in text:
        return "timeout"
    # Docker or the harness broke: no class fits, and every task after this one would break the same way
    raise RuntimeError("grading %s failed: see %s" % (iid, logs[0] if logs else os.path.join(d, "grade.log")))


def swe_row(d, iid, run, wall_s):
    """The row of one swe-mini task from what mini-extra and the harness wrote to task dir d."""
    traj_path = os.path.join(d, iid, iid + ".traj.json")
    if not os.path.exists(traj_path):
        raise RuntimeError("%s: no trajectory, the scaffold never started the task: see %s" % (
            iid, os.path.join(d, "scaffold.log")))
    traj = _load(traj_path)
    patch = _load(os.path.join(d, "preds.json"))[iid]["model_patch"] or ""
    steps, prompt, completion = usage(traj)
    failure = agent_failure(traj["info"]["exit_status"])
    if failure is None:
        failure = graded(d, iid, run) if patch else "no_patch"
    return {"task": iid, "run": run, "resolved": failure is None, "failure": failure, "steps": steps, "wall_s": wall_s,
            "prompt_tokens": prompt, "completion_tokens": completion, "patch_bytes": len(patch.encode("utf-8"))}


def _seconds(timing):
    if not (timing and timing.get("started_at") and timing.get("finished_at")):
        return 0.0
    # Python before 3.11 reads no "Z" suffix
    start, end = (datetime.datetime.fromisoformat(timing[k].replace("Z", "+00:00")) for k in ("started_at", "finished_at"))
    return (end - start).total_seconds()


def tb2_row(job_dir, task, run):
    """The row of one tb2-slice task from the trial Harbor wrote under job_dir."""
    result_path = _one(os.path.join(job_dir, "*", "result.json"))
    result = _load(result_path)
    traj_path = os.path.join(os.path.dirname(result_path), "agent", "mini-swe-agent.trajectory.json")
    traj = _load(traj_path) if os.path.exists(traj_path) else None  # none when the agent never started
    agent = result.get("agent_result") or {}
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (result.get("exception_info") or {}).get("exception_type") or ""
    if reward == 1:  # Terminal-Bench scores the verifier, also after an agent timeout
        failure = None
    elif exc in HARBOR_INFRA:
        raise RuntimeError("%s: Harbor raised %s: see %s" % (task, exc, result_path))
    elif exc:  # the agent process failed (Harbor's NonZeroAgentExitCodeError and its Api* kinds)
        failure = "timeout" if exc in ("AgentTimeoutError", "VerifierTimeoutError") else "server_error"
    else:
        failure = (traj and agent_failure(traj["info"]["exit_status"])) or "tests_fail"
    return {"task": task, "run": run, "resolved": failure is None, "failure": failure,
            "steps": traj["info"]["model_stats"]["api_calls"] if traj else 0,
            "wall_s": _seconds(result.get("agent_execution")), "prompt_tokens": agent.get("n_input_tokens") or 0,
            "completion_tokens": agent.get("n_output_tokens") or 0, "patch_bytes": None}


def summarize(rows, k=4):
    """pass@1 (resolved share), pass^k, steps, wall time, tokens and failures by class over per-run rows."""
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["resolved"])
    wins = [(sum(v), len(v)) for v in by_task.values()]

    def stat(key):
        vals = [r[key] for r in rows]
        return {"total": sum(vals), "mean": sum(vals) / len(vals) if vals else None}

    return {
        "tasks": len(by_task), "runs": len(rows),
        "pass@1": research.pass_k(wins, 1), "pass^%d" % k: research.pass_k(wins, k),
        "steps": stat("steps"), "wall_s": stat("wall_s"),
        "tokens": {"prompt": sum(r["prompt_tokens"] for r in rows),
                   "completion": sum(r["completion_tokens"] for r in rows)},
        "failures": {f: sum(r["failure"] == f for r in rows) for f in FAILURES},
    }


def _bin(name):
    return os.path.join(VENV, "bin", name)


def preflight():
    """None when Docker answers and the venv holds requirements.txt's pins, else what to do."""
    try:
        docker = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except FileNotFoundError:
        docker = False
    if not docker:
        return "Docker is not running: start Docker (Docker Desktop with WSL integration, or `sudo service docker start`) until `docker info` answers"
    install = "python3 -m venv %s && %s install -r %s" % (VENV, _bin("pip"), os.path.join(HERE, "requirements.txt"))
    if not os.path.exists(_bin("python")):
        return "no venv at %s: %s" % (VENV, install)
    want = pins()
    have = subprocess.run([_bin("python"), "-c", "import importlib.metadata as m, sys; print(*(m.version(p) for p in sys.argv[1:]))",
                           *want], capture_output=True, text=True)
    off = ["%s %s, not %s" % (p, h, w) for (p, w), h in zip(want.items(), have.stdout.split()) if h != w]
    if have.returncode or off:
        return "the venv is off requirements.txt (%s): %s" % ("; ".join(off) or have.stderr.strip()[-200:], install)
    return None


def swe_dataset(data):
    """The folder holding SWE-bench Verified's test split at the pinned revision, downloaded once and checked."""
    folder = os.path.join(os.path.abspath(data), "swe-bench-verified")
    path = os.path.join(folder, "test.parquet")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(fetch.get(VERIFIED[0]))
        if research.sha256_file(path + ".part") != VERIFIED[1]:
            raise RuntimeError("%s.part is not the pinned SWE-bench Verified test split" % path)
        os.replace(path + ".part", path)
    if research.sha256_file(path) != VERIFIED[1]:
        raise RuntimeError("%s is not the pinned SWE-bench Verified test split: delete it and rerun" % path)
    return folder


def _call(cmd, cwd, log, env=None):
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, log), "ab") as out:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def _key():
    return os.environ.get("LARCH_API_KEY")


def swe_task(args, work, iid, n):
    """Runs mini-extra swebench on one instance, grades a non-empty patch, and returns the task's row."""
    d = os.path.join(work, "run%d" % n, iid)
    env = dict(os.environ, MSWEA_CONFIGURED="true")
    if _key():
        env["HOSTED_VLLM_API_KEY"] = _key()
    start = time.monotonic()
    _call([_bin("mini-extra"), "swebench", "--subset", args.dataset, "--split", "test", "--filter", swe_filter([iid]),
           "-o", d, "-w", "1", "-m", "hosted_vllm/" + args.model, "-c", "swebench_backticks.yaml",
           "-c", "model.model_class=litellm_textbased", "-c", "model.model_kwargs.api_base=%s/v1" % args.url.rstrip("/"),
           "-c", "model.cost_tracking=ignore_errors"], d, "scaffold.log", env)
    wall_s = time.monotonic() - start
    if _load(os.path.join(d, "preds.json"))[iid]["model_patch"]:
        _call([_bin("python"), "-m", "swebench.harness.run_evaluation", "-d", args.dataset, "-s", "test", "-i", iid,
               "-p", "preds.json", "-id", run_id(n), "--max_workers", "1"], d, "grade.log")
    return swe_row(d, iid, n, wall_s)


def tb2_task(args, work, task, n):
    """Runs one Terminal-Bench 2.0 task through Harbor (agent and verifier) and returns its row."""
    jobs = os.path.join(work, "run%d" % n)
    # the agent runs inside the task's container, so the URL must be one the container reaches
    env = dict(os.environ, OPENAI_BASE_URL=args.url.rstrip("/") + "/v1", OPENAI_API_KEY=_key() or "none")
    _call([_bin("harbor"), "run", "--repo", HARBOR_REGISTRY, "-d", "terminal-bench@2.0", "-i", task,
           "-a", "mini-swe-agent", "--ak", "version=" + pins()["mini-swe-agent"], "-m", "openai/" + args.model,
           "--ae", "UV_EXCLUDE_NEWER=" + EXCLUDE_NEWER, "-o", jobs, "--job-name", task, "-n", "1", "-y", "-q"],
          jobs, task + ".log", env)
    return tb2_row(os.path.join(jobs, task), task, n)


TASKS = {"swe-mini": swe_task, "tb2-slice": tb2_task}


def _report(name, model, url, runs, rows):
    return {"schema": "larch-coding/1", "set": name, "model": model, "url": url, "scaffold": SCAFFOLD[name],
            "scaffold_version": pins()["mini-swe-agent"], "runs": runs,
            "started_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": None, "rows": rows}


def fake(fixtures=os.path.join(HERE, "fixtures")):
    """Feeds the committed tool outputs through the row and report code; 0 when every summary is the expected one."""
    expected = _load(os.path.join(fixtures, "expected.json"))
    bad = []
    for name in SETS:
        work = os.path.join(fixtures, name, "run0")
        rows = [swe_row(os.path.join(work, t), t, 0, 0.0) if name == "swe-mini" else tb2_row(os.path.join(work, t), t, 0)
                for t in sorted(os.listdir(work)) if os.path.isdir(os.path.join(work, t)) and not t.startswith(".")]
        report = _report(name, "fake", None, 1, rows)
        research._save(None, report, summarize)
        print(name, json.dumps(report["summary"], indent=1))
        if report["summary"] != expected[name]:
            bad.append("%s: summary is not fixtures/expected.json's" % name)
    if bad:
        print("fake: FAILED\n" + "\n".join(bad), file=sys.stderr)
        return 1
    print("fake: ok")
    return 0


def main(argv=None):
    """Runs the bench; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--fake", action="store_true", help="feed the committed tool outputs through the report code")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.fake == bool(args.set):
        ap.error("give --set or --fake, not both and not neither")
    if args.fake:
        return fake()
    if args.out and os.path.exists(args.out):  # a rerun after a crash must not overwrite the saved report
        print("%s exists" % args.out, file=sys.stderr)
        return 2
    if not (args.url and args.model and args.out):
        print("a run needs --url, --model and --out", file=sys.stderr)
        return 2
    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 2

    tasks = read_ids(os.path.join(HERE, SETS[args.set]))[:args.limit]
    if args.set == "swe-mini":
        args.dataset = swe_dataset(args.data)
    work = os.path.join(os.path.abspath(args.data), "runs", os.path.splitext(os.path.basename(args.out))[0])
    if os.path.exists(work):  # the tools skip what their folder holds: a rerun there would report old results
        print("%s exists: give --out another name" % work, file=sys.stderr)
        return 2
    rows = []
    report = _report(args.set, args.model, args.url, args.runs, rows)
    pool = concurrent.futures.ThreadPoolExecutor(args.workers)
    futures = [pool.submit(TASKS[args.set], args, work, t, n) for n in range(args.runs) for t in tasks]
    saved = set()
    try:
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result())
            saved.add(f)
            research._save(args.out, report, summarize)
    finally:
        # a task that raised stops the bench: nothing queued starts, the running ones finish and keep their rows
        pool.shutdown(cancel_futures=True)
        rows.extend(f.result() for f in futures
                    if f not in saved and f.done() and not f.cancelled() and f.exception() is None)
        research._save(args.out, report, summarize)
    print(json.dumps(report["summary"], indent=1))
    if rows and all(r["failure"] == "server_error" for r in rows):
        print("every run ended with a server error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
