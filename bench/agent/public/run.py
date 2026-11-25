"""Public tool-calling benchmarks for the v3 wave gate (D10), run against `larch serve`'s OpenAI endpoint.

A BFCL offline subset, When2Call and tau2-bench Retail, on the shipping model and quant.

BFCL and tau2 run through their own packages (each in its own Python 3.12 environment, see README.md); When2Call needs
only the standard library. `gate` compares a report with a baseline report.
"""

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent

WHEN2CALL_URL = "https://raw.githubusercontent.com/NVIDIA/When2Call/main/data/test/when2call_test_llm_judge.jsonl"
WHEN2CALL_SHA256 = "0b710578e3b02479e5acf5688140ec5edf71383fa5c093481c7536b57fd13b25"
BFCL_VERSION = "2026.3.23"
BFCL_MODEL = "larch-FC"
# About 240 items: evenly spaced ids of each offline category, multi-turn (the MoE's weak spot) included.
BFCL_SUBSET = {
    "simple_python": 20, "simple_java": 10, "simple_javascript": 10, "multiple": 20, "parallel": 20,
    "parallel_multiple": 20, "irrelevance": 20, "live_simple": 20, "live_multiple": 20, "live_irrelevance": 20,
    "live_relevance": 8, "multi_turn_base": 20, "multi_turn_miss_func": 10, "multi_turn_miss_param": 10,
    "multi_turn_long_context": 10,
}
# BFCL's own summaries (eval_runner_helper.py), in the order they build on each other: unweighted means.
BFCL_GROUPS = {
    "simple": ["simple_python", "simple_java", "simple_javascript"],
    "non_live_ast": ["simple", "multiple", "parallel", "parallel_multiple"],
    "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
}
TAU2_TAG = "v1.0.1"
# Published numbers to read ours against: a subset and another harness, so a large gap points at our template, parser
# or settings, a small one says nothing. BFCL's own runs sample at temperature 0.001 (its generate default), ours at
# --temperature.
REFERENCE = {
    "bfcl": [
        {"score": "non_live_ast", "value": 0.8577, "model": "Qwen3-30B-A3B-Instruct-2507 (FC, no thinking, t=0.001)",
         "source": "BFCL v4 single-turn AST, aggregator, May 2026: https://benchmarklist.com/benchmarks/bfcl_v4/"},
        {"score": "multi_turn", "value": 0.30, "model": "Qwen3-30B-A3B-Instruct-2507 (FC, no thinking, t=0.001)",
         "source": "BFCL multi-turn, aggregator: https://benchmarklist.com/benchmarks/bfcl_v3_multiturn/"},
        {"score": "overall", "value": 0.691, "model": "Qwen3-30B-A3B, thinking (0.586 without)",
         "source": "BFCL v3 overall, Qwen3 technical report, May 2025: https://arxiv.org/pdf/2505.09388; "
                   "the subset has no overall"},
    ],
    "tau2": [
        {"score": "pass^1", "value": 0.570, "model": "Qwen3-30B-A3B-Instruct-2507",
         "source": "tau2-bench Retail, Qwen model card, 2025, graded before v1.0.1 changed grading: "
                   "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"},
    ],
    "when2call": [
        {"score": "accuracy", "value": None, "model": "none published",
         "source": "the paper's F1 needs a judge for asks-back versus cannot; ours scores the call decision only"},
    ],
}
# Counts, not fractions: the gate leaves them out.
COUNTS = {"false_calls", "errors", "tasks"}
# BFCL's types for When2Call's functions (BFCL v2 Live), in JSON schema's words.
TYPES = {"dict": "object", "float": "number", "any": "string"}


def openai_tool(spec):
    """When2Call's function JSON as an OpenAI tool, its types in JSON schema's words and its dots as underscores (BFCL's
    rule for OpenAI-style names)."""
    f = json.loads(spec)

    def cast(schema):
        schema = dict(schema)
        if isinstance(schema.get("type"), str):
            schema["type"] = TYPES.get(schema["type"], schema["type"])
        if isinstance(schema.get("properties"), dict):
            schema["properties"] = {k: cast(v) for k, v in schema["properties"].items()}
        if isinstance(schema.get("items"), dict):
            schema["items"] = cast(schema["items"])
        return schema

    return {"type": "function", "function": {"name": f["name"].replace(".", "_"), "description": f["description"],
                                             "parameters": cast(f["parameters"])}}


def post(url, body, timeout=600):
    """The JSON reply of a POST to `url`; an HTTP error raises with the server's words."""
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e


def when2call_row(url, model, item, temperature, seed):
    """One When2Call item's row: right when a tool_call item calls its target tool first, and when any other item
    (cannot_answer, request_for_info) calls nothing."""
    tools = [openai_tool(t) for t in item["tools"]]
    body = {"model": model, "messages": [{"role": "user", "content": item["question"]}],
            "temperature": temperature, "seed": seed}
    if tools:
        body["tools"] = tools
    row = {"uuid": item["uuid"], "source": item["source"], "label": item["correct_answer"]}
    try:
        message = post(f"{url}/chat/completions", body)["choices"][0]["message"]
    except (OSError, RuntimeError, KeyError, IndexError, ValueError) as e:
        return {**row, "right": False, "error": str(e)}
    row["called"] = [c["function"]["name"] for c in message.get("tool_calls") or []]
    if item["correct_answer"] == "tool_call":
        target = json.loads(item["target_tool"])["name"].replace(".", "_")
        row["right"] = row["called"][:1] == [target]
    else:
        row["right"] = not row["called"]
    return row


def when2call_scores(rows):
    """Accuracy over all rows and per label, calls made where none fits, and failed requests."""
    scores = {"accuracy": sum(r["right"] for r in rows) / len(rows)}
    for label in sorted({r["label"] for r in rows}):
        of = [r for r in rows if r["label"] == label]
        scores[label] = sum(r["right"] for r in of) / len(of)
    scores["false_calls"] = sum(bool(r.get("called")) for r in rows if r["label"] != "tool_call")
    scores["errors"] = sum("error" in r for r in rows)
    return scores


def when2call_items(data):
    """The 300 items of When2Call's judged test set, downloaded once into `data`, checked against the pin each run."""
    path = data / "when2call_test_llm_judge.jsonl"
    if not path.exists():
        try:
            with urllib.request.urlopen(WHEN2CALL_URL, timeout=120) as r:
                body = r.read()
        except OSError as e:
            sys.exit(f"cannot download When2Call's test set from {WHEN2CALL_URL}: {e}")
        if hashlib.sha256(body).hexdigest() != WHEN2CALL_SHA256:
            sys.exit("When2Call's test set changed upstream: check it, then update WHEN2CALL_SHA256")
        data.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != WHEN2CALL_SHA256:
        sys.exit(f"{path} does not match WHEN2CALL_SHA256: delete it to download it again")
    return [json.loads(line) for line in body.decode().splitlines() if line.strip()]


def bfcl_subset(data, sizes):
    """For each category, `n` ids spread evenly over its BFCL_v4 file in `data`."""
    ids = {}
    for cat, n in sizes.items():
        lines = (data / f"BFCL_v4_{cat}.json").read_text().splitlines()
        every = [json.loads(line)["id"] for line in lines if line.strip()]
        ids[cat] = [every[i * len(every) // n] for i in range(n)]
    return ids


def bfcl_scores(score_dir):
    """Each category's accuracy from BFCL's score files under `score_dir`, and the groups whose parts all ran."""
    scores = {}
    for f in score_dir.rglob("BFCL_v4_*_score.json"):
        cat = f.name[len("BFCL_v4_"):-len("_score.json")]
        scores[cat] = json.loads(f.read_text().splitlines()[0])["accuracy"]
    for group, parts in BFCL_GROUPS.items():
        if all(p in scores for p in parts):
            scores[group] = statistics.fmean(scores[p] for p in parts)
    return scores


def bfcl_data():
    """bfcl-eval's data folder, found without importing bfcl-eval, once its version is the pinned one."""
    try:
        installed = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        sys.exit(f"bfcl needs bfcl-eval=={BFCL_VERSION} in this Python (see README.md)")
    if installed != BFCL_VERSION:
        sys.exit(f"bfcl-eval {installed} is installed; the subset is pinned to {BFCL_VERSION}")
    return pathlib.Path(importlib.util.find_spec("bfcl_eval").origin).parent / "data"


def bfcl_run(args, root):
    """One BFCL generate and evaluate of the subset under `root`, in a process of its own: bfcl-eval fixes its folders
    from BFCL_PROJECT_ROOT when first imported, so a second run in one process would score the first's files."""
    sizes = {c: min(n, args.limit) for c, n in BFCL_SUBSET.items()} if args.limit else BFCL_SUBSET
    ids = bfcl_subset(bfcl_data(), sizes)
    (root / "test_case_ids_to_generate.json").write_text(json.dumps(ids))
    env = dict(os.environ, BFCL_PROJECT_ROOT=str(root), OPENAI_BASE_URL=args.url, OPENAI_API_KEY="none")
    subprocess.run([sys.executable, str(HERE / "run.py"), "bfcl-child", args.model, str(args.temperature),
                    ",".join(ids)], env=env, check=True)
    scores = bfcl_scores(root / "score" / BFCL_MODEL)
    if missing := [c for c in ids if c not in scores]:
        sys.exit(f"BFCL under {root} scored no {', '.join(missing)}")
    return ids, scores


def bfcl_child(model, temperature, categories):
    """`bfcl_run`'s process: larch serve as an OpenAI FC model in bfcl-eval, then BFCL's generate and evaluate."""
    config = importlib.import_module("bfcl_eval.constants.model_config")
    handler = importlib.import_module("bfcl_eval.model_handler.api_inference.openai_completion")
    cli = importlib.import_module("bfcl_eval.__main__").cli
    config.MODEL_CONFIG_MAPPING[BFCL_MODEL] = config.ModelConfig(
        model_name=model, display_name="larch serve (FC)", url="", org="larch", license="",
        model_handler=handler.OpenAICompletionsHandler, input_price=None, output_price=None, is_fc_model=True,
        underscore_to_dot=True)
    # ponytail: one request at a time, as the app's chat; more threads when a gate run's time matters
    cli(["generate", "--model", BFCL_MODEL, "--run-ids", "--temperature", temperature, "--num-threads", "1"],
        standalone_mode=False)
    cli(["evaluate", "--model", BFCL_MODEL, "--test-category", categories, "--partial-eval"], standalone_mode=False)


def tau2_scores(results):
    """pass^1 of a tau2 results.json as tau2 takes it (the mean reward of the simulations that got one), the same with
    a simulation without a reward (an error) counted as a fail, and the errors."""
    sims = json.loads(results.read_text())["simulations"]
    rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
    got = [r for r in rewards if r is not None]
    return {"pass^1": sum(got) / len(got) if got else 0.0, "pass^1_errors_fail": sum(got) / len(rewards),
            "errors": len(rewards) - len(got), "tasks": len(rewards)}


def tau2_run(args, name, n):
    """Run `n` of tau2 Retail in the checkout at `--tau2-dir`, saved as `name`, the agent on larch serve, the customer
    `--user-llm`."""
    agent = {"api_base": args.url, "api_key": "none", "temperature": args.temperature, "seed": n}
    cmd = ["uv", "run", "--directory", args.tau2_dir, "tau2", "run", "--domain", "retail",
           "--agent-llm", f"openai/{args.model}", "--agent-llm-args", json.dumps(agent),
           "--user-llm", args.user_llm, "--num-trials", "1", "--seed", str(n), "--save-to", name]
    if args.limit:
        cmd += ["--num-tasks", str(args.limit)]
    # No stdin: a save name tau2 has seen makes it ask whether to resume, which must fail, not wait.
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return tau2_scores(pathlib.Path(args.tau2_dir) / "data" / "simulations" / name / "results.json")


# A report's settings besides each bench's: the gate compares only reports run the same way.
SETTINGS = ("temperature", "limit")
# A bench's own keys that are results, not settings.
RESULTS = {"runs", "rows", "saved"}


def gate(base, new):
    """The scores of `new` that fell below `base`'s mean by more than its run-to-run spread (D10: the noise of the
    baseline's runs), a bench `new` lacks and a setting the two differ in, each as one line."""
    failed = [f"{k}: {base.get(k)} in the baseline, {new.get(k)} here" for k in SETTINGS if base.get(k) != new.get(k)]
    for bench, b in base.items():
        if not isinstance(b, dict) or "runs" not in b:
            continue
        if bench not in new:
            failed.append(f"{bench}: missing")
            continue
        for k in sorted((b.keys() | new[bench].keys()) - RESULTS):
            if b.get(k) != new[bench].get(k):
                failed.append(f"{bench} {k}: differs from the baseline")
        for score in b["runs"][0]:
            if score in COUNTS:
                continue
            was = [r[score] for r in b["runs"] if score in r]
            now = [r[score] for r in new[bench]["runs"] if score in r]
            if not now:
                failed.append(f"{bench} {score}: missing")
            elif statistics.fmean(now) < statistics.fmean(was) - (max(was) - min(was)):
                failed.append(f"{bench} {score}: {statistics.fmean(now):.3f} against {statistics.fmean(was):.3f} "
                              f"(spread {max(was) - min(was):.3f})")
    return failed


def write(path, report):
    path.write_text(json.dumps(report, indent=1) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="bench", required=True)
    for name in ["bfcl", "when2call", "tau2"]:
        s = sub.add_parser(name)
        s.add_argument("--url", required=True, help="larch serve's OpenAI base URL, e.g. http://127.0.0.1:8080/v1")
        s.add_argument("--model", required=True, help="the model name the server answers to")
        s.add_argument("--out", required=True, help="the report; must not exist yet")
        s.add_argument("--runs", type=int, default=1, help="3 for a baseline, so the gate knows the noise")
        s.add_argument("--temperature", type=float, default=0.7, help="Qwen's recommended sampling")
        s.add_argument("--limit", type=int, help="the first N items (BFCL: per category), to try it out")
    sub.choices["tau2"].add_argument("--tau2-dir", required=True, help=f"a checkout of tau2-bench at {TAU2_TAG}")
    sub.choices["tau2"].add_argument("--user-llm", help="the fixed model that plays the customer (paid)")
    g = sub.add_parser("gate")
    g.add_argument("baseline")
    g.add_argument("report")
    args = p.parse_args(argv)

    if args.bench == "gate":
        failed = gate(json.loads(pathlib.Path(args.baseline).read_text()),
                      json.loads(pathlib.Path(args.report).read_text()))
        print("\n".join(failed) or "gate passed")
        sys.exit(1 if failed else 0)
    if args.bench == "tau2" and not args.user_llm:
        p.error("tau2 needs --user-llm: the customer is a paid model, the same one every run")
    out = pathlib.Path(args.out)
    if out.exists():
        p.error(f"{out} exists; a new run needs a new name")
    roots = [out.parent / f"{out.stem}-bfcl-{n}" for n in range(args.runs if args.bench == "bfcl" else 0)]
    if left := [str(r) for r in roots if r.exists()]:
        p.error(f"{', '.join(left)} left from an earlier run: delete them or pick another --out")
    try:
        with urllib.request.urlopen(f"{args.url}/models", timeout=30) as r:
            models = json.load(r)
    except OSError as e:
        sys.exit(f"cannot reach {args.url}/models: {e}")
    started = datetime.datetime.now(datetime.timezone.utc)
    report = {"date": started.isoformat(timespec="seconds"), "url": args.url,
              "served": [m["id"] for m in models["data"]], "temperature": args.temperature, "limit": args.limit,
              "reference": REFERENCE[args.bench]}
    runs = report.setdefault(args.bench, {"runs": []})["runs"]
    if args.bench == "when2call":
        items = when2call_items(HERE / "data")[: args.limit]
        report["when2call"].update(sha256=WHEN2CALL_SHA256, rows=[])
        for n in range(args.runs):
            rows = []
            for item in items:
                rows.append(when2call_row(args.url, args.model, item, args.temperature, n))
                print(f"run {n} {len(rows)}/{len(items)} {item['uuid']} right={rows[-1]['right']}", file=sys.stderr)
            runs.append(when2call_scores(rows))
            report["when2call"]["rows"].append(rows)
            write(out, report)
    for root in roots:
        root.mkdir(parents=True)
        report["bfcl"]["ids"], scores = bfcl_run(args, root)
        report["bfcl"]["version"] = BFCL_VERSION
        runs.append(scores)
        write(out, report)
    for n in range(args.runs if args.bench == "tau2" else 0):
        # A new save name each run, so tau2 never finds an earlier run's simulations under it.
        name = f"larch-{out.stem}-{started:%Y%m%dT%H%M%SZ}-{n}"
        report["tau2"].update(tag=TAU2_TAG, user_llm=args.user_llm)
        report["tau2"].setdefault("saved", []).append(name)
        runs.append(tau2_run(args, name, n))
        write(out, report)
    write(out, report)
    for score in runs[0] if runs else []:
        mean = statistics.fmean(r[score] for r in runs)
        ref = next((f"  ({r['value']} {r['model']})" for r in REFERENCE[args.bench] if r["score"] == score
                    and r["value"] is not None), "")
        print(f"{args.bench} {score}: {mean:.3f}{ref}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["bfcl-child"]:
        bfcl_child(*sys.argv[2:])
    else:
        main()
