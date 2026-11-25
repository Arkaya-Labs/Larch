#!/usr/bin/env python3
"""The llama.cpp and Ollama baselines (architecture 16), one script for every machine. Python 3.11+, stdlib only.

    python3 scripts/baseline.py --machine <name> --models <dir> <step> [options]

machines: win5070 wsl5070 win4060 sm86 mac, plus cpu (driver checks only, never a baseline number)
steps, in order:
    models        fetch the machine's GGUFs from bench/models.toml as <dir>/<alias>.gguf; verify sha256
    build         llama.cpp b11458 (llama-bench, llama-batched-bench, llama-server) + tools/llama-serve-bench
    manifest      bench/manifest/<alias>.jsonl: every workload turn tokenised by the real GGUF (vocab only)
    flags         the honest flag search: -ngl 99 (0 on cpu) -fa on -ctk f16 -ctv f16 fixed; -t, -b, -ub searched
    run           every row of the machine into <out>/<machine>-<alias>-<row>.json
    check         the workload spec against the manifests, then the honesty checks over <out>
    server        per model, one W1 conversation through a real llama-server over HTTP vs the driver, per turn
    ollama        Modelfiles and `ollama create` for both contexts (8192, 16384)
    ollama-serve  `ollama serve` with the D15 environment; --parallel N
    compare A B   two engines' runs of one row: refuse unless they are complete, clean and comparable
"""

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import time
import tomllib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
PROMPTS = BENCH / "prompts"
FIXED = ["-npp", "512", "-ntg", "128", "-npl", "1,4,8,16", "-c", "16384"]  # W6, as llama-batched-bench runs it
FIXED_SHAPES = {(512, 128, pl) for pl in (1, 4, 8, 16)}
SINGLE = ["-p", "512,2048,4096", "-n", "128"]
SINGLE_TESTS = {(512, 0), (2048, 0), (4096, 0), (0, 128)}  # (n_prompt, n_gen) rows of llama-bench
TOLERANCE = 0.03  # the fixed-shape mirror's S must match llama-batched-bench's within 3 %, at every shape
# Back-to-back llama-batched-bench processes differ by up to 8.4 % at one shape on wsl5070, and in about one process in
# four its npl 4 alone runs 3.5 % slow, so neither one run a side nor a median resolves 3 % (llama-batched-bench against
# itself passes a median-of-5 gate 71-76 % of the time). The fixed row alternates the two this many times in one hold
# and compares S over all runs: 99.8-100 % against itself.
FIXED_REPS = 9

# Rows: (name, workload file, concurrency, cache_prompt, per-sequence context or None for "longest + output").
W1 = [("w1", "w1-chat.jsonl", 1, True, 8192)]
W2 = [(f"w2-{n}", "w2-chats.jsonl", n, True, 8192) for n in (4, 8)]
W3 = [(f"w3-{n}", "w3-agent.jsonl", n, True, 16384) for n in (1, 3)]
W4 = [(f"w4-{n}", "chat.jsonl", n, False, None) for n in (1, 4, 8, 16)]
W5 = [("w5", "w5-longprompt.jsonl", 1, False, 8192)]
CUDA = ["-DGGML_CUDA=ON"]
MACHINES = {
    "win5070": dict(cmake=CUDA, models=["m1", "m5"], rows=W1 + W2 + W3 + W4 + W5, fixed=True),
    "wsl5070": dict(cmake=CUDA, env={"CUDAHOSTCXX": "/usr/bin/g++-14"}, models=["m1", "m5"], rows=W1 + W2 + W3 + W4 + W5, fixed=True),
    "win4060": dict(cmake=CUDA, models=["m1", "m5"], rows=W1 + W2 + W3 + W4 + W5, fixed=True),
    # The 3050: 3B and 8B single-stream.
    "sm86": dict(cmake=CUDA, models=["m6", "m1"], rows=W1 + W5, fixed=False),
    # The base M5: W2 at 4, W3 with one loop, no W4.
    "mac": dict(cmake=["-DGGML_METAL=ON"], models=["m1", "m5"], rows=W1 + W2[:1] + W3[:1] + W5, fixed=True),
    # Every GPU backend off (Metal is on by default on Apple hosts) and nothing offloaded.
    "cpu": dict(cmake=["-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_HIP=OFF", "-DGGML_VULKAN=OFF"], ngl=0,
                models=["m4"], rows=W1 + W2[:1] + W5, fixed=True),
}
OLLAMA_CTX = (8192, 16384)


def die(msg):
    sys.exit(f"baseline: {msg}")


def run(cmd, check=True, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=check, **kw)


def attempt(what, cmd, result, stdout=False):
    """One benchmark row into `result`. The old result goes first, so a failure can never leave an earlier
    success behind; a failure leaves <result>.failed instead, and the next row runs."""
    failed = result.with_suffix(".failed")
    result.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    with open(result if stdout else os.devnull, "w") as f:
        code = run(cmd, check=False, stdout=f if stdout else None).returncode
    if code:
        if stdout:
            result.unlink()  # partial output
        failed.write_text(f"exit {code}\n")  # no command line: results may be committed, and paths name people
        print(f"{what}: FAILED with exit {code} (see the output above)", flush=True)
    return code == 0


def models_toml():
    return tomllib.loads((BENCH / "models.toml").read_text())


def gguf(a, alias):
    path = a.models / f"{alias}.gguf"
    want = models_toml()[alias]["size"]
    if not path.exists() or path.stat().st_size != want:
        die(f"{path} is missing or not {want} bytes: run the models step")
    return path


def tool(a, name):
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (a.build / "bin", a.build / "bin" / "Release"):
        if (d / exe).exists():
            return d / exe
    die(f"{exe} is not under {a.build}/bin: run the build step")


def out_path(a, alias, row, ext="json"):
    return a.out / f"{a.machine}-{alias}-{row}.{ext}"


def step_models(a):
    table = models_toml()
    a.models.mkdir(parents=True, exist_ok=True)
    for alias in a.alias or MACHINES[a.machine]["models"]:
        m, path = table[alias], a.models / f"{alias}.gguf"
        if not path.exists():
            part = path.with_suffix(".part")
            url = f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}"
            print(f"{alias}: downloading {m['size'] / 1e9:.1f} GB", flush=True)
            with urllib.request.urlopen(url) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 24)
            part.rename(path)
        with open(path, "rb") as f:
            got = hashlib.file_digest(f, "sha256").hexdigest()
        if got != m["sha256"]:
            die(f"{alias}: sha256 {got} is not {m['sha256']}; delete {path} and fetch again")
        print(f"{alias}: sha256 verified")


def step_build(a):
    m = MACHINES[a.machine]
    env = {**os.environ, **m.get("env", {})}
    src = ["-DFETCHCONTENT_SOURCE_DIR_LLAMA=" + str(a.llama_src)] if a.llama_src else []
    run(["cmake", "-S", ROOT / "tools/llama-serve-bench", "-B", a.build, "-DCMAKE_BUILD_TYPE=Release", *m["cmake"], *src], env=env)
    run(["cmake", "--build", a.build, "--config", "Release", "-j", str(os.cpu_count()), "--target",
         "llama-serve-bench", "llama-bench", "llama-batched-bench", "llama-server"], env=env)


def step_manifest(a):
    (BENCH / "manifest").mkdir(exist_ok=True)
    workloads = sorted(PROMPTS.glob("*.jsonl"))
    for alias in a.alias or MACHINES[a.machine]["models"]:
        args = [x for w in workloads for x in ("--workload", w)]
        r = run([tool(a, "llama-serve-bench"), "manifest", "-m", gguf(a, alias), *args], stdout=subprocess.PIPE)
        # LF on every OS: a Windows C runtime writes CRLF to stdout.
        (BENCH / "manifest" / f"{alias}.jsonl").write_bytes(r.stdout.replace(b"\r\n", b"\n"))


def manifest(alias):
    path = BENCH / "manifest" / f"{alias}.jsonl"
    if not path.exists():
        die(f"{path} is missing: run the manifest step")
    return [json.loads(line) for line in path.open()]


def ngl(a):
    return MACHINES[a.machine].get("ngl", 99)


def best_flags(a, alias):
    path = out_path(a, alias, "llama-flags")
    if not path.exists():
        die(f"{path} is missing: run the flags step")
    return json.loads(path.read_text())


def base_flags(a, alias):
    """-ngl 99 (0 on cpu) -fa on -ctk f16 -ctv f16, fixed, plus the searched -t -b -ub from the flags step."""
    best = best_flags(a, alias)
    return ["-m", gguf(a, alias), "-ngl", str(ngl(a)), "-fa", "on", "-ctk", "f16", "-ctv", "f16",
            "-t", str(best["t"]), "-b", str(best["b"]), "-ub", str(best["ub"])]


def row_ctx(row, peaks):
    """Per-sequence context of a row; None means the longest prompt plus its output, rounded up to 256."""
    _, workload, _, _, ctx = row
    return ctx or -(-(peaks[workload] + 1) // 256) * 256


def row_of(d):
    """The row of its machine a run is (workload file, concurrency, reuse), or None: a run is labelled and checked by its
    machine's table, never by its own fields alone."""
    rows = MACHINES[d["machine"]]["rows"] if d["machine"] in MACHINES else []
    return next((r for r in rows if r[1:4] == (d["workload"]["file"], d["concurrency"], d["cache_prompt"])), None)


def expected_requests(alias, row):
    """{(conv, turn): manifest line} a row must answer: the first n conversations with reuse, else all."""
    _, workload, n, cache, _ = row
    lines = [r for r in manifest(alias) if r["workload"] == workload]
    convs = list(dict.fromkeys(r["conv"] for r in lines))[: n if cache else None]
    return {(r["conv"], r["turn"]): r for r in lines if r["conv"] in convs}


def step_flags(a):
    n = os.cpu_count() or 4
    threads = ",".join(str(t) for t in sorted({t for t in (4, 8, n // 2, n) if 1 <= t <= n}))
    a.out.mkdir(parents=True, exist_ok=True)
    for alias in a.alias or MACHINES[a.machine]["models"]:
        rows = []
        for b in (512, 1024, 2048):
            ub = ",".join(str(u) for u in (128, 256, 512, 1024) if u <= b)
            r = run([tool(a, "llama-bench"), "-m", gguf(a, alias), "-ngl", ngl(a), "-fa", "on", "-ctk", "f16", "-ctv", "f16",
                     "-t", threads, "-b", b, "-ub", ub, "-p", "2048", "-n", "128", "-o", "csv"], stdout=subprocess.PIPE, text=True)
            rows += list(csv.DictReader(r.stdout.splitlines()))
        with open(out_path(a, alias, "llama-flags", "csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        # Each (t, b, ub) scores the geometric mean of its pp2048 and tg128 speeds; a combination missing
        # either test (llama-bench skipped or failed it) is out.
        speed = {}
        for r in rows:
            key = (int(r["n_threads"]), int(r["n_batch"]), int(r["n_ubatch"]))
            speed.setdefault(key, {})["pp" if int(r["n_prompt"]) else "tg"] = float(r["avg_ts"])
        speed = {k: v for k, v in speed.items() if len(v) == 2}
        if not speed:
            die(f"{alias}: llama-bench produced no complete (t, b, ub) result")
        (t, b, ub), s = max(speed.items(), key=lambda kv: kv[1]["pp"] * kv[1]["tg"])
        best = {"t": t, "b": b, "ub": ub, "pp2048_ts": s["pp"], "tg128_ts": s["tg"], "rule": "max sqrt(pp2048 * tg128)"}
        out_path(a, alias, "llama-flags").write_text(json.dumps(best, indent=1) + "\n")
        print(f"{alias}: {best}")


def basename(path):
    return re.split(r"[\\/]", path)[-1]


def peaks(alias):
    out = {}
    for r in manifest(alias):
        out[r["workload"]] = max(out.get(r["workload"], 0), r["peak_tokens"])
    return out


def step_run(a):
    m = MACHINES[a.machine]
    names = {"single"} | ({"fixed"} if m["fixed"] else set()) | {r[0] for r in m["rows"]}
    if a.row and not set(a.row) <= names:
        die(f"--machine {a.machine} has no rows {sorted(set(a.row) - names)}; it has {sorted(names)}")
    a.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for alias in a.alias or m["models"]:
        flags, peak = base_flags(a, alias), peaks(alias)
        wanted = lambda row: not a.row or row in a.row  # noqa: E731
        jobs = []
        if wanted("single"):
            jobs.append(("single", [tool(a, "llama-bench"), *flags, *SINGLE, "-o", "json"], out_path(a, alias, "single"), True))
        if m["fixed"] and wanted("fixed"):
            jobs.append(("fixed", None, None, None))  # run_fixed: both tools, alternated
        for row in m["rows"]:
            name, workload, n, cache, _ = row
            if wanted(name):
                jobs.append((name, [tool(a, "llama-serve-bench"), "run", "--workload", PROMPTS / workload, "--concurrency", n,
                                    "--conversations", n if cache else 0, "--machine", a.machine, "--json",
                                    out_path(a, alias, name), *flags, "-c", n * row_ctx(row, peak)]
                             + (["--cache-prompt"] if cache else []), out_path(a, alias, name), False))
        for name, cmd, result, stdout in jobs:
            if not (run_fixed(a, alias, flags) if name == "fixed" else attempt(f"{alias} {name}", cmd, result, stdout)):
                failed.append(f"{alias} {name}")
            elif name == "single":
                # llama-bench records the model's full local path; keep only the alias file name, since results
                # may be committed and paths name people.
                d = load(result)
                if not isinstance(d, list):
                    result.unlink()
                    result.with_suffix(".failed").write_text("exit 0, but not llama-bench JSON\n")
                    failed.append(f"{alias} {name}")
                    continue
                for r in d:
                    r["model_filename"] = basename(str(r.get("model_filename", "")))
                result.write_text(json.dumps(d, indent=1) + "\n")
    if failed:
        die("rows failed: " + ", ".join(failed))


def run_fixed(a, alias, flags):
    """W6's gate evidence: llama-batched-bench and the driver's fixed shape alternated FIXED_REPS times, every row of
    every run into <batched>.jsonl and <fixed>.json. A failed run leaves neither file, only its .failed."""
    bb, fx = out_path(a, alias, "batched", "jsonl"), out_path(a, alias, "fixed")
    rows, doc = [], None
    for i in range(1, FIXED_REPS + 1):
        ok = (attempt(f"{alias} batched {i}", [tool(a, "llama-batched-bench"), *flags, *FIXED, "--output-format", "jsonl"],
                      bb, True)
              and attempt(f"{alias} fixed {i}", [tool(a, "llama-serve-bench"), "fixed", *flags, *FIXED, "--machine",
                                                 a.machine, "--json", fx], fx, False))
        d = load(fx) if ok else None
        if not (isinstance(d, dict) and isinstance(d.get("fixed"), list)):
            if ok:
                fx.with_suffix(".failed").write_text("exit 0, but not a fixed-shape result\n")
            bb.unlink(missing_ok=True)
            fx.unlink(missing_ok=True)
            return False
        rows += batched_rows(bb)
        doc = d if doc is None else {**doc, "fixed": doc["fixed"] + d["fixed"]}
    bb.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fx.write_text(json.dumps(doc, indent=1) + "\n")
    return True


def check_spec(alias):
    """The workload definitions of architecture 16, against what the real tokenizer measured."""
    rows = manifest(alias)
    by = {}
    for r in rows:
        by.setdefault(r["workload"], []).append(r)
    bad = []

    def need(ok, what):
        if not ok:
            bad.append(what)

    def convs(w):
        out = {}
        for r in by[w]:
            out.setdefault(r["conv"], []).append(r)
        return out

    chat = by["chat.jsonl"]
    need(len(chat) == 200, "chat: 200 prompts")
    need(all(50 <= r["user_tokens"] and r["prompt_tokens"] <= 2000 for r in chat), "chat: prompts 50 to 2000 tokens")
    need(all(128 <= r["max_tokens"] <= 512 for r in chat), "chat: answers 128 to 512")
    for w, n_conv, n_turn in (("w1-chat.jsonl", 1, 12), ("w2-chats.jsonl", 8, 8)):
        c = convs(w)
        need(len(c) == n_conv and all(len(t) == n_turn for t in c.values()), f"{w}: {n_conv} x {n_turn} turns")
        need(all(40 <= r["user_tokens"] <= 300 for r in by[w]), f"{w}: user messages 40 to 300")
        need(all(100 <= r["max_tokens"] <= 400 for r in by[w]), f"{w}: answers 100 to 400")
        need(all(r["scripted_tokens"] in (None, r["max_tokens"]) for r in by[w]), f"{w}: scripted answers = target")
        need(all(r["peak_tokens"] <= 8192 for r in by[w]), f"{w}: fits 8192")
    c = convs("w3-agent.jsonl")
    need(len(c) == 3 and all(len(t) == 20 for t in c.values()), "w3: 3 loops x 20 calls")
    w3 = by["w3-agent.jsonl"]
    need(all(r["user_tokens"] == 130 and r["max_tokens"] == 120 for r in w3), "w3: 130-token results, 120-token answers")
    need(all(r["scripted_tokens"] in (None, 120) for r in w3), "w3: scripted answers 120")
    need(all(t[0]["prompt_tokens"] - 130 >= 4096 for t in c.values()), "w3: a 4096-token prefix")
    need(all(r["peak_tokens"] < 9600 for r in w3), "w3: peak under 9.6k")
    w5 = by["w5-longprompt.jsonl"]
    need(sorted(r["user_tokens"] for r in w5) == [2048, 4096] and all(r["max_tokens"] == 128 for r in w5), "w5: 2048 and 4096, 128 out")
    return bad


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def check_params(d, a, alias):
    """The run used the machine's fixed flags and the searched ones."""
    best, p = best_flags(a, alias), d["params"]
    want = {"n_gpu_layers": ngl(a), "flash_attn": "enabled", "type_k": "f16", "type_v": "f16", "kv_unified": False,
            "n_batch": best["b"], "n_ubatch": best["ub"], "n_threads": best["t"]}
    return [f"params.{k} = {p.get(k)!r}, not {v!r}" for k, v in want.items() if p.get(k) != v]


def check_header(d, a, alias, mode):
    if not isinstance(d, dict):
        return ["not a result object"]
    want = {"schema": "larch-bench/1", "mode": mode, "engine": "llama.cpp", "machine": a.machine, "model": f"{alias}.gguf"}
    bad = [f"{k} = {d.get(k)!r}, not {v!r}" for k, v in want.items() if d.get(k) != v]
    # The flags only once the object is a result at all; a flag mismatch hides nothing else.
    return bad or check_params(d, a, alias)


def check_run(d, a, alias, row, peak):
    """One workload row: the exact request set of the manifest, every answer its full length, no error."""
    _, workload, n, cache, _ = row
    bad = check_header(d, a, alias, "run")
    if not isinstance(d, dict) or d.get("mode") != "run":
        return bad
    want = expected_requests(alias, row)
    sha = next(iter(want.values()))["workload_sha256"]
    for k, v in {("workload", "file"): workload, ("workload", "sha256"): sha, ("concurrency",): n,
                 ("cache_prompt",): cache, ("params", "n_ctx_seq"): row_ctx(row, peak)}.items():
        got = d
        for key in k:
            got = got.get(key) if isinstance(got, dict) else None
        if got != v:
            bad.append(f"{'.'.join(k)} = {got!r}, not {v!r}")
    return bad + check_requests(d, want)


def check_requests(d, want):
    """Exactly the manifest's requests, once each, with its prompt hashes and full answer lengths, no error."""
    bad, reqs = [], d.get("requests", [])
    keys = [(r["conv"], r["turn"]) for r in reqs]
    if len(keys) != len(set(keys)) or set(keys) != set(want):
        bad.append(f"{len(set(keys))} distinct requests of {len(keys)}, the manifest expects exactly {len(want)}")
    for r in reqs:
        m = want.get((r["conv"], r["turn"]))
        if m and (r["error"] is not None or r["prompt_sha256"] != m["prompt_sha256"]
                  or not r["gen_tokens"] == r["max_tokens"] == m["max_tokens"]):
            bad.append(f"{r['conv']} turn {r['turn']}: error {r['error']!r}, hash or length off the manifest")
    if d["summary"]["failures"] or d["summary"]["requests"] != len(want):
        bad.append(f"summary: {d['summary']['requests']} requests, {d['summary']['failures']} failed")
    return bad


def check_single(d, a, alias):
    if not isinstance(d, list):
        return ["not a llama-bench JSON list"]
    best = best_flags(a, alias)
    got = {(r.get("n_prompt"), r.get("n_gen")) for r in d if r.get("avg_ts", 0) > 0}
    bad = [f"llama-bench test {t} missing" for t in sorted(SINGLE_TESTS - got)]
    flags = {(basename(r.get("model_filename", "")), r.get("flash_attn"), r.get("n_gpu_layers"), r.get("n_batch"),
              r.get("n_ubatch"), r.get("n_threads"), r.get("type_k"), r.get("type_v")) for r in d}
    if flags != {(f"{alias}.gguf", 1, ngl(a), best["b"], best["ub"], best["t"], "f16", "f16")}:
        bad.append(f"llama-bench model or flags {sorted(flags)} are not {alias}.gguf with the machine's flags")
    return bad


def batched_rows(path):
    """llama-batched-bench's JSONL rows (the W6 baseline), or none when the file is missing."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")] if path.exists() else []


def pooled(rows):
    """{(pp, tg, pl): (S over all runs, number of runs)} over fixed-shape rows: every run of a shape processes the same
    tokens, so the sum of tokens over the sum of time is the harmonic mean of the runs' S."""
    by = {}
    for r in rows:
        by.setdefault((r["pp"], r["tg"], r["pl"]), []).append(r["speed"])
    return {t: (statistics.harmonic_mean(v), len(v)) for t, v in by.items()}


def check_fixed(a, alias):
    """Every pp512/tg128 shape in both files, FIXED_REPS runs a side, the driver's S over its runs within 3 % of
    llama-batched-bench's at each."""
    bad = []
    rows = batched_rows(out_path(a, alias, "batched", "jsonl"))
    bb = pooled(rows)
    fx = load(out_path(a, alias, "fixed"))
    bad += [f"batched: {e}" for e in [f"shape {t} missing" for t in sorted(FIXED_SHAPES - bb.keys())]]
    best = best_flags(a, alias)
    # Its JSONL names no model; the driver's fixed file beside it does, and both come from one run step. A shared
    # prompt (-pps) prefills once and copies the KV, which is not the workload the driver runs.
    if {(r["is_pp_shared"], r["flash_attn"], r["n_batch"], r["n_ubatch"], r["n_threads"], r["n_gpu_layers"])
            for r in rows} - {(0, 1, best["b"], best["ub"], best["t"], ngl(a))}:
        bad.append("batched: shared prompt, flash attention or flags are not the machine's")
    head = check_header(fx, a, alias, "fixed")
    bad += [f"fixed: {e}" for e in head]
    if not isinstance(fx, dict) or fx.get("mode") != "fixed":
        return bad
    drvs = pooled(fx.get("fixed", []))
    bad += [f"fixed: shape {t} missing" for t in sorted(FIXED_SHAPES - drvs.keys())]
    for t in sorted(FIXED_SHAPES & drvs.keys() & bb.keys()):
        (drv, n_drv), (ref, n_ref) = drvs[t], bb[t]
        if (n_drv, n_ref) != (FIXED_REPS, FIXED_REPS):
            bad.append(f"fixed npl={t[2]}: {n_drv} driver and {n_ref} llama-batched-bench runs, not {FIXED_REPS} each")
            continue
        if drv <= 0 or ref <= 0:  # a run at 0 t/s makes the harmonic mean 0
            bad.append(f"fixed npl={t[2]}: S {drv} driver, {ref} llama-batched-bench, not positive")
            continue
        dev = drv / ref - 1
        ok = abs(dev) <= TOLERANCE
        print(f"{alias} fixed pp={t[0]} tg={t[1]} npl={t[2]}: S over {FIXED_REPS} runs, driver {drv:.2f} t/s, "
              f"llama-batched-bench {ref:.2f} t/s, {dev:+.1%} {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"fixed npl={t[2]} off by {dev:+.1%}")
    return bad


def check_runs(a, alias):
    """Every row the machine runs must be there, complete and clean; missing or partial evidence fails."""
    m, peak, bad = MACHINES[a.machine], peaks(alias), []
    names = [r[0] for r in m["rows"]] + ["single"] + (["fixed", "batched"] if m["fixed"] else [])
    for name in names:
        ext = "jsonl" if name == "batched" else "json"
        if out_path(a, alias, name, "failed").exists():
            bad.append(f"{name}: failed run ({out_path(a, alias, name, 'failed').read_text().strip()})")
        elif not out_path(a, alias, name, ext).exists():
            bad.append(f"{name}: no result in {a.out}")
    single = out_path(a, alias, "single")
    try:
        if single.exists():
            bad += [f"single: {e}" for e in check_single(load(single), a, alias)]
        if m["fixed"]:
            bad += check_fixed(a, alias)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        bad.append(f"single or fixed: malformed result ({type(e).__name__}: {e})")
    for row in m["rows"]:
        path = out_path(a, alias, row[0])
        if path.exists():
            d = load(path)
            try:
                errs = check_run(d, a, alias, row, peak)
            except (KeyError, TypeError, AttributeError, StopIteration) as e:
                errs = [f"malformed result ({type(e).__name__}: {e})"]
            s = d.get("summary") if isinstance(d, dict) and isinstance(d.get("summary"), dict) else {}
            print(f"{path.name}: {s.get('requests')} requests, {s.get('failures')} failed, "
                  f"{s.get('agg_tok_s') or float('nan'):.2f} tok/s, {len(errs)} problems")
            bad += [f"{row[0]}: {e}" for e in errs]
    return bad


def step_check(a):
    bad = []
    for alias in a.alias or MACHINES[a.machine]["models"]:
        bad += [f"{alias}: {b}" for b in check_spec(alias) + check_runs(a, alias)]
    if bad:
        die("check failed:\n  " + "\n  ".join(bad))
    print("check: all passed")


def http(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def step_server(a):
    """architecture 16: per model, one W1 conversation through a real llama-server reports the driver's per-turn counts."""
    for alias in a.alias or MACHINES[a.machine]["models"]:
        server_check(a, alias)


def server_check(a, alias):
    """One model's server check into <machine>-<alias>-server-check.json, with the driver's flags and llama.cpp build so
    the scoreboard can tell which rows it vouches for."""
    out_path(a, alias, "server-check").unlink(missing_ok=True)  # a failed rerun never leaves an earlier pass behind
    flags = base_flags(a, alias)
    want = {r["turn"]: r for r in manifest(alias) if r["workload"] == "w1-chat.jsonl"}
    conv = json.loads((PROMPTS / "w1-chat.jsonl").read_text().splitlines()[0])
    drv_path = out_path(a, alias, "server-check-driver")
    run([tool(a, "llama-serve-bench"), "run", "--workload", PROMPTS / "w1-chat.jsonl", "--concurrency", 1,
         "--cache-prompt", "--machine", a.machine, "--json", drv_path, *flags, "-c", 8192])
    drv_json = json.loads(drv_path.read_text())
    drv = drv_json["requests"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([str(c) for c in [tool(a, "llama-server"), *flags, "-c", 8192, "-np", 1, "--port", port]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    got = []
    try:
        for _ in range(600):
            if srv.poll() is not None:
                die(f"llama-server exited with {srv.returncode} before it was ready")
            try:
                if http(port, "/health").get("status") == "ok":
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            die("llama-server was not ready after 600 s")
        # The driver's own render date, so both sides render the same prompt.
        kw = {"chat_template_kwargs": {"date_string": drv_json["date_string"]}}
        msgs = conv["messages"]
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for k, i in enumerate(turns):
            history = [{"role": m["role"], "content": m["content"]} for m in msgs[:i]]
            prompt = http(port, "/apply-template", {"messages": history, **kw})["prompt"]
            ids = http(port, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
            sha = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
            r = http(port, "/v1/chat/completions", {"messages": history, "max_tokens": msgs[i]["max_tokens"],
                                                    "temperature": 0, "ignore_eos": True, "cache_prompt": True, **kw})
            t = r["timings"]
            got.append({"turn": k, "prompt_tokens": len(ids), "sha_ok": sha == want[k]["prompt_sha256"],
                        "cache_n": t["cache_n"], "prompt_n": t["prompt_n"], "gen": r["usage"]["completion_tokens"]})
    finally:
        srv.terminate()
        srv.wait()
    bad = 0
    print("turn  prompt  server cache_n/prompt_n  driver cached/evaluated  gen  hash")
    for g, d in zip(got, drv):
        ok = (g["sha_ok"] and g["prompt_tokens"] == d["prompt_tokens"] and g["cache_n"] == d["cached_tokens"]
              and g["prompt_n"] == d["evaluated_tokens"] and g["gen"] == d["gen_tokens"])
        bad += not ok
        print(f"{g['turn']:4}  {g['prompt_tokens']:6}  {g['cache_n']:7}/{g['prompt_n']:<14}  "
              f"{d['cached_tokens']:6}/{d['evaluated_tokens']:<16}  {g['gen']:3}  {'ok' if ok else 'MISMATCH'}")
    ok = not bad and len(got) == len(drv)
    out_path(a, alias, "server-check").write_text(json.dumps(
        {"alias": alias, "machine": a.machine, "params": drv_json["params"], "llama_cpp": drv_json.get("llama_cpp"),
         "build": drv_json.get("build"),
         "server": got, "driver": drv_path.name, "ok": ok}, indent=1) + "\n")
    if not ok:
        die(f"server check: {bad} turns differ")
    print("server check: every turn matches")


def step_ollama(a):
    run(["ollama", "--version"])
    for alias in a.alias or MACHINES[a.machine]["models"]:
        for ctx in OLLAMA_CTX:
            mf = a.models / f"{alias}-{ctx}.Modelfile"
            mf.write_text(f"FROM {gguf(a, alias).resolve()}\nPARAMETER num_ctx {ctx}\n")
            run(["ollama", "create", f"larch-{alias}-{ctx}", "-f", mf])


def step_ollama_serve(a):
    # The server reads these at start, so every concurrency N is its own server run. On Windows, quit the
    # Ollama tray app first: it runs its own server without them.
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(a.parallel), "OLLAMA_FLASH_ATTENTION": "1",
           "OLLAMA_KV_CACHE_TYPE": "f16", "OLLAMA_KEEP_ALIVE": "-1"}
    run(["ollama", "serve"], env=env)


def step_compare(a):
    """Two runs of one row, by any engines. Numbers are shown only when both are complete and clean and ran
    the same thing: machine, model, workload bytes, concurrency, reuse, context, and per turn the prompt and its length."""
    runs = [load(pathlib.Path(p)) for p in a.files]
    bad = compare_problems(a.files, runs)
    if bad:
        die("refused:\n  " + "\n  ".join(bad))
    print(f"{'':10}{'engine':>12}{'agg tok/s':>11}{'TTFT p50':>10}{'TTFT p99':>10}{'TPOT p50':>10}{'TPOT p99':>10}")
    for p, d in zip(a.files, runs):
        s = d["summary"]
        q = lambda k, p: s[k][p] if s[k] else float("nan")  # noqa: E731
        print(f"{pathlib.Path(p).stem[:10]:10}{d['engine']:>12}{s['agg_tok_s']:11.2f}{q('ttft_ms', 'p50'):10.1f}"
              f"{q('ttft_ms', 'p99'):10.1f}{q('tpot_ms', 'p50'):10.2f}{q('tpot_ms', 'p99'):10.2f}")


def compare_problems(files, runs):
    """Every reason two runs are not like for like; empty when they are. Never exits: the scoreboard asks per pair."""
    bad = []
    for p, d in zip(files, runs):
        if not isinstance(d, dict) or d.get("schema") != "larch-bench/1" or d.get("mode") != "run":
            return [f"{p} is not a larch-bench/1 run"]
    x, y = runs
    try:
        for p, d in zip(files, runs):
            # Each run on its own against the manifest of its model: the row's full request set, hashes and lengths.
            alias, workload = d["model"].removesuffix(".gguf"), d["workload"]["file"]
            if alias not in models_toml():
                bad.append(f"{p} names model {d['model']!r}, not an alias of bench/models.toml")
                continue
            if not (BENCH / "manifest" / f"{alias}.jsonl").exists():
                bad.append(f"{p}: bench/manifest/{alias}.jsonl is missing: run the manifest step")
                continue
            row = row_of(d)
            if row is None:
                bad.append(f"{p}: not one of {d['machine']}'s rows ({workload} at {d['concurrency']}, "
                           f"cache_prompt {d['cache_prompt']})")
                continue
            ctx = row_ctx(row, peaks(alias))
            if d["params"]["n_ctx_seq"] != ctx:
                bad.append(f"{p}: params.n_ctx_seq = {d['params']['n_ctx_seq']!r}, not the row's {ctx}")
            if d["engine"] == "llama.cpp":
                # The searched flags of its machine: the flags step's file beside the result.
                try:
                    flags = argparse.Namespace(machine=d["machine"], out=pathlib.Path(p).parent)
                    bad += [f"{p}: {e}" for e in check_params(d, flags, alias)]
                except SystemExit as e:
                    bad.append(f"{p}: {e}")
            want = expected_requests(alias, row)
            if not want or next(iter(want.values()))["workload_sha256"] != d["workload"]["sha256"]:
                bad.append(f"{p}: {workload} is not in bench/manifest/{alias}.jsonl with these bytes")
            else:
                bad += [f"{p}: {e}" for e in check_requests(d, want)]
        for what, get in (("machine", lambda d: d["machine"]), ("model", lambda d: d["model"]),
                          ("workload", lambda d: d["workload"]["sha256"]), ("concurrency", lambda d: d["concurrency"]),
                          ("cache_prompt", lambda d: d["cache_prompt"]), ("n_ctx_seq", lambda d: d["params"]["n_ctx_seq"])):
            if get(x) != get(y):
                bad.append(f"{what} differs: {get(x)!r} vs {get(y)!r}")
        hx = {(r["conv"], r["turn"]): (r["prompt_sha256"], r["max_tokens"]) for r in x["requests"]}
        hy = {(r["conv"], r["turn"]): (r["prompt_sha256"], r["max_tokens"]) for r in y["requests"]}
        diff = sorted(k for k in hx.keys() | hy.keys() if hx.get(k) != hy.get(k))
    except (KeyError, TypeError, AttributeError, ValueError) as e:  # ValueError: a truncated manifest line
        return bad + [f"malformed result or manifest ({type(e).__name__}: {e})"]
    if diff:
        bad.append(f"prompt-token hashes or lengths differ at {len(diff)} turns, first {diff[:3]}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, choices=MACHINES)
    p.add_argument("--models", type=pathlib.Path, help="the directory of <alias>.gguf files")
    p.add_argument("--build", type=pathlib.Path, help="CMake build dir (default target/llama-serve-bench-<machine>)")
    p.add_argument("--out", type=pathlib.Path, default=BENCH / "results")
    p.add_argument("--alias", action="append", help="only these models (default: the machine's)")
    p.add_argument("--row", action="append", help="run: only these rows (single, fixed, w1, w2-4, w4-16, ...)")
    p.add_argument("--llama-src", type=pathlib.Path, help="build: a checkout of llama.cpp b11458, instead of fetching")
    p.add_argument("--parallel", type=int, default=1, help="ollama-serve: OLLAMA_NUM_PARALLEL")
    p.add_argument("step", choices=["models", "build", "manifest", "flags", "run", "check", "server", "ollama",
                                    "ollama-serve", "compare"])
    p.add_argument("files", nargs="*", help="compare: two result JSONs")
    a = p.parse_args()
    a.build = a.build or ROOT / "target" / f"llama-serve-bench-{a.machine}"
    if a.models is None and a.step not in ("build", "check", "ollama-serve", "compare"):
        die("--models <dir> is required for this step")
    if a.step == "compare" and len(a.files) != 2:
        die("compare takes two result files")
    globals()["step_" + a.step.replace("-", "_")](a)


if __name__ == "__main__":
    main()
