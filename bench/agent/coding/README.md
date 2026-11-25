# Coding-agent bench (bash only)

One command scores any OpenAI-compatible `/v1/chat/completions` server (`larch serve` first) on real coding tasks
with mini-swe-agent, the SWE-agent team's bash-only scaffold, graded by each benchmark's own harness, so a row
compares with published bash-only numbers and reruns the same way after every model or agent-loop change. Nothing
here is committed data: the tasks and their images are fetched at run time into `data/` (git-ignored) and Docker.

| Set | What | Size | Licence | Source, pinned |
|---|---|---|---|---|
| `swe-mini` | SWE-bench Verified Mini: GitHub issues in django and sphinx, resolved when the repo's tests pass | 50 instances (`swe-mini.txt`) | the dataset card names none; the script that drew it is MIT, the instances are SWE-bench's (MIT) | ids from `MariusHobbhahn/swe-bench-verified-mini` revision `b316c349947c29963fce3f4a65967c9807a4b673`; run as instances of `princeton-nlp/SWE-bench_Verified` revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, test split `https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet`, sha256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd` |
| `tb2-slice` | Terminal-Bench 2.0: terminal tasks graded by each task's own tests in its container | 20 of 89 tasks (`tb2-slice.txt`), 13 medium and 7 hard | the task repo has no licence file at that commit; each task brings its own image | Harbor dataset `terminal-bench@2.0` from the git registry `laude-institute/harbor@872da9479f4a2078142d22c49c9ffae0f09e2e63`, whose tasks are `laude-institute/terminal-bench-2` at `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` |

How the Mini set was drawn (github.com/mariushobbhahn/make_swe_bench_verified_mini): k-means over 16 models'
per-instance results on all 500 (resolve rate, test pass rate, difficulty), then a linear program that picks 50
keeping the clusters' proportions while minimising Docker storage. It keeps two environments (django and sphinx,
25 each), about 5 GB of images instead of 130 GB, with about the full set's score distribution.

The TB2 slice rule: sort the 89 task names of `terminal-bench@2.0` and take positions `floor(i * 89 / 20)` for
`i = 0..19` (every fourth or fifth name, from the first). `fixtures/tb2-tasks.txt` holds the 89 names and
`fixtures/swe-bench-verified-ids.txt` Verified's 500 ids, so `test_coding.py` checks both id files offline.

## Install

Python 3.12 or later (Harbor's floor), on Linux or WSL with Docker running:

```
python3 -m venv bench/agent/coding/.venv
bench/agent/coding/.venv/bin/pip install -r bench/agent/coding/requirements.txt
```

`run.py` itself is stdlib only. Before a run it checks that `docker info` answers and that the venv holds exactly
`requirements.txt`'s versions, and exits 2 with the fix otherwise.

## Run

```
python3 bench/agent/coding/run.py --set swe-mini  --url http://127.0.0.1:8080 --model <name> [--runs N] [--limit N] [--workers N] --out <report>.json
python3 bench/agent/coding/run.py --set tb2-slice --url http://host.docker.internal:8080 --model <name> [--runs N] [--limit N] [--workers N] --out <report>.json
```

Each task is one scaffold run plus its grading; `--workers` runs that many tasks at once, `--runs` repeats every
task, `--limit N` takes the first N ids. The report is rewritten after every task. The tools' own outputs
(trajectories, predictions, harness logs, Harbor jobs) land in `data/runs/<report name>/`; neither `--out` nor that
folder may exist yet (exit 2), because the tools skip what their folder already holds, so a rerun after a crash
needs a new name. These stop the bench with the log to read, since Docker is broken and every later task would
break too: a task whose scaffold never started (no trajectory, e.g. an image pull failed), a grading failure none of
the classes below explains, an instance the harness itself blames on the host (`infra_failure_ids`, e.g. an OOM
kill), and a Harbor exception for a sandbox that never ran the agent or its tests (image build, setup timeout,
missing reward file). Tasks already running finish and keep their rows; queued ones never start. The exit code is 1 when every run ended with `server_error`, else 0.
`LARCH_API_KEY`, when set, is sent as the API key.

### swe-mini

mini-swe-agent 2.4.6 runs on the host, one instance per `mini-extra swebench` call, with an instance filter built
from `swe-mini.txt`, its commands running in the instance's published `swebench/sweb.eval.x86_64.*` image:

```
mini-extra swebench --subset data/swe-bench-verified --split test --filter '^(<id>)$' -o <task dir> -w 1 \
  -m hosted_vllm/<model> -c swebench_backticks.yaml -c model.model_class=litellm_textbased \
  -c model.model_kwargs.api_base=<url>/v1 -c model.cost_tracking=ignore_errors
```

- `swebench_backticks.yaml` with the text-based model class is the bash-only, one-command-per-turn protocol: the
  model answers with a THOUGHT and exactly one ```` ```mswea_bash_command ```` block, no tool calling. (2.4.6's
  default `swebench.yaml` uses tool calls and allows several commands per reply.)
- `hosted_vllm/<model>` with `api_base` is the route the pinned version's local-model guide documents.
- Limits, from `swebench_backticks.yaml`: 250 model calls (`step_limit`), 60 s per command, an output over 10000
  characters shows only its first and last 5000. The config sets no context limit: the history grows until the
  agent submits or the server refuses it (past `larch serve`'s `max_context`), which ends the run as
  `server_error`. Its `cost_limit` of 3 never trips: a local server costs nothing.
- No model registry file: 2.4.6 raises on a computed cost of 0, whatever a zero-cost registry says, unless cost
  tracking ignores errors, and then the registry changes nothing.

A non-empty patch is graded by swebench 5.0.2, `python -m swebench.harness.run_evaluation -d data/swe-bench-verified
-s test -i <id> -p preds.json -id larch-coding-run<n>`, on local Docker with the same published images; tests time out
at 1800 s. The pinned test split is downloaded once into `data/swe-bench-verified/test.parquet` and checked by its
sha256 on every run; both tools read it from there.

### tb2-slice

Harbor 0.24.0 runs each task (`harbor run --repo <pin> -d terminal-bench@2.0 -i <task> -a mini-swe-agent --ak
version=2.4.6 -m openai/<model>`) with `OPENAI_BASE_URL=<url>/v1`. Harbor installs mini-swe-agent 2.4.6 inside the
task's container and runs it with its own default `mini.yaml` (tool calling; no step limit; cost limit off), bounded
by each task's agent timeout (900 s for 13 of the 20, 1800 s for 4, 3600 s and 7200 s for the last two), then runs
the task's tests: the task is resolved when the reward is 1. The agent runs in the container, so `--url` must be an
address the container reaches (`host.docker.internal` under Docker Desktop; `larch serve --host 0.0.0.0` needs
`LARCH_API_KEY`). Harbor resolves that install unpinned, and litellm 1.105.0 (11 Oct 2026) and later import the
uninstalled `mcp` package on every call that sends tools, so the container resolves as of `UV_EXCLUDE_NEWER=2026-10-06`.

## Failure classes

| `failure` | When |
|---|---|
| `null` | resolved |
| `step_limit` | mini-swe-agent stopped at its step limit (exit `LimitsExceeded`) |
| `timeout` | LiteLLM's request timed out, Harbor's `AgentTimeoutError` or `VerifierTimeoutError`, or the harness's 1800 s test timeout |
| `server_error` | an exception ended the run: the server's after LiteLLM's retries (a refused request, an overlong history), or the sandbox's; the trajectory's `info.exit_status` names it (TB2: also any other Harbor exception, chiefly the agent process failing) |
| `no_patch` | the agent submitted an empty patch (also after three format errors in a row) |
| `patch_does_not_apply` | the harness could not apply the patch (`>>>>> Patch Apply Failed` in its log) |
| `tests_fail` | graded and not resolved |

The agent's own stop (`step_limit`, `timeout`, `server_error`) wins over what its empty patch would give.

## Report (`larch-coding/1`)

`schema, set, model, url, scaffold, scaffold_version, runs, started_at, summary, rows`, where `scaffold` names the
config (and Harbor's version for TB2). A row is `{task, run, resolved, failure, steps, wall_s, prompt_tokens,
completion_tokens, patch_bytes}`: `steps` is model calls; `wall_s` is the scaffold call's wall time for swe-mini
(environment start and agent, no grading) and Harbor's agent execution time for TB2; tokens are the server's
`usage` summed over the trajectory (TB2: Harbor's totals); `patch_bytes` is the submitted patch (null for TB2).

| Summary field | How |
|---|---|
| `pass@1` | resolved share: mean over tasks of resolved runs / runs (× 100 is the published resolved %) |
| `pass^4` | mean over tasks of C(c,4)/C(n,4), the research bench's code; null unless every task has 4 or more runs |
| `steps`, `wall_s` | total and mean per run |
| `tokens` | `prompt` and `completion` totals |
| `failures` | runs per failure class |

`pass^4` needs runs that differ: neither config sets a temperature, so the runs sample at the server's default.

## The --fake gate

`python3 bench/agent/coding/run.py --fake` (stdlib only: no Docker, network or venv) feeds the committed outputs of
the real tools through the same row, summary and report code and exits 1 unless each set's summary equals
`fixtures/expected.json`: for swe-mini, `fixtures/swe-mini/run0/` holds four task dirs that mini-extra 2.4.6 wrote
(predictions, trajectories; resolved, tests_fail, patch_does_not_apply, step_limit) with swebench 5.0.2's run reports
and log; for tb2-slice, `fixtures/tb2-slice/run0/` holds two Harbor trial results (resolved, agent timeout) with the
agent's trajectory. CI runs it and `test_coding.py` under `cargo test` (`crates/larch-agent/tests/bench.rs`).

## Server compatibility

`fixtures/requests/` holds the exact request bodies the pinned agents send, recorded against a stdlib
`http.server` with scripted replies: `mini-swebench-{1,2}.json` from the swe-mini command above (first turn, and
the turn after one command), `harbor-mini-{1,2}.json` from the command Harbor's `MiniSweAgent` builds (`-c mini`, tool
calling; turn two carries a `tool` message). They hold the model, messages and, for Harbor, `tools`: none uses `n`,
penalties, `logprobs`, `response_format` or `tool_choice`. Replay them against a server before a long run; every one
must answer 200:

```
for f in bench/agent/coding/fixtures/requests/*.json; do
  curl -s -o /dev/null -w "%{http_code} $f\n" -H 'Content-Type: application/json' -d @"$f" http://127.0.0.1:8080/v1/chat/completions
done
```

## GPU check (RTX 5070 Ti, WSL, Docker Desktop): NOT RUN

Install the venv as above, serve m5 as the research bench does, reachable from containers, and replay the requests:

```
LARCH_API_KEY=<key> larch serve -m <models>/m5.gguf --host 0.0.0.0 --set max_context=32768
LARCH_API_KEY=<key> python3 bench/agent/coding/run.py --set swe-mini --url http://127.0.0.1:8080 --model m5 --runs 1 --workers 4 --out bench/results/agent-coding-m5-swe-mini.json
LARCH_API_KEY=<key> python3 bench/agent/coding/run.py --set tb2-slice --url http://host.docker.internal:8080 --model m5 --runs 1 --out bench/results/agent-coding-m5-tb2.json
```

(the replay loop above needs `-H "Authorization: Bearer <key>"` once the key is set). Expected wall time: swe-mini
about 4 h (50 tasks at roughly 20 min each over 4 workers, plus about 5 GB of first-time image pulls); tb2-slice up to
about 9 h, bounded by the 20 agent timeouts (29700 s in all) plus image builds and tests, less when tasks finish early.
A run whose history outgrows 32768 tokens ends as `server_error`; a larger `max_context`, if it fits, lets more runs
finish.
