# Public tool-calling benchmarks (v3 D10)

Three public benchmarks, each run against `larch serve`'s OpenAI endpoint on the shipping model and quant
(Qwen3-30B-A3B Q3_K_M on the M5). Our own rows catch our own regressions; these catch tool-call mistakes we have not
seen yet, and give numbers an outsider can check.

| Bench | What it checks | Items | Needs |
|---|---|---|---|
| `bfcl` | BFCL's offline categories, scored by BFCL's own checker with no judge: the tool name and arguments, no call when no tool fits (irrelevance), and multi-turn tasks | 238, spread evenly over 15 categories (`BFCL_SUBSET` in `run.py`) | `bfcl-eval==2026.3.23` (Apache 2.0) in a Python 3.12 environment (its pins do not install on 3.14) |
| `when2call` | the call decision: a fitting tool is called (`tool_call`), no call when no tool fits (`cannot_answer`) or an argument is missing (`request_for_info`) | 300, When2Call's judged test set (`main`, pinned by sha256) | the standard library |
| `tau2` | tau2-bench Retail: customer-service tasks over several turns with tools and a simulated customer | 114 tasks | a tau2-bench checkout at `v1.0.1` (MIT; Python 3.12, `uv`) and a paid model that plays the customer |

## Run

```
larch serve -m <Qwen3-30B-A3B-Q3_K_M.gguf> --name qwen3moe --tool-grammar &
python3 bench/agent/public/run.py when2call --url http://127.0.0.1:8080/v1 --model qwen3moe --runs 3 --out w2c-base.json

uv venv --python 3.12 ~/.venvs/bfcl && uv pip install --python ~/.venvs/bfcl bfcl-eval==2026.3.23
~/.venvs/bfcl/bin/python bench/agent/public/run.py bfcl --url http://127.0.0.1:8080/v1 --model qwen3moe --runs 3 --out bfcl-base.json

git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench ~/tau2-bench && (cd ~/tau2-bench && uv sync)
OPENAI_API_KEY=<key> python3 bench/agent/public/run.py tau2 --url http://127.0.0.1:8080/v1 --model qwen3moe \
  --tau2-dir ~/tau2-bench --user-llm gpt-4.1 --runs 3 --out tau2-base.json

python3 bench/agent/public/run.py gate w2c-base.json w2c-new.json
```

`--limit N` takes the first N items (BFCL: per category) to try a setup. `--out` must not exist yet. BFCL keeps each
run's files next to the report (`<out>-bfcl-<run>/`, which must not exist yet either) and runs each in a process of
its own. tau2 keeps its own in the checkout, under a new name each run (`data/simulations/larch-<out>-<time>-<run>/`,
listed in the report's `tau2.saved`).

Reading a report: `<bench>.runs` holds each run's scores, `reference` the published numbers to read them against.
`when2call` scores `accuracy` (all items), one accuracy per label, `false_calls` (calls where none fits) and `errors`
(failed requests, counted as wrong); `when2call.rows` holds each run's rows. `bfcl` scores each category and BFCL's own
`simple`, `non_live_ast` and `multi_turn` means, built as BFCL's leaderboard builds them. `tau2` scores `pass^1` as tau2
takes it (a simulation that ended in an error left out), `pass^1_errors_fail` (such a simulation counted as a fail) and
`errors`.

The gate fails a score whose mean drops below the baseline's mean by more than the baseline's run-to-run spread, so
the baseline runs 3 times (`--runs 3`). It also fails when the report lacks a bench the baseline has, or was run
another way (temperature, `--limit`, BFCL's ids and version, When2Call's sha256, tau2's tag and customer model). Runs
sample at `--temperature` (default 0.7, Qwen's recommended setting; BFCL's leaderboard samples at 0.001). Run `n` of
When2Call and of tau2's agent sends `seed = n`; BFCL's handler sends no seed, so its runs differ by sampling alone. The
customer model in tau2 must stay the same across runs.

Limits: these are subsets on another harness, so a score near the published one says little; a large gap points at our
template, parser or settings. `--tool-grammar` (#343) holds Qwen3's JSON calls to their tool's schema, as the app does;
without it `larch serve` holds only a forced call (`tool_choice`) and XML-body calls, so the scores show the model alone.
`larch serve` renders the chat template's default, so the Qwen3-30B-A3B GGUF thinks and the 2507 Instruct does not.
When2Call's paper scores asks-back against cannot-answer with a judge; this driver scores the call decision only.

skipped: When2Call's judged F1 (asks back versus cannot), add when a fixed judge model is chosen.
