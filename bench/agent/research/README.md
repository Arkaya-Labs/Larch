# Research-agent bench (offline)

An agent answers each task with two tools, `search` (BM25 over a frozen `docs.jsonl`, SQLite FTS5) and `fetch`
(one document by id, first 12000 characters), against any OpenAI-compatible `/v1/chat/completions` server. Every
download lands in `data/<set>/` (git-ignored); a run needs no network.

`run.py` is its own agent loop over `larch serve`'s OpenAI endpoint, as IR-01 asked: a driver under `bench/` with no
engine change. It measures the model behind the endpoint plus this loop, not the `larch-agent` crate. With `--agent` it
runs larch's own `research` skill instead (below), which is what users get.

| Set | What | Size | Licence | Source, pinned |
|---|---|---|---|---|
| `browsecomp-plus` | hard multi-hop questions over a fixed web corpus | 830 queries, 100195 docs | MIT | `Tevatron/browsecomp-plus` revision `144cff8e35b5eaef7e526346aa60774a9deb941f`; corpus `Tevatron/browsecomp-plus-corpus` revision `b27b02bc3e45511b8b82a13e6f90ce761df726f6`, both through `https://datasets-server.huggingface.co/rows?dataset=<id>&config=default&split=<test\|train>&offset=<o>&length=<n>`. Queries are XOR-encrypted with the BrowseComp canary and decrypted only into `data/` |
| `frames` | multi-hop questions over the Wikipedia pages each one links | 824 | Apache-2.0 (Wikipedia text CC BY-SA) | `https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv`, sha256 `4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff` |
| `simpleqa` | short factual questions; the first 200 rows in file order with an English Wikipedia URL | 200 of 4326 | MIT (Wikipedia text CC BY-SA) | `https://huggingface.co/datasets/basicv8vc/SimpleQA/resolve/e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94/simple_qa_test_set.csv` (mirror of openai/simple-evals), sha256 `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032` |
| `iris` | our own: 40 asks from building a local inference engine for Windows/RTX GPUs, 8 each of `competitor`, `survey`, `changes`, `compare`, `howto`, graded by rubric | 40 tasks, ~95 sources | tasks: this repo's; each source keeps its own | `iris.jsonl` (`as_of` 2026-10-25); the docs are its `sources` URLs, snapshotted as fetched; a GitHub releases list becomes one doc per release |

Wikipedia pages (FRAMES, SimpleQA) are the revisions current at `2024-11-01T00:00:00Z`, read through the MediaWiki API.
The FRAMES and SimpleQA corpora are the gold Wikipedia pages of the fetched tasks (oracle documents): they measure reading
and multi-hop reasoning, not retrieval at web scale. The corpus holds every page the set's fetches have pulled (a later
`--limit` fetch never removes pages), so compare runs by `docs_sha256`.

The BrowseComp-Plus pin is checked at fetch time only: the rows API serves the dataset's current revision, so `fetch.py`
refuses once upstream moves (update the pin and the row counts deliberately). The report's `tasks_sha256` and
`docs_sha256` keep runs comparable.

## Fetch

```
python3 bench/agent/research/fetch.py browsecomp-plus   # corpus ~3.3 GB of JSON; queries ~5 GB via the rows API (each row embeds its docs)
python3 bench/agent/research/fetch.py frames            # test.tsv + about 2475 Wikipedia pages
python3 bench/agent/research/fetch.py simpleqa          # the CSV + about 219 Wikipedia pages
python3 bench/agent/research/fetch.py iris              # every sources URL of iris.jsonl
```

`--limit N` takes the first N tasks (browsecomp-plus: and N*100 docs) to try it out. For browsecomp-plus the N*100
corpus rows are the first rows of the corpus, unrelated to the queries: a limited fetch is a download smoke test whose
corpus does not hold the queries' documents, never to be scored (it prints a warning to stderr). An interrupted download resumes
from the whole lines already in `<file>.part`. Every request carries `User-Agent: larch-research-evals/1`. Iris
sources change over time: a run is graded against the snapshot in `data/iris/docs.jsonl`, and every iris doc there carries
`fetched_at` (UTC) and the `sha256` of the body it came from. A response over 1 GiB, or a redirect out of https, fails the fetch.

## Run

```
python3 bench/agent/research/run.py --set browsecomp-plus --url <server> --model <name> [--runs 4] [--limit N] --grader judge --judge-url <server> --judge-model <name> --out <report>.json
python3 bench/agent/research/run.py --set frames   --url <server> --model <name> --grader judge --judge-url <server> --judge-model <name> --out <report>.json
python3 bench/agent/research/run.py --set simpleqa --url <server> --model <name> --grader judge --judge-url <server> --judge-model <name> --out <report>.json
python3 bench/agent/research/run.py --set iris     --url <server> --model <name> --judge-url <server> --judge-model <name> --out <report>.json
```

`--grader` is `contains` (the default, for quick local checks: the gold answer as whole words in the answer line, after lowercasing and dropping
accents, punctuation and articles; a number stays whole, `1,000` is `1000`), `exact` (also for quick local checks), or `judge` (a model compares the answer with the gold); the GPU numbers use the judge for every set. A task's own `"grader"` key wins over `--grader` for an answer task (`fixtures/` uses it to grade `fake-exact` by `exact`), and the row records the grader used (`rubric` for a rubric task). Rubric tasks
(iris) always need `--judge-url/--judge-model` (exit 2 without them); a task passes only when the judge says yes to
every point. The judge may be the same server, but use a judge model other than the one under test when one is available, and
spot-check a sample of judge verdicts by hand before quoting a score; the fake judge only tests the plumbing. Its prompts are pinned by `judge_prompt_sha256` in the report. A run
that hits a server error, a judge without a verdict, 24 turns without an answer, or 1800 s of wall time fails with its `error` and the
bench goes on; the exit code is 1 when every run ended with an error, else 0. The report is rewritten after every task. `--out` must not exist yet (exit 2), so a rerun after a crash needs a new name.

`pass^4` needs sampling: at temperature 0 with no seed the four runs of a task would be one run four times. The agent
model therefore runs at `--temperature` (default 0.7) and run `n` sends `seed = n` (0 to runs-1); the judge stays at temperature 0.

## larch's own research skill (`--agent`)

```
python3 bench/agent/research/run.py --set iris --agent target/release/larch-agent --model m2 --agent-data <data folder> --runs 1 --judge-url <server> --judge-model <name> --out bench/results/agent-research-m2-iris-skill.json
```

Each run is `larch-agent run research -m <model> --data-dir <data folder> --json --effort <effort> --question <task>` on
the live web, at the task row's `effort` (`standard`, the default, or `deep`), with the search keys and sampling of that
folder's `config.toml`, so `temperature` and `seeds` are null in the report and the
corpus is not read. Only the iris set: the report is graded by its rubric. A row adds `stopped` (why the research
stopped), its `effort`, `unsourced_fixed` and `unsourced_removed` (the report's verification line: sentences with a
number or a date and no source that their writer's retry fixed, and that were cut; the summary's `unsourced` adds them
up) and `checks`, the hard gates, which exit 1 when any is broken: `invented_urls` (a source whose URL no fetch step
asked for), `quotes_missing` (a source whose quote its saved page under `<data folder>/agent/web/` lacks) and
`dangling_markers` (a `[n]` in a section that names no source). `quotes_found / quotes_total` counts the sources whose
saved page holds the quote. `tool_calls` is the searches, `pages` the pages fetched and the tokens every model turn's,
from the run's audit steps.

`--live` (with `--agent`) scores facts right in a bounded time: a run whose report takes longer than `--budget`
seconds (300 by default) does not pass, whatever it read. Each row adds `in_time` and `facts_kept` (the report's
sources whose quotes were found again, `quotes_found`); the summary adds `facts_kept` and `in_time`; and the run
prints each run's wall clock, pages read and facts kept. Pages read is reported, never scored. A row's `wall_s` leaves
out the model's load, which the agent logs as its own `load` step and the row keeps as `load_s`.

## Research briefs (`briefs.jsonl`)

Rows for larch's own research skill on saved pages: a request the agent must not guess at silently. Each row has
`request`, `pages` (a folder of `hits.json` and `1.json`, `2.json`, ... as the replay set saves them; these pages are
written for the set, `synthetic` in `hits.json`), `brief` (regexes the run's brief must each match, case aside: the
generation, country or model year it assumed), `report_not` (regexes the report must not match: the off-subject
product), optionally `brief_not` (regexes the brief must not match: an `About:` list for a general topic), `then` (the
user stops the run at its brief and says this; the chat calls research again with it, which the scorer runs as
`request (then)`) and `off_subject` (pages the subject lock must drop unread). `cargo test -p larch-agent --test
research` checks every row parses, its pages load, and runs `lexus-isf-m3` through the skill on a scripted model (the
lock, the query that gets the IS F's name, the report's note). The brief's wording is the real model's, so score the
rows on the Mac with `larch-agent run research-briefs --model <gguf>` (`--briefs <file>` for another file): each row
runs the standard research skill on its saved pages, served on this machine, and prints `pass <id>` or `FAIL <id>:`
with each regex missed or matched; the exit code is 1 when a row fails.

## Answer protocol

The system prompt asks the agent to search, read and cross-check, to support the answer with quotes copied exactly
from the documents, each in double quotes followed at once by `[doc:<docid>]`, and to end with one line
`Answer: <short answer>`. Graders read the first non-empty line after the last `Answer:` or `Final answer:` (markdown `*`, `_`, `#`, `>` around the marker are
ignored); a reply without one is graded on its last non-empty line, never on the whole reply.

## Report (`larch-research/1`)

`schema, set, model, url, agent, grader, judge_model, judge_prompt_sha256, runs, temperature, seeds ("run index"),
tasks_sha256, docs_sha256, system_prompt_sha256, started_at, summary, rows`; the three hashes are of the tasks file's bytes,
the docs file's bytes and the system prompt, so a report says what it scored. A row is
`{task, run, grader, answer, passed, points, error, tool_calls, pages, wall_s, prompt_tokens, completion_tokens,
quotes_found, quotes_total}`. The summary:

| Field | How |
|---|---|
| `pass@1` | mean over tasks of passes / runs |
| `pass^4` | mean over tasks of C(c,4)/C(n,4) for c passes in n runs; with 4 runs, the share of tasks right 4 runs in a row. null if a task has fewer than 4 runs |
| `citation_accuracy` | cited spans (3+ words in double quotes on one line, followed at once by `[doc:<docid>]`) whose cited document's full text contains them (case and whitespace folded) / all cited spans; quotes without a citation are not counted |
| `tool_calls` | every tool call, malformed ones included: total and mean per run |
| `pages` | distinct documents fetched per run: total and mean |
| `wall_s` | seconds per run: total and mean |
| `tokens` | `prompt` and `completion` totals from the server's `usage` |
| `errors` | runs that ended with an error |
| `points_share` | mean over rubric runs of the share of rubric points judged yes: partial credit, which shows a gain all-or-nothing passes hide; null without rubric runs |
| `checks` | `--agent` only: `invented_urls`, `quotes_missing` and `dangling_markers` over every run; each must be 0 |
| `stopped` | `--agent` only: runs by why the research stopped |
| `unsourced` | `--agent` only: `fixed` and `removed`, the rows' `unsourced_fixed` and `unsourced_removed` added up |

## The --fake gate

`python3 bench/agent/research/run.py --fake` runs `fixtures/` (two answer tasks, one graded `exact` by its own key and one by `--grader`, contains by default, and one rubric task) 4 times against a
scripted model and judge and exits 1 unless every run passes with full citation accuracy, 2 tool calls and 1 page.
CI runs it, and `test_research.py`, under `cargo test` (`crates/larch-agent/tests/bench.rs`).

## GPU check (RTX 5070 Ti): NOT RUN

First, on a machine with network access, take the Iris snapshot and check it. Every doc needs real text, and the NVIDIA
and Wikipedia pages must hold the VRAM, bandwidth, power and launch-price figures that iris-025 to iris-032 ask for:

```
python3 bench/agent/research/fetch.py iris
python3 -c "import json; [print(d['docid'], len(d['text']), 'SHORT' if len(d['text']) < 500 else '') for d in map(json.loads, open('bench/agent/research/data/iris/docs.jsonl', encoding='utf-8'))]"
python3 -c "import json; [print(d['docid'], 'GB/s' in d['text'], '$' in d['text']) for d in map(json.loads, open('bench/agent/research/data/iris/docs.jsonl', encoding='utf-8')) if d['docid'].startswith(('www-nvidia-com', 'en-wikipedia-org'))]"
```

Then serve m5 and run the sets. Every set is graded by the judge. BrowseComp-Plus reports hold decrypted answers, so its
command writes to `bench/agent/research/data/` (git-ignored); commit only a summary of it (e.g. as
`bench/results/agent-research-m5-bcp-summary.json`). The other sets write to `bench/results/`. BrowseComp-Plus and FRAMES take `--limit 100`: a run is up to 24 model turns, so 4 runs
of all 830 or 824 tasks is too long for one GPU session. The first 100 tasks in file order are a fixed slice that
reruns compare against. SimpleQA (200) and Iris (40) run whole.

```
larch serve -m <models>/m5.gguf --set max_context=32768
python3 bench/agent/research/run.py --set browsecomp-plus --url http://127.0.0.1:8080 --model m5 --runs 4 --limit 100 --grader judge --judge-url http://127.0.0.1:8080 --judge-model m5 --out bench/agent/research/data/agent-research-m5-bcp.json
python3 bench/agent/research/run.py --set frames   --url http://127.0.0.1:8080 --model m5 --runs 4 --limit 100 --grader judge --judge-url http://127.0.0.1:8080 --judge-model m5 --out bench/results/agent-research-m5-frames.json
python3 bench/agent/research/run.py --set simpleqa --url http://127.0.0.1:8080 --model m5 --runs 4 --grader judge --judge-url http://127.0.0.1:8080 --judge-model m5 --out bench/results/agent-research-m5-simpleqa.json
python3 bench/agent/research/run.py --set iris     --url http://127.0.0.1:8080 --model m5 --runs 4 --judge-url http://127.0.0.1:8080 --judge-model m5 --out bench/results/agent-research-m5-iris.json
```
