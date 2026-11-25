# Document follow-ups (research v2, plan 10.2 and 12)

64 follow-ups over 17 documents (`docs.json`, `questions.jsonl`, edited copies in `docs/`) in the kinds of
`test-cases.md`, each with its chat turns, gold passages copied from the text with their pages, the outcome the
corrected rules expect (plan 10.2 item 6) and a list's items. `panel.json` is the fixed panel (seed 20261113, 20 ids,
run.py's `pick`); `--set sample --seed S` draws a fresh one and records S.

## On the Mac: the large model at `--context 32768`, the small one at `--context 8192`

```
python3 bench/agent/docfollow/run.py fetch            # into data/cache/ (git-ignored); put the FY2025 10-K PDF there as nvda10k.pdf
python3 bench/agent/docfollow/score.py gold --fill    # fills the anchored gold of lm317, nvda10k, nist600: read what it wrote
python3 bench/agent/docfollow/score.py gold           # must print "N gold passages on their pages"
python3 bench/agent/docfollow/run.py run --agent target/release/larch-agent --plain target/release/larch \
  -m <model.gguf> --context 32768 --thinking-budget 512 --out ~/larch-research-v2-results/docfollow-large
python3 bench/agent/docfollow/score.py pack <out>/results.json    # a judge reads judge.md and the pack, writes verdicts.jsonl
python3 bench/agent/docfollow/score.py score <out>/results.json --verdicts verdicts.jsonl
```

`--source link` pastes URLs instead of attaching files; `--modes`, `--orders`, `--reps`, `--turn-timeout` and
`--agent-config` (a config.toml each data folder starts from) shape a run. Every session has its own data folder under
`<out>/runs/`. Whole and gold read PDFs with `pdftotext` (poppler), else PDFKit. `larch-agent chat` gets `--context` as
`--max-context` (without it, 8192, as the app). `larch chat` has no thinking budget (`--max-tokens` caps thinking and
answer together): `results.json` records what each CLI ran at, and a flag is passed only when a CLI's `--help` lists it.
`run.py dry-run` runs everything on fake CLIs and exits 1 on any failure.

Modes. `today`: `larch-agent chat` as it is. `v2`: with `doc_followups = "on"`. `whole`: the plain model with the whole
document as its system message, where it fits (the second floor). `gold`: the plain model with only the gold passages
(plan 10.2 item 4). `cold` reopens a saved copy of the chat's summary for each question, so the engine holds nothing;
`warm` asks a document's questions in one live chat; `interleaved` runs the four-paper chat switching paper each time.
A turn past `--turn-timeout` is a miss with that reason, and its session restarts for the questions after it.

Scores, kept apart. Extraction: the gold passage is in the agent's extracted text (`<data>/agent/extract/`, else the local text).
Retrieval: a range in the reader's trace covers the gold passage. Citation support: matched over cited quotes in the
trace. Completeness: item recall on lists. Correctness: the judge's verdicts; no answer counts as wrong. The gold check
names each v2 miss that gold mode got right (our flaw). Times are send to first visible answer word and total, p50 and
p95, with the share answered and the tools called per question.
