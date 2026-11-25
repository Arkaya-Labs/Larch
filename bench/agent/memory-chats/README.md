# Memory chats: does the agent remember what the user says?

Plan PR 2 (`larch-memory/plan.md`). A bench, not a product change. It measures **capture**, which
the older memory set (`bench/agent/memory`) does not: that set puts the hits in the prompt itself.

```
larch-agent bench --memory-chats [chats.jsonl] -m <gguf> --machine <m> --json <out>.json [--app]
```

## What a run does

For each case in `chats.jsonl`, in its own throwaway data folder:

1. Saves the case's `memory` facts, as the `remember` tool saves them (a store that is not empty).
2. Runs each chat in `chats` as a new chat, through the real chat turns (`Agent::turn_with`), each
   turn to its end. Nothing is injected. The only help is the y/n answers a user would give: yes
   to `remember` (which asks only after an untrusted result, such as a web answer), no to every
   other write or research plan. Both are counted.
3. Reads `memory::list()` after every turn, so each save is tied to the message whose turn made it.
4. Forgets every fact that holds a `forget` word, as the app's Forget button does.
5. Asks each probe in a new chat.

`--app` opens every chat with the desktop app's options (`AgentOptions::app`): the home folder
allowed, no prefix cache anchor. The report's `app` lists where these differ from the bench's
options. The chats read the data folder's config.toml, so web tools search with its keys.

## Data

`chats.jsonl` has one case per line. It holds 40 cases: the 24 #436 fixtures (`from`) after a
review, plus 16 new ones. The new ones are multi-turn chats, facts mixed into task messages, a
store that already holds facts, and negative cases (a pasted email, an API key, an injected
instruction, plain questions, "I don't remember where I parked"). Counts: 51 gold facts, 27
negative messages and 36 probes. This is short of the plan's data
(about 120 gold facts and 60 negative messages); add cases before PR 4's gate.

| Key | Meaning |
|---|---|
| `id` | `mc-<nn>-<name>` |
| `memory` | facts saved before the first chat |
| `chats` | the chats in order, each a list of `{"user", "no_save", "explicit"}`. `explicit` marks a message that asks for the save |
| `facts` | gold facts: `details` a saved fact must hold. The first detail is the key that finds the fact. `explicit` means the user asked for it to be saved. `stale` lists the details a correction replaced |

A gold fact is saved only through a fact that a turn of the case saved or changed: a seeded fact,
or one an earlier message saved and a correction left as it was, never counts. A gold fact is saved
only through a fact that holds its key and one more of its own details, so a fact that names the
key alone (the dog's name) does not capture each fact about it; a correction (`stale`) thereby
needs what it changed. The gold facts,
`keep`, `never` and secrets are scored on the facts after the chats, before the forgets. A fact's
text is scored without a `Said: "..."` quote line (plan 2.1), as the quote holds every detail.
| `keep` | detail sets some fact must still hold at the end: a seeded fact that a wrong merge would overwrite |
| `never` | detail sets no fact may hold at the end: a hypothetical, a quote, "don't save" |
| `forget` | after the chats, every fact holding one of these words is forgotten |
| `probes` | `{"user", "answer", "not"}`: the reply must hold every `answer` detail and none of `not`. With neither, a human checks the reply |
| `off` | the chats run with memory off, and it is on again for the probes (plan PR 5) |
| `skip` | why the case cannot run yet |

A detail is a word or phrase. It is matched on word boundaries, ignoring case. A trailing `*`
lets the word run on (`designer*`), and `|` separates alternatives (`Lucía|Lucia`).

### Review of the #436 fixtures

- Each `entity/relation/value` triple became a list of details. The plan drops splitting facts
  into pieces, so a fact is scored by what the saved sentence holds.
- 16 (memory off) runs its chat with memory off (`off`): nothing may be saved, and the probe, with
  memory on again, must not know the city.
- 15 (deletion): the same-chat probe is dropped, because that chat's history still holds the fact.
  The new-chat probe stays.
- 11 (private medical) and 22 (hard week) are negative cases. The user asked for a message, not
  for a record of their health or mood.
- 17 (interruption): the bench cannot cancel a turn, so it tests a constraint inside a task.
- 18: the probe asks when the office moves, not where it is today, so the answer does not depend
  on the run date.
- Rubric text and `must_not_use` became `never`, `keep`, `stale` and the probes' `not` lists.
- Probes whose right answer may name what to avoid ("no chicken, since Biscuit is allergic") are
  human checks: mc-06, mc-26's treat, mc-40. mc-14 scores the budget only.
- mc-28: the email is from a coworker's neighbour, so no fact in it is about the user's people.

## Scores (plan section 4)

`report.memory_chats.summary` counts every case that ran. A failed turn is recorded, the case
goes on, and its facts count as they are; `errors` counts the cases with one. Each case row
keeps every turn (its saves, tools, reply, TTFT and tokens), the facts at the end, and each gold
fact's `saved`, `missing` and `stale`.

| Measure | Field |
|---|---|
| Remembers on its own | `remembered_share` = `unasked_saved / unasked_facts` |
| Explicit saves | `explicit_saved / explicit_facts` |
| `remember` calls | `remember_explicit` (messages that ask), `remember_unasked` (other fact messages), `remember_negative` (negatives and probes); `approvals` counts the y/n asked after an untrusted result |
| Whole facts | `whole_share` = saved gold facts with every detail, over those saved |
| No false saves | `false_save_share` = chat messages with nothing to keep whose turn saved a fact, over those messages. Probe saves are counted apart (`probe_saves / probe_turns`). Also `never_held` and `secrets` |
| Updates | `corrected / corrections`, and `merges` (an update that took a gold or `keep` key out of a fact, plus each `keep` set lost) |
| Used later | `used_share` = probes answered right, over probes scored. `probe_checks` are left for a human |
| First-token time | `later_ttft_ms` (p50, p95) over every turn after a chat's first |
| Cache hits | `later_cached_tokens`, over the same turns |

Each probe is scored three ways, so a miss shows which step failed:
- `captured`: every gold fact of the case was saved.
- `retrieved`: the new chat's memory block holds the answer, or the turn called `recall`.
- `right`: the reply uses the fact (use).

Yield time, web speed and capture cost come from PR 4's audit line: `kind` "step", `step`
"memory capture", with `prompt_tokens`, `generated_tokens`, `ttft_ms` and `wall_ms`, and in
`detail` the chat ID, `messages` read, `shown` facts, `cancelled`, `yield_ms` (from the next
turn's cancel to the capture's end), `retry` (a message read again alone after a cut), `saved`
(how many facts it saved: a count, not slugs) and `dropped` (one reason per item). This
bench does not sum them yet; read them from `-audit.jsonl`.

**Background capture (PR 4).** `--memory-chats` runs with capture on, as the app does (the other
benches run without it, unless `--capture` turns it on for `--turn` or `--everyday`: with
`--everyday --live` that measures what a capture costs the web turns after it: each next turn
cancels the capture still running, so it is the cost of a cancelled capture, not a whole one).
Turn it off with `memory_capture = false` in the data folder's
config.toml, which the bench copies. A save is tied to the turn after which `memory::list()`
first shows it. The runner calls `Agent::wait_for_capture()` after every turn, which waits for the
chat's capture to save, before it reads the facts, and so before each probe. Because it waits,
`later_ttft_ms` here holds no contention; a second pass that does not wait is still to come.

## Mac run (baseline, today's code)

Who runs it: a thread with Remote Control in a larch checkout on the Mac (24 GB). One run at
a time. Stop the Actions runner first (let a running macOS job finish), and restart it at the end
by trap. Close other heavy apps. The run uses both models from plan section 4:
- `$M35`: Qwen3.5 35B-A3B Q3_K_M, the demo model
- `$M8`: Qwen3 8B Q4_K_M, the default model (`m2`)

Run it twice per model from PR 4 on: with capture on (the default), then with
`memory_capture = false` added to `$D/agent/config.toml`, to compare. The bench reads
`$D/agent/config.toml` and ignores a missing file without a word, so check it is there.

### 0. Build the branch
```
cd ~/larch && git fetch origin agent/memory-auto && git checkout --detach FETCH_HEAD
LARCH_BACKEND=METAL cargo build --release -p larch-agent; echo "exit=$?"
A=target/release/larch-agent; M35=<35B-A3B Q3_K_M gguf>; M8=<Qwen3-8B-Q4_K_M gguf>
D=$(mktemp -d)   # a scratch data dir, so no fact reaches the real one
mkdir -p "$D/agent" && cp "<real data dir>/agent/config.toml" "$D/agent/" && test -s "$D/agent/config.toml" || { echo "no config.toml: the web turns need its search keys"; exit 1; }
mkdir -p ~/memory; shasum -a 256 "$M35" "$M8" > ~/memory/gguf-sha256.txt; git rev-parse HEAD > ~/memory/commit.txt
```

### 1. The baseline, both models, as the app runs them
```
$A bench --memory-chats --app -m "$M35" --machine m5 --data-dir "$D" --json bench/results/agent-memory-chats-m5-$(basename "$M35" .gguf).json
$A bench --memory-chats --app -m "$M8" --machine m5 --data-dir "$D" --json bench/results/agent-memory-chats-m5-$(basename "$M8" .gguf).json
```
Each report has `-audit.jsonl` (every case's chats, case after case) and, when a turn searched,
`-search.jsonl` beside it. A turn error makes the run exit 1; the case still runs to its end and
is scored.

### 2. Read and send back
- The summary table the run prints at its end, for each model.
- Per case: the line for each case with a save, a false save, a merge or an error, and the facts
  in its `facts`.
- The probe replies left for a human check (`right: null`), with the case ID.
- `remember_unasked`: how often the model called `remember` on its own; also
  `remember_explicit`, `remember_negative` and `approvals`.
- The GGUF SHA-256s, the commit, and the config.toml keys that differ from the defaults.

Gates for this run: no case errors, and every report written. The quality targets in plan section
4 apply from PR 4 on, when this bench runs again with capture on and off.
