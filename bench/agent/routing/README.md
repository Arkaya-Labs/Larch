# Routing rows (v3 D8)

`dev.jsonl` holds rows from real chat misses, each scored on the reply to its last turn after a planted history. An id is
`routing-<family>-<label>-<n>`. The label says what the row takes: `web` (web_answer), `research`, `read` (read_papers),
`list` (list_dir), `none` (no tool) or `refuse` (the user's own fact the files lack: a refusal with no number or date).
Every family but `absent` has a mirror row that expects the opposite.

`multistep.jsonl` (v3 D11, written by `../gen_multistep.py`) holds 30 turns of several calls on the same tools, where the last
call takes what an earlier result found: a file found by listing its folder or in memory, a near name after a path that does
not exist, a fact written to another file, and a skill call (web_answer, research, read_papers) on the subject or papers found.
Each step is scored, and so are the skill call's arguments, since the skill ends the app's turn. The find and fix rows'
listing step takes `find_files` on the folder as well as `list_dir`, and a fix row's is optional, as the not-found error
already lists the folder (a skipped step counts as made in `steps`). These rows run on the bare
system prompt, as `../multistep.jsonl`'s do: `--app-prompt` lists two folders whose files their scripted listings contradict.

```
larch-agent bench -m <model> --turn --prompts bench/agent/routing/dev.jsonl --machine <name> --json <report>.json
larch-agent bench -m <model> --multistep bench/agent/routing/multistep.jsonl --machine <name> --json <report>.json
```

`--turn` runs each row as a chat's turn after its planted history, as the app runs it: the agent's own prompt, the
turn note on each message (the planted messages get the notes they would have had, so a kept limit holds), the calls
plain words force and the checks that take a reply back (v3 D2 to D6), with the router off and the home folder
allowed. The turn stops at its first call. Only the reads it runs before showing the call run: `list_dir`,
`read_file`, `find_files` and `search_files` on the home folder, `recall` on a throwaway data folder.
`counters.withdrawn` counts replies a check took back. The agent's own retry runs inside the turn, so
`unknown_tools`, `schema_fails`, `retries` and `parse_rate_before` are not measured. The prompt lists the home
folder's names, so compare `--turn` reports from the same machine. With `--app-prompt` in place of `--turn` the rows
measure the model and the call grammar alone.

Reading the report: `by_class` counts each `<family>-<label>`; a `none` class's `wrong_tool` rows searched when they
should not, a `web` class's did not search when they should. `counters.offline_claims` counts answers that say they
cannot reach the web, and each row's `cached_tokens` is what the prefix cache reused of its prompt. The call grammar is
on, as in the app (`--no-tool-grammar` turns it off; the report's `tool_grammar` says which), so
`counters.unknown_tools` (a call to a tool not in the list) and `counters.schema_fails` (a call that fails its tool's
schema) should be rare, not 0: the grammar does not hold bounds (`minimum`, `minLength`) or a Llama 3 call written
without `<|python_tag|>`. With it off they count the model's own mistakes (v3 D11).
`multistep.summary.right_rate` is the share of multi-step turns with every step right; each row's `steps` says which step
went wrong, and `answered` whether the last step (the answer or the skill call) did.

skipped: the "must ask back" label, the 3 wordings and seeded sampling of D8; add them with the wave gate's runs.
