# Benchmark inputs and the llama.cpp baseline (architecture 16)

| Path | What |
|---|---|
| `models.toml` | alias → Hugging Face repo, file, size, sha256 (the LFS hash, checked against the bytes). Models live outside git as `<dir>/<alias>.gguf` |
| `prompts/*.jsonl` | the frozen workloads, raw text, written by `prompts/gen.py` (`--check` proves the files are its output). Never edited after day 1 |
| `manifest/<alias>.jsonl` | per-model evidence: every turn of every workload tokenised by that GGUF (`baseline.py manifest`) |
| `results/<machine>-<alias>-<row>.json` | baseline runs (`baseline.py run`); `mac-*` and `sm86-*` come from the Mac and RTX 3050 machines |
| `results/history.csv` | the scoreboard: one line per merge to `dev` (`scripts/bench-merge`), never edited |

Results dated before the public repository started were measured during private development. Commit
hashes, branch names and `#` numbers in them refer to that history, not to this repository's commits,
issues or pull requests.

## Workloads

| Row | File | Run |
|---|---|---|
| W1 chat | `w1-chat.jsonl`: 1 conversation, 12 turns, user 40-300 tokens, answers 100-400 | c=1, slot reuse, 8192 per sequence |
| W2 chats | `w2-chats.jsonl`: 8 conversations x 8 turns, W1 lengths | c=4 (first 4) and c=8, slot reuse, 8192 per sequence |
| W3 agent | `w3-agent.jsonl`: 3 loops; 4096-token system prefix, 20 x (130-token tool result as a user message, 120-token answer) | c=1 (first loop) and c=3, slot reuse, 16384 per sequence |
| W4 batch | `chat.jsonl`: 200 single-turn prompts, 50-2000 tokens rendered, answers 128-512 | c=1, 4, 8, 16, no reuse, longest prompt + output per sequence |
| W5 longprompt | `w5-longprompt.jsonl`: 2048- and 4096-token user messages, 128-token answers | c=1, no reuse |
| W6 fixed | none: `-npp 512 -ntg 128 -npl 1,4,8,16 -c 16384` | `llama-batched-bench` and the driver's mirror |

One line per conversation: `{"id", "messages": [{"role": "system"|"user"|"assistant", "content", "max_tokens"}]}`. Every
assistant message is a timed turn: its prompt is all messages before it, rendered with the generation prompt, and the
answer is exactly `max_tokens` long (greedy, EOS ignored). Its `content` is the scripted answer the history continues
with (`--replay scripted`), so each turn's prompt is the same on every engine; the last assistant message has no
content. Message lengths are exact on the vocabularies of m1 to m5 (`user_tokens` and `scripted_tokens` in the
manifests equal the targets); rendered lengths add each template's overhead and are in the manifests.

## Rules every engine follows

- **Render:** the GGUF's own Jinja template (llama.cpp's `common_chat_templates_apply`, `use_jinja`), generation prompt
  on, `date_string = "26 Jul 2024"` and, for `strftime_now`, 2024-07-26 12:00 UTC. Tokenise the whole rendered string
  with `add_special` and `parse_special` (llama-server's chat path), every turn, never incrementally.
- **Prompt hash:** `prompt_sha256` = sha256 of the prompt token IDs as consecutive little-endian int32.
- **Sampling:** greedy, end-of-generation tokens excluded (llama-server's `ignore_eos`), exactly `max_tokens` sampled.
- **Clients:** N closed-loop clients take conversations in file order and run each one's turns back to back; a client
  submits its next turn the moment the previous answer is done.
- **Reuse (W1-W3 only):** llama-server's slot cache. One slot per client; a request takes the free slot whose tokens
  cover the largest fraction of its prompt (> 0.10, first best wins), else the least recently used; it keeps the
  longest common token prefix with the slot (the previous prompt plus the fed answer tokens, the last sampled token
  never fed) and always evaluates at least one token. W4-W6 reuse nothing.
- **Times:** `t_submit` when the client submits, `t_first` when the first token is sampled, `t_done` when the last one
  is. The driver tokenises every prompt before the clock starts, so llama.cpp's TTFT excludes tokenisation.

## Result schema `larch-bench/1` (one JSON object; `larch bench --json` writes the same)

```
schema, engine ("llama.cpp" | "larch" | "ollama" | "openai"), driver, mode ("run" | "fixed"), llama_cpp {tag, commit},
system_info, machine, started_at (UTC), model (<alias>.gguf), date_string, server (openai),
params {n_ctx, n_ctx_seq, n_seq_max, n_batch, n_ubatch, n_gpu_layers, flash_attn, type_k, type_v, n_threads,
        n_threads_batch, kv_unified}
run:   workload {file, sha256, conversations}, concurrency, cache_prompt,
       summary {requests, failures, prompt_tokens, cached_tokens, evaluated_tokens, gen_tokens, wall_s, agg_tok_s,
                ttft_ms, tpot_ms, e2e_ms: {p50, p99, mean, max} | null},
       requests [{client, conv, turn, slot, prompt_tokens, cached_tokens, evaluated_tokens, prompt_sha256, max_tokens,
                  gen_tokens, t_submit_ms, t_first_ms, t_done_ms, ttft_ms, tpot_ms, e2e_ms, error}]
fixed: fixed [{pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed}]   (one per shape per run)
```

TTFT = first - submit; TPOT = (done - first) / (generated - 1); e2e = done - submit; percentiles are nearest-rank over
the successful requests; `agg_tok_s` = generated tokens / (last done - first submit). `fixed` feeds pp + tg tokens per
sequence in `llama-batched-bench`'s phases (every prompt, then npl-token decode steps), so all three speeds
compare with its own.

## Running a machine

```
python3 scripts/baseline.py --machine <win5070|wsl5070|win4060|sm86|mac> --models <dir> models
python3 scripts/baseline.py --machine <m> build            # CUDA or Metal per machine; WSL sets CUDAHOSTCXX=g++-14
python3 scripts/baseline.py --machine <m> --models <dir> manifest
python3 scripts/baseline.py --machine <m> --models <dir> flags
python3 scripts/baseline.py --machine <m> --models <dir> run    # exit 1 if any row failed
python3 scripts/baseline.py --machine <m> check
python3 scripts/baseline.py --machine <m> --models <dir> server
python3 scripts/baseline.py --machine <m> --models <dir> ollama
python3 scripts/baseline.py --machine <m> --parallel N ollama-serve
python3 scripts/baseline.py --machine <m> compare <a.json> <b.json>
```

- `run` deletes a row's old result before running it; a failed row leaves `<row>.failed` (its exit code) instead. The
  fixed row alternates `llama-batched-bench` and the driver 9 times in one hold, every run's rows into
  `<machine>-<alias>-batched.jsonl` and `-fixed.json`; a failed run leaves neither, only its `.failed`.
- `check` passes only on complete evidence for every row of the machine: the workload spec against the manifests;
  each run with this machine, model, workload bytes, concurrency, reuse, per-sequence context and searched flags,
  exactly the manifest's requests with their hashes and full answer lengths, no error; llama-bench's four tests on
  this model with flash attention and the searched flags (its local model path is reduced to `<alias>.gguf`); all
  four fixed shapes in both files with 9 runs a side, flash attention on and no shared prompt, the driver's S over its
  runs (tokens over time, the harmonic mean of the runs' S) within 3 % of `llama-batched-bench`'s at each: one run a
  side cannot resolve 3 % on wsl5070, where back-to-back `llama-batched-bench` processes differ by up to 8.4 %.
- `compare` refuses unless each run on its own holds exactly its manifest's requests, hashes and full answer lengths
  with no failure, and the two match in model, workload bytes, concurrency, reuse and per-sequence context; engine
  flags may differ.
- `--machine cpu` builds with every GPU backend off and runs `-ngl 0`: driver checks, never a baseline number.

Ollama (D15): `FROM <gguf>`, `PARAMETER num_ctx 8192 | 16384` as `larch-<alias>-<ctx>`; the server runs with
`OLLAMA_NUM_PARALLEL=N`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_KEEP_ALIVE=-1`. It is driven
later by `larch bench --engine ollama`.

## larch bench (same rules, same schema)

```
larch bench -m <dir>/<alias>.gguf --workload chat|chats|agent|batch|longprompt --concurrency 1,4,8 \
    [--replay scripted|generated] [--duration S] [--cancel-rate F] [--telemetry off|t0|t1] \
    [--set k=v[@S]]... --machine <m> --json <out>.json
larch bench -m <gguf> --mode serve ...                      # W4, chat.jsonl
larch bench -m <gguf> --mode serve --concurrency 1 --slice K/N ... --json <out>-K.json   # part K of N, one GPU hold each
python3 scripts/bench-join <out>.json <out>-0.json ... <out>-<N-1>.json     # the whole row; wall time = sum of the parts
larch bench -m <gguf> --mode batched --pp 512 --tg 128 --npl 1,4,8,16 --json <out>.json   # W6 mirror
larch bench -m <gguf> --engine ollama --url http://127.0.0.1:11434 --workload chat ...    # D15 column
larch bench -m <dir>/<alias>.gguf --engine openai --url http://127.0.0.1:8000 --workload chats --concurrency 8 [--cache-prompt true|false] ...   # one --concurrency per server start when it caches prefixes
```

- **Rows:** as the driver: with reuse (W1-W3) the first N conversations, client k starting on conversation k; W4-W5
  every line. Per-sequence context is `--set max_context` (else 16384 for W3, 8192 otherwise; Ollama's `num_ctx`
  and the only `--set` it takes); for W4 pass the driver's `n_ctx_seq` so `compare` accepts the pair. Several
  `--concurrency` values write `<stem>-c<N>.json` each; a live `--set k=v@S` is never undone, so it takes one
  concurrency per invocation and no fixed shape. Changes due at 0 s are in force before the first request (and in
  `params`); a change the engine does not apply, or the run ends before (`NotReached`), is `applied: false` in
  `knob_timeline` and fails the invocation. Default prompts are the checkout's `bench/prompts`; every argument and
  the workload file are checked before the model loads.
- **larch:** prompts are rendered and tokenised before the clock (the driver's benefit of the doubt, both sides);
  scripted turns are greedy engine requests with EOS ignored, parked per conversation (`keep_kv`) with reuse.
  `--replay generated` goes through `Session` and fails a turn unless `cached_tokens == tokens_in_kv()` before it
  (`kv_tokens` per request): run W1-W3 generated, clean, before publishing a scripted conversation number.
  With `ignore_eos` the engine never samples the EOG ids, as llama.cpp masks them (#61).
- **W6:** the driver's phases: every prompt is prefilled and parked (one token sampled), then all decode together
  from the parked KV; `t = t_pp + t_tg`, each phase timed on its own. A row is refused unless every prompt is still
  parked whole after the prefill phase and the engine ran the decode phase as `tg` steps of exactly `pl` decode rows
  (its t1 samples, so `--telemetry t1` only). Each phase is one `submit_group`, so the parked continuations decode
  together from the first step (#64). Its requests carry the EOG ids #61 masks.
- **Extra fields:** `replay`, `cancel_rate`, `summary.cancelled` (a requested cancel is not a failure);
  larch adds `knob_timeline` (startup values, then each `@S` change with the engine's reply), and `telemetry`
  (`engine_samples`, mean `achieved_gbps`, the ring's `revisions`, `guardrail_trips`, `dropped`).
- **Ollama:** loaded and warmed at the run's `num_ctx` before the clock. The same rendered prompt as `raw` text on
  `/api/generate` (its chat API would apply Ollama's own template), greedy with `repeat_penalty 1.0`, `keep_alive -1`;
  a turn fails unless its `prompt_eval_count` (cached tokens included) equals the hashed prompt's length and its
  cached count is at most that. `cached_tokens` is its `prompt_eval_cached_count`, null when the server omits it,
  never inferred. Its runner always caches prompts, so every Ollama result says `cache_prompt: true` and `compare`
  refuses it against W4 and W5, which reuse nothing.
  It has no `ignore_eos`: an answer that ends early fails `compare`'s length check. One `--concurrency` per
  invocation, against a server started for that N (`baseline.py ollama-serve --parallel N`). `http_overhead_ms`
  (per request and in the summary) = client e2e - its `total_duration`.
- **OpenAI-compatible:** each turn is one streamed `/v1/completions` of `{model: the first /v1/models id, prompt: the
  hashed token ids, max_tokens, temperature 0, ignore_eos, stream, stream_options.include_usage}`: token ids, so no
  server template or tokenizer touches the prompt; `--natural-eos` sends `ignore_eos: false` (`larch serve` has no
  `ignore_eos`, so run it with `--natural-eos`). `t_first` is the first event carrying `choices[0]`, even with empty
  text (servers hold back partial UTF-8). A turn fails unless `usage.prompt_tokens` equals the hashed count and, with
  `ignore_eos`, `completion_tokens` equals `max_tokens`; an error event (or llama-server's `error:` line), a non-200
  status, or a stream without usage or `[DONE]` fails it too. `cached_tokens` is
  `prompt_tokens_details.cached_tokens`, null when absent. `cache_prompt` is the server's own: vLLM's `/metrics`
  (`enable_prefix_caching`), else `--cache-prompt` (refused when `/metrics` says otherwise); a cached token under
  `cache_prompt: false` fails the turn. `server` {model, root, max_model_len, owned_by, version} is what `/v1/models`
  and `/version` (vLLM) or `/props` (llama-server) say, null where silent; `n_ctx_seq` is its `max_model_len`. No
  `--set` (a server's knobs are its startup flags). Several `--concurrency` values only with `cache_prompt: false`: a
  server that caches keeps its prefix cache between runs, so it takes one `--concurrency` and no `--duration`; start
  the server fresh for each row that reuses.
- **Scoreboard:** `scripts/bench-merge` refuses a dirty tree, a parity failure or a missing llama.cpp W4-at-8 result,
  builds HEAD (`--locked`) and runs the binary Cargo reports (wherever `CARGO_TARGET_DIR` puts it) on W4 at 8 and the
  W6 mirror into its own `merge-w4-8`/`merge-fixed` results,
  refuses unless `baseline.py compare` accepts the W4 pair, then appends `history.csv` (commit, machine, model,
  agg tok/s at 8 = W6 `pl=8` `speed`, TTFT p99 and TPOT p99 of W4 at 8, single-stream tok/s = W6 `pl=1`
  `speed_tg`, the W4 serve aggregate at 8); a > 3 % loss in any of them against the machine and model's last line
  exits 1 without appending unless `--correctness-fix`.

