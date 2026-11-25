# CLI reference

This page lists the `larch` commands and their options. Every command also prints its options with
`--help`. For the agent's commands, run `larch-agent --help`; for the installer, see
[Install](install.md#update-roll-back-remove).

Examples use `<dir>` for the folder that holds your GGUF files and the aliases from
[Models](models.md#the-tested-models). Binaries built from source are in `./target/release/`; installed
ones are on your `PATH` as `larch`.

## Chat

```
./target/release/larch chat -m <dir>/m1.gguf --max-tokens 512
```

Wait for `larch: ready`, then type a message at the `>` prompt. After each answer, larch prints the speed
of that turn. The numbers below are an example and change with the model and the machine:

```
larch: ready
transcript 1791831519825476000-28246; /status, /knob, /ab, /reset, /quit
> What is the capital of France?
The capital of France is Paris.
213.47 tok/s · TTFT 44.13 ms · 42 prompt + 8 generated · 0 cached · 49 parked
```

TTFT is the time to the first token. `cached` is the number of prompt tokens that larch did not compute
again. The conversation kept their attention state (the KV cache) from the previous turn.

| Option | Effect |
|---|---|
| `--max-tokens <n>` | Maximum tokens for each answer. If unset, runs until the model ends or the context fills |
| `--system <text>` | System message for a new conversation |
| `--temperature <t>` | Sampling temperature. If you do not set it, the current knob value applies |
| `--stop <text>` | Stop string. You can repeat it |
| `--set <knob>=<value>` | Knob value at startup. You can repeat it |
| `--device <n>` | GPU ordinal. The default is 0 |
| `--no-keep-kv` | Don't keep the conversation's state between turns. Each turn computes the whole prompt again |
| `--mmproj <file>` | Vision projector GGUF for the model, so images in a resumed chat are seen again |
| `--drafter <file>` | Draft model GGUF for speculative decoding, or `mtp` to use the model's own MTP block |
| `--cpu` | Run on the CPU reference |
| `--resume <id>` | Continue the transcript with this ID |
| `--data-dir <dir>` | Base directory for transcripts |
| `--fail-overflow` | Reject a prompt that is too long. The default drops the oldest exchanges |

A knob is a runtime parameter of the engine, for example `max_batch_seqs`. `larch knob list` prints all
knobs with their defaults and ranges.

These commands work inside a conversation:

- `/status` prints the health of the engine and the number of parked tokens.
- `/knob list`, `/knob get <name>`, `/knob set <name> <value>` and `/knob revert <revision>` read and change knobs.
- `/ab <knob>=<value> [period_steps]` alternates between two values of a knob. `/ab stop <trial>` stops it.
- `/reset` starts the conversation again and releases its KV cache.
- `/quit` exits.

Larch writes the transcript after every turn to `chat/<id>.jsonl` in the data directory. The data directory
is `~/Library/Application Support/larch` on macOS, `%LOCALAPPDATA%\larch` on Windows and
`$XDG_DATA_HOME/larch` on Linux, or `~/.local/share/larch` when `XDG_DATA_HOME` is not set.

Do not start two larch GPU processes on one device at the same time. Each engine sizes its buffers from the
free memory that the device reports when it loads, so two engines that load together can both count the
same free memory and exhaust it.

## Serve

```sh
larch serve -m <dir>/m4.gguf
```

| Option | Effect |
|---|---|
| `--host <addr>` | Address to listen on. The default is `127.0.0.1`. Any other address needs `LARCH_API_KEY` |
| `--port <n>` | Port. The default is `8080` |
| `--name <name>` | The model name `/v1/models` shows. The default is the file name without `.gguf` |
| `--tool-format <f>` | `qwen3`, `llama3` or `mistral`, for a model whose GGUF names the wrong one (Mistral GGUFs say `llama`) |
| `--tool-grammar` | Constrain tool calls to their schema, as the agent does |
| `--set <knob>=<value>` | Knob value at startup. You can repeat it |
| `--device <n>`, `--cpu` | As for `chat` |

Endpoints, request fields and authentication are in [HTTP API](api.md).

## Benchmark

`larch bench` runs a fixed workload and prints one summary line. With `--json`, it also writes the full
result in the `larch-bench/1` schema.

```
./target/release/larch bench -m <dir>/m1.gguf --workload chat --concurrency 1 --machine mac --json w1.json
```

```
w1-chat.jsonl c=1: 12 requests, 0 failed, 0 cancelled, 157.52 tok/s, TTFT p50 52.4 p99 130.0 ms, TPOT p50 5.70 p99 7.28 ms
```

TPOT is the time for each token after the first.

| `--workload` | Content | Usual `--concurrency` |
|---|---|---|
| `chat` | One conversation of 12 turns | 1 |
| `chats` | 8 conversations of 8 turns | 4, 8 |
| `agent` | Agent loops on a system prefix of 4096 tokens | 1, 3 |
| `batch` | 200 independent prompts | 1, 4, 8, 16 |
| `longprompt` | Prompts of 2048 and 4096 tokens | 1 |

- `--concurrency 1,4,8` runs the values in sequence and writes `<stem>-c<N>.json` for each.
- `--mode serve` is the `batch` workload. `--slice K/N` runs part K of N, and `scripts/bench-join` merges the parts.
- `--mode batched --pp 512 --tg 128 --npl 1,4,8,16` runs the fixed shape of `llama-batched-bench`.
- `--set <knob>=<value>` sets a knob at startup. `--set <knob>=<value>@<seconds>` changes it during the run.

`bench/README.md` has the workload rules, the result schema and the llama.cpp and Ollama baselines.
`python3 scripts/baseline.py --machine <m> compare <a.json> <b.json>` compares two results. If the two runs
differ in model, workload, concurrency or context size, the command refuses.

## Other commands

| Command | Purpose |
|---|---|
| `larch info` | Lists the devices, the backend and the driver versions |
| `larch knob list` | Prints the knob registry |
| `larch parity -m <gguf> --ref <dump.jsonl> --bound <kl> --margin <nats>` | Compares the logits of the engine with a llama.cpp dump. Exit code 0 is a pass, 1 is a missed gate and 2 means that it did not run |
| `larch tokcheck -m <gguf> --ref <tokens.jsonl>` | Compares the tokenizer with a llama.cpp tokenization dump |
| `larch supervise -- <command>` | Runs a command and starts it again each time it fails, with a backoff. An engine fault (exit code 3) restarts it at once. An exit with code 0 ends the supervisor |

Each command prints its options with `--help`.
