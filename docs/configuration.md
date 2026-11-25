# Configuration

This page lists where Larch keeps its data, the agent's settings file, the environment variables
Larch reads, and the engine's knobs.

## Where data is kept

| OS | Data directory |
|---|---|
| Windows | `%LOCALAPPDATA%\larch` |
| macOS | `~/Library/Application Support/larch` |
| Linux | `$XDG_DATA_HOME/larch`, or `~/.local/share/larch` when `XDG_DATA_HOME` isn't set |

Inside it:

| Folder | Contents |
|---|---|
| `chat/` | Transcripts from `larch chat`, one `.jsonl` file per conversation |
| `models/` | Models downloaded by the installer |
| `agent/config.toml` | The agent's settings |
| `agent/memory/` | What the agent remembers about you |
| `agent/log/` | The agent's audit log, one `.jsonl` file per day |
| `doc-cache/` | Documents Larch has read, kept for after a restart |

Chats, memory and logs can hold anything you or the model wrote. Treat them as private and redact
them before you share them in a bug report.

## Agent settings

The agent reads `agent/config.toml` in the data directory. A missing file means every default. The
desktop app writes the same file when you change a setting. An unknown key is an error that names the
line and lists the valid keys.

```toml
# Folders the tools may read and write, after any --allow. ~ is your home folder.
allowed = ["~/Documents", "~/Projects"]

# Keep documents read on disk, so a restart doesn't read them again.
docs_keep_ready = true

# Read each of your messages in the background and save lasting facts to memory.
memory_capture = true

# Offer the git tools (changed_files, diff_file, show_file, log).
git_tools = false

# Search provider keys. Each key moves its provider ahead of the free one.
brave_key = ""
```

| Key | Default | Effect |
|---|---|---|
| `allowed` | none | Folders the file tools may use |
| `mcp_allow` | none | Folders MCP clients of `larch-agent mcp` may read when it gets no `--allow` |
| `docs_keep_ready` | `true` | Keep read documents on disk across restarts |
| `docs_cache_bytes` | 8 GiB | Most bytes the document cache may use |
| `memory_capture` | `true` | Save facts from your messages in the background |
| `recall_hits` | `5` | Lines a memory lookup returns, 1 to 10 |
| `git_tools` | `false` | Offer the git tools |
| `web_pages` | `5` | Pages a web answer reads |
| `thinking_budget` | `100` | Tokens the model may think before answering, up to 4096. `0` turns thinking off |
| `answer_max_tokens` | `0` | Longest answer after thinking. `0` is no limit |
| `tool_format` | from the model | `qwen3`, `llama3` or `mistral` |
| `brave_key`, `serper_key`, `tavily_key`, `exa_key` | none | Web search provider keys |

The remaining keys (`router`, `router_margin`, `tool_grammar`, `probe_low`, `probe_high`,
`outside_pick`, `code_experts`, `code_verify`, `docs_workers`, `docs_window`, `max_workers`,
`paper_tokens`, `doc_followups`, `followup_tokens`, `memory_keys`, `web_reader`) tune experiments and
benchmarks. Leave them at their defaults.

## Environment variables

| Variable | Used by | Effect |
|---|---|---|
| `LARCH_API_KEY` | `larch serve` | Shared key every request must send. Required for a non-loopback `--host` |
| `HF_TOKEN` | App, model downloads | Hugging Face token, for gated models. Sent only to the Hub, never across a redirect |
| `LARCH_HF_ENDPOINT` | App, model downloads | A Hugging Face mirror to use instead of `https://huggingface.co` |
| `LARCH_SETUP_CHANNEL` | `larch-setup` | Release channel: `stable` (default) or `candidate` |
| `LARCH_PREFIX_CACHE_BLOCKS` | `larch-agent` | Blocks of 16 tokens kept for the prefix cache. `0` turns it off |
| `LARCH_FIT_DIAGNOSTICS` | Engine | Set to `1` to print how the model was fitted into GPU memory |
| `XDG_DATA_HOME` | Linux | Moves the data directory |

Build-time variables (`LARCH_BACKEND`, `LARCH_CUDA_ARCH` and others) are in
[Building from source](build.md#build-configuration).

## Knobs

Knobs are engine settings. Most of them change while the engine runs, between two steps, without
interrupting a chat. `larch knob list` prints all of them. These are the ones worth knowing:

| Knob | Default | Range | Takes effect | Effect |
|---|---|---|---|---|
| `max_batch_seqs` | `16` | 1 to 64 | Live | Most requests computed together in one step |
| `prefill_chunk_tokens` | `512` | 64 to 8192 | Live | How much of a long prompt is read per step, so one long prompt doesn't stall other chats |
| `schedule_policy` | `decode_first` | `decode_first`, `balanced`, `prefill_first` | Live | Whether running chats or new prompts go first |
| `max_waiting` | `256` | 0 to 4096 | Live | Requests that may queue before new ones are refused |
| `temperature_default` | `0.7` | 0 to 2 | Next request | Temperature when a request doesn't set one |
| `top_p_default` | `0.95` | 0 to 1 | Next request | |
| `top_k_default` | `40` | 0 to 64 | Next request | |
| `repeat_penalty_default` | `1` | 1 to 2 | Next request | `1` is off |
| `max_context` | `8192` | 512 to 131072 | Model reload | Tokens of context per chat |
| `kv_pool_fraction` | `0.85` | 0.3 to 0.95 | Model reload | Share of free GPU memory used for chats' state |
| `device` | `0` | 0 to 63 | Restart | Which GPU to use |

Set them at startup with `--set`, or live inside `larch chat`:

```sh
larch chat -m model.gguf --set max_context=32768 --set max_batch_seqs=8
```

```text
> /knob set prefill_chunk_tokens 1024
> /knob revert 3
```

See [CLI](cli.md#chat) for the other in-chat commands.
