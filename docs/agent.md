# The agent

The agent lets the model use tools: read and write files in folders you allow, search the web,
remember things between chats, run code in a sandbox and call MCP servers. It runs in the desktop app
and as the `larch-agent` command. It is a preview, so its settings may change.

## Start a chat with tools

In the app, every chat has the agent. From a terminal:

```sh
larch-agent chat -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --allow ~/Projects/notes
```

`-m` takes a path, or the name of a file in the `models` folder of your data directory.

`--allow` names a folder the tools may read and write. Repeat it for more folders. Without it, the
file tools see only the folders listed in [the agent's settings](configuration.md#agent-settings).

## The agent asks before it changes anything

Tools fall into four kinds, and each kind has its own rule:

| Kind | Examples | Asks first |
|---|---|---|
| Reads | Reading files, searching the web, recalling memory | Never |
| Changes on this computer | Writing a file, running a command, saving to memory | Yes, unless you chose **Always in this folder** for that folder |
| Changes outside this computer | An MCP tool that sends a message or edits a remote document | Yes, unless you allowed that tool |
| Spending money | A tool that buys something or calls a paid service | Always |

Once a chat has read anything from outside your control, such as a web page or an MCP result, every
change asks again for the rest of that chat, even where you had allowed it. That stops a web page
from telling the model to write files on your behalf.

In the app, requests show under **Needs you** with **Allow once**, **Always in this folder** or
**No**. In the terminal you answer `y` or `n`. Every request and its outcome are written to the
audit log, `agent/log/<date>.jsonl` in your data directory.

## Tools

| Tool | What it does | Goes online |
|---|---|---|
| `list_dir`, `read_file`, `find_files`, `search_files` | Read files in allowed folders | No |
| `write_file` | Write a file in an allowed folder, after you approve | No |
| `remember`, `recall`, `forget` | Keep short notes about you between chats | No |
| `ask_docs`, `read_papers` | Answer from documents; see [Documents](documents.md) | No |
| `web_answer`, `research` | Search the web, read pages and answer with citations | Yes |
| `changed_files`, `diff_file`, `show_file`, `log` | Read a git repository. Off unless `git_tools = true` | No |
| MCP tools | Whatever the MCP servers you added provide | Depends on the server |

In code mode the agent also gets `bash` and `run_tests`, which run inside the sandbox described
below.

## Web search

`web_answer` and `research` use the first search provider that answers. With no keys set, that's
DuckDuckGo's HTML page. If you add a key for Brave, Serper, Tavily or Exa in the settings, that
provider is used first.

Your search query goes to that provider, and each page the agent reads is fetched from its site,
from your computer. Nothing else is sent. In the app, a chat only searches the web when you press
**Web** or **Research** for that message. Tasks use the web by default; turn it off in the task's
settings.

## Code mode and the sandbox

Commands the agent runs in code mode are confined to the project's worktree:

- no network,
- writes only inside the worktree and a private temporary folder,
- limits on CPU time, memory and number of processes,
- the whole process group is killed on timeout.

Linux uses `unshare`, Landlock, seccomp and `prlimit`. macOS uses `sandbox-exec`. Windows uses an
AppContainer and a Job Object. If the sandbox can't start, the command doesn't run, and the agent
reports why instead.

## MCP servers

Add servers in the app under **Tools > Other apps**, or in the agent's `config.toml`. Larch supports
servers over stdio and HTTP, including servers that sign in with OAuth.

An MCP tool's arguments go to that server. A remote server sees what the model sends it, so only
add servers you trust.

Larch can also be an MCP server for other apps, such as a coding assistant: `larch-agent mcp` serves
the agent's tools on your GPU. Reads are allowed, writes only inside the folders you pass with
`--write`, and the web only with `--web`.

## Memory

Larch remembers short facts about you, like your name or the project you're on, and uses them in
later chats. Facts get saved two ways:

- The model calls `remember` during a chat. You see the call in the chat.
- After each turn, the loaded model reads your message in the background and saves any lasting
  facts in it. This is on by default; set `memory_capture = false` in the agent's `config.toml` to
  turn it off.

Memory is a set of plain files in `agent/memory` in your data directory. Nothing leaves your
computer. You can read, edit or delete it under **Settings > Memory** or with any text editor, and
`forget` works from any chat.

## Limits

- How well the agent picks tools depends on the model. `larch-agent bench` measures how often a
  model picks the right tool, on your machine.
- The agent is a preview. Review what it wants to write or run before you allow it.
