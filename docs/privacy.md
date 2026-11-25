# Privacy and network

The model runs on your computer, and your chats, documents and memory stay on its disk. This page
lists every case where Larch connects to the internet, what it sends and how to avoid it.

Larch sends no usage data, crash reports or analytics. The engine's performance telemetry stays in
memory on your machine, where `larch chat`'s `/status` and the app read it.

## When Larch goes online

| When | Connects to | Sends | To avoid it |
|---|---|---|---|
| You run the installer, or `larch-setup update` | `github.com`, `raw.githubusercontent.com` | Standard HTTPS requests for release files | Install from source |
| The installed app checks for updates, at launch and once a day | `raw.githubusercontent.com`, `github.com` | Requests for the signed release list, through Larch Setup | Turn off **Settings > General > Check for updates** |
| You download a model (app, `larch-setup model get`) | `huggingface.co`, or `LARCH_HF_ENDPOINT` | The model's name. `HF_TOKEN` if you set it | Download GGUF files yourself and pass them with `-m` |
| The agent searches the web (`web_answer`, `research`, **Research** or **Web** in the app) | Your search provider: DuckDuckGo, or Brave, Serper, Tavily or Exa if you added a key | The search query. The provider's key if you added one | Don't use web tools |
| A task runs a step that uses the web. Tasks may use the web by default | Your search provider and the pages' sites | As above | Turn the web off in the task's settings |
| The agent reads a web page | The page's site | A normal page request | Don't use web tools |
| The agent reads an arXiv paper you named | `arxiv.org` | A request for that paper's PDF | Attach the PDF yourself |
| You browse **Tools > Other apps** for MCP servers | `registry.modelcontextprotocol.io`, the official MCP registry | Your search words | Add servers by hand in `config.toml` |
| A tool from a remote MCP server runs | That server | The tool's arguments, which the model writes and can include text from your chat | Only add MCP servers you trust, or local ones |

Every request goes from your computer directly to that host. None of them pass through a server run
by Arkaya Labs.

Requests identify themselves only as Larch and its version (`User-Agent: larch/0.1.0`), with nothing
that identifies you or your machine.

## Working fully offline

Larch works with no network once it's installed and has a model:

1. Download the GGUF files you want while you're online.
2. Don't use **Research**, **Web**, or remote MCP servers.

Chat, documents, memory, tasks with the web turned off, and `larch serve` all work offline.

## What's stored on disk

Your data directory ([where it is](configuration.md#where-data-is-kept)) holds chat transcripts,
agent memory, the document cache and the agent's audit log. They aren't encrypted by Larch; they're
protected by your user account like your other files. `larch-setup uninstall --purge` deletes them.

## The local server

`larch serve` listens only on `127.0.0.1` by default and refuses requests from web pages. See
[HTTP API](api.md#authentication) and [SECURITY.md](../SECURITY.md) before exposing it on a network.
