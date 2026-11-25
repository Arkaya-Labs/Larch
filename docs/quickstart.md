# Quickstart

This walks you through a first chat, a question about a document and a local API that other apps can
use. It uses the command line on Linux and an NVIDIA GPU; the same steps work on Windows and macOS
with the paths changed. It takes about ten minutes, most of it downloading the model.

You need Larch installed with the command line tools. If you haven't done that, follow
[Install](install.md) first.

## 1. Check that Larch sees your GPU

```sh
larch info
```

It prints one line per device. Your GPU is the `device 0:` line, with its name, the backend in
brackets (`Cuda` or `Metal`), the driver version and its free and total memory in bytes. The
`device cpu` line is always there. If that's the only line and you have an NVIDIA card, see
[GPU not found](troubleshooting.md#larch-doesnt-find-my-gpu).

## 2. Download a model

List the catalog models that fit your machine:

```sh
larch-setup model list
```

Download Llama 3.1 8B Instruct. It is 4.9 GB and needs about 7 GB of GPU memory:

```sh
larch-setup model get Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Larch checks the file's hash after the download and saves it in the `models` folder of your data
directory, here `~/.local/share/larch/models/`. If your GPU has less than 8 GB, pick a smaller model
from the list instead.

## 3. Chat

```sh
larch chat -m ~/.local/share/larch/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Wait for `larch: ready`, then type a question at the `>` prompt. Here are two turns copied from a
real run. It used a 0.5B model on a computer without a GPU, so its speeds are far below what you'll
see on a graphics card:

```text
larch: ready
transcript 1795506112634097322-5145; /status, /knob, /ab, /reset, /quit
CPU reference/debug mode
> What is the capital of France?
The capital of France is Paris.
1.20 tok/s · TTFT 11859.82 ms · 36 prompt + 8 generated · 0 cached · 43 parked
> And of Italy?
The capital of Italy is Rome.
1.12 tok/s · TTFT 4522.51 ms · 57 prompt + 8 generated · 43 cached · 64 parked
```

The line after each answer is that turn's numbers: speed in tokens per second, time to the first
token, the tokens read and written, and how many came from the cache. On the second turn, 43 of the
57 prompt tokens were `cached`: Larch kept the conversation's state from the first turn (`parked`),
so it only computed the new words. [Hardware](hardware.md#measured-speeds) lists measured speeds on
real GPUs.

Type `/quit` to leave. The conversation is saved, and `larch chat --resume <id>` picks it up again,
where `<id>` is the number on the `transcript` line.

## 4. Ask about a document

Document questions go through the agent. Point it at a folder that holds a PDF or text file:

```sh
larch-agent chat -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --allow ~/Documents
```

`larch-agent` finds a model by its file name in the `models` folder, so you don't need the full path.

Then ask about a file by name:

```text
> What does report.pdf say about Q3 revenue?
```

The answer quotes the document and gives the page for each quote. Larch checks every quote against
that page and adds a note under the answer if one doesn't match. [Documents](documents.md) explains
how this works with long files.

## 5. Use Larch from another app

Start the server:

```sh
larch serve -m ~/.local/share/larch/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

It listens on `127.0.0.1:8080`, only on your machine. In a second terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hello in French."}]}'
```

Any client that speaks the OpenAI or Anthropic API can use the same address. Set its base URL to
`http://127.0.0.1:8080/v1`. [HTTP API](api.md) lists what's supported.

## Next

- [The app](app.md) does all of this in one window.
- [Models](models.md) explains how to load any GGUF file, not only the catalog.
- [The agent](agent.md) covers web search, code and MCP tools.
