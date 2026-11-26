![Larch](docs/assets/larch-banner-light.svg#gh-light-mode-only)
![Larch](docs/assets/larch-banner-dark.svg#gh-dark-mode-only)

Larch runs GGUF language models on your own computer. It serves many chats from one graphics card,
reads long documents and answers with quotes it has checked, and comes with a desktop app and an
agent. The model runs on your machine. Larch only goes online when you download a model or use a
tool that needs the web, and [Privacy and network](docs/privacy.md) lists each of those.

Larch is made by [Arkaya Labs](https://arkaya-labs.com) and is free software under the Apache 2.0
licence. It is early: expect rough edges, and read [what isn't supported yet](#what-isnt-there-yet).

![The Larch app's Home screen, with sample chats](docs/assets/app-home-light.png#gh-light-mode-only)
![The Larch app's Home screen, with sample chats](docs/assets/app-home-dark.png#gh-dark-mode-only)

## Install

On Windows, in PowerShell:

```powershell
irm https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.ps1 | iex
```

On macOS (Apple silicon) or Linux (x86-64):

```sh
curl -fsSL https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.sh | bash
```

The command opens the Larch Setup window, which asks whether you want the app, the command line
tools or both, picks the build for your GPU, and offers a model that fits your memory. Every file it
downloads is checked against a signed list before anything is installed. [More about installing](docs/install.md).

On a new Mac, macOS may first ask to install its command line developer tools, which the installer
needs for `python3`. Click Install, wait for it to finish, then run the install command again.

## First chat

Open Larch from the Start menu, Launchpad or your app launcher, pick a model, and type.

From a terminal:

```sh
larch chat -m path/to/model.gguf
```

[The quickstart](docs/quickstart.md) takes you from a fresh install to a chat, a document question
and a local API in about ten minutes.

## What it runs on

| GPU | Status |
|---|---|
| NVIDIA RTX 30, 40 and 50 series, Windows and Linux | Supported. Most of the testing happens here |
| Apple silicon, macOS 26 or later | Supported |
| AMD Radeon, Windows and Linux | Preview. Tested on two GPUs only, not in the installer yet |
| Any 64-bit CPU | Works, slowly. Used as the reference for correctness tests |

Models: Llama 3.x, Mistral, Qwen 2.5, Qwen 3 and Qwen 3.5, dense and mixture-of-experts, as GGUF
files. The full list, and which sizes fit which cards, is in [Hardware](docs/hardware.md) and
[Models](docs/models.md).

## How fast it is

On one RTX 5070 Ti, with Llama 3.1 8B and Qwen 2.5 7B at Q4_K_M, Larch served 8 and 16 chats at
once with 11 to 15 % more total throughput than llama.cpp's server loop and 42 to 50 % lower p99
time per output token. Over HTTP it served 20 to 49 % more than Ollama at the same loads.

It was slower in other cases. Its p99 time to first token was 16 to 46 % higher than llama.cpp's at
those loads, a single chat ran 4 % behind llama.cpp on Llama 3.1 8B, in llama.cpp's fixed batched
benchmark it was 10 to 20 % slower at 4 to 16 sequences, and on a 12-turn single chat its first token
took 2.0 to 2.5 times as long as Ollama's.

These numbers are from 17 October 2026 (Larch `4a540ae`, llama.cpp b11458, Ollama 0.40.1). The
method, the commands and the raw data are in
[bench/results/final-sweep-2026-10-17.md](bench/results/final-sweep-2026-10-17.md), and
[Performance](docs/performance.md) explains how to run the same tests on your machine.

## What isn't there yet

- AMD GPUs are a preview and the installer doesn't offer them.
- Prefill (reading a long prompt) is slower than llama.cpp's when many requests arrive at once.
- On a single chat, Ollama returns the first token sooner.
- Architectures other than Llama, Mistral and Qwen don't load. Gemma, Phi, DeepSeek and gpt-oss are
  not supported.

## Documentation

| | Pages |
|---|---|
| ![](docs/assets/icon-sprout-light.svg#gh-light-mode-only)![](docs/assets/icon-sprout-dark.svg#gh-dark-mode-only) | **Get started**: [Install](docs/install.md) · [Quickstart](docs/quickstart.md) · [Hardware](docs/hardware.md) · [Models](docs/models.md) |
| ![](docs/assets/icon-trail-light.svg#gh-light-mode-only)![](docs/assets/icon-trail-dark.svg#gh-dark-mode-only) | **Guides**: [The app](docs/app.md) · [Documents](docs/documents.md) · [The agent](docs/agent.md) · [Troubleshooting](docs/troubleshooting.md) |
| ![](docs/assets/icon-page-light.svg#gh-light-mode-only)![](docs/assets/icon-page-dark.svg#gh-dark-mode-only) | **Reference**: [CLI](docs/cli.md) · [HTTP API](docs/api.md) · [Configuration](docs/configuration.md) |
| ![](docs/assets/icon-cone-light.svg#gh-light-mode-only)![](docs/assets/icon-cone-dark.svg#gh-dark-mode-only) | **How it works**: [Architecture](larch-architecture.md) · [Performance](docs/performance.md) · [Privacy and network](docs/privacy.md) · [Building from source](docs/build.md) |

## Contributing

Bug reports, model requests and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) lists
the checks each kind of change needs; kernel and model changes are checked against a CPU reference
and against llama.cpp on the same GGUF file. Report security problems privately, as described in
[SECURITY.md](SECURITY.md). For questions, see [SUPPORT.md](SUPPORT.md).

## Licence

Apache 2.0, see [LICENSE](LICENSE). Third-party code is listed in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Model weights are not covered by this licence; each
model has its own terms.
