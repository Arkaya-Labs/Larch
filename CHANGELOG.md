# Changelog

Notable changes to Larch, newest first. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, a minor version can change interfaces and file formats.

## 0.1.1

Fixes for the Mac install.

### Install
- macOS: the install command no longer needs `python3`, so a Mac without Apple's command line
  developer tools installs without first downloading them.
- The installer window wraps a long install path inside its box.
- The installer downloads its first model over several connections at once, as the app's Models
  page already did.

### Server
- `larch serve` starts with a 32,768-token context unless `--set max_context` chooses one, so
  Claude Code's first request (about 12,000 tokens) fits. It halves the context down to 8,192 when
  the model does not fit.
- Thinking can be turned off per request: OpenAI's `reasoning_effort: "none"`,
  `chat_template_kwargs.enable_thinking` or Ollama's `think`, and Anthropic's `thinking`. An
  Anthropic request without `thinking` no longer thinks, as in Anthropic's API. `--thinking on|off`
  sets the default.
- A thinking model's reasoning streams as it is written: OpenAI `reasoning_content` deltas and
  Anthropic `thinking` blocks, so clients no longer wait in silence through a long think.

### Engine
- Mixture-of-experts models that keep some expert layers in system RAM beside an NVIDIA GPU run
  faster with several chats, and `larch faster` writes a smaller copy of such a model that fits on
  the GPU.

## 0.1.0 (25 Nov 2026)

The first public release, for Apple silicon Macs (Metal and CPU). Windows and Linux builds are
coming soon; until then their install commands say so and stop.

### Engine
- Runs GGUF models of the `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe`
  architectures on CUDA (RTX 30, 40, 50), Metal (Apple silicon) and the CPU. HIP (AMD) as a preview.
- Serves many conversations from one GPU, keeping each conversation's state between turns.
- Keeps some of a mixture-of-experts model's expert layers in system RAM when it doesn't fit.
- `larch serve`: OpenAI- and Anthropic-compatible HTTP API on localhost.
- Runtime knobs that change between engine steps without interrupting a chat.
- One fault path: a GPU fault or hang fails active requests cleanly and exits with code 3;
  `larch supervise` kills and restarts in under 2 s.

### App and agent (preview)
- Desktop app with chats, documents, tasks, models and tools. A first launch asks what to call you
  and offers a short tour of the app, which Settings › General replays.
- Folio: answers from attached documents with checked page quotes, kept ready across restarts.
- Agent tools for files, memory, web research, git, code in a sandbox and MCP servers.
- Discover recommends models by the GPU's memory, up to a 35B A3B model (Q3_K_M) at 18 GB or more.

### Install
- Installers for Windows, macOS and Linux that check every download against a signed manifest, with
  update, rollback and uninstall.
