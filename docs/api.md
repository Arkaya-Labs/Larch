# HTTP API

`larch serve` runs a local server that speaks the OpenAI and Anthropic APIs, so existing clients and
SDKs can use a model on your GPU. This page lists the endpoints, the request fields Larch supports
and how authentication works.

## Start the server

```sh
larch serve -m model.gguf
```

It listens on `127.0.0.1:8080`. `--host` and `--port` change that; see
[Remote access](#remote-access) before you listen on anything other than localhost.

Agent clients such as Claude Code send 10,000 to 20,000 tokens of system prompt and tools before
the conversation starts, so `larch serve` asks for a 32,768-token context unless you set one with
`--set max_context=N`. If that does not fit in memory, it halves the context down to 8,192 and
prints the context it got.

## Endpoints

| Method | Path | API | Notes |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI | Streaming, tools, JSON output |
| `POST` | `/v1/completions` | OpenAI | Plain text completion |
| `GET` | `/v1/models` | OpenAI | Lists the one loaded model |
| `POST` | `/v1/messages` | Anthropic | Streaming, tools, JSON output |
| `POST` | `/v1/messages/count_tokens` | Anthropic | Counts the prompt's tokens |
| `GET` | `/health` | Larch | Always unauthenticated |
| `GET` | `/api/status` | Larch | Engine status and telemetry |

## Examples

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Name three larch species."}],
)
print(reply.choices[0].message.content)
```

The `model` field is required by the SDK but Larch ignores it and uses the loaded model.

### Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="unused")
reply = client.messages.create(
    model="local",
    max_tokens=256,
    messages=[{"role": "user", "content": "Name three larch species."}],
)
print(reply.content[0].text)
```

### curl, streaming

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream": true, "messages": [{"role": "user", "content": "Count to five."}]}'
```

If a client hangs with no reply, it may be sending local requests through an HTTP proxy. Set
`NO_PROXY=127.0.0.1,localhost` in its environment.

## Supported request fields

Larch ignores fields it doesn't know. Fields it knows but can't honour are refused with a `400`
error that says why, so you never get an answer that silently ignored your settings.

### `/v1/chat/completions`

| Field | Support |
|---|---|
| `messages` | Yes. Text content; image parts are refused |
| `stream`, `stream_options.include_usage` | Yes |
| `max_tokens`, `max_completion_tokens` | Yes. A value larger than the context allows is treated as no limit |
| `temperature`, `top_p`, `seed`, `stop` | Yes |
| `tools`, `tool_choice` | Yes, when the model has a tool-call format. `tool_choice` may be `auto`, `none`, `required` or a named function |
| `response_format` | `text`, `json_object` and `json_schema` |
| `n` | Only `1` |
| `logprobs` | Refused: not supported yet |
| `frequency_penalty`, `presence_penalty` | Refused unless `0`. Use the `repeat_penalty_default` knob instead |
| `reasoning_effort`, `chat_template_kwargs.enable_thinking`, `think` | Turn thinking on or off for a model that thinks. `reasoning_effort: "none"` is off; any other effort is on. Unset, the model thinks as its chat template does by default |

A thinking model's reasoning streams as `reasoning_content` deltas, and a non-streamed answer
carries it in `message.reasoning_content`. It is never part of `content`.

### `/v1/completions`

`prompt` (a string, one string in an array, or token ids), `stream`, `max_tokens`, `temperature`,
`top_p`, `seed` and `stop`. `logprobs`, `echo` and `suffix` are refused.

### `/v1/messages`

`messages`, `system`, `max_tokens` (required), `stream`, `temperature`, `top_p`, `top_k`,
`stop_sequences`, `tools`, `tool_choice`, and `output_config.format` (or the older `output_format`)
for JSON output.

`thinking` works as in Anthropic's API: with no `thinking` field the model does not think, and
`{"type": "enabled"}` or `{"type": "adaptive"}` turns it on. The reasoning comes back as a
`thinking` block before the answer, with an empty `signature`, unless the request sets
`"display": "omitted"`. `budget_tokens` is accepted and ignored.

`larch serve --thinking on` or `--thinking off` sets the default for requests to either API that
do not choose.

## Authentication

By default the server needs no key and accepts requests only from this machine. It refuses a request
whose `Host` isn't a loopback name or whose `Origin` is another website, so a web page you visit
can't use it.

Set `LARCH_API_KEY` to require a key:

```sh
LARCH_API_KEY=choose-a-long-random-string larch serve -m model.gguf
```

Clients send it as `Authorization: Bearer <key>` or `x-api-key: <key>`. `/health` stays open.

## Remote access

`larch serve` refuses to listen on a non-loopback address unless `LARCH_API_KEY` is set. Even with a
key, the server speaks plain HTTP, has no connection limit and has one shared key for everyone. To
use it from another machine, put it behind a reverse proxy that adds TLS and limits connections. See
[SECURITY.md](../SECURITY.md#larch-serve-boundaries).
