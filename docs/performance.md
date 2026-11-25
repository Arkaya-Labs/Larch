# Performance

This page explains what Larch is built to be fast at, the published results with their method, where
it's slower, and how to measure it on your own machine.

## What Larch optimizes for

Larch is built for one GPU serving several conversations at once: a chat app, an agent running
helpers in parallel, or a few people sharing one machine. One engine thread puts every active
conversation into each step on the GPU, keeps a finished turn's state so the next turn only computes
the new words, and reads long prompts in chunks so one document doesn't stall the other chats.

That design trades some single-chat latency for total throughput under load. The results below show
both sides.

## Published results

From the 17 October 2026 sweep. Method, commands and raw data:
[bench/results/final-sweep-2026-10-17.md](../bench/results/final-sweep-2026-10-17.md).

> On one RTX 5070 Ti (16 GB, sm_120, WSL2, driver 617.14), we compared larch (dev `4a540ae`) with
> llama.cpp b11458 and Ollama 0.40.1 on the same two 7–8B instruct models at Q4_K_M (m1 and m5 in
> `bench/models.toml`). On 200 independent prompts with exact output lengths (W4, 2 runs per engine,
> alternated), larch's aggregate throughput was 11–15 % above llama.cpp's server loop at 8 and 16
> concurrent requests, with 42–50 % lower p99 time per output token. Its p99 time to first token was
> 16–46 % higher at the same loads, and single-stream throughput was 4 % behind on m1 and within noise
> on m5. In llama-batched-bench's fixed pp512/tg128 shape (9 runs a side), larch matched llama.cpp at
> one sequence and was 10–20 % slower at 4–16 sequences, limited mainly by prefill. Over localhost HTTP
> with natural end of sequence (3 runs per engine, alternated), larch's W4 aggregate was 20–49 % above
> Ollama's at 8 and 16 concurrent, but on a 12-turn single chat (W1) its time to first token was
> 2.0–2.5× Ollama's at p50.

Against vLLM on the same card, Larch tied at one request and was behind at 4, 8 and 16. The numbers
and method are in [bench/results/vllm-5070ti-2026-10-23.md](../bench/results/vllm-5070ti-2026-10-23.md).

## Where Larch is slower

| Case | Compared with | Source |
|---|---|---|
| p99 time to first token at 8 and 16 requests | llama.cpp | October sweep, above. Mainly prefill |
| First token in a 12-turn single chat | Ollama | October sweep |
| Fixed batched benchmark at 4 to 16 sequences | llama.cpp | October sweep. Mainly prefill |
| 4 to 16 requests at once | vLLM | [vLLM comparison](../bench/results/vllm-5070ti-2026-10-23.md) |
| Small models (1B, 3B) on an RTX 3050 Laptop | llama.cpp | Measured in October. Later kernel work narrowed the gap; not remeasured since |

If first-token latency matters more to you than throughput, `schedule_policy` and
`prefill_chunk_tokens` are the knobs to try. See [Configuration](configuration.md#knobs).

## Measure it yourself

`larch bench` runs the same workloads against Larch, llama.cpp's server, Ollama or any
OpenAI-compatible server:

```sh
larch bench -m <dir>/m1.gguf --workload batch --concurrency 1,8,16 --json w4.json
larch bench -m <dir>/m1.gguf --engine ollama --url http://127.0.0.1:11434 --workload batch --concurrency 1,8,16 --json w4-ollama.json
```

Run each engine more than once and alternate them; one run on a warm GPU after a cold one will
mislead you. [bench/README.md](../bench/README.md) has the workloads, the rules and the result format,
and `python3 scripts/baseline.py compare a.json b.json` compares two results.

## Correctness comes first

Every kernel is tested against a CPU reference implementation, and every supported model is checked
against llama.cpp's output on the same GGUF file before a speed change merges. A change that makes
Larch faster and less accurate is not accepted. See [CONTRIBUTING.md](../CONTRIBUTING.md#validation).
