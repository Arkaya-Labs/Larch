# Models

Larch loads language models in the GGUF format, the same files llama.cpp and Ollama use. This page
covers which ones load, where to get them and their licences.

## Architectures

| Architecture (GGUF `general.architecture`) | Families |
|---|---|
| `llama` | Llama 3.x, Mistral |
| `qwen2` | Qwen 2.5 |
| `qwen3` | Qwen 3 |
| `qwen3moe` | Qwen 3 mixture-of-experts, for example Qwen3 30B-A3B |
| `qwen35` | Qwen 3.5 |
| `qwen35moe` | Qwen 3.5 mixture-of-experts, for example Qwen3.5 35B-A3B |

Any other architecture is refused when the file loads, with a message naming it. Gemma, Phi,
DeepSeek and gpt-oss are not supported yet. If you'd like one added,
[open a model request](https://github.com/Arkaya-Labs/Larch/issues/new?template=model_request.yml).

## Quantizations

Larch loads tensors in F32, F16, Q8_0, Q6_K, Q5_K, Q5_0, Q4_K, Q4_0, Q3_K and Q2_K, so files sold as
Q4_K_M, Q5_K_M, Q6_K, Q8_0, Q3_K_M and similar load. Q4_K_M is the format most testing happens on.

BF16, Q4_1, Q5_1, the IQ formats and MXFP4 are not supported. A file that uses one of them is refused
when it loads, with a message naming the type.

## Where to get models

**In the app:** Models > Discover searches Hugging Face and shows which files fit your GPU before you
download.

**With the installer:** `larch-setup model list` shows the catalog models that fit your memory, and
`larch-setup model get <file>` downloads one and checks its hash.

**By hand:** any GGUF file of a supported architecture works. Pass its path with `-m`:

```sh
larch chat -m path/to/model.gguf
```

The app downloads into the shared Hugging Face cache, the folder llama.cpp and the `huggingface_hub`
tools also use, so a file you already have there isn't downloaded twice.

## The tested models

These are the models every change is checked against. Their exact files and hashes are in
[bench/models.toml](../bench/models.toml), where each has a short alias (`m1`, `m5`) that the
benchmarks and the codebase use.

| Alias | Model | File size | Checked for |
|---|---|---|---|
| `m1` | Llama 3.1 8B Instruct, Q4_K_M | 4.9 GB | Benchmarks, parity, catalog |
| `m5` | Qwen 2.5 7B Instruct, Q4_K_M | 4.7 GB | Benchmarks, catalog |
| `m2` | Qwen 3 8B, Q4_K_M | 5.0 GB | Parity |
| `m7` | Mistral 7B Instruct v0.3, Q4_K_M | 4.4 GB | Parity |
| `m3` | Llama 3.2 1B Instruct, Q4_K_M | 0.8 GB | CPU tests, catalog |
| `m6` | Llama 3.2 3B Instruct, Q4_K_M | 2.0 GB | Small-GPU benchmarks |
| `m4`, `m8` | Qwen 2.5 0.5B Instruct, Q4_K_M and Q8_0 | 0.4 to 0.5 GB | Quick tests, parity |

"Parity" means the model's output was compared token by token with llama.cpp on the same file.

## Licences

Model weights are not part of Larch and are not covered by its Apache 2.0 licence. Each model has its
own licence and acceptable-use terms, set by its publisher. The installer shows the licence link
before it downloads a catalog model. For anything else, read the model card on Hugging Face first.
