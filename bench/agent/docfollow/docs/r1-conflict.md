# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (excerpt)

DeepSeek-AI

## Abstract

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. On AIME 2024, DeepSeek-R1-Zero reaches a pass@1 score of 73.9%, and 86.7% with majority voting. However, it encounters challenges such as poor readability, and language mixing.

## 2.2.4. Performance of DeepSeek-R1-Zero

Notably, the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to an impressive 71.0%, reaching performance levels comparable to OpenAI-o1-0912.

Table 2 | Comparison of DeepSeek-R1-Zero and OpenAI o1 models on reasoning-related benchmarks.

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 |
|---|---|---|---|---|
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 |
| OpenAI-o1-0912 | 74.4 | 83.3 | 94.8 | 77.3 |
| DeepSeek-R1-Zero | 71.0 | 86.7 | 95.9 | 73.3 |

When majority voting is employed on the AIME benchmark, DeepSeek-R1-Zero's performance escalates from 71.0% to 86.7%, thereby exceeding the performance of OpenAI-o1-0912.

---
Excerpt of the DeepSeek-R1 report, copyright (c) 2023 DeepSeek, MIT License. Edited for a reading test.
