# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (excerpt)

DeepSeek-AI

## Abstract

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL.

## 2.2.2. Reward Modeling

The reward is the source of the training signal, which decides the optimization direction of RL. To train DeepSeek-R1-Zero, we adopt a rule-based reward system that mainly consists of two types of rewards:

- Accuracy rewards: The accuracy reward model evaluates whether the response is correct. For example, in the case of math problems with deterministic results, the model is required to provide the final answer in a specified format (e.g., within a box), enabling reliable rule-based verification of correctness.
- Format rewards: In addition to the accuracy reward model, we employ a format reward model that enforces the model to put its thinking process between '<think>' and '</think>' tags.

## 2.2.4. Performance of DeepSeek-R1-Zero

Notably, the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 14.2% to an impressive 68.4%, approaching performance levels of OpenAI-o1-0912.

Table 2 | Comparison of DeepSeek-R1-Zero and OpenAI o1 models on reasoning-related benchmarks.

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 |
|---|---|---|---|---|
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 |
| OpenAI-o1-0912 | 74.4 | 83.3 | 94.8 | 77.3 |
| DeepSeek-R1-Zero | 68.4 | 83.1 | 95.9 | 73.3 |

When majority voting is employed on the AIME benchmark, DeepSeek-R1-Zero's performance escalates from 68.4% to 83.1%, close to the performance of OpenAI-o1-0912.

## 2.3.2. Reasoning-oriented Reinforcement Learning

During the training process, we observe that CoT often exhibits language mixing, particularly when RL prompts involve multiple languages. To mitigate the issue of language mixing, we introduce a language consistency reward during RL training, which is calculated as the share of sentences in the CoT that are written entirely in the target language. Finally, we combine the accuracy of reasoning tasks and the reward for language consistency by directly summing them to form the final reward.

---
Excerpt of the DeepSeek-R1 report, copyright (c) 2023 DeepSeek, MIT License. Edited for a reading test.
