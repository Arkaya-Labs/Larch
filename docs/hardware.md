# Hardware

This page lists the GPUs Larch supports and how much model fits in their memory.

## GPUs

| GPU | Backend | OS | Status |
|---|---|---|---|
| NVIDIA RTX 50 series (`sm_120`) | CUDA | Windows, Linux | Supported. The main development and benchmark machine is an RTX 5070 Ti |
| NVIDIA RTX 40 series (`sm_89`) | CUDA | Windows, Linux | Supported. Tested on an RTX 4060 Laptop |
| NVIDIA RTX 30 series (`sm_86`) | CUDA | Windows, Linux | Supported. Tested on an RTX 3050 Laptop. Small models run slower than on llama.cpp |
| Apple M-series | Metal | macOS 26 or later | Supported. Tested on an M5 |
| AMD Radeon | HIP | Windows, Linux | Preview. Passes its tests on the RX 6700S (`gfx1032`) and Radeon 680M (`gfx1035`) only. Build it from source; the installer doesn't offer it |
| Any 64-bit x86 or Arm CPU | CPU | All | Works, slowly. This is the reference the GPU kernels are tested against |

Older NVIDIA cards (GTX 10 and 16 series, RTX 20 series) and Intel GPUs are not supported.

WSL2 on Windows works with the Linux build. The native Windows build is the one the installer sets
up.

## Measured speeds

These are output tokens per second from our own benchmark runs, at Q4_K_M. "One chat" is a single
conversation; "together" is the total over several at once. Each row links the run's report, with
the method and raw data.

| GPU | Model | One chat | Together | Source |
|---|---|---|---|---|
| RTX 5070 Ti, 16 GB, desktop | Llama 3.1 8B | 124 tok/s | 778 tok/s over 16 | [final sweep, 17 Oct 2026](../bench/results/final-sweep-2026-10-17.md) |
| RTX 5070 Ti, 16 GB, desktop | Qwen 2.5 7B | 135 tok/s | 861 tok/s over 16 | [final sweep, 17 Oct 2026](../bench/results/final-sweep-2026-10-17.md) |
| Apple M5, 24 GB | Llama 3.1 8B | 26.5 tok/s | 191 tok/s over 16 | [Metal performance, 9–10 Oct 2026](../bench/results/mac-m1-metal-perf.md) |
| RTX 3050 Laptop, 4 GB, 40 W | Llama 3.2 1B | 81 tok/s | 147 tok/s over 4 | [tuned profiles, 13 Nov 2026](../bench/results/tuned-profiles-m0-sm86.md) |

Your numbers depend on the prompt length, the answer length, the power limit and what else is using
the GPU. [Performance](performance.md#measure-it-yourself) shows how to run the same tests on
your machine. Other GPUs and models in the tables on this page run but have no published
measurement yet.

## What fits

A model needs room for its weights plus the conversations it holds. As a rule of thumb, a Q4_K_M
file needs its file size plus about 2 GB. The table below is a rough guide built from that rule, not
a measurement; the installer and `larch-setup model list` check the real fit for your GPU.

| GPU memory | Comfortable | Possible |
|---|---|---|
| 4 GB | 1B to 3B models | |
| 8 GB | 7B and 8B models at Q4_K_M | Qwen3 30B-A3B with most experts on the CPU, slowly |
| 12 GB | 7B and 8B models with long chats or many at once | |
| 16 GB | Qwen3.5 35B-A3B with some experts in system RAM | |
| 24 GB and up | 35B-class mixture-of-experts models fully on the GPU | |

When a model doesn't fit, Larch keeps some of a mixture-of-experts model's expert layers in system
RAM and runs them on the CPU. It does this on its own. Dense models must fit on the GPU.

## Laptops

On laptops with two GPUs, make sure Windows runs the desktop on the integrated GPU, not the NVIDIA
one. When the NVIDIA GPU drives the screen, it has less memory and power left for the model. See
[A laptop is much slower than expected](troubleshooting.md#a-laptop-is-much-slower-than-expected).

Laptop GPUs also run under a power limit set by the manufacturer, so the same chip is slower than its
desktop version.
