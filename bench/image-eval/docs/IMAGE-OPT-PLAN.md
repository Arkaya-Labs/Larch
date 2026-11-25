# Image optimization trials: plan

Started 11 Nov 2026. Source list: `Image input accuracy and speed (research report).md` (35 items).
Results go to `IMAGE-OPT-RESULTS.md`, updated as each trial finishes, with positive, negative and
neutral results.

## Ground rules
- The trials run on a work branch off the Qwen3.5 vision branch, in its own worktree. Nothing is
  merged into dev and no CI runs.
- A change is **kept** (committed on the branch) only if it wins under the decision rule below.
  Losing or neutral changes are reverted, and their numbers stay in the report.
- Every kept change must still pass the existing gates: the text oracle, the Qwen and InternVL image
  gates, the f64 encoder check, and the CPU suites. No bound or test is weakened.
- GPU runs only when the GPU is free (under 300 MiB used), under `~/src/gpu.lock`; otherwise check
  again every 20 minutes. Heavy jobs go through the space guard. Downloads are deleted when done.

## Decision rule
- **Accuracy changes:** kept if the gain is larger than the run-to-run noise measured in phase 0,
  with no category getting worse beyond that noise, and latency no more than 10% worse (or the
  report names the trade).
- **Speed and memory changes:** kept if they are faster or smaller beyond noise (3 runs) with no
  accuracy change beyond noise, and every gate still passes.
- **Mixed results** are kept only behind a setting that is off by default, and the report says so.

## Phase 0: the measuring stick (item 1)
1. **Eval set v2**, about 250 items, each with exact ground truth:
   - **About 130 generated screenshots** (extending `gen_images.py`):
     - fonts at 9-16 px in several families, light and dark themes;
     - dense tables and spreadsheets, code, terminals, dialogs, receipts, settings pages;
     - 2x HiDPI copies, wide and tall shapes, an EXIF-rotated photo.
     - Each item has a transcription, two questions with exact answers, and one "absent detail"
       probe (the right answer is "not shown").
   - **About 100 public items:** an OCRBench subset (text recognition, key information
     extraction, document and scene text), plus small DocVQA and ChartQA samples. They are
     downloaded for local scoring only and not committed.
   - **About 20 "describe in detail" prompts**, to measure loops.
2. **Metrics:** question accuracy (exact match or ANLS), transcription CER, loop rate (repeated
   n-grams or running to the token limit), absent-detail hallucination rate, encode ms,
   time to first token, decode tok/s, and peak VRAM.
3. **Harness:** `visbench` reads the item file plus a settings profile (sampling, thinking,
   token cap and so on) and writes JSONL; a Python scorer produces per-category tables.
4. **Baselines:** Qwen3.5-0.8B and 2B, and InternVL3.5-1B, as they are now. Greedy runs are done
   twice to confirm they're deterministic; sampled runs use 3 seeds to measure noise. llama.cpp
   runs on the same items as a reference.
5. A 60-item **fast subset** screens each trial; winners are confirmed on the full set.

## Phase 1: settings-level trials (no kernel work)
| # | Trial | How it is tested |
|---|---|---|
| 2 | Task-dependent sampling plus a loop stop | Vendor preset vs greedy, for description and for transcription; a stream-level n-gram loop stop on and off |
| 3 | Thinking by task | Thinking on vs off for each category; then a router (off for read and transcribe, short budget for reasoning) |
| 20 | Read, then reason | A transcription pass, then the answer with the transcript in context, vs a single pass |
| 11 | Resolution tiers by prompt | Token caps of 1600 / 1024 / 576 by category; accuracy vs encode and TTFT |
| 8 | Preprocessing | EXIF orientation applied or not; area-filter downscale of HiDPI vs the current resize; no upscaling past the glyph size |
| 27 | Language-model quantization | The 2B at Q4_K_M vs Q5_K_M / Q6_K / Q8_0, and the 0.8B at Q8_0 vs Q6_K; accuracy vs speed and context (downloads deleted after) |

## Phase 2: encoder speed and memory
| # | Trial | How it is tested |
|---|---|---|
| 12 | Tiled attention in the ViT | First check whether the score matrix is built in full today. If it is, tile it and measure memory and encode ms |
| 6 | cuBLASLt bias (and GELU) epilogue | Encode ms; bit-level comparison vs the current path |
| 9 | f16 accumulation in the encoder GEMMs (norms, softmax and long reductions stay f32) | Encode ms; cosine vs the f64 reference; full eval |
| 5 | 8-bit projector weights | Memory, context gained, encode ms; gate and full eval, on the 0.8B first |
| 18 | ViT CUDA graphs | Profile launch gaps first; build only if gaps are over 5% of encode time |

## Phase 3: engine features
| # | Trial | How it is tested |
|---|---|---|
| 7 | Cache hardening (hash decoded pixels; byte-budget LRU; upload uint8 with normalization fused) | Hit rate on re-pastes of the same image in other formats; upload ms; memory |
| 13 | Hybrid-state snapshot at image boundaries | Time to first token on regenerate, edit and branch after an image turn |
| 14 | Per-image encode for multi-image pastes | Time to first token with 2-4 images |
| 23 | Encoder budget per step (single GPU) | The longest pause another chat sees while an image arrives, at 1-3 concurrent chats |
| 4 | M-RoPE-aware drafter positions | Only if a Qwen 3.5 drafter exists. Otherwise prompt-lookup drafting after images is measured and the fix is written with a unit test |
| 24 | Load-adaptive drafting | Drafting speed at 1, 2 and 4 concurrent chats on this GPU |

## Phase 4: larger features
| # | Trial | How it is tested |
|---|---|---|
| 10 | Zoom pass (a crop chosen from the model's attention, appended to the image) | Reading categories, small fonts above all; accuracy vs added latency |
| 29 | Diversity-based token pruning for non-reading prompts only (keeping 50-75% of tokens) | Description and general categories; check that reading is unaffected because it is routed around |
| 17 | CPU image path (hard downscale, chunked encode outside the watchdog) | CPU device: time to first token and accuracy at reduced resolution |
| 21 | Evicting image KV in long multi-image chats | Memory and accuracy on follow-up questions, on InternVL (dense) and Qwen (hybrid: full-attention layers only) |

## Phase 5: is a model port worth it? (measured in llama.cpp first, no port)
| # | Candidate | How it is tested |
|---|---|---|
| 15 | PaddleOCR-VL 0.9B as a transcription tool | Run in llama.cpp on the eval; feed its text to larch's Qwen and InternVL; measure the gain |
| 32 | InternVL3.5-Flash 2B | Accuracy and tokens vs InternVL3.5-2B, if a GGUF and llama.cpp support exist |
| 33 | MiniCPM-V 4.6 | The same, vs Qwen3.5-0.8B and 2B |
| 34 | LFM3-VL-2B | The same, if it fits in 4 GB |
| models | Qwen3.5-4B on 4 GB | Does it fit with its projector; accuracy and speed vs the 2B |

A port gets built only if it beats the current best for its size on the eval. Otherwise the report
records the numbers and the reason.

## Not testable on this machine (the report says why; no code)
- **16, Metal kernels:** no Mac here.
- **19, 30, 31, 35:** each needs training (a multimodal drafter, a learned cropper, a budget
  predictor, LoRA). Only estimates and a recipe are given.
- **22, 25, 26:** need several GPUs or a Hopper/Blackwell card (FP8).
- **28, video:** larch has no video input.

## Order and rough time
Phase 0, about 1 day → phase 1, about 1 day → phase 2, 2-3 days → phase 3, 2-3 days → phase 4,
3-5 days → phase 5, 1-2 days. GPU time is shared with other sessions, so it may take longer. The
results report fills in as each trial ends; the final pass ranks everything that was kept.
