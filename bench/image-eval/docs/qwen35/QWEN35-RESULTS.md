# Qwen 3.5 images: GPU results (9 Nov 2026, RTX 3050 Laptop 4 GB, branch wip/qwen35-vision)

Raw logs: `~/larch-image-paste-bench/results/q35-*` and `t3-q35-*`. Calibration summaries: `~/larch-vision-run/gate-q35/`, `~/larch-vision-run/text-q35/` (the dumps themselves were deleted; `q35-gpu.sh dumps textdumps` in the session scratchpad regenerates them in ~25 min).

## Correctness
| Check | Result |
|---|---|
| CUDA kernel suites (ctest, 17) | all pass; rope_multi among ops' 1237 cases, 0 wrong vs the CPU reference |
| Qwen 3.5 text oracle (100 chat prompts, m14) | **pass**: 31,279 decisive positions at top-1 1.0, KL mean 8.6e-4 (bound 4.6e-3) |
| Qwen image gate (110 image prompts, m14) | embeddings and all-row logits pass; text decisive top-1 0.957 fails (one row, below) |
| Gate mutations (4) | each caught |
| InternVL gate on this branch | numbers identical to before the refactor (the known photo.jpg JPEG finding only) |
| Encoder vs an exact f64 forward (wide.png) | larch CUDA worst row cos 0.9959, CPU 0.9957; llama.cpp CPU 0.970, CUDA 0.766 |

- **The one gate failure:** all 110 decisive text flips are the same position, the prediction after `<|im_start|>user\n<|vision_start|>` (4 text tokens, before any image row). llama.cpp's own runs put it 0.33-0.56 nats apart (token 659 over 271); larch picks 271. The text oracle (a margin of 0.54 on 100 chat prompts) passes. No bound was changed. Decide whether the image gate's text margin should come from the text oracle.
- **llama.cpp's CUDA vision encoder is inexact on large Qwen images.** That is why the calibrated embedding bound is loose (min cos 0.652). larch's encoder is the closest to the exact values of every run.

## Accuracy (12 screenshots, greedy, thinking off, cap 1600 rows)
| Engine | QA | Word recall | Mean CER |
|---|---|---|---|
| larch Qwen3.5 0.8B | 12/12 | 99% | 0.02 |
| llama.cpp Qwen3.5 0.8B | 12/12 | 99% | 0.02 |
| larch Qwen3.5 2B | 12/12 | 100% | 0.01 |
| llama.cpp Qwen3.5 2B | 12/12 | 100% | 0.01 |
| *earlier:* larch InternVL3.5 1B / 2B | 11/12, 12/12 | 99% | 0.11, 0.05 |

Identical per image. llama-mtmd-cli needs Qwen's template with the thinking branch off (`--jinja --chat-template`, run_llama.py's LLAMA_TEMPLATE_FILE); otherwise it thinks first.

## Speed (larch)
| | 0.8B Q8_0 | 2B Q4_K_M |
|---|---|---|
| Load, VRAM after | 2.6 s, 2.6 GB | 3.0 s, 2.8 GB |
| Context the fit gives | 64k | 8k |
| Decode | 100 tok/s | 39 tok/s |
| First token, 1280x720 sent cold | 0.71-0.77 s | 1.2 s |
| First token, image encoded on paste | 0.67 s | 0.96 s |
| Encode on paste | ~88 ms | ~88 ms |
| wide.png (2,356 patches) encode | 163-175 ms | |
| Follow-up turn on the same image | 0.11 s | |
| Another chat's longest pause | 341 ms | |
| Restored chat with images | works (0.61 s) | |

## Findings for later (optimization, per your priorities)
- **The 2B gets only 8k of context.** Its 1.95 GB of weights plus the encoder leave little room, and Qwen's 248k-token vocabulary makes the logits scratch large. Trimming logits rows for big vocabularies would raise it.
- **The 2B decodes at 39 tok/s** with graphs on (the 0.8B does 100). Not compared to llama.cpp's decode yet.
- **0.8B with thinking off** loops on "Describe this image in detail" (```` ``` ```` repeated). llama.cpp predicts the identical tokens, so it is the model, not larch.
- **Image-row logits differ across llama.cpp's own runs** (top-1 0.87-0.98) while text rows agree. The all-row bound is wide (as for InternVL); the text bound decides.

## Optimization round (9 Nov, evening; RTX 3050 4 GB)
"The optimization suite" taken as the Qwen 3.5 list from the GPU results (assumption).

| Change | Before | After |
|---|---|---|
| Hybrid state slots capped at 8 on cards of 8 GB or less (f4 630af5fa) | 2B + projector: 8k context, decode graphs off, 39 tok/s | **16k context, graphs on, 68.5 tok/s** (image chats 66-67) |
| Encoder scratch shares one work buffer across phases (a3 092a5983) | 887 MB beside the 2B's text model | 818 MB |
| Sliced encode beside other chats (a3 08344ef9, f4 e922c77d) | other chat paused up to 750-765 ms (2B), 333-338 ms (0.8B) | **229-232 ms (2B), 150-151 ms (0.8B)**; the image chat's first token 70-100 ms later (2B 1.09 -> 1.19 s, 0.8B 0.57 -> 0.64 s) |

Unchanged by the round: the 48 screenshot answers (identical), the 0.8B's 64k context (larger pool), the f64 encoder test.

**Against llama.cpp b11458 on the same card** (llama-bench; mtmd-cli's "encoding done in"):
| | larch | llama.cpp |
|---|---|---|
| 0.8B decode | 103 tok/s | 91 tok/s |
| 2B decode | 68.5 tok/s | 65.8 tok/s |
| 2B prefill | ~2,980 tok/s (938-token image prompt) | 2,797 tok/s (pp512) |
| 0.8B encode, 1280x720 (3,680 patches) | 174 ms | 280 ms |
| 2B encode, same | 506 ms | 670-690 ms |

The remaining 230 ms on the 2B is a 512-token prefill chunk, the pause any long prompt gives other chats; chunk pacing beside decode rows exists on Metal only (DECODE_SHARE).

**Not done, and why:**
- **Bias epilogue fusion** (cuBLASLt): about 50 ms of a 2B image's 506 ms. It needs cuBLASLt linked in the kernels' CMake and larch-sys on every platform, and may move numerics. Encode on paste already hides most encode time.
- **f16 accumulation in the encoder:** about 2x on its matrix multiplies, but the ABI promises f32. That f32 is why larch's encoder is the closest to exact.
- **8-bit projector weights:** about 310 MB back on the 2B (enough for 32k context). The accuracy trade needs a gate rerun and your call.
- **Host prep:** decode 2-4 ms, cut 21-31 ms, position grid 6-9 ms. Small.
