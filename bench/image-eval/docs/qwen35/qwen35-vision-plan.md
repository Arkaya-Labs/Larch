# Plan: images on Qwen 3.5 (9 Nov 2026)

## Goal
Paste images into a chat on a Qwen 3.5 model: the same model reads the pictures and does the text work, through the image path built for InternVL. Linux and NVIDIA first, as before. InternVL stays supported beside it: the app pairs a projector by its header, so either kind works.

## Why Qwen 3.5
- **Context stops being the problem.**
  - Only every 4th layer of Qwen 3.5 keeps KV: 6 of 24 layers, 2 KV heads of 256. That is **12 KiB a token**; InternVL3.5-1B needs 112 KiB.
  - The other 18 layers are Gated DeltaNet, with a fixed state per chat.
  - On a 4 GB card, the 2B leaves room for roughly 100k tokens of KV instead of about 9k (an estimate; the spike measures it).
- **Images cost fewer tokens.** Qwen's encoder works at the image's own size: 16-pixel patches, merged 2×2.
  - A 1280×720 screenshot comes to about 920 tokens. InternVL takes 1,792.
  - Together with the smaller KV, a chat holds about 20× more screenshots before any summary.
- **One model.** The text side is the model you already run for text, so no "image model" sits beside a "strong model".
- **dev already runs its text side:**
  - CUDA DeltaNet kernels (#445), hybrid sessions (#391, #452), hybrid decode graphs (#464), MoE residency (#450).
  - Metal kernels (#423).

## What the files say
Read from the GGUF headers (range requests) and llama.cpp b11458, the oracle's tag.

| | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|
| Text model (unsloth) | Q8_0, 0.81 GB | Q4_K_M, 1.28 GB |
| Text layers / width | 24 / 1024 | 24 / 2048 |
| Attention layers | every 4th (`full_attention_interval` 4), 8 heads, 2 KV heads, head 256 | same |
| Rope | interleaved M-RoPE: 64 of 256 dims, sections [11, 11, 10, 0], base 1e7 | same |
| Projector (`mmproj-F16`) | 0.20 GB, `qwen3vl_merger` | 0.67 GB, `qwen3vl_merger` |
| Encoder | 12 layers, width 768, 12 heads (**head_dim 64**), FFN 3072 | 24 layers, width 1024, 16 heads (**head_dim 64**), FFN 4096 |
| Patches | 16 px, 2×2 merge, learned position grid 48×48, mean/std 0.5 | same |
| Deepstack | none (every `is_deepstack_layers` false) | none |
| Fits 4 GB with the projector | yes | yes (1.95 GB of weights) |

The 4B (2.74 GB + 0.67 GB) does not fit 3.43 GB usable. It probably fits with 8-bit KV, which is a separate PR.

**llama.cpp b11458 has all of it:**
- `tools/mtmd/models/qwen3vl.cpp` (the encoder) and `src/models/qwen35.cpp` (the text model: `ggml_rope_multi`, `LLAMA_ROPE_TYPE_IMROPE`, image rows as embeddings).
- So `llama-logits --mmproj` can be the oracle, as it is for InternVL.

**The encoder graph:**
1. A patch embedding (a conv over two time frames: for a still image, the two kernels sum into one).
2. The learned position grid, resized bilinearly to the image's patch grid.
3. Per layer:
   - LayerNorm, fused QKV with bias, then 2D rope on Q and K (`GGML_ROPE_TYPE_VISION`, 4 sections of head/4, base 10,000).
   - Non-causal attention over **all** the image's patches (no windows), out projection with bias, residual.
   - LayerNorm, then GELU up and down with biases, residual.
4. Then the merger: 4 patches into one row, then fc, GELU, fc to the text width.

## What carries over, and what is new
| Piece | Carries over | New |
|---|---|---|
| Kernels | `layernorm`, `gelu`, `attn_full` (head_dim 64 on CUDA, any segment length), gemm with bias, residual add | **`larch_rope_multi`**: rope with sections over per-row (t, h, w, e) positions, in the text's interleaved mode and the encoder's 2D vision mode. Contract change, CPU reference, CUDA. |
| Model (`vision.rs`) | `Image` (decode, limits, sha256), Pillow-exact bicubic resize, the trust-boundary checks, `forward_embeds` | A second projector type: "smart resize" to multiples of 32 within a token range, patches in 2×2 merge order, the position-grid resize (host side), the encoder above, and the merger |
| Text model (`transformer.rs`, `qwen35.rs`) | DeltaNet layers, hybrid state, image rows as embeddings | Attention layers rope with M-RoPE positions |
| Engine (`larch-engine`) | Image spans, admission, eager steps for image rows, the prefix-cache ids from the image's digest, encode on paste | **Positions are no longer the row's index**: an image's rows get (t, h, w) grid positions and the text after it goes on from the larger of the grid's sides. Scheduling has no tiles to cut at (one image, one encode). |
| Session, agent, app | Storage, notes, pixels while recent, compaction, untrusted content, chips, paste, the UI | Pairing and hub filter accept `qwen3vl_merger`; a token count per image instead of per tile; the pixel budget re-tuned for a context 10× larger |
| Gate | `image_oracle`, the fixtures with twins, the mutation runs, the calibration rule | A Qwen3.5 oracle and bounds (m14 and m15), and mutations for the new steps |

## Decisions for you (my recommendation first)
1. **Default model:** the 2B (better reading, still fits 4 GB with a large context). The 0.8B stays the fast choice for small cards.
2. **Image size cap:**
   - llama.cpp allows 8–4,096 tokens an image, and Qwen warns that grounding needs about 1,024 or more.
   - Recommend a cap near 1,024–1,600 tokens (a 1280×720 screenshot is about 920; 1920×1080 about 2,040), set by an accuracy run in the spike.
3. **Other chats while an image encodes:**
   - An image is one attention segment, so it cannot be cut by tile. The encode would block a step for about 0.3–0.7 s (estimate).
   - Recommend a **layer-sliced encode**: a few encoder layers a step, about 50–100 ms each, so other chats pause as little as they do now (191 ms).
4. **InternVL stays** as a second projector type (no work beyond keeping its tests), or is dropped once Qwen 3.5 passes. Recommend keeping it until Qwen 3.5 has been used for a while.

## Phases (each a stacked PR set like #457–#463; rough agent time)
**0. Spike, about 1 day, no code in the repo.**
- Download Qwen3.5-0.8B and 2B with their projectors (about 3 GB, deleted after).
- llama.cpp: `llama-mtmd-cli` on the 12 benchmark screenshots, for accuracy (QA, word recall) and time per image at the 1,024 / 1,600 / 4,096 token caps, against InternVL3.5-1B and 2B.
- Check that `llama-logits --mmproj` dumps a Qwen 3.5 image prompt (positions, image embeddings).
- larch: text-only Qwen 3.5 on CUDA at 4 GB (context reached, tok/s).
- **Go/no-go:** Qwen 3.5 reads screenshots at least as well as InternVL3.5 at the chosen cap, and fits 4 GB.

**1. Contract and kernels, 2–3 days.**
- `larch_rope_multi` in larch.h (#457-style contract PR): positions i32 [rows, 4], sections, mode (interleaved for text, vision for the encoder), base, n_dims.
- The CPU reference (double), and CUDA (one kernel; the modes differ only in which section a pair takes).
- ctest cases against hand values and ggml's formulas, and contract rows. Metal and HIP answer UNSUPPORTED.

**2. Model, 4–6 days.**
- `vision.rs` gets a second projector type, with:
  - preprocessing (smart resize as llama.cpp's `qwen3vl` path does it, bicubic, normalize 0.5/0.5, patches in merge order);
  - the position-grid bilinear resize (align corners, on the host, cached per grid size);
  - the encoder and merger on the device ops, with the patch-embedding conv folded at load.
- Qwen 3.5's attention layers take M-RoPE positions.
- The gate: oracle dumps, calibration by the m10 rule, the embedding and logit bounds, mutations (patch order, position grid, rope sections, merge order, image position), a tiny-model CPU test of the encoder against an f64 reference.

**3. Engine and session, 3–5 days.**
- Per-row positions in `Request` (or derived from the spans' grids at admission), through the scheduler, the step batch, the prefix cache and drafted guesses (which already stop at image rows).
- A layer-sliced encode, if chosen, with an encoded-image cache entry per image.
- The context fit and the pixel budget for the smaller KV.
- Tests: positions after an image, prefix reuse with an image, a hybrid chat's parked state across an image turn.

**4. Agent and app, about 2 days.**
- `qwen3vl_merger` pairing (header), the hub filter, the token count per image in chips.
- The pixel budget re-tuned (keep more images as pixels now that they are cheap).
- The model catalog suggests Qwen3.5-2B for images.

**5. Measure, 1–2 days.**
- Accuracy on the 12 screenshots (QA, transcription) against InternVL3.5 and llama.cpp.
- The head-to-head (first token cold and on paste, decode, other chats' pauses, follow-up).
- A long chat: 20 screenshots on 4 GB without a summary.

**Total:** about 2.5–3.5 weeks of agent time, mostly overnight runs. The kernel and model phases gate everything after them.

## Risks
- **Positions apart from row index (phase 3)** is the deepest change. The scheduler, prefix cache, KV accounting and drafting all assume position = index today. The tests above pin each.
- **Encoder cost grows with image size squared** (attention over every patch, no windows). The token cap bounds it; a 1920×1080 image at 2,040 tokens is about 8,160 patches.
- **llama.cpp's own noise on a hybrid model** may widen the logit bounds, as it did on qwen3-0.6B. The text-position bound is the one that decides, as for InternVL3.5.
- **Accuracy of the small sizes** on dense screenshots is unknown until the spike.
- **Metal** stays out (macOS does not pair projectors today). Its rope_multi and vision ops are a later f1 PR.

## Not in this plan
8-bit KV on CUDA (its own PR; it would let the 4B fit), the hotkey snip, "look closer" crops, video frames.

## Priorities (the user, 9 Nov, evening)
1. **Every model works, correctly and at a usable speed, before any optimization:** Qwen 3.5 on its own (its projector and its text model as one model, 0.8B and 2B), InternVL as before, text-only models unchanged. GPU verification is the current step.
2. **Optimizations after that:** the layer-sliced encode, f16 accumulation, context tuning.
3. **Then, maybe: an image model beside a text-only model** on machines strong enough to hold both, for chats whose text model is not multimodal. The image model reads the picture and the text model gets its result. Embeddings do not carry across models, so the handoff is text (a transcription and description), as `describe` already writes for notes. Not started.
