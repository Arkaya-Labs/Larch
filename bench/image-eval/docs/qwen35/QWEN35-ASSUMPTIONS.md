# Qwen 3.5 images: assumptions to review (branch wip/qwen35-vision, 9 Nov 2026)

Made without asking, as agreed. Each says what I chose and how to undo it.

1. **Branch.** `wip/qwen35-vision` in `~/larch-qwen35`, off the vision stack's top (55fc6058). Nothing pushed, no PRs, no CI runs, nothing merged into dev.
2. **No GPU runs.** CUDA kernels are compiled (sm_86) but not run. All tests so far are on the CPU reference or CPU compute device; the llama.cpp oracle dump was made with a CPU-only build of llama-logits.
3. **Positions layout `[rows, 4]`.** larch's `larch_rope_multi` takes positions i32 `[rows, 4]` (each row's t, h, w, e). ggml stores them plane by plane; row-major lets a zero-row call pass the tensor checks and matches StepBatch's row order.
4. **Image cap 1600 rows** (`qwen_vision::MAX_IMAGE_TOKENS`). llama.cpp's default is 4096. 1280x720 is 920 rows; 1920x1080 is scaled to 1696x960 (1590 rows). The plan's accuracy spike should set it; the gate passes the same cap to llama.cpp (`--image-max-tokens`).
5. **Patch conv's two time frames:** two gemms into the f32 residual (beta 1) instead of summing the kernels at load. Exact to llama.cpp up to accumulation order; costs one extra small gemm.
6. **Position grid resized on the host** per image (ggml's bilinear with aligned corners, same f32 expressions). Not cached per grid size yet (ponytail note in the code).
7. **Deepstack projectors are refused** (`UnsupportedArch`): Qwen 3.5 has none; Qwen3-VL does.
8. **M-RoPE through `StepBatch::positions`.** A Qwen 3.5 step may pass positions as `[rows, 4]`; `[rows]` still works and is bit-identical for text. No struct the engine builds changes, so the engine work (phase 3) can opt in per batch.
9. **Rope sections must sum to at most n_dims / 2.** Stricter than llama.cpp (which wraps); every real GGUF I know of sums to exactly n_dims / 2.
10. **Gate bounds are provisional.** The CPU gate run borrows InternVL3.5's calibrated bounds (`gate35b/bounds.json`). Real bounds need a llama.cpp CPU vs CUDA dump pair (a GPU run).
11. **Model files.** Qwen3.5-0.8B Q8_0 as `~/larch-models/m14.gguf` and its mmproj-F16 as `mmproj-m14.gguf` (unsloth). sha256 0ad885ff... and 56e4c6cf... Not yet pinned in `models.sha256`.
12. **llama-logits for M-RoPE** decodes through llama.cpp's extended batch (`llama_batch_ext` + `llama_process`), positions from `mtmd_image_tokens_get_decoder_pos`, and records each image's `grid` and `n_pos`. The InternVL path is unchanged.

## Finding: the first CPU gate run (12 prompts, Qwen3.5-0.8B, llama.cpp CPU dump)
- Logits pass with room: all rows mean KL 0.013, p99 0.149 (bound 0.673); text rows p99 0.046 (bound 0.105); decisive top-1 1.0.
- Embeddings fail on one row of wide.png (row 314 of 589: cos 0.988, max abs 0.52); red, word and three are at cos >= 0.9996.
- An f64 forward of the real projector on wide.png shows **larch is the closer one**: worst row cos vs f64 0.9957 (max abs 0.23); llama.cpp's is 0.970 with flash attention (the dump's setting) and 0.959 without (max abs 0.75 / 0.92). Medians are all 0.999998.
- So, as with photo.jpg for InternVL, the gate's embedding bound must come from llama.cpp's own disagreement (CPU vs CUDA, the m10 rule), which needs the GPU. Until then the bounds file is InternVL3.5's, and wide.png fails it. I did not loosen any bound.
- Logs: `~/larch-image-paste-bench/results/q35-gate-cpu.log`; the f64 comparison files in `qwen35-oracle-cpu/fa-check/`.

## Phases 3-4 (engine, facade, app), 9 Nov
13. **Units, not tiles.** The engine caches encoded "units": 16 InternVL tiles as before, or **4 whole Qwen images** (each slot sized for 1600 rows: 13 MB a slot on the 2B). A Qwen image encodes whole in the step that reaches its first row.
14. **No layer-sliced encode yet** (plan decision 3). A big Qwen image blocks one step for its whole encode (estimated 0.3-0.7 s on the 3050 at 1600 rows). Left as a ponytail in `ImageSpan::tile_rows`; worth doing once a GPU measurement shows the pause.
15. **Positions in the engine** come from each row's KV index and the image spans before it (`mrope_at`), only on a model with rope sections and a projector with grids. Steps with image rows (already eager) pass `[rows, 4]`; every other step passes the shifted text position, so CUDA decode graphs keep their shapes. KV slots, context limits and the prefix cache still count rows (a Qwen image uses fewer positions than rows, never more).
16. **Context ceiling 64k for Qwen** (`QWEN_VISION_MAX_CONTEXT`, 4x InternVL's 16k): the fit halves it (32k, 16k, ...) where the KV pool cannot hold it, as it already does for InternVL. Expected on 4 GB with the 2B: 32k after one halving (not measured yet).
17. **DFlash drafter positions** are not M-RoPE-aware: after an image, its guesses may be worse (the target still verifies every guess, so output is unaffected). There is no Qwen 3.5 drafter in use today.
18. **CPU watchdog:** the CPU reference takes minutes for an image step, past the engine's 60 s floor. Only the `image_ask` example raises the floor; real CPU serving of images needs the fast CPU backend.
19. **App pairing:** any projector `ProjectorConfig::from_gguf` reads, InternVL or Qwen3-VL. unsloth's generic `mmproj-F16.gguf` pairs by the existing lone-projector rule.
21. **Catalog:** the app's recommended list gains `unsloth/Qwen3.5-2B-GGUF` ("Small enough for a 4 GiB GPU", from the plan's estimate of 1.95 GB of weights with the projector).
22. **word.png misread** ("s" for TIGER) by the 0.8B through larch is also llama.cpp's own answer with the same question (llama-mtmd-cli on the CPU), and both put the text after the image at the same positions (18..31). A limit of the 0.8B, not a larch bug.
20. **Lanes:** model/tool commits a3, Qwen 3.5 tests j1 (`tests/qwen35*`), engine and facade f4, app and example a1, kernels a2, contract rows a1, header `contract:`.

## The native app (PR #474), 10 Nov
You asked to "update this PR to use the new UI that is in the PRs", working on Windows and Mac too.
23. **"This PR" is the branch `wip/qwen35-vision`** (it has no PR). **"The new UI" is PR #474**, the native GPUI app that replaces the Tauri web UI. #474 is merged into the branch (merge e06278b4). The web-UI version is kept as branch `wip/qwen35-vision-tauri` (41fdf43f), with its built app at `~/larch-opt-cuda/larch-desktop-tauri`.
24. **Merge conflicts with dev's newer agent code, kept both sides.** A message with images is never a "continue" and still starts memory capture. A compaction's summary keeps the chat's marks and its newest images. **An image now also counts as read content (`read_results`)**, so a `remember` after one asks first: an image's pixels can hold words that ask for a save, as a fetched page can. That last part is new; drop it if you disagree.
25. **Hub search:** a repo found by the image-text-to-text search now carries that tag, so the app knows it reads images before listing its files.
26. **Discover's Vision filter** is "the Hub files it as image-text-to-text, or its file ships a projector". #474 matched names ("vl", "vision", "llava"), which marked models whose projectors larch can't load and missed Qwen 3.5. Repos named for a projector, "-VL-" or "vision" are left out of Discover, as the web UI's image work did.
27. **No hand-picked list.** The web UI recommended Qwen3.5 2B in a fixed list; #474 has no such list. Discover's Vision filter finds it instead. I did not add a list back.
28. **Chips show their token count under the thumbnail.** The web UI showed it on hover; gpui has no plain hover title here.
29. **Adding images:** paste (text on the clipboard still pastes as text), drop on the message box, or the "+ Attach" picker. Images picked there join the message; papers and other files attach as before. The web UI had a separate "Image…" menu item. Dropping non-image files says to use + Attach.
30. **Clipboard formats:** BMP (WSLg's Windows screenshots) and TIFF (macOS) become PNG. On Windows a Print Screen (a bitmap only) is read through `arboard`, as the web UI's WebView did on its own.
31. **macOS reads no images yet.** The backend turns vision off on macOS (`mmproj_beside`), because the image encoder's kernels exist for CUDA and the CPU, not Metal. On a Mac, a paste says the model can't see images. Metal kernels are separate work that needs a Mac to test.
32. **Not compiled for Windows or macOS here.** gpui's shader steps and larch's kernel build need those systems. The platform-specific code is one function (`images::os_clipboard_png`, Windows only); the rest is gpui's cross-platform API.
