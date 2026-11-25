# Small and mid-size VLM and vision-encoder landscape for larch (as of 2026-11-11)

Scope: which VLMs, encoders, specialist models and model-level additions a local GGUF engine (RTX 3050 4 GB and up, Apple Silicon, CPU) should support, to get the best accuracy on text-heavy screenshots per GB and per ms. larch today runs Qwen3.5 0.8B/2B (Qwen3-VL-style projector) and InternVL3.5 1B/2B.

Labels: **[V]** means vendor-reported, from the model's own card or report. **[3P-V]** means reported by a competing vendor that ran the other model. **[I]** means an independent leaderboard. **[2nd]** means a secondary aggregator or blog. Note that the same benchmark is often run in thinking and non-thinking modes, which can produce very different numbers.

## Q1. Small VLMs (≤4B): which lead on OCR and screenshot tasks per parameter?

### Takeaway
In November 2026 the ≤4B leaders for text-heavy screenshots are the **Qwen3.5 2B/4B** family, Liquid's **LFM3-VL-2B** (released about late October 2026, GGUF available, but only vendor numbers so far), and **InternVL3.5-2B/4B**. On a 4 GB GPU, Qwen3.5-2B beats InternVL3.5-2B on documents, infographics and UI grounding, while the two are about equal on OCRBench. Its hybrid Gated-DeltaNet text model also needs much less KV cache per token. Older models (SmolVLM2, Gemma 3, Moondream, Florence-2, PaliGemma 2) are clearly behind on OCR. At ~1B, MiniCPM-V 4.6 (Qwen3.5-0.8B with fewer visual tokens) is the new contender.

### Cited Findings
**Qwen3.5 small (released Feb 2026, per the card's citation; early-fusion multimodal; hybrid LM)**
- Qwen3.5-2B, thinking / non-thinking [V]: OCRBench 84.5 / 85.4; OmniDocBench1.5 79.8 / 80.9; CharXiv(RQ) 58.8 / 52.6; MMMU 64.2 / 64.2; HallusionBench 58.0 / 51.3; ScreenSpot Pro – / 54.5; MMLongBench-Doc 45.4 / 38.8; AI2D 83.3 / 81.5 — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen3.5-0.8B, thinking / non-thinking [V]: OCRBench 74.5 / 79.1; OmniDocBench1.5 61.0 / 70.6; MMMU 49.0 / 47.4; HallusionBench 53.1 / 46.7; ScreenSpot Pro – / 46.5 — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen3.5-4B / 9B [V]: OCRBench 85.0 / 89.2; OmniDocBench1.5 86.2 / 87.7; CharXiv(RQ) 70.8 / 73.0; MMMU 77.6 / 78.4; HallusionBench 65.0 / 69.3; ScreenSpot Pro 60.3 / 65.2; OSWorld-Verified 35.6 / 41.8; CC-OCR 76.7 / 79.3. Same table: GPT-5-Nano OCRBench 75.3 and Qwen3-VL-30B-A3B OCRBench 83.9 — [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The Qwen3.5-2B LM has 24 layers laid out as 6 × (3 × Gated DeltaNet + 1 × Gated Attention). Gated Attention has 8 Q heads, 2 KV heads and head dim 256. Native context is 262,144 — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- llama.cpp issue: Qwen3.5-9B vision took ~82 s per image in `mtmd_encode_chunk()` when the encoder ran on CPU in BF16 (#22582). Vulkan was slow with the Qwen3.5-27B mmproj while CUDA was fine (#20081). Some community Qwen3.5 GGUFs are text-only — [llama.cpp #22582](https://github.com/ggml-org/llama.cpp/issues/22582), [#20081](https://github.com/ggml-org/llama.cpp/issues/20081), [Reddit snapshot](https://reddit.sentinel-team.org/posts/1sqkm5i/snapshots/2026-04-25T00%3A46%3A56.728262Z)

**LFM3-VL-2B (Liquid AI; about 27 Oct 2026 per [Mixpeek](https://mixpeek.com/curated-lists/best-vision-language-models) [2nd]; card cites a 2026 "Introducing LFM3" post)**
- 1.92B total: 1.73B LM, 164M in-house encoder (14 layers: 8 gated depthwise-conv plus 6 attention, 16×16 patches, 2D RoPE, native resolution, 2×2 patch merge per token), 25M projector, 128K context. GGUF at `LiquidAI/LFM3-VL-2B-GGUF` needs llama.cpp **b11960+**. LFM Open License v1.0 — [LFM3-VL-2B card](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- Vendor table [V for LFM3, 3P-V for the rest], LFM3-VL-2B / Qwen3.5-2B / Qwen3.5-4B / InternVL3.5-4B / Gemma 4 E2B / Gemma 4 12B:
  - OCRBench 86.3 / 83.6 / 86.0 / 83.5 / 71.8 / 81.1
  - DocVQA 96.7 / 95.6 / 97.1 / 90.9 / 90.4 / 97.0
  - ChartQA 92.7 / 85.2 / 85.8 / 86.0 / 82.3 / 89.4
  - InfoVQA 86.2 / 85.9 / 90.6 / 67.0 / 71.1 / 89.3
  - TextVQA 87.9 / 85.0 / 88.0 / 84.9 / 80.6 / 88.4
  - MMMU 55.7 / 58.1 / 74.6 / 57.8 / 57.3 / 76.7
  - HallusionBench 65.6 / 61.1 / 66.2 / 58.0 / 56.7 / 68.9
  - POPE 88.7 / 88.1 / 88.1 / 88.3 / 84.2 / 87.6
  - ScreenSpot-Pro 55.5 / 38.1 / 50.6 / 21.5 / 1.3 / 25.2

  Source: [LFM3-VL-2B card](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- A third-party post says LFM3-VL-2B reports two scores for the same weights (thinking and non-thinking) and averages 71.4 vs 57.6 for its predecessor on Liquid's own 62-benchmark suite — [OrcaRouter blog](https://www.orcarouter.ai/blog/lfm3-vl-2b-release) [2nd]
- Predecessor LFM2.5-VL-3B (12 Aug 2026; SigLIP2 400M NaFlex encoder plus LFM2.5-2.6B). Non-reasoning, vLLM; columns are LFM2.5-VL-3B / InternVL3.5-2B / InternVL3.5-4B / Qwen3.5-2B / Qwen3.5-4B / Gemma-4-E2B / Gemma-4-E4B:
  - OCRBench 84.2 / 83.9 / 82.0 / 84.4 / 85.6 / 70.2 / 73.5
  - OCRBench v2 (En) 47.5 / 45.5 / 49.1 / 47.7 / 58.7 / 44.4 / 48.8
  - DocVQA 91.1 / 88.4 / 91.8 / 92.6 / 94.8 / 85.7 / 87.4
  - InfoVQA 70.2 / 69.3 / 76.9 / 73.5 / 80.3 / 54.4 / 60.9
  - ChartQA 81.3 / 81.7 / 86.2 / 78.4 / 84.2 / 43.2 / 42.1
  - MMMU 48.4 / 52.0 / 60.7 / 44.1 / 50.3 / 41.1 / 49.3
  - HallusionBench 47.2 / 47.6 / 52.1 / 49.3 / 51.7 / 41.8 / 49.8
  - ScreenSpot-v2 Desktop 78.7 / 79.9 / 82.0 / 63.8 / 76.3 / 28.1 / 45.8
  - ScreenSpot-v2 Web 82.2 / 79.9 / 82.6 / 65.9 / 77.8 / 22.4 / 47.6
  - Speed: ~228 tok/s on M5 Max, 116 tok/s on Ryzen AI Max+ 395, about 3 GB of memory. Day-one llama.cpp, MLX and vLLM support.

  Source: [Liquid HF blog](https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b) [V/3P-V]
- Earlier Liquid models: LFM2-VL-450M OCRBench 655, LFM2-VL-1.6B 742 — [Liquid blog](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models) [V]

**InternVL3.5 (Aug 2025; InternViT-300M, 448 px tiles, 1024 → 256 tokens per tile via pixel shuffle)**
- Report table 4 [V]. Columns are ChartQA / TextVQA / DocVQA / InfoVQA / OCRBench / CharXiv RQ:
  - 1B: 77.7 / 71.5 / 85.6 / 60.5 / 795 / 26.9
  - 2B: 80.7 / 76.5 / 89.4 / 70.8 / 836 / 31.6
  - 4B: 86.0 / 77.9 / 92.4 / 78.0 / 822 / 39.6
  - 8B: 86.7 / 78.2 / 92.3 / 79.1 / 840 / 44.4
  - MMMU: 1B 44.2, 2B 59.0, 4B 66.6
  - 4B GUI grounding: ScreenSpot 83.6, ScreenSpot-v2 85.1

  Source: [InternVL3.5 report](https://arxiv.org/html/2508.18265)
- The same table has older comparators: Qwen2.5-VL-3B OCRBench 797 / DocVQA 93.9; MiniCPM-V-4 (4B) OCRBench 862 / DocVQA 94.0 / ChartQA 73.0; Gemma3-12B OCRBench 702 — [InternVL3.5 report](https://arxiv.org/html/2508.18265)
- Flash variant: the Visual Resolution Router (ViR) picks 256 or 64 tokens per tile, giving about 50% fewer visual tokens at "nearly 100%" of the performance. The overall speedup over InternVL3 is 4.05× (ViR plus DvD combined) — [InternVL3.5 report](https://arxiv.org/html/2508.18265), [InternVL3.5-2B card](https://huggingface.co/OpenGVLab/InternVL3_5-2B)

**Qwen3-VL 2B/4B (Oct 2025; superseded by Qwen3.5 for larch)**
- Encoder: SigLIP2-Large (300M) for 2B/4B and SigLIP2-SO-400M for larger models. Dynamic resolution with 2D-RoPE, 2×2 merge into one token through an MLP, and DeepStack injection of 3 ViT levels into the first 3 LM layers — [Qwen3-VL report](https://arxiv.org/html/2511.21631)
- I could not extract exact 2B/4B vision numbers: the report tables would not parse and the HF cards show them as images. Secondary figures for 4B Instruct: DocVQA ~91–95.3, OCRBench ~85, ScreenSpot ~90; for 8B Instruct: OCRBench 89.6, DocVQA 96.1, ScreenSpot 94.4. Sources disagree — [Codersera, Jun 2026](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/) [2nd, unsourced], [datalearner](https://www.datalearner.com/en/ai-models/pretrained-models/qwen3-vl-4b-instruct/analysis) [2nd]
- llama.cpp gained Qwen3-VL (dense, MoE, DeepStack in the mmproj path) around 27–30 Oct 2025 ("add support for qwen3vl series (#16780)") — [Debian source qwen3vl.cpp](https://sources.debian.org/src/llama.cpp/8064%2Bdfsg-2/tools/mtmd/models/qwen3vl.cpp), [PR #16780](https://github.com/ggml-org/llama.cpp/pull/16780/files)

**MiniCPM-V 4.6 (11 May 2026)**
- About 1–1.3B: SigLIP2-400M plus Qwen3.5-0.8B. Visual token compression is switchable between 4× and 16× (LLaVA-UHD v4), and visual encoding FLOPs drop by more than 50%. Claims about 1.5× the token throughput of Qwen3.5-0.8B and "Qwen3.5 2B-level" OCRBench and HallusionBench, without numbers in text. Official GGUF `openbmb/MiniCPM-V-4.6-gguf`, llama.cpp and Ollama support. Apache-2.0 — [MiniCPM-V-4.6 card](https://huggingface.co/openbmb/MiniCPM-V-4.6), [ITHome](https://www.ithome.com/0/949/729.htm)
- OCRBench "876" appears only in an unverified third-party cache — [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/minicpm-v-4-6-a-1-3b-multimodal-model-built-for-phones/) [2nd]
- Independent [I]: OCRBench v2 EN avg 40.4 and ZH 52.3 (2026.06 round) — [OCRBench v2 leaderboard](https://99franklin.github.io/ocrbench_v2/)

**Older or weaker small models**
- SmolVLM2-2.2B [V]: OCRBench 72.9, DocVQA val 79.98, ChartQA 68.84, TextVQA 73.21, MMMU 42. SigLIP so400m-patch14-384 with image splitting (Idefics3) — [SmolVLM2 card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- Gemma 4 E2B/E4B (report Jul 2026; Apache 2.0): ~150M vision encoder, variable aspect ratio and resolution, fixed token budgets of 70/140/280/560/1120. OmniDocBench1.5 edit distance (lower is better) 0.290 / 0.181. MMMU-Pro 44.2 / 52.6 [V] — [Gemma 4 E4B card](https://huggingface.co/google/gemma-4-E4B-it)
- Gemma 4 E2B on OCRBench: 71.8 per Liquid [3P-V] — [LFM3 card](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- Gemma 4 in llama.cpp: `ggml-org/gemma-4-E2B-it-GGUF`, `gemma-4-E4B-it-GGUF`, `26B-A4B`, `31B` — [llama.cpp multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- Moondream 3 Preview (9B MoE, ~2B active, Sep 2025, BSL 1.1). Vendor numbers: ScreenSpot F1@0.5 80.4, DocVQA 79.3, ChartQA 77.5, OCRBench 61.2. The vendor says the encoder struggles with tiny fonts. One report gives ScreenSpot 60.3 instead — [Moondream models](https://moondream.ai/models), [Moondream blog](https://moondream.ai/blog/moondream-3-preview), [AIBase](https://news.aibase.com/news/21621) [2nd]
- Moondream 3.1 (9B-A2B) card: OlmOCR 32.37 (preliminary) — [Moondream 3.1](https://moondream.ai/models/moondream_3-1_9B_A2B)
- llama.cpp lists only `moondream2-20250414` — [multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

### Inferences
- **Per GB on 4 GB:** Qwen3.5-2B and InternVL3.5-2B are about equal on OCRBench (84–85 vs 83.6–83.9 when the same lab ran both). Qwen3.5-2B is ahead on DocVQA (+4), InfoVQA (+4 to +15), OmniDocBench and ScreenSpot-Pro (54.5 vs ~21 for InternVL3.5-4B). InternVL3.5-2B is ahead on ChartQA in Liquid's non-reasoning run (81.7 vs 78.4) and on ScreenSpot-v2 (79.9 vs 63.8 Desktop). The order on ScreenSpot-v2 (InternVL3.5 ahead) and ScreenSpot-Pro (Qwen3.5 ahead) conflicts, likely because of prompt or coordinate format. Test both on your own screenshot set.
- **KV cost, computed from the card's architecture (not a published number):**
  - Qwen3.5-2B has 6 full-attention layers × 2 KV heads × 256 dim × 2 (K, V) × 2 bytes = **12 KiB per token** at fp16.
  - InternVL3.5-1B (Qwen3-0.6B text model) uses ~112 KiB per token (matches larch's own measurement).
  - So a 2–4k-token screenshot costs roughly 9× less KV on Qwen3.5. On a 4 GB card this matters more than a ±1 point OCRBench gap, and it argues for making Qwen3.5-2B the default 4 GB image model once CUDA GDN kernels land.
- **LFM3-VL-2B is the strongest new candidate to evaluate.** Its numbers are vendor-only, and it is a new architecture (LFM3 hybrid conv LM plus conv/attention encoder), so larch would need new kernels. That is a large cost. Wait for independent OCRBench v2 or OpenCompass numbers before investing.
- **MiniCPM-V 4.6** reuses the Qwen3.5-0.8B text model larch already supports. Adding it means only a SigLIP2 encoder plus the LLaVA-UHD v4 resampler. That is a cheap way to get a ~1B model at near-2B quality with 4–16× fewer visual tokens. It is a good 0.8B/CPU-tier option, provided the vendor claim holds.
- **Drop or skip:** SmolVLM2, Moondream, and Gemma 3 at ≤4B. Gemma 4 E2B/E4B for OCR (OCRBench 70–74; ChartQA is unstable across evaluators: 82.3 vs 43.2).

### Gaps
- No exact, authoritative Qwen3-VL-2B/4B OCRBench, DocVQA or ScreenSpot numbers. The report tables could not be parsed.
- No independent (OpenCompass or OCRBench v2) numbers yet for LFM3-VL-2B, Qwen3.5-2B/4B, or InternVL3.5-2B. OCRBench v2 covers only Qwen3.5-9B, LFM2.5-VL-3B and MiniCPM-V-4.6 among the relevant small models.
- Not researched in depth (time budget): Phi-4-multimodal, PaliGemma 2, Florence-2, Ovis2.5-2B, granite-vision, GOT-OCR2, SAIL-VL2, Youtu-VL-4B (OCRBench v2 EN 42.3, ZH 54.1 [I]). Ovis2.5-8B gets OCRBench v2 EN 54.1 [I] — [OCRBench v2](https://99franklin.github.io/ocrbench_v2/)
- No measured visual-token counts per typical 1920×1080 screenshot, for any model, from a primary source.

## Q2. Mid-size (7–32B) and MoE VLMs for 8–24 GB GPUs and Macs

### Takeaway
The Qwen3.5/3.6 line dominates open mid-size OCR and document benchmarks. Qwen3.5-9B is the best dense model for 8–12 GB. **Qwen3.6-35B-A3B** (MoE, about 3B active) is the best fit for 24 GB GPUs and 32 GB+ Macs, because it decodes at near-3B speed. Gemma 4 26B/31B and Qwen3-VL-30B-A3B trail on document OCR. Nemotron 3 Nano Omni 30B-A3B tops the independent OCRBench v2.

### Cited Findings
- **Qwen3.6-35B-A3B (Apr 2026)** [V]. Columns are Qwen3.6-35B-A3B / Qwen3.5-35B-A3B / Qwen3.5-27B / Gemma4-31B / Gemma4-26B-A4B:
  - OmniDocBench1.5: 89.9 / 89.3 / 88.9 / 80.1 / 74.4
  - CC-OCR: 81.9 / 80.7 / 81.0 / 75.7 / 74.5
  - CharXiv RQ: 78.0 / 77.5 / 79.5 / 67.9 / 69.0
  - HallusionBench: 69.8 / 67.9 / 70.0 / 67.4 / 66.1
  - MMMU: 81.7 / 81.4 / 82.3 / 80.4 / 78.4
  - "975" community quantizations exist (llama.cpp/Ollama/LM Studio)

  Source: [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Qwen3.5-9B [V]: OCRBench 89.2, OmniDocBench 87.7, ScreenSpot Pro 65.2, OSWorld-Verified 41.8. The older Qwen3-VL-30B-A3B in the same table has OCRBench 83.9, OmniDoc 86.8, ScreenSpot Pro 60.5 — [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Independent OCRBench v2, EN average, 2026.09 round [I]:
  - Nemotron 3 Nano Omni 30B-A3B 65.8 (#2 overall)
  - Qwen3-Omni-30B-A3B 61.3
  - SenseNova-U1.5-8B 59.9
  - GLM-4.6V-Flash (9B) 59.0, with the best recognition among open models (75.3)
  - Qwen3.5-9B 58.7
  - Llama-3.1-Nemotron-Nano-VL-8B 56.4
  - LFM2.5-VL-3B 42.6
  - ZH track: Qwen3.5-9B 64.1 first, MiniCPM-V-4.5-8B 58.8

  Source: [OCRBench v2 leaderboard](https://99franklin.github.io/ocrbench_v2/)
- InternVL3.5 mid-size [V]: 30B-A3B OCRBench 880, DocVQA 94.2, InfoVQA 81.4, ScreenSpot-v2 87.3, HallusionBench 53.8. 38B OCRBench 870, DocVQA 94.0. 14B OCRBench 836 — [InternVL3.5 report](https://arxiv.org/html/2508.18265)
- MiniCPM-V 4.5 (Aug 2025): 8B, Qwen3-8B plus SigLIP2-400M. Vendor claims SOTA on OCRBench and OmniDocBench, but no number was retrievable. MiniCPM-o 4.5 (9B) has OCRBench 879 in thinking mode [V] — [MiniCPM-V-4_5 card](https://huggingface.co/openbmb/MiniCPM-V-4_5), [MiniCPM-o 4.5 report](https://arxiv.org/pdf/2604.27393)
- Q4_K_M sizes: Qwen3-VL-4B GGUF ~3.3 GB (6 GB VRAM minimum), 8B ~6.1 GB (8 GB minimum). Claimed 3–5% OCR accuracy drop at Q4_K_M vs full precision, and Q2_K not recommended for vision — [Codersera](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/) [2nd, unsourced]
- llama.cpp official GGUFs for mid-size VLMs: Gemma 4 26B-A4B and 31B, Qwen3-Omni-30B-A3B, Qwen2.5-VL 7B/32B/72B, InternVL3 8B/14B, Mistral Small 3.1 24B, Pixtral 12B, Llama 4 Scout — [multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

### Inferences
- Tier picks, using only the numbers above:
  - **8 GB:** Qwen3.5-4B, or Qwen3.5-9B at Q4.
  - **12–16 GB:** Qwen3.5-9B at Q6/Q8. GLM-4.6V-Flash 9B is a strong OCR alternative but needs a new architecture.
  - **24 GB / 32–64 GB Macs:** Qwen3.6-35B-A3B. About 3B active makes it decode near 3B speed, and the expert weights can be offloaded.
- Supporting the Qwen3.5 family end to end (0.8B → 35B-A3B; the same encoder style and GDN hybrid LM) is the highest-leverage choice. One code path covers every tier.
- Gemma 4 is attractive for license and multilingual reasons, but trails on OCR and documents (OmniDocBench1.5 74–80 vs ~89). It is a lower priority for screenshots.

### Gaps
- Qwen3.5-27B and 35B-A3B have no OCRBench or ScreenSpot-Pro numbers on the cards I read.
- No independent tokens/s or VRAM measurements for these mid-size models with image input in llama.cpp.
- Not confirmed whether the Qwen3.5/3.6 MoE vision mmprojs are in llama.cpp upstream with GPU encoders on all backends (Vulkan issue #20081).

## Q3. Vision encoders: efficiency, OCR strength, native aspect ratio

### Takeaway
SigLIP2 (especially the NaFlex native-aspect variant) is the de facto small-VLM encoder in 2026. Qwen3-VL/3.5, LFM2.5-VL, MiniCPM-V 4.5/4.6 and Gemma-family derivatives use it or something similar. It beats AIMv2 on OCR in a controlled ablation. Native-resolution encoders (Qwen ViT, SigLIP2 NaFlex, LFM3's conv/attention encoder) avoid InternViT's fixed 448-px tiling waste on wide screenshots. Token-compression levers (2×2 merge, pixel shuffle, 4×/16× resamplers, ViR routing, budgeted tokens) matter more for speed than the encoder choice.

### Cited Findings
- SigLIP 2 NaFlex (native aspect ratio, variable sequence length) outperforms the fixed-resolution SigLIP 2 on most retrieval benchmarks. The biggest gains are at short sequence lengths and on OCR/document sets such as TextCaps and HierText. The paper also reports that SigLIP 2 beats SigLIP and AIMv2 as a VLM encoder at several sizes — [SigLIP 2 paper](https://arxiv.org/pdf/2502.14786)
- Flash-VL 2B ablation: SigLIP2-so400m@512 scored OCRBench 386 vs 184 for AIMv2-H/14@448 (an early-stage ablation, not a final model). In that setup dynamic resolution hurt SigLIP2 — [Flash-VL 2B](https://arxiv.org/pdf/2505.09498)
- SigLIP 2 at 400M is reported to beat InternViT-6B (5.9B) on most VLM benchmarks — [Jina vision encoder survey](https://jina.ai/vision-encoder-survey.pdf) [2nd]
- Qwen3-VL: SigLIP-2 continued with dynamic resolution, 2D-RoPE plus interpolated absolute position embeddings, a 2×2 merge into one token, and DeepStack multi-level features. Size: 300M (SigLIP2-L) for 2B/4B and 400M (SO) for larger models — [Qwen3-VL report](https://arxiv.org/html/2511.21631)
- InternVL3.5: InternViT-300M (6B for the largest models) with "Dynamic High Resolution" tiling, 448-px tiles, and 1024 → 256 tokens per tile (pixel shuffle). Flash adds a 64-token path per tile — [InternVL3.5-2B card](https://huggingface.co/OpenGVLab/InternVL3_5-2B)
- LFM3-VL: 164M in-house encoder (gated depthwise conv plus attention), 16×16 patches, 2D RoPE, native resolution, 2×2 → 1 token — [LFM3-VL-2B card](https://huggingface.co/LiquidAI/LFM3-VL-2B)
- LFM2.5-VL-3B: SigLIP2 400M NaFlex — [Liquid HF blog](https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b)
- Gemma 4: ~150M encoder (E2B/E4B) or ~550M (26B/31B); variable aspect ratio and resolution; token budgets 70/140/280/560/1120 — [Gemma 4 card](https://huggingface.co/google/gemma-4-E4B-it)
- MiniCPM-V 4.6: SigLIP2-400M, mixed 4×/16× compression, `max_slice_nums` 9 by default (36 recommended for images) — [MiniCPM-V-4.6 card](https://huggingface.co/openbmb/MiniCPM-V-4.6)
- MiniCPM-V 4.5: SigLIP2-400M — [MiniCPM-V-4_5 card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- DeepSeek-OCR (Oct 2025) shows how far visual tokens can be compressed:
  - Precision is 96%+ at 9–10× text compression, ~90% at 10–12×, and ~60% at 20× (Fox benchmark).
  - Modes: Tiny = 64 tokens at 512², Small = 100 tokens at 640².
  - On OmniDocBench it beats GOT-OCR2.0 (256 tokens per page) with 100 tokens, and beats MinerU2.0 (6000+ tokens) with fewer than 800.

  Sources: [DeepSeek-OCR arXiv](https://arxiv.org/abs/2510.18234); detail from [themoonlight review](https://www.themoonlight.io/review/deepseek-ocr-contexts-optical-compression) [2nd]
- CARES: a lightweight selector that picks the minimal sufficient input resolution per query before tokenization, with a frozen VLM — [CARES arXiv 2510.19496](https://arxiv.org/html/2510.19496v1)

### Inferences
- **Pixels per token:**
  - InternVL's 448² tile → 256 tokens works out to 784 px² per token (28×28 px).
  - Qwen3-VL and LFM3 merge 2×2 patches into one token. With 16-px patches that is 32×32 px (1024 px²) per token, about 23% fewer tokens per pixel. They also avoid padding and thumbnail tiles on odd aspect ratios such as wide terminals.
  - The 16-px patch is stated for LFM3. For Qwen3-VL it is inferred, not confirmed from a primary source here.
- **Model-level additions larch could make, cheapest first:**
  1. A per-image token budget or `max_pixels` knob, as Gemma 4's budget list and Qwen's min/max pixels already do.
  2. An InternVL3.5-Flash-style 64-token path for low-information tiles.
  3. A CARES-like resolution selector.
  4. MiniCPM-V-style 16× compression for the ~1B tier.
- None of these needs a new base model.

### Gaps
- No head-to-head OCR comparison of SigLIP2 NaFlex vs Qwen ViT vs InternViT vs LFM3's encoder in the same setup.
- No published encoder latency (ms per 1080p image) on consumer GPUs for these encoders.
- AIMv2 evidence comes from a single ablation.

## Q4. Specialist models (OCR/document parsers, GUI grounding, detectors) and router/cascade practice

### Takeaway
Sub-1.2B document-OCR VLMs are now very strong: PaddleOCR-VL 1.5/1.6, GLM-OCR, OvisOCR2, TeleOCR and MinerU2.5-Pro all score OmniDocBench v1.6 above 95. PaddleOCR-VL and DeepSeek-OCR (plus OCR 2) are merged in llama.cpp. They need fixed prompts, which makes them good **tools** for an agent ("transcribe this page"), not chat models. General small VLMs now ground UI elements well (Qwen3.5-2B ScreenSpot-Pro 54.5; LFM3-VL-2B 55.5, vendor-reported), so a separate GUI-grounding model is optional unless computer-use is a goal. Holo2-4B/8B lead that niche. Cascades and routers appear in papers and design proposals with **projected** savings of 70–78%. I found no measured production numbers.

### Cited Findings
- **OmniDocBench v1.6 (checked 29 Sep 2026):**
  - TeleOCR 1.2B (Aug 2026) 96.91; community GGUF runs on llama.cpp
  - OvisOCR2 0.8B (Jul 2026) 96.47
  - PaddleOCR-VL-1.6 0.9B (May 2026) 96.34
  - MinerU2.5-Pro 1.2B 95.75
  - GLM-OCR 0.9B (Feb 2026, MIT) 95.22
  - DeepSeek-OCR 2 3B (Jan 2026) 90.25
  - Gemini 3 Pro 92.91, Qwen3-VL-235B 89.78
  - olmOCR-Bench: Chandra OCR 2 (5.3B) 85.8, dots.mocr (3B) 83.9, olmOCR 2 (7B) 82.4
  - HunyuanOCR 89.95, but its license grants no rights in the EU, UK or South Korea

  Source: [Docsumo, 30 Sep 2026](https://www.docsumo.com/blog/best-ocr-models) [2nd; verify against the official leaderboard]
- **llama.cpp support for OCR models:**
  - PaddleOCR-VL: PR #18825 merged 19 Feb 2026 (b8110). There is an official `PaddlePaddle/PaddleOCR-VL-1.5-GGUF` and community 1.6 GGUFs.
  - DeepSeek-OCR: PR #17400. DeepSeek-OCR 2 with multi-tile dynamic resolution: PR #20975, in b9414.
  - GLM-OCR (#19677), Dots.OCR (#17575) and HunyuanOCR (#21395) are linked from the docs as needing specific prompts. I could not confirm merge status.

  Sources: [llama.cpp multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), [b8110 release](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b8110), [b9414 release](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b9414), [PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-GGUF/blob/main/README.md)
- **GUI grounding, ScreenSpot-Pro:**
  - Holo2-4B 68.6, Holo2-8B 71.4, Holo2-30B-A3B 75.2, Holo1.5-7B 57.9
  - UI-Venus-1.5-8B 68.4, UI-Venus-Ground-7B 50.8, GUI-Actor-7B 44.6, UI-AGILE-3B 45.0
  - These are aggregator rows imported 2026-05-06, mixing self-reported scores and setups

  Sources: [benchmarklist ScreenSpot-Pro](https://benchmarklist.com/benchmarks/screenspot_pro/) [2nd]. Vendor claim: Indeed-UI-8B 81% and 32B 82.7% [V, press release 2026-09-10] — [ai-indeed](https://www.ai-indeed.com/aboutNews/31372.html). Official board: gui-agent.github.io/grounding-leaderboard (not opened)
- ZonUI-3B: ScreenSpot 84.9 and ScreenSpot-v2 86.4, trained on a single consumer GPU with balanced sampling and two-stage fine-tuning — [ZonUI-3B arXiv 2506.23491](https://arxiv.org/html/2506.23491v3)
- Moondream 3 Preview is positioned for pointing and detection (ScreenSpot F1 80.4, COCO mAP 51.2), but has weak OCR (OCRBench 61.2) and a BSL license — [moondream.ai/models](https://moondream.ai/models)
- **Routing and cascades:**
  - AVR (CVPR 2026, computer-use agents) puts a light router in front of a VLM pool. Its cost model assumes a small model 10× cheaper and a 20% escalation rate, which gives ~70% cost reduction. Projected savings reach 78% while staying within 2 points of an all-large baseline. The small-model probe costs ~10% of a full generation and scales with screenshot size. These are projections, not measured savings — [AVR arXiv 2603.12823](https://arxiv.org/abs/2603.12823), [summary](https://www.emergentmind.com/papers/2603.12823)
  - ECVL-Router routes between an edge small VLM and a cloud large VLM by predicted quality — [arXiv 2510.27256](https://arxiv.org/html/2510.27256v1)
  - Roboflow describes filter cascades (a small model decides whether, or which crop, goes to a large VLM), without numbers — [Roboflow](https://roboflow.com/blog/model-cascades-for-computer-vision)
  - Within one model, InternVL3.5-Flash's ViR routes tiles between 256 and 64 tokens for ~50% fewer tokens — [InternVL3.5 report](https://arxiv.org/html/2508.18265)

### Inferences
- The best specialist to add is **PaddleOCR-VL-1.5/1.6 (0.9B)**: it is upstream in llama.cpp, Apache-2.0, has an official GGUF, and is SOTA-class on documents. Use it as an "exact transcription" tool the chat model can call on dense pages, tables or spreadsheets. Second choice: GLM-OCR 0.9B (MIT), once its llama.cpp status is confirmed. DeepSeek-OCR suits very low token budgets (64–100 tokens per page).
- Skip a separate GUI-grounding model unless larch adds computer-use. If it does, Holo2-4B is the top ≤4B option on the aggregator.
- A practical cascade for larch is intra-model (a token budget or ViR-style tile routing) plus "describe once, reuse text", which larch's "pixels while recent" design already does. Cross-model escalation (2B → 9B/35B-A3B on low confidence) has only projected evidence. Treat it as an experiment with its own measurements.

### Gaps
- Official OmniDocBench v1.6 and olmOCR-Bench leaderboards not opened directly. The Docsumo figures are secondary.
- No measured real-world savings from small→large VLM cascades. No standard confidence signal for VLM escalation.
- Small detectors (e.g., YOLO/RT-DETR for UI elements) not researched.

## Q5. Fine-tuning and distillation options and expected gains

### Takeaway
LoRA or full fine-tuning on GUI and screenshot data is cheap and well documented for 2–3B models. ZonUI-3B reaches ScreenSpot 84.9 on a single consumer GPU, and a Qwen3-VL-2B LoRA took about $5 of compute. Reported distillation gains for 2B students are modest (+3.5 to +5.8 points on ScreenSpot-Pro/v2). The informal 7B → 2B screenshot-description distillation showed small gains on a weak metric. Base-model upgrades (InternVL3.5 → Qwen3.5 → LFM3/MiniCPM-V 4.6) currently move scores more than fine-tuning does.

### Cited Findings
- WinDOM (ICML 2026): on a Qwen3.5-2B student, RL from a distilled cold start gives +5.8 on ScreenSpot-V2 and +3.5 on ScreenSpot-Pro over the base model. Source is an abstract listing — [ICML 2026 WinDOM](https://icml.cc/virtual/2026/67777)
- ZonUI-3B: 84.9 ScreenSpot and 86.4 ScreenSpot-v2. Ablations credit balanced cross-platform sampling, redundancy reduction and two-stage fine-tuning, on a single consumer GPU — [arXiv 2506.23491](https://arxiv.org/html/2506.23491v3)
- browserground: a Qwen3-VL-2B LoRA (rank 32, 26k examples) trained for about $5 on an L40S. No ScreenSpot score reported — [HF browserground](https://huggingface.co/renezander030/browserground) [2nd/self-published]
- Informal 7B → 2B screenshot-description distillation with a LoRA: BLEU 0.178 for the student vs 0.164 for the teacher vs 0.153 for the base 2B — [DEV post](https://dev.to/p0rt/i-distilled-a-7b-vision-model-into-a-2b-one-for-screenshots-and-the-7b-teacher-scored-worse-3akh) [informal]
- Smol2Operator shows post-training SmolVLM2 for GUI use — [HF blog](https://huggingface.co/blog/smol2operator)
- A study of LoRA restricted to visual-token pathways, evaluated on ScreenSpot-Pro (numbers not extracted) — [arXiv 2512.19219](https://arxiv.org/pdf/2512.19219v1)

### Inferences
- If larch offers fine-tuning:
  - Ship LoRA adapters (GGUF LoRA on the text model; the mmproj stays frozen) trained on the user's real paste distribution: terminals, IDE errors, spreadsheets.
  - Expect gains of a few points on in-domain screenshot QA, not the 10+ points that model swaps give.
- Distilling Qwen3.5-9B or 35B-A3B answers on screenshots into Qwen3.5-2B is plausible. The published evidence (WinDOM) suggests single-digit gains.

### Gaps
- No published LoRA results for screenshot OCR/QA (as opposed to grounding) at 2B, and no figures on quantization-aware LoRA for vision GGUFs.
- No source on how much LoRA gains survive Q4 quantization of the text model.
