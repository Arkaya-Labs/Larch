# Decode-side speedups and quantization effects for VLMs (local to at-scale)

Research date: 2026-11-11. Context: larch (Qwen3.5 0.8B Q8_0 / 2B Q4_K_M hybrid GDN + full attention every 4th layer, M-RoPE; InternVL3.5 1B/2B; RTX 3050 4 GB; existing exact spec-decode with prompt lookup and a DFlash drafter whose positions are not M-RoPE-aware after an image).

Caveat for the whole file: almost all speedup numbers are self-reported by method authors, on 3B-13B targets, on datacenter GPUs, with different benchmarks and temperatures. They are not directly comparable to each other or to a 0.8B/2B target on a 4 GB laptop GPU.

## 1. Speculative decoding when the prompt contains image tokens

### Takeaway
Exact speculative decoding works for VLMs, but autoregressive one-layer drafters (EAGLE-2/EAGLE-3, Medusa) lose much of their text-only gain once an image is in the prompt (often below 1.5x, sometimes below 1x). Block-parallel drafters (DFlash-style, GLANCE, Uni-LaSpec) are the 2026 state of the art at 2.2-3.2x on multimodal prompts at batch 1, including on Qwen3.5 and InternVL3.5 targets. A DFlash drafter trained on text only already gets most of the multimodal gain; multimodal training or a few visual tokens add about 5-15% acceptance length, with the largest gain on document/OCR prompts. Gains collapse under heavy serving load.

### Cited Findings
**Text-only / image-free drafters**
- Gagrani et al. (CVPR 2024 workshop, arXiv 2404.08856, Apr 2024): a language-only draft model can draft for LLaVA-7B without image tokens or the image encoder; a 115M model trained from scratch gave up to 2.37x memory-bound speedup. — [arXiv 2404.08856](https://arxiv.org/abs/2404.08856), [CVPRW paper](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Gagrani_On_Speculative_Decoding_for_Multimodal_Large_Language_Models_CVPRW_2024_paper.pdf). Note the conflict: the ViSpec paper summarizes Gagrani et al. as "up to 1.5x" on LLaVA-7B and says a small VLM draft with an image encoder gave "only marginal gains" — [ViSpec](https://arxiv.org/html/2509.15235v1). The 2.37x is a memory-bound (theoretical) figure, not wall-clock.
- HiViS (arXiv 2509.23928, Sep 2025, rev. Aug 2026, CVPR 2026 Findings): hides visual tokens from the drafter; the target acts as a "semantic fusion model", so drafter prefill length equals the text token count; uses time-step-aware aligned training with bias-correction residuals. Abstract gives no numbers. — [arXiv 2509.23928](https://arxiv.org/abs/2509.23928)

**EAGLE/Medusa-style heads with image input (ViSpec, Sep 2025)**
- ViSpec (arXiv 2509.15235, NeurIPS 2025): Q-Former-style vision adaptor compresses image features (final version uses a single compressed embedding) inserted into draft attention, plus a global image feature added to all later text hidden states; drafter is one transformer layer; trained on 68k ShareGPT + 68k LLaVA samples plus target-generated long responses ("at least 1000 words"), 20 epochs on 8 GPUs. — [ViSpec](https://arxiv.org/html/2509.15235v1)
- ViSpec average speedups (T=0 / T=1): LLaVA-1.6-7B 2.58x / 2.14x; LLaVA-1.6-13B 2.38x / 2.13x; Qwen2.5-VL-3B 1.87x / 1.47x; Qwen2.5-VL-7B 1.80x / 1.49x. EAGLE-2 baselines (image-embedding-fed, 1 layer, 30-token tree): 1.62x, 1.86x, 1.39x, 1.40x at T=0 and 1.18-1.54x at T=1. Medusa: 1.01-1.48x. — [ViSpec](https://arxiv.org/html/2509.15235v1)
- ViSpec acceptance length tau at T=0: about 2.9-3.0 on LLaVA-7B/13B, about 2.2 on both Qwen2.5-VL models; EAGLE-2 1.16-1.89; Medusa 0.60-0.82. Per-task ViSpec speedups at T=0 range 1.37x-3.22x. ViSpec does not mention M-RoPE. — [ViSpec](https://arxiv.org/html/2509.15235v1)
- Qwen-family targets get clearly lower speedups than LLaVA in the same paper (1.8x vs 2.4-2.6x). — [ViSpec](https://arxiv.org/html/2509.15235v1)

**Other LLaVA-era methods (2025-2026)**
- MSD (Multimodal Speculative Decoding, arXiv 2505.14260, May 2025): up to 2.29x on LLaVA-1.5-7B and 2.46x on LLaVA-1.5-13B; notes acceptance length similar to or above text-only SD but lower end-to-end speedup on multimodal tasks. — [arXiv 2505.14260](https://arxiv.org/pdf/2505.14260)
- Spec-LLaVA (arXiv 2509.11961, Sep 2025): up to 3.28x on LLaVA-1.5-7B with a 160M drafter, 2.95x on 13B; dynamic tree branching adds 15-20% acceptance length. — [arXiv 2509.11961](https://arxiv.org/pdf/2509.11961)
- DREAM-S (arXiv 2606.00535, Jun 2026): 2.58x on LLaVA-v1.6-Vicuna-7B, 3.62x on 13B; uses attention-head pruning and visual token compression in the drafter. — [arXiv 2606.00535](https://arxiv.org/pdf/2606.00535)
- Variational Speculative Decoding (arXiv 2602.05774, Feb 2026): +7.9% acceptance length and +7.3% speedup over ViSpec on LLaVA-1.5. — [arXiv 2602.05774](https://arxiv.org/html/2602.05774v1)
- MASSV (Ganesan et al., 2025): reuses the target's vision-encoder outputs through a lightweight projector into a small LM drafter plus self-data distillation (described in later papers; I could not retrieve its own numbers). — [ViSpec/related search result summary](https://arxiv.org/pdf/2609.00355)

**Block-parallel / diffusion drafters with images (most relevant to larch's DFlash)**
- Survey + diagnosis (arXiv 2608.20743, Aug 2026): block-parallel diffusion drafting (DFlash, DSpark) reaches up to 3.6x on daily chat tasks; block-parallel drafting for multimodal models is "largely unexplored"; compares methods on OCR, VQA, visual reasoning and captioning. — [arXiv 2608.20743](https://arxiv.org/abs/2608.20743)
- GLANCE ("Vision Is Not Overhead", arXiv 2609.00355, Aug-Oct 2026): block-diffusion head drafts a whole block in one pass over the target's already-fused vision-language hidden states (no image compression or hiding), with an unmodified target and wide tree verification, committing exactly the greedy output; up to 3.05x vs autoregressive; beats the production EAGLE3-VL head on average and by about 11% on grounded tasks; an "entropy law" predicts longest accepted blocks on grounded (low-entropy) tasks. Code: github.com/js-lee-AI/GLANCE. — [arXiv 2609.00355](https://arxiv.org/abs/2609.00355)
- Uni-LaSpec (arXiv 2611.01617 v2, Nov 2026) targets Qwen3.5-4B/9B, InternVL3.5-4B/8B, Gemma 4 E2B/E4B — the same families larch runs. — [arXiv 2611.01617](https://arxiv.org/html/2611.01617)
  - Qwen3.5-4B, batch 1, T=0, 15-token block: EAGLE-3 1.41x text / 0.89x multimodal (tau 3.82 / 2.34); MTP 1.55x / 1.14x (tau 5.61 / 4.20); DFlash text-trained 3.41x / 2.19x (tau 5.16 / 3.49); DFlash multimodal-trained 3.40x / 2.39x (tau 5.23 / 3.79); Uni-LaSpec 3.60x / 2.60x (tau 5.38 / 4.07). Abstract: up to 5.11x text and 3.24x multimodal.
  - InternVL3.5-4B: Uni-LaSpec 2.78x (tau 3.62 general, 2.80 on second benchmark group, which I read as document/OCR); DFlash multimodal 2.63x; DFlash text-only 2.58x (tau 2.44 on the second group). InternVL3.5-8B: 3.12x / 2.96x / 2.97x. Gemma 4 E2B: 4.57x / 4.47x / 4.40x.
  - The largest tau gain from visual information is on the second (doc/OCR) group: e.g. Gemma 4 E2B DFlash text-only 2.35 -> multimodal-trained 2.79 -> Uni-LaSpec 3.01. (Group labels were not fully legible in the fetched table; treat the doc/OCR label as my reading.)
  - Mechanism: the verifier's attention (cross-layer vote over three full-attention layers) picks about eight visual tokens per round for a gated cross-attention drafter (gate initialized to zero); 8 tokens retain at least 98.6% of full-image acceptance; verifier visual keys cached in int8 with per-position scales; verifier image embeddings cached per request because the default serving path drops them during decode.
  - Positions: draft-tree nodes use rotary positions pos(u) = j + |u| (j = current position, |u| = depth), independent of packed KV-cache indices. M-RoPE is not discussed in the fetched portion.
  - Load: at concurrency 64, the fixed-block strategy drops to 1.06x text and 0.82x multimodal on Qwen3.5-4B; a load-adaptive policy (draft trees at light load, truncated linear drafts at heavy load) beats autoregressive in all settings over six load levels and four GPUs, matches or beats the best fixed strategy in 157/192 settings, and cuts mean end-to-end latency 47.6% vs linear verification under bursty arrivals.
  - Small model: a fine-tuned Qwen3.5-2B ("Q3.5-2B-SFT", in-house data) reached 5.59x at batch 1 with tau 9.92; its MTP heads fell to 0.54x at concurrency 64.
- The ViSpec abstract states prior VLM speculative decoding methods gave "only modest speedups (<1.5x)". — [ViSpec](https://arxiv.org/html/2509.15235v1)

### Inferences
- For larch (batch 1, local), the existing DFlash drafter is the right base: in Uni-LaSpec's tables a text-only-trained DFlash kept about 85-95% of the best multimodal speedup on Qwen3.5-4B and InternVL3.5. The cheap first step is to fix positions; retraining with multimodal data or adding about 8 attended visual tokens is a second step worth roughly 0.2-0.6 tau, mostly on OCR/document answers.
- M-RoPE handling (inference from how Qwen2-VL/Qwen3.5 M-RoPE works, not from a cited source): text tokens after an image get equal temporal/height/width components, and the text position counter resumes at max(image position ids)+1, not at the KV slot index. A drafter that only sees text (or fused hidden states) should therefore use the target's current 1D text rope position plus draft offset (the same pos(u) = j + |u| rule Uni-LaSpec uses, with j taken from the target's M-RoPE text counter rather than the KV length). The bug in larch's DFlash (positions not M-RoPE-aware after an image) likely reduces acceptance rather than correctness, since verification is exact; a quick check is tau with and without an image whose token count differs from its position span.
- Prompt lookup on OCR'd text: no paper found that measures it, but GLANCE's "entropy law" (longest accepted blocks on grounded, low-entropy tasks) and the higher doc-group gains for visual-aware drafters suggest transcription/OCR answers are where drafting pays most. Prompt lookup cannot copy from image tokens; it would need an OCR text side-channel or the model's own earlier transcription in the chat history.
- Lower speedups on Qwen-family targets than on LLaVA (ViSpec) and on InternVL3.5 than on Gemma 4 E2B (Uni-LaSpec) suggest small, already-fast targets leave less headroom for drafting; on a 0.8B target at 103 tok/s the drafter's relative cost is larger.

### Gaps
- No source measured prompt-lookup decoding on OCR text or on image-containing prompts.
- MASSV's own numbers and any EAGLE-3 results on Qwen2.5-VL/Qwen3-VL from the EAGLE authors were not retrieved.
- No source explicitly describes M-RoPE-aware drafter position handling; Uni-LaSpec's tree-depth rule is the closest. The GLANCE and survey PDFs could not be text-extracted here (no PDF tools in the environment).
- No results found for targets under 2B except Uni-LaSpec's fine-tuned Qwen3.5-2B.

## 2. Weight quantization of the language model: are vision/OCR tasks more sensitive?

### Takeaway
At 4-bit weight-only (W4A16, AWQ/GPTQ), VLMs are close to lossless on OCR/document benchmarks for 2B-7B models (about 0-1.5 points on DocVQA). Damage appears at 3-bit and with activation quantization (W4A8), where OCRBench and TextVQA are the benchmarks that drop most, and modality-agnostic calibration (AWQ, SmoothQuant) fails more often. Language tokens are about 10x more sensitive than vision tokens in gradient terms, which is why modality-balanced calibration (MBQ) helps.

### Cited Findings
- Official Qwen2-VL quantization results (VLMEvalKit): 2B DocVQA BF16 88.34 vs AWQ 86.96 (-1.4); 7B DocVQA BF16 93.89, GPTQ-Int8 93.94, GPTQ-Int4 93.16, AWQ 93.10. Benchmarks: MMMU_VAL, DocVQA_VAL, MMBench_DEV_EN, MathVista_MINI. — [Qwen quantization benchmark docs](https://qwen.readthedocs.io/en/latest/getting_started/quantization_benchmark.html), [Qwen2-VL-2B-Instruct-GPTQ-Int8 card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8)
- A separate paper reports 4-bit GPTQ on Qwen2-VL-2B costs about 6.8% on MMMU, more than DocVQA-style tasks. — [Dual Precision Quantization, arXiv 2505.14638](https://arxiv.org/pdf/2505.14638)
- MBQ (CVPR 2025, arXiv 2412.19509): on LLaVA-onevision-7B the average absolute gradient of language tokens is "over 10x" that of vision tokens; MBQ weights calibration loss by modality sensitivity. Average accuracy FP16 66.9 -> W3 AWQ 60.4 vs MBQ 64.8; W4A8 SmoothQuant 50.9 vs MBQ 62.5; W4 weight-only MBQ 67.1 (paper calls W4 and W8A8 "nearly lossless"). — [MBQ](https://arxiv.org/html/2412.19509v1)
- MBQ per-benchmark: at W4A8 the biggest drops are OCRBench (62.2 -> 52.3) and TextVQA (76.1 -> 68.3) on LLaVA-onevision-7B; InternVL2-8B W4A8 TextVQA 77.0 -> 72.3, OCRBench 76.5 -> 73.0; Qwen2-VL-7B W3 OCRBench 80.7 -> 76.8; at W3, RTN collapses OCRBench to 35.9; ScienceQA and VizWiz stay within about 0.4 points. LLaVA-onevision-72B W3: AWQ 55.1 vs FP16 74.3 vs MBQ 73.6. — [MBQ](https://arxiv.org/html/2412.19509v1)
- MBQ: AWQ had 86 bad cases vs MBQ 38, most common AWQ failure "No Output". W4A8 on the ViT encoder caused no significant drop. MAE reconstruction loss beat MSE by >1% on MMMU/SEED. Abstract: up to 4.4% (W3) and 11.6% (W4A8) better than baselines on 7B-70B VLMs. — [MBQ arXiv abs](https://arxiv.org/abs/2412.19509), [MBQ html](https://arxiv.org/html/2412.19509v1)
- MBQ W3 fused GEMV kernel on RTX 4090: 1.2-5.0x per layer shape; end-to-end LLaVA-onevision-7B decode 29.6 ms -> 21.1 ms per token (1.4x), 1.23x faster than W4A8. — [MBQ](https://arxiv.org/html/2412.19509v1)
- QIG (arXiv 2603.17809, Mar 2026): at W4A8 with 128 calibration samples, +3.52% average over MBQ, +4.12 on DocVQA, +6.20 on OCRBench (MBQ 70.60 -> QIG 76.80 OCRBench in its setting). — [arXiv 2603.17809](https://arxiv.org/html/2603.17809)
- VLMQ (arXiv 2508.03351): evaluates INT3/INT2 on OCRBench, DocVQA, ChartQA, TextVQA; transformation-based methods (AWQ, MBQ) fail to give good results at INT2. — [VLMQ](https://arxiv.org/html/2508.03351v2)
- Further 2026 low-bit VLM quantization work exists (modality heterogeneity arXiv 2605.19929, SPHQuant 2609.24875, SAB-LVLM 2607.01876, MiX 2609.19683, RGSQ 2609.25492) but I did not retrieve their numbers. — [arXiv 2605.19929](https://arxiv.org/pdf/2605.19929), [SPHQuant](https://arxiv.org/pdf/2609.24875)
- NVIDIA Nemotron Nano V2 VL FP8 PTQ scores 94.32 DocVQA-val. — [arXiv 2511.03929](https://arxiv.org/pdf/2511.03929)
- Third-party (unsourced) blog claims of a 3-5% OCR-task drop for Qwen3-VL at Q4_K_M and <1% at FP8 should not be relied on. — [Codersera](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/)

### Inferences
- larch's choices (0.8B Q8_0, 2B Q4_K_M, f16 projector) match where the literature says the risk is low: 8-bit is effectively lossless, 4-bit weight-only costs about 1-1.5 DocVQA points at 2B (Qwen2-VL AWQ data). Keeping the projector and vision encoder at f16 avoids the part MBQ did not test at low bits, though MBQ found W4A8 on the ViT harmless at 7B.
- Small models (0.8B-2B) likely have less redundancy than the 7B-72B models studied, so the 2B's Q4_K_M OCR loss should be measured directly (OCRBench / DocVQA subset) rather than assumed; llama.cpp k-quants use importance-matrix calibration on text, which may under-weight vision-token activations (the MBQ argument), though weight-only 4-bit appears robust.
- Activation quantization (W4A8/W8A8) is where OCR breaks; larch does not need it for decode at batch 1, which is memory-bound.

### Gaps
- No official Qwen3-VL / Qwen3.5 / InternVL3.5 quantization tables found (Qwen's page is marked "to be updated for Qwen3").
- No llama.cpp discussion with measured GGUF k-quant vs f16 OCRBench numbers for small VLMs was found in this pass.
- Q-VLM numbers were not retrieved.

## 3. Image KV cache: quantization, eviction, multi-turn memory

### Takeaway
Image tokens dominate the KV cache and are highly redundant after the first read: methods keep 5-10% of multimodal KV with near-full accuracy on 7B models, and int8 visual keys are used in production-style drafters. For larch's hybrid model, only 1 in 4 layers holds KV, so image KV is already about 4x cheaper than in a dense model; eviction would mainly help long multi-image chats.

### Cited Findings
- VL-Cache (arXiv 2410.23317, Oct 2024): sparsity- and modality-aware layer-wise budget, separate scoring of visual and text tokens; keeping about 10% of KV gives accuracy comparable to full cache; up to 2.33x end-to-end latency (100 generated tokens), up to 7.08x decoding speedup, about 90% less KV memory. Argues text-only LLM KV compression methods perform poorly on VLMs. — [arXiv 2410.23317](https://arxiv.org/abs/2410.23317)
- LOOK-M (arXiv 2406.18139, EMNLP Findings 2024): look-once optimization for multimodal long context (multi-image, multi-turn); merges evicted KV into kept KV instead of dropping; approaches full-cache performance at 5% KV on LLaVA-v1.5-7B. — [arXiv 2406.18139](https://arxiv.org/pdf/2406.18139), [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.235.pdf)
- A later benchmark (arXiv 2511.16786, Nov 2025, "Flash-Cache", frequency-domain outlier-KV-aware) found LOOK-M ran out of memory in some Qwen2.5-VL-7B tests and compared on MileBench, MUIRBench and high-resolution benchmarks. — [arXiv 2511.16786](https://arxiv.org/pdf/2511.16786)
- Uni-LaSpec caches the verifier's visual keys in int8 with per-position scales for its drafter's cross-attention. — [arXiv 2611.01617](https://arxiv.org/html/2611.01617)
- MLX-VLM caches vision-encoder features so the same image across requests is encoded once (feature cache, not KV); LM Studio's mlx-engine has an open issue on cross-turn KV cache for vision models. — [mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm), [mlx-engine issue #287](https://github.com/lmstudio-ai/mlx-engine/issues/287)

### Inferences
- For Qwen3.5 (full attention every 4th layer), image KV lives only in the full-attention layers; GDN layers fold the image into a fixed-size state at prefill. Evicting image KV after the answer would therefore only affect those layers, and the GDN state already "remembers" the image in compressed form. This is untested in the literature found.
- Re-encoding vs keeping: at larch's ~3,000 tok/s prefill, re-prefilling a 1,000-token image costs roughly 0.3 s, so keeping a per-image encoder-feature cache (MLX-VLM style) plus recomputation is a reasonable alternative to holding image KV across many turns; but for the hybrid model, dropping KV mid-history forces re-prefill of everything after the image because GDN state is sequential (see section 5).
- "Elastic Cache" and HiRED were not found in this pass (HiRED is a token-dropping method for high-resolution inputs, applied before the LLM, not KV eviction; not verified here).

### Gaps
- No memory numbers specific to multi-turn chats with eviction of image KV after answering; no paper on image-KV eviction for hybrid linear-attention models.
- Elastic Cache and HiRED not retrieved.

## 4. At scale: video, multi-image, batching, multi-GPU

### Takeaway
For video, post-encoder temporal token pruning (EVS in vLLM) gives about 2-4x lower TTFT and 2.6x throughput with roughly flat accuracy; it complicates CUDA graphs and M-RoPE/timestamp prompt layout. Speculative decoding gains shrink and can turn negative at high concurrency, so serving needs load-adaptive drafting.

### Cited Findings
- EVS (arXiv 2510.14624, Oct 2025): prunes temporally redundant video tokens after the encoder; up to 4x lower TTFT with minimal accuracy loss. — [arXiv 2510.14624](https://arxiv.org/abs/2510.14624)
- Nemotron Nano V2 VL, one GPU, 128 frames: 75% EVS pruning cuts TTFT 4131 ms -> 2072 ms and raises throughput 34 -> 88 tok/s with accuracy roughly flat. — [arXiv 2511.03929](https://arxiv.org/pdf/2511.03929)
- vLLM supports EVS ("evs", removes tokens that changed least from the previous frame) and "vidcom2" (Qwen3-VL only); enabling pruning disables encoder CUDA graphs because retained token count varies. — [vLLM multimodal inputs docs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- Qwen3-VL EVS integration had a mismatch between prompt timestamp/placeholder layout and actual retained token count, since pruned counts are only known after the encoder runs. — [vLLM issue #30847](https://github.com/vllm-project/vllm/issues/30847), [vLLM forum](https://discuss.vllm.ai/t/is-there-a-plan-for-evs-to-support-qwen3vl-in-response-to-the-issue-of-sparse-video-tokens/2215)
- RFC to enable EVS for Qwen3.5-VL on Ascend notes the final prompt reduction will not equal the configured rate because text and timestamp tokens are kept. — [vllm-ascend RFC #20525](https://github.com/vllm-project/vllm-ascend/issues/20525)
- Speculative decoding under load: Uni-LaSpec fixed-block drafting drops to 0.82x multimodal at concurrency 64 on Qwen3.5-4B; MTP on a Qwen3.5-2B to 0.54x; load-adaptive policy stays above 1x everywhere. — [arXiv 2611.01617](https://arxiv.org/html/2611.01617)

### Inferences
- For larch (single user), video support would be dominated by prefill/encoder cost; EVS-style pruning is the cheapest win, but the M-RoPE temporal ids must follow the retained tokens.
- Batch-many-chats findings (spec decode turning negative) matter only if larch adds concurrent serving.

### Gaps
- No sourced numbers found on multi-GPU VLM serving (encoder/LLM disaggregation, tensor parallel for ViT) or on vLLM's encoder cache throughput in this pass.

## 5. Hybrid / linear-attention models (Qwen3.5 / Qwen3-Next) with images

### Takeaway
Prefix caching for hybrid GDN models is the main known serving issue: recurrent state can only be reused at saved checkpoints, so vLLM's default "align" mode gave near-zero hit rates, and block-level state checkpointing ("all" mode) needed float32 states for exactness. Speculative decoding on Qwen3.5 works (DFlash, MTP, Uni-LaSpec) but needs state rollback/checkpointing for rejected drafts.

### Cited Findings
- vLLM PR #36649 (opened Mar 2026, closed unmerged May 2026 in favor of PR #26807): "align" mode cached only 1-2 tail blocks and GDN null blocks were never cached, giving 0% hits; "all" mode saves GDN SSM and conv1d states at block boundaries during prefill, sets mamba_chunk_size=64 and auto-sets mamba_ssm_cache_dtype=float32 because a bf16 round trip gave max diff 2.0 in kernel tests; Qwen3.5-35B-A3B second run 37 s -> 2.6 s (14x) with 1,088 cached tokens and matching outputs; a user saw about 2% output divergence vs no cache (about 1% for #26807). Tested with --language-model-only; multimodal not discussed. — [vLLM PR #36649](https://github.com/vllm-project/vllm/pull/36649)
- vLLM PR #50172 adds "all"-mode prefix caching combined with MTP speculative decoding for Qwen3-Next on V1. — [vLLM PR #50172](https://github.com/vllm-project/vllm/pull/50172)
- vLLM RFC #45702: aligning full-attention block size to the Mamba state block size hurts hit rates; proposes finer hash block sizes / partial hits. — [vLLM issue #45702](https://github.com/vllm-project/vllm/issues/45702)
- Align-mode prefix caching for Qwen3.5/Next added on Ascend (PR #7103) and TPU (PR #3422); vllm-metal (PR #584) and Tenstorrent (issue #270) work on the same problem; the Tenstorrent proposal keys state snapshots on vLLM block hashes that include multimodal hashes. — [vllm-ascend #7103](https://github.com/vllm-project/vllm-ascend/pull/7103), [tpu-inference #3422](https://github.com/vllm-project/tpu-inference/pull/3422), [vllm-metal #584](https://github.com/vllm-project/vllm-metal/pull/584), [Tenstorrent #270](https://github.com/tenstorrent/vllm-tt-plugin/issues/270)
- Tail-Replay (arXiv 2608.30310, Aug 2026) addresses "the curse of linear attention in prefix caching for hybrid LLMs" (numbers not retrieved). — [arXiv 2608.30310](https://arxiv.org/pdf/2608.30310)
- vLLM blog on Qwen3-Next support (Sep 2025) describes the hybrid architecture serving support. — [vLLM blog](https://vllm.ai/blog/2025-09-11-qwen3-next)
- Uni-LaSpec reports DFlash, MTP and EAGLE-3 all running exact speculative decoding on Qwen3.5-4B/9B with images (numbers in section 1). — [arXiv 2611.01617](https://arxiv.org/html/2611.01617)

### Inferences
- For larch multi-turn image chats on Qwen3.5: snapshot the GDN state (in fp32) right after each image's tokens and after each completed turn; that gives exact reuse of the image prefix without keeping a long recurrent history replay, and is the hybrid analogue of image-KV caching. Multimodal content must be part of the cache key (image hash), as vLLM's block hashing does.
- Image KV eviction interacts badly with hybrid state: you cannot drop image tokens from GDN state after the fact, so "evict image after answer" can only shrink the full-attention layers' KV.

### Gaps
- No source found on image-specific issues (M-RoPE ids, image-hash cache keys) in hybrid-model prefix caching beyond the Tenstorrent design note.
- No measured memory cost per GDN state snapshot was found (the vLLM PR does not quantify it).
