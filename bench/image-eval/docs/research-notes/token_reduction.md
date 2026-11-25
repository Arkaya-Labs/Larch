# Visual token reduction and compression for VLMs (focus: text-heavy images, small models, runtime-only)

Scope note: researched 2026-11-11. Many 2026 sources are arXiv preprints (IDs 26xx.xxxxx) not yet peer reviewed. Dates are given per source. Numbers from different papers use different models, budgets and baselines and are not directly comparable. larch context used for inferences: Qwen 3.5 0.8B/2B (Qwen3-VL-style ViT, 2x2 merge, M-RoPE), InternVL3.5 1B/2B (448px tiles, pixel shuffle to 256 tokens/tile); 2B image chat: encode 506 ms + ~920 visual tokens prefill at ~3,000 tok/s (~0.3 s) gives ~1 s first token.

## Q1. Training-free pruning/merging (FastV, SparseVLM, PDrop, VisionZip, PruMerge, ToMe, DART, FasterVLM, HiRED, and newer): text-heavy vs general results, and which hurt OCR most

### Takeaway
Training-free pruning keeps general VQA (GQA/MME/POPE/MMB) close to baseline down to ~20-25% of tokens, but OCR, grounding and fine-text benchmarks (TextVQA, InfoVQA, ChartQA, DocVQA, RefCOCO) degrade first and fastest. In-LLM attention pruning at early layers (FastV-style) and ViT-internal merging (ToMe, SAINT) hurt OCR most. Diversity/duplication-based pre-LLM selection (VisionZip, DART, DivPrune, AIM) does much better. Several 2025-2026 studies find that importance scoring often does no better than random or spatially uniform sampling.

### Cited Findings
**Classic LLaVA-1.5-7B numbers (576 tokens baseline), from the VisionZip paper (arXiv 2412.04467, Dec 2024; v2 revised 2026):**
- Baseline TextVQA 58.2, GQA 61.9, POPE 85.9, MME 1862 — [VisionZip](https://arxiv.org/html/2412.04467)
- 192 tokens: FastV TextVQA 52.5 / POPE 64.8; SparseVLM 56.1 / 83.6; VisionZip 57.3 / 85.3 — [VisionZip](https://arxiv.org/html/2412.04467)
- 128 tokens: FastV 50.6 / 59.6; SparseVLM 54.9 / 80.5; VisionZip 56.8 / 83.2 — [VisionZip](https://arxiv.org/html/2412.04467)
- 64 tokens (89% cut): FastV 47.8 / 48.0; SparseVLM 51.8 / 75.1; VisionZip 55.5 / 77.0; VisionZip with a short projector fine-tune 56.0 / 80.9 — [VisionZip](https://arxiv.org/html/2412.04467)
- VisionZip reports ~98.5% average retention at 192 tokens, 97.6% at 128 and 95.2% at 64 (fine-tuned variant); about 8x prefill speedup on LLaVA-NeXT 7B (Table 4: 218 ms to 27.8 ms prefill, 3.0x total, 160 tokens) — [VisionZip](https://arxiv.org/html/2412.04467)
- VisionZip selects "dominant" tokens by vision-encoder [CLS]/attention and merges the rest into a few "contextual" tokens, all before the LLM. It does not reduce ViT encode cost — [VisionZip](https://arxiv.org/html/2412.04467)

**FastV failure mode on text and localization (Feather the Throttle, arXiv 2412.13180, Dec 2024, rev. Jul 2025; ICCV 2025):**
- FastV at layer K=3 with 75% of tokens dropped: TextVQA 54.9 to 31.8 (-42%), RefCOCO 60.9 to 6.7 (-89%), at a 68% FLOPs cut (SigLIP + Llama-2-7B VLM) — [Feather the Throttle](https://arxiv.org/html/2412.13180)
- Cause: RoPE long-range decay makes early-layer text-to-image attention favor tokens late in the sequence (bottom of the image). At R=0.75 the average kept token sits 80.7% of the way down the image — [Feather the Throttle](https://arxiv.org/html/2412.13180)
- Many benchmarks barely move under bad pruning. The authors argue these benchmarks do not test fine-grained visual ability — [Feather the Throttle](https://arxiv.org/html/2412.13180)
- The FEATHER fix (prune later, at layers 8 and 16, with a RoPE-free attention criterion plus uniform sampling) scored TextVQA 51.4 / RefCOCO 44.1 at a 64% FLOPs cut and 54.6 / 56.7 at 48% — [Feather the Throttle](https://arxiv.org/html/2412.13180)

**DART (duplication-aware, arXiv 2502.11494, Feb 2025; EMNLP 2025 main):**
- Picks a few pivot tokens and keeps the tokens least similar to them. It is training-free and compatible with FlashAttention because it needs no attention maps. It prunes 88.9% of vision tokens with "comparable performance", giving 1.99x total and 2.99x prefill speedup — [DART arXiv](https://arxiv.org/abs/2502.11494); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.505.pdf)
- The DART paper argues importance-based ranking is often worse than random pruning — [DART arXiv](https://arxiv.org/abs/2502.11494)
- On the same LLaVA-NeXT-7B setup (320 tokens, ~88.9% cut), the DART paper's table lists average retention of HiRED 91.8%, FasterVLM 89.8% and PruMerge 79.9%. Column labels were not recoverable in my extraction — [DART, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.505.pdf)

**HiRED vs PruMerge (arXiv 2408.10945, Aug 2024; AAAI 2025):**
- PruMerge and PruMerge+ keep about 10% and 55% of tokens on transcription tasks. With 55% of tokens, PruMerge+ is still 11% (TextVQA) and 26% (DocVQA) less accurate than HiRED at 20%. PruMerge at 10% is 13% and 37% less accurate. These are the HiRED authors' own comparisons — [HiRED](https://arxiv.org/pdf/2408.10945)
- PruMerge's adaptive (IQR-outlier) selection keeps very few tokens on document images, which explains the large DocVQA loss — [HiRED](https://arxiv.org/pdf/2408.10945); [LLaVA-PruMerge](https://arxiv.org/html/2403.15388v1)
- FasterVLM ranks tokens by [CLS]-to-patch attention in the vision encoder and prunes before the LLM — [PACT (describing FasterVLM)](https://arxiv.org/pdf/2504.08966)

**Modern models (Qwen2.5-VL-7B, LLaVA-OV-1.5-8B), "Rescuing Your Pruning", arXiv 2610.33688, 24 Oct 2026. Values are retention % vs dense:**
- Encoder-side ToMe at 40% keep: OCR 66.7, grounding 44.4, general 89.5. At 10% keep: OCR 24.9, grounding 4.5, general 76.2 — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)
- Encoder-side SAINT at 40% keep: OCR 68.5, grounding 45.1. At 10%: OCR 25.6, grounding 4.0 — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)
- Projector-side DivPrune at 40%: OCR 94.0, grounding 91.5, general 96.0. At 10%: OCR 63.8, grounding 27.2. AIM at 40%: OCR 94.1. At 10%: OCR 59.8 — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)
- At a 25% budget (general retention GRR / ATR): DART 88.9/68.1, VisionZip 86.9/80.6, PDrop 82.5/63.7, ToMe 85.7/57.6, PACT 92.5/81.6, VLM-Pruner 93.6/80.1, AIM 93.6/84.9, EADP 91.8/82.1 — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)
- Summary statement: "training-free pruning largely preserves coarse-grained understanding but sharply degrades OCR and grounding" — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)

**Text-aware two-stage pruning ("When Text Matters", KAIST, arXiv 2609.34861, 28 Sep 2026):**
- Prunes before the LLM using vision-encoder attention but keeps a 20% reserve. It then makes the final selection with text-to-visual attention at the decoder midpoint (L/2), because text-to-visual attention is weak in early layers — [When Text Matters](https://arxiv.org/html/2609.34861v1)
- Qwen3-VL-8B at 80% pruning: TextVQA 80.48 vs 82.95 dense, but InfoVQA 64.57 vs 81.19 dense (-16.6 points). LLM prefill 121 ms vs 227 ms — [When Text Matters](https://arxiv.org/html/2609.34861v1)
- At 90% pruning it retains 90.65% of dense performance on Qwen3-VL-8B, with all baselines (FastV, SparseVLM, VisPruner, ZOO-Prune, RESTORE) below 70%. Its margin over the best baseline is +11.1 points at 80% and +16.8 at 90%. Also tested on Qwen3-VL-4B — [When Text Matters](https://arxiv.org/html/2609.34861v1)
- Image-only selection can drop the region holding the requested text. Text-guided selection can keep the surrounding context but lose the value itself. The biggest losses are on TextVQA, ChartQA and InfoVQA — [When Text Matters](https://arxiv.org/html/2609.34861v1)

**Other 2025-2026 observations:**
- A CVPR 2026 study ("When Token Pruning is Worse than Random") finds that past a certain decoder depth, training-free methods do no better than random pruning, because of "vanishing token information". That depth is deeper for OCR than for general VQA — [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.pdf); [arXiv 2512.07580](https://arxiv.org/html/2512.07580v1)
- ResPrune reports that every pruning method retains less at the same ratio when moving from LLaVA to Qwen2.5-VL — [ResPrune, arXiv 2603.21105](https://arxiv.org/pdf/2603.21105)
- EADP likewise reports larger degradation on Qwen2.5-VL than on LLaVA — [EADP, arXiv 2607.02484](https://arxiv.org/pdf/2607.02484)
- With InternVL2-2B guiding InternVL2-26B, ToMe and FastV start dropping once retention falls to 35%, "particularly in OCR-related tasks, including TextVQA, ChartQA, and DocVQA" — [A Stitch in Time / SGP, arXiv 2412.03324](https://arxiv.org/pdf/2412.03324)
- An audit paper warns that accuracy can hide lost evidence: an answer can stay correct even when no kept token traces back to the supporting OCR region. DocVQA questions with multi-token answers are a hard case — [Auditing spatial provenance, arXiv 2608.00077](https://arxiv.org/html/2608.00077)
- ET-Prune (arXiv 2608.01979, Aug 2026) proposes evidence-aware dynamic budgets specifically for text-rich inputs. I could not extract its numbers (PDF too large to fetch) — [ET-Prune](https://arxiv.org/pdf/2608.01979)

### Inferences
- Methods ranked by OCR safety, best first: (1) diversity/duplication-based pre-LLM selection (DivPrune, DART, AIM, VisionZip) and text-aware mid-layer selection; (2) late-layer in-LLM pruning (FEATHER-style, mid-depth); (3) PyramidDrop/SparseVLM; (4) early-layer FastV; (5) ViT-internal merging (ToMe/SAINT) and PruMerge-style adaptive selection, which are worst on OCR and grounding.
- No method in the sources is near lossless on InfoVQA/DocVQA-style dense text below ~40% retention without training. A screenshot with fine UI text resembles InfoVQA/DocVQA far more than GQA, so headline "95% retained at 11% tokens" claims (measured mostly on LLaVA-1.5 general benchmarks) should not be trusted for larch's workload.
- Attention-based methods (FastV, SparseVLM, FasterVLM-in-LLM) need materialized attention scores. That conflicts with fused/flash attention kernels in larch and, in the GUI study, nearly doubled peak memory (5.1 GB to 9.1 GB). On a 4 GB card this matters more than the speedup. DART, DivPrune, VisionZip-style selection and random/uniform sampling need no attention map.

### Gaps
- PyramidDrop and SparseVLM numbers on DocVQA/ChartQA/OCRBench for Qwen-family models were not collected; only LLaVA-1.5 TextVQA numbers (via VisionZip) were.
- TokenCarve: no TextVQA/OCR numbers found in this session.
- ET-Prune's tables could not be fetched (file too large).

## Q2. Compatibility with M-RoPE / Qwen2-VL / 2.5-VL / 3-VL position layouts

### Takeaway
The near-universal practice is to drop tokens but keep each survivor's original (t, h, w) M-RoPE position ids, with no re-indexing. Text after the image continues from max position + 1 as in unpruned inference. This works for VQA but can degrade fine spatial localization. Fixes include keeping a marker token per pruned run (FocusUI POSPAD), per-axis compaction (HERMES, for video KV reuse), and spatially uniform selection.

### Cited Findings
- Qwen2-VL M-RoPE: for image tokens the temporal id is constant and the height/width ids come from the token's 2D grid position. Text after the image continues from the maximum position + 1 (transformers `get_rope_index`) — [Qwen2-VL paper](https://arxiv.org/html/2409.12191v1); [Spark NLP issue #14904](https://github.com/JohnSnowLabs/spark-nlp/issues/14904)
- GUI-agent pruning study (Qwen2.5-VL-3B, Qwen3-VL-2B): retained tokens keep their original position indices and are not re-indexed. The authors argue, for RoPE and for M-RoPE / interleaved M-RoPE (appendix), that retained tokens concentrated in one area weaken global position inference. This favors spatially spread selection — [Where and How to Prune, arXiv 2603.26041](https://arxiv.org/html/2603.26041)
- FocusUI: direct pruning creates positional jumps in the (h, w) M-RoPE dimensions and causes "pronounced localization offsets on fine-grained targets". Its POSPAD replaces each run of dropped tokens with one learnable marker carrying the (h, w) position of the run's last index. This requires training — [FocusUI, arXiv 2601.03928](https://arxiv.org/pdf/2601.03928)
- HERMES (Qwen2.5-VL, video) instead applies per-axis left-compaction of the retained (t, h, w) coordinates and an M-RoPE correction to reuse cached keys — [HERMES, arXiv 2601.14724](https://arxiv.org/pdf/2601.14724)
- vLLM PR #38888 adds attention-score-based image token pruning for Qwen VL models, with M-RoPE position "reattachment" after pruning and a fallback that recomputes positions for pruned placeholders. This is a concrete engine-level reference implementation — [vLLM PR #38888](https://github.com/vllm-project/vllm/pull/38888)
- "When Text Matters" (Qwen3-VL) does not describe its position-id handling — [When Text Matters](https://arxiv.org/html/2609.34861v1)
- GlimpsePrune (Qwen2.5-VL) does not describe M-RoPE handling either — [GlimpsePrune](https://arxiv.org/html/2508.01548v1)

### Inferences
- For larch, the low-risk implementation is: compute the full M-RoPE (t, h, w) ids for the unpruned grid, gather the ids of the kept rows, and leave text positions unchanged (still max + 1 of the full grid). This needs only an index gather in the projector-output and position buffers, and no kernel changes to RoPE.
- Pre-LLM pruning (after the 2x2 merger) with preserved positions is the cleanest fit. In-LLM pruning at layer K requires compacting the KV cache mid-prefill, which is a much more invasive change to the prefill pipeline.
- InternVL3.5 uses plain 1D RoPE positions over flattened tile tokens, so pruning there is simpler (gather the ids). But the attention-bias issue (bottom-of-image preference) applies, per Feather the Throttle.

### Gaps
- No source directly compared preserved ids, compaction and marker tokens on Qwen2-VL/3-VL OCR benchmarks.
- I did not verify how vLLM's PR reattaches positions in detail, or whether it was merged.

## Q3. Resolution budgeting: dynamic or adaptive resolution, ViR, Qwen min/max pixels, per-image or per-question budgets

### Takeaway
For text-heavy images, setting the input resolution or token budget at the image level is the most reliable training-free lever, and it is the only lever that also cuts ViT encode cost (most of larch's TTFT). Pruning inside or after the encoder does not reduce encode cost. InternVL3.5-Flash's ViR (a trained router) halves visual tokens at "nearly 100%" performance, but per-size Flash numbers were not found. GUI-agent evidence shows mild downscaling (x0.8) costs ~1%, while patch removal costs more.

### Cited Findings
- Qwen2.5-VL defaults to native resolution with 4 to 16,384 visual tokens per image. The model cards recommend `min_pixels`/`max_pixels` to trade cost against accuracy (an example range is 256-1280 tokens). They also note that "higher resolutions can enhance performance at the cost of more computation" — [Qwen2.5-VL-3B model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct); [Qwen2.5-VL-7B card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- InternVL3.5 ViR: each 448px patch yields 1024 ViT tokens. Standard pixel shuffle compresses these to 256 (1/4). Flash adds a second pixel shuffle to 64 (1/16), and a per-patch binary router picks the rate based on "semantic richness" — [InternVL3.5 paper](https://arxiv.org/html/2508.18265v2)
- ViR training (ViCO): a consistency stage minimizes the KL divergence between outputs at 1/16 and at 1/4 (frozen reference), with the rate sampled uniformly. The router is then trained as a binary classifier on per-patch loss ratios, mostly on OCR/VQA data — [InternVL3.5 paper](https://arxiv.org/html/2508.18265v2)
- Claimed result: 50% fewer visual tokens with "nearly 100%" of InternVL3.5 performance. The 4.05x speedup headline is vs InternVL3 and combines ViR with DvD (decoupled vision/language servers), so it is not attributable to ViR alone — [InternVL3.5 paper](https://arxiv.org/html/2508.18265v2); [InternVL3.5-2B-Flash card](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Flash)
- A 2B Flash checkpoint exists (OpenGVLab/InternVL3_5-2B-Flash). Its card shows no 2B-specific Flash vs non-Flash OCR/Doc numbers. A secondary summary cites a 38B Flash vs non-Flash comparison of 83.9 vs 83.4 — [InternVL3.5-2B-Flash card](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Flash); secondary: [liner review](https://liner.com/review/internvl35-advancing-opensource-multimodal-models-in-versatility-reasoning-and-efficiency)
- GUI navigation (Qwen3-VL-2B, AITW, upper bound 0.7035): downscaling the screenshot to x0.8 scored 0.6965 (-1.0% relative), and pre-ViT patch removal (36% of patches) scored 0.6859 (-2.5%). Caching ViT features for an unchanged screenshot gave 0.7030 and cut encoder latency from 212 to 70 ms/step (-67%) — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- In the same study, LLM-side pruning at layer 4 with R=50% kept ~99.7% of performance but cut LLM prefill only ~24% and wall clock only ~4.2%. At R=20%: ~98.6% kept, prefill -37.8%, wall clock -5.4%. The encoder and decode dominate the remaining time — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Qwen2.5-VL-7B on MATH (rendered text) needed enough resolution: 224x224 (16 tokens) lost accuracy "due to insufficient text clarity", while 1024x1024 (~256 tokens, as reported) gave the best balance (76.3%). This is indirect evidence, not an OCR benchmark — [Cross-modal memory compression, arXiv 2602.00454](https://arxiv.org/pdf/2602.00454)
- DocVLM (Qwen2-VL): replacing most visual tokens with OCR-text encodings, using 128 visual tokens plus 800 OCR tokens, reached 84.3% DocVQA with OCR words and 90.1% with OCR encodings. This shows a text channel can substitute for visual resolution, but it needs a trained module — [DocVLM, arXiv 2412.08746](https://arxiv.org/pdf/2412.08746)
- GlimpsePrune (Qwen2.5-VL, lightweight-trained predictor, base model frozen) chooses retention per image and question. It kept only 3.6% of tokens on DocVQA (0.962 vs 0.964 baseline) and 7.1% on TextVQA (0.939 vs 0.927), with an average of 7.4%. On 100 DocVQA samples, the KV length fell from 5,074 to 203 tokens and prefill FLOPs from 77.8T to 53.8T — [GlimpsePrune, arXiv 2508.01548](https://arxiv.org/html/2508.01548v1)
- Under the same 11.1% cap, VisionZip and VScan scored only ~0.66-0.78 on TextVQA and ran out of memory on several high-resolution datasets — [GlimpsePrune](https://arxiv.org/html/2508.01548v1)

### Inferences
- In larch's 2B path the ViT (506 ms) costs more than LLM prefill of ~920 tokens (~0.3 s at 3,000 tok/s). Any method that prunes after the encoder can save at most ~0.3 s of the ~1 s TTFT. A smaller resolution cap (Qwen `max_pixels`, fewer InternVL tiles) cuts encode (super-linearly, because ViT attention is quadratic in each window or full image) and prefill together. However, it directly costs fine-text legibility on small models that already misread fine text.
- Per-question budgeting (a GlimpsePrune-style query-aware choice) gives the best reported trade-off but requires a trained predictor. A training-free approximation would be heuristic tiers: full resolution when the prompt asks to read, transcribe or quote text, and a lower cap for "describe this" prompts. No source evaluates such a heuristic.
- Encoder-output caching for repeated screenshots (multi-turn about the same image, or agent loops with unchanged screens) is lossless and gives the largest encoder win measured in the GUI study (-67% encoder latency).
- InternVL3.5-2B-Flash is a drop-in checkpoint that halves tokens, but larch would have to implement the router MLP and a second pixel-shuffle path. Its OCR deltas at 1B/2B are unverified.

### Gaps
- No primary-source table of InternVL3.5-Flash vs non-Flash at 1B/2B on DocVQA/OCRBench was found; the paper's Table 4 lists only standard models.
- I did not find a 2024-2026 "less is more resolution" study with OCR numbers across max_pixels settings for Qwen2.5-VL/3-VL small models.
- Qwen3-VL technical report details on default pixel limits were not fetched in this session.

## Q4. Token compression trained into the model (pixel shuffle, Q-Former, resamplers, TokenPacker, DeepSeek-OCR): relevance when you cannot retrain

### Takeaway
Trained compressors set the token rate per model. A runtime engine can only use the rates the checkpoint was trained with: Qwen's fixed 2x2 merge, InternVL's 1/4 pixel shuffle, and Flash's optional 1/16 path. DeepSeek-OCR shows ~10x text-to-vision-token compression is near-lossless (96%+ precision) only for a model trained for it, falling to ~60% at 20x. Runtime-only options are therefore picking checkpoints (e.g., Flash) and resolution, not new compressors.

### Cited Findings
- DeepSeek-OCR (arXiv 2510.18234, Oct 2025) reports 96%+ OCR decoding precision at 9-10x text-to-vision-token compression, ~90% at 10-12x and ~60% at ~20x on Fox (documents of 600-1,300 text tokens). Secondary sources round the first figure to "97%" — [DeepSeek-OCR](https://arxiv.org/html/2510.18234v1); secondary: [alphaxiv overview](https://alphaxiv.org/overview/2510.18234v1)
- The ~20x point corresponds to 64 vision tokens (19.7x), per secondary summaries — [emergentmind](https://www.emergentmind.com/topics/deepseek-ocr)
- InternVL3.5's pixel shuffle compresses 1024 ViT tokens to 256 per 448px tile. The 1/16 path (64 tokens) needed dedicated consistency training (ViCO) to be usable — [InternVL3.5 paper](https://arxiv.org/html/2508.18265v2)
- Training a projector briefly after pruning recovers some accuracy: VisionZip's fine-tuned variant raised 64-token TextVQA from 55.5 to 56.0 and POPE from 77.0 to 80.9 — [VisionZip](https://arxiv.org/html/2412.04467)
- "Rescuing Your Pruning" studies post-training recovery of OCR and grounding after pruning, which is out of scope for a runtime-only engine — [Rescuing Your Pruning](https://arxiv.org/html/2610.33688)

### Inferences
- For larch, the trained-compression ideas worth importing are checkpoint choices (InternVL3.5-Flash) and possibly DeepSeek-OCR-style dedicated OCR models for document transcription. None of these is a runtime toggle on the existing Qwen 3.5 / InternVL3.5 weights.
- Applying an untrained extra pixel shuffle or 2x2 average-pool at runtime is the same in spirit as ToMe/SAINT-style merging. The encoder-side merging numbers above (OCR retention ~67% at 40% keep) suggest it would hurt fine text badly.

### Gaps
- Q-Former, Perceiver-resampler and TokenPacker numbers on OCR benchmarks were not gathered in this session. My prior knowledge is that Q-Former-style fixed-count resamplers underperform on text-rich tasks, but I did not verify this against a source.

## Q5. Small (≤3B) models vs 7B+

### Takeaway
Direct small-vs-large comparisons under identical pruning are scarce. The evidence that exists suggests stronger, more visually dependent models (Qwen2.5-VL vs LLaVA) lose more from pruning, while on 2-3B GUI agents, moderate pruning (50%) costs <1%, with random sampling matching importance methods. Small-model results on OCR benchmarks under pruning were not found.

### Cited Findings
- Qwen3-VL-2B on AITW at R=50%: upper bound 0.7035, Random 0.7027, FastV 0.6996, SparseVLM 0.6996, DART 0.6987, PDrop 0.6964, DivPrune 0.6957 — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Qwen3-VL-2B on AITW at R=10%: Random 0.6842 vs FastV 0.6824 — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Qwen2.5-VL-3B: Mind2Web 0.5055 upper bound vs 0.4987 Random / 0.4979 FastV; AndroidControl 0.7113 vs 0.7115 Random / 0.7084 FastV — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Pruning only the current screenshot's tokens to 50% (FastV, AITW) dropped the score from 0.7035 to 0.6612 (-6%). Most of the "safe" headline result came from pruning history screenshots — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Background patches are ~64% of screenshot patches. Keeping background-only beat keeping foreground-only, and spatially uniform random pruning was competitive with or better than importance methods — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- In the same study, FastV and SparseVLM raised peak memory from 5,067 MB to 9,061 MB and 8,287 MB, because they materialize attention maps — [Where and How to Prune](https://arxiv.org/html/2603.26041)
- Stronger models (Qwen2.5-VL-7B) draw information from visual tokens at deeper layers than weaker ones (LLaVA-1.5-7B), so pruning at a fixed early layer removes more useful information. Qwen2.5-VL-7B still reaches ~80% on TextVQA/OCRBench using visual tokens up to layer 27 — [arXiv 2512.07580](https://arxiv.org/html/2512.07580v1)
- All pruning methods retain less on Qwen2.5-VL than on LLaVA at equal ratios — [ResPrune](https://arxiv.org/pdf/2603.21105); [EADP](https://arxiv.org/pdf/2607.02484)
- GlimpsePrune is evaluated on Qwen2.5-VL-3B and 7B (trained predictor). Its document-task retention figures above are for 7B — [GlimpsePrune](https://arxiv.org/html/2508.01548v1)
- "When Text Matters" includes Qwen3-VL-4B, the smallest model in that study — [When Text Matters](https://arxiv.org/html/2609.34861v1)

### Inferences
- larch's small models already misread fine text at full resolution, which means they have little visual redundancy to spare on text regions. Pruning would most likely add errors exactly where they already fail. For OCR-heavy prompts, the safe default is no pruning, or very mild pruning (≥50% kept) that is spatially uniform or diversity-based and preserves positions.
- GUI-agent success metrics are coarse (click or step accuracy). The near-zero losses at 50% on 2-3B models may not transfer to exact-string reading tasks; compare the OCR retention numbers in Q1.
- The most promising training-free direction for larch's 0.8B/2B is: (a) encoder-output caching, (b) prompt-aware resolution tiers, and (c) for non-OCR prompts only, pre-LLM DART/DivPrune-style or uniform selection at 25-50% with preserved M-RoPE ids. Expected prefill savings at 50% are ~0.15 s per 920-token image on the 2B. Encode time is unchanged unless the resolution changes.

### Gaps
- No paper found reporting TextVQA/DocVQA/OCRBench under pruning for Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen3-VL-2B or InternVL 1B/2B side by side with their 7B+ counterparts.
- No numbers found for pruning on Qwen 3.5 small VLMs specifically (larch's models). Results for Qwen2.5-VL and Qwen3-VL are the nearest proxies.
