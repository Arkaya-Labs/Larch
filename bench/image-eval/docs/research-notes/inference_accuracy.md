# Inference-time accuracy techniques for small VLMs (<=4B) on screenshots, documents, UI and fine text

Researched 2026-11-11. About 18 tool calls. Tables were fetched from arXiv HTML pages where possible. Several key papers (Zoom Eye, OPERA, ICD, DeepEyes, V*) could not be fetched within the budget and appear under Gaps with pointers. Most published inference-time methods were measured on 7B models (LLaVA-1.5-7B, InstructBLIP-7B, Qwen-VL-7B). Results on 3B or smaller are flagged explicitly where they exist.

## 1. Resolution, token budget, cropping and zoom

### Takeaway
For fine text, the visual token budget is the strongest lever, with diminishing or negative returns past about 1,600 tokens on OCRBench (measured at 7B). A crop-and-re-ask pass that feeds the original image plus a zoomed crop gives +2 to +7 points on TextVQA/DocVQA/InfoVQA for Qwen2.5-VL-3B with no training, and more with a small learned cropper. Small models gain the most when the base resolution is low.

### Cited Findings
- **Token budget vs OCR (Qwen2-VL-7B, so not a small model).** Fixed 64 tokens gives OCRBench 572 and InfoVQA 28.85. 576 tokens gives 828 / 65.72. **1600 tokens gives 824 / 74.99.** 3136 tokens gives 786 / 77.27. Dynamic resolution (about 1,924 tokens on average) gives **866 / 75.89**. MMMU is flat (about 53) at every budget. So OCRBench peaks around 576 to 1,600 fixed tokens and drops at 3,136, while InfoVQA (dense documents) keeps improving with more tokens. — [Qwen2-VL report, arXiv 2409.12191, Table 7](https://arxiv.org/html/2409.12191v1)
- **Upscaling small images (min_pixels), also from Qwen2-VL.** Upscaling small images within a reasonable range improves InfoVQA, HallusionBench and OCRBench. An overly high min_pixels causes a *severe* OCRBench decline, because tiny images become out of distribution once enlarged. MMMU is unaffected. The exact values are only in Figure 4, which I could not read. — [Qwen2-VL report](https://arxiv.org/html/2409.12191v1)
- Qwen2-VL-2B reference scores: DocVQA test 90.1, TextVQA val 79.7, OCRBench 809, ChartQA 73.5. — [Qwen2-VL report, Table 2](https://arxiv.org/html/2409.12191v1). A search snippet gave OCRBench 794 for the 2B, which conflicts. 809 is the paper's number.
- **Resolution matters most for tiny models.** SmolVLM-256M, baseline averaged over TextVQA/DocVQA/InfoVQA/ST-VQA: 28.45 at 512px, 44.55 at 1024px, 50.16 at 2048px. DocVQA alone goes 13.68, then 47.86, then 60.13. — [CropVLM, arXiv 2511.19820 (Nov 2025, CVPRW 2026), Table 3](https://arxiv.org/html/2511.19820v2)
- **ViCrop (training-free; "MLLMs know where to look", ICLR 2025, Feb 2025).** It uses the model's own attention or gradient maps to choose a crop. The crop's tokens are *concatenated to the original image tokens*, so the model keeps the global view. Results on LLaVA-1.5-**7B** with rel-att: TextVQA 47.80 → 55.17, V* 42.41 → 62.30, POPE 85.27 → 87.25, DocVQA 15.97 → 19.63. General VQA benchmarks (AOKVQA, GQA, VQAv2) are flat or slightly up, with no regressions. On InstructBLIP-7B, TextVQA goes 33.48 → 45.44. On LLaVA-NeXT-7B (a higher-resolution model), TextVQA goes 65.17 → 68.65 and V* 58.11 → 61.78. Overhead is constant because the method only needs the first answer token: on an RTX A6000, rel-att adds 1.16 s and grad-att 0.89 s, against 0.17 s per answer token. On CPU, rel-att adds 14.4 s. For very large images, a two-stage strategy splits the image into blocks under 1024², scores each block, then crops. — [ViCrop, arXiv 2502.17422](https://arxiv.org/html/2502.17422v1)
- **ViCrop on Qwen2.5-VL-3B at 448px (third-party run, so it applies to a small model).** TextVQA 68.20 → 72.91 (rel-att) or 74.15 (grad-att). DocVQA 69.52 → 71.40 / 72.27. InfoVQA 44.03 → 47.94 / 49.43. ST-VQA 65.49 → 68.96. V* 51.31 → 53.40. HR-Bench-4k 51.88 → 56.25. The 7-benchmark average goes 56.42 → 59.75. Box-generation cost on Qwen2.5-VL is 396 to 525 ms and about 19 GB peak memory on an A6000. ViCrop needs attention or gradient access, which is heavy. — [CropVLM, Tables 4 and 6](https://arxiv.org/html/2511.19820v2)
- **CropVLM (a learned 256M SmolVLM cropper, plug-in, target VLM frozen).** With Qwen2.5-VL-3B at 448px, the average goes 56.42 → 67.14 (2048px cropper). DocVQA goes 69.52 → 84.41, InfoVQA 44.03 → 55.95, and HR-8k 44.50 → 60.75–63.63. The cropper costs 418 to 691 ms per box and only 1.2 to 1.7 GB peak memory. It beats ViCrop on every dataset except ST-VQA. On GPT-4.1-nano, the average goes 41.27 → 47.41. With SmolVLM-256M as both cropper and answerer, the 1024px cropper (50.89) beats the 2048px no-crop baseline (50.16). — [CropVLM](https://arxiv.org/html/2511.19820v2)
- **Learned resolution selection.** CARES uses a 350M VLM to predict the minimal resolution at which the target VLM still answers correctly. It reports "up to 80%" compute reduction while preserving task performance on 5 doc and natural-image benchmarks. The abstract gives no per-benchmark numbers. arXiv Oct 2025, v3 May 2026, ACL 2026. — [CARES, arXiv 2510.19496](https://arxiv.org/abs/2510.19496)
- Qwen2.5-VL visual tokens grow roughly quadratically with side length: 672² gives 576 tokens and 1024² gives 1,369 tokens. — [CARES via search snippet](https://arxiv.org/pdf/2510.19496)
- Token pruning hurts OCR at aggressive budgets. On Qwen2.5-VL-7B with 10% of tokens kept, PACE reaches OCRBench 70.90%, POPE 84.99% and MMStar 59.08%. I did not see the unpruned baseline in the snippet. — [PACE, arXiv 2608.27206](https://arxiv.org/pdf/2608.27206)
- A RL-trained zoom tool on Qwen2.5-VL-3B (GRPO, TextVQA data) learned to use zoom but generalized poorly. — [arXiv 2506.14821 (search snippet)](https://arxiv.org/pdf/2506.14821)

### Inferences
- larch's 1,600-token Qwen cap is near the OCRBench sweet spot from the 7B ablation. Raising it mainly helps dense documents (InfoVQA-like content such as receipts, spreadsheets and wide banners), not scene-text OCR. A per-image adaptive budget looks better than a single cap.
- The cheapest high-value addition for a local engine is a crop-and-re-ask pass that appends a zoomed crop to the original image. Region choice can come from (a) attention maps from the first answer token (ViCrop rel-att needs only forward attention, at the cost of one extra prefill plus attention extraction), (b) a tiny cropper model (256M class), or (c) deterministic tiling for screenshots. Gains at 3B are +2 to +5 points training-free and +8 to +15 points on DocVQA with a learned cropper.
- Do not upscale already-legible screenshots aggressively. The Qwen2-VL min_pixels result shows that over-enlarging small images hurts OCR. Upscale only when the text glyph height falls below what one 14px patch (28px after 2×2 merge) can resolve.

### Gaps
- No max_pixels sweep for Qwen2.5/3/3.5 at ≤4B on OCRBench or DocVQA was found. The available curve is for 7B.
- Zoom Eye (arXiv 2411.16044), V* / SEAL (arXiv 2312.14135), DeepEyes (arXiv 2505.14362) and "thinking with images" (o3-style) numbers were not fetched. DeepEyes is RL-trained at 7B, so it is not training-free. Zoom Eye is a training-free tree search with many VLM calls per query, and its small-model numbers were not verified.
- No source covered InternVL 448px tiles plus a global thumbnail ablation at 1 to 2B, nor HiDPI downscale choices (for example 2× Retina screenshots).

## 2. Decoding: hallucination-mitigating, repetition control, constrained

### Takeaway
Contrastive decoding methods (VCD, M3ID) give about +2 to +4 points on yes/no hallucination benchmarks (POPE/MME). They were measured at 7B, roughly double per-token compute, and can flip correct answers to wrong ones. No evidence was found that they help OCR or transcription at ≤4B. For small Qwen models, the vendor-documented lever against loops is `presence_penalty` (0 to 2) plus streaming loop detection.

### Cited Findings
- **VCD and variants on POPE/MME** (LLaVA-1.5, InstructBLIP and Qwen-VL; sizes not stated in the re-evaluation, and the originals are 7B). Overall accuracy: original 0.785, VCD (noised image) 0.810, M3ID (no image as contrast) 0.821, *downsampled image as contrast* 0.824, entropy-weighted fusion of all contrasts 0.833. For Qwen-VL, POPE-random goes 0.847 → 0.865 with VCD. VCD-style methods "flip more correct answers to wrong" than the reverse, with downsampling and diffusion noise flipping the most. The evaluation looks only at the first token of yes/no answers, so it says little about free-form generation. — [Delve into VCD, arXiv 2412.06775 (Dec 2024)](https://arxiv.org/html/2412.06775v1)
- VCD-family gains vary by benchmark and model, and Qwen-VL behaves differently from the other two models tested. — [arXiv 2412.06775](https://arxiv.org/html/2412.06775v1)
- POPE is fragile to setup changes. On Qwen2.5-VL, putting the prompt inside the image raised POPE from 80.2% to 84.3%. On LLaVA-1.5, the same change dropped POPE from 84.0% to 55.0%. — [Cure or Poison?, arXiv 2508.01678](https://arxiv.org/html/2508.01678v1)
- **Qwen3.5-0.8B official sampling (model card).** The 0.8B is "more prone to thinking loops than other Qwen3.5 models", and the card recommends streaming so loops can be detected and interrupted. Recommended settings, all with top_k 20, min_p 0 and repetition_penalty 1.0:

  | Mode | temperature | top_p | presence_penalty |
  |---|---|---|---|
  | Non-thinking, VL | 0.7 | 0.8 | 1.5 |
  | Non-thinking, text | 1.0 | 1.0 | 2.0 |
  | Thinking, VL | 0.6 | 0.95 | 0.0 |

  presence_penalty can range from 0 to 2. Higher values may cause language mixing and slightly lower performance. Thinking is off by default, and the `/think` and `/nothink` soft switches are not officially supported. — [Qwen/Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- A runtime caveat: Ollama accepted presence, frequency and repeat penalties for Qwen3.5 and Qwen3-VL but silently discarded them, so the reported loops partly came from the runtime. — [ollama issue #14493](https://github.com/ollama/ollama/issues/14493)
- Community reports are mixed (anecdotal). One user says presence_penalty 0.3 reduced verbose reasoning loops. Another found the recommended penalties hurt output. — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1ryb028/snapshots/2026-03-20T18%3A55%3A41.336186Z)

### Inferences
- larch's "describe in detail" loop on the 0.8B with thinking off matches the documented non-thinking VL setting (presence_penalty 1.5, temperature 0.7, top_p 0.8). Verify that the engine actually applies presence_penalty, given the Ollama precedent. Add n-gram or line-repeat loop detection with early stop as a backstop.
- For transcription (CER-scored), presence_penalty penalizes legitimately repeated tokens such as table cells, digits and repeated words, so it may raise CER. A task-dependent penalty is a reasonable design: about 0 for transcribe or extract, and about 1.5 for free description. This is untested and derived from the card's "slightly lower performance" note.
- VCD, M3ID and OPERA need a second forward pass per token (VCD, M3ID) or beam search with attention inspection (OPERA). That roughly doubles decode cost on a 4 GB GPU. The evidence covers object hallucination at 7B, not text reading, so it is low priority for larch. The downsampled-image contrast (0.824) is the cheapest variant to try because no noise model is needed.

### Gaps
- OPERA (arXiv 2311.17911), ICD (arXiv 2403.18715), DoLa-for-VLM and M3ID original CHAIR numbers were not fetched. All were originally reported on 7B models.
- No source tested VCD-family decoding on OCR or transcription, or on ≤4B models such as Qwen2-VL-2B or InternVL 1–2B.
- No benchmark numbers were found for grammar or JSON-constrained decoding on receipt or form extraction (for example SROIE or CORD) with small VLMs.

## 3. Thinking / reasoning modes and test-time scaling

### Takeaway
For OCR and document parsing, thinking *hurts* the 0.8B: OCRBench drops from 79.1 to 74.5 and OmniDocBench from 70.6 to 61.0. It helps reasoning and hallucination-QA benchmarks such as HallusionBench (+6.4) and MMMU (+1.6). Longer reasoning chains reduce visual attention, and 3B reasoning models are less stable than 7B ones. Best-of-N with a process reward model gives +4 to +8 points, but only at 7B+ with an 8B critic.

### Cited Findings
- **Qwen3.5-0.8B, thinking vs non-thinking** (official card):

  | Benchmark | Thinking | Non-thinking |
  |---|---|---|
  | OCRBench | 74.5 | **79.1** |
  | CC-OCR | 63.2 | **66.7** |
  | OmniDocBench 1.5 | 61.0 | **70.6** |
  | MMMU | **49.0** | 47.4 |
  | MathVista-mini | **62.2** | 58.6 |
  | HallusionBench | **53.1** | 46.7 |
  | MMLongBench-Doc | **33.6** | 28.1 |
  | CharXiv-RQ | **41.3** | 38.2 |
  | RealWorldQA | **63.4** | 61.6 |
  | MMStar | **58.3** | 55.9 |

  ScreenSpot-Pro is 46.5 non-thinking, with no thinking number reported. — [Qwen3.5-0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **"More Thinking, Less Seeing?"** (NeurIPS 2025, arXiv May 2025). Reasoning models derived from Qwen2.5-VL-3B and 7B hallucinate more than their base models. The perception ordering is base > RL-only > SFT+RL. Accuracy vs reasoning length is non-monotonic: perception and hallucination tasks peak at short or moderate length, while math benefits from longer chains. Forcing zero thinking also hurt. Visual attention decreases as chains grow. The 3B reasoning models have RH-AUC 0.46–0.53 and perception accuracy 48.7–52.8. The 7B models have RH-AUC 0.46–0.63 and perception accuracy 55.7–63.3. The 7B models "are more stable as reasoning chains get longer, while smaller models decline noticeably." — [arXiv 2505.21523](https://arxiv.org/html/2505.21523v2)
- Prompted CoT improves clean OCR-reasoning accuracy but does not prevent worst-case failures under perturbation. Qwen3-VL-8B-Thinking has the best clean OCR2.0 accuracy (78.18%), but its worst-case ratio is 0.624. This is an 8B model. — [OCR-Robust, arXiv 2606.26041](https://arxiv.org/html/2606.26041)
- VISTA-Bench reports that highly structured prompts such as CoT can hurt on visualized text. — [arXiv 2602.04802](https://arxiv.org/html/2602.04802v1)
- Gemini 2.5 Flash on Korean KIE with OCR text: 212 without thinking and 184 with thinking. 25 KIE errors were spacing mistakes. — [arXiv 2510.02543](https://arxiv.org/html/2510.02543)
- **VisualPRM (Best-of-N with an 8B process reward model).** Best-of-8 gains: InternVL2.5-8B 32.8 → 41.2 (+8.4), Qwen2.5-VL-7B +3.7, MiniCPM-V2.6 +8.0, InternVL2.5-78B +5.9. The PRM beats self-consistency by 2.4 points and an ORM by 1.5 at N=8 on InternVL2.5-8B. InternVL2.5-8B is a poor critic of itself. ORM Best-of-128 underperforms Best-of-64 (the N=128 figures come from a secondary summary). The benchmarks are reasoning benchmarks (MathVista, MMMU and others), not OCR. — [VisualPRM, arXiv 2503.10291](https://arxiv.org/html/2503.10291v1); [InternVL blog](https://internvl.github.io/blog/2025-03-13-VisualPRM)

### Inferences
- Route by task. Keep thinking off for transcription, OCR, UI reading and document parsing. Allow a short thinking budget only for reasoning questions over the image (charts, math, "why is this error happening"). On the 0.8B, thinking costs 4.6 OCRBench points and 9.6 OmniDocBench points.
- Best-of-N with an 8B verifier does not fit a 4 GB GPU next to the policy. Self-consistency (majority vote over N samples) is feasible for short answers (QA, field extraction), but no OCR-specific gain at ≤4B was found. For transcription, voting is ill-defined. A per-line or per-field vote across 3 samples, or across the original plus a zoomed crop, is a plausible cheap variant, but it is untested.

### Gaps
- No source measured self-consistency or best-of-N on OCRBench, DocVQA or ScreenSpot at ≤4B.
- No thinking-budget sweep (as opposed to on/off) for Qwen3.5 2B or InternVL3.5 1–2B on OCR was found.
- The Qwen3.5-2B card's thinking vs non-thinking table was not fetched. It likely shows a similar pattern, but this is unverified.

## 4. Prompting (system prompts, read-then-reason)

### Takeaway
Little rigorous, small-model-specific evidence exists. What exists says that CoT-style or highly structured prompts can hurt text perception, and that prompt placement effects are model-specific.

### Cited Findings
- Highly structured or very short prompts such as CoT can hurt performance on visualized text. — [VISTA-Bench, arXiv 2602.04802](https://arxiv.org/html/2602.04802v1)
- An extra CoT prompt at inference creates a train/test mismatch, which may explain CoT degradation on OCR-reasoning tasks. — [OCR-Reasoning, ICLR 2026](https://arxiv.org/html/2505.17163v2)
- Putting instructions inside the image helped Qwen2.5-VL on POPE (+4.1) and devastated LLaVA-1.5 (−29). — [arXiv 2508.01678](https://arxiv.org/html/2508.01678v1)

### Inferences
- "Read then reason" can be implemented as a two-pass pipeline: transcribe in non-thinking mode, then answer with the transcription in context. This mirrors the OCR-augmentation gains in section 5, using the VLM itself as the OCR engine. It also lets section 3's finding (thinking hurts reading, helps reasoning) be applied per stage.

### Gaps
- No paper found that quantifies "OCR first, then answer" self-prompting on ≤4B VLMs, or system-prompt wording effects on transcription CER.

## 5. Hybrid pipelines (dedicated OCR plus VLM)

### Takeaway
Adding external OCR text to the prompt helps weaker VLMs a lot and strong readers little or not at all. Gains scale with OCR quality.

### Cited Findings
- **Adding OCR text to the prompt** (KOCRBench, 250 Korean VQA questions, KLOCR 55M OCR model):

  | Model | Without OCR text | With OCR text |
  |---|---|---|
  | InternVL2.5-**7B** | 87 | **162** (+75) |
  | Qwen2.5-VL-7B | 198 | 212 (+14) |
  | Qwen2.5-VL-32B | 176 | 205 |
  | Gemini 2.5 Flash | 182 | 212 |
  | Gemini 2.0 Flash | 200 | 203 |

  Swapping in the weaker TrOCR engine lowers the gains: InternVL +40 instead of +75, and Qwen +4 instead of +14. Key-information extraction gains the most (InternVL 20 → 81). "The largest improvements are observed from smaller models with a weaker base performance." All of these models are 7B or larger, and the benchmark is Korean. — [Exploring OCR-augmented Generation for Bilingual VQA, arXiv 2510.02543 (Oct 2025)](https://arxiv.org/html/2510.02543)
- **Contrary result.** At ImageCLEF 2026, injecting an OCR file into the prompts of two Qwen3-VL models did not improve visual open-ended QA. The authors suggest strong readers gain only a noisier second view of the same image. — [FAU at ImageCLEF 2026, arXiv 2608.01664](https://arxiv.org/pdf/2608.01664)
- A trained, not training-free, variant: DocVLM compresses OCR text and layout into 64 learned queries for a frozen VLM and lifts InternVL2 DocVQA from 56.0% to 86.6% under a low image-token budget. — [DocVLM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nacson_DocVLM_Make_Your_VLM_an_Efficient_Reader_CVPR_2025_paper.pdf)

### Inferences
- larch's InternVL3.5-1B (CER 0.11) fits the profile of a model that benefits from OCR augmentation. The Qwen3.5 models (CER 0.01–0.02) fit the profile of models that gain little. An OCR side channel should be optional and model-dependent. It is most useful for InternVL and for very dense or tiny text, where the token cap bites.

### Gaps
- No English DocVQA, OCRBench or ScreenSpot numbers were found for ≤4B VLMs with PaddleOCR, Tesseract, Florence-2, GOT-OCR2, DeepSeek-OCR or docTR text in the prompt. The VRAM and latency of those OCR models alongside a VLM on 4 GB were not researched.

## 6. Image preprocessing (EXIF rotation, contrast, HiDPI)

### Takeaway
No benchmarked sources were found within budget. The only related evidence is the Qwen2-VL min_pixels result: upscaling helps within limits and over-upscaling small images hurts OCRBench.

### Cited Findings
- Upscaling small images helps InfoVQA, HallusionBench and OCRBench within a range. Too-high min_pixels severely hurts OCRBench. — [Qwen2-VL report](https://arxiv.org/html/2409.12191v1)
- Visual perturbations sharply degrade worst-case OCR-reasoning even for 8B thinking models (WCR 0.624). — [OCR-Robust, arXiv 2606.26041](https://arxiv.org/html/2606.26041)

### Inferences
- Applying EXIF orientation is a correctness fix rather than a tuning choice. VLM encoders see raw pixel order, and rotated text is effectively out of distribution. This is general knowledge and was not benchmarked here.
- For HiDPI (2×) screenshots, downscaling to logical resolution halves tokens without losing glyph legibility if the glyphs stay above patch scale. Whether to downscale should depend on text size versus the 1,600-token cap. This is an inference that needs local measurement on larch's hidpi test case.

### Gaps
- No sourced numbers on contrast or binarization, EXIF handling, or HiDPI downscale strategies for VLM OCR accuracy.
