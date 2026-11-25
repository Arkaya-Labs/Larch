# Vision encoder and multimodal serving optimizations in production engines and papers (as of Nov 2026)

Scope: systems-level optimizations for the ViT / projector and the multimodal serving path (llama.cpp mtmd/clip, vLLM, SGLang, TensorRT-LLM/Dynamo, MLX / vllm-mlx, research). Target system: larch (Rust + CUDA/Metal/CPU), Qwen 3.5 0.8B/2B (Qwen3-VL-style ViT, full attention, 2D rope, 2x2 merge) and InternVL3.5 1B/2B, RTX 3050 4 GB (Ampere sm86). Current larch numbers: 1280x720 (3,680 patches) encodes in 174 ms (0.8B) / 506 ms (2B), against llama.cpp at 280 / 670-690 ms.

Source quality note: vLLM and SGLang docs, the llama.cpp PR, and the arXiv papers are primary sources. Blog figures (contracollective) and Reddit anecdotes are flagged where used. Several search-tool summaries were checked against the fetched page; where only a search snippet was available, that is stated.

## Encoder kernels and precision: FlashAttention, window attention, fused ops, FP16/BF16/FP8/INT8/4-bit encoders and projectors

### Takeaway
Production engines run the ViT in BF16/FP16 with varlen FlashAttention. FP8 is used only for ViT attention (vLLM, Qwen3-VL family only, on Hopper/Blackwell/MI300). That gives near-zero accuracy loss and helps only at QHD/4K or with multiple images. Papers consistently find the vision encoder less sensitive to quantization than the LLM. FlashAttention in llama.cpp's clip cut Metal compute-buffer memory about 9x and encode time 16-26%. No source reports an accuracy study for FP16 accumulation (as opposed to FP16 inputs with FP32 accumulation) in a ViT.

### Cited Findings
**FlashAttention in the encoder**
- llama.cpp PR #16837 "clip : use FA" (merged 2 Nov 2025) enables flash attention in the CLIP/mtmd vision encoder, default "auto": on when the backend supports all the graph's ops, and warmup warns about any unsupported ops. Metal gained head size 72 for Gemma. — [llama.cpp PR #16837](https://github.com/ggml-org/llama.cpp/pull/16837)
- On Gemma 12B, two images, Metal, that PR reports: Metal compute buffer 1132.00 MiB → 121.25 MiB, image 1 1653 → 1386 ms (-16%), image 2 1093 → 810 ms (-26%). The PR does not discuss accuracy. The PR page links a later report titled as CUDA FA producing wrong image embeddings on Qwen3.5/Qwen3-VL; I could not find or confirm that issue. — [llama.cpp PR #16837](https://github.com/ggml-org/llama.cpp/pull/16837)
- llama.cpp issue #17801 / draft PR #17802: very large images hit a 2 GB / INT_MAX limit in `ggml_cuda_cpy` because of large FA masks. The proposed fix tiles Q into chunks, runs FA per chunk, and concatenates. The same issue notes that `--no-mmproj-offload` keeps encoding on the CPU, which is slow. — [llama.cpp issue #17801](https://github.com/ggml-org/llama.cpp/issues/17801)
- The EPD paper's baseline setup runs the encoders in FP16 with FlashAttention-2 on A100. — [EPD paper, arXiv 2501.05460v4](https://arxiv.org/html/2501.05460v4)
- vLLM encoder CUDA graphs were tested with FLASH_ATTN and FLASHINFER backends on Blackwell, and with FA2/FA3 for Qwen2/2.5-VL. — [vLLM CUDA graphs multimodal design doc](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/)

**Window attention (Qwen2.5-VL)**
- The Qwen2.5-VL ViT has only 4 full-attention layers. The rest use windowed attention with a maximum window of 112x112 px (8x8 patches), so cost scales linearly with patch count. The official report and README give no speedup percentage. — [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923)
- A secondary aggregator estimates about 4x fewer encoder attention FLOPs for typical images. This is an unverified derivation. — [Emergentmind topic page](https://www.emergentmind.com/topics/qwen2-5-vl-multimodal-encoder)

**FP8 ViT attention (vLLM)**
- vLLM quantizes only the ViT attention Q/K/V to FP8, on the fly before the attention call. It supports dynamic scaling (a 16-entry amax ring buffer, no calibration) and static scaling (calibrated with `vllm bench mm-processor`, default 1.5x margin; recommended for production). NVIDIA requires the FlashInfer cuDNN backend with cuDNN ≥ 9.17.1; AMD uses AITER on gfx942/gfx950. Only the Qwen3-VL family is supported (`qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`, `qwen3_5_moe`). — [vLLM FP8 ViT attention doc](https://docs.vllm.ai/en/latest/features/quantization/fp8_vit_attn/)
- Accuracy on ChartQA, Qwen3-VL-8B, 500 samples, relaxed accuracy: BF16 0.780, FP8 dynamic 0.776, FP8 static 0.780; exact match 0.584 / 0.582 / 0.578. The doc calls these equal within noise. — [vLLM FP8 ViT attention doc](https://docs.vllm.ai/en/latest/features/quantization/fp8_vit_attn/)
- Speed: the cuDNN kernel at seq 8192 runs 1.12x faster on GB200 and 1.42x on GB300. End to end (Qwen3-VL-30B-A3B, GB200, 3 images/request) it is 0.87x at HD (slower), about 1.0x at FullHD, 1.08x at QHD, and 1.18x at 4K. Small images lose because of three quantization kernel launches plus un-padding. Dynamic scaling is incompatible with ViT full CUDA graphs. — [vLLM FP8 ViT attention doc](https://docs.vllm.ai/en/latest/features/quantization/fp8_vit_attn/)

**Whole-ViT / VLM quantization**
- LiteVLM: FP8 PTQ via ModelOpt gave about 1.5x speedup on the ViT and 1.6x on LLM generation, with "slight" task accuracy degradation (search snippet). — [LiteVLM, arXiv 2506.07416](https://arxiv.org/pdf/2506.07416)
- A microscaling-format PTQ benchmark (Jan 2026, abstract): "quantization sensitivity is dominated by the language model rather than the vision encoder in multimodal LLMs". MXFP8 is near-lossless, and MXFP4 degrades substantially. A search snippet attributed "W4A4 on the ViT ≈1% accuracy cost on Qwen2.5-VL-7B" to this paper, but I could not extract the PDF tables to verify it. — [arXiv 2601.09555](https://arxiv.org/abs/2601.09555)
- An AAAI paper reports a "substantial sensitivity discrepancy" between ViT and LLM under quantization-aware training (search snippet only). — [AAAI article 39296](https://ojs.aaai.org/index.php/AAAI/article/view/39296/43257)
- Red Hat / Neural Magic report that FP8 and INT8 (W8A8) quantized VLMs (Qwen2-VL-72B, Qwen2.5-VL 3B/7B/72B, Pixtral) recover over 99% across five vision benchmarks, and that INT4 drops modestly on smaller models. The snippets do not say whether the vision tower itself was quantized. — [Red Hat Developer, 1 Apr 2025](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization); [HF model card Qwen2.5-VL-7B w8a8](https://huggingface.co/neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w8a8)
- Q-VLM (NeurIPS 2024): PTQ that models cross-layer dependency and tunes the visual encoder to shrink the search space. On LLaVA-13B at 4-bit it reports about 2.78x memory compression and 1.44x faster generation. — [Q-VLM, arXiv 2410.08119](https://arxiv.org/html/2410.08119v1)
- MBQ (Modality-Balanced Quantization): calibration weighted by the different sensitivity of vision and language tokens. It reports up to +4% accuracy at W3 and +11% at W4A8 over SOTA baselines on 7B-70B VLMs. — [MBQ, arXiv 2412.19509](https://ar5iv.labs.arxiv.org/html/2412.19509)
- VLM-PTQ (CVPR 2026 supplement), 3-bit, LLaVA-OneVision-7B average accuracy: VLM-PTQ 69.5%, MBQ 64.4%, AWQ 60.6%. Self-reported. — [CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Deng_VLM-PTQ_Efficient_Post-Training_CVPR_2026_supplemental.pdf)

**llama.cpp mmproj quantization**
- llama.cpp removed `clip-quantize-cli` with libllava in PR #13460, a breaking change. The old README listed q4_0/q8_0 mmproj options. — [llama.cpp commit for #13460](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/b4726345aca49e2ad62d615e6e370b3dbad6434f)
- Feature request #18881: mmproj files for models like Qwen2-VL ship only in f16/f32. A Q8_0 mmproj is estimated at about 665 MB, 50% smaller than f16. — [llama.cpp issue #18881](https://github.com/ggml-org/llama.cpp/issues/18881)
- Anecdotal (Reddit, Apr 2026, explicitly non-scientific): Q8 vs F16 mmproj differences were insignificant **except on Qwen3.5 0.8B and 4B**. One commenter saw Q8_0 beat F16 on a Gemma model. — [r/LocalLLaMA snapshot](https://reddit.sentinel-team.org/posts/1sgmgtj/snapshots/2026-04-09T16%3A11%3A00.655487Z)
- Issue #22582: llama-server spent 82 s in `mtmd_encode_chunk()`, apparently on CPU in BF16, while llama-cli was fast on the same image. This shows how easily the encoder silently ends up on a slow path. — [llama.cpp issue #22582](https://github.com/ggml-org/llama.cpp/issues/22582)

### Inferences
- **FP8 is not available on larch's target card.** The RTX 3050 is Ampere sm86 with no FP8 tensor cores, and vLLM's FP8 ViT attention needs cuDNN FP8 on Hopper/Blackwell-class parts. Even on GB200 it is slower than BF16 below FullHD. For a 1280x720 single image, FP8 attention would not help even on supported hardware.
- **FP16/BF16 inputs with FP32 accumulation are the industry baseline.** Every engine benchmark found (vLLM FP8 doc baseline, EPD paper) runs the ViT in BF16/FP16, so larch's FP32-everything encoder is more conservative than production. Moving GEMM inputs to f16/bf16 while keeping FP32 accumulation (cuBLAS `COMPUTE_32F` with 16-bit A/B) matches what all engines ship. True FP16 accumulation (`COMPUTE_16F`) is a separate, riskier step for which I found no ViT accuracy data.
- **The INT8 projector is supported by the literature, with a Qwen3.5 0.8B caveat.** Papers say the ViT and projector are less sensitive than the LLM, which supports the planned 8-bit projector weights (about 310 MB saved). The anecdotal Qwen3.5 0.8B Q8-mmproj sensitivity is exactly larch's model, so validate on the 0.8B before shipping.
- **Window attention cannot be retrofitted.** It is an architectural property of Qwen2.5-VL. The Qwen3-VL/Qwen3.5 ViT that larch runs uses full attention, so the attention kernel (flash/varlen, no materialized mask) is the lever. llama.cpp's FA change mainly saved memory (about 9x compute buffer), which matters on a 4 GB card.
- **A fused bias/GELU epilogue is a small win.** No engine-level source quantified it for ViTs. The roughly 10% cuBLASLt estimate is plausible but unverified here.

### Gaps
- No published accuracy study of FP16-accumulate (as opposed to FP32-accumulate) GEMMs in ViTs, or for Qwen3-VL/Qwen3.5 specifically.
- No primary benchmark of llama.cpp q8_0 vs f16 mmproj encode speed or accuracy.
- Could not confirm the reported llama.cpp CUDA FA wrong-embeddings issue on Qwen3.5/Qwen3-VL (possibly issue #19543). Worth checking, since larch targets the same models.
- Could not extract per-component (ViT vs LLM) numbers from arXiv 2601.09555; the PDF text was not recoverable.
- No source found for which engines ship INT8/FP8 *projector* weights by default. TensorRT-LLM FP8 vision encoder results were not found.

## Caching: image embedding caches, multimodal prefix caching, KV reuse across turns, cache sizing

### Takeaway
Every major engine keys encoder outputs by a hash of the image content and keeps that cache separate from KV prefix caching. vLLM uses `mm_hash` with an encoder cache plus a processor cache, SGLang a 4 GB default `--enable-prefix-mm-cache`, and vllm-mlx SHA-256 of decoded pixels with 512 MB LRU. The largest measured multi-turn wins come from caching both vision embeddings and KV: 19-28x per turn on Apple Silicon. Embeddings alone give about 7.8x, and KV alone about 1.2x.

### Cited Findings
- **vLLM V1** `EncoderCacheManager`: shares embeddings of the same multimodal item (identified by its hash) across requests, and evicts at allocation time when full. `can_allocate` rejects an item whose token count exceeds the per-step `encoder_compute_budget`. — [vLLM encoder cache manager v0.11.2](https://docs.vllm.ai/en/v0.11.2/api/vllm/v1/core/encoder_cache_manager/)
- The vLLM(-Omni) prefix caching doc treats the image hash/encoder cache as separate from the KV block hash, and says the encoder cache covers images that do not end on a KV block boundary. — [vLLM-Omni prefix caching design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/)
- vLLM also caches HF processor outputs because some processors (Qwen2-VL) are very slow. It batches only cache-missing items into the HF processor. A forum answer suggests `--mm-processor-cache-type shm` for workloads with repeated images. — [vLLM mm_processing design](https://docs.vllm.ai/en/latest/design/mm_processing/); [vLLM forum](https://discuss.vllm.ai/t/qps-doesnt-scale-with-multi-card-gpu/2603)
- **SGLang**: encoder servers cache vision embeddings via `--enable-prefix-mm-cache`, default 4 GB (`SGLANG_VLM_CACHE_SIZE_MB`), so repeated images skip the ViT and the transfer. — [LMSYS EPD blog, 12 Jan 2026](https://lmsys.org/blog/2026-01-12-epd)
- With SGLang under Dynamo, each media item gets a per-media pad value in the RadixAttention prefix key, derived from the image hash so router and cache keys align. — [NVIDIA Dynamo SGLang multimodal docs](https://docs.nvidia.com/dynamo/zh-CN/dev/knowledge-base/modular-components/backends/sg-lang/sglang-multimodal)
- **vllm-mlx (Apple Silicon paper, arXiv 2601.19139)**: images are keyed by SHA-256 of *decoded pixel values*, so URL, base64 and file inputs all hit. Each entry stores vision embeddings plus KV state, with LRU eviction and a 512 MB default limit. — [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139)
- Same paper, M4 Max, Qwen3-VL-8B, 1024x1024 multi-turn: no cache 21.7 s/turn; turn 2 cached 1.15 s (19x); turn 3+ 0.78 s (28x). Ablation: embeddings only 2.8 s (7.8x), KV only 18.2 s (1.2x), both 1.15 s (19x). The text claims KV reuse "adds 2.4x", which conflicts with its own table. — [arXiv 2601.19139](https://arxiv.org/html/2601.19139)
- Same paper, Qwen3-VL-4B cache entry sizes and cold→cached: 224² 48 MB, 0.8→0.12 s; 448² 89 MB, 1.2→0.14 s; 768² 124 MB, 1.8→0.15 s; 1024² 156 MB, 2.1→0.16 s. — [arXiv 2601.19139](https://arxiv.org/html/2601.19139)
- **VLCache** (research, arXiv 2512.12977): stores encoder output and KV, reuses both on an image-hash hit, and recomputes a selected subset of tokens (about 2%) to limit accuracy loss when the image appears in a different prefix context. The prototype is built on SGLang. — [VLCache, arXiv 2512.12977](https://arxiv.org/pdf/2512.12977)

### Inferences
- **larch already has the core feature.** Its content-hash image cache with 4 Qwen slots matches the industry design. What production engines add:
  - a **byte budget** instead of a slot count (SGLang 4 GB, vllm-mlx 512 MB LRU);
  - hashing **decoded pixels** (dedups the same image arriving through different encodings);
  - a **processor/preprocess cache** layer;
  - **KV reuse across turns** for the image span, keyed to the image hash so a changed preceding prefix still reuses embeddings.
- **Sizing check.** For Qwen 3.5 0.8B, a 3,680-patch image becomes 920 merged tokens after 2x2 merge. At LLM hidden size × 2 bytes (bf16) this is a few MB, so four slots are cheap. The ablation suggests per-turn wins come mostly from embedding reuse, plus KV reuse to avoid re-prefilling the 920 image tokens.
- **VLCache-style partial recompute** is what would let KV for an image be reused when the image moves to a new position or context. It is research-grade.

### Gaps
- Exact vLLM V1 `mm_hash` algorithm (bytes vs decoded pixels) and default encoder cache size in the current release were not confirmed; the docs found were v0.10.2-v0.11.2.
- No engine reports the hit-rate distribution of image caches in local, single-user chat.

## Scheduling: encoder/prefill/decode disaggregation, chunked prefill, encoder batching, encoder budget, ViT CUDA graphs, DP vs TP

### Takeaway
At scale, separating the encoder (EPD) yields large TTFT and goodput wins for image-heavy workloads: up to 71.9% lower TTFT and 15x lower peak memory (EPD paper), 3.3-5.5x throughput (ModServe), and 6-8x TTFT in SGLang. It is counterproductive for image-light traffic. Tensor-parallelizing a small ViT is wasteful, so engines replicate it (DP) instead. ViT CUDA graphs help mostly at small batch, which is exactly the single-user case.

### Cited Findings
- **EPD paper** (Singh et al., arXiv 2501.05460, ICML/PMLR v267 2025; 8x A100 FP16 FA2; MiniCPM-V 2.6, InternVL2-8B/26B):
  - **Memory:** peak memory up to 15x lower; encoder workers save 95-96% of weight memory.
  - **Capacity:** images per request up to 10x higher (InternVL2-26B at 4032x3024: 1 → 10), batch size up to 22x, KV capacity up to 2.2x.
  - **TTFT:** up to 71.9% lower vs DistServe; up to 69.2% vs vLLM on Video-MME.
  - **Intra-request parallelism:** splitting one request's image patches across encoder workers matters. Disabling it raises TTFT 1.6x at 2 images and 2.9x at 6.
  - **Role switching:** dynamic role switching takes under 0.7 s.
  - [EPD paper v4](https://arxiv.org/html/2501.05460v4); [PMLR](https://proceedings.mlr.press/v267/singh25d.html)
- **ModServe** (Microsoft, SoCC 2025, arXiv 2502.00937): calls image encoding a major TTFT bottleneck. Images in a request do not attend to each other, so encoding is split at image level across Image Instances, routed to the instance with the fewest pending image tokens. On a 128-GPU cluster replaying production traces it reports 3.3-5.5x throughput and 25-41.3% cost savings. — [ModServe arXiv](https://arxiv.org/abs/2502.00937)
- **HydraInfer**: hybrid EPD that picks the disaggregation scheme from workload and SLO profiles. — [HydraInfer, arXiv 2505.12658](https://arxiv.org/html/2505.12658v2)
- **SGLang EPD** (12 Jan 2026), 8x H20, Qwen3-VL-235B-A22B-FP8, about 4x 1080p images per request:
  - **Gains:** 2E1P vs colocate gives about 6-8x lower TTFT at 1 QPS, about 8-10x lower TPOT, and about 2x throughput, using 6 GPUs instead of 4.
  - **Embedding transport:** ZMQ (default) or Mooncake RDMA.
  - **When it hurts:** EPD is "less efficient or even counterproductive" for image-light workloads, because transfer can exceed the saved encode time.
  - [LMSYS blog](https://lmsys.org/blog/2026-01-12-epd)
- **SGLang ViT TP scaling.** Qwen2.5-VL-72B ViT on H20 with 4 images: 492 ms at TP2, 466 ms at TP4, 524 ms at TP8. More TP does not help. — [LMSYS blog](https://lmsys.org/blog/2026-01-12-epd)
- **vLLM `--mm-encoder-tp-mode data`** (AMD blog, 2 Jan 2026, 8x MI300X): replicates the ViT per rank instead of sharding it. Mean request throughput gains: InternVL3.5-241B +17.7%, step3 +15.0%, Qwen3-VL-235B +6.2%; range -1.5% to +44.9%. With TP, step3's ViT needed 63 all-reduces, about 39.9% of encoder time. The Qwen3-VL-235B ViT is only 0.4B params (0.2% of total). — [AMD ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html)
- **vLLM encoder CUDA graphs** (`cudagraph_mm_encoder`, off by default):
  - **Capture scheme:** graphs are captured at a grid of token budgets (36 levels over [64, 16384], default max 5120 tokens). Inputs are zero-padded to the smallest fitting budget, and capture buffers are shared so memory is about 1x the largest budget. Batches beyond the max run eagerly.
  - **Models:** supported for Qwen2/2.5/3-VL, Qwen3.5/3.6, InternVL 2-3.5, Gemma 3/4 and others.
  - **Scope of gains:** "Small batches gain the most. Eager becomes faster than graphs at large sizes."
  - **Numbers:** 1x GB200, Qwen3-VL-30B-A3B: mean 5.13→4.52 ms (FA) / 5.42→4.36 ms (FlashInfer), P99 9.16→6.26 / 10.87→6.49 ms. 4x GB200 with DP encoder: FlashInfer mean 23.24→12.91 ms, P99 172.41→26.05 ms.
  - [vLLM CUDA graphs multimodal doc](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/)
- **SGLang ViT CUDA graphs** (`SGLANG_VIT_ENABLE_CUDA_GRAPH=1`, Qwen2.5-VL/Qwen3-VL): captures the ViT blocks, merger and deepstack merger. Graphs are keyed by sequence length, and `cu_seqlens` must match; every distinct length adds graph-private memory. — [SGLang docs](https://docs.sglang.io/advanced_features/cuda_graph_for_multi_modal_encoder.html)
- **Dynamo / TensorRT-LLM** support aggregated, EP/D (prefill encodes) and E/P/D (standalone `MultimodalEncoder` worker running vision encoder plus projector, with embeddings sent over NIXL). Older docs list E/P/D as WIP. — [Dynamo TRT-LLM multimodal docs](https://docs.nvidia.com/dynamo/latest/multimodal/trtllm.html)

### Inferences
- **Single-user local engine.** EPD across devices does not apply. Its intra-engine analogues do:
  - larch's "encode on paste" is the local analogue of EPD's asynchronous encode stage.
  - Slicing the encode across scheduler steps is the analogue of vLLM's per-step `encoder_compute_budget`.
  - ModServe's image-level split suggests encoding multi-image pastes per image, so the first image's tokens can start prefill early.
- **ViT CUDA graphs suit larch's regime** (batch 1, small models). Launch overhead is a bigger fraction for a 24-layer 1024-dim ViT on a slow GPU than on a 30B model. vLLM's padded-budget scheme (about 36 levels, shared buffers) is a template for capturing per-slice graphs despite variable patch counts. Padding wastes compute, though, and at 174 ms per image launch overhead is likely a small share. Profile the gap between kernels before investing.
- **Multi-GPU users should replicate, not shard.** The ViT should be replicated (DP), not tensor-sharded; all engines found converge on this.

### Gaps
- No public numbers for chunked prefill interacting with image tokens (images split across chunks) in vLLM V1. The encoder budget mechanism is documented, but its tuning impact is not.
- No primary benchmark of ViT CUDA graphs on consumer GPUs (all vLLM numbers are GB200/MI350X).

## Apple / Metal: how mlx-vlm and llama.cpp Metal run vision encoders; M-series numbers

### Takeaway
Primary M-series numbers for the vision encoder are sparse. The best primary data are from vllm-mlx on an M4 Max: Qwen3-VL-4B cold 0.8-2.1 s for 224²-1024² (encode plus prefill), and the llama.cpp FA PR on Metal (Gemma 12B: 1.1-1.65 s per image, 16-26% faster with FA, about 9x less compute-buffer memory). FastVLM shows vision encoding dominates TTFT at high resolution on an M1.

### Cited Findings
- llama.cpp Metal, Gemma 12B, 2 images: 1653→1386 ms and 1093→810 ms with clip FA; Metal compute buffer 1132→121 MiB. — [llama.cpp PR #16837](https://github.com/ggml-org/llama.cpp/pull/16837)
- vllm-mlx on M4 Max 128 GB:
  - **Encoder cost:** vision encoding costs 1.5-4 s per request depending on resolution.
  - **Qwen3-VL-4B cold (uncached) turn:** 0.8 s at 224², 1.2 s at 448², 1.8 s at 768², 2.1 s at 1024².
  - **Text throughput:** 21-87% higher than llama.cpp (Qwen3-0.6B: 525.5 vs 281.5 tok/s).
  - **Missing comparison:** no multimodal comparison against llama.cpp; the paper claims llama.cpp lacks VLM support, which is inaccurate as of 2025-2026 given mtmd.
  - [arXiv 2601.19139](https://arxiv.org/html/2601.19139)
- FastVLM (Apple, CVPR 2025), measured on an M1 MacBook Pro: at high resolution, vision encoder latency becomes the dominant TTFT bottleneck. Its hybrid FastViTHD encoder gives 3.2x better TTFT in the LLaVA-1.5 setup. — [Apple ML Research](https://machinelearning.apple.com/research/fast-vision-language-models); [FastVLM paper](https://arxiv.org/html/2412.13303v1)
- Unverified blog (Jun 2026, no methodology): Qwen2.5-VL-7B image encode 480 ms in llama.cpp Metal (likely M4 Pro); on M5 Pro, 360 ms llama.cpp vs 240 ms MLX. — [contracollective blog](https://contracollective.com/blog/local-vision-llm-apple-silicon-mlx-qwen-vl-moondream-2026)
- A Reddit comment says Vulkan builds lag CUDA/Metal for multimodal features (anecdotal). — [r/LocalLLaMA](https://reddit.sentinel-team.org/posts/1ssbop3/snapshots/2026-04-25T00%3A46%3A56.728262Z)

### Inferences
- **Metal port priorities.** For a Metal port, llama.cpp's experience says flash attention in the ViT is the first priority, mostly for memory: a 9x smaller compute buffer matters on 8-16 GB unified-memory Macs. Content-hash caching of embeddings plus KV is the biggest multi-turn win on Apple Silicon.
- **Rough Metal targets** (low confidence, from unverified blog data): MLX appears about 1.5x faster than llama.cpp Metal on Qwen2.5-VL-7B encoding. Hundreds of ms for a 7B-class ViT on M4/M5 Pro is the competitive bar.

### Gaps
- No primary mlx-vlm benchmark of encoder-only time for Qwen3-VL/Qwen3.5 small models on M1-M4.
- No source detailing mlx-vlm's ViT attention kernel (e.g., whether it uses `mx.fast.scaled_dot_product_attention` with varlen masking).

## CPU fallback: fast CPU ViT inference

### Takeaway
I found no credible recent CPU benchmarks for small ViT encoders (oneDNN or ggml CPU) in VLM pipelines. The only evidence is negative: llama.cpp's CPU path for mmproj is "slow", with one 82 s outlier.

### Cited Findings
- `--no-mmproj-offload` keeps encoding on the CPU, "which makes image encoding slow". — [llama.cpp issue #17801](https://github.com/ggml-org/llama.cpp/issues/17801)
- llama-server took 82 s per image in `mtmd_encode_chunk()` (apparently CPU BF16). — [llama.cpp issue #22582](https://github.com/ggml-org/llama.cpp/issues/22582)
- I-ViT (integer-only ViT quantization) reports small ImageNet top-1 drops at INT8, for classification ViTs only, not VLM encoders. — [I-ViT, arXiv 2207.01405](https://arxiv.org/pdf/2207.01405)

### Inferences
- With full attention over 3,680 patches and a 1024-dim, 24-layer ViT, CPU encoding is likely seconds per image. This suggests a CPU fallback should downscale aggressively (fewer patches) rather than rely on kernel tuning. This is an inference; no source measures it.

### Gaps
- No oneDNN / OpenVINO / ggml-CPU numbers found for SigLIP / Qwen-VL / InternViT encoders at 2024-2026 dates.

## Engine features larch may be missing (preprocessing on GPU, nvJPEG, async encode, etc.)

### Takeaway
vLLM moved only normalization and rescaling to the GPU: a fused Triton kernel on uint8 input that halves PCIe bytes. Decode and resize remain on the CPU, and I found no engine using nvJPEG. Engines also add processor-output caches, DP encoders, ViT CUDA graphs and image-level parallel encoding. CPU-side preprocessing is a recurring throughput bottleneck at scale.

### Cited Findings
- **vLLM `mm_device_do_normalize`** (default on for Qwen2-VL through Qwen3.5, Gemma 3/4, Kimi and others):
  - **Kernel:** `FusedMMInputNorm` runs a Triton per-channel affine `y = x*w[c] + b[c]`, with the /255, mean and std folded in. It accepts uint8 input and computes in fp32.
  - **PCIe saving:** uploading uint8 instead of bf16 cuts PCIe volume 50%.
  - **What stays on CPU:** decode and resize still run on the CPU, and no GPU decoding is described.
  - [vLLM mm_processing doc](https://docs.vllm.ai/en/latest/design/mm_processing/)
- Forum reports of Qwen2.5-VL in vLLM describe CPU-bound, poorly parallelized image preprocessing that lowers throughput while the GPU is underutilized. These are community reports, and an older claim of "no built-in feature" predates the GPU-normalize path. — [vLLM forum: Qwen2.5-VL slow with image](https://discuss.vllm.ai/t/why-is-inference-for-qwen-2-5-vl-so-slow-when-we-send-an-image/1438); [vLLM forum: high CPU utilization](https://discuss.vllm.ai/t/cpu-utilization-is-extremely-high-during-inference-and-becomes-the-primary-performance-bottleneck/1876)
- Research (FlashCodec) reports that multimodal preprocessing, especially video decoding, often dominates TTFT, and proposes collaborative multi-GPU video decoding. — [chatpaper summary](https://chatpaper.com/paper/220384)
- The FP8 ViT attention and the DP encoder (above) are further vLLM features. The EPD paper's intra-request parallelism (splitting one request's images or patches across workers) is another pattern. — [vLLM FP8 ViT doc](https://docs.vllm.ai/en/latest/features/quantization/fp8_vit_attn/); [EPD paper](https://arxiv.org/html/2501.05460v4)
- A Jun 2026 dev.to post reports `torch.compile` of the Qwen3-VL vision encoder on Hopper giving about 3% more throughput (title only; not fetched). — [dev.to post](https://dev.to/mketkar/compiling-the-vision-encoder-squeezing-3-more-throughput-from-qwen3-vl-on-hopper-gpus-24ma)

### Inferences
Candidate additions for larch, with an estimated value for a single-user engine on 4 GB:
1. Upload uint8 and fuse normalization into the patch-embed or first kernel. This is cheap: half or a quarter of the PCIe bytes vs f16/f32, and one less CPU pass.
2. Hash decoded pixels for the image cache.
3. Byte-budget LRU for the encoder cache, plus KV reuse of the image span across turns.
4. ViT CUDA graphs at a few padded patch budgets, only if profiling shows launch gaps.
5. FlashAttention-style tiled attention in the ViT, mainly for memory headroom on 4 GB; llama.cpp also hit an INT_MAX copy limit with large masks.
6. Per-image encoding of multi-image pastes, so prefill can start on image 1.

GPU JPEG decode (nvJPEG) has no precedent in the engines surveyed. For a single 1280x720 paste, CPU decode is probably a few ms, so it is low priority (inference, not measured).

### Gaps
- Found no engine using nvJPEG or GPU resize for VLM input. Confirming absence would need source-code inspection of vLLM/SGLang/TensorRT-LLM.
- No numbers on CPU preprocessing time for a single 720p image in any engine.
- The async encode stream / encode-decode overlap within one GPU is not documented with numbers in any engine found. larch's scheduler-slice approach has no direct published comparator.
