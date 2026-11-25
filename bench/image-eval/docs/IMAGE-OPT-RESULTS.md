# Image optimization trials: results

Plan: `IMAGE-OPT-PLAN.md`. The trials ran on a work branch that was not merged.
Each trial lists what changed, the numbers before and after, the verdict (kept / reverted /
behind a setting / not testable), and side effects. Filled in as trials finish.

## Summary (12 Nov, 23:10; paused, GPU handed to the tool-call session)
Scores are the mean over the 256-item eval set (0-1). "Better / worse" counts items, paired against the
baseline from the same build, with a sign test. The 0.8B baseline is 0.835 and the 2B 0.902.

### What stuck (kept, on this branch)
| Item | What it is | Result | Commits |
|---|---|---|---|
| 2. Loop stop | Ends a reply stuck repeating one short unit (1-64 chars, over 200 chars and 8 copies) as a stop sequence; the app adds a notice | Catches exactly the 3 degenerate loops, cuts nothing else on either model; −15 s decode per 0.8B run; that transcription's CER 4.99 → 0.56 | f4 70366f52, agent e877de84 |
| 3b. No thinking on the step that sends images | The app's chat thinks (hidden, 100-token budget) on every Qwen 3.5 step; the step that sends the user's images now does not | The app's setting lost 0.045 on the 0.8B (18 better / 33 worse, p=0.049) and 0.016 on the 2B (14 / 19), and showed the first words 1.05-1.55 s later | agent 183c3ce1 |
| Decode graphs before a raised vision context | The engine raises a vision model's context to 64k and halves it until it fits; a context that fit only by dropping the decode graphs was kept | Found when a token cap freed 90 MB: the 2B took 32k without graphs and decoded at 38 tok/s instead of 68 on every turn. Now it halves to 16k with graphs (67.9 tok/s). The caller's own context is never cut | f4 1bffe75a |
| Image warm-up at a real size | The load's warm-up encoded one row; it now encodes a 256-row blank in idle time | With 3 s idle after load: first image 456 → 377 ms (2B), 364 → 290 ms (0.8B), three loads each. Answers unchanged (30 / 30 identical) | f4 853bfd51 |
| 10. Small images scaled up for the 0.8B | A Qwen projector whose text model is 512-1,024 wide (Qwen3.5-0.8B) defaults to a 1,024-row floor: a smaller image is scaled up toward it by at most 2x a side, one under 16 rows (a word crop) only to llama.cpp's 8 | Eval: +0.032 (35 better / 10 worse, p=0.0002; small fonts +0.22, code +0.17, scene text +0.06). **Held out** (180 new public items): +0.044 (10 / 2, p=0.04), no category worse, the 0.8B within 0.022 of the 2B there (was 0.067 behind). About 0.4 s more to the first token on the images it scales. The default build reproduces the experiment run exactly (256 / 256 answers); the 2B is unchanged (30 / 30) | f4 bad91346 |
| Eval harness | `examples/image_eval` (now with `pause_ms`), eval set v2, scorer, GPU queue | Greedy runs reproduce exactly within one build | a1 84a71b0a, 98d7237c, c8d42cf6 |

Tests: 1,647 pass in larch-model, larch-engine and larch-agent (20 ignored) with the first three kept changes; after the 0.8B rule, 724 pass in larch-model, larch-engine and larch (16 ignored). Four tests are new.

### Measured, not adopted (options)
| Item | Result |
|---|---|
| 11. Image token cap 1024 on the 0.8B | Eval −0.004 (1 / 1), held out +0.000 (1 / 1): free in accuracy, and on large images the time to first token goes 1,084 → 599 ms (whole held-out run −14%). Combined with the upscale it gave +0.033 (12 / 6) on the held-out set against +0.044 for the upscale alone (charts and infographics lose), so it is left as an option. On the 2B it costs 0.016 (1 / 6) on large document pages |
| 11. Image token cap 576 | 0.8B −0.007 (4 / 6), 2B −0.019 (2 / 8); time to first token on capped images about halves |

### Tried and rejected (none of these changes the shipped behaviour)
The five encoder switches were removed on 12 Nov (27f834d3): the kernels, CMakeLists and build.rs are back
to their pre-switch state; the bench scripts that set them need the experiment-switch commit (on the work branch, not in this history) to rerun.

| Item | Result | Code left behind |
|---|---|---|
| 9. f16 accumulation in encoder GEMMs | −8% encode on the 2B, −5% on the 0.8B; worst row cosine 0.865 vs f32 | `LARCH_EXP_GEMM16` switch: removed |
| 12b. Larger K/V tiles in encoder attention (32, 64) | −1 to −4% encode | `LARCH_EXP_ATTN_KT` template: removed |
| 6 (bound). Bias added in place after f16 rounding | −6% encode, worst cosine 0.902; only a speed bound | `LARCH_EXP_BIAS_INPLACE`: removed |
| 6. cuBLASLt bias epilogue | 2B −2.4% (1280x720) / −5% (wide); 0.8B about 0; worst cosine 0.994, so not the same arithmetic after all (cuBLASLt picks other kernels) | `LARCH_EXP_LT_BIAS` and the cublasLt link: removed |
| Single-f16 attention weights (drop P's low half) | 0% on both models; worst cosine 0.984 | `LARCH_EXP_ATTN_NOLO`: removed |
| 3. Full thinking on image turns | 0.8B −0.019 (6 better / 9 worse), 2B −0.089 (4 / 8) on fast60; 3.7-4.6x slower | none (a setting) |
| Repeat penalty 1.1 | 0.8B +0.013 (22 / 17, n.s.); 2B −0.004 (11 / 19, n.s.); hurts tables and code | none (a setting) |
| Model card sampled preset (0.8B) | −0.031 (17 / 33, p=0.03); tables −0.29, small fonts −0.27 | none (a setting) |
| 7. Read then reason (transcribe, then answer in the same chat) | 0.8B −0.023 (2 / 3) and 1.8x the time per answer; 2B −0.045 (1 / 3) and 2.9x (44 questions) | none (harness option) |
| 2B at Q6_K instead of Q4_K_M | +0.011 (10 / 11, n.s.); decode 66 → 34 tok/s, run 645 → 1,163 s | none |
| 2B at Q8_0 | Does not fit: needs 3.79 GB, 3.43 GB free on the 4 GB card | none |
| 2B at Q3_K_M instead of Q4_K_M | −0.001 (16 / 16); decode +7%, run −17% (fewer loops); receipts −0.17, so it fails "no category worse". Still beats the 0.8B by 0.063 (p=0.0004). An option for tight memory, not a default | none |
| 0.8B at Q4_K_M instead of Q8_0 | −0.015 (17 / 19, n.s.); decode 99 → 131 tok/s (+32%), run 454 → 352 s; but tables −0.20 and documents −0.12, so it fails "no category worse" | none |
| 10. Upscale small images to at least 1024 tokens, 2B | −0.001 (21 / 11, p=0.11): terminals +0.16 and small fonts +0.08, code −0.09, scene text −0.06, charts −0.05; time to first token 465 → 1,272 ms | none (the switch stays for the 0.8B follow-up) |
| 10. Upscale to at least 1600 tokens, 0.8B | Invalid run: a floor at the 1,600-token cap rounded images past it (228 of 256 items `InvalidRequest(25)`). Not rerun; the fit now falls back to the cap (663085b8) | none |
| Eager CUDA module loading (`CUDA_MODULE_LOADING=EAGER`) for the first image | Worse: first image 520 ms on the 2B (377 with the default), 337 on the 0.8B (290); load +1-2 s, VRAM +30-40 MB | none (an environment variable) |
| 18. CUDA graphs for the encoder | Not built: launch gaps are about 3% of a 2B encode (522 ms of kernels in 540) and 4-6% on the 0.8B, at or under the plan's 5% bar | none |
| 8-word repeat rule for the loop stop | Cuts a real spreadsheet (CER 0.07 → 0.41) | none (simulated only) |
| 12. Tiled (flash-style) encoder attention | Already in place | none |
| 10. Upscale to at least 1024 tokens, at most 2x a side, 2B | +0.004 (22 / 8, p=0.016) but code −0.07, scene text −0.13; time to first token on upscaled images 235 → 634 ms. The 2B reads small text well enough already | none |

### Other findings
- **The first image after a load** is slower than later ones. The new warm-up removes about 80 ms of it.
  Most of the remaining ~160 ms is on the language model's side: a text turn before the first image
  halved it in the probe. Eager CUDA module loading made it worse, so it is not lazy kernel loading.
  Not built: a short text prefill in the load's warm-up.
- **Held-out set** (`heldout/`, 180 items: 140 OCRBench, 20 ChartQA, 20 DocVQA, other rows than the eval
  set, `fetch_heldout.py`): the 0.8B scores 0.789 and the 2B 0.067 more (p=0.008).
- **The experiment-switch commit (the experiment-switch commit (on the work branch, not in this history)) changed the default output** with every switch off (6 of 30
  answers matched the pre-switch build). Its effect on scores was negligible (0.8B +0.000, 2B −0.003).
  Removing the switches restored bit-identity (30 of 30, 22 of 22), and every trial was scored against a
  baseline from its own build.
- **Encoder profile:** attention 39-43% of the time, GEMMs 37-45%, bias passes 10%. Attention runs at about
  a third of the tensor-core peak; none of the tried kernel changes moved it more than 5%.

### Not testable here
16 (Metal), 19 / 30 / 31 / 35 (training), 22 / 25 / 26 (multi-GPU, Hopper), 28 (video), 4 (no drafter
for models that fit 4 GB).

## Phase 0: the measuring stick
- **Eval set v2** (`eval-v2/items.jsonl`): 256 items over 141 images. Of these, 147 are generated
  with exact ground truth (41 transcriptions, 71 questions, 24 "not shown" probes, 11 describe
  prompts), 73 come from OCRBench, 18 from ChartQA and 18 from DocVQA. `fast60.txt` is a stratified
  60-item subset. Build and licence notes are in `eval-v2/README.md`.
- **Harness** `crates/larch/examples/image_eval.rs` (settings profiles in JSON); **scorer**
  `eval-v2/score_eval.py`, which gives per-kind, per-category and per-source tables plus a paired
  comparison with a sign test.
- **Harness fix before any result counted:** with a 96-token limit on questions, the 0.8B's
  explain-first answers ("Looking at the table: row 8, column F...") were cut off before they gave the
  answer. That run was thrown away; questions and probes now get 384 tokens.
- **GPU sharing:** jobs run through `tools/gpu-queue.sh`, taking turns with another user of the GPU.
- **Baselines (greedy, thinking off):**

  | Model | Overall | Questions | "Not shown" | Transcription (CER) | Describe | Loops | TTFT median / p90 | Decode |
  |---|---|---|---|---|---|---|---|---|
  | Qwen3.5-0.8B | 0.835 | 0.813 | 0.917 | 0.898 (0.199) | 0.800 | 7 | 265 / 1,099 ms | 99 tok/s |
  | Qwen3.5-2B | 0.902 | 0.883 | 0.958 | 0.949 (0.051) | 0.900 | 3 | 465 / 2,135 ms | 67 tok/s |

  The 2B beats the 0.8B on 43 items and loses on 16 (sign test p=0.0006). The weakest categories
  for both are charts (0.8B 0.58, 2B 0.78), handwriting and small fonts.
- **Noise:** two runs of each model gave identical text on all 256 items. Greedy decoding is
  deterministic here, so a trial's change is real; whether it generalizes is judged with the paired
  sign test over items.

## Trials

### 12. Tiled (flash-style) attention in the ViT: already in place, nothing to gain
- **Checked in code (11 Nov):** `attn_full_k` and `attn_full_tc_k` (`kernels/src/cuda/ops.cu`)
  already stream K and V through shared memory in tiles of 64 or 16 rows, with an f32 online
  softmax, as flash attention does. No score matrix is materialized. The research report's concern
  (llama.cpp's clip path materialized the scores before PR #16837) does not apply to larch.
- **Verdict:** nothing to change.

### 2. Loop stop: kept (repeat penalty and sampling preset still running)
- **What the "loops" were:** of the 7 answers the scorer flags on the 0.8B, only 3 are degenerate
  loops: a run of `^` under a compiler error, a run of ```` ``` ```` lines, and "command, option"
  alternating. The other 4 are long answers that reached the token limit.
- **Two rules simulated on the finished runs (`eval-v2/loopstop_sim.py`, `periodic_sim.py`):**
  - **An 8-word sequence repeated 3 times:** catches 3 of 7 and also cuts real content. On the 0.8B
    it stops 1 non-loop answer, a spreadsheet whose CER goes from 0.07 to 0.41. Looser settings cut
    up to 12 non-loop answers. **Rejected.**
  - **One unit of 1-64 characters repeated over 200 characters and at least 8 copies:** catches
    exactly the 3 degenerate loops and nothing else on either model. It saves 15 s of decoding over
    the 0.8B run, and the cargo transcription's CER goes from 4.99 to 0.56. **Kept.**
- **Built:**
  - `TextStream::stop_loops` (f4 70366f52) ends the answer at the loop as a stop sequence would,
    and is off by default.
  - `Agent::stop_loops` (agent e877de84), which the app turns on for every chat, adds the notice
    "The reply began repeating itself, so it was stopped there."
  - Tests: 1,457 pass in larch, larch-agent and larch-app, including a new loop test. Six
    existing cap tests deliberately use a fake model that repeats "a", which is why the setting is
    off by default and none of those tests had to change.
- **Repeat penalty 1.1 (llama.cpp's, on a window): rejected.**
  - 0.8B: +0.013 (better on 23 items, worse on 17, p=0.43). Loops go from 7 to 4. Charts improve
    (+0.16), but tables, code and small-font transcription get worse (−0.07 to −0.08), where tokens
    legitimately repeat.
  - 2B: −0.007 (better on 13, worse on 17, p=0.59). Loops stay at 3, code transcription is −0.08,
    and the time to first token is slightly higher.
  - The loop stop handles the degenerate loops without these losses.
- **Pending (paused):** the model card's sampled preset (temperature 0.7, top-p 0.8, top-k 20) on the
  0.8B. A true presence penalty would need a change to the kernel contract (`larch_sample_params`),
  so it is not worth building on these results.

### Encoder profile (nsys, 1280x720 screenshot)
| Model | GPU time per encode | Attention (`attn_full_tc_k`) | GEMMs | Separate bias passes | GELU, layernorm, rope |
|---|---|---|---|---|---|
| 2B projector | 522 ms | 203 ms (39%) | 233 ms (45%) | 50 ms (10%) | 33 ms (6%) |
| 0.8B projector | 173 ms | 74 ms (43%) | 65 ms (37%) | 19 ms (11%) | 15 ms (9%) |

Attention runs at about 6.5 TFLOPS, a third of this GPU's tensor-core peak. It is the biggest single
piece of encoder time, and none of the 35 research items targeted it.

### 9. f16 accumulation in the encoder GEMMs: rejected
- **Built:** `LARCH_EXP_GEMM16`, an experiment switch on f16-output GEMMs (QKV and MLP up) only. The
  output and down projections accumulate into the f32 residual, so f16 compute cannot apply to
  them. Not committed.
- **Speed (median of 6, `diag_encode`):** 2B 539 → 495 ms (−8%), 0.8B 179 → 170 ms (−5%). That is
  far from the "about 2x" estimate, because only 58% of GEMM FLOPs are eligible and attention is
  untouched.
- **Precision:** worst row cosine against the current f32 path is 0.865 (2B, wide.png) and 0.932
  (0.8B). The current encoder sits at 0.996 against an f64 reference. Losing that for an 8% gain is
  not worth it.

### 12b. Larger K/V tiles in the encoder attention: rejected (marginal)
- **Built:** `attn_full_tc_k` templated on the tile size (16, the shipped kernel; 32; 64), behind
  `LARCH_EXP_ATTN_KT`.
- **Result:** 2B 539 → 518 ms (−4%), 0.8B 179 → 176 ms (−1%). Worst cosine 0.995-0.9995 against the
  shipped tile size, from the re-ordered online softmax. Tile overhead is not what limits this kernel.

### 6. Bias epilogue: the in-place bound shows a real gain; doing it properly
- **The bound:** the in-place switch (`LARCH_EXP_BIAS_INPLACE`: f16 GEMM output, then a bias add in
  place) gives 2B 539 → 507 ms (−6%) and 0.8B 179 → 172 ms. But adding the bias after the f16
  rounding moves rows (worst cosine 0.902), so it is only a speed bound.
- **The proper version, being built:** the cuBLASLt bias epilogue adds the f32 bias to the f32
  accumulator before rounding, which is the same arithmetic as the shipped two-pass route, in one
  pass. It also covers the f32-residual GEMMs (`bias_k<float>`, about 14 ms per 2B encode). cuBLAS
  already loads cuBLASLt, so linking it adds no runtime dependency.

### New finding: the first image after a model load is slow
- **Measured:** in the baseline runs, the first image's time to first token is 1,602 ms on the 2B
  against about 200 ms for similar later images, and 476 ms against about 140 ms on the 0.8B. In a
  fresh process the encoder alone takes 700-1,000 ms on its first encode against about 540 ms after.
- **What's already there:** the engine's load-time warm-up encodes only the smallest blank image,
  whose GEMM shapes and kernels differ from a real screenshot's.
- **Next:** a probe (`warm_probe.sh`, queued) compares no warm-up, a text turn first and an image
  turn first, to find which warm-up removes the penalty.

### 3. Thinking by task (in progress)
- **Unlimited thinking on the 0.8B (fast60):** a mean score change of −0.019 (better on 6 items, worse
  on 9, p=0.61). It is worse on charts (−0.20), documents (−0.14) and scene text (−0.33), and better
  on terminal and build-log transcription (+0.40, +0.47), where it avoided loops. It costs 4x the
  tokens (median 80 → 320) and 3.7x the time per answer (median 1.3 → 4.8 s).
- **What the app does today:** Qwen 3.5 counts as the Qwen3 family (`larch::toolcall::family_for`), so every
  chat step thinks, hidden, with the default 100-token budget (`config::Thinking`). The eval
  baselines had thinking off. Queued: the app's own setting (thinking with a 100-token budget) on
  both models over all 256 items, to decide whether image turns should stop thinking.

## Resumed 12 Nov, 15:06 (the other session finished; the GPU is ours)

### The experiment-switch build changes the default output (found 12 Nov)
- **Seen:** the first trials after the pause changed answers on items they could not affect (the 1024 cap
  changed 70 answers, only 42 images were capped).
- **Checked:** fresh baselines with today's build (`base-08b-c`, `base-2b-c`), and the harness rebuilt from
  84a71b0a, before the switches (`~/larch-imgref`, run `ref-2b-sub` on 30 items).
  - The pre-switch build reproduces the 11 Nov baseline exactly (30 of 30 answers identical).
  - Today's build, all switches off, matches it on 6 of 30. Within one build, greedy runs are still fully
    deterministic (the cap runs differ from `base-2b-c` only on capped images).
  - Score effect is negligible: 0.8B +0.000 (17 answers' text differ, 1 item better), 2B −0.003 (5 better,
    2 worse, p=0.45).
- **So:** every trial is now scored against the fresh baselines from the same build. Commit the experiment-switch commit (on the work branch, not in this history)
  ("experiment switches, off by default") is not bit-identical when off. Suspect: the templated
  `attn_full_tc_k` (restructured V loop, the new `lo` argument). To bisect after the timed queue.
- **Rescored against the fresh baselines; verdicts unchanged:** 0.8B full thinking −0.019 (6/9), repeat penalty
  0.8B +0.013 (22/17, n.s.), 2B −0.004 (11/19, n.s.), 2B full thinking −0.089 on fast60 (4/8).

### Model card sampled preset on the 0.8B: rejected
- Temperature 0.7, top-p 0.8, top-k 20, repeat penalty 1.1, seed 1: −0.031 (better 17, worse 33, p=0.03).
  Tables −0.29, small fonts −0.27, receipts −0.08; build logs +0.50. Loops 7 → 7; speed unchanged.

### 11. Image token cap on the 2B (the shipped cap is 1,600 tokens; the eval images use 64-1,600)
| Cap | Images capped | Their tokens (median) | Their TTFT median / total | Score vs base-2b-c | Lost on |
|---|---|---|---|---|---|
| 1024 | 42 | 1,551 → 999 | 2,132 → 1,180 ms / 76 → 46 s | −0.016 (1 better, 6 worse, p=0.13) | DocVQA (−0.12 on docs), OCRBench document pages |
| 576 | 76 | 1,026 → 551 | 1,167 → 654 ms / 102 → 45 s | −0.016 (2 better, 7 worse, p=0.18) | KIE −0.12, the wide status bar, OCRBench pages |
- The 576 run's decode speed (36 tok/s) is not valid: a CPU build overlapped it. A clean timing rerun
  (`p1-cap576-2b-t`) is queued.
- **Reading so far:** capping trades about 1.5 points, all on large document pages, for roughly halving the
  time to first token on those images. Not a default. 0.8B caps are next.

## Paused (11 Nov, ~13:35) for agent development
- **Last job run:** p1-rp110-2b. The remaining 18 phase-1 jobs are listed in
  `eval-v2/paused-phase1.txt`; to pause them they were appended to `queue-phase1.txt.done`.
- **To resume:** remove those names from `queue-phase1.txt.done`
  (`grep -v -x -F -f paused-phase1.txt queue-phase1.txt.done > d && mv d queue-phase1.txt.done`),
  then run `FIRST_FREE=0 ./chain.sh queue-phase1.txt` in `eval-v2`.
- **Built but not yet measured:** the cuBLASLt bias epilogue (`LARCH_EXP_LT_BIAS`), single-f16
  attention weights (`LARCH_EXP_ATTN_NOLO`, job p2-encode-ab2), the warm-up probe, two-pass,
  quantization variants (downloads in `~/larch-models/q`, about 5 GB, to delete when done),
  upscaling to a minimum token count (`LARCH_IMAGE_MIN_TOKENS`), and the app's 100-token thinking
  budget.
- **Uncommitted experiment hooks** in `~/larch-imgopt`: `kernels/src/cuda/{quant,ops}.cu`,
  `kernels/CMakeLists.txt`, `crates/larch-sys/build.rs`, `crates/larch-model/src/qwen_vision.rs`,
  `crates/larch-model/examples/diag_encode.rs`. The harness changes since 84a71b0a (warm, two-pass,
  thinking budget) are also uncommitted.
