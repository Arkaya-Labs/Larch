# Image eval v2 and the image optimization trials (work in progress)

Phase 1 (settings, caps, quantization, warm-up, encoder kernels) finished 12 Nov 2026; everything needed
to continue is here or regenerates from here.
Start with `docs/IMAGE-OPT-RESULTS.md` (what was tested, kept, rejected, pending) and
`docs/IMAGE-OPT-PLAN.md` (the 35 items and how each is tested).

## Layout
| Path | What |
|---|---|
| `docs/IMAGE-OPT-PLAN.md`, `docs/IMAGE-OPT-RESULTS.md` | The plan and the results so far |
| `docs/Image input accuracy and speed (research report).md` | The 35-item research report the trials work through; its notes are in `docs/research-notes/` |
| `docs/qwen35/` | The Qwen 3.5 image work's assumptions, GPU results and plan (PR #488) |
| `EVAL-SET.md` | How the 256-item eval set was built, counts, sources and licences |
| `tools/` | Generator, fetcher, item builder, scorer, simulations, the GPU queue and job scripts |
| `queue/` | Item-id lists for the queue (the job files themselves are not published) |
| `heldout/` | The held-out set (180 public items, other rows than the eval set; images not committed, `tools/fetch_heldout.py <dir>` downloads them) |

The harness is `crates/larch/examples/image_eval.rs`. The encode timer (`diag_encode`) and the
experiment switches were removed before the merge into dev. `LARCH_IMAGE_MAX_TOKENS` stays as an
option: it lowers the Qwen projector's image token cap.

## Not committed, and how to get it back
- **Generated screenshots** (`gen/`, 2.2 MB): `uv run --with pillow python -I tools/gen_eval.py <dir>/gen`
  rebuilds them byte for byte.
- **Public items** (OCRBench, ChartQA, DocVQA, 14 MB of images): `python3 -I tools/fetch_public.py <dir>`
  downloads the same rows (revisions pinned in `EVAL-SET.md`). They are not committed because of
  their licences (ChartQA GPL-3.0, DocVQA research-use, OCRBench none stated).
- **`items.jsonl`**: `uv run --with pillow python -I tools/build_items.py <dir>` writes it with
  absolute image paths for the machine it runs on.
- **Models:** Qwen3.5-0.8B Q8_0 (`m14.gguf`) and 2B Q4_K_M (`m15.gguf`) with their `mmproj-F16.gguf`,
  from `unsloth/Qwen3.5-0.8B-GGUF` and `unsloth/Qwen3.5-2B-GGUF`. The quantization trial also uses
  2B Q3_K_M, Q6_K and Q8_0 and 0.8B Q4_K_M from the same repos. InternVL3.5-1B Q8_0 is from
  `bartowski/OpenGVLab_InternVL3_5-1B-GGUF`.
- **Profiles and encoder dumps** (nsys reports, 127 MB): their numbers are in the results doc; rerun
  `tools/encode_ab*.sh` or `nsys profile ... diag_encode` (from the experiment-switch commit on the work branch) to get them again.

## Resuming
1. Regenerate the eval set as above (adjust the absolute paths in `tools/*.sh` and `queue/*.txt`,
   written for `~/...`).
2. Build on every core: `LARCH_BACKEND=CUDA LARCH_CUDA_ARCH=86 CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) cargo build -j $(nproc)
   --release -p larch --example image_eval` (under a memory cap, allow
   about 11 GB; a tighter cap makes a parallel build thrash).
3. Run a queue: `FIRST_FREE=1 FAIR_WAIT_MIN=2 tools/chain.sh <queue file>` from the eval directory. Each
   line is `<name> <command>`; finished names go to `<queue>.done` and are skipped on a rerun. The queue
   holds `~/src/gpu.lock` for each job and shares the GPU in turns of at most 40 minutes.
4. Score: `python3 -I tools/score_eval.py items.jsonl runs/<base>.jsonl runs/<trial>.jsonl`, and
   `python3 -I tools/affected.py runs/<base>.jsonl runs/<trial>.jsonl` for timings on the images a trial
   changed. Compare against a baseline from the same build (`base-*-a` for builds without the experiment switches).
