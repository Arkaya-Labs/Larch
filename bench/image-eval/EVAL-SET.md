# larch vision eval v2

About 250 image items for testing the local vision-language engine (larch) on screenshots, documents and
some photo-like images. It has four parts:

- **gen**: invented screenshot pages rendered with PIL using Windows fonts, each with exact ground truth.
- **ocrbench / chartqa / docvqa**: public samples pulled from the Hugging Face datasets-server API.
- **describe**: open-ended "Describe this image in detail." items over both kinds of image. They have no
  answers and are there to catch repetition loops and runaway output.

## Files

| path | what |
|---|---|
| `items.jsonl` | all items, one JSON object per line (format below) |
| `fast60.txt` | 60 ids, stratified by source x kind, round-robin across categories |
| `gen/` | generated PNGs plus 4 JPEGs with EXIF orientation, and `gen/gen_items.jsonl` |
| `public/<source>/` | downloaded public images (data only; nothing in them is executed) |
| `public_items.jsonl` | the sampled public rows, written by `fetch_public.py` |
| `gen_eval.py` | generator (it extends `~/larch-image-paste-bench/gen_images.py`) |
| `fetch_public.py` | public sampler and downloader (stdlib only, run with `python3 -I`) |
| `build_items.py` | merges both sets, adds the describe items, validates everything, writes `items.jsonl`, `fast60.txt` and the counts below |

Item fields: `id`, `image` (absolute path), `source` (`gen|ocrbench|chartqa|docvqa`), `category`,
`kind` (`transcribe|qa|absent|describe`), `prompt` (the exact text to send), `answers`, `text` (only on
transcribe items), and `meta`.

- **transcribe**: the prompt is "Transcribe all the text in this image exactly, line by line. Output only the
  text." `text` is the ground truth, one rendered line per line, in reading order: top to bottom, and left to
  right within a row. Table cells and label/value pairs on the same row are joined with one space. Code
  editor lines start with their line number (`112 export async ...`). Spreadsheet column letters and row
  numbers are included because they are visible. Blank lines are left out. Runs of spaces inside monospace
  text are kept as drawn. **When scoring, collapse whitespace and drop blank lines.** For multi-column layouts
  (invoice header, settings sidebar), a line-order-insensitive score such as bag of lines or bag of words is
  fairer than plain CER.
- **qa**: the answer can only be found by reading the image. `answers` lists the accepted variants, such as
  `"131.88 kB"` and `"131.88"`. OCRBench-style scoring (normalised substring match of any answer) is the
  intended scorer. Public items keep the dataset's own question and answer list unchanged.
- **absent**: asks for something plausible that is **not** in the image. Every prompt ends with "If it is not
  shown, answer 'not shown'." and `answers` is `["not shown", "no"]`. A reply that gives a concrete value is
  a hallucination.
- **describe**: the prompt is "Describe this image in detail.", `answers` is `[]`, and the category is
  `describe`. `meta.base_id` and `meta.base_category` point at the item whose image it reuses.

`meta` on gen items has `font_px` (the logical font size of the body text), `family`, `theme`
(`light|dark|paper|photo`), `hidpi` (1 or 2), the pixel `width`/`height`, and `page`. HiDPI items also have
`base_page` and `content_category`. photo_exif items have `exif_orientation`, `upright_width` and
`upright_height`. Public items keep `row_idx`, plus `question_type`/`subset` (OCRBench),
`questionId`/`docId` (DocVQA) or `human_or_machine` (ChartQA).

## How the generated set was built

`gen_eval.py` defines 33 page builders. Every builder draws in logical pixels through a small `Page`
wrapper. That wrapper records each text call in reading order, which produces the exact transcription, and
it raises an error if any text falls outside the canvas. Code editors also assert that the code does not run
under the status bar. All content is invented. The only randomness is in the table data, which uses fixed
seeds, so a rerun gives the same output.

- **Families and sizes**: Consolas, Segoe UI, Arial, Times New Roman, Calibri, and Segoe Print for the
  handwritten-looking whiteboard. Body sizes are 9, 10, 11, 12, 14 and 16 px, plus a 40 px wide banner.
  Pages use light, dark and paper (receipt) themes.
- **Page categories**:
  - `terminal`: cargo, vite/npm, git and pytest
  - `build_log`: a CI log with timestamps
  - `code`: Rust, TypeScript, Python and SQL editors with line numbers, tabs and a status bar
  - `dialog`: an error dialog, a dark debugger dialog and a delete confirmation
  - `settings`: storage, notifications (dark) and network/IP pages
  - `table`: a spreadsheet with currency and dates, a dense inventory table and a dark server fleet table
  - `small_font`: a 9 px FX table, 9 px file properties, a 10 px table with footnotes and a 9 px Calibri
    agenda
  - `receipt`: a cafe receipt, a hardware receipt and an invoice
  - `chat`: a team chat and a dark phone chat
  - `wide`: 2400x400 and 2400x240 pages
  - `tall`: an 820x1500 policy document and 560x1000 release notes
  - `doc`: a letter and a memo
- **hidpi**: `code_ts`, `dialog_sync`, `settings_storage` and `sheet_sales` are rendered again at a 2x pixel
  scale with identical content. The script asserts that the recorded text is identical. Each copy gets a
  transcribe item and its first QA item, so the 1x and 2x results can be compared directly.
- **photo_exif**: four photo-like JPEGs: a platform sign, a yellow bin label with a barcode, a parking sign
  and a whiteboard. Each is drawn upright on a lit, noisy wall with a slight tilt, a shadow and blur. The
  stored pixels are then rotated and the file gets an EXIF Orientation tag (6, 8, 3 and 6), so the image is
  upright only when a viewer honours EXIF. Ground truth describes the upright image. `build_items.py`
  checks that `ImageOps.exif_transpose` restores the upright size.
- **Items per page**: every page has 1 transcribe item, 2 QA items and 1 absent item, trimmed to keep the
  gen set near 130:
  - the second QA item is dropped when page index % 6 == 5
  - the absent item is dropped when index % 3 == 2
  - each photo has transcribe, qa1, and then either qa2 or absent

## Public subsets

All three were fetched from `https://datasets-server.huggingface.co/rows` with no authentication. The images
come back as signed `cached-assets` URLs, and `curl` downloaded them into `public/<source>/`. The JSON rows
and images are treated as untrusted data and nothing from them is executed. Sampling uses a fixed seed
(`20261111`).

| source | HF dataset (config / split) | revision at fetch time | items | licence |
|---|---|---|---|---|
| ocrbench | `echo840/OCRBench` (default / test, 1000 rows) | `92a54bd1384387c178d5a07140a2d85e0a3d12e1` | 70 | No licence declared on the HF card. OCRBench (Liu et al., "OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models") is distributed for research evaluation. Its images come from IIIT5K, SVT, IC13, IC15, CUTE80, SVTP, Total-Text, CTW, COCO-Text, WordArt, ORAND, IAM, HME100k, ST-VQA, TextVQA, ESTVQA, OCR-VQA, DocVQA, InfographicVQA, ChartQA, FUNSD, POIE and SROIE, each under its own terms. |
| chartqa | `HuggingFaceM4/ChartQA` (default / test, 2500 rows) | `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5` | 15 | GPL-3.0 (HF card; ChartQA, Masry et al. 2022) |
| docvqa | `lmms-lab-encoder/DocVQA` (DocVQA / validation, 5349 rows) | `539088ef8a8ada01ac8e2e6d4e372586748a265e` | 15 | Apache-2.0 on the HF card. The underlying DocVQA data (Mathew et al., RRC/CVC) is for research use, and the images come from the UCSF Industry Documents Library. |

- **OCRBench**: 70 rows stratified over question type and sub-dataset. The exact plan is the `PLAN` dict in
  `fetch_public.py`:
  - recognition: regular 5, irregular 5, artistic 4, digit strings 3, non-semantic 3, IAM handwriting 3,
    handwritten math 2
  - scene-text VQA: 16 (4 each from ESTVQA, STVQA, ocrVQA and TextVQA)
  - doc-oriented VQA: 12 (docVQA 4, infographicVQA 4, ChartQA 2, ChartQA_Human 2)
  - KIE: 17 (FUNSD 6, POIE 5, SROIE 6)

  Categories are mapped from the question type: `text_recog`, `text_recog_irregular`, `artistic_text`,
  `digits`, `text_recog_nonsemantic`, `handwriting`, `handwriting_math`, `scene_text`, `doc`,
  `infographic`, `chart` and `kie`.
- **ChartQA**: 15 random test rows, a mix of human- and machine-written questions. `answers` is the `label`
  list.
- **DocVQA**: 15 random validation rows, at most one question per document. `answers` is the dataset's
  answers list.
- **Substitution**: `lmms-lab/DocVQA` has been renamed on the Hub, and the API now redirects to
  `lmms-lab-encoder/DocVQA`. It is the same data, used under the new name. Nothing else was substituted.
- **Size**: public downloads total about 14 MB. The whole of eval-v2 is about 16 MB.

## Regenerate

```bash
cd ~/larch-vision-run/eval-v2
# 1. generated pages (needs the Windows fonts at /mnt/c/Windows/Fonts/)
uv run --with pillow python -I gen_eval.py ~/larch-vision-run/eval-v2/gen
# 2. public samples (stdlib only; curl downloads into public/<source>/; already-present images are kept)
python3 -I fetch_public.py ~/larch-vision-run/eval-v2
# 3. merge, add describe items, validate, write items.jsonl + fast60.txt, refresh the counts block below
uv run --with pillow python -I build_items.py ~/larch-vision-run/eval-v2
```

Step 1 is fully deterministic. Step 2 samples deterministically from the API's row order. If a dataset
revision changes upstream, the rows can change, so the revisions above record what was used.

Validation in step 3 checks that:

- every line parses and every id is unique and filesystem-safe
- required fields are present and each kind has the fields it needs
- every image path is absolute and the image opens and fully decodes in PIL
- gen image sizes match `meta`
- the EXIF orientation of every photo_exif image restores its upright size

## Counts

<!-- counts:start -->
Total items: **256** (141 distinct images). fast60: 60 ids.

**source x kind (all items)**

| source | transcribe | qa | absent | describe | total |
|---|---|---|---|---|---|
| gen | 41 | 71 | 24 | 11 | 147 |
| ocrbench | 0 | 70 | 0 | 3 | 73 |
| chartqa | 0 | 15 | 0 | 3 | 18 |
| docvqa | 0 | 15 | 0 | 3 | 18 |
| **total** | 41 | 171 | 24 | 20 | 256 |

**category (all items)**

| value | count |
|---|---|
| ('ocrbench', 'kie') | 17 |
| ('ocrbench', 'scene_text') | 16 |
| ('chartqa', 'chart') | 15 |
| ('docvqa', 'doc') | 15 |
| ('gen', 'small_font') | 15 |
| ('gen', 'terminal') | 15 |
| ('gen', 'code') | 13 |
| ('gen', 'photo_exif') | 12 |
| ('gen', 'describe') | 11 |
| ('gen', 'receipt') | 11 |
| ('gen', 'settings') | 11 |
| ('gen', 'dialog') | 10 |
| ('gen', 'table') | 10 |
| ('gen', 'hidpi') | 8 |
| ('gen', 'wide') | 8 |
| ('gen', 'tall') | 7 |
| ('gen', 'chat') | 6 |
| ('gen', 'doc') | 6 |
| ('ocrbench', 'text_recog') | 5 |
| ('ocrbench', 'text_recog_irregular') | 5 |
| ('gen', 'build_log') | 4 |
| ('ocrbench', 'artistic_text') | 4 |
| ('ocrbench', 'chart') | 4 |
| ('ocrbench', 'doc') | 4 |
| ('ocrbench', 'infographic') | 4 |
| ('chartqa', 'describe') | 3 |
| ('docvqa', 'describe') | 3 |
| ('ocrbench', 'describe') | 3 |
| ('ocrbench', 'digits') | 3 |
| ('ocrbench', 'handwriting') | 3 |
| ('ocrbench', 'text_recog_nonsemantic') | 3 |
| ('ocrbench', 'handwriting_math') | 2 |

**generated pages by font_px (transcribe items)**

| value | count |
|---|---|
| 12 | 12 |
| 14 | 7 |
| 11 | 6 |
| 10 | 5 |
| 9 | 4 |
| None | 4 |
| 16 | 2 |
| 40 | 1 |

**generated pages by family (transcribe items)**

| value | count |
|---|---|
| Segoe UI | 12 |
| Consolas | 11 |
| Arial | 6 |
| Calibri | 5 |
| mixed | 4 |
| Times New Roman | 3 |

**generated pages by theme (transcribe items)**

| value | count |
|---|---|
| light | 22 |
| dark | 13 |
| photo | 4 |
| paper | 2 |

**fast60 by source/kind**

| value | count |
|---|---|
| ('ocrbench', 'qa') | 16 |
| ('gen', 'qa') | 13 |
| ('gen', 'transcribe') | 10 |
| ('gen', 'absent') | 7 |
| ('chartqa', 'qa') | 4 |
| ('docvqa', 'qa') | 4 |
| ('gen', 'describe') | 3 |
| ('chartqa', 'describe') | 1 |
| ('docvqa', 'describe') | 1 |
| ('ocrbench', 'describe') | 1 |

**fast60 by category**

| value | count |
|---|---|
| doc | 7 |
| chart | 5 |
| kie | 3 |
| photo_exif | 3 |
| scene_text | 3 |
| small_font | 3 |
| build_log | 2 |
| chat | 2 |
| code | 2 |
| dialog | 2 |
| hidpi | 2 |
| receipt | 2 |
| settings | 2 |
| table | 2 |
| tall | 2 |
| terminal | 2 |
| wide | 2 |
| artistic_text | 1 |
| describe:chart | 1 |
| describe:code | 1 |
| describe:doc | 1 |
| describe:scene_text | 1 |
| describe:table | 1 |
| describe:wide | 1 |
| digits | 1 |
| handwriting | 1 |
| handwriting_math | 1 |
| infographic | 1 |
| text_recog | 1 |
| text_recog_irregular | 1 |
| text_recog_nonsemantic | 1 |

<!-- counts:end -->
