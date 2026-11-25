# Samples public OCR / doc VQA items via the Hugging Face datasets-server API and downloads
# their images with curl into <eval-v2>/public/<source>/.  Downloads are treated as data only.
# Held-out variant: other rows than eval-v2 (fetch_heldout.py <dir> [eval-v2 items.jsonl]).
# Run: python3 -I fetch_public.py <eval-v2 dir>     (stdlib only; writes public_items.jsonl)
import json, os, random, subprocess, sys, time, urllib.parse

ROOT = os.path.abspath(sys.argv[1])
API = "https://datasets-server.huggingface.co/rows"
SEED = 20261112  # held-out: a different seed, and every eval-v2 row excluded
# The eval set's items.jsonl, whose rows are excluded: the second argument, or the one in the
# parent of <dir> (<eval-v2>/heldout).
ITEMS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "..", "items.jsonl")
EXCLUDE = {json.loads(l)["id"] for l in open(ITEMS)}


def curl(url, out=None):
    for attempt in range(8):
        args = ["curl", "-sSfL", "--max-time", "120", url]
        if out:
            args += ["-o", out]
        r = subprocess.run(args, capture_output=True)
        if r.returncode == 0:
            return r.stdout
        time.sleep(5 + 15 * attempt)
    raise RuntimeError(f"curl failed: {url[:120]} {r.stderr[:200]!r}")


def rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=length))
    return json.loads(curl(f"{API}?{q}"))["rows"]


def get_image(src, source, ident):
    d = os.path.join(ROOT, "public", source)
    os.makedirs(d, exist_ok=True)
    path_part = urllib.parse.urlparse(src).path
    ext = os.path.splitext(path_part)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".img"
    out = os.path.join(d, ident + ext)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        curl(src, out)
    return out


items = []

# ---------------- OCRBench (echo840/OCRBench, test, 1000 rows) ----------------
ocr = []
for off in range(0, 1000, 100):
    ocr += rows("echo840/OCRBench", "default", "test", off, 100)
groups = {}
for r in ocr:
    groups.setdefault((r["row"]["question_type"], r["row"]["dataset"]), []).append(r)
# per (question_type, sub-dataset): how many to sample  -> 70 total
PLAN = {
    ("Regular Text Recognition", "IIIT5K"): 2, ("Regular Text Recognition", "IC13_857"): 2,
    ("Regular Text Recognition", "svt"): 1,
    ("Irregular Text Recognition", "IC15_1811"): 1, ("Irregular Text Recognition", "cocotext"): 1,
    ("Irregular Text Recognition", "ct80"): 1, ("Irregular Text Recognition", "svtp"): 1,
    ("Irregular Text Recognition", "totaltext"): 1,
    ("Artistic Text Recognition", "WordArt"): 4,
    ("Digit String Recognition", "ORAND"): 3,
    ("Non-Semantic Text Recognition", "NonSemanticText"): 3,
    ("Handwriting Recognition", "IAM"): 3,
    ("Handwritten Mathematical Expression Recognition", "HME100k"): 2,
    ("Scene Text-centric VQA", "ESTVQA"): 4, ("Scene Text-centric VQA", "STVQA"): 4,
    ("Scene Text-centric VQA", "ocrVQA"): 4, ("Scene Text-centric VQA", "textVQA"): 4,
    ("Doc-oriented VQA", "docVQA"): 4, ("Doc-oriented VQA", "infographicVQA"): 4,
    ("Doc-oriented VQA", "ChartQA"): 2, ("Doc-oriented VQA", "ChartQA_Human"): 2,
    ("Key Information Extraction", "FUNSD"): 6, ("Key Information Extraction", "POIE"): 5,
    ("Key Information Extraction", "SROIE"): 6,
}
CAT = {
    "Regular Text Recognition": "text_recog", "Irregular Text Recognition": "text_recog_irregular",
    "Artistic Text Recognition": "artistic_text", "Digit String Recognition": "digits",
    "Non-Semantic Text Recognition": "text_recog_nonsemantic", "Handwriting Recognition": "handwriting",
    "Handwritten Mathematical Expression Recognition": "handwriting_math",
    "Scene Text-centric VQA": "scene_text", "Key Information Extraction": "kie",
}
rng = random.Random(SEED)
for key in sorted(PLAN):
    pool = sorted(groups[key], key=lambda r: r["row_idx"])
    pool = [r for r in pool if f"ocrbench_{r['row_idx']:04d}" not in EXCLUDE]
    for r in rng.sample(pool, min(len(pool), 2 * PLAN[key])):
        row, idx = r["row"], r["row_idx"]
        qt, sub = key
        cat = CAT.get(qt)
        if cat is None:  # Doc-oriented VQA
            cat = {"ChartQA": "chart", "ChartQA_Human": "chart", "infographicVQA": "infographic"}.get(sub, "doc")
        ident = f"ocrbench_{idx:04d}"
        img = get_image(row["image"]["src"], "ocrbench", ident)
        items.append({
            "id": ident, "image": img, "source": "ocrbench", "category": cat, "kind": "qa",
            "prompt": row["question"], "answers": list(row["answer"]),
            "meta": {"row_idx": idx, "question_type": qt, "subset": sub,
                     "width": row["image"].get("width"), "height": row["image"].get("height")},
        })
    print("ocrbench", key, PLAN[key], flush=True)

# ---------------- ChartQA (HuggingFaceM4/ChartQA, test, 2500 rows) ----------------
rng = random.Random(SEED + 1)
for idx in sorted(i for i in rng.sample(range(2500), 40) if f"chartqa_{i:04d}" not in EXCLUDE)[:20]:
    r = rows("HuggingFaceM4/ChartQA", "default", "test", idx, 1)[0]
    row = r["row"]
    ident = f"chartqa_{idx:04d}"
    img = get_image(row["image"]["src"], "chartqa", ident)
    hm = row.get("human_or_machine")
    items.append({
        "id": ident, "image": img, "source": "chartqa", "category": "chart", "kind": "qa",
        "prompt": row["query"], "answers": list(row["label"]),
        "meta": {"row_idx": idx, "human_or_machine": {0: "human", 1: "machine"}.get(hm, hm),
                 "width": row["image"].get("width"), "height": row["image"].get("height")},
    })
    print("chartqa", idx, flush=True)

# ---------------- DocVQA (lmms-lab-encoder/DocVQA [was lmms-lab/DocVQA], DocVQA config, validation) ----------------
rng = random.Random(SEED + 2)
seen_docs = set()
for idx in sorted(rng.sample(range(5349), 40)):
    if len(seen_docs) >= 20:
        break
    r = rows("lmms-lab-encoder/DocVQA", "DocVQA", "validation", idx, 1)[0]
    row = r["row"]
    if row["docId"] in seen_docs or f"docvqa_{idx:04d}" in EXCLUDE:
        continue
    seen_docs.add(row["docId"])
    ident = f"docvqa_{idx:04d}"
    img = get_image(row["image"]["src"], "docvqa", ident)
    items.append({
        "id": ident, "image": img, "source": "docvqa", "category": "doc", "kind": "qa",
        "prompt": row["question"], "answers": list(row["answers"]),
        "meta": {"row_idx": idx, "questionId": row["questionId"], "docId": row["docId"],
                 "question_types": row.get("question_types"),
                 "width": row["image"].get("width"), "height": row["image"].get("height")},
    })
    print("docvqa", idx, flush=True)

with open(os.path.join(ROOT, "public_items.jsonl"), "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(items), "public items")
