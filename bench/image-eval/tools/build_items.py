# Merges gen/gen_items.jsonl + public_items.jsonl, adds "describe" items, validates everything,
# writes items.jsonl + fast60.txt and refreshes the counts block in README.md.
# Run: uv run --with pillow python -I build_items.py <eval-v2 dir>
import collections, json, os, random, re, sys
from PIL import Image, ImageOps

ROOT = os.path.abspath(sys.argv[1])
SEED = 20261111
DESCRIBE = "Describe this image in detail."


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


gen = load(os.path.join(ROOT, "gen", "gen_items.jsonl"))
pub = load(os.path.join(ROOT, "public_items.jsonl"))
items = gen + pub

# ---- describe items: 11 generated pages + 9 public images
GEN_DESC = ["term_cargo", "code_rust", "dialog_sync", "settings_storage", "sheet_sales", "receipt_cafe", "chat_phone",
            "banner_wide", "tall_release", "photo_platform", "small_props"]
PUB_DESC = [("ocrbench", "scene_text", 2), ("ocrbench", "kie", 1), ("chartqa", "chart", 3), ("docvqa", "doc", 3)]
desc = []
for page in GEN_DESC:
    base = next(i for i in gen if i["meta"].get("page") == page and i["kind"] == "transcribe" and i["meta"]["hidpi"] == 1)
    desc.append((base, f"describe_gen_{page}"))
seen = set()
for src, cat, n in PUB_DESC:
    pool = [i for i in pub if i["source"] == src and i["category"] == cat and i["image"] not in seen]
    for base in random.Random(f"{SEED}{src}{cat}").sample(sorted(pool, key=lambda i: i["id"]), n):
        seen.add(base["image"])
        desc.append((base, f"describe_{base['id']}"))
for base, ident in desc:
    meta = {k: v for k, v in base["meta"].items()}
    meta["base_id"], meta["base_category"] = base["id"], base["category"]
    items.append({"id": ident, "image": base["image"], "source": base["source"], "category": "describe",
                  "kind": "describe", "prompt": DESCRIBE, "answers": [], "meta": meta})

# ---- validation
errors = []
ids = collections.Counter(i["id"] for i in items)
for k, v in ids.items():
    if v > 1:
        errors.append(f"duplicate id {k}")
img_ok = {}
for it in items:
    i = it["id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", i):
        errors.append(f"{i}: id not filesystem-safe")
    for k in ("image", "source", "category", "kind", "prompt", "answers", "meta"):
        if k not in it:
            errors.append(f"{i}: missing {k}")
    if it["source"] not in ("gen", "ocrbench", "chartqa", "docvqa"):
        errors.append(f"{i}: bad source")
    kind = it["kind"]
    if kind not in ("transcribe", "qa", "absent", "describe"):
        errors.append(f"{i}: bad kind")
    if kind == "transcribe" and not it.get("text"):
        errors.append(f"{i}: transcribe without text")
    if kind != "transcribe" and "text" in it:
        errors.append(f"{i}: text on non-transcribe item")
    if kind in ("qa", "absent") and not (it["answers"] and all(isinstance(a, str) and a for a in it["answers"])):
        errors.append(f"{i}: empty answers")
    if kind == "describe" and it["answers"]:
        errors.append(f"{i}: describe with answers")
    if kind == "absent" and not it["prompt"].endswith("If it is not shown, answer 'not shown'."):
        errors.append(f"{i}: absent prompt suffix")
    p = it["image"]
    if not os.path.isabs(p):
        errors.append(f"{i}: image path not absolute")
    if p not in img_ok:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                size, fmt = im.size, im.format
                up = ImageOps.exif_transpose(im).size
            img_ok[p] = (size, fmt, up)
        except Exception as e:  # noqa
            img_ok[p] = None
            errors.append(f"{i}: image does not open: {p}: {e}")
    info = img_ok[p]
    if info and it["category"] == "photo_exif":
        m = it["meta"]
        if info[2] != (m["upright_width"], m["upright_height"]) or info[0] == info[2] and m["exif_orientation"] in (6, 8):
            errors.append(f"{i}: EXIF orientation does not restore the upright size")
    if info and it["source"] == "gen" and it["category"] != "photo_exif":
        if info[0] != (it["meta"]["width"], it["meta"]["height"]):
            errors.append(f"{i}: meta size mismatch")
# json round trip of the serialised lines
lines = [json.dumps(it, ensure_ascii=False) for it in items]
for l in lines:
    json.loads(l)
if errors:
    print("\n".join(errors))
    sys.exit(1)

with open(os.path.join(ROOT, "items.jsonl"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ---- fast60: stratified by (source, kind), round-robin across categories, balancing category use
QUOTA = [(("gen", "transcribe"), 10), (("gen", "qa"), 13), (("gen", "absent"), 7), (("gen", "describe"), 3),
         (("ocrbench", "qa"), 16), (("ocrbench", "describe"), 1), (("chartqa", "qa"), 4), (("chartqa", "describe"), 1),
         (("docvqa", "qa"), 4), (("docvqa", "describe"), 1)]
assert sum(q for _, q in QUOTA) == 60
PRIORITY = ["photo_exif", "hidpi", "small_font", "wide", "table", "code", "terminal", "dialog", "receipt"]
rng = random.Random(SEED)
used = collections.Counter()
fast = []
for (src, kind), q in QUOTA:
    pool = [i for i in items if i["source"] == src and i["kind"] == kind]
    bycat = collections.defaultdict(list)
    for it in sorted(pool, key=lambda i: i["id"]):
        c = it["meta"].get("base_category", it["category"]) if kind == "describe" else it["category"]
        bycat[c].append(it)
    for c in bycat:
        rng.shuffle(bycat[c])
    tiebreak = {c: rng.random() for c in bycat}
    size = {c: len(v) for c, v in bycat.items()}
    here = collections.Counter()
    picked = []
    while len(picked) < q and any(bycat.values()):
        # least-sampled share of the category within this group first, then least used overall
        order = sorted((c for c in bycat if bycat[c]),
                       key=lambda c: (here[c] / size[c], used[c], PRIORITY.index(c) if c in PRIORITY else 99,
                                      tiebreak[c]))
        c = order[0]
        picked.append(bycat[c].pop())
        used[c] += 1
        here[c] += 1
    assert len(picked) == q, (src, kind)
    fast += [i["id"] for i in picked]
assert len(set(fast)) == 60
with open(os.path.join(ROOT, "fast60.txt"), "w") as f:
    f.write("\n".join(fast) + "\n")


# ---- counts
def table(title, key, rows):
    c = collections.Counter(key(i) for i in rows)
    out = [f"**{title}**", "", "| value | count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    return "\n".join(out) + "\n"


fset = set(fast)
fitems = [i for i in items if i["id"] in fset]
sk = collections.Counter((i["source"], i["kind"]) for i in items)
grid = ["**source x kind (all items)**", "", "| source | transcribe | qa | absent | describe | total |", "|---|---|---|---|---|---|"]
for s in ("gen", "ocrbench", "chartqa", "docvqa"):
    r = [sk[(s, k)] for k in ("transcribe", "qa", "absent", "describe")]
    grid.append(f"| {s} | " + " | ".join(map(str, r)) + f" | {sum(r)} |")
tot = [sum(sk[(s, k)] for s in ("gen", "ocrbench", "chartqa", "docvqa")) for k in ("transcribe", "qa", "absent", "describe")]
grid.append("| **total** | " + " | ".join(map(str, tot)) + f" | {sum(tot)} |")
counts = "\n".join([
    f"Total items: **{len(items)}** ({len(img_ok)} distinct images). fast60: {len(fast)} ids.", "",
    "\n".join(grid), "",
    table("category (all items)", lambda i: (i["source"], i["category"]), items),
    table("generated pages by font_px (transcribe items)", lambda i: i["meta"].get("font_px"),
          [i for i in gen if i["kind"] == "transcribe"]),
    table("generated pages by family (transcribe items)", lambda i: i["meta"].get("family"),
          [i for i in gen if i["kind"] == "transcribe"]),
    table("generated pages by theme (transcribe items)", lambda i: i["meta"].get("theme"),
          [i for i in gen if i["kind"] == "transcribe"]),
    table("fast60 by source/kind", lambda i: (i["source"], i["kind"]), fitems),
    table("fast60 by category", lambda i: i["category"] if i["kind"] != "describe" else "describe:" + i["meta"]["base_category"], fitems),
])
print(counts)
rp = os.path.join(ROOT, "README.md")
if os.path.exists(rp):
    s = open(rp).read()
    a, b = "<!-- counts:start -->", "<!-- counts:end -->"
    if a in s and b in s:
        s = s[: s.index(a) + len(a)] + "\n" + counts + "\n" + s[s.index(b):]
        open(rp, "w").write(s)
        print("README counts refreshed")
print("OK", len(items), "items")
