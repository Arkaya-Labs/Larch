"""Scores a docfollow run on plan 10.2's five scores, kept apart, and checks the gold passages.

  score.py gold [--cache DIR] [--fill]        every gold passage is on its page of its document's text
  score.py pack RESULTS --out PACK.jsonl      the rows a judge (a person or a strong model) reads with judge.md
  score.py score RESULTS [--verdicts V.jsonl] [--out S.json] [--md S.md]

Extraction, retrieval, citation support and completeness need no judge; correctness is the judge's
verdicts, and the gold-passage check compares the `gold` mode's verdicts with `v2`'s.
"""

import argparse
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', " ": " "}
# A reader's own words for "the document does not say": a hint beside the judge, never a verdict.
NOT_FOUND = re.compile(
    r"not (?:found|mentioned|stated|discussed|covered|addressed|given|specified)|"
    r"(?:does|did)n[o']?t (?:say|mention|state|discuss|cover|address|give|specify|appear)|"
    r"no (?:mention|literal match|information)|"
    r"(?:is|are) not (?:in|part of) the (?:document|text|paper)|nowhere in", re.IGNORECASE)
# PDFKit's text page by page, a form feed between pages; an empty page stays, so pages never shift.
PDFKIT = """ObjC.import("PDFKit");
function run(argv) {
  var d = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  if (d.isNil()) { throw new Error("PDFKit could not open the file"); }
  var out = [];
  for (var i = 0; i < d.pageCount; i++) { var s = d.pageAtIndex(i).string; out.push(s.isNil() ? "" : s.js); }
  return out.join("\\f");
}"""


def load_json(path):
    """A JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    """A JSON-lines file, blank lines skipped."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_map(s):
    """`s` folded for matching, as plan 7.6's check folds it (spaces, line breaks and curly quotes made
    the same, a line-end hyphen joined only between two letters), plus each folded character's byte
    offset in `s`."""
    out, at, byte, i = [], [], 0, 0
    while i < len(s):
        c = s[i]
        if c == "-" and i + 1 < len(s) and s[i + 1] == "\n" and i > 0 and s[i - 1].isalpha() and i + 2 < len(s) and s[i + 2].isalpha():
            byte += 2
            i += 2
            continue
        for d in unicodedata.normalize("NFKC", QUOTES.get(c, c)).casefold():
            if d.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
                    at.append(byte)
            else:
                out.append(d)
                at.append(byte)
        byte += len(c.encode("utf-8"))
        i += 1
    at.append(byte)
    return "".join(out), at


def norm(s):
    """`s` folded for matching, without the offsets."""
    return norm_map(s)[0].strip()


def locate(text, passage):
    """The byte range of `passage` in `text` under the fold, or None."""
    folded, at = norm_map(text)
    p = norm(passage)
    i = folded.find(p) if p else -1
    return None if i < 0 else (at[i], at[i + len(p) - 1] + 1)


@functools.cache
def doc_text(path):
    """A document's text: a PDF by pdftotext, else macOS PDFKit (both give a form feed between pages);
    any other file as UTF-8."""
    if not path.lower().endswith(".pdf"):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    if shutil.which("pdftotext"):
        cmd = ["pdftotext", "-enc", "UTF-8", path, "-"]
    elif shutil.which("osascript"):
        cmd = ["osascript", "-l", "JavaScript", "-e", PDFKIT, path]
    else:
        raise SystemExit(f"{path}: no PDF text tool; install poppler (pdftotext)")
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=120).stdout


def page_texts(text):
    """The pages of `text`, cut at form feeds."""
    return text.split("\f")


def sentence_at(text, start, end):
    """The sentence of `text` around bytes `start`..`end`, for a gold passage filled from an anchor."""
    b = text.encode("utf-8")
    left = max(b.rfind(x, 0, start) for x in (b". ", b".\n", b"\n\n"))
    right = [i for i in (b.find(x, end) for x in (b". ", b".\n", b"\n\n")) if i >= 0]
    return b[left + 2 if left >= 0 else 0:(min(right) + 1) if right else len(b)].decode("utf-8", "replace").strip()


def check_gold(questions, docs, cache, fill):
    """Problems with each gold passage against the cached documents; with `fill`, a gold entry that
    has only a `find` anchor gets the anchor's sentence and page written into it."""
    problems, texts = [], {}
    for q in questions:
        for g in q.get("gold", []):
            d = docs[g["doc"]]
            path = os.path.join(cache, d["file"]) if d.get("file") else None
            if not path or not os.path.exists(path):
                problems.append(f"{q['id']}: {g['doc']} is not in the cache")
                continue
            if path not in texts:
                texts[path] = doc_text(path)
            pages = page_texts(texts[path])
            if g.get("text") is None:
                if not fill:
                    problems.append(f"{q['id']}: gold to fill from anchor {g.get('find')!r}")
                    continue
                for n, page in enumerate(pages, 1):
                    span = locate(page, g["find"])
                    if span:
                        g["text"], g["page"] = sentence_at(page, *span), n
                        break
                else:
                    problems.append(f"{q['id']}: anchor {g['find']!r} not in {g['doc']}")
                continue
            on = [n for n, page in enumerate(pages, 1) if norm(g["text"]) in norm_map(page)[0]]
            if not on:
                whole = norm(g["text"]) in norm_map(texts[path])[0]
                problems.append(f"{q['id']}: gold not in {g['doc']}" + (" (it spans a page break)" if whole else ""))
            elif g.get("page") not in on:
                problems.append(f"{q['id']}: gold on page {on}, not page {g.get('page')}")
    return problems


def harvest(lines):
    """What score.py reads from the reader's trace lines, whatever their nesting: every range
    ({doc, start, end}), the summed quote-check counts ({cited, matched, ran}), and the statuses."""
    ranges, cited, matched, ran, statuses = [], 0, 0, False, []

    def walk(v):
        nonlocal cited, matched, ran
        if isinstance(v, dict):
            if isinstance(v.get("start"), int) and isinstance(v.get("end"), int):
                ranges.append((v.get("doc", 0), v["start"], v["end"]))
            if isinstance(v.get("cited"), int) and isinstance(v.get("matched"), int):
                cited, matched = cited + v["cited"], matched + v["matched"]
                ran = ran or bool(v.get("ran", v["cited"] > 0))
            if isinstance(v.get("status"), str):
                statuses.append(v["status"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(lines)
    return {"ranges": ranges, "cited": cited, "matched": matched, "ran": ran, "statuses": statuses}


def agent_texts(data_dir):
    """The documents' text as the agent extracted it: every text file, and every long string in a
    JSON file, under `<data>/agent/extract/` and `<data>/agent/docs/` (blobs are bytes, skipped)."""
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir or "", "agent", "extract", "**", "*"), recursive=True)
                       + glob.glob(os.path.join(data_dir or "", "agent", "docs", "*.json"))):
        if not os.path.isfile(path) or os.sep + "blobs" + os.sep in path:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        try:
            stack = [json.loads(raw)]
            while stack:
                v = stack.pop()
                if isinstance(v, str) and len(v) > 200:
                    out.append(v)
                elif isinstance(v, (list, dict)):
                    stack.extend(v.values() if isinstance(v, dict) else v)
        except json.JSONDecodeError:
            out.append(raw)
    return out


def recall(items, answer):
    """The share of `items` the answer names; an item is a list of patterns (regex, any one counts)."""
    if not items:
        return None
    hit = sum(any(re.search(p, answer or "", re.IGNORECASE) for p in item) for item in items)
    return hit / len(items)


def pct(xs, p):
    """Nearest-rank percentile of `xs`, None when empty."""
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, -(-len(xs) * p // 100) - 1)] if xs else None


def key(row):
    """The row's key in the judge pack and the verdicts."""
    return f"{row['mode']}/{row['order']}/{row['source']}/{row['id']}/{row.get('rep', 0)}"


def score_rows(results, questions, docs, cache, verdicts):
    """Each row's five scores and its timings."""
    by_id = {q["id"]: q for q in questions}
    local, agent = {}, {}
    out = []
    for row in results["rows"]:
        q = by_id[row["id"]]
        base = {"key": key(row), "id": row["id"], "kind": q["kind"], "mode": row["mode"], "order": row["order"],
                "source": row["source"], "expect": q["expect"], "skipped": row.get("skipped")}
        if row.get("skipped"):
            out.append(base)
            continue
        h = harvest(row.get("trace", []))
        if row["data_dir"] not in agent:
            agent[row["data_dir"]] = agent_texts(row["data_dir"])
        ext, ret, source = [], [], None
        for g in (g for g in q.get("gold", []) if g.get("text")):
            texts = [t for t in agent[row["data_dir"]] if norm(g["text"]) in norm_map(t)[0]]
            source = "agent" if agent[row["data_dir"]] else "local"
            if not texts and source == "local":
                d = docs[g["doc"]]
                path = os.path.join(cache, d["file"]) if d.get("file") else ""
                if os.path.exists(path):
                    local.setdefault(path, doc_text(path))
                    texts = [local[path]] if norm(g["text"]) in norm_map(local[path])[0] else []
            ext.append(bool(texts))
            index = q["docs"].index(g["doc"])
            span = locate(texts[0], g["text"]) if texts else None
            if h["ranges"] and span:
                need = (span[1] - span[0]) / 2
                ret.append(any(d == index and min(e, span[1]) - max(s, span[0]) >= need for d, s, e in h["ranges"]))
        v = verdicts.get(key(row), {})
        answered = bool((row.get("answer") or "").strip()) and not row.get("error")
        out.append({**base,
            "error": row.get("error"), "tools": [t["name"] for t in row.get("tools", [])],
            "extraction": sum(ext) / len(ext) if ext else None, "extraction_source": source,
            "retrieval": sum(ret) / len(ret) if ret else None,
            "citation": h["matched"] / h["cited"] if h["cited"] else None, "checked": h["ran"],
            "completeness": recall(q.get("items"), row.get("answer")),
            "said_not_found": bool(NOT_FOUND.search(row.get("answer") or "")),
            # A row that never answered is a miss with its reason, never dropped.
            "verdict": v.get("verdict") or (None if answered else "wrong"), "miss": v.get("miss"),
            "miss_reason": v.get("note") or (None if answered else row.get("error") or "no answer"),
            "first_answer_s": row.get("first_answer_s"), "total_s": row.get("total_s"), "answered": answered,
        })
    return out


def mean(xs):
    """The mean of the values that are not None, None when there are none."""
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(scored):
    """Per mode, order and source: the five scores, the not-found rate on unanswerable questions, and the
    first-answer and total times (median, p95), with the share of rows that ended with an answer."""
    groups = {}
    for s in scored:
        groups.setdefault((s["mode"], s["order"], s["source"]), []).append(s)
    out = []
    for (mode, order, source), every in sorted(groups.items()):
        rows = [r for r in every if not r["skipped"]]
        tools = {}
        for r in rows:
            for t in r["tools"]:
                tools[t] = tools.get(t, 0) + 1
        judged = [r for r in rows if r["verdict"]]
        unanswerable = [r for r in rows if r["expect"] in ("not_found", "correct_premise")]
        out.append({
            "mode": mode, "order": order, "source": source, "rows": len(rows), "skipped": len(every) - len(rows),
            "extraction": mean(r["extraction"] for r in rows), "retrieval": mean(r["retrieval"] for r in rows),
            "citation": mean(r["citation"] for r in rows), "completeness": mean(r["completeness"] for r in rows),
            "correct": mean((r["verdict"] == "right") + 0.5 * (r["verdict"] == "partly") for r in judged) if judged else None,
            "judged": len(judged),
            "not_found_said": mean(r["said_not_found"] for r in unanswerable if r["expect"] == "not_found"),
            "first_answer_p50": pct([r["first_answer_s"] for r in rows], 50), "first_answer_p95": pct([r["first_answer_s"] for r in rows], 95),
            "total_p50": pct([r["total_s"] for r in rows], 50), "total_p95": pct([r["total_s"] for r in rows], 95),
            "answered": mean(r["answered"] for r in rows),
            "tools_per_question": mean(len(r["tools"]) for r in rows), "tools": tools,
            # Routing: a follow-up the chat model answered with no call never reached the text.
            "no_tool": mean(not r["tools"] for r in rows),
            "to_read_papers": mean("read_papers" in r["tools"] for r in rows),
        })
    return out


def gold_check(scored):
    """Plan 10.2 item 4: a question right with only its gold passage and wrong in v2 is our flaw; wrong
    in both is a model limit. Only judged rows count."""
    gold = {(s["id"], s["source"]): s for s in scored if s["mode"] == "gold" and s.get("verdict")}
    out = []
    for s in scored:
        g = gold.get((s["id"], s["source"]))
        if s["mode"] == "v2" and s.get("verdict") not in (None, "right") and g:
            out.append({"key": s["key"], "gold_verdict": g["verdict"], "v2_verdict": s["verdict"],
                        "suggested": "our flaw" if g["verdict"] == "right" else "model limit"})
    return out


def fmt(v, pctage=True):
    """A cell: a share as a percentage, seconds with one decimal, or a dash."""
    if v is None:
        return "-"
    return f"{100 * v:.0f} %" if pctage else f"{v:.1f}"


def markdown(summary, flaws):
    """The scores as a Markdown table, then the gold-passage check."""
    lines = ["| mode | order | source | rows (skipped) | extraction | retrieval | citation | complete | correct (judged) | not found said | first word p50/p95 s | total p50/p95 s | answered | no tool | to read_papers | tools per question |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['mode']} | {s['order']} | {s['source']} | {s['rows']} ({s['skipped']}) | {fmt(s['extraction'])} | {fmt(s['retrieval'])} | "
                     f"{fmt(s['citation'])} | {fmt(s['completeness'])} | {fmt(s['correct'])} ({s['judged']}) | {fmt(s['not_found_said'])} | "
                     f"{fmt(s['first_answer_p50'], False)}/{fmt(s['first_answer_p95'], False)} | {fmt(s['total_p50'], False)}/{fmt(s['total_p95'], False)} | {fmt(s['answered'])} | {fmt(s['no_tool'])} | {fmt(s['to_read_papers'])} | "
                     f"{fmt(s['tools_per_question'], False)}: {', '.join(f'{k} {v}' for k, v in sorted(s['tools'].items(), key=lambda kv: -kv[1])) or '-'} |")
    if flaws:
        lines += ["", "Gold-passage check (v2 misses):", "", "| row | gold | v2 | suggested |", "|---|---|---|---|"]
        lines += [f"| {f['key']} | {f['gold_verdict']} | {f['v2_verdict']} | {f['suggested']} |" for f in flaws]
    return "\n".join(lines) + "\n"


def pack(results, questions, docs, out):
    """The judge's rows: the question with its turns, what the corrected rules expect, the gold
    passages and items, and the answer."""
    by_id = {q["id"]: q for q in questions}
    with open(out, "w", encoding="utf-8") as f:
        for row in (r for r in results["rows"] if not r.get("skipped")):
            q = by_id[row["id"]]
            f.write(json.dumps({
                "key": key(row), "kind": q["kind"], "documents": [docs[d]["title"] for d in q["docs"]],
                "turns": q["turns"], "question": q["question"], "expect": q["expect"], "expect_note": q.get("expect_note"),
                "gold": [{"doc": g["doc"], "page": g.get("page"), "text": g.get("text")} for g in q.get("gold", [])],
                "items": q.get("items"), "answer": row.get("answer"), "error": row.get("error"),
            }, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gold", "pack", "score"])
    ap.add_argument("results", nargs="?")
    ap.add_argument("--cache", default=os.path.join(HERE, "data", "cache"))
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.jsonl"))
    ap.add_argument("--fill", action="store_true", help="gold: write anchored gold passages into the questions file")
    ap.add_argument("--verdicts", help="score: the judge's verdicts, one JSON object per line: key, verdict, miss, note")
    ap.add_argument("--out")
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    questions = load_jsonl(args.questions)
    docs = {d["id"]: d for d in load_json(os.path.join(HERE, "docs.json"))["docs"]}
    if args.command == "gold":
        problems = check_gold(questions, docs, args.cache, args.fill)
        if args.fill:
            with open(args.questions, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
        print("\n".join(problems) or f"{sum(len(q.get('gold', [])) for q in questions)} gold passages on their pages")
        return 1 if problems else 0
    if not args.results:
        ap.error(f"{args.command} needs RESULTS")
    results = load_json(args.results)
    if args.command == "pack":
        pack(results, questions, docs, args.out or os.path.splitext(args.results)[0] + "-judge.jsonl")
        return 0
    verdicts = {v["key"]: v for v in load_jsonl(args.verdicts)} if args.verdicts else {}
    scored = score_rows(results, questions, docs, args.cache, verdicts)
    summary, flaws = summarize(scored), gold_check(scored)
    base = os.path.splitext(args.results)[0]
    with open(args.out or base + "-scores.json", "w", encoding="utf-8") as f:
        json.dump({"schema": "larch-docfollow-scores/1", "summary": summary, "gold_check": flaws, "rows": scored}, f, indent=1)
    with open(args.md or base + "-scores.md", "w", encoding="utf-8") as f:
        f.write(markdown(summary, flaws))
    print(markdown(summary, flaws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
