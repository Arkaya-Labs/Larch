#!/usr/bin/env python3
"""Exact-answer recall over long documents, for comparing KV cache page formats (f16 against q8_0).

Each document is filler text of about --tokens tokens with three records whose IDs differ in one
character, placed early (5%), in the middle (50%) and late (92%). Two questions per document ask for
two of the records' IDs, the second on the same prefix (a prefix-cache hit). An answer passes when
the expected ID is the only record ID it contains. Documents, IDs and questions come from --seed
alone, so two servers (one per KV format) get the same requests.

  python3 scripts/kv_exact_answer.py --url http://127.0.0.1:8080 --tokens 8000 --out f16-8k.json

Start the server with LARCH_KV_DTYPE=q8_0 for the q8_0 arm. Prints a summary line and writes every
question's result to --out.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request

WORDS = (
    "the a of to and in that is for on with as by at from this it be are was an which or have has "
    "not but they their more one all been would will can there about when also into other these some "
    "than its only time new year people first could after report market system water program world "
    "city state number group service company river council board summer winter season project local "
    "public office record member team study plan region school field paper result period area model "
    "value change level policy history county station bridge village harbor valley garden library"
).split()
NAMES = ["Harlan Voss", "Harlan Vos", "Harlen Voss"]


def filler(rng: random.Random, words: int) -> list[str]:
    out = []
    while len(out) < words:
        n = rng.randint(8, 18)
        s = [rng.choice(WORDS) for _ in range(n)]
        s[0] = s[0].capitalize()
        out.extend(s[:-1] + [s[-1] + "."])
    return out[:words]


def document(rng: random.Random, tokens: int):
    words = int(tokens / 1.35)
    body = filler(rng, words)
    base = rng.randint(1000000, 9999999)
    swapped = str(base)[:-2] + str(base)[-1] + str(base)[-2]
    if swapped == str(base):
        swapped = str(base)[:-1] + str((base + 1) % 10)
    ids = [f"KX-{base}-A", f"KX-{base}-B", f"KX-{swapped}-A"]
    order = list(range(3))
    rng.shuffle(order)  # which near-identical record sits early, middle, late
    records = []
    for slot, at in zip(order, (0.05, 0.50, 0.92)):
        records.append((NAMES[slot], ids[slot], at))
    for name, rid, at in sorted(records, key=lambda r: -r[2]):  # insert from the end, offsets stay valid
        body.insert(int(at * len(body)), f"Record: the account ID of {name} is {rid}.")
    return " ".join(body), records, ids


def ask(url: str, model: str, doc: str, name: str, max_tokens: int):
    content = (
        f"{doc}\n\nQuestion: what is the account ID of {name}, exactly as written in the text above? "
        "Answer with the ID only. /no_think"
    )
    req = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    r = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        body = json.load(resp)
    return body, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="", help="the name /v1/models lists (any, for one model)")
    p.add_argument("--tokens", type=int, default=8000)
    p.add_argument("--docs", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = random.Random(a.seed * 1000003 + a.tokens)
    results, passed = [], 0
    for d in range(a.docs):
        doc, records, ids = document(rng, a.tokens)
        for q in range(2):
            name, rid, at = records[(d + q) % 3]  # every position asked across the documents
            body, secs = ask(a.url, a.model, doc, name, a.max_tokens)
            text = body["choices"][0]["message"].get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            found = set(re.findall(r"KX-\d{7}-[AB]", text))
            ok = found == {rid}
            passed += ok
            usage = body.get("usage", {})
            results.append({
                "doc": d, "question": q, "position": at, "expected": rid, "found": sorted(found),
                "pass": ok, "answer": text.strip()[:200], "seconds": round(secs, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            })
            print(f"doc {d} q{q} at {at:.2f}: {'PASS' if ok else 'FAIL'} {sorted(found)} "
                  f"want {rid} ({secs:.1f} s)", file=sys.stderr)
    with open(a.out, "w") as f:
        json.dump({"tokens": a.tokens, "docs": a.docs, "seed": a.seed, "passed": passed,
                   "total": len(results), "results": results}, f, indent=1)
    print(f"{passed}/{len(results)} exact answers at ~{a.tokens} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
