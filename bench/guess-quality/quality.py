#!/usr/bin/env python3
"""Guess-check quality check: greedy outputs of larch arms, scored by llama.cpp's llama-logits.

  quality.py gen URL ARM PROMPTS OUT.jsonl [MAX_TOKENS]   one arm through larch serve's /v1/completions
  quality.py prep PROMPTS OUT_IN.jsonl OUT_KEYS.json ARM=FILE...   llama-logits input: each prompt, then prompt + each arm's text
  quality.py score OUT_KEYS.json DUMP.jsonl BASE ARM=FILE...   tables vs BASE (the drafter-off arm), markdown on stdout

Score, in llama.cpp's own tokens of each output (teacher forced on the true prefix):
  top-1: share of generated tokens that are llama.cpp's own greedy pick there;
  mean logprob: llama.cpp's mean log-probability of the generated tokens (a token outside its top 64
  takes the 64th logprob, an upper bound; counted as "outside top 64");
  same text as BASE; for each output that differs, at the first token where it differs from BASE
  (same prefix, so one llama.cpp distribution): each side's logprob and rank, llama.cpp's top-1/top-2 gap;
  math: the 'Answer: N' line against the known answer; speed: decode tok/s and time to first token.
Standard library only.
"""
import json
import re
import statistics
import sys
import time
import urllib.request


def render(content):
    # Qwen chat template, one user turn, thinking closed (as the earlier drafter checks ran).
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gen(url, arm, prompts, out, max_tokens=320):
    rows = []
    for p in load(prompts):
        prompt = render(p["content"])
        body = {
            "model": "larch",
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        t0 = time.time()
        first = last = None
        text, usage = [], None
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                raw = raw.strip()
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for c in ev.get("choices") or []:
                    if c.get("text"):
                        now = time.time()
                        first = first or now
                        last = now
                        text.append(c["text"])
        n = usage["completion_tokens"] if usage else None
        decode = (n - 1) / (last - first) if n and n > 1 and last and last > first else None
        row = {
            "id": p["id"],
            "kind": p["kind"],
            "answer": p.get("answer"),
            "arm": arm,
            "prompt": prompt,
            "text": "".join(text),
            "tokens": n,
            "ttft_s": (first - t0) if first else None,
            "decode_tok_s": decode,
        }
        rows.append(row)
        print(f"{arm} {p['id']}: {n} tokens, {decode or 0:.1f} tok/s", file=sys.stderr)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def prep(prompts, out_in, out_keys, arms):
    outs = {a: {r["id"]: r for r in load(f)} for a, f in arms}
    lines, keys = [], []
    for p in load(prompts):
        lines.append({"prompt": render(p["content"])})
        keys.append([p["id"], None])
        for a, _ in arms:
            r = outs[a][p["id"]]
            lines.append({"prompt": r["prompt"] + r["text"]})
            keys.append([p["id"], a])
    with open(out_in, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    with open(out_keys, "w") as f:
        json.dump(keys, f)


def lp_rank(dist, tok):
    for rank, (t, lp) in enumerate(dist):
        if t == tok:
            return lp, rank + 1
    return dist[-1][1], None


def answer_of(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def score(keys_path, dump_path, base, arms):
    keys = json.load(open(keys_path))
    dump = load(dump_path)
    header, lines = dump[0], dump[1:]
    if len(lines) != len(keys):
        sys.exit(f"dump has {len(lines)} prompts, keys {len(keys)}")
    got = {}
    for k, line in zip(keys, lines):
        got[(k[0], k[1])] = (line["tokens"], line["top64"])
    outs = {a: {r["id"]: r for r in load(f)} for a, f in arms}
    ids = [k[0] for k in keys if k[1] is None]
    print(f"llama.cpp {header.get('llama_cpp_tag', '?')} ({header.get('llama_cpp_commit', '?')}), "
          f"backend {header.get('backend', '?')}, device {header.get('device', '?')}, ngl {header.get('n_gpu_layers', '?')}")
    print()
    print("| arm | prompts | gen tokens (llama.cpp) | top-1 = llama.cpp | mean logprob | outside top 64 | same text as " + base + " | math right |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    div_rows, speed = [], []
    for a, _ in arms:
        agree = total = outside = boundary = same = math_ok = math_n = 0
        lps = []
        for i in ids:
            ptoks = got[(i, None)][0]
            toks, top = got[(i, a)]
            b = len(ptoks)
            if toks[:b] != ptoks:
                boundary += 1
                b = next((j for j, (x, y) in enumerate(zip(toks, ptoks)) if x != y), min(len(toks), b))
            for j in range(max(b, 1), len(toks)):
                lp, rank = lp_rank(top[j - 1], toks[j])
                total += 1
                agree += rank == 1
                outside += rank is None
                lps.append(lp)
            r = outs[a][i]
            same += r["text"] == outs[base][i]["text"]
            if r.get("answer") is not None:
                math_n += 1
                got_a = answer_of(r["text"])
                math_ok += got_a is not None and abs(got_a - float(r["answer"])) < 1e-6
            if a != base and r["text"] != outs[base][i]["text"]:
                btoks, btop = got[(i, base)]
                j = next((j for j, (x, y) in enumerate(zip(btoks, toks)) if x != y), None)
                if j is None or j == 0:
                    div_rows.append((a, i, None))
                else:
                    dist = btop[j - 1]
                    lb, rb = lp_rank(dist, btoks[j])
                    la, ra = lp_rank(dist, toks[j])
                    gap = dist[0][1] - dist[1][1] if len(dist) > 1 else None
                    div_rows.append((a, i, (j - len(ptoks), lb, rb, la, ra, gap)))
            speed.append((a, r["kind"], r["decode_tok_s"], r["ttft_s"]))
        top1 = agree / total if total else 0
        mean = statistics.fmean(lps) if lps else 0
        note = f" ({boundary} boundary mismatches)" if boundary else ""
        print(f"| {a} | {len(ids)} | {total} | {top1:.4f} | {mean:.4f} | {outside} | {same}/{len(ids)} | {math_ok}/{math_n} |{note}")
    print()
    print(f"First token that differs from {base} (in llama.cpp tokens; same prefix, one distribution):")
    print()
    print(f"| arm | prompt | at generated token | {base} logprob (rank) | arm logprob (rank) | llama.cpp top-1/top-2 gap |")
    print("|---|---|---:|---:|---:|---:|")
    for a, i, d in div_rows:
        if d is None:
            print(f"| {a} | {i} | ? | | | |")
            continue
        at, lb, rb, la, ra, gap = d
        print(f"| {a} | {i} | {at} | {lb:.3f} ({rb or '>64'}) | {la:.3f} ({ra or '>64'}) | {gap:.3f} |")
    print()
    print("| arm | kind | prompts | decode tok/s (mean) | TTFT s (p50) |")
    print("|---|---|---:|---:|---:|")
    kinds = sorted({k for _, k, _, _ in speed})
    for a, _ in arms:
        for k in kinds:
            s = [d for x, kk, d, _ in speed if x == a and kk == k and d]
            t = [tt for x, kk, _, tt in speed if x == a and kk == k and tt]
            if s:
                print(f"| {a} | {k} | {len(s)} | {statistics.fmean(s):.1f} | {statistics.median(t):.2f} |")


def arms_of(args):
    return [tuple(a.split("=", 1)) for a in args]


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "gen":
        gen(*rest)
    elif cmd == "prep":
        prep(rest[0], rest[1], rest[2], arms_of(rest[3:]))
    elif cmd == "score":
        score(rest[0], rest[1], rest[2], arms_of(rest[3:]))
    else:
        sys.exit(__doc__)
