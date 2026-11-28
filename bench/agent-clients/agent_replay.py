#!/usr/bin/env python3
"""Scripted coding-agent replay against any OpenAI-compatible server (stdlib only).

Each agent starts from a real OpenCode request (system prompt + 10 tools, ~9k tokens), with its own task
line, then runs --turns turns. A turn sends the whole conversation (stateless, as every agent client
does), times it, then appends a FIXED assistant tool call and a FIXED ~1.5k-token tool result, so every
engine sees exactly the same prompts whatever the model answered. This measures what an agent loop
costs per turn: prefill of the new part, prefix reuse, first token, decode.

Modes:
  --max-tokens 1      prefill/reuse cost only (thinking does not matter)
  --max-tokens 512    a real turn, model thinks or not as the server does by default
Use --extra '{"chat_template_kwargs":{"enable_thinking":false}}' (llama-server, mlx) or
--extra '{"reasoning_effort":"none"}' (Ollama) to ask a server to skip thinking.

Writes one JSON line per request to --out and prints a summary.
"""
import argparse, json, random, sys, threading, time, urllib.request

WORDS = ("fn let mut self impl struct enum match Some None Ok Err return pub use mod crate where "
         "async await loop for while if else true false Vec String usize u32 Option Result Box Arc "
         "Mutex clone iter map collect unwrap expect into from len push insert get key value").split()


def tool_result(agent, turn, n_words):
    r = random.Random(agent * 1000 + turn)
    lines, line = [], []
    for _ in range(n_words):
        line.append(r.choice(WORDS))
        if len(line) >= 12:
            lines.append(" ".join(line)); line = []
    body = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))
    return f"<path>/home/user/project/src/file_{agent}_{turn}.rs</path>\n<type>file</type>\n<content>\n{body}\n</content>"


def post(url, body, stream, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.perf_counter()
    first = first_answer = None
    usage, finish, text, calls, reasoning = None, None, "", 0, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            d = json.loads(r.read())
            c = d["choices"][0]
            first = first_answer = time.perf_counter() - t0
            m = c["message"]
            text = m.get("content") or ""
            calls = len(m.get("tool_calls") or [])
            reasoning = len(m.get("reasoning_content") or m.get("reasoning") or "")
            return dict(first_s=first, first_answer_s=first_answer, e2e_s=time.perf_counter() - t0,
                        usage=d.get("usage"), finish=c.get("finish_reason"), text_chars=len(text),
                        tool_calls=calls, reasoning_chars=reasoning)
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                now = time.perf_counter() - t0
                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if rc:
                    reasoning += len(rc)
                    first = first or now
                if delta.get("content"):
                    text += delta["content"]
                    first = first or now
                    first_answer = first_answer or now
                if delta.get("tool_calls"):
                    calls = max(calls, max(tc.get("index", 0) for tc in delta["tool_calls"]) + 1)
                    first = first or now
                    first_answer = first_answer or now
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    return dict(first_s=first, first_answer_s=first_answer, e2e_s=time.perf_counter() - t0, usage=usage,
                finish=finish, text_chars=len(text), tool_calls=calls, reasoning_chars=reasoning)


def agent(i, a, base, out, lock, start):
    msgs = [dict(m) for m in base["messages"][:2]]
    msgs[1]["content"] = (f"Task {i}: the build in crate_{i} fails. Read the files you need, find the bug and fix it. "
                          + msgs[1]["content"])
    start.wait()
    for t in range(a.turns):
        body = {"model": a.model, "messages": msgs, "tools": base["tools"], "tool_choice": "auto",
                "max_tokens": a.max_tokens, "temperature": 0, "stream": a.stream}
        if a.stream:
            body["stream_options"] = {"include_usage": True}
        body.update(json.loads(a.extra))
        rec = dict(engine=a.engine, agent=i, turn=t, t_start=time.time())
        try:
            rec.update(post(a.url + "/v1/chat/completions", body, a.stream, a.timeout))
        except Exception as e:  # recorded, not hidden
            rec["error"] = f"{type(e).__name__}: {getattr(e, 'read', lambda: b'')()[:300]!r} {e}"
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
            u = rec.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            print(f"{a.engine} a{i} t{t} prompt={u.get('prompt_tokens')} cached={cached} "
                  f"gen={u.get('completion_tokens')} first={rec.get('first_s')} e2e={rec.get('e2e_s', 0):.2f} "
                  f"calls={rec.get('tool_calls')} finish={rec.get('finish')} err={rec.get('error','')[:120]}",
                  flush=True)
        cid = f"call_{i}_{t}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
                     "function": {"name": "read", "arguments": json.dumps({"filePath": f"/home/user/project/src/file_{i}_{t}.rs"})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(i, t, a.result_words)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--model", default="local")
    p.add_argument("--fixture", default="crates/larch/tests/serve-opencode.json")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--result-words", type=int, default=1100, help="~1.5k tokens per tool result")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--extra", default="{}")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    base = json.load(open(a.fixture))
    lock, start = threading.Lock(), threading.Barrier(a.agents)
    t0 = time.perf_counter()
    with open(a.out, "a") as out:
        ts = [threading.Thread(target=agent, args=(i, a, base, out, lock, start)) for i in range(a.agents)]
        [t.start() for t in ts]; [t.join() for t in ts]
    print(f"{a.engine}: {a.agents} agents x {a.turns} turns wall {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
