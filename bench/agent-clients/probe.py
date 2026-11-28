#!/usr/bin/env python3
"""API capability probe for agent clients: OpenAI chat/Responses and Anthropic Messages (stdlib only).

python3 probe.py --url http://127.0.0.1:8080 --engine larch --out probe-larch.json
Every check records ok / fail / error with a short detail; nothing is skipped silently.
"""
import argparse, base64, json, time, urllib.request, urllib.error

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Local time in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
A_WEATHER = {"name": "get_weather", "description": "Current weather for a city",
             "input_schema": WEATHER["function"]["parameters"]}
# 1x1 png
PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cf"
    "c0f01f0005000201e2b1a5f50000000049454e44ae426082")).decode()


def call(url, path, body=None, method="POST", stream=False, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method, headers={
        "Content-Type": "application/json", "Authorization": "Bearer x", "x-api-key": "x",
        "anthropic-version": "2023-06-01"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode("utf-8", "replace"), e.code
    dt = time.perf_counter() - t0
    if stream:
        events = []
        for line in raw.splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except ValueError:
                    pass
        return status, events, raw, dt
    try:
        return status, json.loads(raw), raw, dt
    except ValueError:
        return status, None, raw, dt


def chat(url, model, msgs, **kw):
    return call(url, "/v1/chat/completions", {"model": model, "messages": msgs, "temperature": 0, **kw},
                stream=kw.get("stream", False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--model", default="local")
    p.add_argument("--max-tokens", type=int, default=2048, help="room for a thinking model to finish")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    U, M, N = a.url, a.model, a.max_tokens
    res = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = None, f"{type(e).__name__}: {e}"
        res.append({"check": name, "ok": ok, "detail": str(detail)[:400]})
        print(f"{a.engine:10} {name:34} {'OK' if ok else ('ERROR' if ok is None else 'FAIL')}  {str(detail)[:160]}",
              flush=True)

    ask = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    both = [{"role": "user", "content": "Get the weather in Paris and in Tokyo. Call the tool once per city, "
             "both calls in this one reply."}]

    def models():
        s, d, raw, _ = call(U, "/v1/models", method="GET")
        return s == 200 and bool(d and d.get("data")), raw[:200]
    check("openai.models", models)

    def basic():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "Say hello in one word."}], max_tokens=N)
        m = d["choices"][0]["message"] if s == 200 else {}
        return s == 200 and bool(m.get("content")), f"{dt:.1f}s usage={d.get('usage') if d else None} " \
            f"reasoning_field={'reasoning_content' in m or 'reasoning' in m} content={str(m.get('content'))[:80]!r}"
    check("openai.chat", basic)

    def stream_usage():
        s, ev, raw, dt = chat(U, M, [{"role": "user", "content": "Say hello."}], max_tokens=N, stream=True,
                              stream_options={"include_usage": True})
        usage = [e["usage"] for e in ev if e.get("usage")]
        reasoning = any((c.get("delta") or {}).get("reasoning_content") or (c.get("delta") or {}).get("reasoning")
                        for e in ev for c in e.get("choices") or [])
        return s == 200 and bool(usage), f"usage={usage[-1] if usage else None} streams_reasoning={reasoning}"
    check("openai.stream+usage", stream_usage)

    def tools_auto():
        s, d, raw, dt = chat(U, M, ask, tools=[WEATHER], max_tokens=N)
        c = d["choices"][0] if s == 200 else {}
        calls = (c.get("message") or {}).get("tool_calls") or []
        args = json.loads(calls[0]["function"]["arguments"]) if calls else None
        return bool(calls) and c.get("finish_reason") == "tool_calls" and "city" in (args or {}), \
            f"{dt:.1f}s finish={c.get('finish_reason')} calls={calls[:1]} raw={raw[:200] if not calls else ''}"
    check("openai.tools.auto", tools_auto)

    def tools_stream():
        s, ev, raw, dt = chat(U, M, ask, tools=[WEATHER], max_tokens=N, stream=True)
        tc = [t for e in ev for c in e.get("choices") or [] for t in (c.get("delta") or {}).get("tool_calls") or []]
        fin = [c.get("finish_reason") for e in ev for c in e.get("choices") or [] if c.get("finish_reason")]
        argtext = "".join((t.get("function") or {}).get("arguments") or "" for t in tc)
        try:
            json.loads(argtext)
            good = True
        except ValueError:
            good = False
        return bool(tc) and good, f"chunks_with_calls={len(tc)} finish={fin} args={argtext[:80]!r}"
    check("openai.tools.stream", tools_stream)

    def parallel():
        s, d, raw, dt = chat(U, M, both, tools=[WEATHER], max_tokens=N)
        calls = ((d or {}).get("choices", [{}])[0].get("message") or {}).get("tool_calls") or []
        cities = sorted(json.loads(c["function"]["arguments"]).get("city", "") for c in calls)
        ids = len({c.get("id") for c in calls}) == len(calls)
        return len(calls) >= 2 and ids, f"{len(calls)} calls {cities} unique_ids={ids}"
    check("openai.tools.parallel", parallel)

    def required():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "Hi there"}], tools=[WEATHER, TIME],
                             tool_choice="required", max_tokens=N)
        calls = ((d or {}).get("choices", [{}])[0].get("message") or {}).get("tool_calls") or []
        return bool(calls), f"s={s} calls={calls[:1]} {raw[:150] if not calls else ''}"
    check("openai.tool_choice.required", required)

    def named():
        s, d, raw, dt = chat(U, M, ask, tools=[WEATHER, TIME], max_tokens=N,
                             tool_choice={"type": "function", "function": {"name": "get_time"}})
        calls = ((d or {}).get("choices", [{}])[0].get("message") or {}).get("tool_calls") or []
        return bool(calls) and calls[0]["function"]["name"] == "get_time", f"s={s} calls={calls[:1]}"
    check("openai.tool_choice.named", named)

    def tool_result():
        msgs = ask + [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                      {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 17, \"sky\": \"rain\"}"}]
        s, d, raw, dt = chat(U, M, msgs, tools=[WEATHER], max_tokens=N)
        txt = ((d or {}).get("choices", [{}])[0].get("message") or {}).get("content") or ""
        return s == 200 and "17" in txt, f"{txt[:120]!r}"
    check("openai.tool_result_turn", tool_result)

    def schema():
        fmt = {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": {
            "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"], "additionalProperties": False}}}
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "Invent a person. Reply as JSON."}],
                             response_format=fmt, max_tokens=N)
        txt = ((d or {}).get("choices", [{}])[0].get("message") or {}).get("content") or ""
        o = json.loads(txt)
        return set(o) == {"name", "age"} and isinstance(o["age"], int), txt[:120]
    check("openai.json_schema", schema)

    def length():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "Count from 1 to 500."}], max_tokens=5)
        c = (d or {}).get("choices", [{}])[0]
        return c.get("finish_reason") == "length", f"finish={c.get('finish_reason')} usage={(d or {}).get('usage')}"
    check("openai.finish_length", length)

    def think_off():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "What is 17*23? Answer only the number."}],
                             max_tokens=N, chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none")
        s2, d2, raw2, dt2 = chat(U, M, [{"role": "user", "content": "What is 17*23? Answer only the number."}],
                                 max_tokens=N)
        u1, u2 = (d or {}).get("usage") or {}, (d2 or {}).get("usage") or {}
        return s == 200 and u1.get("completion_tokens", 1e9) < 20, \
            f"off: {u1.get('completion_tokens')} tok {dt:.1f}s / default: {u2.get('completion_tokens')} tok {dt2:.1f}s"
    check("openai.thinking_off", think_off)

    def image():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": [{"type": "text", "text": "What colour is it?"},
                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]}],
                             max_tokens=64)
        return s == 200, f"s={s} {raw[:150]}"
    check("openai.image_part", image)

    def too_long():
        s, d, raw, dt = chat(U, M, [{"role": "user", "content": "word " * 300000}], max_tokens=8)
        return 400 <= s < 500, f"s={s} {dt:.1f}s {raw[:200]}"
    check("openai.too_long_is_4xx", too_long)

    def responses():
        s, d, raw, dt = call(U, "/v1/responses", {"model": M, "input": "Say hello.", "max_output_tokens": N})
        return s == 200 and bool(d and d.get("output")), f"s={s} {raw[:150]}"
    check("openai.responses", responses)

    def responses_tools():
        s, d, raw, dt = call(U, "/v1/responses", {"model": M, "input": "What is the weather in Paris?",
                             "max_output_tokens": N, "tools": [{"type": "function", "name": "get_weather",
                             "parameters": WEATHER["function"]["parameters"]}]})
        calls = [o for o in (d or {}).get("output") or [] if o.get("type") == "function_call"]
        return bool(calls), f"s={s} {raw[:150] if not calls else calls[0]}"
    check("openai.responses.tools", responses_tools)

    def am(body, stream=False):
        return call(U, "/v1/messages", {"model": M, "max_tokens": N, **body}, stream=stream)

    def a_basic():
        s, d, raw, dt = am({"messages": [{"role": "user", "content": "Say hello."}]})
        kinds = [b.get("type") for b in (d or {}).get("content") or []]
        return s == 200 and "text" in kinds, f"{dt:.1f}s kinds={kinds} usage={(d or {}).get('usage')} " \
            f"stop={(d or {}).get('stop_reason')}"
    check("anthropic.messages", a_basic)

    def a_tools_stream():
        s, ev, raw, dt = am({"messages": ask, "tools": [A_WEATHER], "stream": True}, stream=True)
        types = [e.get("type") for e in ev]
        starts = [e["content_block"] for e in ev if e.get("type") == "content_block_start"]
        uses = [b for b in starts if b.get("type") == "tool_use"]
        stop = [e["delta"].get("stop_reason") for e in ev if e.get("type") == "message_delta"]
        return bool(uses) and stop == ["tool_use"] and types[:1] == ["message_start"], \
            f"blocks={[b.get('type') for b in starts]} stop={stop}"
    check("anthropic.tools.stream", a_tools_stream)

    def a_parallel():
        s, d, raw, dt = am({"messages": both, "tools": [A_WEATHER]})
        uses = [b for b in (d or {}).get("content") or [] if b.get("type") == "tool_use"]
        return len(uses) >= 2, f"{len(uses)} tool_use {[u.get('input') for u in uses]}"
    check("anthropic.tools.parallel", a_parallel)

    def a_count():
        s, d, raw, dt = call(U, "/v1/messages/count_tokens", {"model": M, "messages": ask, "tools": [A_WEATHER]})
        return s == 200 and (d or {}).get("input_tokens", 0) > 0, raw[:120]
    check("anthropic.count_tokens", a_count)

    def a_think_off():
        q = [{"role": "user", "content": "What is 17*23? Answer only the number."}]
        s, d, raw, dt = am({"messages": q, "thinking": {"type": "disabled"}})
        s2, d2, raw2, dt2 = am({"messages": q})
        o1 = ((d or {}).get("usage") or {}).get("output_tokens")
        o2 = ((d2 or {}).get("usage") or {}).get("output_tokens")
        return s == 200 and (o1 or 1e9) < 20, f"disabled: {o1} tok {dt:.1f}s / default: {o2} tok {dt2:.1f}s"
    check("anthropic.thinking_disabled", a_think_off)

    def a_think_blocks():
        s, d, raw, dt = am({"messages": [{"role": "user", "content": "Is 391 prime? Think, then answer."}],
                            "thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": N + 1024})
        kinds = [b.get("type") for b in (d or {}).get("content") or []]
        return "thinking" in kinds, f"kinds={kinds} usage={(d or {}).get('usage')}"
    check("anthropic.thinking_blocks", a_think_blocks)

    def a_too_long():
        s, d, raw, dt = am({"messages": [{"role": "user", "content": "word " * 300000}], "max_tokens": 8})
        return 400 <= s < 500 and "too long" in raw.lower(), f"s={s} {raw[:200]}"
    check("anthropic.too_long_is_4xx", a_too_long)

    json.dump({"engine": a.engine, "url": U, "results": res}, open(a.out, "w"), indent=1)
    ok = sum(1 for r in res if r["ok"])
    print(f"{a.engine}: {ok}/{len(res)} checks pass")


if __name__ == "__main__":
    main()
