"""Offline research-agent bench: a local BM25 corpus, graders, citation checks and the scorer.

Tasks are answered from a frozen document set (`docs.jsonl`) searched through an FTS5 index, so a run
needs no network. The agent loop, the models and the CLI follow the scorer.
"""

import argparse
import datetime
import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request

# -I (isolated mode) drops the script's directory from sys.path; the directory is the repo's own
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import UA  # noqa: E402

# ponytail: answers past 12000 characters are unreadable, add an offset argument to fetch when a run's fetched docs show the answer past the cut
FETCH_CHARS = 12000

JUDGE_PROMPT = """You grade a response to a question against the gold answer.

Question: {question}

Gold answer: {gold}

Response:
{response}

Judge only the meaning of the response's final answer. Accept equivalent forms and small numeric rounding. Ignore the reasoning, the quotes and the citations. Say no if the final answer is missing, contradicts the gold answer or hedges between answers.

End with exactly one line: VERDICT: yes or VERDICT: no
"""

RUBRIC_PROMPT = """You check whether a response to a question satisfies one rubric point.

Question: {question}

Rubric point: {point}

Response:
{response}

Judge only whether the response states what the point asks for, in any wording. Say no if the point is missing or contradicted.

End with exactly one line: VERDICT: yes or VERDICT: no
"""

JUDGE_SHA256 = "be0718e0733fe577aa81d1e4145e241c8ae30069366e3b24dcd8c53848c60c39"


class Corpus:
    """FTS5 index over a docs.jsonl file, rebuilt when the file's size or mtime changes."""

    def __init__(self, docs_path, index_path):
        st = os.stat(docs_path)
        source = "%d:%d" % (st.st_size, st.st_mtime_ns)
        self.db = sqlite3.connect(index_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        row = self.db.execute("SELECT value FROM meta WHERE key = 'source'").fetchone()
        if row is None or row[0] != source:
            self._build(docs_path, source)

    def _build(self, docs_path, source):
        db = self.db
        db.execute("DELETE FROM meta")  # no source row until the last insert commits
        db.execute("DROP TABLE IF EXISTS docs")
        db.execute("DROP TABLE IF EXISTS ids")
        db.execute(
            "CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, url UNINDEXED, title, text, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        # docid is UNINDEXED, so a lookup by it would scan every document; ids maps it to the rowid
        db.execute("CREATE TABLE ids(docid TEXT PRIMARY KEY, n INTEGER)")
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    cur = db.execute(
                        "INSERT INTO docs(docid, url, title, text) VALUES (?, ?, ?, ?)",
                        (d["docid"], d["url"], d["title"], d["text"]),
                    )
                    db.execute("INSERT INTO ids VALUES (?, ?)", (d["docid"], cur.lastrowid))
        db.execute("INSERT INTO meta VALUES ('source', ?)", (source,))
        db.commit()

    def search(self, query, k=5):
        """Top-k documents for the query's word tokens, best BM25 first."""
        q = fts_query(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT docid, title, url, snippet(docs, 3, '', '', '...', 32) FROM docs "
            "WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (q, k),
        )
        return [{"docid": r[0], "title": r[1], "url": r[2], "snippet": r[3]} for r in rows]

    def fetch(self, docid):
        """The document with its text cut to FETCH_CHARS, or None for an unknown docid."""
        row = self._row(docid, "docid, title, url, text")
        if row is None:
            return None
        text = row[3]
        if len(text) > FETCH_CHARS:
            text = text[:FETCH_CHARS] + "\n[cut at %d of %d characters]" % (FETCH_CHARS, len(text))
        return {"docid": row[0], "title": row[1], "url": row[2], "text": text}

    def text(self, docid):
        """The full text of a document, or None for an unknown docid."""
        row = self._row(docid, "text")
        return None if row is None else row[0]

    def _row(self, docid, columns):
        return self.db.execute(
            "SELECT %s FROM docs WHERE rowid = (SELECT n FROM ids WHERE docid = ?)" % columns, (docid,)
        ).fetchone()

    def close(self):
        """Closes the index."""
        self.db.close()


def fts_query(query):
    """OR of the query's word tokens, each quoted so FTS5 syntax in the text is inert."""
    return " OR ".join('"%s"' % t for t in re.findall(r"\w+", query))


def normalize(s):
    """Lowercase, accent-free, punctuation-free form of s without the articles a, an, the; numbers stay whole (1,000 is 1000, 3.5 is one word)."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"(?<=\d),(?=\d)", "", s.casefold())
    return " ".join(w for w in re.findall(r"\d+(?:\.\d+)*|[^\W_]+", s) if w not in ("a", "an", "the"))


def answer_line(response):
    """The first non-empty line after the last `Answer:` marker (markdown `*`, `_`, `#`, `>` around the marker are ignored), else the last non-empty line, without any [doc:ID] citation.

    One line only: text after the answer ("I hope this helps", a second candidate) must not change the grade,
    and the gold answer in the reasoning must not pass the contains grader.
    """
    last = None
    for m in re.finditer(r"^[\s*_#>]*(?:final\s+)?answer[\s*_#>]*:", response, re.I | re.M):
        last = m
    lines = [line for line in (response if last is None else response[last.end():]).splitlines() if line.strip()]
    if not lines:
        return ""
    # a [doc:ID] after the answer is a citation, not part of it
    return re.sub(r"\[doc:[^\]]*\]", "", lines[-1] if last is None else lines[0]).strip(" \t\r*_")


def grade_exact(response, gold):
    """True when the normalized answer equals the normalized gold."""
    return normalize(answer_line(response)) == normalize(gold)


def grade_contains(response, gold):
    """True when the normalized gold occurs as whole words in the normalized answer."""
    g = normalize(gold)
    return bool(g) and " " + g + " " in " " + normalize(answer_line(response)) + " "


def parse_verdict(text):
    """The last `VERDICT: yes|no` in text as a bool, or None when there is none."""
    found = re.findall(r"VERDICT:\s*(yes|no)", text, re.I)
    return None if not found else found[-1].lower() == "yes"


class Judge:
    """Asks a model to grade answers and rubric points; the prompts are pinned by JUDGE_SHA256."""

    def __init__(self, model):
        self.model = model

    def _ask(self, prompt):
        reply = self.model.chat([{"role": "user", "content": prompt}], tools=None)
        verdict = parse_verdict(reply["message"]["content"] or "")
        if verdict is None:
            raise RuntimeError("judge gave no verdict")
        return verdict

    def answer(self, question, gold, response):
        """Whether the response's final answer matches the gold answer."""
        return self._ask(JUDGE_PROMPT.format(question=question, gold=gold, response=response))

    def point(self, question, point, response):
        """Whether the response satisfies one rubric point."""
        return self._ask(RUBRIC_PROMPT.format(question=question, point=point, response=response))


def task_grader(task, default):
    """How a task is graded: "rubric", else the task's own `grader` key (fake-exact needs exact), else the default."""
    return "rubric" if "rubric" in task else task.get("grader", default)


def grade(task, response, grader, judge):
    """Grades a response (None = no answer) as {"passed", "points"}; points only for rubric tasks."""
    if "rubric" in task:
        if judge is None:
            raise ValueError("rubric tasks need a judge")
        points = [response is not None and judge.point(task["question"], p, response) for p in task["rubric"]]
        return {"passed": all(points), "points": points}
    grader = task_grader(task, grader)
    if grader == "judge" and judge is None:
        raise ValueError("the judge grader needs a judge")
    if grader not in ("exact", "contains", "judge"):
        raise ValueError("unknown grader %r" % grader)
    if response is None:
        return {"passed": False, "points": None}
    if grader == "judge":
        passed = judge.answer(task["question"], task["answer"], response)
    else:
        passed = (grade_exact if grader == "exact" else grade_contains)(response, task["answer"])
    return {"passed": passed, "points": None}


# anchored on the citation: pairing bare quote marks goes wrong at the first stray one (an inch mark)
CITED = re.compile(r'(?:"([^"\n]+)"|“([^”\n]+)”)\s*\[doc:([^\]\s]+)\]')


def quotes(response):
    """(span, docid) for every double-quoted run of 3+ words on one line that is followed at once by [doc:ID]."""
    return [(m.group(1) or m.group(2), m.group(3)) for m in CITED.finditer(response)
            if len((m.group(1) or m.group(2)).split()) >= 3]


def _squash(s):
    s = s.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
    return " ".join(s.casefold().split())


def citations(response, corpus):
    """(found, total): cited spans whose cited document really contains them, and all cited spans."""
    found = 0
    spans = quotes(response)
    for span, docid in spans:
        text = corpus.text(docid)
        found += text is not None and _squash(span) in _squash(text)
    return found, len(spans)


# a report's citation markers: [3], [1, 4]
MARKERS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def saved_page(data_dir, url, sha256):
    """The text of the copy larch saved of the page asked for at url whose body hashed to sha256 (`fetch::saved_at`), or None."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    try:
        with open(os.path.join(data_dir, "agent", "web", key, sha256 + ".json"), encoding="utf-8") as f:
            return json.load(f).get("text")
    except (OSError, ValueError):
        return None


def report_checks(report, steps, data_dir):
    """The hard gates on a research report: sources whose URL the run never fetched (invented), sources whose quote their saved page lacks, and markers in the sections that name no source."""
    fetched = {u for s in steps if s.get("step") == "fetch" for u in (s.get("detail") or {}).get("urls", [])}
    sources = report["sources"]
    missing = 0
    for e in sources:
        text = saved_page(data_dir, e["url"], e["sha256"])
        missing += text is None or _squash(e["cited"]["quote"]) not in _squash(text)
    numbers = {e["n"] for e in sources}
    dangling = sorted({n for sec in report["sections"] for m in MARKERS.finditer(sec["text"])
                       for n in map(int, m.group(1).split(",")) if n not in numbers})
    return {"invented_urls": [e["url"] for e in sources if e["url"] not in fetched],
            "quotes_missing": missing, "dangling_markers": dangling}


def pass_k(wins, k):
    """Mean over tasks of C(c,k)/C(n,k) for (c passes, n runs) per task; None if a task has fewer than k runs."""
    if not wins or any(n < k for _, n in wins):
        return None
    return sum(math.comb(c, k) / math.comb(n, k) for c, n in wins) / len(wins)


def summarize(rows, k=4):
    """Aggregates per-run rows into pass@1, pass^k, citation accuracy and cost totals."""
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(bool(r["passed"]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def stat(key):
        vals = [r[key] for r in rows]
        return {"total": sum(vals), "mean": mean(vals)}

    wins = [(sum(v), len(v)) for v in by_task.values()]
    total = sum(r["quotes_total"] for r in rows)
    # partial credit: a rubric run's share of yes points shows a gain all-or-nothing passes hide
    shares = [sum(r["points"]) / len(r["points"]) for r in rows if r.get("points")]
    checks = [r["checks"] for r in rows if r.get("checks")]
    stopped = {}
    for r in rows:
        if r.get("stopped"):
            stopped[r["stopped"]] = stopped.get(r["stopped"], 0) + 1
    return {
        "tasks": len(by_task),
        "runs": len(rows),
        "pass@1": mean([c / n for c, n in wins]),
        "pass^%d" % k: pass_k(wins, k),
        "citation_accuracy": sum(r["quotes_found"] for r in rows) / total if total else None,
        "tool_calls": stat("tool_calls"),
        "pages": stat("pages"),
        "wall_s": stat("wall_s"),
        "tokens": {
            "prompt": sum(r["prompt_tokens"] for r in rows),
            "completion": sum(r["completion_tokens"] for r in rows),
        },
        "errors": sum(1 for r in rows if r["error"]),
        "points_share": mean(shares),
        "checks": {
            "invented_urls": sum(len(c["invented_urls"]) for c in checks),
            "quotes_missing": sum(c["quotes_missing"] for c in checks),
            "dangling_markers": sum(len(c["dangling_markers"]) for c in checks),
        } if checks else None,
        "stopped": stopped or None,
        # R6: sentences with a number or a date and no source, fixed by their writer or cut
        "unsourced": {"fixed": sum(r["unsourced_fixed"] for r in rows),
                      "removed": sum(r["unsourced_removed"] for r in rows)} if checks else None,
        # --live: the facts each report kept, and the runs that came back inside the budget
        "facts_kept": stat("facts_kept") if rows and all("facts_kept" in r for r in rows) else None,
        "in_time": sum(1 for r in rows if r["in_time"]) if rows and all("in_time" in r for r in rows) else None,
    }


TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Full-text search of the document set. Returns the best matches with a snippet each.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "fetch",
        "description": "Read one document by its doc_id.",
        "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
    }},
]

SYSTEM = """You answer a question by researching a fixed set of documents with two tools: search (words to look for) and fetch (read one document by doc_id). Search as often as you need, read the documents that matter and check them against each other before you answer. Use only doc ids that search returned; never invent one.

Support the answer with short quotes copied exactly from the documents you read. Put each quote in double quotes and follow it at once with the document it came from, like "an exact phrase from the text" [doc:<docid>].

End your reply with one last line: Answer: <short answer>"""

MAX_TURNS = 24
# ponytail: a long-thinking model on a slow box can need more than 30 min; raise it when a GPU run shows correct runs cut at the cap
RUN_SECONDS = 1800

# what a failing server or judge raises: the run fails, the bench goes on
# (a ValueError is a bug or a config error, so it is not one of them: sources raise RuntimeError)
MODEL_ERRORS = (OSError, RuntimeError, http.client.HTTPException)


def call_tool(corpus, name, arguments_json):
    """Runs one tool call: (JSON result, the docid fetched or None); a bad call answers {"error": ...}."""
    def err(msg):
        return json.dumps({"error": msg}), None

    try:
        args = json.loads(arguments_json)
    except (ValueError, TypeError):
        return err("arguments are not valid JSON")
    if not isinstance(args, dict):
        return err("arguments must be a JSON object")
    if name == "search":
        if not isinstance(args.get("query"), str):
            return err("search needs a string query")
        return json.dumps(corpus.search(args["query"])), None
    if name == "fetch":
        if not isinstance(args.get("doc_id"), str):
            return err("fetch needs a string doc_id")
        doc = corpus.fetch(args["doc_id"])
        if doc is None:
            return err("unknown doc_id %r" % args["doc_id"])
        return json.dumps(doc), doc["docid"]
    return err("unknown tool %r" % name)


def run_task(model, corpus, task):
    """One agent run: the answer (None when it gave none), the cost counters and any error."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["question"]}]
    out = {"answer": None, "tool_calls": 0, "pages": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "wall_s": 0.0, "error": None}
    pages = set()
    start = time.monotonic()
    for _ in range(MAX_TURNS):
        if time.monotonic() - start > RUN_SECONDS:
            out["error"] = "over %d s" % RUN_SECONDS
            break
        try:
            reply = model.chat(messages, tools=TOOLS)
        except MODEL_ERRORS as exc:
            out["error"] = str(exc)
            break
        usage = reply.get("usage") or {}
        out["prompt_tokens"] += usage.get("prompt_tokens") or 0
        out["completion_tokens"] += usage.get("completion_tokens") or 0
        msg = reply["message"]
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            out["answer"] = msg.get("content") or ""
            break
        if not isinstance(calls, list):
            out["error"] = "the reply's tool_calls is not a list"
            break
        for tc in calls:
            # a server may leave any part out: the call still counts and is answered with an error
            tc = tc if isinstance(tc, dict) else {}
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            out["tool_calls"] += 1
            result, fetched = call_tool(corpus, fn.get("name"), fn.get("arguments"))
            if fetched is not None:
                pages.add(fetched)
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "", "content": result})
    else:
        out["error"] = "no answer after %d turns" % MAX_TURNS
    out["pages"] = len(pages)
    out["wall_s"] = time.monotonic() - start
    return out


def run_agent_task(agent, model, data_dir, task):
    """One run of larch's own research skill (`<agent> run research --json`) at the task's effort, on the live web: the report as the answer, the cost counters from its audit steps, why it stopped, its verification counts and its code checks."""
    effort = task.get("effort", "standard")
    out = {"answer": None, "tool_calls": 0, "pages": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "wall_s": 0.0, "error": None, "stopped": None, "checks": None, "effort": effort,
           "unsourced_fixed": 0, "unsourced_removed": 0, "facts_kept": 0}
    cmd = agent + ["run", "research", "-m", model, "--data-dir", data_dir, "--json", "--effort", effort,
                   "--question", task["question"]]
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=RUN_SECONDS)
    except subprocess.TimeoutExpired:
        out["error"] = "over %d s" % RUN_SECONDS
        return out
    finally:
        out["wall_s"] = time.monotonic() - start
    if p.returncode != 0:
        out["error"] = "exit %d: %s" % (p.returncode, (p.stderr.strip().splitlines() or [""])[-1])
        return out
    # stderr holds one audit step per JSON line, among the engine's other lines
    steps = []
    for line in p.stderr.splitlines():
        try:
            step = json.loads(line)
        except ValueError:
            continue
        if isinstance(step, dict):
            steps.append(step)
    try:
        result = json.loads(p.stdout)
        report = result["report"]
        checks = report_checks(report, steps, data_dir)
        # each source is one quote: found when its saved page has it
        total = len(report["sources"])
        out.update(answer=result["markdown"], stopped=report["stopped"], checks=checks,
                   quotes_total=total, quotes_found=total - checks["quotes_missing"],
                   facts_kept=total - checks["quotes_missing"],
                   unsourced_fixed=report.get("fixed", 0), unsourced_removed=report.get("removed", 0))
    except (ValueError, KeyError, TypeError) as exc:
        out["error"] = "the report is not larch's JSON: %r" % exc
        return out
    # the model's load is its own step line: the run's time is the research's, not the load's
    out["load_s"] = sum(s.get("wall_ms") or 0 for s in steps if s.get("step") == "load") / 1000
    out["wall_s"] = max(0.0, out["wall_s"] - out["load_s"])
    detail = [(s.get("step"), s.get("detail") or {}) for s in steps]
    out["tool_calls"] = sum(len(d.get("queries", [])) for name, d in detail if name == "search")
    out["pages"] = sum(d.get("pages", 0) for name, d in detail if name == "fetch")
    out["prompt_tokens"] = sum(s.get("prompt_tokens") or 0 for s in steps)
    out["completion_tokens"] = sum(s.get("generated_tokens") or 0 for s in steps)
    return out


class OpenAIModel:
    """A chat model behind an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, url, name, timeout=600, temperature=0, seed=None):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.name = name
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

    def chat(self, messages, tools=None):
        """One completion: {"message": the reply message, "usage": the token counts}."""
        body = {"model": self.name, "messages": messages, "temperature": self.temperature, "max_tokens": 2048}
        if self.seed is not None:
            body["seed"] = self.seed
        if tools:
            body["tools"] = tools
        headers = {"Content-Type": "application/json", "User-Agent": UA}
        if os.environ.get("LARCH_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LARCH_API_KEY"]
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                reply = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError("HTTP %d: %s" % (exc.code, exc.read()[:300].decode("utf-8", "replace")))
        except ValueError as exc:  # a server that answers with something else than JSON
            raise RuntimeError("the reply is not JSON: %s" % exc)
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None
        if not isinstance(message, dict):
            raise RuntimeError("the reply has no completion")
        return {"message": message, "usage": reply.get("usage") or {}}


def _words(*texts):
    return sum(len((t or "").split()) for t in texts)


class FakeModel:
    """Scripted for --fake: search, fetch the first hit, then answer; tokens are word counts."""

    def __init__(self, task):
        self.script = task["fake"]
        self.turn = 0

    def chat(self, messages, tools=None):
        """The next scripted reply."""
        if self.turn == 0:
            msg = self._call("search", {"query": self.script["search"]})
        elif self.turn == 1:
            hits = json.loads(messages[-1]["content"])
            self.fetched = hits[0]["docid"] if hits else ""
            msg = self._call("fetch", {"doc_id": self.fetched})
        else:
            # the gate's point: the search must rank the doc the scripted answer cites first
            if "[doc:%s]" % self.fetched not in self.script["answer"]:
                raise RuntimeError("search ranked %r first, the answer cites another" % self.fetched)
            msg = {"role": "assistant", "content": self.script["answer"]}
        self.turn += 1
        used = msg["content"] or msg["tool_calls"][0]["function"]["arguments"]
        return {"message": msg, "usage": {"prompt_tokens": _words(*[m.get("content") for m in messages]),
                                          "completion_tokens": _words(used)}}

    def _call(self, name, args):
        call = {"id": "call-%d" % self.turn, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        return {"role": "assistant", "content": None, "tool_calls": [call]}


class FakeJudge:
    """Approves everything, so --fake exercises the judge path offline."""

    def chat(self, messages, tools=None):
        """Always a yes verdict."""
        return {"message": {"role": "assistant", "content": "VERDICT: yes"}, "usage": {}}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fake_failures(rows, summary):
    bad = ["%s run %d: %s" % (r["task"], r["run"], r["error"] or "not passed") for r in rows
           if r["error"] or not r["passed"]]
    if summary["citation_accuracy"] != 1.0:
        bad.append("citation_accuracy %s" % summary["citation_accuracy"])
    bad += ["%s run %d: %d tool calls, %d pages" % (r["task"], r["run"], r["tool_calls"], r["pages"])
            for r in rows if (r["tool_calls"], r["pages"]) != (2, 1)]
    return bad


def sha256_file(path):
    """Hex sha256 of a file's bytes, read in 1 MiB chunks (a BrowseComp-Plus corpus is gigabytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(path, report, summarize_rows=None):
    """Sets the report's summary from its rows so far (by summarize_rows, default this bench's summarize) and writes it to path (if any); replaced whole, so a crash leaves no torn file."""
    report["summary"] = (summarize_rows or summarize)(report["rows"])
    if path:
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        os.replace(path + ".part", path)


def main(argv=None):
    """Runs the bench; returns the exit code."""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["browsecomp-plus", "frames", "simpleqa", "iris"])
    ap.add_argument("--fake", action="store_true", help="run the fixtures against a scripted model")
    ap.add_argument("--url")
    ap.add_argument("--model", help="the served model's name; with --agent, the agent's -m")
    ap.add_argument("--agent", help="runs larch's own research skill: the larch-agent binary, such as target/release/larch-agent")
    ap.add_argument("--agent-data", help="with --agent, the agent's data folder: its config.toml, and where it saves the pages the checks read")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--grader", choices=["contains", "exact", "judge"], default="contains")
    ap.add_argument("--temperature", type=float, default=0.7, help="the agent's; pass^4 needs sampling (the judge stays at 0)")
    ap.add_argument("--judge-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--out")
    ap.add_argument("--live", action="store_true", help="with --agent: a run passes only inside --budget seconds, and each run's wall clock, pages read and facts kept are printed")
    ap.add_argument("--budget", type=float, default=300, help="with --live: the seconds a report may take and still pass")
    args = ap.parse_args(argv)
    if args.fake == bool(args.set):
        ap.error("give --set or --fake, not both and not neither")
    if args.out and os.path.exists(args.out):  # a rerun after a crash must not overwrite the saved report
        print("%s exists" % args.out, file=sys.stderr)
        return 2
    if args.live and not args.agent:
        print("--live runs larch's own research on the web: it needs --agent", file=sys.stderr)
        return 2
    if args.agent and not (args.set == "iris" and args.model and args.agent_data):
        # the skill writes a report, which only a rubric grades, from the live web, not a corpus
        print("--agent runs the iris set and needs --model and --agent-data", file=sys.stderr)
        return 2

    if args.fake:
        tasks_path = os.path.join(here, "fixtures", "tasks.jsonl")
        docs_path = os.path.join(here, "fixtures", "docs.jsonl")
    else:
        if not (args.agent or args.url) or not args.model:
            print("a real run needs --url (or --agent) and --model", file=sys.stderr)
            return 2
        folder = os.path.join(args.data, args.set)
        tasks_path = os.path.join(here, "iris.jsonl") if args.set == "iris" else os.path.join(folder, "tasks.jsonl")
        docs_path = os.path.join(folder, "docs.jsonl")
    if not os.path.exists(tasks_path):
        print("missing %s" % tasks_path, file=sys.stderr)
        return 2
    tasks = _read_jsonl(tasks_path)[:args.limit]

    if args.fake:
        judge = Judge(FakeJudge())
    elif args.judge_url and args.judge_model:
        judge = Judge(OpenAIModel(args.judge_url, args.judge_model))
    elif any(task_grader(t, args.grader) in ("judge", "rubric") for t in tasks):
        print("the judge grader and rubric tasks need --judge-url and --judge-model", file=sys.stderr)
        return 2
    else:
        judge = None
    # after the judge check, so a fresh checkout asks for the judge flags before the iris download
    if not args.agent and not os.path.exists(docs_path):
        print("missing %s" % docs_path, file=sys.stderr)
        return 2

    rows = []
    report = {
        "schema": "larch-research/1", "set": "fake" if args.fake else args.set,
        "model": "fake" if args.fake else args.model, "url": args.url, "grader": args.grader,
        "judge_model": "fake" if args.fake else args.judge_model, "judge_prompt_sha256": JUDGE_SHA256,
        "agent": args.agent, "live_budget_s": args.budget if args.live else None,
        # the agent samples, searches and prompts as its own config.toml and code say
        "runs": args.runs, "temperature": None if args.agent else args.temperature,
        "seeds": None if args.agent else "run index", "tasks_sha256": sha256_file(tasks_path),
        "docs_sha256": None if args.agent else sha256_file(docs_path),
        "system_prompt_sha256": None if args.agent else hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "summary": None, "rows": rows,
        "started_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    corpus = None if args.agent else Corpus(docs_path, ":memory:" if args.fake else os.path.join(folder, "index.db"))
    try:
        for task in tasks:
            for n in range(args.runs):
                if args.agent:
                    row = run_agent_task([args.agent], args.model, args.agent_data, task)
                else:
                    model = FakeModel(task) if args.fake else OpenAIModel(
                        args.url, args.model, temperature=args.temperature, seed=n)
                    row = run_task(model, corpus, task)
                try:
                    verdict = grade(task, row["answer"], args.grader, judge)
                except MODEL_ERRORS as exc:  # a judge that fails fails this run only
                    row["error"] = str(exc)
                    verdict = {"passed": False, "points": None}
                if args.agent:
                    found, total = row.pop("quotes_found", 0), row.pop("quotes_total", 0)
                else:
                    found, total = (0, 0) if row["answer"] is None else citations(row["answer"], corpus)
                if args.live:
                    # facts right in a bounded time: a report after the budget does not pass, however many pages it read
                    row["in_time"] = row["wall_s"] <= args.budget
                    verdict["passed"] = verdict["passed"] and row["in_time"]
                rows.append(dict(row, task=task["id"], run=n, grader=task_grader(task, args.grader),
                                 passed=verdict["passed"], points=verdict["points"], quotes_found=found, quotes_total=total))
            if args.out:
                _save(args.out, report)
    finally:
        if corpus:
            corpus.close()

    _save(args.out, report)
    summary = report["summary"]
    if args.live:
        print("%-12s %3s %8s %5s %10s %6s" % ("task", "run", "wall_s", "pages", "facts_kept", "passed"))
        for r in rows:
            print("%-12s %3d %8.1f %5d %10d %6s" % (r["task"], r["run"], r["wall_s"], r["pages"], r["facts_kept"], r["passed"]))
    print(json.dumps(summary, indent=1, ensure_ascii=True))
    if rows and all(r["error"] for r in rows):
        print("every run ended with an error", file=sys.stderr)
        return 1
    # the hard gates: zero invented URLs, zero quotes their page lacks, zero markers without a source
    broken = ["%s run %d: %s" % (r["task"], r["run"], r["checks"]) for r in rows
              if r.get("checks") and any(r["checks"].values())]
    if broken:
        print("code checks FAILED\n" + "\n".join(broken), file=sys.stderr)
        return 1
    if args.fake:
        bad = _fake_failures(rows, summary)
        if bad:
            print("fake: FAILED\n" + "\n".join(bad), file=sys.stderr)
            return 1
        print("fake: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
