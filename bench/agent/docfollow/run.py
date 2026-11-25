"""Drives the follow-up questions through the agent CLI and the plain chat, and records each answer.

  run.py fetch [--cache DIR]                      downloads docs.json's documents and checks their hashes
  run.py panel [--seed S] [--size N] [--write]    draws a balanced panel; --write records it in panel.json
  run.py run --agent BIN --plain BIN -m GGUF --context N --thinking-budget N --out DIR [options]
  run.py dry-run [--out DIR]                      the whole path on fake CLIs, then score.py on its results

Modes: today (`larch-agent chat`, setting off), v2 (`doc_followups = "on"`), whole (`larch chat` with the
whole document as its system message, where it fits) and gold (the same with only the gold passages, plan 10.2
item 4). Orders: cold (each question reopens its chat's saved summary, so nothing is kept in the engine), warm
(one live chat per document set, questions in turn) and interleaved (the chats over several documents, questions
switching document each time).
"""

import argparse
import codecs
import glob
import hashlib
import json
import os
import platform
import random
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score

MODES = ("today", "v2", "whole", "gold")
ORDERS = ("cold", "warm", "interleaved")
STATS = re.compile(r" tok/s · TTFT |engine usage/timings unavailable")
# Lines the CLIs write around the answer: tool calls and results, notices, progress, a write's question.
PROTOCOL = ("-> ", "<- ", "note: ", "... ", "allow ", "(dropped ")
TRACE_GLOBS = ("agent/log/*.jsonl",)  # the audit log; the reader's lines carry skill "doc"
QUIET_S = 3.0  # a prompt with no stats line before it (an error) ends the turn after this much silence


def load_docs():
    """docs.json's documents by id."""
    return {d["id"]: d for d in score.load_json(os.path.join(HERE, "docs.json"))["docs"]}


def sha256(path):
    """A file's SHA-256, hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(docs, cache):
    """Downloads each document with a URL into `cache` (once), copies the repository's own, and returns
    one line per document that is missing or whose hash differs from docs.json's."""
    os.makedirs(cache, exist_ok=True)
    problems = []
    for d in docs.values():
        if not d.get("file"):
            continue
        out = os.path.join(cache, d["file"])
        if d.get("path"):
            shutil.copyfile(os.path.join(HERE, d["path"]), out)
            continue
        if not os.path.exists(out) and d.get("url"):
            req = urllib.request.Request(d["url"], headers={"User-Agent": "larch-docfollow-bench"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(out + ".part", "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(out + ".part", out)
            except OSError as e:
                problems.append(f"{d['id']}: {d['url']}: {e}")
                continue
        if not os.path.exists(out):
            problems.append(f"{d['id']}: not in the cache; {d['checked']}")
        elif d.get("sha256") and sha256(out) != d["sha256"]:
            problems.append(f"{d['id']}: sha256 {sha256(out)} is not docs.json's {d['sha256']}")
        elif not d.get("sha256"):
            problems.append(f"{d['id']}: unpinned, got sha256 {sha256(out)}; record it in docs.json")
    return problems


def pick(questions, size, seed):
    """`size` question ids balanced across kinds and documents: kinds in turn, each giving its next
    question whose document set has fewer than its share, from a shuffle seeded by `seed`."""
    rng = random.Random(seed)
    by_kind = {}
    for q in sorted(questions, key=lambda q: q["id"]):
        by_kind.setdefault(q["kind"], []).append(q)
    for qs in by_kind.values():
        rng.shuffle(qs)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    cap = -(-size // len({tuple(q["docs"]) for q in questions})) + 1
    picked, per_docs = [], {}
    while len(picked) < size and any(by_kind.values()):
        for k in kinds:
            for i, q in enumerate(by_kind[k]):
                if per_docs.get(tuple(q["docs"]), 0) < cap:
                    picked.append(by_kind[k].pop(i)["id"])
                    per_docs[tuple(q["docs"])] = per_docs.get(tuple(q["docs"]), 0) + 1
                    break
            else:
                by_kind[k] = []
            if len(picked) == size:
                break
    return sorted(picked)


class Repl:
    """One running chat CLI: a prompt per line in, and per turn the output with each character's time."""

    def __init__(self, cmd, env=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.dec = {fd: codecs.getincrementaldecoder("utf-8")("replace") for fd in (self.p.stdout.fileno(), self.p.stderr.fileno())}
        self.err = ""

    def _read(self, until, timeout):
        """Output (text, per-character times) until `until(text, quiet_s)` holds, the process ends, or `timeout`."""
        text, times, start, last = "", [], time.monotonic(), time.monotonic()
        out, err = self.p.stdout.fileno(), self.p.stderr.fileno()
        while True:
            now = time.monotonic()
            if until(text, now - last):
                return text, times, None
            if now - start > timeout:
                return text, times, f"timeout after {timeout:.0f} s"
            ready, _, _ = select.select([out, err], [], [], 0.05)
            for fd in ready:
                chunk = self.dec[fd].decode(os.read(fd, 65536))
                if fd == err:
                    self.err += chunk
                elif chunk:
                    text, last = text + chunk, time.monotonic()
                    times += [last] * len(chunk)
            if not ready and self.p.poll() is not None:
                return text, times, f"exited {self.p.returncode}: {self.err.strip()[-300:]}"

    def start(self, timeout):
        """Waits for the first prompt; the error when the CLI never gave one."""
        return self._read(lambda t, q: t.endswith("> "), timeout)[2]

    def turn(self, line, timeout):
        """Sends `line` and returns the turn: its answer, tool calls, notes, the times from the send to the
        first visible word and to the end, and an error when it never ended."""
        t0 = time.monotonic()
        self.p.stdin.write((line + "\n").encode("utf-8"))
        self.p.stdin.flush()
        text, times, confirms = "", [], 0

        def ended(t, quiet):
            body = t[:-2].rstrip("\n").rsplit("\n", 1)[-1]
            return (t.endswith("\n> ") or t == "> ") and (STATS.search(body) or body.startswith(("(dropped", "note: ")) or quiet > QUIET_S)

        while True:
            more, more_times, error = self._read(lambda t, q, seen=text: ended(seen + t, q) or t.endswith("? [y/n] "), timeout - (time.monotonic() - t0))
            text, times = text + more, times + more_times
            if error or not more.endswith("? [y/n] "):
                break
            # Every write is declined: the bench reads, it never changes a file.
            confirms += 1
            self.p.stdin.write(b"n\n")
            self.p.stdin.flush()
        return parse_turn(text, times, t0, error, confirms)

    def close(self, kill=False):
        """Ends the CLI with /quit, or kills it: one stopped mid-answer would not read the line."""
        try:
            if not kill:
                self.p.stdin.write(b"/quit\n")
                self.p.stdin.close()
                self.p.wait(30)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.p.kill()
        self.p.wait()


def parse_turn(text, times, t0, error, confirms):
    """A turn's output split into the answer and the CLI's own lines, with its times in seconds."""
    if text.endswith("> ") and not error:
        text, times = text[:-2], times[:-2]
    tools, notes, answer, first_text, first_answer, stats, total = [], [], [], None, None, None, None
    # The answer is the last text after a tool line: a reader streams it before its "<- " line.
    fresh = True
    segments, at = [], 0
    for line in text.split("\n"):
        # A write's question ends without a newline: what follows it on that line is the turn's own text.
        if line.startswith("allow ") and "? [y/n] " in line:
            cut = line.index("? [y/n] ") + 8
            segments += [(at, line[:cut]), (at + cut, line[cut:])]
        else:
            segments.append((at, line))
        at += len(line) + 1
    for start, line in segments:
        if line.startswith("-> "):
            name, _, args = line[3:].partition(" ")
            tools.append({"name": name, "args": args})
            fresh = True
        elif line.startswith("<- "):
            fresh = True
        elif line.startswith("note: "):
            notes.append(line[6:])
        elif STATS.search(line):
            stats, total = line, times[start] - t0 if start < len(times) else None
        elif not line.startswith(PROTOCOL):
            answer.append(line)
            word = re.search(r"\w", line)
            if word and start + word.start() < len(times):
                t = times[start + word.start()] - t0
                first_text = t if first_text is None else first_text
                first_answer, fresh = (t, False) if fresh else (first_answer, fresh)
    if total is None and times and not error:
        total = times[-1] - t0
    return {"answer": "\n".join(answer).strip(), "tools": tools, "notes": notes, "stats": stats, "confirms": confirms,
            "first_text_s": first_text, "first_answer_s": first_answer, "total_s": total, "error": error}


class Trace:
    """The reader's trace lines added to the data folder since the last mark."""

    def __init__(self, data_dir, globs):
        self.data_dir, self.globs, self.sizes = data_dir, globs, {}
        self.mark()

    def files(self):
        """The trace files that exist now."""
        return sorted({p for g in self.globs for p in glob.glob(os.path.join(self.data_dir, g), recursive=True)})

    def mark(self):
        """Remembers how long each trace file is."""
        self.sizes = {p: os.path.getsize(p) for p in self.files()}

    def new(self):
        """The JSON lines written since the mark, and how many lines did not parse."""
        lines, bad = [], 0
        for p in self.files():
            with open(p, "rb") as f:
                f.seek(self.sizes.get(p, 0))
                for raw in f.read().decode("utf-8", "replace").splitlines():
                    try:
                        line = json.loads(raw) if raw.strip() else None
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    if isinstance(line, dict) and line.get("skill") == "doc":
                        lines.append(line)
        self.mark()
        return lines, bad


def fill(turn, docs, files, source):
    """`turn` with each `{doc}` replaced by the document's file path, link, or (plain modes) file name."""
    def one(m):
        d = docs[m.group(1)]
        if source == "name":
            return d.get("file") or d["url"]
        if source == "link" or d.get("link_only"):
            return d["url"]
        return os.path.join(files, d["file"])
    return re.sub(r"\{(\w+)\}", one, turn)


def system_text(q, docs, cache, gold_only):
    """The plain model's system message: the documents' whole text with page markers, or only the gold
    passages with their pages."""
    parts = []
    for i, doc_id in enumerate(q["docs"]):
        d = docs[doc_id]
        if gold_only:
            body = "\n\n".join(f"[page {g['page']}] {g['text']}" for g in q.get("gold", []) if g["doc"] == doc_id and g.get("text"))
        else:
            pages = score.page_texts(score.doc_text(os.path.join(cache, d["file"])))
            body = "\n".join(f"[page {n}]\n{p}" for n, p in enumerate(pages, 1) if p.strip() or n < len(pages))
        parts.append(f"=== Document {chr(65 + i)}: {d['file']} ===\n{body}")
    lead = "Passages from the documents the user attached follow." if gold_only else "The user attached these documents; their whole text follows."
    return lead + "\n\n" + "\n\n".join(parts)


def why_skip(q, docs, cache, mode, source, context, reserve):
    """Why the question cannot run in this mode and source, or None."""
    for doc_id in q["docs"]:
        d = docs[doc_id]
        if source == "link" and not d.get("url") and mode in ("today", "v2"):
            return f"{doc_id} has no link"
        if (source == "file" or mode in ("whole", "gold")) and not d.get("link_only") and not os.path.exists(os.path.join(cache, d.get("file") or "")):
            return f"{doc_id} is not in the cache"
        if mode in ("whole", "gold") and d.get("link_only"):
            return f"{doc_id} is only a link; the plain model cannot fetch it"
    if mode == "gold" and not any(g.get("text") for g in q.get("gold", [])):
        return "no gold passage"
    if mode == "whole" and len(system_text(q, docs, cache, False).encode("utf-8")) // 4 + reserve > context:
        return f"the documents do not fit {context} tokens with {reserve} kept for the turns and the answer"
    return None


def sessions(questions, order):
    """The order's sessions: (chat's opening line, document ids, whether the summary is reopened from a
    snapshot, questions in the order asked)."""
    chats = {}
    for q in questions:
        chats.setdefault(q["turns"][0], []).append(q)
    out = []
    for opening, qs in chats.items():
        if order == "cold":
            out += [(opening, qs[0]["docs"], True, [q]) for q in qs]
        elif order == "warm":
            out.append((opening, qs[0]["docs"], False, qs))
        elif len(qs[0]["docs"]) > 1:
            lanes = {}
            for q in qs:
                lanes.setdefault((q.get("gold") or [{"doc": q["docs"][0]}])[0]["doc"], []).append(q)
            mixed = []
            while any(lanes.values()):
                mixed += [lane.pop(0) for lane in lanes.values() if lane]
            out.append((opening, qs[0]["docs"], False, mixed))
    return out


class Runner:
    """One run's settings, its output folder and its rows."""

    def __init__(self, args, docs, questions):
        self.a, self.docs, self.questions = args, docs, questions
        self.rows, self.summaries = [], []
        self.agent_context = self.context_arg(args.agent, "--max-context", ["--max-context", str(args.context)])
        self.plain_thinking = self.context_arg(args.plain, "--thinking-budget", ["--thinking-budget", str(args.thinking_budget)])
        self.results = {
            "schema": "larch-docfollow/1", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "machine": platform.platform(), "commit": git_commit(),
            "config": {
                "modes": args.modes, "orders": args.orders, "source": args.source, "reps": args.reps,
                "model": os.path.basename(args.model), "context": args.context, "thinking_budget": args.thinking_budget,
                "answer_tokens": args.answer_tokens, "turn_timeout_s": args.turn_timeout,
                "agent": args.agent, "plain": args.plain, "agent_config": args.agent_config,
                "agent_context": "--set max_context" if self.agent_context else "8192: this agent CLI has no context flag",
                "plain_thinking": "--thinking-budget" if self.plain_thinking else
                f"this chat CLI has no thinking budget: --max-tokens {args.thinking_budget + args.answer_tokens} caps thinking and answer together",
                "set": args.set, "seed": args.seed, "questions": [q["id"] for q in questions],
            },
            "summaries": self.summaries, "rows": self.rows,
        }

    def context_arg(self, cmd, flag, value):
        """`value` when `cmd chat --help` lists `flag`, else [], so an older CLI still runs and the
        results say what it ran at."""
        try:
            help_text = subprocess.run(cmd + ["chat", "--help"], capture_output=True, text=True, timeout=60, check=False).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        return value if flag in help_text else []

    def save(self):
        """Rewrites the results and the summary, so a crash keeps every finished row."""
        with open(os.path.join(self.a.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=1, ensure_ascii=False)
        with open(os.path.join(self.a.out, "summary.md"), "w", encoding="utf-8") as f:
            f.write(summary_md(self.results))

    def command(self, mode, data_dir, files, chat_id, q):
        """The CLI command for a session."""
        if mode in ("today", "v2"):
            cmd = self.a.agent + ["chat", "-m", self.a.model, "--data-dir", data_dir] + self.agent_context
            cmd += ["--allow", files] if self.a.source == "file" else []
            return cmd + (["--chat", chat_id] if chat_id else [])
        cmd = self.a.plain + ["chat", "-m", self.a.model, "--data-dir", data_dir, "--set", f"max_context={self.a.context}",
                              "--system", system_text(q, self.docs, self.a.cache, mode == "gold"), "--resume", chat_id]
        if self.plain_thinking:
            return cmd + self.plain_thinking + ["--max-tokens", str(self.a.thinking_budget + self.a.answer_tokens)]
        return cmd + ["--max-tokens", str(self.a.thinking_budget + self.a.answer_tokens)]

    def prepare(self, mode, data_dir, docs):
        """A fresh data folder with the mode's config.toml, and a folder holding only the chat's files."""
        os.makedirs(os.path.join(data_dir, "agent"))
        files = os.path.join(data_dir + "-files")
        os.makedirs(files)
        for doc_id in docs:
            d = self.docs[doc_id]
            src = os.path.join(self.a.cache, d.get("file") or "")
            if d.get("file") and os.path.exists(src):
                # A link, not a copy: every session has its own folder, and the PDFs are megabytes.
                try:
                    os.link(src, os.path.join(files, d["file"]))
                except OSError:
                    shutil.copyfile(src, os.path.join(files, d["file"]))
        base = ""
        if self.a.agent_config:
            with open(self.a.agent_config, encoding="utf-8") as f:
                base = "".join(line for line in f if not line.startswith(("thinking_budget", "doc_followups")))
        with open(os.path.join(data_dir, "agent", "config.toml"), "w", encoding="utf-8") as f:
            f.write(base + f"\nthinking_budget = {self.a.thinking_budget}\n" + ('doc_followups = "on"\n' if mode == "v2" else ""))
        return files

    def open(self, mode, data_dir, files, chat_id, q):
        """Starts the CLI; the error when it gave no prompt."""
        repl = Repl(self.command(mode, data_dir, files, chat_id, q))
        # A cold load of the large model from disk takes minutes on its first run.
        error = repl.start(max(600, self.a.turn_timeout))
        if error:
            repl.close()
        return repl, error

    def summary(self, mode, order, rep, key, t):
        """Records a chat's opening turn."""
        self.summaries.append({"mode": mode, "order": order, "rep": rep, "chat": key, "answer": t["answer"],
                               "total_s": t["total_s"], "tools": [x["name"] for x in t["tools"]], "error": t["error"]})

    def run_session(self, mode, order, rep, n, opening, doc_ids, snapshot, qs):
        """One session: the chat's opening turn (or a copy of its saved one), then each question with the
        follow-ups it builds on. A turn that fails or passes the cap is a miss with its reason, and the
        session's later questions start again in a new one."""
        plain = mode in ("whole", "gold")
        root = os.path.join(self.a.out, "runs", f"{mode}-{order}-{rep}")
        key = re.sub(r"\W+", "-", "-".join(doc_ids))
        base, timeout = os.path.join(root, f"base-{key}"), self.a.turn_timeout
        source = "name" if plain else self.a.source
        new_id = f"docfollow-{key}" if plain else None
        if snapshot and base not in self.bases:
            # The opening turn runs once and is saved; every question reopens a copy, so its engine is cold.
            files = self.prepare(mode, base, doc_ids)
            repl, error = self.open(mode, base, files, new_id, qs[0])
            chat_id = new_id
            if not error:
                t = repl.turn(fill(opening, self.docs, files, source), timeout)
                self.summary(mode, order, rep, key, t)
                error, chat_id = t["error"], new_id or chat_id_of(repl.err)
                repl.close(kill=bool(error))
            self.bases[base] = (chat_id, error)
        data_dir = os.path.join(root, f"s{n:04d}")
        if snapshot:
            (chat_id, error), files, asked = self.bases[base], base + "-files", [opening]
            if not error:
                shutil.copytree(base, data_dir)
        else:
            files = self.prepare(mode, data_dir, doc_ids)
            chat_id, error, asked = new_id, None, [opening] if mode == "gold" else []
        repl = None
        if not error:
            repl, error = self.open(mode, data_dir, files, chat_id, qs[0])
            repl = None if error else repl
        if not error and not asked:
            t = repl.turn(fill(opening, self.docs, files, source), timeout)
            self.summary(mode, order, rep, key, t)
            error, asked = t["error"], [opening]
        trace = Trace(data_dir, self.a.trace_glob)
        for position, q in enumerate(qs):
            row = {"id": q["id"], "mode": mode, "order": order, "source": self.a.source, "rep": rep, "session": n,
                   "position": position, "data_dir": data_dir, "skipped": None, "history_exact": asked == q["turns"][:len(asked)]}
            if error:
                row.update({"answer": "", "error": f"the chat's opening: {error}", "tools": [], "trace": []})
                self.rows.append(row)
                continue
            failed = None
            for line in [t for t in q["turns"] if t not in asked]:
                t = repl.turn(fill(line, self.docs, files, source), timeout)
                asked.append(line)
                if t["error"]:
                    failed = f"a follow-up before the question: {t['error']}"
                    break
            if failed:
                row.update({"answer": "", "error": failed, "tools": [], "trace": []})
            else:
                trace.mark()
                t = repl.turn(q["question"], timeout)
                lines, bad = trace.new()
                row.update(t)
                row.update({"trace": lines, "trace_errors": bad})
                asked.append(q["question"])
                failed = t["error"]
            row["context_fitted"] = fitted(repl.err)
            self.rows.append(row)
            self.save()
            if failed:
                repl.close(kill=True)
                if position + 1 < len(qs):
                    self.run_session(mode, order, rep, n + 1000, opening, doc_ids, snapshot, qs[position + 1:])
                return
        if repl:
            repl.close(kill=bool(error))

    def run(self):
        """Every mode, order and repetition; a skipped question gets a row saying why. The gold mode runs
        each question alone and only cold, since its passages are that question's own."""
        os.makedirs(self.a.out, exist_ok=True)
        self.bases = {}
        reserve = self.a.thinking_budget + self.a.answer_tokens + 1024
        for rep in range(self.a.reps):
            for mode in self.a.modes:
                for order in self.a.orders:
                    if mode == "gold" and order != "cold":
                        continue
                    n = 0
                    plan = [(q["turns"][0], q["docs"], False, [q]) for q in self.questions] if mode == "gold" else sessions(self.questions, order)
                    for opening, doc_ids, snapshot, qs in plan:
                        ok = []
                        for q in qs:
                            why = why_skip(q, self.docs, self.a.cache, mode, self.a.source, self.a.context, reserve)
                            if why:
                                self.rows.append({"id": q["id"], "mode": mode, "order": order, "source": self.a.source,
                                                  "rep": rep, "skipped": why, "answer": None, "data_dir": None, "trace": []})
                            else:
                                ok.append(q)
                        if ok:
                            n += 1
                            self.run_session(mode, order, rep, n, opening, doc_ids, snapshot, ok)
                        self.save()
        self.results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()


def chat_id_of(err):
    """The chat's ID from the agent CLI's stderr ("chat <id>")."""
    m = re.search(r"^(?:chat|transcript) ([^\s;]+)", err, re.MULTILINE)
    return m.group(1) if m else None


def fitted(err):
    """The context the engine fitted when it says it is smaller than asked, else None."""
    m = re.search(r"with a (\d+)-token context", err)
    return int(m.group(1)) if m else None


def git_commit():
    """The checkout's commit, or None outside git."""
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False).stdout.strip() or None
    except OSError:
        return None


def summary_md(results):
    """Per mode and order: rows run, skipped and failed, the share answered, the first-word and total
    times (median, p95), and the tools the agent called per question."""
    c = results["config"]
    lines = [f"# docfollow run {results['started_at']}", "",
             (f"model {c['model']}, context {c['context']} (agent: {c['agent_context']}), thinking budget {c['thinking_budget']} "
              f"(plain: {c['plain_thinking']}), turn cap {c['turn_timeout_s']} s, source {c['source']}, set {c['set']} seed {c['seed']}"), "",
             "| mode | order | run | skipped | failed | answered | first word p50/p95 s | total p50/p95 s | tool calls per question | tools |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    groups = {}
    for r in results["rows"]:
        groups.setdefault((r["mode"], r["order"]), []).append(r)
    for (mode, order), rows in groups.items():
        ran = [r for r in rows if not r["skipped"]]
        names = {}
        for r in ran:
            for t in r.get("tools", []):
                names[t["name"]] = names.get(t["name"], 0) + 1
        f = lambda v: "-" if v is None else f"{v:.1f}"
        firsts, totals = [r.get("first_answer_s") for r in ran], [r.get("total_s") for r in ran]
        lines.append(f"| {mode} | {order} | {len(ran)} | {len(rows) - len(ran)} | {sum(bool(r.get('error')) for r in ran)} | "
                     f"{sum(bool(r.get('answer')) and not r.get('error') for r in ran)}/{len(ran)} | "
                     f"{f(score.pct(firsts, 50))}/{f(score.pct(firsts, 95))} | {f(score.pct(totals, 50))}/{f(score.pct(totals, 95))} | "
                     f"{f(sum(len(r.get('tools', [])) for r in ran) / len(ran) if ran else None)} | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(names.items(), key=lambda kv: -kv[1])) or '-'} |")
    failed = [r for r in results["rows"] if r.get("error")]
    if failed:
        lines += ["", "Failed rows (scored as misses):", ""] + [f"- {r['mode']}/{r['order']}/{r['id']}: {r['error']}" for r in failed]
    return "\n".join(lines) + "\n"


def fake_cli(argv):
    """A stand-in for both CLIs: `fake agent|plain chat ...`, answering from questions.jsonl's gold so the
    dry run exercises every path (snapshots, reopening, tool lines, a declined write, trace lines)."""
    kind, args = argv[0], argv[2:]
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    if "--help" in args:
        print("--set <KNOB=VALUE>" if kind == "plain" else "--allow <ALLOW>  --chat <CHAT>")
        return 0
    data = opt("--data-dir")
    questions = {q["question"]: q for q in score.load_jsonl(os.path.join(HERE, "questions.jsonl"))}
    docs = load_docs()
    chat = opt("--chat") or opt("--resume") or f"c{os.getpid()}"
    marker = os.path.join(data, "fake-chats", chat)
    if opt("--chat") and not os.path.exists(marker):
        print(f"error: no saved chat {chat}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "a"):
        pass
    v2 = False
    if kind == "agent":
        with open(os.path.join(data, "agent", "config.toml"), encoding="utf-8") as f:
            v2 = 'doc_followups = "on"' in f.read()
    print(f"chat {chat}" if kind == "agent" else f"transcript {chat}; /status, /quit", file=sys.stderr, flush=True)
    system = opt("--system") or ""
    while True:
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line or line.strip() == "/quit":
            return 0
        q = questions.get(line.strip())
        if kind == "plain" and q and q["id"] == "kimi-pipeline" and system.startswith("The user attached"):
            time.sleep(3600)  # the floor's unbounded thinking that never answers
        out = sys.stdout
        if kind == "agent":
            # A v2 follow-up's answer streams inside the call, before its "<- " line, as the reader's does.
            out.write('-> read_papers {"papers": ["x"]}\n' + ("" if v2 and q else "<- read_papers: 1 summary\n"))
            if q and q["kind"] == "false_premise":
                out.write("allow write_file notes.md? [y/n] ")
                out.flush()
                sys.stdin.readline()
        if q is None:
            out.write("A summary of the document.")
        elif kind == "agent" and not v2:
            out.write(q.get("expect_note") or "")
        else:
            golds = [g for g in q.get("gold", []) if g.get("text") and (kind == "agent" or g["text"] in system)]
            out.write(" ".join(f"\"{g['text']}\" [p. {g['page']}]" for g in golds) + " " + (q.get("expect_note") or ""))
            if kind == "agent":
                ranges = []
                for g in golds:
                    text = score.doc_text(os.path.join(HERE, "data", "cache", docs[g["doc"]]["file"]))
                    os.makedirs(os.path.join(data, "agent", "extract"), exist_ok=True)
                    with open(os.path.join(data, "agent", "extract", g["doc"] + ".txt"), "w", encoding="utf-8") as f:
                        f.write(text)
                    s, e = score.locate(text, g["text"])
                    ranges.append({"doc": q["docs"].index(g["doc"]), "start": s, "end": e})
                os.makedirs(os.path.join(data, "agent", "log"), exist_ok=True)
                with open(os.path.join(data, "agent", "log", time.strftime("%Y-%m-%d") + ".jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"skill": "doc", "question": q["question"], "first_read": ranges,
                                        "check": {"cited": len(golds), "matched": len(golds), "ran": bool(golds)}, "status": "answered"}) + "\n")
                out.write("\n<- read_papers: answered")
        out.flush()
        time.sleep(0.02)
        out.write("\n12.00 tok/s · TTFT 40.00 ms · 0 cached\n")
        out.flush()


def dry_run(out):
    """Runs the panel through the fake CLIs in every mode and order, scores it, and returns the failures:
    a row with no answer or time, or a v2 row whose gold was not found, read and matched."""
    if os.path.exists(out):
        shutil.rmtree(out)
    me = [sys.executable, os.path.abspath(__file__), "fake"]
    rc = main(["run", "--agent", " ".join(me + ["agent"]), "--plain", " ".join(me + ["plain"]), "-m", "fake.gguf",
               "--context", "32768", "--thinking-budget", "512", "--set", "all", "--out", out, "--turn-timeout", "5"])
    if rc:
        return [f"run exited {rc}"]
    results = score.load_json(os.path.join(out, "results.json"))
    questions = score.load_jsonl(os.path.join(HERE, "questions.jsonl"))
    scored = score.score_rows(results, questions, load_docs(), os.path.join(HERE, "data", "cache"), {})
    fails = []
    ran = [r for r in results["rows"] if not r["skipped"]]
    hung = [s for s in scored if s["mode"] == "whole" and s["id"] == "kimi-pipeline" and not s["skipped"]]
    if not hung or any((s["verdict"], s["miss_reason"]) != ("wrong", "timeout after 5 s") for s in hung):
        fails.append(f"a whole-mode turn that never answered is not a miss with its reason: {hung}")
    for r in ran:
        if r["mode"] == "whole" and r["id"] == "kimi-pipeline":
            continue
        if r.get("error") or not r.get("answer") or r.get("first_answer_s") is None or r.get("total_s") is None:
            fails.append(f"{r['mode']}/{r['order']}/{r['id']}: {r.get('error') or 'no answer or time'}")
    for s in scored:
        if s["mode"] == "v2" and s.get("extraction") is not None and (s["extraction"], s["retrieval"], s["citation"], s["extraction_source"]) != (1, 1, 1, "agent"):
            fails.append(f"{s['key']}: extraction {s['extraction']} ({s['extraction_source']}), retrieval {s['retrieval']}, citation {s['citation']}")
    if not any(r["mode"] == "whole" and r["skipped"] for r in results["rows"]):
        fails.append("no whole-mode row was skipped for not fitting")
    if {r["order"] for r in ran} != set(ORDERS):
        fails.append(f"orders run: {sorted({r['order'] for r in ran})}")
    print(f"dry run: {len(ran)} rows run, {len(results['rows']) - len(ran)} skipped, {len(fails)} failures")
    return fails


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fake"]:
        return fake_cli(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    p = sub.add_parser("panel")
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--write", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--agent", required=True, type=str.split, help="the larch-agent binary (a command line may follow)")
    r.add_argument("--plain", required=True, type=str.split, help="the larch binary, for whole and gold")
    r.add_argument("-m", "--model", required=True)
    r.add_argument("--context", type=int, required=True, help="tokens of context, given to both CLIs and recorded")
    r.add_argument("--thinking-budget", type=int, required=True, help="the agent's config.toml thinking_budget; the plain chat's budget or cap")
    r.add_argument("--answer-tokens", type=int, default=1024, help="the plain chat's answer room after thinking")
    r.add_argument("--turn-timeout", type=float, default=300, help="seconds a turn may take; a turn past it is a miss with that reason")
    r.add_argument("--modes", type=lambda s: s.split(","), default=list(MODES))
    r.add_argument("--orders", type=lambda s: s.split(","), default=list(ORDERS))
    r.add_argument("--source", choices=["file", "link"], default="file")
    r.add_argument("--set", choices=["panel", "all", "sample"], default="panel")
    r.add_argument("--size", type=int, default=20, help="with --set sample")
    r.add_argument("--seed", type=int, help="with --set sample: recorded, drawn from the clock when absent")
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--agent-config", help="a config.toml every agent data folder starts from")
    r.add_argument("--trace-glob", action="append", default=list(TRACE_GLOBS), help="where the reader's trace lines are, under the data folder")
    r.add_argument("--out", required=True)
    for s in (f, r):
        s.add_argument("--cache", default=os.path.join(HERE, "data", "cache"))
    d = sub.add_parser("dry-run")
    d.add_argument("--out", default=os.path.join(HERE, "data", "dry-run"))
    args = ap.parse_args(argv)
    questions = score.load_jsonl(os.path.join(HERE, "questions.jsonl"))
    if args.command == "fetch":
        problems = fetch(load_docs(), args.cache)
        print("\n".join(problems) or "every document is in the cache")
        return 1 if any("sha256" in p and "unpinned" not in p for p in problems) else 0
    if args.command == "panel":
        panel = {"seed": args.seed, "size": args.size, "rule": pick.__doc__, "ids": pick(questions, args.size, args.seed)}
        if args.write:
            with open(os.path.join(HERE, "panel.json"), "w", encoding="utf-8") as f:
                json.dump(panel, f, indent=1)
                f.write("\n")
        print(json.dumps(panel, indent=1))
        return 0
    if args.command == "dry-run":
        fetch(load_docs(), os.path.join(HERE, "data", "cache"))
        fails = dry_run(args.out)
        print("\n".join(fails) or "dry run passed")
        return 1 if fails else 0
    if bad := [m for m in args.modes if m not in MODES] + [o for o in args.orders if o not in ORDERS]:
        ap.error(f"unknown mode or order: {bad}")
    if os.path.exists(os.path.join(args.out, "results.json")):
        ap.error(f"{args.out} already holds a run; pick a new --out")
    if args.set == "panel":
        ids = set(score.load_json(os.path.join(HERE, "panel.json"))["ids"])
    elif args.set == "sample":
        args.seed = args.seed if args.seed is not None else int(time.time())
        ids = set(pick(questions, args.size, args.seed))
    else:
        ids = {q["id"] for q in questions}
    args.out = os.path.abspath(args.out)
    Runner(args, load_docs(), [q for q in questions if q["id"] in ids]).run()
    with open(os.path.join(args.out, "summary.md"), encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
