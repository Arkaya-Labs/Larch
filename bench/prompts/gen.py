#!/usr/bin/env python3
"""Writes the frozen benchmark workloads (architecture 16) into this directory.

    python3 bench/prompts/gen.py           # (re)write the files
    python3 bench/prompts/gen.py --check   # exit 1 unless the committed files are exactly what this writes

The text is built from words that are one token with a leading space in both primary vocabularies
(m1 and m5 of bench/models.toml), so every message has an exact content length in tokens on both. Those lengths
are a design target: bench/manifest/<alias>.jsonl is the evidence, measured on the real GGUFs.
"""

import json
import math
import pathlib
import random
import sys

NOUN = """cache server request user model queue block engine thread kernel buffer driver memory session message
answer question system process worker client socket packet file folder record table index page token batch window
budget limit report review team project plan schedule task ticket release build branch commit test result error
warning signal metric chart graph value number price order customer product service account invoice payment market
garden river mountain forest city village road bridge station train ship car truck plane airport hotel kitchen recipe
bread soup coffee tea cake dinner lunch breakfast morning evening night week month year season summer winter spring
autumn weather storm rain snow wind sun moon star planet ocean island book story chapter letter paper note list map
picture music song movie game player coach school student teacher lesson course class library museum hospital doctor
nurse patient office manager meeting call email phone screen keyboard printer camera light door wall floor roof house
room chair desk lamp clock bottle glass cup plate knife box bag shirt shoe hat coat problem solution idea reason method
step detail example change update version feature option setting rule policy goal risk cost""".split()
VERB = """keeps sends reads writes checks moves builds starts stops opens closes finds holds needs uses makes takes gives
shows tells asks helps runs pays saves loads stores counts tracks follows leads joins sorts fixes breaks marks calls
waits brings carries covers adds drops pulls pushes cleans changes creates reports returns receives describes explains
improves reduces increases""".split()
ADJ = """large small slow fast new old early late short long simple clear quiet busy empty full open free main local
public private final first second third next last best better strong weak warm cold hot dry wet heavy light bright
dark green blue red white black high low deep wide narrow safe careful quick happy calm fresh clean""".split()
ADV = "quickly slowly carefully often always never sometimes usually again also still only already".split()
PREP = "in on for with from to into over under after before near during without about across through behind".split()
DET = "the a each every this that our their its one another".split()
CONJ = "because when while after before and but so although since unless".split()
# Message openers: one token with no leading space.
OPEN = "The A Each Every This That Our Their Please We You It Then When After Before Today Here".split()


class Text:
    """Seeded text with exact token counts: every word is one token, and so are '.' and '.\\n\\n'."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def pick(self, xs):
        # Indexing by random() rather than choice(): random()'s stream is stable across Python versions.
        return xs[int(self.r.random() * len(xs))]

    def clause(self):
        w = [self.pick(DET)]
        if self.r.random() < 0.5:
            w.append(self.pick(ADJ))
        w += [self.pick(NOUN), self.pick(VERB)]
        if self.r.random() < 0.2:
            w.append(self.pick(ADV))
        w += [self.pick(DET), self.pick(NOUN)]
        while self.r.random() < 0.4:
            w += [self.pick(PREP), self.pick(DET), self.pick(NOUN)]
        return w

    def words(self, n):
        w = self.clause()
        while len(w) < n:
            w += [self.pick(CONJ)] + self.clause()
        return w[:n]

    def content(self, n):
        """Exactly n tokens (n >= 3): sentences of 8 to 24 tokens, paragraphs of 3 to 6 sentences."""
        assert n >= 3
        out, left, first, in_par, par_len = [], n, True, 0, 3 + int(self.r.random() * 4)
        while left:
            k = 8 + int(self.r.random() * 17)
            k = left if left - k < 3 else k  # never leave a remainder too short for a sentence
            w = self.words(k - 1)
            head = self.pick(OPEN) if first else " " + w[0].capitalize()
            out.append(head + "".join(" " + x for x in w[1:]))
            left -= k
            in_par += 1
            if left and in_par == par_len:
                out.append(".\n\n")
                first, in_par, par_len = True, 0, 3 + int(self.r.random() * 4)
            else:
                out.append(".")
                first = False
        return "".join(out)


def between(t, lo, hi):
    return lo + int(t.r.random() * (hi - lo + 1))


def conversation(cid, system, turns):
    """turns: (user_tokens, answer_tokens); the scripted answer has the timed answer's exact length."""
    t = Text(cid)
    msgs = [{"role": "system", "content": t.content(system)}] if system else []
    for i, (u, a) in enumerate(turns):
        msgs.append({"role": "user", "content": t.content(u)})
        msgs.append({"role": "assistant", "max_tokens": a} | ({"content": t.content(a)} if i + 1 < len(turns) else {}))
    return {"id": cid, "messages": msgs}


def chat_turns(cid, n):
    """W1 lengths: user messages 40 to 300 tokens, answers 100 to 400."""
    t = Text(cid + ":lengths")
    return [(between(t, 40, 300), between(t, 100, 400)) for _ in range(n)]


def workloads():
    # The 200-prompt set (W4, and A3's parity calibration): prompts 50 to 2000 tokens, log-uniform,
    # with the content capped at 1960 so the rendered prompt stays within 2000 on every template;
    # answers 128 to 512.
    t = Text("chat:lengths")
    chat = []
    for i in range(200):
        n = round(math.exp(math.log(50) + t.r.random() * (math.log(1960) - math.log(50))))
        chat.append(conversation(f"chat-{i:03}", 0, [(n, between(t, 128, 512))]))
    return {
        "chat.jsonl": chat,
        # W1: one conversation, 12 turns.
        "w1-chat.jsonl": [conversation("w1-c0", 48, chat_turns("w1-c0", 12))],
        # W2: 8 conversations of 8 turns; the 4-concurrent run takes the first 4.
        "w2-chats.jsonl": [conversation(f"w2-c{i}", 48, chat_turns(f"w2-c{i}", 8)) for i in range(8)],
        # W3: 3 agent loops; a 4096-token prefix, then 20 calls of a 130-token tool result (sent as a
        # user message, which every template renders) and a 120-token answer. The 1-loop run takes the first.
        "w3-agent.jsonl": [conversation(f"w3-l{i}", 4096, [(130, 120)] * 20) for i in range(3)],
        # W5: a 2048- and a 4096-token prompt, 128-token answers.
        "w5-longprompt.jsonl": [conversation(f"w5-{n}", 0, [(n, 128)]) for n in (2048, 4096)],
    }


def main():
    here = pathlib.Path(__file__).resolve().parent
    check = sys.argv[1:] == ["--check"]
    stale = []
    for name, convs in workloads().items():
        text = "".join(json.dumps(c) + "\n" for c in convs)
        path = here / name
        if check:
            if not path.exists() or path.read_bytes() != text.encode():
                stale.append(name)
        else:
            path.write_bytes(text.encode())
    if stale:
        sys.exit("not what gen.py writes: " + ", ".join(stale))


if __name__ == "__main__":
    main()
