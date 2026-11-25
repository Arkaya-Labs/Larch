#!/usr/bin/env python3
"""Writes the memory set of `larch-agent bench --memory` into this directory: the daily notes
(daily/) and facts (facts/) that ten chats over the 60 days before the bench's date (2024-07-26)
left, as the agent writes them, and questions.json, 20 questions that name the file, the line and
the value that answer each; 5 ask about a date range and give it.

    python3 bench/agent/memory/make_memory.py           # (re)write the files
    python3 bench/agent/memory/make_memory.py --check   # exit 1 unless the committed files are
                                                        # exactly what this writes, each value is
                                                        # on its line, and this file parses as
                                                        # Python 3.9

Everything is fixed text, so the output never changes. The bench copies daily/ and facts/ into a
throwaway data folder (agent/memory/) and asks `memory::recall` each question (tier 1), then the
model, with the five hits (tier 2). Topics come back across sessions (the boiler in June and July,
the car insurance chosen in June and noted again in July), so a date-range question has a hit
outside its range to lose to.
"""

import ast
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

# (date, chat ID, [(HH:MM, daily line or None, fact (type, text) or None)]): a fact is written as
# `remember` writes it, and its daily line names it as the loop notes the call.
SESSIONS = [
    ("2024-05-28", "1716885720000000000-3121", [
        ("08:42", "read_file notes/move.txt: Moving day is Saturday 8 June; the van is booked with Rapid "
                  "Removals", None),
        ("08:47", None, ("user", "My new address from 8 June is 41 Alder Street, flat 3")),
    ]),
    ("2024-06-04", "1717492320000000000-4410", [
        ("09:12", "read_file health/dentist.txt: Dentist: Dr Okafor, Tuesday 18 June at 10:30", None),
        ("09:15", None, ("reference", "The dentist is Dr Okafor at Leith Walk Dental")),
    ]),
    ("2024-06-11", "1718132580000000000-5120", [
        ("19:03", "read_file documents/lease-alder.md: Rent is 1,850 USD a month, due on the 1st; deposit "
                  "900 USD", None),
        ("19:06", None, ("project", "The rent is 1,850 USD a month, paid to Harbour Lettings")),
    ]),
    ("2024-06-18", "1718713200000000000-2290", [
        ("12:20", "read_file notes/boiler.txt: Boiler quote from Tomasz Nowak: 640 EUR to replace the pump",
         None),
        ("12:25", None, ("reference", "The plumber is Tomasz Nowak, phone 07700 900417")),
    ]),
    ("2024-06-27", "1719509940000000000-6034", [
        ("17:40", "read_file insurance/quotes.txt: Car insurance: chose Harbour Mutual at 412 EUR a year "
                  "over Northwind at 455 EUR", None),
        ("17:44", "write_file insurance/choice.txt: wrote 27 bytes to insurance/choice.txt", None),
    ]),
    ("2024-07-02", "1719906900000000000-1877", [
        ("07:55", "read_file travel/porto.txt: Porto trip 19 to 23 July; flight TP 1352 leaves at 09:40", None),
        ("07:58", None, ("user", "I am vegetarian and allergic to peanuts")),
    ]),
    ("2024-07-09", "1720555800000000000-7302", [
        ("20:10", "read_file notes/boiler.txt: Tomasz came back about the boiler: the pump is fixed, invoice "
                  "180 EUR", None),
        ("20:14", None, ("user", "My sister Ines lives at 4 Rua das Flores, Porto")),
    ]),
    ("2024-07-15", "1721068200000000000-2551", [
        ("18:30", "read_file school/term.txt: Parents evening is Thursday 18 July at 18:00 with Mr Adeyemi",
         None),
        ("18:33", None, ("reference", "The gym timetable is in health/gym-times.md")),
    ]),
    ("2024-07-20", "1721469900000000000-3988", [
        ("10:05", "read_file travel/porto-packing.txt: Packing list: raincoat, boots, camera, adapter", None),
        ("21:47", "read_file travel/booking.txt: Dinner with Ines at Cantina 32, Monday 22 July at 20:00",
         None),
    ]),
    ("2024-07-25", "1721895300000000000-4127", [
        ("08:15", "read_file notes/car.txt: Car service booked at Leith Motors for 2 August; MOT due in "
                  "September", None),
        ("08:20", None, ("project", "My car insurance renews on 27 June each year with Harbour Mutual")),
    ]),
]

# (question, file, line, value, range): the line of `file` (relative to this folder) holds `value`,
# the text the answer must contain; `range` is (from, to) for a date-range question.
JUNE = ("2024-06-01", "2024-06-30")
JULY = ("2024-07-01", "2024-07-31")
QUESTIONS = [
    ("What is my new address?", "facts/my-new-address-from-8-june.md", 7, "41 Alder Street", None),
    ("Which company is the van booked with?", "daily/2024-05-28.md", 1, "Rapid Removals", None),
    ("Who is my dentist?", "facts/the-dentist-is-dr-okafor-at.md", 7, "Okafor", None),
    ("When is the dentist appointment?", "daily/2024-06-04.md", 1, "18 June", None),
    ("How much is the rent?", "facts/the-rent-is-1-850-usd.md", 7, "850", None),
    ("What deposit did I pay on the lease?", "daily/2024-06-11.md", 1, "900", None),
    ("What is the plumber's phone number?", "facts/the-plumber-is-tomasz-nowak-phone.md", 7,
     "07700 900417", None),
    ("How much was the boiler quote in June?", "daily/2024-06-18.md", 1, "640", JUNE),
    ("When does my car insurance renew?", "facts/my-car-insurance-renews-on-27.md", 7, "27 June", None),
    ("When does my flight to Porto leave?", "daily/2024-07-02.md", 1, "09:40", None),
    ("Am I allergic to anything?", "facts/i-am-vegetarian-and-allergic-to.md", 7, "peanut", None),
    ("Where does my sister Ines live?", "facts/my-sister-ines-lives-at-4.md", 7, "Rua das Flores", None),
    ("Where is the gym timetable?", "facts/the-gym-timetable-is-in-health.md", 7, "gym-times", None),
    ("Who is the teacher at parents evening?", "daily/2024-07-15.md", 1, "Adeyemi", None),
    ("What was on the Porto packing list last week?", "daily/2024-07-20.md", 1, "raincoat",
     ("2024-07-19", "2024-07-26")),
    ("Where is the dinner with Ines?", "daily/2024-07-20.md", 2, "Cantina 32", None),
    ("What did I book for the car this week?", "daily/2024-07-25.md", 1, "Leith Motors",
     ("2024-07-22", "2024-07-26")),
    ("What did Tomasz say about the boiler in July?", "daily/2024-07-09.md", 1, "180", JULY),
    ("Which car insurance did I choose in June?", "daily/2024-06-27.md", 1, "Harbour Mutual", JUNE),
    ("What is the name of the dental practice?", "facts/the-dentist-is-dr-okafor-at.md", 7, "Leith Walk",
     None),
]


def slug(text):
    """The first six words, lowercased and joined by dashes, as `memory::remember` names a fact."""
    words = [w.lower() for w in re.split(r"[^0-9A-Za-z]+", text) if w]
    return "-".join(words[:6])[:48].rstrip("-")


def files():
    """Every file of the set, by its path relative to this folder."""
    out = {}
    for date, chat, steps in SESSIONS:
        lines = []
        for turn, (time, line, fact) in enumerate(steps, 1):
            if fact:
                kind, text = fact
                name = slug(text)
                out[f"facts/{name}.md"] = (
                    f"---\ntype: {kind}\ncreated: {date}T{time}:05Z\nsource: chat {chat} turn {turn}\n"
                    f"confidence: medium\n---\n{text}\n")
                line = f"remember {text}: {name}"
            lines.append(f"- {time}Z chat {chat}: {line}\n")
        out[f"daily/{date}.md"] = "".join(lines)
    questions = [{"id": f"m{i:02}", "question": q, "file": file, "line": line, "value": value,
                  "range": {"from": r[0], "to": r[1]} if r else None}
                 for i, (q, file, line, value, r) in enumerate(QUESTIONS, 1)]
    out["questions.json"] = json.dumps(questions, indent=2) + "\n"
    return out


def problems(out):
    """What is wrong with the set, if anything."""
    errors = []
    for q in json.loads(out["questions.json"]):
        text = out.get(q["file"])
        lines = text.splitlines() if text else []
        if not 0 < q["line"] <= len(lines) or q["value"] not in lines[q["line"] - 1]:
            errors.append(f"{q['id']}: {q['value']!r} is not on {q['file']} line {q['line']}")
    dated = sum(1 for q in QUESTIONS if q[4])
    if (len(SESSIONS), len(QUESTIONS), dated) != (10, 20, 5):
        errors.append(f"{len(SESSIONS)} sessions, {len(QUESTIONS)} questions, {dated} dated: not 10, 20, 5")
    return errors


def build(folder, out):
    for name, text in out.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(text.encode())


def main():
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] not in ([], ["--check"]):
        sys.exit(__doc__)
    out = files()
    errors = problems(out)
    if check:
        # The Mac's Python is 3.9.
        ast.parse(pathlib.Path(__file__).read_text(), feature_version=(3, 9))
        for name, text in out.items():
            if not (HERE / name).exists():
                errors.append(f"{name} is missing")
            elif (HERE / name).read_bytes() != text.encode():
                errors.append(f"{name} differs from what make_memory.py writes")
        ours = {"make_memory.py", "questions.json", ".DS_Store"}
        extra = sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*")
                       if p.is_file() and p.name not in ours and str(p.relative_to(HERE)) not in out)
        errors += [f"{name} is not written by make_memory.py" for name in extra]
    elif not errors:
        build(HERE, out)
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
