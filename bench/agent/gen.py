#!/usr/bin/env python3
"""Writes the frozen tool-call prompts of `larch-agent bench` next to this script.

    python3 bench/agent/gen.py             # (re)write toolcalls.jsonl
    python3 bench/agent/gen.py --holdout   # (re)write holdout.jsonl, the held-out set
    python3 bench/agent/gen.py --docs      # (re)write docs-routing.jsonl
    python3 bench/agent/gen.py --check     # exit 1 unless the committed files are exactly what
                                           # this writes, the held-out set shares no prompt with
                                           # the 200, and this file parses as Python 3.9

200 prompts: 120 that need one tool with clear arguments (clear-*), 40 that need none (none-*), 20
that name a tool almost right (near-*) and 20 whose argument must be inferred from today's date or
an earlier line (infer-*). `args_required` maps each argument the call must have to text its value
must contain (case aside), a number it must equal, or null for any value. The runner's system
prompt says the date is DATE.
"""

import ast
import json
import pathlib
import random
import sys

DATE = "2024-07-26"
CHAT = ["list_dir", "read_file", "write_file", "recall", "remember", "forget", "ask_docs"]
GIT = CHAT + ["changed_files", "diff_file", "show_file", "log"]


def prompt(kind, n, tools, user, tool, args):
    return {"id": f"{kind}-{n:03}", "tools": tools, "user": user,
            "expect": {"tool": tool, "args_required": args}}


# (tools, tool, phrasings, cases, args): case i fills phrasing i % len(phrasings), and args(case) is
# what its call must carry.
CLEAR = [
    (CHAT, "list_dir", ["List the files in {p}.", "What is inside the folder {p}?",
                        "Show me what {p} holds."],
     [{"p": p} for p in ["notes", "src", "docs/drafts", "photos/2023", "recipes", "invoices",
                         "projects/garden", "music", "reports/q3", "letters", "scripts", "backups",
                         "school/math", "travel"]],
     lambda c: {"path": c["p"]}),
    (CHAT, "read_file", ["Read {p} and tell me what it says.", "Open {p} for me.",
                         "What is written in {p}?"],
     [{"p": p} for p in ["notes/todo.txt", "src/main.rs", "README.md", "recipes/bread.md",
                         "letters/landlord.txt", "reports/q3/summary.md", "travel/packing.txt",
                         "docs/drafts/intro.md", "scripts/backup.sh", "invoices/march.csv",
                         "music/playlist.txt", "school/math/homework.md", "notes/ideas.md",
                         "projects/garden/plan.md", "Cargo.toml", "notes/meeting.md",
                         "photos/2023/captions.txt", "backups/log.txt"]],
     lambda c: {"path": c["p"]}),
    (CHAT, "write_file", ["Write \"{t}\" to {p}.", "Save the text \"{t}\" in {p}.",
                          "Put \"{t}\" into the file {p}."],
     [{"p": p, "t": t} for p, t in [
         ("notes/shopping.txt", "milk, eggs, bread"), ("notes/todo.txt", "call the plumber"),
         ("letters/thanks.txt", "Thank you for the flowers"), ("travel/hotel.txt", "Room 214"),
         ("recipes/soup.md", "Boil the carrots for ten minutes"), ("notes/wifi.txt", "network: garden"),
         ("reports/q3/status.md", "On track"), ("school/math/answer.txt", "x = 42"),
         ("projects/garden/seeds.txt", "tomato, basil, pepper"), ("notes/reminder.txt", "pay rent"),
         ("docs/drafts/title.md", "A Quiet Harbour"), ("music/next.txt", "learn the blues scale"),
         ("scripts/hello.sh", "echo hello"), ("notes/birthday.txt", "Mum, 3 March")]],
     lambda c: {"path": c["p"], "text": c["t"]}),
    (CHAT, "recall", ["What did I tell you about my {q}? Check your memory.",
                      "Search your memory for anything about {q}.",
                      "Do you remember what I said about {q}?"],
     [{"q": q} for q in ["dentist", "landlord", "garden", "budget", "sister", "laptop", "holiday",
                         "car", "gym", "piano", "manager", "allergies", "passport", "thesis"]],
     lambda c: {"query": c["q"]}),
    (CHAT, "remember", ["Remember that {f}.", "Please keep this in memory: {f}.",
                        "Save this fact about me: {f}."],
     [{"f": f, "k": k} for f, k in [
         ("I am allergic to peanuts", "peanut"), ("my sister lives in Lisbon", "Lisbon"),
         ("I prefer short answers", "short"), ("my laptop runs Linux", "Linux"),
         ("the project deadline is in March", "March"), ("I water the garden on Sundays", "Sunday"),
         ("my manager is called Priya", "Priya"), ("I take the train to work", "train"),
         ("I am learning the piano", "piano"), ("my passport expires next year", "passport"),
         ("I like tea, not coffee", "tea"), ("my thesis is about rivers", "river"),
         ("I go to the gym at seven", "gym"), ("our team meets on Mondays", "Monday")]],
     lambda c: {"text": c["k"], "type": None}),
    (CHAT, "forget", ["Forget the fact [{s}].", "Delete [{s}] from your memory.",
                      "Remove the memory [{s}]; it is out of date."],
     [{"s": s} for s in ["old-address", "dentist-tuesday", "laptop-mac", "gym-at-six",
                         "sister-in-porto", "coffee-preference", "car-insurance", "team-fridays",
                         "piano-teacher", "budget-2023"]],
     lambda c: {"slug": c["s"]}),
    (GIT, "changed_files", ["Which files did branch {b} change?",
                            "List the files changed on the {b} branch."],
     [{"b": b} for b in ["feature/login", "fix/typo", "dev", "release/1.2", "docs/readme",
                         "feature/search", "fix/crash", "experiment/cache", "feature/export"]],
     lambda c: {"branch": c["b"]}),
    (GIT, "diff_file", ["Show the diff of {p} on branch {b}.",
                        "What changed in {p} on the {b} branch?"],
     [{"b": b, "p": p} for b, p in [
         ("feature/login", "src/auth.rs"), ("fix/typo", "README.md"), ("dev", "Cargo.toml"),
         ("release/1.2", "CHANGELOG.md"), ("docs/readme", "docs/setup.md"),
         ("feature/search", "src/index.rs"), ("fix/crash", "src/main.rs"),
         ("experiment/cache", "src/cache.rs"), ("feature/export", "src/export.rs")]],
     lambda c: {"branch": c["b"], "path": c["p"]}),
    (GIT, "show_file", ["Show {p} as it is at {r}.", "What did {p} look like at {r}?"],
     [{"r": r, "p": p} for r, p in [
         ("v1.0", "README.md"), ("main", "src/lib.rs"), ("v0.9", "Cargo.toml"),
         ("release/1.2", "src/config.rs"), ("dev", "docs/setup.md"), ("v1.1", "CHANGELOG.md"),
         ("main", "LICENSE"), ("feature/login", "src/auth.rs"), ("v2.0", "src/main.rs")]],
     lambda c: {"ref": c["r"], "path": c["p"]}),
    (GIT, "log", ["List the last {n} commits on {b}.", "Show me the latest {n} commits of branch {b}."],
     [{"b": b, "n": n} for b, n in [
         ("main", 5), ("dev", 10), ("feature/login", 3), ("fix/typo", 1), ("release/1.2", 7),
         ("docs/readme", 2), ("feature/search", 4), ("fix/crash", 6), ("experiment/cache", 8)]],
     lambda c: {"branch": c["b"], "n": c["n"]}),
]

NONE = [
    "What is the capital of France?", "Explain what a hash map is in two sentences.",
    "Hello! How are you today?", "What is 17 times 23?", "Write a haiku about autumn rain.",
    "Translate 'good morning' into Spanish.", "Give me three tips for sleeping better.",
    "What does HTTP stand for?", "Thanks, that was helpful.", "Is a tomato a fruit or a vegetable?",
    "Suggest a name for a small black cat.", "How many minutes are in a day?",
    "What is the difference between a list and a tuple in Python?", "Tell me a short joke.",
    "Summarize the plot of Romeo and Juliet in one sentence.", "How do I boil an egg?",
    "What is the boiling point of water at sea level?", "Which is bigger, a kilobyte or a megabyte?",
    "Write one sentence that uses the word 'harbour'.", "What does a compiler do?",
    "How far is the Moon from the Earth, roughly?", "What rhymes with 'garden'?",
    "Give me a synonym for 'quick'.", "Why is the sky blue?", "What year did the Second World War end?",
    "Explain recursion to a child.", "Good night!", "What is a prime number?",
    "Convert 10 kilometres to miles.", "What is the plural of 'mouse'?",
    "Name three primary colours.", "What does 'RAM' mean in a computer?",
    "Write a polite way to decline a meeting.", "How many sides does a hexagon have?",
    "What is the square root of 144?", "Which planet is closest to the Sun?",
    "What is the opposite of 'ancient'?", "Describe the smell of rain in one line.",
    "How do you say 'thank you' in Japanese?", "What is the main ingredient of guacamole?",
]

# The user names a tool almost right; the right call is the real tool.
NEAR = [
    ("read_file", "Use read_files on notes/todo.txt.", {"path": "notes/todo.txt"}),
    ("list_dir", "Run list_directory on src.", {"path": "src"}),
    ("read_file", "Call readfile for README.md.", {"path": "README.md"}),
    ("list_dir", "listdir recipes please.", {"path": "recipes"}),
    ("write_file", "write_files \"buy stamps\" to notes/post.txt.",
     {"path": "notes/post.txt", "text": "stamps"}),
    ("write_file", "save_file notes/plan.txt with the text \"start at nine\".",
     {"path": "notes/plan.txt", "text": "nine"}),
    ("remember", "remember_fact: I live in Leeds.", {"text": "Leeds", "type": None}),
    ("remember", "remember_this: my bike is blue.", {"text": "bike", "type": None}),
    ("recall", "search_memory for my dentist.", {"query": "dentist"}),
    ("recall", "Use recall_memory to find what I said about the budget.",
     {"query": "budget"}),
    ("forget", "forget_fact [old-address].", {"slug": "old-address"}),
    ("forget", "delete_memory [gym-at-six].", {"slug": "gym-at-six"}),
    ("log", "git_log the last 5 commits on main.", {"branch": "main", "n": 5}),
    ("log", "Show the commit_log of dev, 3 commits.", {"branch": "dev", "n": 3}),
    ("changed_files", "changed_file for branch fix/typo.", {"branch": "fix/typo"}),
    ("changed_files", "files_changed on feature/login?", {"branch": "feature/login"}),
    ("diff_file", "file_diff src/main.rs on fix/crash.",
     {"branch": "fix/crash", "path": "src/main.rs"}),
    ("diff_file", "git_diff README.md on branch docs/readme.",
     {"branch": "docs/readme", "path": "README.md"}),
    ("show_file", "cat_file Cargo.toml at v1.0.", {"ref": "v1.0", "path": "Cargo.toml"}),
    ("show_file", "show_files src/lib.rs at main.", {"ref": "main", "path": "src/lib.rs"}),
]

# The argument comes from the date in the system prompt or from an earlier line.
INFER = [
    ("read_file", "Read today's journal entry. Entries are named journal/YYYY-MM-DD.md.", {"path": DATE}),
    ("read_file", "Open the daily log for today; the logs are logs/<date>.txt.", {"path": DATE}),
    ("write_file", "Start today's journal entry, journal/YYYY-MM-DD.md, with \"Slept well\".",
     {"path": DATE, "text": "Slept"}),
    ("write_file", "Save \"water the plants\" as today's task file: tasks/<today's date>.txt.",
     {"path": DATE, "text": "water"}),
    ("list_dir", "Photos go in a folder per day, photos/YYYY-MM-DD. What is in today's?", {"path": DATE}),
    ("recall", "What did I ask you to do today? Search your memory for today's date.", {"query": DATE}),
    ("read_file", "My meeting notes are named notes/meeting-<date>.md. Read today's.", {"path": DATE}),
    ("write_file", "Write \"no meetings\" to the calendar file for today, calendar/YYYY-MM-DD.txt.",
     {"path": DATE, "text": "meetings"}),
    ("list_dir", "List the receipts folder for today: receipts/YYYY-MM-DD.", {"path": DATE}),
    ("read_file", "Read the backup report from today, backups/report-YYYY-MM-DD.txt.", {"path": DATE}),
    ("list_dir", "My project lives in projects/garden.\nWhat is in that folder?", {"path": "projects/garden"}),
    ("read_file", "The config file is settings/app.toml.\nWhat does it say?", {"path": "settings/app.toml"}),
    ("write_file", "My shopping list is notes/shop.txt.\nPut \"apples and pears\" in it.",
     {"path": "notes/shop.txt", "text": "apples"}),
    ("read_file", "I keep my recipes in recipes/.\nRead the one called soup.md.", {"path": "recipes/soup.md"}),
    ("list_dir", "Last week I saved the scans under docs/scans.\nList them.", {"path": "docs/scans"}),
    ("read_file", "The draft is in docs/drafts/letter.md.\nOpen it and tell me how it starts.",
     {"path": "docs/drafts/letter.md"}),
    ("write_file", "The file for ideas is notes/ideas.md.\nReplace what it holds with \"a garden blog\".",
     {"path": "notes/ideas.md", "text": "garden"}),
    ("list_dir", "Our invoices folder is accounts/invoices.\nWhat files are there?",
     {"path": "accounts/invoices"}),
    ("read_file", "The report I mentioned is reports/q3/final.md.\nWhat is in it?",
     {"path": "reports/q3/final.md"}),
    ("read_file", "Look in the travel folder.\nThe file packing.txt there has my list; read it.",
     {"path": "travel/packing.txt"}),
]


# The held-out set (`--holdout`, holdout.jsonl): 50 prompts that need a tool (40 clear, 5 near, 5
# infer) and 50 that need none, with cases and phrasings the 200 do not use, so a change tuned on
# toolcalls.jsonl can be checked on text it never saw. Its phrasings are drawn with HOLDOUT_SEED.
HOLDOUT_SEED = 2

HOLDOUT_CLEAR = [
    (CHAT, "list_dir", ["What files are in {p}?", "Give me a listing of {p}.", "Could you list {p} for me?"],
     [{"p": p} for p in ["videos", "receipts/2024", "garden/photos", "work/contracts", "fonts",
                         "downloads"]],
     lambda c: {"path": c["p"]}),
    (CHAT, "read_file", ["Show me the contents of {p}.", "Please read {p}.", "I need to know what {p} says."],
     [{"p": p} for p in ["notes/groceries.txt", "work/agenda.md", "recipes/curry.md", "letters/bank.txt",
                         "health/plan.md", "travel/itinerary.txt", "src/util.rs", "docs/faq.md"]],
     lambda c: {"path": c["p"]}),
    (CHAT, "write_file", ["Create {p} containing \"{t}\".", "Store \"{t}\" in {p}.",
                          "Make the file {p} say \"{t}\"."],
     [{"p": p, "t": t} for p, t in [
         ("notes/gift.txt", "a blue scarf"), ("work/status.txt", "waiting on review"),
         ("recipes/tea.md", "steep for four minutes"), ("notes/parking.txt", "level 3, bay 12"),
         ("letters/reply.txt", "See you on Friday"), ("travel/flight.txt", "TP 1352 at 09:40")]],
     lambda c: {"path": c["p"], "text": c["t"]}),
    (CHAT, "recall", ["Look up {q} in your memory.", "Have I mentioned my {q} before? Check your memory.",
                      "What do you have in memory about my {q}?"],
     [{"q": q} for q in ["plumber", "neighbour", "doctor", "insurance", "flight"]],
     lambda c: {"query": c["q"]}),
    (CHAT, "remember", ["Note for the future: {f}.", "Keep in mind that {f}.", "Store this about me: {f}."],
     [{"f": f, "k": k} for f, k in [
         ("my neighbour is called Oskar", "Oskar"), ("I cycle to work", "cycle"),
         ("my flat is on the fourth floor", "fourth"), ("I am saving for a kayak", "kayak"),
         ("my doctor is Dr Mensah", "Mensah")]],
     lambda c: {"text": c["k"], "type": None}),
    (CHAT, "forget", ["Erase the memory [{s}].", "Drop [{s}] from what you remember.", "Please forget [{s}]."],
     [{"s": s} for s in ["old-phone", "flat-third-floor", "tea-with-sugar", "gym-on-fridays"]],
     lambda c: {"slug": c["s"]}),
    (GIT, "changed_files", ["What did the {b} branch touch?", "Name the files that {b} changed."],
     [{"b": b} for b in ["feature/upload", "fix/memory-leak"]],
     lambda c: {"branch": c["b"]}),
    (GIT, "diff_file", ["Diff {p} on {b}.", "How does {p} differ on branch {b}?"],
     [{"b": b, "p": p} for b, p in [("feature/upload", "src/upload.rs"), ("fix/memory-leak", "src/pool.rs")]],
     lambda c: {"branch": c["b"], "path": c["p"]}),
    (GIT, "show_file", ["Print {p} from {r}.", "Give me {p} at revision {r}."],
     [{"r": "v1.3", "p": "src/net.rs"}],
     lambda c: {"ref": c["r"], "path": c["p"]}),
    (GIT, "log", ["Give me {n} recent commits from {b}.", "What are the newest {n} commits on {b}?"],
     [{"b": "feature/upload", "n": 4}],
     lambda c: {"branch": c["b"], "n": c["n"]}),
]

HOLDOUT_NONE = [
    "What is the capital of Canada?", "Explain what a linked list is in one sentence.",
    "Good morning! Any plans for the weekend?", "What is 19 times 21?", "Write a haiku about the sea.",
    "Translate 'see you tomorrow' into French.", "Give me two tips for staying focused.",
    "What does DNS stand for?", "Great, thank you.", "Is a peanut a nut?",
    "Suggest a name for a brown dog.", "How many seconds are in an hour?",
    "What is the difference between a stack and a queue?", "Tell me a riddle.",
    "Summarize the plot of Hamlet in one sentence.", "How do I make toast?",
    "At what temperature does water freeze?", "Which is longer, a mile or a kilometre?",
    "Write one sentence that uses the word 'lantern'.", "What does an operating system do?",
    "How far is the Sun from the Earth, roughly?", "What rhymes with 'window'?",
    "Give me a synonym for 'happy'.", "Why do leaves change colour in autumn?",
    "What year did the first person walk on the Moon?", "Explain gravity to a child.",
    "See you later!", "What is an even number?", "Convert 5 miles to kilometres.",
    "What is the plural of 'goose'?", "Name three kinds of cloud.", "What does 'CPU' mean?",
    "Write a polite way to ask for more time.", "How many sides does an octagon have?",
    "What is 15 percent of 200?", "Which planet is the largest?", "What is the opposite of 'generous'?",
    "Describe the sound of snow in one line.", "How do you say 'hello' in Italian?",
    "What is the main ingredient of hummus?", "Who wrote Pride and Prejudice?",
    "What is a verb?", "How many legs does a spider have?", "What colour do you get from blue and yellow?",
    "Give me a word that means 'very big'.", "What is the freezing point of water in Fahrenheit?",
    "Explain what a password manager is.", "What is the tallest animal?",
    "Write a two-line poem about coffee.", "How many days are in a leap year?",
]

HOLDOUT_NEAR = [
    ("read_file", "Use file_read on work/agenda.md.", {"path": "work/agenda.md"}),
    ("list_dir", "Run ls_dir on downloads.", {"path": "downloads"}),
    ("recall", "memory_search for my neighbour.", {"query": "neighbour"}),
    ("write_file", "file_write \"hello\" to notes/hi.txt.", {"path": "notes/hi.txt", "text": "hello"}),
    ("forget", "forget_memory [old-phone].", {"slug": "old-phone"}),
]

HOLDOUT_INFER = [
    ("read_file", "Today's shopping list is lists/YYYY-MM-DD.txt. Read it.", {"path": DATE}),
    ("write_file", "Add \"call the bank\" to today's notes file, notes/<date>.md.",
     {"path": DATE, "text": "bank"}),
    ("list_dir", "Scans are filed by day under scans/YYYY-MM-DD. List today's.", {"path": DATE}),
    ("read_file", "The plan is in work/plan-v2.md.\nRead it.", {"path": "work/plan-v2.md"}),
    ("list_dir", "I moved my essays to school/essays.\nWhat is in there?", {"path": "school/essays"}),
]

# The document rows (`--prompts docs-routing.jsonl`, docs-routing.jsonl): questions about what
# bench/agent/docs says, where ask_docs is the right call, 6 the documents answer (answered-*) and
# 4 they do not (absent-*), whose answer must then be ask_docs's refusal and no made-up value. The
# first absent row is the chat question that once got an invented password.
DOCS_ANSWERED = [
    "What is the monthly rent for the apartment at 42 Alder Street?",
    "What time does my flight to Lisbon depart?",
    "When is my next dentist appointment?",
    "What is the lockbox code for the cabin key?",
    "How often should I clean the washer's drain pump filter?",
    "Is the next book club meeting before the Pine Hollow cabin weekend?",
]
DOCS_ABSENT = [
    "what's the wifi password at the cabin?",
    "What is my passport number?",
    "How long is the warranty on the dishwasher?",
    "What is the monthly rent for 17 Maple Avenue in 2027?",
]


def docs_routing():
    rows = [prompt("answered", i + 1, CHAT, user, "ask_docs", {"question": None})
            for i, user in enumerate(DOCS_ANSWERED)]
    return rows + [prompt("absent", i + 1, CHAT, user, "ask_docs", {"question": None})
                   for i, user in enumerate(DOCS_ABSENT)]


def build(clear, none, near, infer, phrase):
    """The prompts in file order; phrase(i, phrasings) picks case i's phrasing."""
    out = []
    for tools, tool, phrasings, cases, args in clear:
        for i, case in enumerate(cases):
            out.append(prompt("clear", len(out) + 1, tools, phrase(i, phrasings).format(**case),
                              tool, args(case)))
    out += [prompt("none", i + 1, CHAT, user, None, {}) for i, user in enumerate(none)]
    for i, (tool, user, args) in enumerate(near):
        out.append(prompt("near", i + 1, CHAT if tool in CHAT else GIT, user, tool, args))
    out += [prompt("infer", i + 1, CHAT, user, tool, args) for i, (tool, user, args) in enumerate(infer)]
    return out


def prompts():
    return build(CLEAR, NONE, NEAR, INFER, lambda i, phrasings: phrasings[i % len(phrasings)])


def holdout():
    rng = random.Random(HOLDOUT_SEED)
    return build(HOLDOUT_CLEAR, HOLDOUT_NONE, HOLDOUT_NEAR, HOLDOUT_INFER,
                 lambda i, phrasings: rng.choice(phrasings))


def same(user):
    """User text as the overlap check compares it: case and spacing aside."""
    return " ".join(user.lower().split())


def errors(main, held):
    """What is wrong with the held-out set, if anything."""
    out = []
    seen = {same(p["user"]) for p in main}
    out += [f"holdout {p['id']} repeats a toolcalls.jsonl prompt" for p in held if same(p["user"]) in seen]
    tool = sum(1 for p in held if p["expect"]["tool"] is not None)
    if (tool, len(held) - tool) != (50, 50):
        out.append(f"holdout has {tool} tool and {len(held) - tool} no-tool prompts, not 50 and 50")
    return out


def text(rows):
    return "".join(json.dumps(p) + "\n" for p in rows).encode()


def main():
    folder = pathlib.Path(__file__).resolve().parent
    files = {"toolcalls.jsonl": text(prompts()), "holdout.jsonl": text(holdout()),
             "docs-routing.jsonl": text(docs_routing())}
    found = errors(prompts(), holdout())
    if sys.argv[1:] == ["--check"]:
        # The Mac's Python is 3.9.
        ast.parse(pathlib.Path(__file__).read_text(), feature_version=(3, 9))
        for name, want in files.items():
            if not (folder / name).exists() or (folder / name).read_bytes() != want:
                found.append(f"not what gen.py writes: {name}")
        if found:
            sys.exit("\n".join(found))
    elif found:
        sys.exit("\n".join(found))
    else:
        flags = {"--holdout": "holdout.jsonl", "--docs": "docs-routing.jsonl"}
        name = flags.get(" ".join(sys.argv[1:]), "toolcalls.jsonl")
        (folder / name).write_bytes(files[name])


if __name__ == "__main__":
    main()
