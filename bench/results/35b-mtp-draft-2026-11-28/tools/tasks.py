"""Draft pass task sets (bench only): 6 tasks x 16 prompts, from the repo's own source and docs plus fixed
question lists. Writes <dir>/<task>.jsonl. Held out from the parity prompts (llama.cpp dumps of wiki text)."""
import json, os, re, sys
root, out = sys.argv[1], sys.argv[2]
def read(p): return open(os.path.join(root, p), encoding='utf-8').read()
def w(task, rows):
    with open(os.path.join(out, task + '.jsonl'), 'w') as f:
        for i, msgs in enumerate(rows):
            f.write(json.dumps({'id': f'{task}-{i:02d}', 'messages': msgs}) + '\n')
# code edit: real functions from the repo, with an edit to make
src = read('crates/larch-engine/src/guess.rs') + read('crates/larch-engine/src/fit.rs') + read('crates/larch-model/src/qwen35_mtp.rs')
fns = [m.group(0) for m in re.finditer(r'(?ms)^(?:    )?(?:pub(?:\(crate\))? )?fn [a-z_]+\(.*?^(?:    )?\}\n', src)]
fns = [f for f in fns if 600 < len(f) < 3500][:16]
edits = ['Rename every local variable to a longer descriptive name and return the full updated function.',
 'Add a doc comment to every statement block explaining it, and return the whole function.',
 'Convert this function to return a Result with a custom error enum where it can fail; show the full code.',
 'Rewrite this function so it never panics; return the full updated function only.']
w('code_edit', [[{'role': 'user', 'content': f'{edits[i % 4]}\n\n```rust\n{f}```'}] for i, f in enumerate(fns)])
gen = ['a Python function that parses an ISO 8601 duration string into seconds, with tests',
 'a Rust struct for an LRU cache with get and put in O(1), with unit tests',
 'a TypeScript React hook that debounces a value, with a usage example',
 'a Go HTTP handler that streams server-sent events from a channel',
 'a Python script that deduplicates lines of a large file using constant memory per line hash',
 'a C function that reverses a singly linked list in place, with a small main()',
 'a SQL schema for a library (books, members, loans) and three example queries',
 'a bash script that rotates log files older than 7 days into gzip archives',
 'a Rust function that merges k sorted vectors using a binary heap, with tests',
 'a Python class implementing a token bucket rate limiter with thread safety',
 'a JavaScript function that deep-merges two objects, handling arrays and cycles',
 'a Python asyncio crawler that fetches URLs with a concurrency limit of 5',
 'a Rust command-line tool with clap that counts words, lines and bytes like wc',
 'a Java method that validates a Sudoku board, with JUnit tests',
 'a Python function computing the Levenshtein distance with O(min(n,m)) memory',
 'a Kotlin data class and repository for a to-do list backed by an in-memory map']
w('code_gen', [[{'role': 'user', 'content': f'Write {g}.'}] for g in gen])
chat = ['What are some good ways to keep basil alive indoors over the winter?',
 'Can you explain how compound interest works with a simple example?',
 'I have a job interview for a product manager role tomorrow. How should I prepare tonight?',
 'What is the difference between weather and climate?',
 'Suggest a 3-day itinerary for a first visit to Lisbon.',
 'How do noise-cancelling headphones actually work?',
 'My sourdough bread comes out dense. What could I be doing wrong?',
 'Explain the rules of cricket to someone who knows baseball.',
 'What should I consider when adopting a rescue dog?',
 'How can I get better at remembering names when I meet people?',
 'Write a short, friendly email declining a wedding invitation.',
 'Why is the sky blue, and why are sunsets red?',
 'What are the pros and cons of renting versus buying a home?',
 'Give me a beginner weekly running plan to reach 5 km.',
 'How do vaccines train the immune system?',
 'What are good strategies for a productive work-from-home day?']
w('chat', [[{'role': 'user', 'content': c}] for c in chat])
# long-document QA: repo docs, ~6000 characters each, with a question
docs = sorted(os.listdir(os.path.join(root, 'docs')))
docs = ['docs/' + d for d in docs if d.endswith('.md')][:15] + ['README.md']
qs = ['Summarize the main points of this document in a bulleted list.',
 'List every command or setting this document mentions and what each does.',
 'What problems might a new user run into, according to this document?',
 'Write five questions a reader could answer from this document, with answers.']
w('doc_qa', [[{'role': 'user', 'content': f'Read the document below and answer.\n\n<document>\n{read(d)[:6000]}\n</document>\n\n{qs[i % 4]}'}] for i, d in enumerate(docs)])
math = ['A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?',
 'Find all integer solutions of x^2 - y^2 = 45.',
 'What is the sum of the first 100 odd numbers? Prove your formula.',
 'A rectangle has perimeter 46 and area 120. What are its sides?',
 'How many ways can 8 people sit at a round table if two of them refuse to sit together?',
 'Solve for x: log2(x) + log2(x - 2) = 3.',
 'A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the probability both are blue?',
 'Find the remainder when 7^100 is divided by 13.',
 'The sum of three consecutive even integers is 162. Find them.',
 'Compute the derivative of f(x) = x^3 ln(x) and find its critical points.',
 'A tank fills in 6 hours with pipe A and 9 hours with pipe B. How long with both?',
 'How many positive divisors does 3600 have?',
 'Find the area of a triangle with sides 13, 14 and 15.',
 'If a car depreciates 15 % a year, after how many years is it worth under half its price?',
 'Prove that the square root of 3 is irrational.',
 'Evaluate the integral of x e^(2x) from 0 to 1.']
w('math', [[{'role': 'user', 'content': m + ' Think it through step by step.'}] for m in math])
# agent multi-turn: system with tools, earlier assistant turns with think blocks and tool calls, tool results
sysmsg = ('You are a coding agent. You can call tools: read_file(path), grep(pattern), run(cmd). Call one tool per turn '
 'as <tool_call>{"name": ..., "arguments": {...}}</tool_call>, then continue when the result arrives.')
files = ['crates/larch-engine/src/guess.rs', 'crates/larch-model/src/qwen35_mtp.rs', 'crates/larch-engine/src/fit.rs',
 'docs/configuration.md', 'docs/cli.md', 'docs/troubleshooting.md', 'docs/performance.md', 'docs/api.md']
asks = ['Find where the number of speculative guesses is capped and explain how it is chosen.',
 'Explain what the MTP drafter caches per sequence.', 'Find how the drafter device bytes are counted in the fit.',
 'Which environment variables change drafting? Make a table.', 'How do I run a benchmark from the command line?',
 'The server says it does not fit in memory. What can I do?', 'How is throughput measured here?',
 'What endpoints does the API serve?']
rows = []
for i in range(16):
    f = files[i % 8]; body = read(f)[(i // 8) * 3000:(i // 8) * 3000 + 3000]
    rows.append([{'role': 'system', 'content': sysmsg},
      {'role': 'user', 'content': asks[i % 8]},
      {'role': 'assistant', 'content': f'<think>\nI should look at the relevant file first. {f} looks right.\n</think>\n\n<tool_call>\n{{"name": "read_file", "arguments": {{"path": "{f}"}}}}\n</tool_call>'},
      {'role': 'tool', 'content': body},
      {'role': 'assistant', 'content': '<think>\nThe file has part of it. Let me grep for the key terms too.\n</think>\n\n<tool_call>\n{"name": "grep", "arguments": {"pattern": "guess|draft|fit"}}\n</tool_call>'},
      {'role': 'tool', 'content': '\n'.join(l for l in body.splitlines() if re.search('guess|draft|fit', l))[:1200] or '(no matches)'},
      {'role': 'user', 'content': 'Thanks. Now answer my question fully, quoting the code or text you found.'}])
w('agent', rows)
