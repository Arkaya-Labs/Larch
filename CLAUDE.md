@AGENTS.md

# Working in larch

Follow [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Read affected code and its callers
before editing; reuse the standard library, platform facilities and existing dependencies before adding
new machinery. Prefer a small, readable diff that fixes the cause and preserves the repository's
correctness, safety and data-durability invariants.

Use checks appropriate to the change and report exact results, including blocked or skipped commands.
No assistant-specific review command, sprint ritual or lane assignment replaces human review.
