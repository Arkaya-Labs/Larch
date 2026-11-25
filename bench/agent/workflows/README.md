# Workflow bench

20 goals (`workflows.jsonl`) run through the app's task service: the planner, model, tool and delegate steps,
then each goal's "done when" check. The tools are four scripted MCP servers (`fixture/`: notes, calendar, issues,
files), so a run needs no network. Each tool step is approved, as a user who read it would approve it.

```
target/release/larch-agent workflow-bench -m <model> --runs 4 --json bench/results/agent-workflows-<model>-<gpu>.json
```

A run passes when it succeeds, calls every tool in its row's `calls` and gets an answer from each, and its result
holds every word of `answer`, case aside; a write tool's reply shows the arguments it wrote. It fails with its
reason if it waits unchanged for 30 s with a failure, no question and no helper at work, or runs past 30 minutes;
a settled run waits out its "done when" check. Run it on a build with the task service's "done when" checks: without
them every row, as each has acceptance criteria, settles, is never accepted and times out. The runs sample at `--temperature`
(0.7) with seed = run number, so `pass^4` (the share of goals right all 4 times) measures consistency. `pass@1` is
the mean share of passing runs.
