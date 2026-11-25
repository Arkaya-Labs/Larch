//! The workflow bench: fixed goals run through the app's task service (planner, model, tool and
//! delegate steps, "done when" checks) against scripted MCP servers, `runs` times each, scored
//! pass@1 and pass^k.

use super::{McpFixture, Scripted};
use crate::runtime::driver::{StepTool, StepToolSource};
use crate::tool::Tool;
use anyhow::Context;
use larch_task::types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// A run that has not ended by then fails with "timed out".
const RUN_LIMIT: Duration = Duration::from_secs(1800);
/// A run `stuck` and unchanged this long has stopped; a retry's backoff is shorter.
const STALLED: Duration = Duration::from_secs(30);

/// One goal of `workflows.jsonl`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub id: String,
    pub goal: String,
    /// The fixture's servers whose tools the run is granted (`mcp.<server>`).
    pub servers: Vec<String>,
    /// The run's "done when" criteria.
    #[serde(default)]
    pub acceptance: Vec<String>,
    /// Tools a passing run called and got a result from, by their agent names.
    pub calls: Vec<String>,
    /// Words its result holds, case aside.
    #[serde(default)]
    pub answer: Vec<String>,
}

/// One run of one workflow.
#[derive(Clone, Debug, Serialize)]
pub struct WorkflowRun {
    pub id: String,
    pub run: usize,
    pub status: String,
    pub passed: bool,
    /// Tools that answered, in call order.
    pub calls: Vec<String>,
    /// Tool steps approved: the bench approves each one, as a user who reads it would.
    pub approvals: usize,
    pub result: Option<String>,
    /// Why it did not pass.
    pub failure: Option<String>,
    pub seconds: f64,
}

/// The report `bench/results/agent-workflows-*.json` keeps, rewritten after every run.
#[derive(Debug, Default, Serialize)]
pub struct WorkflowReport {
    pub model: String,
    pub runs: usize,
    pub temperature: f32,
    pub context_tokens: usize,
    pub rows: Vec<WorkflowRun>,
    pub summary: WorkflowSummary,
}

/// Pass rates over the workflows that have all their runs.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct WorkflowSummary {
    pub workflows: usize,
    /// Mean over workflows of passes / runs.
    pub pass_at_1: f64,
    /// The share of workflows right in every run.
    pub pass_pow_k: f64,
}

/// Read `workflows.jsonl`; every server a row names is one of the fixture's.
pub fn load(path: &Path, fixture: &McpFixture) -> anyhow::Result<Vec<Workflow>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let rows: Vec<Workflow> = (text.lines().filter(|l| !l.trim().is_empty()))
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    for row in &rows {
        if let Some(s) = row
            .servers
            .iter()
            .find(|s| !fixture.servers.contains_key(*s))
        {
            anyhow::bail!("{}: server {s} is not in the fixture", row.id);
        }
    }
    Ok(rows)
}

/// Run each workflow `runs` times in its own data folder under `scratch`, sampling at
/// `temperature` with seed = run, and `save` the report after every run.
pub fn run(
    larch: &larch::Larch,
    fixture: &McpFixture,
    rows: &[Workflow],
    report: &mut WorkflowReport,
    scratch: &Path,
    mut save: impl FnMut(&WorkflowReport) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    for row in rows {
        for n in 0..report.runs {
            let started = Instant::now();
            let dir = scratch.join(format!("{}-{n}", row.id));
            std::fs::create_dir_all(&dir)?;
            let mut done = once(larch, fixture, row, n, report, &dir)?;
            done.seconds = started.elapsed().as_secs_f64();
            report.rows.push(done);
            report.summary = summary(&report.rows, report.runs);
            save(report)?;
        }
    }
    Ok(())
}

/// One run: host the service on `dir`, start the goal, approve each tool step, wait for its end.
fn once(
    larch: &larch::Larch,
    fixture: &McpFixture,
    row: &Workflow,
    n: usize,
    report: &WorkflowReport,
    dir: &Path,
) -> anyhow::Result<WorkflowRun> {
    let called = Arc::new(Mutex::new(Vec::new()));
    let tools = step_tools(fixture, &row.servers, called.clone());
    // The app's step parameters (`larch_app::task::step_params`), sampled so k runs differ.
    let params = larch::GenParams {
        max_tokens: report.context_tokens as u32 / 2,
        thinking: Some(false),
        class: larch_engine::Class::Background,
        temperature: Some(report.temperature),
        seed: Some(n as u64),
        ..Default::default()
    };
    let context = crate::tool::ToolCtx {
        allowed: Vec::new(),
        data_dir: dir.to_owned(),
        source: "workflow".into(),
    };
    let (tell, told) = std::sync::mpsc::channel();
    let host = {
        let (larch, dir, seats) = (
            larch.clone(),
            dir.to_owned(),
            crate::runtime::inference::SeatAdmission::of(larch),
        );
        let context_tokens = report.context_tokens;
        std::thread::spawn(move || {
            crate::runtime::service::serve_with_tools(
                &dir,
                larch,
                seats,
                params,
                context_tokens,
                1,
                Some((tools, context)),
                Box::new(move |token: &str| {
                    let _ = tell.send(token.to_owned());
                }),
            )
        })
    };
    // The token, or nothing once the service ended before it listened.
    if told.recv().is_err() {
        let error = host
            .join()
            .map_err(|_| anyhow::anyhow!("the service panicked"))?;
        anyhow::bail!("the task service did not start: {error:?}");
    }
    let client = larch_task::ipc::Client::connect(dir).map_err(anyhow::Error::msg)?;
    let ended = drive(&client, row);
    client.shutdown().map_err(anyhow::Error::msg)?;
    let hosted = host
        .join()
        .map_err(|_| anyhow::anyhow!("the service panicked"))?;
    let (state, approvals, why) = ended?;
    hosted.map_err(anyhow::Error::msg)?;
    let calls = called
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .clone();
    let failure = why.or_else(|| verdict(row, &state, &calls));
    Ok(WorkflowRun {
        id: row.id.clone(),
        run: n,
        status: format!("{:?}", state.status),
        passed: failure.is_none(),
        calls,
        approvals,
        result: state.result,
        failure,
        seconds: 0.0,
    })
}

/// Create and start the goal, approve every decision it asks for, and return its last state,
/// the approvals and why it ended early, if it did.
fn drive(
    client: &larch_task::ipc::Client,
    row: &Workflow,
) -> anyhow::Result<(RunState, usize, Option<String>)> {
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: row.goal.clone(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: row.servers.iter().map(|s| format!("mcp.{s}")).collect(),
        budgets: Default::default(),
        acceptance: row.acceptance.clone(),
    };
    let msg = anyhow::Error::msg;
    let created = (client.create(
        "create",
        &RunState::new("run".into(), definition).map_err(msg)?,
    ))
    .map_err(msg)?;
    client
        .start_goal("run", "start", created.revision)
        .map_err(msg)?;
    let (started, mut approvals) = (Instant::now(), 0);
    let mut seen = (0, Instant::now());
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let state = client.get("run").map_err(msg)?.context("the run is gone")?;
        if matches!(
            state.status,
            RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled
        ) {
            return Ok((state, approvals, None));
        }
        if let Some(request) = &state.waiting_decision
            && let Some(decision) = state.decisions.get(request)
            && decision.approved.is_none()
        {
            let command = format!("approve-{approvals}");
            (client.respond_decision(
                "run",
                &command,
                state.revision,
                request,
                &decision.action_digest,
                true,
            ))
            .map_err(msg)?;
            approvals += 1;
            continue;
        }
        if state.revision != seen.0 {
            seen = (state.revision, Instant::now());
        }
        let why = match seen.1.elapsed() > STALLED {
            true => stuck(&state),
            false => None,
        };
        let why = why.or_else(|| (started.elapsed() > RUN_LIMIT).then(|| "timed out".into()));
        if why.is_some() {
            return Ok((state, approvals, why));
        }
    }
}

/// Why `state` will not move on its own, if it will not: a failed step or helper with no question
/// for the user and no helper still working. A settled graph stays Ready through its "done when"
/// check, which records nothing until the model answers, so only `RUN_LIMIT` ends it.
fn stuck(state: &RunState) -> Option<String> {
    let helping = state.waiting_child.is_some() && state.child_failure.is_none();
    match state.status {
        RunStatus::Waiting if state.waiting_decision.is_none() && !helping => {
            let helper = (state.child_failure.as_ref())
                .map(|f| f.reason.clone().unwrap_or("a helper failed".into()));
            helper.or(state.attempt_failure.clone())
        }
        _ => None,
    }
}

/// Why a run that ended did not pass: not succeeded, a tool it should have called, a word its
/// result lacks.
fn verdict(row: &Workflow, state: &RunState, calls: &[String]) -> Option<String> {
    if state.status != RunStatus::Succeeded {
        return Some(
            state
                .attempt_failure
                .clone()
                .unwrap_or("not succeeded".into()),
        );
    }
    if let Some(tool) = row.calls.iter().find(|t| !calls.contains(t)) {
        return Some(format!("never called {tool}"));
    }
    let result = state.result.as_deref().unwrap_or_default().to_lowercase();
    (row.answer.iter())
        .find(|w| !result.contains(&w.to_lowercase()))
        .map(|w| format!("the result lacks \"{w}\""))
}

/// The fixture's tools of `servers` as step tools under `mcp.<server>`, each answered call
/// noted in `called`.
fn step_tools(
    fixture: &McpFixture,
    servers: &[String],
    called: Arc<Mutex<Vec<String>>>,
) -> StepToolSource {
    let tools: Vec<StepTool> = (servers.iter())
        .flat_map(|s| (fixture.servers.get(s).into_iter().flatten()).map(move |spec| (s, spec)))
        .map(|(server, spec)| StepTool {
            tool: Arc::new(Noted {
                tool: Scripted {
                    spec: spec.clone(),
                    result: fixture.results.get(&spec.name).cloned().unwrap_or_default(),
                },
                called: called.clone(),
            }),
            capability: format!("mcp.{server}"),
        })
        .collect();
    Arc::new(move || tools.clone())
}

/// A scripted tool that notes each call it answers.
struct Noted {
    tool: Scripted,
    called: Arc<Mutex<Vec<String>>>,
}

impl Tool for Noted {
    fn spec(&self) -> crate::tool::ToolSpec {
        self.tool.spec()
    }

    fn call(
        &self,
        args: &serde_json::Value,
        ctx: &crate::tool::ToolCtx,
    ) -> Result<String, crate::tool::ToolError> {
        let out = self.tool.call(args, ctx)?;
        if let Ok(mut called) = self.called.lock() {
            called.push(self.tool.spec.name.clone());
        }
        // A write's reply shows what it wrote, as a real server's created record does: a tool
        // step's reply is the run's result, which the answer and the "done when" check read.
        Ok(match self.tool.spec.permission {
            crate::tool::Permission::Read => out,
            _ => format!("{out}: {args}"),
        })
    }
}

/// pass@1 and pass^k over the workflows with all `k` runs.
pub fn summary(rows: &[WorkflowRun], k: usize) -> WorkflowSummary {
    let mut by_id = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    for r in rows {
        let (n, c) = by_id.entry(&r.id).or_default();
        *n += 1;
        *c += r.passed as usize;
    }
    let full: Vec<_> = by_id.values().filter(|(n, _)| *n == k).collect();
    let mean = |f: &dyn Fn(usize, usize) -> f64| match full.len() {
        0 => 0.0,
        len => full.iter().map(|(n, c)| f(*n, *c)).sum::<f64>() / len as f64,
    };
    WorkflowSummary {
        workflows: full.len(),
        pass_at_1: mean(&|n, c| c as f64 / n as f64),
        // With n = k runs, C(c, k) / C(n, k) is 1 when every run passed, else 0.
        pass_pow_k: mean(&|n, c| (c == n) as u8 as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, passed: bool) -> WorkflowRun {
        WorkflowRun {
            id: id.into(),
            run: 0,
            status: "Succeeded".into(),
            passed,
            calls: vec![],
            approvals: 0,
            result: None,
            failure: None,
            seconds: 0.0,
        }
    }

    #[test]
    fn pass_pow_k_counts_only_workflows_right_every_run() {
        let rows = [
            row("a", true),
            row("a", true),
            row("b", true),
            row("b", false),
            // An unfinished workflow is left out.
            row("c", true),
        ];
        let s = summary(&rows, 2);
        assert_eq!((s.workflows, s.pass_at_1, s.pass_pow_k), (2, 0.75, 0.5));
    }

    #[test]
    fn the_bench_workflows_name_only_fixture_servers() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/agent/workflows");
        let fixture = McpFixture::load(&dir.join("fixture")).unwrap();
        let rows = load(&dir.join("workflows.jsonl"), &fixture).unwrap();
        assert_eq!(rows.len(), 20);
        let names: std::collections::BTreeSet<_> = (fixture.servers.values().flatten())
            .map(|t| &t.name)
            .collect();
        for r in &rows {
            assert!(r.calls.iter().all(|c| names.contains(c)), "{}", r.id);
        }
    }

    #[test]
    fn a_write_rows_answer_is_in_what_its_last_call_wrote() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/agent/workflows");
        let fixture = McpFixture::load(&dir.join("fixture")).unwrap();
        let rows = load(&dir.join("workflows.jsonl"), &fixture).unwrap();
        // What a right run writes last, by row.
        let wrote = serde_json::json!({
            "note-launch-summary": {"title": "Launch status", "body": "Lantern 1.0 ships 2026-12-03; blocked on issue 42 and the pricing page."},
            "comment-backup-server": {"number": 42, "body": "Priya, please try the backup timestamp server before the certificate expires."},
            "schedule-cert-renewal": {"title": "Renew signing certificate", "day": "2026-11-27", "time": "10:00"},
            "schedule-after-free": {"title": "Launch checklist review", "day": "2026-11-20", "time": "17:00"},
            "note-q3": {"title": "Q3 GPU spend", "body": "Jul 1640, Aug 2080, Sep 2440 USD: 6160 USD in all."},
            "assign-57": {"number": 57, "body": "The October outage retro's action item is this issue; Dev should own it."},
            "week-plan": {"title": "Fix resume crash", "day": "2026-11-20", "time": "17:00"},
        });
        let context = crate::tool::ToolCtx {
            allowed: Vec::new(),
            data_dir: std::env::temp_dir(),
            source: "workflow".into(),
        };
        let mut writes = 0;
        for r in &rows {
            let tools = step_tools(&fixture, &r.servers, Default::default())();
            let last = r.calls.last().unwrap();
            let tool = &tools
                .iter()
                .find(|t| t.tool.spec().name == *last)
                .unwrap()
                .tool;
            if tool.spec().permission == crate::tool::Permission::Read {
                continue;
            }
            writes += 1;
            let reply = tool.call(&wrote[&r.id], &context).unwrap().to_lowercase();
            for word in &r.answer {
                assert!(reply.contains(&word.to_lowercase()), "{}: {reply}", r.id);
            }
        }
        assert_eq!(writes, 7);
    }

    #[test]
    fn a_run_is_stuck_only_when_nothing_will_move_it() {
        use larch_task::graph::{Graph, Node, NodeKind};
        use larch_task::types::{ChildFailure, ChildFailureKind, ChildWait};
        let node = Node {
            when: None,
            optional: Vec::new(),
            id: "a".into(),
            depends_on: vec![],
            kind: NodeKind::Activity,
            inputs: Default::default(),
            outputs: Default::default(),
        };
        let definition = WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "g".into(),
            initial_graph: Some(Graph { nodes: vec![node] }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: Default::default(),
            acceptance: vec!["done".into()],
        };
        let ready = RunState::new("run".into(), definition).unwrap();
        assert_eq!(stuck(&ready), None, "work left");
        let mut settled = ready.clone();
        settled.completed_nodes.insert("a".into());
        assert_eq!(stuck(&settled), None, "a done-when check at work");
        let mut helping = ready.clone();
        helping.status = RunStatus::Waiting;
        helping.waiting_child = Some(ChildWait {
            child_id: "run/a".into(),
            node_id: "a".into(),
            graph_revision: 0,
            origin_attempt_id: None,
            origin_epoch: None,
        });
        // Even with an earlier try's failure on record.
        helping.attempt_failure = Some("an earlier try failed".into());
        assert_eq!(stuck(&helping), None, "a helper still working");
        let mut helper_failed = helping.clone();
        helper_failed.child_failure = Some(ChildFailure {
            child_id: "run/a".into(),
            kind: ChildFailureKind::FailedAttempt,
            reason: Some("no such note".into()),
        });
        assert_eq!(stuck(&helper_failed).as_deref(), Some("no such note"));
        let mut failed = ready.clone();
        failed.status = RunStatus::Waiting;
        failed.attempt_failure = Some("tool dispatch: Uncertain".into());
        assert_eq!(stuck(&failed).as_deref(), Some("tool dispatch: Uncertain"));
        failed.waiting_decision = Some("ask".into());
        assert_eq!(stuck(&failed), None, "a question for the user");
    }
}
