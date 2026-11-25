use larch_task::{
    host::{self, Action, LIST_PREVIEW, MAX_FRAME},
    runtime::ActorHandle,
    store::SqliteStore,
    types::{PlanningPolicy, RunState, SCHEMA_VERSION, WorkflowDefinition},
};
use std::collections::BTreeMap;

/// Forty runs of three 20 KB step results: about 2.4 MB whole, past a frame in one `List`;
/// `ListPage` answers each page inside a frame, every run once, results cut to a preview.
#[test]
fn a_long_history_lists_in_pages_inside_a_frame() {
    let dir = std::env::temp_dir().join(format!("larch-list-pages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tasks.sqlite");
    let mut store = SqliteStore::open(&path).unwrap();
    let mut runs = Vec::new();
    for i in 0..40 {
        let def = WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "goal".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        };
        let run = RunState::new(format!("run-{i:02}"), def).unwrap();
        store.create(&run).unwrap();
        runs.push(run);
    }
    drop(store);
    // The results finished steps leave, written as the reducer stores them.
    let db = rusqlite::Connection::open(&path).unwrap();
    for mut run in runs {
        for step in ["a", "b", "c"] {
            run.node_results.insert(step.into(), "é".repeat(10_000));
        }
        let state = serde_json::to_string(&run).unwrap();
        db.execute("UPDATE runs SET state = ?1 WHERE id = ?2", (state, &run.id))
            .unwrap();
    }
    drop(db);
    let actor = ActorHandle::start(&path, 4).unwrap();
    let whole = host::handle(&actor, Action::List, false, false)
        .0
        .result
        .unwrap();
    assert!(serde_json::to_vec(&whole).unwrap().len() > MAX_FRAME);
    let mut seen = Vec::new();
    loop {
        let (reply, _) = host::handle(
            &actor,
            Action::ListPage { offset: seen.len() },
            false,
            false,
        );
        let bytes = serde_json::to_vec(&reply).unwrap();
        assert!(bytes.len() < MAX_FRAME, "{} bytes", bytes.len());
        let page: Vec<RunState> = serde_json::from_value(reply.result.unwrap()).unwrap();
        if page.is_empty() {
            break;
        }
        seen.extend(page);
    }
    assert_eq!(seen.len(), 40);
    assert_eq!(seen[39].id, "run-39");
    assert_eq!(seen[0].node_results["a"].chars().count(), LIST_PREVIEW);
}
