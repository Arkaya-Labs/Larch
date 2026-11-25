use chrono::{TimeZone, Utc};
use larch_task::{
    schedule::{Policy, Schedule},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::collections::BTreeMap;
#[test]
fn occurrence_claim_commits_start_and_outbox_ack_once_across_reopen() {
    let file =
        std::env::temp_dir().join(format!("larch-claim-{}-{}.sqlite", std::process::id(), {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }));
    let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "plan".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let outbox;
    let command = CommandEnvelope {
        id: "start".into(),
        expected_revision: 0,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "attempt".into(),
                node_id: "__planning__".into(),
                epoch: 1,
            },
        },
    };
    {
        let mut s = SqliteStore::open(&file).unwrap();
        s.create_schedule(
            "schedule",
            &Schedule::Once { at: now },
            Policy::default(),
            &definition,
            now,
        )
        .unwrap();
        s.enable_schedule("schedule", 0, true, now).unwrap();
        s.pump_due(now, 10).unwrap();
        outbox = s.pending_outbox(1).unwrap().pop().unwrap();
        assert!(s.acknowledge_outbox(&outbox.id).is_err());
        assert!(
            s.claim_scheduled_start_once(&outbox.id, &command)
                .unwrap()
                .1
        );
    }
    {
        let mut s = SqliteStore::open(&file).unwrap();
        assert!(s.pending_outbox(10).unwrap().is_empty());
        assert!(
            !s.claim_scheduled_start_once(&outbox.id, &command)
                .unwrap()
                .1
        );
        let state = s.get(&outbox.payload).unwrap().unwrap();
        assert_eq!(state.revision, 1);
        assert_eq!(state.attempt.unwrap().id, "attempt");
    }
    let _ = std::fs::remove_file(file);
}
