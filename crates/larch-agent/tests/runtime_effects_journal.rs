use larch_agent::{
    runtime::{
        effects::{Declaration, DispatchError, TaskJournal, dispatch},
        permissions::{LiveGrants, Scope},
    },
    tool::{Permission, Tool, ToolCtx, ToolError, ToolSpec},
};
use larch_task::{
    store::SqliteStore,
    types::{Attempt, Command, CommandEnvelope, RunState, WorkflowDefinition},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Counter {
    calls: AtomicUsize,
    fails: bool,
}
impl Tool for Counter {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "count".into(),
            description: "test".into(),
            schema: json!({"type":"object", "properties":{"value":{"type":"string"}}, "required":["value"], "additionalProperties":false}),
            permission: Permission::WriteExternal,
            untrusted_results: true,
            max_result_tokens: 64,
            max_args_tokens: 64,
        }
    }
    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fails {
            Err(ToolError::Invalid("unknown outcome".into()))
        } else {
            Ok("done".into())
        }
    }
}
struct Decl;
impl Declaration for Decl {
    fn tool_name(&self) -> &str {
        "count"
    }
    fn capability(&self, _: &Value, _: &ToolCtx) -> Result<String, String> {
        Ok("write:test".into())
    }
    fn max_argument_bytes(&self) -> usize {
        1000
    }
    fn max_result_bytes(&self) -> usize {
        1000
    }
}
fn scope() -> Scope {
    Scope(BTreeSet::from(["write:test".to_owned()]))
}
fn setup(path: &std::path::Path) -> SqliteStore {
    let mut store = SqliteStore::open(path).unwrap();
    let definition: WorkflowDefinition = serde_json::from_value(
        json!({"version":1,"goal":"test","planning_policy":"initial","grants":["write:test"]}),
    )
    .unwrap();
    store
        .create(&RunState::new("run".into(), definition).unwrap())
        .unwrap();
    store
        .execute(
            "run",
            &CommandEnvelope {
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
            },
        )
        .unwrap();
    store
}
fn ctx() -> ToolCtx {
    ToolCtx {
        allowed: vec![],
        data_dir: std::env::temp_dir(),
        source: "test".into(),
    }
}
#[test]
fn durable_receipt_replays_without_call_and_conflicting_arguments_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("task.sqlite");
    let mut store = setup(&path);
    let grants = LiveGrants::new(scope());
    let tool = Counter {
        calls: AtomicUsize::new(0),
        fails: false,
    };
    let first = {
        let mut journal = TaskJournal {
            store: &mut store,
            run_id: "run",
            attempt_id: "attempt",
            epoch: 1,
        };
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":"a"}),
            &ctx(),
            "effect",
        )
        .unwrap()
    };
    assert!(first.untrusted);
    drop(store);
    let mut store = SqliteStore::open(path).unwrap();
    let mut journal = TaskJournal {
        store: &mut store,
        run_id: "run",
        attempt_id: "attempt",
        epoch: 1,
    };
    assert_eq!(
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":"a"}),
            &ctx(),
            "effect"
        )
        .unwrap(),
        first
    );
    assert_eq!(
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":"b"}),
            &ctx(),
            "effect"
        ),
        Err(DispatchError::Conflict)
    );
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
}
#[test]
fn revoked_or_invalid_calls_never_execute_and_an_error_is_not_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = setup(&dir.path().join("db"));
    let grants = LiveGrants::new(scope());
    let tool = Counter {
        calls: AtomicUsize::new(0),
        fails: true,
    };
    let mut journal = TaskJournal {
        store: &mut store,
        run_id: "run",
        attempt_id: "attempt",
        epoch: 1,
    };
    assert!(matches!(
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":7}),
            &ctx(),
            "invalid"
        ),
        Err(DispatchError::Rejected(_))
    ));
    grants.replace(Scope::default()).unwrap();
    assert!(matches!(
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":"a"}),
            &ctx(),
            "revoked"
        ),
        Err(DispatchError::Rejected(_))
    ));
    grants.replace(scope()).unwrap();
    for _ in 0..2 {
        assert_eq!(
            dispatch(
                &mut journal,
                &grants,
                &scope(),
                &tool,
                &Decl,
                &json!({"value":"a"}),
                &ctx(),
                "effect"
            ),
            Err(DispatchError::Uncertain)
        );
    }
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn admitted_effect_receipt_is_reconciled_after_cancel() {
    use larch_agent::runtime::effects::{EffectJournal, Intent, Receipt, ReplayClass, Reservation};
    use larch_task::types::EffectStatus;
    let dir = tempfile::tempdir().unwrap();
    let mut store = setup(&dir.path().join("db"));
    let intent = Intent {
        effect_id: "effect".into(),
        tool: "count".into(),
        args_digest: "digest".into(),
        capability: "write:test".into(),
        replay: ReplayClass::Opaque,
        provider_key: None,
    };
    {
        let mut journal = TaskJournal {
            store: &mut store,
            run_id: "run",
            attempt_id: "attempt",
            epoch: 1,
        };
        assert_eq!(journal.reserve(&intent).unwrap(), Reservation::New);
    }
    let state = store.get("run").unwrap().unwrap();
    store
        .execute(
            "run",
            &CommandEnvelope {
                id: "cancel".into(),
                expected_revision: state.revision,
                command: Command::Cancel,
            },
        )
        .unwrap();
    let receipt = Receipt {
        result: Ok("completed".into()),
        untrusted: false,
    };
    let mut journal = TaskJournal {
        store: &mut store,
        run_id: "run",
        attempt_id: "attempt",
        epoch: 1,
    };
    journal.receipt("effect", &receipt).unwrap();
    assert!(matches!(
        &store.get("run").unwrap().unwrap().effects["effect"].status,
        EffectStatus::Completed { .. }
    ));
}

#[test]
fn durable_revocation_blocks_effect_even_when_process_grants_are_stale() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = setup(&dir.path().join("db"));
    let state = store.get("run").unwrap().unwrap();
    store
        .execute(
            "run",
            &CommandEnvelope {
                id: "revoke".into(),
                expected_revision: state.revision,
                command: Command::RevokeGrant {
                    grant: "write:test".into(),
                },
            },
        )
        .unwrap();
    let grants = LiveGrants::new(scope());
    let tool = Counter {
        calls: AtomicUsize::new(0),
        fails: false,
    };
    let mut journal = TaskJournal {
        store: &mut store,
        run_id: "run",
        attempt_id: "attempt",
        epoch: 1,
    };
    assert!(
        dispatch(
            &mut journal,
            &grants,
            &scope(),
            &tool,
            &Decl,
            &json!({"value":"a"}),
            &ctx(),
            "e"
        )
        .is_err()
    );
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn only_a_read_only_tool_may_declare_itself_replay_safe() {
    struct PureDecl;
    impl Declaration for PureDecl {
        fn tool_name(&self) -> &str {
            "count"
        }
        fn capability(&self, _: &Value, _: &ToolCtx) -> Result<String, String> {
            Ok("write:test".into())
        }
        fn replay_class(&self) -> larch_agent::runtime::effects::ReplayClass {
            larch_agent::runtime::effects::ReplayClass::Pure
        }
        fn max_argument_bytes(&self) -> usize {
            1000
        }
        fn max_result_bytes(&self) -> usize {
            1000
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let mut store = setup(&dir.path().join("task.sqlite"));
    let tool = Counter {
        calls: AtomicUsize::new(0),
        fails: true,
    };
    let mut journal = TaskJournal {
        store: &mut store,
        run_id: "run",
        attempt_id: "attempt",
        epoch: 1,
    };
    // A write that failed may still have landed, so it cannot pass its error off as a receipt.
    assert!(matches!(
        dispatch(
            &mut journal,
            &LiveGrants::new(scope()),
            &scope(),
            &tool,
            &PureDecl,
            &json!({"value":"a"}),
            &ctx(),
            "e"
        ),
        Err(DispatchError::Rejected(_))
    ));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn web_search_requires_its_explicit_network_capability() {
    use larch_agent::runtime::effects::{WebSearchDeclaration, web_search_tool};
    let tool = web_search_tool();
    assert_eq!(tool.spec().name, "web_search");
    assert_eq!(
        WebSearchDeclaration
            .capability(&json!({"query":"test"}), &ctx())
            .unwrap(),
        "network.web_search"
    );
    assert_eq!(WebSearchDeclaration.tool_name(), "web_search");
}

#[test]
fn fetch_page_is_a_read_only_step_tool() {
    let tool = larch_agent::runtime::effects::fetch_page_tool();
    assert_eq!(tool.spec().name, "fetch_page");
    assert_eq!(tool.spec().permission, larch_agent::tool::Permission::Read);
}
