//! Standalone tests for the native task presentation model.
#[path = "tasks_model.rs"]
mod tasks_model;
use tasks_model::{
    Attempt, BlockState, CapabilitySet, Category, Control, Draft, Host, InFlight, Model, Record,
    Start, StepNode, StepState, Tab,
};

fn record(id: &str, revision: u64, status: &str) -> Record {
    Record {
        id: id.into(),
        revision,
        status: status.into(),
        goal: "Find evidence".into(),
        nodes: vec![step("collect", &[]), step("summarize", &["collect"])],
        has_initial_graph: true,
        startable_steps: true,
        planning: "adaptive".into(),
        scope: vec![],
        acceptance: vec![],
        result: None,
        // The service's sections, as TaskCard::from assigns them.
        category: match status {
            "waiting" => Category::NeedsYou,
            "succeeded" | "failed" | "cancelled" => Category::History,
            _ => Category::Running,
        },
        recovery: None,
        decision: None,
        skipped: vec![],
        attempt_failure: None,
        unresolved_effects: false,
        started: vec![],
        completed: vec![],
        results: Default::default(),
        origin: None,
    }
}

fn step(id: &str, depends_on: &[&str]) -> StepNode {
    StepNode {
        id: id.into(),
        depends_on: depends_on.iter().map(|d| (*d).into()).collect(),
        kind: "step",
    }
}

#[test]
fn sections_and_workflow_card_keep_the_shared_definition() {
    let mut model = Model::default();
    model.replace(vec![
        record("one", 2, "running"),
        record("two", 3, "waiting"),
        record("three", 4, "succeeded"),
    ]);
    let now = model.in_tab(Tab::Now);
    assert_eq!(
        now.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert_eq!(now[0].goal, "Find evidence");
    assert_eq!(
        now[0]
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect::<Vec<_>>(),
        ["collect", "summarize"]
    );
    assert_eq!(model.in_tab(Tab::Done)[0].id, "three");
    assert!(model.in_tab(Tab::Scheduled).is_empty());
    assert_eq!((model.count(Tab::Now), model.count(Tab::Done)), (2, 1));
}

#[test]
fn a_finished_run_waits_on_now_until_the_user_files_it_done() {
    let ids = |m: &Model, tab| {
        m.in_tab(tab)
            .iter()
            .map(|r| r.id.clone())
            .collect::<Vec<_>>()
    };
    let mut model = Model::default();
    // Runs that ended before the first snapshot are already filed.
    model.replace(vec![
        record("old", 2, "succeeded"),
        record("run", 1, "running"),
    ]);
    let ended = vec![
        record("old", 2, "succeeded"),
        record("run", 2, "succeeded"),
        record("bad", 2, "failed"),
        record("stop", 2, "cancelled"),
    ];
    model.replace(ended.clone());
    assert_eq!(ids(&model, Tab::Now), ["run", "bad"]);
    assert_eq!(ids(&model, Tab::Done), ["old", "stop"]);
    assert_eq!(model.headline(&ended[1]), "Ready for review");
    assert_eq!(model.headline(&ended[2]), "Failed");
    model.mark_done("run");
    // A resync keeps what the user filed rather than seeding again.
    model.resync();
    model.replace(ended);
    assert_eq!(ids(&model, Tab::Now), ["bad"]);
    assert_eq!(ids(&model, Tab::Done), ["old", "run", "stop"]);
}

#[test]
fn authoring_and_actions_never_infer_autonomous_grants() {
    let goal = Draft::new("Find evidence", vec![], "initial").unwrap();
    let steps = Draft::new("Find evidence", vec!["collect".into()], "disabled").unwrap();
    assert_eq!(goal.goal, steps.goal);
    assert!(Draft::new("", vec![], "initial").is_err());
    assert!(Draft::new("goal", vec![], "disabled").is_err());
    // Each non-empty line of "done when" is one criterion, trimmed.
    let done = goal.done_when("  cites each paper \n\n names a winner");
    assert_eq!(done.done_when, ["cites each paper", "names a winner"]);
    assert!(!CapabilitySet::default().autonomous_application);
    assert!(
        tasks_model::command(
            &record("one", 8, "running"),
            "cmd-1",
            Control::Pause,
            CapabilitySet::default()
        )
        .is_err()
    );
    let caps = CapabilitySet {
        controls: true,
        ..CapabilitySet::default()
    };
    let command =
        tasks_model::command(&record("one", 8, "running"), "cmd-1", Control::Pause, caps).unwrap();
    assert_eq!(command.expected_revision, 8);
    assert_eq!(command.command_id, "cmd-1");
}

#[test]
fn host_errors_and_reconnect_do_not_discard_the_last_snapshot() {
    struct Fake {
        fail: bool,
    }
    impl Host for Fake {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            if self.fail {
                Err("service unavailable".into())
            } else {
                Ok(vec![record("one", 2, "running")])
            }
        }
    }
    let mut model = Model::default();
    model.refresh(&Fake { fail: false }).unwrap();
    assert!(model.refresh(&Fake { fail: true }).is_err());
    assert_eq!(model.in_tab(Tab::Now)[0].revision, 2);
    assert!(model.error().is_some());
}

#[test]
fn stale_snapshot_cannot_replace_a_newer_revision() {
    struct Stale;
    impl Host for Stale {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            Ok(vec![record("one", 1, "running")])
        }
    }
    let mut model = Model::default();
    model.replace(vec![record("one", 3, "running")]);
    assert!(model.refresh(&Stale).is_err());
    assert_eq!(model.in_tab(Tab::Now)[0].revision, 3);
}

/// A settled run can be deleted where the service deletes, but not one still running, one whose
/// effect waits to be checked, or one its saved schedule holds; a deleted run leaves every tab.
#[test]
fn only_a_settled_run_no_schedule_holds_can_be_deleted() {
    use tasks_model::ScheduledRecord;
    let mut model = Model::default();
    let done = record("done", 3, "succeeded");
    let scheduled = Record {
        id: "schedule:5:daily:at".into(),
        origin: Some(("daily".into(), 0)),
        ..done.clone()
    };
    let unsure = Record {
        unresolved_effects: true,
        ..done.clone()
    };
    model.replace(vec![done.clone(), record("live", 1, "running")]);
    assert!(!model.can_delete(&done), "an older service deletes nothing");
    model.capabilities.deletes = true;
    assert!(model.can_delete(&done));
    assert!(!model.can_delete(&record("live", 1, "running")));
    assert!(!model.can_delete(&unsure));
    assert!(model.can_delete(&scheduled), "its schedule is gone");
    model.replace_schedules(vec![ScheduledRecord {
        id: "daily".into(),
        revision: 0,
        goal: "Find evidence".into(),
        steps: vec![],
        planning: "initial".into(),
        scope: vec![],
        acceptance: vec![],
        schedule: "daily".into(),
        next_due: None,
        enabled: false,
        catch_up: "Run the latest one".into(),
        overlap: "Skip this one".into(),
    }]);
    assert!(!model.can_delete(&scheduled));
    model.mark_done("done");
    assert_eq!(model.in_tab(Tab::Done).len(), 1);
    model.forget("done");
    assert!(model.in_tab(Tab::Done).is_empty() && model.in_tab(Tab::Now).len() == 1);
}

#[test]
fn scheduled_items_are_separate_from_active_runs() {
    use tasks_model::ScheduledRecord;
    let mut model = Model::default();
    model.replace(vec![record("run", 3, "running")]);
    model.replace_schedules(vec![ScheduledRecord {
        id: "daily".into(),
        revision: 2,
        goal: "Find evidence".into(),
        steps: vec![],
        planning: "initial".into(),
        scope: vec![],
        acceptance: vec![],
        schedule: "daily".into(),
        next_due: Some("2026-11-13T09:00:00Z".into()),
        enabled: true,
        catch_up: "Run the latest one".into(),
        overlap: "Skip this one".into(),
    }]);
    assert_eq!(model.schedules()[0].goal, "Find evidence");
    assert_eq!(
        model.schedules()[0].next_due.as_deref(),
        Some("2026-11-13T09:00:00Z")
    );
    assert_eq!(model.in_tab(Tab::Now)[0].id, "run");
}

#[test]
fn committed_events_advance_the_cursor_only_after_a_complete_ordered_page() {
    let mut model = Model::default();
    model
        .apply_events("run", &[(1, "created".into()), (2, "paused".into())])
        .unwrap();
    assert_eq!(model.cursor("run"), 2);
    assert!(
        model
            .apply_events("run", &[(4, "cancelled".into())])
            .is_err()
    );
    assert_eq!(model.cursor("run"), 2);
    model.apply_events("run", &[(3, "resumed".into())]).unwrap();
    assert_eq!(model.cursor("run"), 3);
}

/// What a paused card offers under `caps`, with no events yet (started paused, not by the user).
fn start(caps: CapabilitySet, item: &Record) -> Option<Result<Start, &'static str>> {
    let mut model = Model::default();
    model.capabilities = caps;
    model.start(item)
}

/// A paused workflow with a goal and no steps of its own.
fn goal_only() -> Record {
    Record {
        nodes: vec![],
        has_initial_graph: false,
        startable_steps: false,
        ..record("goal", 0, "paused")
    }
}

#[test]
fn old_services_cannot_enable_authoring_or_start() {
    let old = CapabilitySet {
        authoring: true,
        controls: true,
        execution: true,
        planning: true,
        ..CapabilitySet::default()
    };
    assert!(!tasks_model::can_author(old));
    assert_eq!(start(old, &goal_only()), Some(Err(tasks_model::TOO_OLD)));
    let safe = CapabilitySet {
        explicit_start: true,
        ..old
    };
    assert!(tasks_model::can_author(safe));
    assert_eq!(start(safe, &goal_only()), Some(Ok(Start::Goal)));
}

#[test]
fn steps_start_only_where_model_steps_run() {
    let full = CapabilitySet {
        controls: true,
        explicit_start: true,
        execution: true,
        planning: true,
        ..CapabilitySet::default()
    };
    let steps = record("steps", 0, "paused");
    assert_eq!(start(full, &steps), Some(Ok(Start::Steps)));
    // A driver but no model, as a web-search-only service: its plan would fail at the first step.
    let driver_only = CapabilitySet {
        planning: false,
        ..full
    };
    for item in [&steps, &goal_only()] {
        assert_eq!(start(driver_only, item), Some(Err(tasks_model::NO_MODEL)));
    }
    let no_execution = CapabilitySet {
        execution: false,
        ..full
    };
    assert_eq!(
        start(no_execution, &steps),
        Some(Err(tasks_model::NO_MODEL))
    );
    let old = CapabilitySet {
        explicit_start: false,
        ..full
    };
    assert_eq!(start(old, &steps), Some(Err(tasks_model::TOO_OLD)));
    let repeating = Record {
        startable_steps: false,
        ..steps.clone()
    };
    assert_eq!(start(full, &repeating), Some(Err(tasks_model::REPEATING)));
}

#[test]
fn start_offers_nothing_without_controls_or_once_started() {
    let full = CapabilitySet {
        controls: true,
        explicit_start: true,
        execution: true,
        planning: true,
        ..CapabilitySet::default()
    };
    let steps = record("steps", 2, "paused");
    let no_controls = CapabilitySet {
        controls: false,
        ..full
    };
    assert_eq!(start(no_controls, &steps), None, "no hint either");
    assert_eq!(start(no_controls, &goal_only()), None);
    assert_eq!(start(full, &record("steps", 2, "ready")), None);
    let mut model = Model::default();
    model.capabilities = full;
    model
        .apply_events("steps", &[(1, "created".into()), (2, "paused".into())])
        .unwrap();
    assert_eq!(model.start(&steps), None, "Resume answers a user's pause");
}

#[test]
fn a_start_the_workflow_overtook_says_it_refreshed() {
    for stale in ["stale or inactive start request", "revision conflict"] {
        assert_eq!(tasks_model::start_error(stale.into()), tasks_model::CHANGED);
    }
    let other = "activity execution is not configured";
    assert_eq!(tasks_model::start_error(other.into()), other);
}

#[path = "tasks_pending_tests.rs"]
mod tasks_pending_tests;

#[test]
fn reconnect_rejects_snapshot_behind_applied_event_cursor() {
    struct Snapshot(u64);
    impl Host for Snapshot {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            Ok(vec![record("run", self.0, "paused")])
        }
    }
    let mut model = Model::default();
    model.refresh(&Snapshot(1)).unwrap();
    // An event can commit between the snapshot and event-page reads.
    model
        .apply_events("run", &[(1, "started".into()), (2, "paused".into())])
        .unwrap();
    assert!(model.refresh(&Snapshot(1)).is_err());
    assert!(!model.connected);
    assert_eq!(model.cursor("run"), 2);
    assert_eq!(model.in_tab(Tab::Now)[0].revision, 1);
}

#[test]
fn reconnect_accepts_snapshot_caught_up_to_the_event_cursor() {
    struct Snapshot(u64);
    impl Host for Snapshot {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            Ok(vec![record("run", self.0, "paused")])
        }
    }
    let mut model = Model::default();
    model.refresh(&Snapshot(1)).unwrap();
    model
        .apply_events("run", &[(1, "started".into()), (2, "paused".into())])
        .unwrap();
    model.refresh(&Snapshot(2)).unwrap();
    assert!(model.connected);
    assert_eq!(model.cursor("run"), 2);
}

#[test]
fn needs_you_explains_a_persisted_child_failure_without_inventing_a_reason() {
    assert_eq!(
        tasks_model::child_recovery_explanation(
            "waiting",
            "child-7",
            "failed",
            Some("permission denied")
        )
        .as_deref(),
        Some("Child task child-7 failed: permission denied. This task is waiting.")
    );
    assert_eq!(
        tasks_model::child_recovery_explanation(
            "waiting",
            "child-7",
            "failed_attempt",
            Some("permission denied")
        )
        .as_deref(),
        Some("Child task child-7 failed: permission denied. This task is waiting.")
    );
    assert_eq!(
        tasks_model::child_recovery_explanation("waiting", "child-7", "cancelled", None).as_deref(),
        Some("Child task child-7 was cancelled. This task is waiting.")
    );
    assert_eq!(
        tasks_model::child_recovery_explanation("waiting", "child-7", "failed", Some("   "))
            .as_deref(),
        Some("Child task child-7 failed. This task is waiting.")
    );
}

#[test]
fn stale_or_nonterminal_child_outcomes_are_not_shown_as_recovery_failures() {
    assert!(
        tasks_model::child_recovery_explanation("running", "child-7", "failed", Some("old"))
            .is_none()
    );
    assert!(
        tasks_model::child_recovery_explanation("waiting", "child-7", "succeeded", None).is_none()
    );
    assert!(tasks_model::child_recovery_explanation("waiting", "", "failed", None).is_none());
}

#[test]
fn waiting_record_projects_the_persisted_child_failure() {
    let mut item = record("parent", 3, "waiting");
    item.recovery = Some(tasks_model::ChildRecovery {
        child_id: "child-7".into(),
        kind: "failed_attempt".into(),
        reason: Some("permission denied".into()),
    });
    assert_eq!(
        item.recovery_explanation().as_deref(),
        Some("Child task child-7 failed: permission denied. This task is waiting.")
    );
    item.status = "running".into();
    assert!(item.recovery_explanation().is_none());
}

#[test]
fn retry_child_uses_the_committed_failure_and_parent_revision() {
    let mut item = record("parent", 3, "waiting");
    item.recovery = Some(tasks_model::ChildRecovery {
        child_id: "child-7".into(),
        kind: "failed_attempt".into(),
        reason: Some("permission denied".into()),
    });
    let old_service = CapabilitySet {
        controls: true,
        ..CapabilitySet::default()
    };
    assert!(tasks_model::retry_child_request(&item, "retry-1", old_service).is_err());
    let enabled = CapabilitySet {
        retry_child: true,
        ..old_service
    };
    assert!(tasks_model::retry_child_request(&item, "retry-1", CapabilitySet::default()).is_err());
    let command = tasks_model::retry_child_request(&item, "retry-1", enabled).unwrap();
    assert_eq!(command.run_id, "parent");
    assert_eq!(command.child_id, "child-7");
    assert_eq!(command.expected_revision, 3);
    assert_eq!(command.command_id, "retry-1");
    item.status = "running".into();
    assert!(tasks_model::retry_child_request(&item, "retry-2", enabled).is_err());
}

#[test]
fn an_answer_carries_the_exact_digest_and_only_while_the_run_waits() {
    let mut item = record("run", 4, "waiting");
    item.decision = Some(tasks_model::Decision {
        request_id: "deliver:sha256:abc".into(),
        step: "deliver".into(),
        action_digest: "sha256:abc".into(),
        scope: "apply abc to refs/heads/main".into(),
        expires_at: 100,
    });
    let caps = CapabilitySet {
        decisions: true,
        ..CapabilitySet::default()
    };
    assert!(tasks_model::decision_answer(&item, true, "a", CapabilitySet::default(), 10).is_err());
    assert!(
        tasks_model::decision_answer(&item, true, "a", caps, 100).is_err(),
        "expired"
    );
    let decision = item.decision.as_ref().unwrap();
    assert_eq!(decision.expires_in(40), Some(60));
    assert_eq!(decision.expires_in(100), None);
    assert_eq!(decision.expires_in(500), None);
    let answer = tasks_model::decision_answer(&item, false, "a", caps, 10).unwrap();
    assert_eq!(answer.decision.action_digest, "sha256:abc");
    assert_eq!((answer.expected_revision, answer.approve), (4, false));
    item.status = "ready".into();
    assert!(tasks_model::decision_answer(&item, true, "a", caps, 10).is_err());
}

#[test]
fn only_a_run_the_user_paused_offers_resume() {
    let mut model = Model::default();
    // A saved run is paused at revision 0 and commits no event until it starts.
    let item = record("run", 0, "paused");
    model.apply_events("run", &[]).unwrap();
    assert!(!model.user_paused(&item), "saved paused needs a start");
    let item = record("run", 3, "paused");
    model
        .apply_events(
            "run",
            &[
                (1, "resumed".into()),
                (2, "attempt_started".into()),
                (3, "paused".into()),
            ],
        )
        .unwrap();
    assert!(model.user_paused(&item));
    assert!(!model.user_paused(&record("run", 3, "ready")));
}

#[test]
fn pause_then_grant_revocation_or_restore_keeps_resume() {
    let mut model = Model::default();
    model
        .apply_events("run", &[(1, "resumed".into()), (2, "paused".into())])
        .unwrap();
    model
        .apply_events("run", &[(3, "grant_revoked".into())])
        .unwrap();
    assert!(model.user_paused(&record("run", 3, "paused")));
    model
        .apply_events("run", &[(4, "restored_paused".into())])
        .unwrap();
    assert!(model.user_paused(&record("run", 4, "paused")));
    // The same events on a run that never started leave it needing a start.
    model
        .apply_events(
            "saved",
            &[(1, "grant_revoked".into()), (2, "restored_paused".into())],
        )
        .unwrap();
    assert!(!model.user_paused(&record("saved", 2, "paused")));
}

#[test]
fn backup_rollback_is_refused_until_the_user_resyncs() {
    struct Restored;
    impl Host for Restored {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            Ok(vec![record("run", 4, "paused")])
        }
    }
    let mut model = Model::default();
    model.replace(vec![record("run", 10, "running")]);
    let history: Vec<_> = (1..=10).map(|r| (r, "attempt_started".into())).collect();
    model.apply_events("run", &history).unwrap();
    for _ in 0..3 {
        assert!(
            model.refresh(&Restored).is_err(),
            "an older reply is refused"
        );
        assert!(model.behind() && model.needs_resync());
        assert_eq!(model.in_tab(Tab::Now)[0].revision, 10);
    }
    model.resync();
    assert!(!model.loaded() && model.cursor("run") == 0);
    model.refresh(&Restored).unwrap();
    assert!(model.connected && !model.behind());
    // The restored history is read again from the start.
    let restored = [
        (1, "resumed".into()),
        (2, "paused".into()),
        (3, "grant_revoked".into()),
        (4, "restored_paused".into()),
    ];
    model.apply_events("run", &restored).unwrap();
    assert!(model.user_paused(&record("run", 4, "paused")));
}

#[test]
fn a_paused_run_whose_history_has_a_gap_offers_no_start() {
    let mut model = Model::default();
    model.capabilities = CapabilitySet {
        controls: true,
        explicit_start: true,
        execution: true,
        planning: true,
        ..CapabilitySet::default()
    };
    let paused = record("paused", 4, "paused");
    model.replace(vec![paused.clone()]);
    assert!(model.start(&paused).is_some());
    assert!(
        model
            .apply_events("paused", &[(1, "created".into()), (4, "paused".into())])
            .is_err()
    );
    assert_eq!(model.start(&paused), None);
}

#[test]
fn an_event_gap_turns_off_only_that_run() {
    let mut model = Model::default();
    model.replace(vec![record("bad", 3, "ready"), record("good", 1, "ready")]);
    assert!(
        model
            .apply_events("bad", &[(1, "resumed".into()), (3, "paused".into())])
            .is_err()
    );
    model
        .apply_events("good", &[(1, "resumed".into())])
        .unwrap();
    assert!(model.run_error("bad").is_some() && model.run_error("good").is_none());
    assert!(model.connected && model.needs_resync() && !model.behind());
    model
        .apply_events(
            "bad",
            &[
                (1, "resumed".into()),
                (2, "paused".into()),
                (3, "resumed".into()),
            ],
        )
        .unwrap();
    assert!(model.run_error("bad").is_none() && !model.needs_resync());
}

#[test]
fn runs_gone_from_the_service_drop_their_history() {
    let mut model = Model::default();
    model.replace(vec![record("old", 1, "ready")]);
    model.apply_events("old", &[(1, "resumed".into())]).unwrap();
    model.replace(vec![record("new", 0, "paused")]);
    assert_eq!(model.cursor("old"), 0);
    assert!(model.recent("old").is_empty());
    assert!(model.cursors().is_empty());
}

#[test]
fn pause_is_offered_where_the_core_takes_it() {
    let decision = tasks_model::Decision {
        request_id: "deliver:1".into(),
        step: "deliver".into(),
        action_digest: "sha256:abc".into(),
        scope: "repo".into(),
        expires_at: 9,
    };
    assert!(record("run", 1, "ready").can_pause());
    let waiting = record("run", 1, "waiting");
    assert!(
        Record {
            decision: Some(decision),
            ..waiting.clone()
        }
        .can_pause()
    );
    assert!(
        Record {
            recovery: Some(tasks_model::ChildRecovery {
                child_id: "child".into(),
                kind: "failed".into(),
                reason: None,
            }),
            ..waiting.clone()
        }
        .can_pause()
    );
    let failed_attempt = Record {
        attempt_failure: Some("tool unavailable".into()),
        ..waiting
    };
    assert!(!failed_attempt.can_pause());
    assert!(!record("run", 1, "paused").can_pause());
    assert!(!record("run", 1, "running").can_pause());
}

#[test]
fn a_run_takes_one_command_until_a_later_refresh_lands() {
    let mut busy = InFlight::default();
    assert!(busy.begin("run"));
    assert!(!busy.begin("run"), "second click while sending");
    busy.refreshed(7);
    assert!(busy.contains("run"), "unanswered command stays claimed");
    busy.answered("run", 7);
    busy.refreshed(7);
    assert!(busy.contains("run"), "refresh 7 may predate the command");
    assert!(busy.begin("other"));
    busy.refreshed(8);
    assert!(!busy.contains("run") && busy.contains("other"));
    assert!(busy.begin("run"));
}

#[test]
fn only_an_ordinary_failed_attempt_offers_retry() {
    let mut item = record("run", 5, "waiting");
    assert!(!item.can_retry(), "a wait with no failure");
    item.attempt_failure = Some("tool unavailable".into());
    assert!(item.can_retry());
    let caps = CapabilitySet {
        controls: true,
        ..CapabilitySet::default()
    };
    let retry = tasks_model::command(&item, "r", Control::Resume, caps).unwrap();
    assert_eq!(
        (retry.expected_revision, retry.control),
        (5, Control::Resume)
    );
    let blocked = [
        Record {
            unresolved_effects: true,
            ..item.clone()
        },
        Record {
            decision: Some(tasks_model::Decision {
                request_id: "apply:1".into(),
                step: "deliver".into(),
                action_digest: "sha256:abc".into(),
                scope: "repo".into(),
                expires_at: 9,
            }),
            ..item.clone()
        },
        Record {
            recovery: Some(tasks_model::ChildRecovery {
                child_id: "child".into(),
                kind: "failed".into(),
                reason: None,
            }),
            ..item.clone()
        },
        // Resume leaves the reason on the record until the next attempt starts.
        Record {
            status: "ready".into(),
            ..item.clone()
        },
        Record {
            status: "paused".into(),
            attempt_failure: None,
            ..item.clone()
        },
    ];
    for record in blocked {
        assert!(!record.can_retry(), "{record:?}");
    }
}

#[test]
fn steps_read_their_state_from_the_committed_run() {
    let mut run = Record {
        started: vec!["collect".into(), "summarize".into()],
        completed: vec!["collect".into()],
        ..record("r", 4, "running")
    };
    assert_eq!(run.progress(), [StepState::Done, StepState::Running]);
    assert_eq!(run.headline(), "Running · step 2 of 2");
    assert_eq!(run.focus_step(None), Some("summarize"));
    assert_eq!(run.focus_step(Some("collect")), Some("collect"));
    assert_eq!(run.focus_step(Some("gone")), Some("summarize"));
    run.status = "waiting".into();
    run.attempt_failure = Some("timed out".into());
    assert_eq!(run.step_state("summarize"), StepState::Failed);
    assert_eq!(run.headline(), "Step failed");
    run.decision = Some(tasks_model::Decision {
        request_id: "q".into(),
        step: "summarize".into(),
        action_digest: "d".into(),
        scope: "s".into(),
        expires_at: 9,
    });
    assert_eq!(run.step_state("summarize"), StepState::NeedsYou);
    run.status = "cancelled".into();
    assert_eq!(run.progress(), [StepState::Done, StepState::Stopped]);
    run.skipped = vec!["summarize".into()];
    assert_eq!(run.step_state("summarize"), StepState::Skipped);
    let fresh = record("p", 0, "paused");
    assert_eq!(fresh.progress(), [StepState::Upcoming, StepState::Upcoming]);
    assert_eq!(fresh.waits_on("summarize"), 1);
    assert_eq!(fresh.focus_step(None), Some("collect"));
}

#[test]
fn layers_put_each_step_below_its_longest_chain_of_dependencies() {
    let run = Record {
        // Listed out of order, with a join after two parallel reads and a dangling dependency.
        nodes: vec![
            step("write", &["read_a", "read_b"]),
            step("read_b", &["find"]),
            step("find", &["missing"]),
            step("read_a", &["find"]),
            step("save", &["write", "find"]),
        ],
        ..record("r", 1, "running")
    };
    assert_eq!(run.layers(), vec![vec![2], vec![1, 3], vec![0], vec![4]]);
    assert!(
        Record {
            nodes: vec![],
            ..run
        }
        .layers()
        .is_empty()
    );
}

#[test]
fn the_center_opens_the_selected_run_or_the_tabs_first_and_a_schedules_runs_newest_first() {
    let mut model = Model::default();
    let from = |id: &str, at| Record {
        origin: Some(("digest".into(), at)),
        ..record(id, 1, "succeeded")
    };
    model.replace(vec![
        record("a", 1, "running"),
        record("b", 1, "waiting"),
        from("d1", 100),
        from("d2", 200),
    ]);
    assert_eq!(model.current().map(|r| r.id.as_str()), Some("a"));
    model.selected = Some("b".into());
    assert_eq!(model.current().map(|r| r.id.as_str()), Some("b"));
    model.tab = Tab::Done;
    assert_eq!(
        model.current().map(|r| r.id.as_str()),
        Some("d1"),
        "b is not on Done"
    );
    let runs: Vec<_> = model
        .runs_of("digest")
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(runs, ["d2", "d1"]);
    assert!(model.runs_of("other").is_empty());
}

#[test]
fn a_calendar_block_shows_its_run_or_whether_it_is_still_to_come() {
    use tasks_model::block_state;
    assert_eq!(
        block_state(true, 10, 20, Some("succeeded")),
        BlockState::Ran
    );
    assert_eq!(
        block_state(true, 10, 20, Some("failed")),
        BlockState::Stopped
    );
    assert_eq!(
        block_state(false, 10, 20, Some("cancelled")),
        BlockState::Stopped
    );
    assert_eq!(
        block_state(true, 10, 20, Some("waiting")),
        BlockState::Running
    );
    assert_eq!(block_state(true, 10, 20, None), BlockState::Missed);
    assert_eq!(block_state(true, 30, 20, None), BlockState::Upcoming);
    assert_eq!(block_state(false, 30, 20, None), BlockState::Paused);
}

#[test]
fn activity_keeps_only_the_latest_committed_events() {
    let mut model = Model::default();
    model.replace(vec![record("r", 9, "running")]);
    let events: Vec<_> = (1..=9).map(|n| (n, format!("e{n}"))).collect();
    model.apply_events("r", &events).unwrap();
    let recent: Vec<_> = model.recent("r").iter().map(|(n, _)| *n).collect();
    assert_eq!(
        recent,
        (10 - tasks_model::RECENT_EVENTS as u64..10).collect::<Vec<_>>()
    );
}

#[test]
fn a_typed_time_of_day_reads_in_either_clock() {
    use tasks_model::parse_clock;
    for (text, want) in [
        ("7:45", (7, 45)),
        ("07:45", (7, 45)),
        ("7:45 am", (7, 45)),
        ("7:45 PM", (19, 45)),
        ("6pm", (18, 0)),
        ("12 am", (0, 0)),
        ("12:30 pm", (12, 30)),
        ("18:30", (18, 30)),
        (" 0:05 ", (0, 5)),
    ] {
        assert_eq!(parse_clock(text), Ok(want), "{text}");
    }
    for text in ["", "24:00", "7:60", "13 pm", "0 am", "noon", "7:4x"] {
        assert!(parse_clock(text).is_err(), "{text}");
    }
    assert_eq!(
        parse_clock(" "),
        Err("Set the time of day it runs, such as 7:45 am or 18:30.".into())
    );
}

#[test]
fn a_steps_attempts_add_up_and_its_rate_needs_every_count() {
    let attempt = |step: &str, start: i64, end: i64, tokens| Attempt {
        step: step.into(),
        started_ms: start,
        finished_ms: end,
        tokens,
    };
    let mut model = Model::default();
    model.replace(vec![record("run", 3, "running")]);
    model.set_attempts(
        "run",
        vec![
            // A failed try, then the retry: both count.
            attempt("collect", 0, 2_000, Some((100, 20))),
            attempt("collect", 5_000, 9_000, Some((150, 40))),
            attempt("summarize", 10_000, 10_400, None),
            // A clock that stepped back reads as no time.
            attempt("summarize", 20_000, 19_000, Some((10, 5))),
        ],
    );
    let t = model.timings("run");
    let collect = t["collect"];
    assert_eq!(
        (
            collect.attempts,
            collect.millis,
            collect.prompt,
            collect.completion
        ),
        (2, 6_000, 250, 60)
    );
    assert_eq!(collect.rate(), Some(10.));
    assert_eq!(
        (
            t["summarize"].millis,
            t["summarize"].counted,
            t["summarize"].rate()
        ),
        (400, false, None)
    );
    assert!(model.timings("other").is_empty());
    model.set_names(
        "run",
        [("s1".to_string(), "Collect the notes".to_string())].into(),
    );
    assert_eq!(
        (model.name("run", "s1"), model.name("run", "s2")),
        ("Collect the notes", "s2")
    );
    // A run gone from the service takes its attempts and names with it.
    model.replace(vec![]);
    assert!(model.timings("run").is_empty());
    assert_eq!(model.name("run", "s1"), "s1");
}

#[test]
fn a_steps_time_reads_in_its_largest_units() {
    use tasks_model::duration;
    assert_eq!(duration(400), "0.4 s");
    assert_eq!(duration(9_960), "10.0 s");
    assert_eq!(duration(38_000), "38 s");
    assert_eq!(duration(120_000), "2 min");
    assert_eq!(duration(125_000), "2 min 5 s");
    assert_eq!(duration(3_780_000), "1 h 3 min");
}

#[test]
fn a_run_that_newly_needs_the_user_or_fails_is_told_once() {
    let mut model = Model::default();
    assert!(
        model.alerts(&[record("a", 1, "waiting")]).is_empty(),
        "not on the first load"
    );
    model.replace(vec![record("a", 1, "running"), record("b", 1, "running")]);
    let next = [record("a", 2, "waiting"), record("b", 2, "failed")];
    let alerts = model.alerts(&next);
    assert_eq!(
        alerts,
        vec![
            ("A task needs you", "Find evidence".to_owned()),
            ("A task stopped on an error", "Find evidence".to_owned())
        ]
    );
    model.replace(next.to_vec());
    assert!(
        model
            .alerts(&[record("a", 3, "waiting"), record("b", 2, "failed")])
            .is_empty()
    );
}

#[test]
fn a_refused_snapshot_alerts_once_not_on_every_poll() {
    struct Polls(std::cell::Cell<usize>);
    impl Host for Polls {
        fn snapshot(&self) -> Result<Vec<Record>, String> {
            let n = self.0.replace(self.0.get() + 1);
            // b failed, but the first polls carry a restored backup of a, so they are refused.
            let a = if n < 3 { 4 } else { 6 };
            Ok(vec![record("a", a, "running"), record("b", 2, "failed")])
        }
    }
    let mut model = Model::default();
    model.replace(vec![record("a", 5, "running"), record("b", 1, "running")]);
    let host = Polls(std::cell::Cell::new(0));
    let told: Vec<_> = (0..4)
        .flat_map(|_| model.refresh(&host).unwrap_or_default())
        .collect();
    assert_eq!(
        told,
        vec![("A task stopped on an error", "Find evidence".to_owned())]
    );
    assert!(!model.behind(), "the last poll was taken");
}

/// An edit opened before the schedule ran saves on its newer revision; any other change since,
/// someone's edit, is refused rather than overwritten.
#[test]
fn an_edit_open_across_a_run_saves_but_not_across_an_edit() {
    use tasks_model::{ScheduledRecord, edit_revision};
    let opened = ScheduledRecord {
        id: "daily".into(),
        revision: 2,
        goal: "Find evidence".into(),
        steps: vec!["collect".into()],
        planning: "initial".into(),
        scope: vec![],
        acceptance: vec![],
        schedule: "Every day at 09:00".into(),
        next_due: Some("2026-11-13T09:00:00Z".into()),
        enabled: true,
        catch_up: "Run the latest one".into(),
        overlap: "Skip this one".into(),
    };
    assert_eq!(edit_revision(&opened, &opened), Some(2));
    let ran = ScheduledRecord {
        revision: 3,
        next_due: Some("2026-11-14T09:00:00Z".into()),
        ..opened.clone()
    };
    assert_eq!(edit_revision(&opened, &ran), Some(3));
    let edited = ScheduledRecord {
        goal: "Find more evidence".into(),
        ..ran.clone()
    };
    assert_eq!(edit_revision(&opened, &edited), None);
    let off = ScheduledRecord {
        enabled: false,
        ..ran
    };
    assert_eq!(edit_revision(&opened, &off), None);
}
