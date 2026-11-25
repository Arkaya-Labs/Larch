//! The service-card projection; the model's own tests live in tasks_model_tests.rs.
use crate::tasks::project;
use crate::tasks_model::Category;

fn card(section: &str, status: &str) -> larch_app::task::TaskCard {
    serde_json::from_value(serde_json::json!({
        "version": 1, "runId": "run", "revision": 3, "section": section, "status": status,
        "definition": {"version": 1, "goal": "Find evidence", "planning_policy": "adaptive"},
    }))
    .unwrap()
}

/// A step cut off by a restart has no failure in the core, and its run waits on nothing: Retry is
/// what moves it.
#[test]
fn an_interrupted_step_is_offered_retry() {
    let mut c = card("needsYou", "waiting");
    assert!(!project(c.clone()).can_retry());
    c.interrupted = true;
    assert!(project(c).can_retry());
}

#[test]
fn a_card_keeps_the_service_section_and_maps_status_and_planning_by_name() {
    // The section is the service's, even where the model would have put the status elsewhere.
    let record = project(card("needsYou", "paused"));
    assert_eq!(
        (
            record.category,
            record.status.as_str(),
            record.planning.as_str()
        ),
        (Category::NeedsYou, "paused", "adaptive")
    );
    for (status, key) in [
        ("ready", "ready"),
        ("running", "running"),
        ("waiting", "waiting"),
        ("succeeded", "succeeded"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ] {
        assert_eq!(project(card("running", status)).status, key);
    }
    assert_eq!(
        project(card("history", "failed")).category,
        Category::History
    );
}
