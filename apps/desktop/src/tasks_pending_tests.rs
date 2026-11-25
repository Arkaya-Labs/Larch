//! Test-first regression for overlapping workflow submissions.
#[path = "tasks_model.rs"]
mod tasks_model;

use tasks_model::{Draft, PendingSubmission};

#[test]
fn older_success_must_not_discard_newer_uncertain_submission() {
    let first = Draft::new("First goal", vec![], "initial").unwrap();
    let second = Draft::new("Second goal", vec![], "initial").unwrap();
    let mut pending = PendingSubmission::default();
    let a = pending.begin(&first, "run-a", "command-a");
    let b = pending.begin(&second, "run-b", "command-b");

    // A completes after B has been submitted. B's response is then lost.
    pending.complete(&a.0, &a.1);
    let retry = pending.begin(&second, "new-run", "new-command");
    assert_eq!(retry, b, "retry must reuse B's durable deduplication keys");

    pending.complete(&b.0, &b.1);
    assert_eq!(
        pending.begin(&second, "run-c", "command-c"),
        ("run-c".into(), "command-c".into())
    );
}

#[test]
fn superseded_failure_and_retry_keep_the_newer_identity() {
    let first = Draft::new("First goal", vec![], "initial").unwrap();
    let second = Draft::new("Second goal", vec![], "initial").unwrap();
    let mut pending = PendingSubmission::default();
    let a = pending.begin(&first, "run-a", "command-a");
    let b = pending.begin(&second, "run-b", "command-b");

    pending.fail(&a.0, &a.1);
    assert_eq!(
        pending.begin(&second, "replacement-run", "replacement-command"),
        b
    );
    pending.fail(&b.0, &b.1);
    assert_eq!(pending.begin(&second, "another-run", "another-command"), b);
}
