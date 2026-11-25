//! The host hands watch notifications to the app: claim before showing, resolve after, and a
//! crash in between leaves the notification visible as uncertain instead of lost or resent.
use chrono::{Duration, TimeZone, Utc};
use larch_task::{
    host::{self, Action},
    runtime::ActorHandle,
    store::SqliteStore,
    watch::{NOTIFICATION_LEASE_SECONDS, NotificationClaim, PendingNotification, Truth},
};

fn file(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "larch-transport-{name}-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
fn call(actor: &ActorHandle, action: Action) -> serde_json::Value {
    let (reply, _) = host::handle(actor, action, true, false);
    assert_eq!(reply.error, None);
    reply.result.unwrap()
}
fn refused(actor: &ActorHandle, action: Action) -> bool {
    host::handle(actor, action, true, false).0.error.is_some()
}
fn pending(actor: &ActorHandle) -> Vec<PendingNotification> {
    serde_json::from_value(call(actor, Action::Notifications { limit: 10 })).unwrap()
}
fn claim(id: &str, transport: &str) -> Action {
    Action::ClaimNotification {
        id: id.into(),
        transport: transport.into(),
    }
}
/// Raise one notification per id, in this order.
fn raise(s: &mut SqliteStore, ids: &[&str]) {
    s.create_watch("w").unwrap();
    for (version, id) in ids.iter().enumerate() {
        s.update_watch_notify("w", version as u64, Truth::True, None, id, "price dropped")
            .unwrap();
    }
}

#[test]
fn notifications_flow_through_the_host_and_survive_a_crash_as_uncertain() {
    let file = file("flow");
    {
        let mut s = SqliteStore::open(&file).unwrap();
        raise(&mut s, &["n1"]);
        drop(s);
        let actor = ActorHandle::start(&file, 16).unwrap();
        let item = PendingNotification {
            id: "n1".into(),
            payload: "price dropped".into(),
            uncertain: false,
            claimed_by: None,
        };
        assert_eq!(pending(&actor), vec![item]);
        let first: NotificationClaim =
            serde_json::from_value(call(&actor, claim("n1", "app"))).unwrap();
        assert!(!first.may_repeat);
        // A second window, another transport, is refused while the claim's lease holds.
        assert!(refused(&actor, claim("n1", "window-2")));
    }
    // The app crashed between showing and resolving.
    let actor = ActorHandle::start(&file, 16).unwrap();
    let listed = pending(&actor);
    assert!(listed[0].uncertain);
    assert_eq!(listed[0].claimed_by.as_deref(), Some("app"));
    // Only the owner resolves; after a restart it reclaims and shows it as a possible repeat.
    let resolve = |transport: &str| Action::ResolveNotification {
        id: "n1".into(),
        transport: transport.into(),
        receipt: Some("shown".into()),
    };
    assert!(refused(&actor, resolve("window-2")));
    let again: NotificationClaim =
        serde_json::from_value(call(&actor, claim("n1", "app"))).unwrap();
    assert!(again.may_repeat);
    call(&actor, resolve("app"));
    assert!(pending(&actor).is_empty());
    drop(actor);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_lapsed_lease_passes_the_claim_on_and_an_explicit_absence_allows_a_clean_retry() {
    let file = file("lease");
    let mut s = SqliteStore::open(&file).unwrap();
    raise(&mut s, &["n1"]);
    let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    assert!(
        s.resolve_notification("n1", "app", None).is_err(),
        "a pending notification has no claim to resolve"
    );
    s.claim_notification("n1", "app", t0).unwrap();
    assert!(
        s.claim_notification("n1", "cli", t0 + Duration::seconds(30))
            .is_err()
    );
    let lapsed = t0 + Duration::seconds(NOTIFICATION_LEASE_SECONDS);
    assert!(
        s.claim_notification("n1", "cli", lapsed)
            .unwrap()
            .may_repeat
    );
    assert!(
        s.resolve_notification("n1", "app", Some("late".into()))
            .is_err(),
        "the old owner lost the claim"
    );
    // The new owner knows it was not shown: back to pending, and the next claim is not a repeat.
    s.resolve_notification("n1", "cli", None).unwrap();
    assert!(
        !s.claim_notification("n1", "app", lapsed)
            .unwrap()
            .may_repeat
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn uncertain_notifications_never_hide_new_ones_and_listing_follows_raise_order() {
    let file = file("order");
    let mut s = SqliteStore::open(&file).unwrap();
    raise(&mut s, &["a1", "a2", "a3", "z-new", "b-later"]);
    let now = Utc::now();
    for id in ["a1", "a2", "a3"] {
        s.claim_notification(id, "crashed", now).unwrap();
    }
    let ids = |limit| -> Vec<String> {
        s.pending_notifications(limit)
            .unwrap()
            .into_iter()
            .map(|n| n.id)
            .collect()
    };
    assert_eq!(ids(1), vec!["z-new"]);
    assert_eq!(ids(10), vec!["z-new", "b-later", "a1", "a2", "a3"]);
    drop(s);
    let _ = std::fs::remove_file(file);
}
