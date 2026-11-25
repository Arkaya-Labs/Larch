use larch_task::{
    store::SqliteStore,
    watch::{NotificationStatus, Truth},
};
#[test]
fn uncertain_notifications_are_not_resent_and_receipts_survive_restart() {
    let file =
        std::env::temp_dir().join(format!("larch-notify-{}-{}.sqlite", std::process::id(), {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }));
    {
        let mut s = SqliteStore::open(&file).unwrap();
        s.create_watch("watch").unwrap();
        s.update_watch_notify(
            "watch",
            0,
            Truth::Unknown,
            Some("partial".into()),
            "notification",
            "payload",
        )
        .unwrap();
        assert!(s.pending_outbox(10).unwrap().is_empty());
        s.update_watch_notify(
            "watch",
            1,
            Truth::True,
            Some("successful".into()),
            "notification",
            "payload",
        )
        .unwrap();
        assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
        s.claim_notification("notification", "app", chrono::Utc::now())
            .unwrap();
        assert!(s.pending_outbox(10).unwrap().is_empty());
    }
    {
        let mut s = SqliteStore::open(&file).unwrap();
        assert_eq!(
            s.notification_status("notification").unwrap(),
            NotificationStatus::Uncertain
        );
        s.resolve_notification("notification", "app", Some("provider-receipt".into()))
            .unwrap();
        assert_eq!(
            s.notification_status("notification").unwrap(),
            NotificationStatus::Delivered {
                receipt: "provider-receipt".into()
            }
        );
        assert!(
            s.resolve_notification("notification", "app", Some("other".into()))
                .is_err()
        );
    }
    let _ = std::fs::remove_file(file);
}
