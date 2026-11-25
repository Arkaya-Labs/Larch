use larch_task::watch::{Truth, WatchState};
#[test]
fn unknown_does_not_advance_cursor_and_stale_updates_fail() {
    let mut state = WatchState::default();
    state.update(0, Truth::True, Some("a".into())).unwrap();
    state.update(1, Truth::Unknown, Some("b".into())).unwrap();
    assert_eq!(state.cursor.as_deref(), Some("a"));
    assert_eq!(state.last_success, Some(true));
    assert!(state.update(1, Truth::False, Some("c".into())).is_err());
    assert_eq!(Truth::Unknown.and(Truth::False), Truth::False);
    assert_eq!(Truth::Unknown.or(Truth::False), Truth::Unknown);
}
