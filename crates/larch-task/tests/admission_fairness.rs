use larch_task::admission::{Admission, Request};
#[test]
fn roots_share_one_seat_and_release_on_wait() {
    let mut a = Admission::new(1);
    a.set_budget("a".into(), 100).unwrap();
    a.set_budget("b".into(), 100).unwrap();
    for (id, root) in [("a1", "a"), ("a2", "a"), ("b1", "b")] {
        a.enqueue(Request {
            id: id.into(),
            root: root.into(),
            reserve: 10,
        })
        .unwrap();
    }
    assert_eq!(a.admit().unwrap().id, "a1");
    assert!(a.admit().is_none());
    a.finish("a1", 4).unwrap();
    assert_eq!(a.admit().unwrap().id, "b1");
    a.finish("b1", 2).unwrap();
    assert_eq!(a.admit().unwrap().id, "a2");
    a.finish("a2", 12).unwrap();
    assert_eq!(a.usage("a"), Some((16, 0)));
    assert!(a.finish("a2", 0).is_err());
}
