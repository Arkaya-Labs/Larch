// Service and CLI may open a fresh store together; WAL setup must not fail one of them.
#[test]
fn concurrent_first_opens_all_succeed() {
    for i in 0..50 {
        let dir = std::env::temp_dir().join(format!("vt-open-race-{}-{i}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("db.sqlite");
        let hs: Vec<_> = (0..4)
            .map(|_| {
                let p = p.clone();
                std::thread::spawn(move || larch_task::store::SqliteStore::open(p).map(|_| ()))
            })
            .collect();
        for h in hs {
            h.join().unwrap().unwrap();
        }
    }
}
