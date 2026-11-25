use larch_task::artifacts::ArtifactStore;
#[test]
fn detects_corruption_and_rejects_paths() {
    let path = std::env::temp_dir().join(format!("larch-artifact-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    let store = ArtifactStore::open(&path).unwrap();
    let digest = store.put(b"data").unwrap();
    assert_eq!(store.read(&digest).unwrap(), b"data");
    std::fs::write(path.join(&digest), b"corrupt").unwrap();
    assert!(store.read(&digest).is_err());
    assert!(store.read("../secret").is_err());
    std::fs::remove_dir_all(path).unwrap();
}
