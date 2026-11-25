use fs2::FileExt;
use larch_task::artifacts::ArtifactStore;
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    sync::mpsc,
    time::Duration,
};
#[test]
fn lock_child() {
    let Some(root) = std::env::var_os("LARCH_LOCK_TEST_ROOT") else {
        return;
    };
    println!("ready");
    std::io::stdout().flush().unwrap();
    ArtifactStore::open(root)
        .unwrap()
        .put(b"protected")
        .unwrap();
    println!("done");
    std::io::stdout().flush().unwrap();
}
#[test]
fn publication_waits_for_another_process_holding_the_artifact_lock() {
    let root =
        std::env::temp_dir().join(format!("larch-process-lock-{}-{}", std::process::id(), {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }));
    fs::create_dir_all(&root).unwrap();
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(".larch-artifacts.lock"))
        .unwrap();
    lock.lock_exclusive().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "lock_child", "--nocapture"])
        .env("LARCH_LOCK_TEST_ROOT", &root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let output = child.stdout.take().unwrap();
    let (send, recv) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(output).lines() {
            let _ = send.send(line.unwrap());
        }
    });
    loop {
        if recv
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .ends_with("ready")
        {
            break;
        }
    }
    assert!(recv.recv_timeout(Duration::from_millis(200)).is_err());
    FileExt::unlock(&lock).unwrap();
    loop {
        if recv
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .ends_with("done")
        {
            break;
        }
    }
    assert!(child.wait().unwrap().success());
    reader.join().unwrap();
    fs::remove_dir_all(root).unwrap();
}
