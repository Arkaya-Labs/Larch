//! Diagnostic retention is bounded and never follows a foreign log symlink.
use larch_setup::log::{Log, MAX_BYTES, clean};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-log-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn rotation_keeps_three_old_logs_and_redacts_secrets_and_control_characters() {
    let t = Temp::new();
    let log = Log::with_token(&t.0, Some("private-token".into()));
    for n in 0..5 {
        fs::write(t.0.join("setup.log"), vec![b'0' + n; MAX_BYTES as usize]).unwrap();
        log.line("private-token\r\n\x1bnext").unwrap();
        assert_eq!(
            fs::read_to_string(t.0.join("setup.log")).unwrap(),
            "[redacted]   next\n"
        );
    }
    for n in 1..=3 {
        let bytes = fs::read(t.0.join(format!("setup.log.{n}"))).unwrap();
        assert_eq!(bytes.len(), MAX_BYTES as usize);
        assert_eq!(bytes[0], b'5' - n as u8);
    }
    assert!(!t.0.join("setup.log.4").exists());
    assert!(!clean("token\nline", Some("token")).contains('\n'));
}
#[cfg(unix)]
#[test]
fn foreign_symlink_is_refused_without_writing_its_target() {
    let t = Temp::new();
    let target = t.0.join("foreign");
    fs::write(&target, "keep").unwrap();
    std::os::unix::fs::symlink(&target, t.0.join("setup.log")).unwrap();
    assert!(Log::with_token(&t.0, None).line("bad").is_err());
    assert_eq!(fs::read_to_string(target).unwrap(), "keep");
}
