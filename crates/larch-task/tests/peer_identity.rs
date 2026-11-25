#![cfg(unix)]

use larch_task::peer::{authenticate_peer, authenticate_peer_for_uid};
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]
use std::io::{ErrorKind, Read, Write};
use std::{
    fs,
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct SocketDirectory(PathBuf);
impl SocketDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-peer-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for SocketDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A real filesystem-bound Unix socket; no UID changes or credential stubs are used.
fn connected() -> (SocketDirectory, UnixStream, UnixStream) {
    let directory = SocketDirectory::new();
    let listener = UnixListener::bind(directory.0.join("peer.sock")).unwrap();
    let client = UnixStream::connect(directory.0.join("peer.sock")).unwrap();
    let (server, _) = listener.accept().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    server
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    (directory, server, client)
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]
#[test]
fn kernel_peer_credentials_accept_the_current_effective_uid_on_both_ends() {
    let (_directory, server, client) = connected();
    let uid = nix::unistd::geteuid().as_raw();
    authenticate_peer(&server).unwrap();
    authenticate_peer_for_uid(&server, uid).unwrap();
    authenticate_peer(&client).unwrap();
    authenticate_peer_for_uid(&client, uid).unwrap();
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]
#[test]
fn different_expected_uid_is_rejected_without_sending_any_bytes() {
    let (_directory, server, mut client) = connected();
    let uid = nix::unistd::geteuid().as_raw();
    let other_uid = uid.wrapping_add(1);
    assert_ne!(uid, other_uid);
    assert!(authenticate_peer_for_uid(&server, other_uid).is_err());
    client
        .set_read_timeout(Some(Duration::from_millis(150)))
        .unwrap();
    let mut byte = [0u8; 1];
    let error = client.read(&mut byte).unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::WouldBlock | ErrorKind::TimedOut
    ));
    // Rechecking the same kernel connection with its actual UID must still succeed.
    authenticate_peer_for_uid(&server, uid).unwrap();
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]
#[test]
fn checking_peer_identity_does_not_consume_the_request_stream() {
    let (_directory, mut server, mut client) = connected();
    client.write_all(b"pending request").unwrap();
    authenticate_peer(&server).unwrap();
    let mut data = [0u8; 15];
    server.read_exact(&mut data).unwrap();
    assert_eq!(&data, b"pending request");
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
)))]
#[test]
fn unsupported_unix_targets_fail_closed() {
    let (_directory, server, _client) = connected();
    assert!(authenticate_peer(&server).is_err());
    assert!(authenticate_peer_for_uid(&server, 0).is_err());
}
