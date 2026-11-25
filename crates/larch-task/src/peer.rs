//! Operating-system identity checks for connected local sockets.
use std::os::unix::net::UnixStream;

/// Reject a socket unless its kernel-reported peer has this process's effective UID.
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
pub fn authenticate_peer(stream: &UnixStream) -> Result<(), String> {
    authenticate_peer_for_uid(stream, nix::unistd::geteuid().as_raw())
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
pub fn authenticate_peer(_: &UnixStream) -> Result<(), String> {
    Err("local socket peer authentication is unavailable on this platform".into())
}

/// Check a connected peer against a caller-supplied effective UID.
pub fn authenticate_peer_for_uid(stream: &UnixStream, expected_uid: u32) -> Result<(), String> {
    let actual_uid = peer_uid(stream)?;
    if actual_uid != expected_uid {
        return Err("local socket peer identity mismatch".into());
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn peer_uid(stream: &UnixStream) -> Result<u32, String> {
    nix::sys::socket::getsockopt(stream, nix::sys::socket::sockopt::PeerCredentials)
        .map(|credentials| credentials.uid())
        .map_err(|error| format!("unable to authenticate local socket peer: {error}"))
}

#[cfg(any(
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
fn peer_uid(stream: &UnixStream) -> Result<u32, String> {
    nix::unistd::getpeereid(stream)
        .map(|(uid, _)| uid.as_raw())
        .map_err(|error| format!("unable to authenticate local socket peer: {error}"))
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
fn peer_uid(_: &UnixStream) -> Result<u32, String> {
    Err("local socket peer authentication is unavailable on this platform".into())
}
