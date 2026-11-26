//! Offline release trust anchors.

/// The release owner's offline primary public key, the only key that signs releases.
pub const PRIMARY: &str = "untrusted comment: minisign public key EAD12064D0704DD3\nRWTTTXDQZCDR6gcdAISWt5CmUyoRp2fXzRv1g3e9QiDAwCrYHK0XBpA5";
/// The release owner's offline recovery public key, used only to rotate both keys.
pub const BACKUP: &str = "untrusted comment: minisign public key 48F381111AE19655\nRWRVluEaEYHzSDeCoyMmRwruaYPC8dt/tLxE8QkY+mioTqMjNMG6bt65";
