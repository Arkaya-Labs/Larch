//! Offline release trust anchors.

/// The release owner's offline primary public key, the only key that signs releases.
pub const PRIMARY: &str = "untrusted comment: minisign public key 352755DF231FA415\nRWQVpB8j31UnNTtZdjlYzi71ZGHAvhmaa5P0NrKn5OhW0oO0eZUQRO7s";
/// The release owner's offline recovery public key, used only to rotate both keys.
pub const BACKUP: &str = "untrusted comment: minisign public key 370D7DAFB582945B\nRWRblIK1r30NN63RdfzTO33Bwcrpni0p9VLtPu2Hfwo3tckrCnDon37q";
