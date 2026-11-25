//! Stands in for xattr 0.2.3, which gpui's tar unpacker pulls and which names the Linux ENOATTR that
//! libc 0.2.190 removed. The unpacker calls `set` only when asked to keep extended attributes, which
//! gpui never asks, so the stand-in refuses instead of carrying the original's unsafe FFI.
//! ponytail: delete this crate and its [patch] when gpui ships an async-tar on xattr 1.x.

use std::ffi::OsStr;
use std::io;
use std::path::Path;

/// Refuses: Larch never keeps extended attributes from an archive.
pub fn set<N: AsRef<OsStr>, P: AsRef<Path>>(_: P, _: N, _: &[u8]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "extended attributes are not kept",
    ))
}
