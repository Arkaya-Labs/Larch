//! The larch facade: library API, CLI commands and the supervisor.

/// The release version and source commit, including source archives without Git metadata.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("LARCH_COMMIT"), ")");

pub mod api;
pub mod bench;
pub mod chat;
pub mod info;
pub mod knob;
pub mod parity;
pub mod pin;
pub mod repack;
pub mod routes_check;
pub mod serve;
pub mod session;
pub mod supervise;
pub mod text;
pub mod tokcheck;
pub mod toolcall;

pub use api::{DocFit, EngineError, GenParams, Larch, LoadError, LoadOptions, Prompt, ToolMessage};
pub use pin::{Pin, PinError};
pub use session::{ChatMessage, Overflow, Role, Session, SessionOptions};
pub use text::{Finish, TextEvent, TextStream, Timings, TokenLogprob, Usage};
