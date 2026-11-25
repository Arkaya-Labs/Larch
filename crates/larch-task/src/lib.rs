//! Durable, local workflow state and execution primitives.
//! External activities must execute outside a store transaction.

pub mod admission;
pub mod artifacts;
pub mod execution;
pub mod graph;
pub mod host;
pub mod ipc;
#[cfg(unix)]
pub mod peer;
pub mod reducer;
pub mod runtime;
pub mod schedule;
pub mod store;
pub mod types;
pub mod watch;
