//! Coding tasks: the agent changes a repository in its own worktree and a human approves the
//! packet before it is merged, or changes any folder in place after a checkpoint, asking before
//! each write the user has not allowed.
// ponytail: moves behind the run-model crate when v20-19 lands

pub mod allow;
pub mod checks;
mod experts;
pub(crate) mod lookup;
pub mod r#loop;
pub mod map;
mod plan;
pub mod record;
mod review;
pub mod room;
mod session;
pub mod task;
