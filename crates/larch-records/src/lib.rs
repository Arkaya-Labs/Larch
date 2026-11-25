//! The larch contract: telemetry records, the ring that carries them, the knob schema and API,
//! and the errors shared by larch, strided, gait and cuprum (architecture section 4).
//! No GPU code, no unsafe, no I/O.

#[macro_use]
mod raw_enum;

mod control;
mod error;
mod fixed_str;
mod knob;
mod measure;
mod record;
mod ring;

pub use control::*;
pub use error::*;
pub use fixed_str::FixedStr;
pub use knob::*;
pub use measure::*;
pub use record::*;
pub use ring::*;
