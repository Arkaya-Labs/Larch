//! Errors shared across crates (4.7).

use std::fmt;

use crate::HealthState;

/// A request failure. [`EngineError::code`] goes into `RequestSpan.error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum EngineError {
    PromptTooLong { tokens: u32, limit: u32 },
    Overloaded { waiting: u32, limit: u32 },
    NotReady(HealthState),
    Cancelled,
    DeviceLost,
    Internal(u16),
    InvalidRequest(u16),
}

impl EngineError {
    /// Stable, never 0 (0 in `RequestSpan.error` means no error).
    pub const fn code(&self) -> u16 {
        match self {
            EngineError::PromptTooLong { .. } => 1,
            EngineError::Overloaded { .. } => 2,
            EngineError::NotReady(_) => 3,
            EngineError::Cancelled => 4,
            EngineError::DeviceLost => 5,
            EngineError::Internal(_) => 6,
            EngineError::InvalidRequest(_) => 7,
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::PromptTooLong { tokens, limit } => {
                write!(f, "prompt is {tokens} tokens, the limit is {limit}")
            }
            EngineError::Overloaded { waiting, limit } => {
                write!(
                    f,
                    "overloaded: {waiting} requests waiting, the limit is {limit}"
                )
            }
            EngineError::NotReady(state) => write!(f, "engine not ready ({state:?})"),
            EngineError::Cancelled => f.write_str("cancelled"),
            EngineError::DeviceLost => f.write_str("device lost; the process must restart"),
            EngineError::Internal(code) => write!(f, "internal error {code}"),
            EngineError::InvalidRequest(code) => write!(f, "invalid request {code}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// A failure loading a model or opening the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum LoadError {
    Io,
    /// A GGUF parse error code and the byte offset it was found at.
    BadGguf(u16, u64),
    UnsupportedArch,
    /// The `ggml_type` of the tensor.
    UnsupportedQuant(u32),
    /// Bytes.
    DoesNotFit {
        need: u64,
        have: u64,
    },
    DriverTooOld {
        have: u32,
        need: u32,
    },
    Template(u16),
    Backend(u16),
}

impl LoadError {
    /// Stable, never 0.
    pub const fn code(&self) -> u16 {
        match self {
            LoadError::Io => 1,
            LoadError::BadGguf(..) => 2,
            LoadError::UnsupportedArch => 3,
            LoadError::UnsupportedQuant(_) => 4,
            LoadError::DoesNotFit { .. } => 5,
            LoadError::DriverTooOld { .. } => 6,
            LoadError::Template(_) => 7,
            LoadError::Backend(_) => 8,
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io => f.write_str("I/O error reading the model"),
            LoadError::BadGguf(code, offset) => {
                write!(f, "bad GGUF (error {code} at byte {offset})")
            }
            LoadError::UnsupportedArch => f.write_str("unsupported model architecture"),
            LoadError::UnsupportedQuant(ty) => {
                write!(f, "unsupported quantization (ggml_type {ty})")
            }
            LoadError::DoesNotFit { need, have } => {
                write!(
                    f,
                    "model does not fit: needs {need} bytes, {have} available"
                )
            }
            LoadError::DriverTooOld { have, need } => {
                write!(f, "driver too old: have {have}, need {need} or newer")
            }
            LoadError::Template(code) => write!(f, "chat template error {code}"),
            LoadError::Backend(code) => write!(f, "backend error {code}"),
        }
    }
}

impl std::error::Error for LoadError {}
