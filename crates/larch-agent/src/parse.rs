//! Argument validation against the tool's JSON Schema; the call format is `larch::toolcall`.
//! Every error's `Display` is the sentence the loop feeds back to the model.

use std::fmt;

use jsonschema::error::{TypeKind, ValidationErrorKind};
use serde_json::Value;

pub use larch::toolcall::{Family, ParseError, Parsed, ToolCall, family_for, parse};

use crate::tool::ToolSpec;

/// A call that does not fit a tool; `Display` is the message the model gets back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidateError {
    /// No tool of that name among the ones the model was shown.
    UnknownTool {
        /// The name the model wrote.
        name: String,
        /// The names it was shown.
        available: Vec<String>,
    },
    /// A required argument is absent.
    MissingArgument(String),
    /// An argument has the wrong JSON type.
    WrongType {
        /// The argument's path in the arguments object.
        arg: String,
        /// The JSON Schema type it needs.
        expected: String,
    },
    /// An argument the schema does not name.
    UnknownArgument {
        /// The name the model wrote.
        arg: String,
        /// The names the schema has.
        expected: Vec<String>,
    },
    /// Any other schema failure, as one sentence.
    Invalid(String),
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTool { name, available } => {
                // Edits on up to a third of the name are a typo worth naming ('ls' is not
                // 'log'); further apart, the model gets the whole list.
                let near = name.chars().count().div_ceil(3).max(1);
                let near = available
                    .iter()
                    .map(|a| (strsim::levenshtein(name, a), a))
                    .min()
                    .filter(|(d, _)| *d <= near);
                match near {
                    Some((_, a)) => write!(f, "unknown tool '{name}'; did you mean '{a}'?"),
                    None => write!(
                        f,
                        "unknown tool '{name}'; available: {}",
                        available.join(", ")
                    ),
                }
            }
            Self::MissingArgument(arg) => write!(f, "missing required argument '{arg}'"),
            Self::WrongType { arg, expected } => {
                write!(f, "argument '{arg}' must be of type {expected}")
            }
            Self::UnknownArgument { arg, expected } => write!(
                f,
                "unknown argument '{arg}'; arguments: {}",
                expected.join(", ")
            ),
            Self::Invalid(sentence) => f.write_str(sentence),
        }
    }
}

/// Types `call`'s string arguments by their schemas in `tools` (`toolcall::coerce`), and reads a
/// Python `True`, `False` or `None`, as an XML-body value may be written, as JSON where the schema
/// admits it. Runs before `validate`.
pub fn coerce(call: &mut ToolCall, tools: &[Value]) {
    larch::toolcall::coerce(call, tools);
    let mut py = call.clone();
    let Some(args) = py.args.as_object_mut() else {
        return;
    };
    for v in args.values_mut() {
        *v = match v.as_str() {
            Some("True") => "true".into(),
            Some("False") => "false".into(),
            Some("None") => "null".into(),
            _ => continue,
        };
    }
    // A literal that stays a string (its schema takes a string) keeps the model's spelling.
    larch::toolcall::coerce(&mut py, tools);
    if let (Some(typed), Some(args)) = (py.args.as_object(), call.args.as_object_mut()) {
        for (key, v) in typed.iter().filter(|(_, v)| !v.is_string()) {
            args.insert(key.clone(), v.clone());
        }
    }
}

/// Checks the call's name and its arguments against `spec`'s JSON Schema; the first failure wins.
pub fn validate(call: &ToolCall, spec: &ToolSpec) -> Result<(), ValidateError> {
    if call.name != spec.name {
        return Err(ValidateError::UnknownTool {
            name: call.name.clone(),
            available: vec![spec.name.clone()],
        });
    }
    let validator = jsonschema::validator_for(&spec.schema).map_err(|e| {
        ValidateError::Invalid(format!("tool '{}' has a broken schema: {e}", spec.name))
    })?;
    let Some(error) = validator.iter_errors(&call.args).next() else {
        return Ok(());
    };
    let arg = error
        .instance_path()
        .as_str()
        .trim_start_matches('/')
        .to_owned();
    Err(match error.kind() {
        ValidationErrorKind::Required { property } => {
            ValidateError::MissingArgument(property.as_str().unwrap_or_default().into())
        }
        ValidationErrorKind::AdditionalProperties { unexpected } if arg.is_empty() => {
            let names = spec.schema["properties"].as_object();
            let mut expected: Vec<String> =
                names.into_iter().flat_map(|p| p.keys().cloned()).collect();
            // Sorted: the map's own order depends on a serde_json feature any crate in the
            // build may turn on (larch-model does).
            expected.sort();
            ValidateError::UnknownArgument {
                arg: unexpected.first().cloned().unwrap_or_default(),
                expected,
            }
        }
        ValidationErrorKind::Type {
            kind: TypeKind::Single(expected),
        } if !arg.is_empty() => ValidateError::WrongType {
            arg,
            expected: expected.to_string(),
        },
        _ if arg.is_empty() => ValidateError::Invalid(format!("arguments are invalid: {error}")),
        _ => ValidateError::Invalid(format!("argument '{arg}' is invalid: {error}")),
    })
}
