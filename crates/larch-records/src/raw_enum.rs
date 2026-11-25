/// Declares a fieldless enum stored as a raw integer in records (4.1), with an extra `Unknown`
/// variant (the integer's `MAX`) so a value from a newer producer never makes an older reader panic.
/// Generates `from_raw` (unknown values decode to `Unknown`) and `raw`.
macro_rules! raw_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident: $raw:ty {
            $($(#[$vmeta:meta])* $variant:ident = $value:literal,)+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
        #[repr($raw)]
        pub enum $name {
            $($(#[$vmeta])* $variant = $value,)+
            /// A value this build does not know (written by a newer producer).
            Unknown = <$raw>::MAX,
        }

        impl $name {
            /// Decodes a raw field; values this build does not know become `Unknown`.
            pub const fn from_raw(v: $raw) -> Self {
                match v {
                    $($value => Self::$variant,)+
                    _ => Self::Unknown,
                }
            }

            /// The value stored in a record field.
            pub const fn raw(self) -> $raw {
                self as $raw
            }
        }
    };
}
