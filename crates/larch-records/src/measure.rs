use bytemuck::{Pod, Zeroable};

/// The state of a measured value (4.2). Zeroed memory is `Missing`, so a field a producer forgot
/// to fill is honest by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
#[repr(u8)]
pub enum MState {
    /// Should be measurable, but no sample arrived.
    Missing = 0,
    Measured = 1,
    Estimated = 2,
    /// This backend cannot measure the value at all (e.g. SM occupancy on Metal).
    Unsupported = 3,
}

impl MState {
    /// Decodes a raw state. A state this build does not know reads as `Missing`: the value is not
    /// usable, which is what `Missing` tells a rule.
    pub const fn from_raw(v: u8) -> Self {
        match v {
            1 => Self::Measured,
            2 => Self::Estimated,
            3 => Self::Unsupported,
            _ => Self::Missing,
        }
    }
}

macro_rules! measurement {
    ($(#[$meta:meta])* $name:ident, $t:ty, $pad:literal) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
        #[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
        pub struct $name {
            pub v: $t,
            s: u8,
            #[cfg_attr(feature = "schema", serde(skip))]
            _pad: [u8; $pad],
        }

        impl $name {
            const fn with(v: $t, s: MState) -> Self {
                Self { v, s: s as u8, _pad: [0; $pad] }
            }

            pub const fn measured(v: $t) -> Self {
                Self::with(v, MState::Measured)
            }

            pub const fn estimated(v: $t) -> Self {
                Self::with(v, MState::Estimated)
            }

            pub const fn unsupported() -> Self {
                Self::with(0 as $t, MState::Unsupported)
            }

            pub const fn missing() -> Self {
                Self::with(0 as $t, MState::Missing)
            }

            pub const fn state(&self) -> MState {
                MState::from_raw(self.s)
            }

            /// The value if it was `Measured` or `Estimated`.
            pub const fn get(&self) -> Option<$t> {
                match self.state() {
                    MState::Measured | MState::Estimated => Some(self.v),
                    MState::Missing | MState::Unsupported => None,
                }
            }
        }
    };
}

measurement!(
    /// An `f32` with its [`MState`]. 8 bytes.
    MF32, f32, 3
);
measurement!(
    /// An `f64` with its [`MState`]. 16 bytes.
    MF64, f64, 7
);
measurement!(
    /// A `u64` with its [`MState`]. 16 bytes.
    MU64, u64, 7
);

const _: () = assert!(size_of::<MF32>() == 8);
const _: () = assert!(size_of::<MF64>() == 16);
const _: () = assert!(size_of::<MU64>() == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn zeroed_is_missing() {
        assert_eq!(MF32::zeroed().state(), MState::Missing);
        assert_eq!(MU64::zeroed().get(), None);
    }

    #[test]
    fn states() {
        assert_eq!(MF32::measured(1.5).get(), Some(1.5));
        assert_eq!(MF64::estimated(2.0).get(), Some(2.0));
        assert_eq!(MU64::estimated(7).state(), MState::Estimated);
        assert_eq!(MF32::unsupported().get(), None);
        assert_eq!(MF32::unsupported().state(), MState::Unsupported);
        assert_eq!(MF64::missing().state(), MState::Missing);
        assert_eq!(MState::from_raw(200), MState::Missing);
    }
}
