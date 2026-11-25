use bytemuck::{Pod, Zeroable};

/// A UTF-8 string stored inline in a record: `[u8; N]`, NUL-padded (4.1).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct FixedStr<const N: usize>(pub [u8; N]);

impl<const N: usize> FixedStr<N> {
    /// Copies `s`, cut at the last character boundary that fits in `N` bytes.
    pub const fn from_str_truncating(s: &str) -> Self {
        let bytes = s.as_bytes();
        let mut len = if bytes.len() < N { bytes.len() } else { N };
        // Back up over UTF-8 continuation bytes (0b10xx_xxxx) so no character is split.
        while len < bytes.len() && len > 0 && (bytes[len] & 0xC0) == 0x80 {
            len -= 1;
        }
        let mut out = [0u8; N];
        let mut i = 0;
        while i < len {
            out[i] = bytes[i];
            i += 1;
        }
        Self(out)
    }

    /// The text up to the first NUL. Bytes that are not valid UTF-8 (only possible if a producer
    /// wrote the array directly) end the string.
    pub fn as_str(&self) -> &str {
        let end = self.0.iter().position(|&b| b == 0).unwrap_or(N);
        match std::str::from_utf8(&self.0[..end]) {
            Ok(s) => s,
            Err(e) => std::str::from_utf8(&self.0[..e.valid_up_to()]).unwrap_or_default(),
        }
    }
}

impl<const N: usize> Default for FixedStr<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> std::fmt::Debug for FixedStr<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> std::fmt::Display for FixedStr<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "schema")]
impl<const N: usize> serde::Serialize for FixedStr<N> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

#[cfg(feature = "schema")]
impl<const N: usize> schemars::JsonSchema for FixedStr<N> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("FixedStr{N}").into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": format!("UTF-8, at most {N} bytes"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::FixedStr;

    #[test]
    fn round_trip_and_truncation() {
        assert_eq!(FixedStr::<8>::from_str_truncating("abc").as_str(), "abc");
        assert_eq!(
            FixedStr::<4>::from_str_truncating("abcdef").as_str(),
            "abcd"
        );
        assert_eq!(FixedStr::<4>::from_str_truncating("abcd").as_str(), "abcd");
        assert_eq!(FixedStr::<4>::from_str_truncating("").as_str(), "");
        // "é" is 2 bytes; a 4-byte field must not keep half of the second one.
        assert_eq!(FixedStr::<4>::from_str_truncating("aéé").as_str(), "aé");
        assert_eq!(FixedStr::<1>::from_str_truncating("é").as_str(), "");
        // A 4-byte character never splits either.
        assert_eq!(FixedStr::<3>::from_str_truncating("😀").as_str(), "");
    }

    #[test]
    fn invalid_bytes_end_the_string() {
        let s = FixedStr::<4>([b'a', 0xFF, b'b', 0]);
        assert_eq!(s.as_str(), "a");
    }
}
