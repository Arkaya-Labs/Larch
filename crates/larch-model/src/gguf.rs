//! GGUF v2/v3: the header, metadata and tensor table. Every length, count, offset and dims product is
//! checked against the file size before use, so a hostile file is an error, never a panic (row R9), and
//! memory stays within about 32 bytes (one `Value`) per byte of header.
// ponytail: typed arrays (`Vec<u8>`, `Vec<String>`, ...) would bring that to about 1x; it changes the
// `Value` API the tokenizer reads.

use std::collections::BTreeMap;
use std::io::Read;

use larch_kernels::DType;
use larch_records::LoadError;

/// `LoadError::BadGguf` codes.
pub mod code {
    pub const MAGIC: u16 = 1;
    pub const VERSION: u16 = 2;
    pub const TRUNCATED: u16 = 3;
    pub const VALUE_TYPE: u16 = 4;
    pub const UTF8: u16 = 5;
    pub const TENSOR: u16 = 6;
    pub const ALIGNMENT: u16 = 7;
    /// A model key missing or out of range (offset 0).
    pub const KEY: u16 = 8;
    /// A tensor the model needs is missing (offset 0).
    pub const MISSING_TENSOR: u16 = 9;
    /// A file that does not fit the model it is loaded with: a projector of another width (offset 0).
    pub const MISMATCH: u16 = 10;
}

/// A metadata value, one variant per GGUF value type.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// Any non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Any float or non-negative integer, as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    /// The string of a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The items of an `Arr`.
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// One entry of the tensor table.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    /// GGML order: `dims[0]` is the contiguous one (columns).
    pub dims: Vec<u64>,
    /// From the start of the data section.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// Row-major larch shape: GGML's dims reversed.
    pub fn shape(&self) -> Vec<i64> {
        self.dims.iter().rev().map(|&d| d as i64).collect()
    }
}

/// A parsed GGUF header: metadata and tensor table, without the data section.
#[derive(Debug)]
pub struct Gguf {
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset of the data section in the file.
    pub data_offset: u64,
}

/// GGML type id ↔ larch dtype, for the types the ops take (architecture 6.1).
const GGML_TYPES: [(u32, DType); 10] = [
    (0, DType::F32),
    (1, DType::F16),
    (2, DType::Q4_0),
    (6, DType::Q5_0),
    (8, DType::Q8_0),
    (10, DType::Q2K),
    (11, DType::Q3K),
    (12, DType::Q4K),
    (13, DType::Q5K),
    (14, DType::Q6K),
];

/// The larch dtype of a GGML type id, or `UnsupportedQuant`.
pub fn dtype(ggml_type: u32) -> Result<DType, LoadError> {
    GGML_TYPES
        .iter()
        .find(|e| e.0 == ggml_type)
        .map(|e| e.1)
        .ok_or(LoadError::UnsupportedQuant(ggml_type))
}

/// The GGML type id of a dtype (the inverse of [`dtype`]).
pub fn ggml_type(t: DType) -> Option<u32> {
    GGML_TYPES.iter().find(|e| e.1 == t).map(|e| e.0)
}

struct Reader<R> {
    r: R,
    at: u64,
    len: u64,
}

impl<R: Read> Reader<R> {
    fn bad(&self, c: u16) -> LoadError {
        LoadError::BadGguf(c, self.at)
    }

    fn take(&mut self, n: u64) -> Result<Vec<u8>, LoadError> {
        if n > self.len - self.at {
            return Err(self.bad(code::TRUNCATED));
        }
        let mut v = vec![0; n as usize]; // n <= the file size
        self.r
            .read_exact(&mut v)
            .map_err(|_| self.bad(code::TRUNCATED))?;
        self.at += n;
        Ok(v)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], LoadError> {
        if N as u64 > self.len - self.at {
            return Err(self.bad(code::TRUNCATED));
        }
        let mut a = [0; N];
        self.r
            .read_exact(&mut a)
            .map_err(|_| self.bad(code::TRUNCATED))?;
        self.at += N as u64;
        Ok(a)
    }

    fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64, LoadError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    /// A count of items each at least `min` bytes long, so it cannot exceed what is left.
    fn count(&mut self, min: u64) -> Result<u64, LoadError> {
        let n = self.u64()?;
        if n.saturating_mul(min) > self.len - self.at {
            return Err(self.bad(code::TRUNCATED));
        }
        Ok(n)
    }

    fn string(&mut self) -> Result<String, LoadError> {
        let n = self.count(1)?;
        let at = self.at;
        String::from_utf8(self.take(n)?).map_err(|_| LoadError::BadGguf(code::UTF8, at))
    }

    fn value(&mut self, ty: u32) -> Result<Value, LoadError> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                if elem == 9 {
                    return Err(self.bad(code::VALUE_TYPE)); // nested arrays: no model uses them
                }
                let n = self.count(1)?;
                (0..n)
                    .map(|_| self.value(elem))
                    .collect::<Result<_, _>>()
                    .map(Value::Arr)?
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            _ => return Err(self.bad(code::VALUE_TYPE)),
        })
    }
}

impl Gguf {
    /// Parses the header and tables of a whole GGUF file held in `b`.
    pub fn parse(b: &[u8]) -> Result<Gguf, LoadError> {
        Gguf::read(b, b.len() as u64)
    }

    /// Parses the header and tables from `r`, the start of a GGUF file of `len` bytes; reads up to the
    /// end of the tensor table, not the data section.
    pub fn read(r: impl Read, len: u64) -> Result<Gguf, LoadError> {
        let mut r = Reader { r, at: 0, len };
        if r.take(4)? != b"GGUF" {
            return Err(LoadError::BadGguf(code::MAGIC, 0));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(LoadError::BadGguf(code::VERSION, 4));
        }
        let n_tensors = r.count(8)?;
        let n_kv = r.count(8)?;
        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            kv.insert(key, r.value(ty)?);
        }
        let mut tensors = Vec::new();
        for _ in 0..n_tensors {
            let bad = LoadError::BadGguf(code::TENSOR, r.at);
            let name = r.string()?;
            let n_dims = r.u32()?;
            if !(1..=4).contains(&n_dims) {
                return Err(bad);
            }
            let dims = (0..n_dims)
                .map(|_| r.u64())
                .collect::<Result<Vec<_>, _>>()?;
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            let elems = dims
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d))
                .ok_or(bad)?;
            let bytes = match dtype(ggml_type) {
                Ok(t) if dims[0].is_multiple_of(t.block()) => (elems / t.block())
                    .checked_mul(t.block_bytes())
                    .ok_or(bad)?,
                Ok(_) => return Err(bad),
                Err(_) => 0, // unknown types are rejected when the model asks for the tensor
            };
            tensors.push(TensorInfo {
                name,
                ggml_type,
                dims,
                offset,
                bytes,
            });
        }
        let align = match kv.get("general.alignment") {
            None => 32,
            Some(v) => v
                .as_u64()
                .filter(|a| a.is_power_of_two())
                .ok_or(LoadError::BadGguf(code::ALIGNMENT, r.at))?,
        };
        let data_offset = r.at.next_multiple_of(align);
        for t in &tensors {
            let start = data_offset.checked_add(t.offset);
            let fits = t.offset.is_multiple_of(align)
                && start
                    .and_then(|s| s.checked_add(t.bytes))
                    .is_some_and(|e| e <= len);
            if !fits {
                return Err(LoadError::BadGguf(code::TENSOR, start.unwrap_or(u64::MAX)));
            }
        }
        Ok(Gguf {
            kv,
            tensors,
            data_offset,
        })
    }

    /// The metadata value under `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    /// The tensor named `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Writes a GGUF v3 file: metadata, then each tensor as `(name, GGML dims, type, bytes)`.
pub fn write(
    kv: &[(impl AsRef<str>, Value)],
    tensors: &[(impl AsRef<str>, Vec<u64>, u32, Vec<u8>)],
) -> Vec<u8> {
    let table: Vec<(&str, Vec<u64>, u32, u64)> = tensors
        .iter()
        .map(|(n, d, t, data)| (n.as_ref(), d.clone(), *t, data.len() as u64))
        .collect();
    let mut o = header(kv, &table);
    for (_, _, _, data) in tensors {
        o.extend(data);
        o.resize(o.len().next_multiple_of(32), 0);
    }
    o
}

/// A GGUF v3 file up to its data section: metadata, then each tensor as `(name, GGML dims, type,
/// byte length)`, whose data follows in the order given, each at the next multiple of 32 (the
/// default `general.alignment`).
pub fn header(
    kv: &[(impl AsRef<str>, Value)],
    tensors: &[(impl AsRef<str>, Vec<u64>, u32, u64)],
) -> Vec<u8> {
    fn string(o: &mut Vec<u8>, s: &str) {
        o.extend((s.len() as u64).to_le_bytes());
        o.extend(s.as_bytes());
    }
    fn value(o: &mut Vec<u8>, v: &Value) {
        match v {
            Value::U8(x) => o.push(*x),
            Value::I8(x) => o.push(*x as u8),
            Value::U16(x) => o.extend(x.to_le_bytes()),
            Value::I16(x) => o.extend(x.to_le_bytes()),
            Value::U32(x) => o.extend(x.to_le_bytes()),
            Value::I32(x) => o.extend(x.to_le_bytes()),
            Value::F32(x) => o.extend(x.to_le_bytes()),
            Value::Bool(x) => o.push(u8::from(*x)),
            Value::Str(s) => string(o, s),
            Value::Arr(a) => {
                o.extend(a.first().map_or(4, ty).to_le_bytes());
                o.extend((a.len() as u64).to_le_bytes());
                a.iter().for_each(|x| value(o, x));
            }
            Value::U64(x) => o.extend(x.to_le_bytes()),
            Value::I64(x) => o.extend(x.to_le_bytes()),
            Value::F64(x) => o.extend(x.to_le_bytes()),
        }
    }
    fn ty(v: &Value) -> u32 {
        match v {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Arr(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }
    let mut o = b"GGUF".to_vec();
    o.extend(3u32.to_le_bytes());
    o.extend((tensors.len() as u64).to_le_bytes());
    o.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        string(&mut o, k.as_ref());
        o.extend(ty(v).to_le_bytes());
        value(&mut o, v);
    }
    let mut off = 0u64;
    for (name, dims, t, len) in tensors {
        string(&mut o, name.as_ref());
        o.extend((dims.len() as u32).to_le_bytes());
        dims.iter().for_each(|d| o.extend(d.to_le_bytes()));
        o.extend(t.to_le_bytes());
        o.extend(off.to_le_bytes());
        off = (off + len).next_multiple_of(32);
    }
    o.resize(o.len().next_multiple_of(32), 0);
    o
}
