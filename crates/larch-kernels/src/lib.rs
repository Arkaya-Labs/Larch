//! Safe wrappers over `kernels/include/larch.h` (architecture 7): the only GPU API the model and the
//! engine use. Handles free themselves on drop and keep their [`Device`] alive. A [`Tensor`] is a view
//! into a [`Buffer`], bounds-checked when made. Ops are in [`ops`], which says what else is checked
//! and by whom. [`Graph`] says who owns the memory a replay touches; [`Mmap`] and
//! [`Buffer::wrap_host`] are Metal's zero-copy weights.

pub mod ops;

/// The CUDA routes' environment switches, each read once. `LARCH_QUANT_I8`, `LARCH_PREFILL_TC` and
/// `LARCH_DECODE_TC` are tri-state (`1` on, `0` off, anything else the measured-arch default);
/// `LARCH_MIXED_GRAPHS` is on only for `1`. Each route also needs a device and shape that run it; bench rows
/// record the switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Routes {
    /// `LARCH_QUANT_I8`: activation-INT8 Q4_K projections, chosen by larch-model at load. `1` forces
    /// them on, `0` off; `None` (unset or anything else) keeps the measured-arch default.
    pub quant_i8: Option<bool>,
    /// `LARCH_PREFILL_TC`: tensor-core paged prefill. The kernel library reads it once at load
    /// itself (that read routes, with the same `1`/`0`/default rule and its own measured-arch list);
    /// this one reports it.
    pub prefill_tc: Option<bool>,
    /// `LARCH_DECODE_TC`: tensor-core paged decode, read and reported like `prefill_tc`.
    pub decode_tc: Option<bool>,
    /// `LARCH_MIXED_GRAPHS`: bounded mixed-step graphs in the engine.
    pub mixed_graphs: bool,
    /// `LARCH_PREFILL_STREAM_ROWS`: prefill rows from which host experts stream to the device, 0 for
    /// never; unset leaves the model's measured default. Non-zero also opts in to the pool's KV cost:
    /// the fit waives its KV floor and admits the pool wherever it fits past one context and the graphs.
    pub prefill_stream_rows: Option<u32>,
    /// `LARCH_FUSED`: the fused layer entries (larch.h ABI 4), chosen by larch-model at load. `0` keeps every
    /// layer on its unfused ops, `1` takes every entry, including those off by default; `None` takes the entries
    /// on by default (larch-model's `FUSED`, on CUDA sm_120 only), where a gemm-replacing entry is on only once it
    /// gives the unfused path's bytes or passes parity against it.
    pub fused: Option<bool>,
}

/// A tri-state switch: `1` forces the route on, `0` off, anything else leaves the measured-arch default.
fn switch(value: Option<&str>) -> Option<bool> {
    match value {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// The route switches, read on first use and fixed for the process.
pub fn routes() -> Routes {
    static ROUTES: std::sync::OnceLock<Routes> = std::sync::OnceLock::new();
    *ROUTES.get_or_init(|| {
        let env = |name| switch(std::env::var(name).ok().as_deref());
        Routes {
            quant_i8: env("LARCH_QUANT_I8"),
            prefill_tc: env("LARCH_PREFILL_TC"),
            decode_tc: env("LARCH_DECODE_TC"),
            mixed_graphs: env("LARCH_MIXED_GRAPHS") == Some(true),
            prefill_stream_rows: std::env::var("LARCH_PREFILL_STREAM_ROWS")
                .ok()
                .and_then(|v| v.parse().ok()),
            fused: env("LARCH_FUSED"),
        }
    })
}

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::fmt;
use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use bytemuck::Pod;
use larch_sys as sys;

/// `"cuda"`, `"hip"`, `"metal"` or `"cpu"`: the GPU backend these kernels were built with.
pub use sys::BACKEND;
/// One logits row's sampling settings, as `larch_sample_prep` reads them.
pub use sys::larch_sample_params as SampleParams;

/// The ordinal of the CPU reference backend; GPUs are 0, 1, ...
pub const CPU: i32 = -1;
/// The ordinal of the CPU device the engine computes on (an MoE model's routed experts in host memory);
/// it runs the reference's ops until the fast CPU backend lands.
pub const CPU_COMPUTE: i32 = -2;

/// Why a kernel-library call failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KError {
    InvalidArg,
    Oom,
    Unsupported,
    Launch,
    DeviceLost,
    Timeout,
    /// The CUDA driver supports an older major version than the runtime the kernels were built with
    /// (`have` < `need`, CUDA's encoding: 13000 = 13.0), or a driver failed outright (both 0; see
    /// [`last_error`]).
    Driver {
        have: u32,
        need: u32,
    },
    /// The kernel library was built for another ABI version: rebuild it.
    Stale,
}

impl fmt::Display for KError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KError::Driver { have: 0, need: 0 } => f.write_str("GPU driver error"),
            KError::Driver { have: 0, need } => write!(
                f,
                "no GPU driver found; the kernels need a CUDA {} driver",
                need / 1000
            ),
            KError::Driver { have, need } => write!(
                f,
                "the GPU driver supports CUDA {}.{}, the kernels need a CUDA {} driver (R580 or newer \
                 for CUDA 13)",
                have / 1000,
                have % 1000 / 10,
                need / 1000
            ),
            KError::Stale => f.write_str("the kernel library was built for another ABI version"),
            e => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for KError {}

impl KError {
    /// Stable, never 0; carried in `EngineError::Internal` and `LoadError::Backend`.
    pub const fn code(&self) -> u16 {
        match self {
            KError::InvalidArg => 1,
            KError::Oom => 2,
            KError::Unsupported => 3,
            KError::Launch => 4,
            KError::DeviceLost => 5,
            KError::Timeout => 6,
            KError::Driver { .. } => 7,
            KError::Stale => 8,
        }
    }
}

fn check(s: sys::larch_status) -> Result<(), KError> {
    use sys::larch_status as S;
    match s {
        S::LARCH_OK => Ok(()),
        S::LARCH_ERR_INVALID_ARG => Err(KError::InvalidArg),
        S::LARCH_ERR_OOM => Err(KError::Oom),
        S::LARCH_ERR_UNSUPPORTED => Err(KError::Unsupported),
        S::LARCH_ERR_DEVICE_LOST => Err(KError::DeviceLost),
        S::LARCH_ERR_TIMEOUT => Err(KError::Timeout),
        S::LARCH_ERR_DRIVER => Err(KError::Driver { have: 0, need: 0 }),
        _ => Err(KError::Launch),
    }
}

/// The detail string of this thread's last failed call into the kernel library, for logs. A check
/// made here in Rust (bounds, devices) fails without a call and leaves it unchanged.
pub fn last_error() -> String {
    // SAFETY: larch_last_error accepts NULL and returns a NUL-terminated thread-local string that stays
    // valid until this thread's next larch call; it is copied before returning.
    unsafe { CStr::from_ptr(sys::larch_last_error(ptr::null())) }
        .to_string_lossy()
        .into_owned()
}

/// GPUs of the backend this build has (0 in a CPU-only build).
pub fn device_count() -> Result<i32, KError> {
    let mut n = 0;
    // SAFETY: `n` is a valid out pointer.
    check(unsafe { sys::larch_device_count(&mut n) })?;
    Ok(n)
}

/// The backend a device belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Hip,
    Metal,
}

impl Backend {
    fn from_raw(b: sys::larch_backend) -> Backend {
        match b {
            sys::larch_backend::LARCH_BACKEND_CUDA => Backend::Cuda,
            sys::larch_backend::LARCH_BACKEND_HIP => Backend::Hip,
            sys::larch_backend::LARCH_BACKEND_METAL => Backend::Metal,
            _ => Backend::Cpu,
        }
    }
}

/// Element types, GGML's block layouts byte for byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Q4_0,
    Q5_0,
    Q3K,
    Q2K,
}

impl DType {
    /// Elements per block (1 for plain types).
    pub const fn block(self) -> u64 {
        match self {
            DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K => 256,
            DType::Q8_0 | DType::Q4_0 | DType::Q5_0 => 32,
            _ => 1,
        }
    }

    /// Bytes per block (per element for plain types).
    pub const fn block_bytes(self) -> u64 {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4K => 144,
            DType::Q5K => 176,
            DType::Q6K => 210,
            DType::Q8_0 => 34,
            DType::Q4_0 => 18,
            DType::Q5_0 => 22,
            DType::Q3K => 110,
            DType::Q2K => 84,
        }
    }

    /// Bytes of `n` elements; panics unless `n` is a whole number of blocks whose size fits in u64.
    pub const fn bytes(self, n: u64) -> u64 {
        self.checked_bytes(n)
            .expect("DType::bytes: a partial block or past u64")
    }

    /// [`DType::bytes`], `None` where that panics.
    pub const fn checked_bytes(self, n: u64) -> Option<u64> {
        if !n.is_multiple_of(self.block()) {
            return None;
        }
        (n / self.block()).checked_mul(self.block_bytes())
    }

    /// The byte alignment of a tensor's offset: the element size, or 2 for a quant block (every block
    /// size is even, with its f16 scales at even offsets).
    const fn align(self) -> u64 {
        if self.block() == 1 {
            self.block_bytes()
        } else {
            2
        }
    }

    fn raw(self) -> sys::larch_dtype {
        use sys::larch_dtype as T;
        match self {
            DType::F32 => T::LARCH_F32,
            DType::F16 => T::LARCH_F16,
            DType::BF16 => T::LARCH_BF16,
            DType::I32 => T::LARCH_I32,
            DType::Q4K => T::LARCH_Q4_K,
            DType::Q5K => T::LARCH_Q5_K,
            DType::Q6K => T::LARCH_Q6_K,
            DType::Q8_0 => T::LARCH_Q8_0,
            DType::Q4_0 => T::LARCH_Q4_0,
            DType::Q5_0 => T::LARCH_Q5_0,
            DType::Q3K => T::LARCH_Q3_K,
            DType::Q2K => T::LARCH_Q2_K,
        }
    }
}

/// `larch_device_props`, read when asked.
#[derive(Clone, Debug)]
pub struct DeviceProps {
    pub name: String,
    pub backend: Backend,
    /// 86/89/120 (CUDA sm), 1030/1100/1200 (gfx), the MTLGPUFamily value on Apple.
    pub arch_code: u32,
    pub compute_units: u32,
    pub warp_size: u32,
    pub max_threads_per_group: u32,
    pub shared_mem_per_group: u64,
    pub l2_bytes: u64,
    pub vram_total: u64,
    pub vram_free: u64,
    /// Apple's GPU memory budget; equal to `vram_free` elsewhere.
    pub recommended_working_set: u64,
    pub peak_gbps: f32,
    pub driver_version: u32,
    pub runtime_version: u32,
    pub flags: u32,
}

impl DeviceProps {
    pub const CAP_GRAPHS: u32 = sys::LARCH_CAP_GRAPHS;
    pub const CAP_BF16: u32 = sys::LARCH_CAP_BF16;
    pub const CAP_ASYNC_COPY: u32 = sys::LARCH_CAP_ASYNC_COPY;
    pub const CAP_UNIFIED: u32 = sys::LARCH_CAP_UNIFIED;
}

struct RawDevice {
    raw: *mut sys::larch_device,
    /// Every file map a buffer of this device wraps, unmapped after the close ([`Buffer::wrap_host`]).
    // ponytail: kept until the device closes (one GGUF per process, a reload is a restart, 9.11);
    // release a map at a stream sync if a process ever swaps models in place.
    maps: Mutex<Vec<Arc<Map>>>,
}

// SAFETY: every backend's device is safe to use from several threads (Metal and CUDA devices are
// thread-safe objects; the CPU reference's device holds no mutable state), it is closed exactly
// once, when the last Arc drops, and `maps` is behind a Mutex.
unsafe impl Send for RawDevice {}
// SAFETY: as above.
unsafe impl Sync for RawDevice {}

impl Drop for RawDevice {
    fn drop(&mut self) {
        // SAFETY: the pointer came from larch_device_open, and every buffer, stream, event and graph of
        // this device holds an Arc to it, so all of them are already freed. `maps` drops after this,
        // once every stream has waited for its work (Stream's drop).
        unsafe { sys::larch_device_close(self.raw) }
    }
}

/// An open device. Cloning shares it.
#[derive(Clone)]
pub struct Device(Arc<RawDevice>);

impl Device {
    /// Opens [`CPU`] or GPU `ordinal` of this build's backend, after checking the kernel ABI version
    /// and, for a CUDA GPU, that the driver supports the major version of the runtime the kernels
    /// were built with (the R580 branch for CUDA 13.x).
    pub fn open(ordinal: i32) -> Result<Device, KError> {
        // SAFETY: a plain query.
        if unsafe { sys::larch_abi_version() } != sys::LARCH_ABI_VERSION {
            return Err(KError::Stale);
        }
        if ordinal >= 0 {
            // Before the open: a driver too old for the runtime fails inside it with no versions.
            let (mut backend, mut have, mut need) = (sys::larch_backend::LARCH_BACKEND_CPU, 0, 0);
            // SAFETY: valid out pointers.
            check(unsafe { sys::larch_backend_info(&mut backend, &mut need, &mut have) })?;
            // Minor-version compatibility: any driver of the runtime's major version runs it.
            if backend == sys::larch_backend::LARCH_BACKEND_CUDA && have / 1000 < need / 1000 {
                return Err(KError::Driver { have, need });
            }
        }
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer; on success the device is owned by RawDevice.
        check(unsafe { sys::larch_device_open(ordinal, &mut raw) })?;
        Ok(Device(Arc::new(RawDevice {
            raw,
            maps: Mutex::default(),
        })))
    }

    /// Current properties (`vram_free` changes as memory is allocated).
    pub fn props(&self) -> Result<DeviceProps, KError> {
        // SAFETY: larch_device_props is plain old data; all zeros is a valid value.
        let mut p: sys::larch_device_props = unsafe { std::mem::zeroed() };
        // SAFETY: the device is open and `p` is a valid out pointer.
        check(unsafe { sys::larch_device_props_get(self.raw(), &mut p) })?;
        let name = CStr::from_bytes_until_nul(bytemuck::cast_slice(&p.name))
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(DeviceProps {
            name,
            backend: Backend::from_raw(p.backend),
            arch_code: p.arch_code,
            compute_units: p.compute_units,
            warp_size: p.warp_size,
            max_threads_per_group: p.max_threads_per_group,
            shared_mem_per_group: p.shared_mem_per_group,
            l2_bytes: p.l2_bytes,
            vram_total: p.vram_total,
            vram_free: p.vram_free,
            recommended_working_set: p.recommended_working_set,
            peak_gbps: p.peak_gbps,
            driver_version: p.driver_version,
            runtime_version: p.runtime_version,
            flags: p.flags,
        })
    }

    fn raw(&self) -> *mut sys::larch_device {
        self.0.raw
    }

    fn same(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// What a [`Buffer`] owns, shared with the [`Graph`] replays in flight: freed when the last drops.
struct RawBuffer {
    raw: *mut sys::larch_buffer,
    bytes: u64,
    /// A wrapped file map: uploads and op outputs are refused.
    readonly: bool,
    dev: Device,
}

// SAFETY: Rust never dereferences `raw`; it only hands it to the backend, and frees it once, when the
// last Arc drops. Another thread reaches it only through a Graph replay, and graphs exist only on GPU
// backends, whose ops touch device memory (or a wrapped map, read-only), never host memory the
// owner's thread could be writing at the same time.
unsafe impl Send for RawBuffer {}
// SAFETY: as above.
unsafe impl Sync for RawBuffer {}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        // SAFETY: made by larch_buffer_alloc or larch_buffer_wrap_host on a device that is still open
        // (self.dev). Work still queued on it must keep it alive in the backend: Metal command buffers
        // retain it, CUDA's cudaFree waits for the device, and the CPU reference queues nothing.
        // larch.h does not say so yet (an A1 contract item); a backend that frees asynchronously
        // breaks this.
        unsafe { sys::larch_buffer_free(self.raw) }
    }
}

/// Device memory. Never exposes a pointer; reach it through [`Tensor`]s and [`Stream`] copies.
pub struct Buffer {
    inner: Arc<RawBuffer>,
    /// Send but not Sync: ops write through shared references, and on the CPU reference they run on
    /// the calling thread.
    _not_sync: PhantomData<Cell<()>>,
}

impl Buffer {
    fn new(raw: *mut sys::larch_buffer, bytes: u64, readonly: bool, dev: &Device) -> Buffer {
        Buffer {
            inner: Arc::new(RawBuffer {
                raw,
                bytes,
                readonly,
                dev: dev.clone(),
            }),
            _not_sync: PhantomData,
        }
    }

    /// `bytes` of device memory on `dev`.
    pub fn alloc(dev: &Device, bytes: u64) -> Result<Buffer, KError> {
        let mut raw = ptr::null_mut();
        // SAFETY: the device is open and `raw` is a valid out pointer; RawBuffer frees the buffer.
        check(unsafe { sys::larch_buffer_alloc(dev.raw(), bytes, &mut raw) })?;
        Ok(Buffer::new(raw, bytes, false, dev))
    }

    /// Bytes `range` of a file map as a read-only buffer, without a copy: Metal's zero-copy weights
    /// (6.4); CUDA and the CPU reference answer `Unsupported`. `range` is not empty, ends inside the
    /// file and starts on a page boundary, so a GGUF wrapped from 0 addresses each tensor by its file
    /// offset. Uploads to the buffer and op outputs on it are `InvalidArg`, and the backend's
    /// `maxBufferLength` is its `Oom`. The device keeps the map until it closes, after every stream
    /// has waited for its work, so no GPU read queued before the buffer drops finds it unmapped.
    pub fn wrap_host(dev: &Device, map: &Mmap, range: Range<u64>) -> Result<Buffer, KError> {
        let m = &map.0;
        if range.is_empty() || range.end > m.len || !range.start.is_multiple_of(m.page) {
            return Err(KError::InvalidArg);
        }
        let bytes = range.end - range.start;
        let host = m.ptr.wrapping_add(range.start as usize);
        let mut raw = ptr::null_mut();
        // SAFETY: `range` lies inside the live map, page-aligned; the backend rounds its length up to
        // whole pages, which the map covers (it spans the file rounded up to whole pages). The device
        // holds the map from here until it closes, and nothing writes the read-only pages: the buffer
        // is read-only on both sides of the ABI.
        check(unsafe { sys::larch_buffer_wrap_host(dev.raw(), host.cast(), bytes, &mut raw) })?;
        let mut maps = dev.0.maps.lock().unwrap_or_else(PoisonError::into_inner);
        if !maps.iter().any(|x| Arc::ptr_eq(x, m)) {
            maps.push(Arc::clone(m));
        }
        Ok(Buffer::new(raw, bytes, true, dev))
    }

    /// Its size as allocated.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    fn in_range(&self, off: u64, bytes: u64) -> bool {
        off.checked_add(bytes)
            .is_some_and(|end| end <= self.inner.bytes)
    }
}

/// A read-only map of a whole file (a GGUF), for [`Buffer::wrap_host`] and [`Stream::upload_map`]. Rust
/// never reads through it, only a GPU or an upload's memcpy does, so a file changed or truncated under
/// the map costs wrong values, a GPU fault (DeviceLost, D2) or SIGBUS, never Rust's memory safety.
/// On Windows the file cannot be deleted, replaced or truncated while a map of it lives, and a failing
/// page read (a drive that went away) raises an in-page error, not SIGBUS. Cloning shares the map.
#[derive(Clone)]
pub struct Mmap(Arc<Map>);

#[cfg_attr(not(any(unix, windows)), allow(dead_code))] // only unix and Windows map files
struct Map {
    ptr: *mut u8,
    /// The file's length when mapped; the map spans it rounded up to whole pages.
    len: u64,
    page: u64,
}

// SAFETY: Rust never reads or writes through `ptr` (the backend's upload memcpy only reads the
// read-only pages); the map is only unmapped, once, by Drop.
unsafe impl Send for Map {}
// SAFETY: as above.
unsafe impl Sync for Map {}

impl Mmap {
    /// Maps all of `file`, read-only. Unix and Windows; elsewhere `Unsupported`.
    pub fn map(file: &File) -> io::Result<Mmap> {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let len = file.metadata()?.len();
            let n = usize::try_from(len)
                .ok()
                .filter(|&n| n > 0)
                .ok_or(io::ErrorKind::InvalidInput)?;
            // SAFETY: a query with no arguments.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
            // SAFETY: a new read-only mapping of an open descriptor at an address the kernel picks, so
            // no existing memory changes; Map's drop unmaps it once.
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    n,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(Mmap(Arc::new(Map {
                ptr: ptr.cast(),
                len,
                page,
            })))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            let len = file.metadata()?.len();
            let n = usize::try_from(len)
                .ok()
                .filter(|&n| n > 0)
                .ok_or(io::ErrorKind::InvalidInput)?;
            // SAFETY: a new unnamed read-only section over an open handle, sized to the file (0, 0).
            let section = unsafe {
                win::CreateFileMappingW(
                    file.as_raw_handle(),
                    ptr::null(),
                    win::PAGE_READONLY,
                    0,
                    0,
                    ptr::null(),
                )
            };
            if section.is_null() {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: a read view of the whole section at an address the system picks, so no existing
            // memory changes; the view keeps the section alive, so its handle closes here once, after
            // the view's error (if any) is read, which a later call may overwrite.
            let (ptr, err) = unsafe {
                let ptr = win::MapViewOfFile(section, win::FILE_MAP_READ, 0, 0, n);
                let err = ptr.is_null().then(io::Error::last_os_error);
                let closed = win::CloseHandle(section);
                debug_assert_ne!(closed, 0, "{}", io::Error::last_os_error());
                (ptr, err)
            };
            if let Some(e) = err {
                return Err(e);
            }
            // ponytail: 4 KiB pages, the only size Windows maps files with (a view's base is 64 KiB
            // aligned, a different figure); read GetSystemInfo if a wrap_host user appears there.
            Ok(Mmap(Arc::new(Map {
                ptr: ptr.cast(),
                len,
                page: 4096,
            })))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = file;
            Err(io::ErrorKind::Unsupported.into())
        }
    }

    /// The file's length when it was mapped.
    pub fn bytes(&self) -> u64 {
        self.0.len
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping Mmap::map made, unmapped only here. No GPU read of it is queued: a device
        // that wrapped it held it until it closed, after its streams waited for their work.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len as usize);
        }
        #[cfg(windows)]
        // SAFETY: the view Mmap::map made, unmapped only here. No GPU read of it is queued: a device
        // that wrapped it held it until it closed, after its streams waited for their work.
        unsafe {
            win::UnmapViewOfFile(self.ptr.cast());
        }
    }
}

/// The four kernel32 calls a file map needs (kernel32 is in every Windows link), declared here
/// rather than a windows-sys edge in both lockfiles.
#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    pub const PAGE_READONLY: u32 = 0x02;
    pub const FILE_MAP_READ: u32 = 0x04;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *const c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn MapViewOfFile(
            section: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(base: *const c_void) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
    }
}

/// An ordered queue of work on one device. Use it from one thread at a time. On a device that wraps
/// a file map, dropping it waits for its queued work.
pub struct Stream {
    raw: *mut sys::larch_stream,
    dev: Device,
    /// Never reused in a process, unlike `raw`: a graph replays only on the stream that captured it.
    id: u64,
    /// The buffers the ops of an open [`Graph::capture`] use; `None` outside one.
    captured: RefCell<Option<Vec<Weak<RawBuffer>>>>,
    /// A capture began, so a backend with a capture arena holds it until the stream drops.
    began: Cell<bool>,
    /// The pieces [`Stream::cut`] ended so far in an open capture, each with its host tag.
    pieces: RefCell<Vec<(Graph, u64)>>,
    /// A cut failed, so the backend capture is closed: the rest of the window's ops are refused with
    /// this error (none may run eagerly) and the capture does not end it again.
    cut_failed: Cell<Option<KError>>,
}

static STREAMS: AtomicU64 = AtomicU64::new(0);

// SAFETY: a stream may move to another thread with its device; it is not Sync (Metal streams keep an
// open command buffer).
unsafe impl Send for Stream {}

impl Stream {
    /// A new stream on `dev`.
    pub fn new(dev: &Device) -> Result<Stream, KError> {
        let mut raw = ptr::null_mut();
        // SAFETY: the device is open and `raw` is a valid out pointer; Drop destroys the stream.
        check(unsafe { sys::larch_stream_create(dev.raw(), &mut raw) })?;
        Ok(Stream {
            raw,
            dev: dev.clone(),
            id: STREAMS.fetch_add(1, Ordering::Relaxed),
            captured: RefCell::default(),
            began: Cell::default(),
            pieces: RefCell::default(),
            cut_failed: Cell::default(),
        })
    }

    /// A capture has begun on this stream. CUDA and HIP then hold its scratch arena (64 MiB) until
    /// the stream drops, so a refit need not charge it again.
    pub fn capture_arena_held(&self) -> bool {
        self.began.get()
    }

    /// The device it runs on.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The stream's id, never reused in a process (a graph replays only on the stream that
    /// captured it): two streams have two ids.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// A [`Graph::capture`] is open on this stream, so host copies and syncs are refused: a model that
    /// needs them mid-pass answers `Unsupported`, and the step runs eagerly.
    pub fn capturing(&self) -> bool {
        self.captured.borrow().is_some()
    }

    /// Inside [`Graph::capture_pieces`]: ends the capture so far into a piece, with `tag` naming the
    /// host step that runs after it, and begins the next piece. `InvalidArg` outside a capture. On
    /// failure the capture is over: return the error and the capture discards every piece.
    pub fn cut(&self, tag: u64) -> Result<(), KError> {
        if let Some(e) = self.cut_failed.get() {
            return Err(e);
        }
        let Some(bufs) = self.captured.borrow_mut().as_mut().map(std::mem::take) else {
            return Err(KError::InvalidArg);
        };
        let mut raw = ptr::null_mut();
        // SAFETY: a capture is open on this stream (`captured` is set only inside one, and no cut has
        // failed), and `raw` is a valid out pointer.
        let ended = check(unsafe { sys::larch_graph_end(self.raw, &mut raw) });
        let graph = (!raw.is_null()).then(|| Graph {
            raw,
            _dev: self.dev.clone(),
            stream: self.id,
            bufs,
        });
        // SAFETY: the stream is live and its capture was just ended; begin runs only after an end
        // that succeeded, so a failed begin leaves nothing open for the capture to end.
        let began = ended.and_then(|()| check(unsafe { sys::larch_graph_begin(self.raw) }));
        match (began, graph) {
            (Ok(()), Some(g)) => {
                self.pieces.borrow_mut().push((g, tag));
                Ok(())
            }
            // Unreachable on a conforming backend; the new capture stays open for the capture to
            // end.
            (Ok(()), None) => Err(KError::InvalidArg),
            (Err(e), _) => {
                self.cut_failed.set(Some(e));
                Err(e)
            }
        }
    }

    /// Host copies, syncs, events, captures and replays never run inside a capture: a replay would
    /// touch host memory that is gone, and a graph only holds the buffers its own ops use.
    fn outside_capture(&self) -> Result<(), KError> {
        if self.captured.borrow().is_some() {
            return Err(KError::InvalidArg);
        }
        Ok(())
    }

    /// `b` is on this stream's device; inside a capture, the graph now holds it.
    fn uses(&self, b: &Buffer) -> Result<(), KError> {
        if !b.inner.dev.same(&self.dev) {
            return Err(KError::InvalidArg);
        }
        if let Some(held) = self.captured.borrow_mut().as_mut() {
            if let Some(e) = self.cut_failed.get() {
                return Err(e);
            }
            if !held
                .iter()
                .any(|w| ptr::eq(w.as_ptr(), Arc::as_ptr(&b.inner)))
            {
                held.push(Arc::downgrade(&b.inner));
            }
        }
        Ok(())
    }

    /// Waits for everything queued; a fault is `DeviceLost` or `Timeout`.
    pub fn sync(&self) -> Result<(), KError> {
        self.outside_capture()?;
        // SAFETY: the stream is live.
        check(unsafe { sys::larch_stream_sync(self.raw) })
    }

    /// Copies `src` into `buf` at byte `off`, in stream order. `src` may be reused on return: every
    /// backend copies or stages it before larch_upload returns.
    pub fn upload<T: Pod>(&self, buf: &Buffer, off: u64, src: &[T]) -> Result<(), KError> {
        let bytes: &[u8] = bytemuck::cast_slice(src);
        self.outside_capture()?;
        if buf.inner.readonly || !buf.in_range(off, bytes.len() as u64) {
            return Err(KError::InvalidArg);
        }
        self.uses(buf)?;
        // SAFETY: the range is inside the buffer, both are on this stream's device, and the backend is
        // done with `src` when the call returns (see above).
        check(unsafe {
            sys::larch_upload(
                self.raw,
                buf.inner.raw,
                off,
                bytes.as_ptr().cast(),
                bytes.len() as u64,
            )
        })
    }

    /// [`Stream::upload`] of bytes `range` of a file map, without a host copy first: the backend's
    /// staging memcpy reads the map, so Rust still never does. A file truncated under the map dies on
    /// that read (SIGBUS), as any reader of a mapped model does.
    pub fn upload_map(
        &self,
        buf: &Buffer,
        off: u64,
        map: &Mmap,
        range: Range<u64>,
    ) -> Result<(), KError> {
        self.outside_capture()?;
        let m = &map.0;
        let bytes = range.end.saturating_sub(range.start);
        if range.is_empty() || range.end > m.len || buf.inner.readonly || !buf.in_range(off, bytes)
        {
            return Err(KError::InvalidArg);
        }
        self.uses(buf)?;
        // SAFETY: `range` is inside the live map, the range `off..off + bytes` inside the buffer, both
        // on this stream's device, and the backend is done with the source when the call returns.
        check(unsafe {
            sys::larch_upload(
                self.raw,
                buf.inner.raw,
                off,
                m.ptr.wrapping_add(range.start as usize).cast(),
                bytes,
            )
        })
    }

    /// Copies from `buf` at byte `off` into `dst` after all queued work, then syncs, so `dst` holds the
    /// data on return. This is the host sync of a step: lay a step's outputs out in one buffer and
    /// download them with one call.
    pub fn download<T: Pod>(&self, buf: &Buffer, off: u64, dst: &mut [T]) -> Result<(), KError> {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
        self.outside_capture()?;
        if !buf.in_range(off, bytes.len() as u64) {
            return Err(KError::InvalidArg);
        }
        self.uses(buf)?;
        // SAFETY: the range is inside the buffer and `dst` is valid for its length. No backend writes
        // `dst` after the sync below returns, whatever its status, and it runs on both paths.
        let copied = check(unsafe {
            sys::larch_download(
                self.raw,
                buf.inner.raw,
                off,
                bytes.as_mut_ptr().cast(),
                bytes.len() as u64,
            )
        });
        let synced = self.sync();
        copied.and(synced) // the download's own error wins
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // A device unmaps the file maps it wraps when it closes, after its last stream, so on such a
        // device nothing a stream queued may outlive it (Buffer::wrap_host). The sync's status has no
        // caller to go to.
        let wrapped = !(self.dev.0.maps.lock())
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty();
        // SAFETY: created by larch_stream_create on a device that is still open.
        unsafe {
            if wrapped {
                sys::larch_stream_sync(self.raw);
            }
            sys::larch_stream_destroy(self.raw);
        }
    }
}

/// Queues a kernel that faults on the device, so the next [`Stream::sync`] is `DeviceLost`, and so is
/// every later one (sticky; nothing is reset in-process, D2). `Ok` means queued; `DeviceLost` when the
/// device was already lost. `Unsupported` off CUDA (HIP included: ROCm aborts on a GPU fault instead of
/// returning it), `InvalidArg` inside a capture. Debug builds and the `fault-inject` feature only: the
/// engine's `device_lost@step` injection (architecture 9.11).
#[cfg(any(debug_assertions, feature = "fault-inject"))]
pub fn fault_device_lost(s: &Stream) -> Result<(), KError> {
    unsafe extern "C" {
        // dispatch.cpp's debug-only entry, outside larch.h and so outside larch-sys' bindings.
        fn vv_fault_device_lost(s: *mut sys::larch_stream) -> sys::larch_status;
    }
    s.outside_capture()?;
    // SAFETY: the stream is live, and the entry is in the kernel library every build links; it checks
    // the handle and the backend's support itself.
    check(unsafe { vv_fault_device_lost(s.raw) })
}

/// A point in a stream, for GPU timing.
pub struct Event {
    raw: *mut sys::larch_event,
    dev: Device,
}

// SAFETY: from another thread elapsed_ns only reads backend state that is safe to read anywhere (CPU:
// a timestamp; Metal: a command buffer's status, waiting on committed ones).
unsafe impl Send for Event {}

impl Event {
    /// Marks the current end of `s`.
    pub fn record(s: &Stream) -> Result<Event, KError> {
        s.outside_capture()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the stream is live and `raw` is a valid out pointer; Drop destroys the event.
        check(unsafe { sys::larch_event_record(s.raw, &mut raw) })?;
        Ok(Event {
            raw,
            dev: s.dev.clone(),
        })
    }

    /// Nanoseconds from `self` to `later`, two events of one device; waits for both.
    pub fn elapsed_ns(&self, later: &Event) -> Result<u64, KError> {
        // The backend of `self` casts `later` to its own event type.
        if !self.dev.same(&later.dev) {
            return Err(KError::InvalidArg);
        }
        let mut ns = 0;
        // SAFETY: both events are live and of one device, and `ns` is a valid out pointer.
        check(unsafe { sys::larch_event_elapsed_ns(self.raw, later.raw, &mut ns) })?;
        Ok(ns)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: created by larch_event_record on a device that is still open.
        unsafe { sys::larch_event_destroy(self.raw) }
    }
}

/// A captured sequence of ops, replayed with [`Graph::launch`] on the stream that captured it.
///
/// Ownership: a graph never owns the memory it replays. While a capture is open its stream records
/// every buffer an op uses, and the graph keeps them weakly. Their owners keep them: the model its
/// weights (a wrapped map on Metal) and scratch, the engine its KV pool, step inputs (token ids,
/// positions, slots, block table, lengths) and logits. An owner may free one at any time, and from
/// then on `launch` is `InvalidArg`, so a replaced buffer means a recapture, never a replay into
/// freed memory. The graph keeps its device open, and a replay needs the capturing stream itself, so
/// scratch a backend keeps per stream or per device (split-K partials, a BLAS workspace) is alive.
///
/// A capture freezes the host values its ops were given (eps, scale, splits, rope params, tensor
/// offsets and shapes), and a replay reads whatever its buffers hold: upload each step's inputs to
/// the offsets the capture used, before `launch`. Host copies, syncs, events, nested captures and
/// replays are `InvalidArg` inside a capture, so the step's download stays outside the graph, and so
/// does `sample_prep`, whose per-row params would freeze (9.5 steps 3 and 4); the engine's sample
/// buffer is then never a graph's.
pub struct Graph {
    raw: *mut sys::larch_graph,
    /// Open until the graph is destroyed.
    _dev: Device,
    /// The id of the stream that captured it.
    stream: u64,
    bufs: Vec<Weak<RawBuffer>>,
}

// SAFETY: a graph is immutable once captured, and a replay from another thread only reaches buffers
// through the Weak handles RawBuffer's Send and Sync cover. Not Sync: a backend graph is not safe to
// launch from two threads at once.
unsafe impl Send for Graph {}

/// Runs `f` on `s` in capture mode: every buffer its ops use is recorded and returned. A panic in `f`
/// is caught, so the caller ends the capture before passing it on (the engine survives a panic under
/// `OnFault::Fail`, 9.11).
fn recorded(
    s: &Stream,
    f: impl FnOnce(&Stream) -> Result<(), KError>,
) -> (
    std::thread::Result<Result<(), KError>>,
    Vec<Weak<RawBuffer>>,
) {
    *s.captured.borrow_mut() = Some(Vec::new());
    s.cut_failed.set(None);
    let issued = panic::catch_unwind(AssertUnwindSafe(|| f(s)));
    let bufs = s.captured.borrow_mut().take().unwrap_or_default();
    (issued, bufs)
}

impl Graph {
    /// Captures the ops `f` issues on `s`. `Ok(None)` when the backend has no graphs; `f` then does not
    /// run. Also returns `Ok(None)` after a successfully ended capture if an op is unsupported
    /// during capture (for example, its bounded scratch arena does not fit); callers may run eagerly.
    /// `InvalidArg` when `s` is already capturing, or when `f` cut it (that is
    /// [`Graph::capture_pieces`]). If `f` panics, the capture is ended and dropped, then the panic goes on.
    pub fn capture(
        s: &Stream,
        f: impl FnOnce(&Stream) -> Result<(), KError>,
    ) -> Result<Option<Graph>, KError> {
        match Self::capture_pieces(s, f)? {
            Some((cut, last)) if cut.is_empty() => Ok(Some(last)),
            Some(_) => Err(KError::InvalidArg),
            None => Ok(None),
        }
    }

    /// [`Graph::capture`] for an `f` that cuts it with [`Stream::cut`]: the pieces in order, each with
    /// the tag of the host step that runs after it, then the last piece. Replay is: each piece, its
    /// host step, ..., the last. Any failure (begin, a cut, `f`, the end) discards every piece.
    #[allow(clippy::type_complexity)]
    pub fn capture_pieces(
        s: &Stream,
        f: impl FnOnce(&Stream) -> Result<(), KError>,
    ) -> Result<Option<(Vec<(Graph, u64)>, Graph)>, KError> {
        s.outside_capture()?;
        // SAFETY: the stream is live.
        match check(unsafe { sys::larch_graph_begin(s.raw) }) {
            Err(KError::Unsupported) => return Ok(None),
            r => r?,
        }
        s.began.set(true);
        let (issued, bufs) = recorded(s, f);
        let cut = std::mem::take(&mut *s.pieces.borrow_mut());
        let mut raw = ptr::null_mut();
        let ended = match s.cut_failed.take() {
            Some(e) => Err(e), // the failed cut left no capture open
            // SAFETY: capture was begun on this stream (by the begin above or the last cut) and `raw`
            // is a valid out pointer.
            None => check(unsafe { sys::larch_graph_end(s.raw, &mut raw) }),
        };
        let graph = (!raw.is_null()).then(|| Graph {
            raw,
            _dev: s.dev.clone(),
            stream: s.id,
            bufs,
        });
        let issued = match issued {
            Ok(r) => r,
            Err(p) => {
                drop((cut, graph));
                panic::resume_unwind(p)
            }
        };
        if issued == Err(KError::Unsupported) {
            if ended != Err(KError::Unsupported) {
                ended?;
            }
            return Ok(None);
        }
        issued?;
        if ended == Err(KError::Unsupported) {
            return Ok(None);
        }
        ended?;
        Ok(graph.map(|g| (cut, g)))
    }

    /// Replays the captured ops on `s`, which must be the stream that captured them and not be
    /// capturing now. `InvalidArg` once any buffer the capture used is gone.
    pub fn launch(&self, s: &Stream) -> Result<(), KError> {
        if s.id != self.stream {
            return Err(KError::InvalidArg);
        }
        s.outside_capture()?;
        // Held until the replay is queued; a buffer freed after that waits for it (RawBuffer's drop).
        let Some(_held) = self
            .bufs
            .iter()
            .map(Weak::upgrade)
            .collect::<Option<Vec<_>>>()
        else {
            return Err(KError::InvalidArg);
        };
        // SAFETY: `s` captured the graph, so the stream and its device are live and of one backend,
        // with any scratch the backend bound into the graph for them. The captured ops reach only the
        // buffers in `bufs` (every op records its tensors' buffers and pool), all held alive here, and
        // the capture made no host copy and launched no other graph (both refused inside a capture).
        check(unsafe { sys::larch_graph_launch(self.raw, s.raw) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: created by larch_graph_end on a device that is still open.
        unsafe { sys::larch_graph_destroy(self.raw) }
    }
}

/// A dense view into a buffer (no strides): `shape` row-major, dims after the first non-empty, the
/// last a whole number of quant blocks. Checked against the buffer when made.
#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    buf: &'a Buffer,
    off: u64,
    dtype: DType,
    ndim: usize,
    shape: [i64; 4],
    /// Row stride in elements of a [`Tensor::cols`] view; 0: dense.
    ld: i64,
}

impl<'a> Tensor<'a> {
    /// A view of `shape` at byte `off` of `buf`; `off` must suit the dtype ([`DType`] sizes, 2 for a
    /// quant block), so no GPU op reads a misaligned element.
    pub fn new(
        buf: &'a Buffer,
        off: u64,
        dtype: DType,
        shape: &[i64],
    ) -> Result<Tensor<'a>, KError> {
        let ndim = shape.len();
        let ok_dims = (1..=4).contains(&ndim)
            && shape[0] >= 0
            && shape[1..].iter().all(|&d| d >= 1)
            && (shape[ndim - 1] as u64).is_multiple_of(dtype.block())
            && off.is_multiple_of(dtype.align());
        if !ok_dims {
            return Err(KError::InvalidArg);
        }
        // One row first, in i64 like the strides, so a zero first dim cannot hide an overflow.
        let bytes = shape[1..]
            .iter()
            .try_fold(1i64, |n, &d| n.checked_mul(d))
            .and_then(|row| dtype.checked_bytes(row as u64))
            .and_then(|row| row.checked_mul(shape[0] as u64));
        if !bytes.is_some_and(|b| buf.in_range(off, b)) {
            return Err(KError::InvalidArg);
        }
        let mut s = [0; 4];
        s[..ndim].copy_from_slice(shape);
        Ok(Tensor {
            buf,
            off,
            dtype,
            ndim,
            shape: s,
            ld: 0,
        })
    }

    /// The dims, row-major.
    pub fn shape(&self) -> &[i64] {
        &self.shape[..self.ndim]
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Byte offset in its buffer.
    pub fn offset(&self) -> u64 {
        self.off
    }

    /// The buffer it views.
    pub fn buffer(&self) -> &'a Buffer {
        self.buf
    }

    fn numel(&self) -> Option<u64> {
        self.shape()
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
    }

    /// Bytes of one row (everything after the first dim).
    pub fn row_bytes(&self) -> u64 {
        if self.ld != 0 {
            return self.dtype.bytes(self.ld as u64);
        }
        let per_row: u64 = self.shape()[1..].iter().map(|&d| d as u64).product();
        self.dtype.bytes(per_row)
    }

    /// Rows `start..start + n` of the first dim.
    pub fn rows(&self, start: i64, n: i64) -> Result<Tensor<'a>, KError> {
        if start < 0 || n < 0 || start.checked_add(n).is_none_or(|end| end > self.shape[0]) {
            return Err(KError::InvalidArg);
        }
        let mut shape = self.shape;
        shape[0] = n;
        let t = Tensor::new(
            self.buf,
            self.off + start as u64 * self.row_bytes(),
            self.dtype,
            &shape[..self.ndim],
        )?;
        Ok(Tensor { ld: self.ld, ..t })
    }

    /// Columns `start..start + n` of a 2-D plain (not quant) tensor, its rows keeping their stride: one
    /// head of `[rows, heads * head_dim]`. Ops take it; [`Tensor::reshape`] refuses it.
    pub fn cols(&self, start: i64, n: i64) -> Result<Tensor<'a>, KError> {
        let cols = self.shape[1];
        if self.ndim != 2
            || self.dtype.block() != 1
            || start < 0
            || n < 1
            || start.checked_add(n).is_none_or(|end| end > cols)
        {
            return Err(KError::InvalidArg);
        }
        Ok(Tensor {
            off: self.off + self.dtype.bytes(start as u64),
            shape: [self.shape[0], n, 0, 0],
            ld: if self.ld != 0 { self.ld } else { cols },
            ..*self
        })
    }

    /// A 2-D plain tensor's rows as `[rows, heads, cols / heads]`, keeping their stride: a
    /// [`Tensor::cols`] view of a fused QKV as its q, k or v.
    pub fn heads(&self, heads: i64) -> Result<Tensor<'a>, KError> {
        if self.ndim != 2 || self.dtype.block() != 1 || heads < 1 || self.shape[1] % heads != 0 {
            return Err(KError::InvalidArg);
        }
        Ok(Tensor {
            ndim: 3,
            shape: [self.shape[0], heads, self.shape[1] / heads, 0],
            ..*self
        })
    }

    /// The same bytes seen with another shape of the same element count.
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor<'a>, KError> {
        let t = Tensor::new(self.buf, self.off, self.dtype, shape)?;
        if self.ld != 0 || t.numel() != self.numel() {
            return Err(KError::InvalidArg);
        }
        Ok(t)
    }

    fn raw(&self) -> sys::larch_tensor {
        let mut stride = [0i64; 4];
        let mut dense = 1;
        for d in (0..self.ndim).rev() {
            stride[d] = dense;
            let blocks = if d == self.ndim - 1 {
                self.shape[d] / self.dtype.block() as i64
            } else {
                self.shape[d]
            };
            dense *= blocks;
        }
        if self.ld != 0 {
            stride[0] = self.ld;
        }
        sys::larch_tensor {
            buf: self.buf.inner.raw,
            offset: self.off,
            dtype: self.dtype.raw(),
            ndim: self.ndim as u32,
            shape: self.shape,
            stride,
        }
    }
}

/// The paged KV pool (architecture 9.4): `k` and `v` are two buffers, each holding
/// `[n_layers][n_blocks * block_tokens][n_kv_heads][head_dim]` of f16 (`dtype` must be F16).
#[derive(Clone, Copy)]
pub struct KvPool<'a> {
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub n_blocks: u32,
    pub block_tokens: u32,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

impl KvPool<'_> {
    /// Bytes each of `k` and `v` needs.
    fn bytes(&self) -> Option<u64> {
        [
            self.n_blocks,
            self.block_tokens,
            self.n_kv_heads,
            self.head_dim,
        ]
        .iter()
        .try_fold(u64::from(self.n_layers), |n, &d| {
            n.checked_mul(u64::from(d))
        })
        .and_then(|n| self.dtype.checked_bytes(n))
    }

    /// The pool at `layer`, checked: both buffers writable (kv_write writes them) and on `s`'s device.
    fn raw(&self, s: &Stream, layer: u32) -> Result<sys::larch_kv_pool, KError> {
        let need = self.bytes().ok_or(KError::InvalidArg)?;
        let ok = layer < self.n_layers
            && self.dtype == DType::F16
            && !ptr::eq(self.k, self.v)
            && [self.k, self.v]
                .iter()
                .all(|b| !b.inner.readonly && b.inner.bytes >= need);
        if !ok {
            return Err(KError::InvalidArg);
        }
        s.uses(self.k)?;
        s.uses(self.v)?;
        Ok(sys::larch_kv_pool {
            k: self.k.inner.raw,
            v: self.v.inner.raw,
            n_blocks: self.n_blocks,
            block_tokens: self.block_tokens,
            n_layers: self.n_layers,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype.raw(),
            layer,
        })
    }
}

/// The recurrent state of one Gated DeltaNet layer group (Qwen 3.5's linear attention), in `n_slots`
/// slots per layer: `conv` f16 `[n_layers][n_slots][kernel - 1][channels]` and `ssm` f32
/// `[n_layers][n_slots][v_heads][head_dim][head_dim]`, channels = (2 k_heads + v_heads) head_dim
/// (larch.h's `larch_gdn_state`).
#[derive(Clone, Copy)]
pub struct GdnState<'a> {
    pub conv: &'a Buffer,
    pub ssm: &'a Buffer,
    pub n_slots: u32,
    pub n_layers: u32,
    pub kernel: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub head_dim: u32,
}

impl GdnState<'_> {
    /// Bytes `conv` and `ssm` need.
    pub fn bytes(&self) -> Option<(u64, u64)> {
        let per = |n: &[u32]| {
            n.iter().try_fold(
                u64::from(self.n_layers) * u64::from(self.n_slots),
                |a, &d| a.checked_mul(u64::from(d)),
            )
        };
        let ch = (2 * u64::from(self.k_heads) + u64::from(self.v_heads))
            .checked_mul(u64::from(self.head_dim))
            .and_then(|c| u32::try_from(c).ok())?;
        let conv = per(&[self.kernel.checked_sub(1)?, ch])?.checked_mul(2)?;
        let ssm = per(&[self.v_heads, self.head_dim, self.head_dim])?.checked_mul(4)?;
        Some((conv, ssm))
    }

    /// The state at `layer`, checked as [`KvPool::raw`] checks the pool: both buffers writable,
    /// distinct, large enough and on `s`'s device. The backend checks the geometry.
    fn raw(&self, s: &Stream, layer: u32) -> Result<sys::larch_gdn_state, KError> {
        let (conv, ssm) = self.bytes().ok_or(KError::InvalidArg)?;
        let ok = layer < self.n_layers
            && !ptr::eq(self.conv, self.ssm)
            && !self.conv.inner.readonly
            && !self.ssm.inner.readonly
            && self.conv.inner.bytes >= conv
            && self.ssm.inner.bytes >= ssm;
        if !ok {
            return Err(KError::InvalidArg);
        }
        s.uses(self.conv)?;
        s.uses(self.ssm)?;
        Ok(sys::larch_gdn_state {
            conv: self.conv.inner.raw,
            ssm: self.ssm.inner.raw,
            n_slots: self.n_slots,
            n_layers: self.n_layers,
            layer,
            kernel: self.kernel,
            k_heads: self.k_heads,
            v_heads: self.v_heads,
            head_dim: self.head_dim,
        })
    }
}

/// The Rust half of graph replay and of read-only buffers on the CPU reference, which has neither:
/// tests open the capture window (between larch_graph_begin and _end) and set the read-only flag
/// (wrap_host's) themselves.
#[cfg(test)]
mod tests {
    use super::*;

    fn f32s<'a>(b: &'a Buffer, off: u64) -> Tensor<'a> {
        Tensor::new(b, off, DType::F32, &[1, 4]).unwrap()
    }

    fn f16s<'a>(b: &'a Buffer, off: u64) -> Tensor<'a> {
        Tensor::new(b, off, DType::F16, &[1, 4]).unwrap()
    }

    /// Ops inside a capture hand their buffers to the graph and host work is refused there; a replay
    /// is refused on any other stream (another device's included), inside a capture and once a buffer
    /// is gone, all before any FFI call (last_error stays empty on a fresh thread), and otherwise
    /// reaches the backend. A panic inside a capture leaves the stream usable.
    #[test]
    fn graphs_hold_buffers_weakly_and_refuse_dead_ones() {
        std::thread::spawn(|| {
            let dev = Device::open(CPU).unwrap();
            let s = Stream::new(&dev).unwrap();
            let (a, b) = (
                Buffer::alloc(&dev, 16).unwrap(),
                Buffer::alloc(&dev, 8).unwrap(),
            );
            s.upload(&a, 0, &[1f32; 4]).unwrap();
            s.upload(&b, 0, &[0u16; 4]).unwrap();
            let (at, bt) = (f32s(&a, 0), f16s(&b, 0));
            let graph = |bufs| Graph {
                raw: ptr::null_mut(),
                _dev: dev.clone(),
                stream: s.id,
                bufs,
            };
            let early = graph(Vec::new());

            // The window between larch_graph_begin and _end, which the CPU reference never opens.
            let (issued, bufs) = recorded(&s, |s| {
                ops::add(s, &at, &bt, &at)?;
                ops::add(s, &at, &bt, &at)?; // the same two buffers, held once
                assert_eq!(s.upload(&a, 0, &[0f32; 4]), Err(KError::InvalidArg));
                assert_eq!(s.download(&a, 0, &mut [0f32; 4]), Err(KError::InvalidArg));
                assert_eq!(s.sync(), Err(KError::InvalidArg));
                #[cfg(any(debug_assertions, feature = "fault-inject"))]
                assert_eq!(fault_device_lost(s), Err(KError::InvalidArg));
                assert!(matches!(Event::record(s), Err(KError::InvalidArg)));
                assert!(matches!(
                    Graph::capture(s, |_| Ok(())),
                    Err(KError::InvalidArg)
                ));
                early.launch(s)
            });
            assert_eq!(issued.unwrap(), Err(KError::InvalidArg));
            assert_eq!(bufs.len(), 2);
            let g = graph(bufs);

            assert_eq!(
                g.launch(&Stream::new(&dev).unwrap()),
                Err(KError::InvalidArg)
            );
            let other = Stream::new(&Device::open(CPU).unwrap()).unwrap();
            assert_eq!(g.launch(&other), Err(KError::InvalidArg));
            drop(b);
            assert_eq!(g.launch(&s), Err(KError::InvalidArg));
            assert_eq!(last_error(), "");

            // Every buffer alive: the call reaches the backend, which refuses this null graph.
            let live = graph(vec![Arc::downgrade(&a.inner)]);
            assert_eq!(live.launch(&s), Err(KError::InvalidArg));
            assert!(last_error().contains("null handle"));

            let (issued, _) = recorded(&s, |_| panic!("inside a capture"));
            assert!(issued.is_err());
            s.upload(&a, 0, &[2f32; 4]).unwrap();
        })
        .join()
        .unwrap();
    }

    /// A cut the backend refuses (the CPU reference's Unsupported end) keeps no piece and refuses
    /// the window's later ops with its error, so none runs outside a capture; the stream is usable
    /// once the window closes.
    #[test]
    fn a_refused_cut_keeps_no_piece_and_stops_the_window() {
        std::thread::spawn(|| {
            let dev = Device::open(CPU).unwrap();
            let s = Stream::new(&dev).unwrap();
            let (a, b) = (
                Buffer::alloc(&dev, 16).unwrap(),
                Buffer::alloc(&dev, 8).unwrap(),
            );
            s.upload(&a, 0, &[1f32; 4]).unwrap();
            s.upload(&b, 0, &[0x3c00u16; 4]).unwrap(); // f16 1.0
            let (at, bt) = (f32s(&a, 0), f16s(&b, 0));
            let (issued, _) = recorded(&s, |s| {
                ops::add(s, &at, &bt, &at)?; // the CPU reference runs it: no real capture
                assert_eq!(s.cut(7), Err(KError::Unsupported));
                assert!(s.pieces.borrow().is_empty());
                assert_eq!(ops::add(s, &at, &bt, &at), Err(KError::Unsupported));
                assert_eq!(s.cut(8), Err(KError::Unsupported));
                Ok(())
            });
            issued.unwrap().unwrap();
            assert!(s.pieces.borrow().is_empty());
            ops::add(&s, &at, &bt, &at).unwrap();
            let mut got = [0f32; 4];
            s.download(&a, 0, &mut got).unwrap();
            assert_eq!(got, [3.0; 4]);
            let (issued, _) = recorded(&s, |s| ops::add(s, &at, &bt, &at));
            assert_eq!(issued.unwrap(), Ok(()), "a new window starts clean");
        })
        .join()
        .unwrap();
    }

    /// A wrapped buffer is read, never written: no upload, no op output (in place or not), no KV pool.
    /// The same calls succeed once the flag is cleared.
    #[test]
    fn read_only_buffers_are_never_written() {
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let (mut ro, rw) = (
            Buffer::alloc(&dev, 64).unwrap(),
            Buffer::alloc(&dev, 64).unwrap(),
        );
        s.upload(&ro, 0, &[0.5f32; 16]).unwrap();
        s.upload(&rw, 0, &[0f32; 16]).unwrap();
        let writes = |ro: &Buffer| {
            let (r, w, h) = (f32s(ro, 0), f32s(&rw, 0), f16s(&rw, 32));
            ops::add(&s, &r, &h, &w).unwrap(); // a read-only input is fine
            let qk = Tensor::new(ro, 0, DType::F16, &[1, 1, 4]).unwrap();
            let pool = KvPool {
                k: ro,
                v: &rw,
                n_blocks: 1,
                block_tokens: 1,
                n_layers: 1,
                n_kv_heads: 1,
                head_dim: 4,
                dtype: DType::F16,
            };
            let slot = Tensor::new(&rw, 48, DType::I32, &[1]).unwrap();
            [
                s.upload(ro, 0, &[0.5f32]),
                ops::add(&s, &w, &h, &r),
                ops::qk_norm(&s, &qk, &w.reshape(&[4]).unwrap(), 1e-6),
                ops::kv_write(
                    &s,
                    &h.reshape(&[1, 1, 4]).unwrap(),
                    &h.reshape(&[1, 1, 4]).unwrap(),
                    &slot,
                    &pool,
                    0,
                ),
            ]
        };

        Arc::get_mut(&mut ro.inner).unwrap().readonly = true;
        assert_eq!(writes(&ro), [Err(KError::InvalidArg); 4]);
        let mut back = [0f32; 16];
        s.download(&ro, 0, &mut back).unwrap();
        assert_eq!(back, [0.5; 16]);
        Arc::get_mut(&mut ro.inner).unwrap().readonly = false;
        assert_eq!(writes(&ro), [Ok(()); 4]);
    }

    /// A GDN state view is refused before the FFI unless its buffers are distinct, writable and large
    /// enough, the layer in range and the geometry's sizes representable; a kernel of 0 has no size.
    #[test]
    fn gdn_state_checks_its_buffers() {
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        // 2 layers x 2 slots; channels (2 x 1 + 2) x 2 = 8.
        let (conv, ssm) = (2 * 2 * 3 * 8 * 2, 2 * 2 * 2 * 2 * 2 * 4);
        fn st<'a>(conv: &'a Buffer, ssm: &'a Buffer, kernel: u32) -> GdnState<'a> {
            GdnState {
                conv,
                ssm,
                n_slots: 2,
                n_layers: 2,
                kernel,
                k_heads: 1,
                v_heads: 2,
                head_dim: 2,
            }
        }
        let c = Buffer::alloc(&dev, conv).unwrap();
        let mut m = Buffer::alloc(&dev, ssm).unwrap();
        assert_eq!(st(&c, &m, 4).bytes(), Some((conv, ssm)));
        assert!(st(&c, &m, 4).raw(&s, 1).is_ok());
        let refused = [
            st(&c, &m, 4).raw(&s, 2),
            st(&c, &m, 5).raw(&s, 0),
            st(&c, &m, 0).raw(&s, 0),
            st(&c, &c, 4).raw(&s, 0),
        ];
        assert_eq!(refused.map(|r| r.err()), [Some(KError::InvalidArg); 4]);
        assert_eq!(st(&c, &m, 0).bytes(), None);
        Arc::get_mut(&mut m.inner).unwrap().readonly = true;
        assert_eq!(st(&c, &m, 4).raw(&s, 0).err(), Some(KError::InvalidArg));
    }

    /// A tri-state route switch is on for `1`, off for `0` and the measured-arch default otherwise.
    #[test]
    fn switch_is_one_zero_or_the_default() {
        assert_eq!(switch(Some("1")), Some(true));
        assert_eq!(switch(Some("0")), Some(false));
        for other in [None, Some(""), Some("true"), Some("2")] {
            assert_eq!(switch(other), None, "{other:?}");
        }
    }
}
