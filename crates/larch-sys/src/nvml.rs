//! NVML, loaded at run time for the engine's 1 Hz device sampler (architecture 9.6): a machine
//! without the NVIDIA driver has no NVML, so nothing links against it.

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};

/// One NVML reading: the value, or why there is none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read<T> {
    Value(T),
    /// NVML answered NOT_SUPPORTED, or the driver lacks the call: this device never has it.
    Unsupported,
    /// Any other error: no value this time.
    Failed,
}

impl<T> Read<T> {
    /// The value through `f`, or the same reason for none.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Read<U> {
        match self {
            Read::Value(v) => Read::Value(f(v)),
            Read::Unsupported => Read::Unsupported,
            Read::Failed => Read::Failed,
        }
    }
}

/// One sample of the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub sm_clock_mhz: Read<u32>,
    pub mem_clock_mhz: Read<u32>,
    pub temp_c: Read<u32>,
    pub power_mw: Read<u32>,
    pub power_limit_mw: Read<u32>,
    /// Percent of the last sample period a kernel ran.
    pub gpu_util: Read<u32>,
    /// Used and total bytes.
    pub memory: Read<(u64, u64)>,
    /// NVML's clock-event (throttle) reasons bitmask.
    pub clocks_event_reasons: Read<u64>,
    /// Compute processes besides this one. Unsupported when NVML's list leaves this process out
    /// (WSL's lists none), since then it cannot tell this process from another.
    pub other_procs: Read<u32>,
}

type Ret = c_int;
const SUCCESS: Ret = 0;
const NOT_SUPPORTED: Ret = 3;
const INSUFFICIENT_SIZE: Ret = 7;
/// The driver has the symbol but not the function behind it.
const FUNCTION_NOT_FOUND: Ret = 13;
const CLOCK_SM: c_int = 1;
const CLOCK_MEM: c_int = 2;
const TEMPERATURE_GPU: c_int = 0;

type Device = *mut c_void;

#[repr(C)]
struct Memory {
    total: c_ulonglong,
    _free: c_ulonglong,
    used: c_ulonglong,
}

#[repr(C)]
struct Utilization {
    gpu: c_uint,
    _memory: c_uint,
}

/// `nvmlProcessInfo_t` (the v2/v3 layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct ProcessInfo {
    pid: c_uint,
    _used_gpu_memory: c_ulonglong,
    _gpu_instance_id: c_uint,
    _compute_instance_id: c_uint,
}

/// The NVML entry points the sampler calls; a test fills it with fakes.
struct Api {
    shutdown: unsafe extern "C" fn() -> Ret,
    count: unsafe extern "C" fn(*mut c_uint) -> Ret,
    handle: unsafe extern "C" fn(c_uint, *mut Device) -> Ret,
    name: unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Ret,
    memory: unsafe extern "C" fn(Device, *mut Memory) -> Ret,
    clock: unsafe extern "C" fn(Device, c_int, *mut c_uint) -> Ret,
    temperature: unsafe extern "C" fn(Device, c_int, *mut c_uint) -> Ret,
    power: unsafe extern "C" fn(Device, *mut c_uint) -> Ret,
    power_limit: unsafe extern "C" fn(Device, *mut c_uint) -> Ret,
    utilization: unsafe extern "C" fn(Device, *mut Utilization) -> Ret,
    /// Older drivers name it ...ClocksThrottleReasons; either may be absent.
    reasons: Option<unsafe extern "C" fn(Device, *mut c_ulonglong) -> Ret>,
    processes: Option<unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> Ret>,
    driver: Option<unsafe extern "C" fn(*mut c_char, c_uint) -> Ret>,
}

/// An open NVML and the device the engine runs on. NVML shuts down when it drops.
pub struct Nvml {
    api: Api,
    device: Device,
    _lib: Option<libloading::Library>,
}

// SAFETY: every NVML call is thread-safe (nvml.h), and `device` is an opaque handle that stays valid
// until nvmlShutdown, which only `drop` calls.
unsafe impl Send for Nvml {}
// SAFETY: as for Send; `sample` takes `&self` and NVML serializes internally.
unsafe impl Sync for Nvml {}

impl Nvml {
    /// Opens NVML and the device CUDA calls `name`: the only device, or the one device of that
    /// name. None without the driver, or with two devices of that name.
    // ponytail: matched by name; carry the PCI bus id in the device props when two identical
    // cards serve.
    pub fn open(name: &str) -> Option<Nvml> {
        const LIB: &str = if cfg!(windows) {
            "nvml.dll"
        } else {
            "libnvidia-ml.so.1"
        };
        // SAFETY: loading the driver's NVML runs only its own initialisers.
        let lib = unsafe { libloading::Library::new(LIB) }.ok()?;
        // SAFETY: each symbol is looked up under the name nvml.h declares with exactly this
        // signature; a missing optional one is None.
        let (init, api) = unsafe {
            macro_rules! sym {
                ($($name:literal)|+) => {
                    None$(.or_else(|| lib.get(concat!($name, "\0").as_bytes()).ok().map(|s| *s)))+
                };
            }
            let init: unsafe extern "C" fn() -> Ret = sym!("nvmlInit_v2")?;
            let api = Api {
                shutdown: sym!("nvmlShutdown")?,
                count: sym!("nvmlDeviceGetCount_v2")?,
                handle: sym!("nvmlDeviceGetHandleByIndex_v2")?,
                name: sym!("nvmlDeviceGetName")?,
                memory: sym!("nvmlDeviceGetMemoryInfo")?,
                clock: sym!("nvmlDeviceGetClockInfo")?,
                temperature: sym!("nvmlDeviceGetTemperature")?,
                power: sym!("nvmlDeviceGetPowerUsage")?,
                power_limit: sym!("nvmlDeviceGetEnforcedPowerLimit")?,
                utilization: sym!("nvmlDeviceGetUtilizationRates")?,
                reasons: sym!(
                    "nvmlDeviceGetCurrentClocksEventReasons"
                        | "nvmlDeviceGetCurrentClocksThrottleReasons"
                ),
                processes: sym!(
                    "nvmlDeviceGetComputeRunningProcesses_v3"
                        | "nvmlDeviceGetComputeRunningProcesses_v2"
                ),
                driver: sym!("nvmlSystemGetDriverVersion"),
            };
            (init, api)
        };
        // SAFETY: nvmlInit_v2 takes no arguments; a failed init needs no shutdown.
        if unsafe { init() } != SUCCESS {
            return None;
        }
        Self::with(api, Some(lib), name)
    }

    /// Picks the device over `api`, which is initialised; shuts it down when none fits.
    fn with(api: Api, lib: Option<libloading::Library>, name: &str) -> Option<Nvml> {
        let mut n: c_uint = 0;
        let mut devices = Vec::new();
        // SAFETY: the out-pointers are live locals for the duration of each call.
        unsafe {
            if (api.count)(&mut n) == SUCCESS {
                for i in 0..n {
                    let mut d: Device = std::ptr::null_mut();
                    if (api.handle)(i, &mut d) == SUCCESS {
                        devices.push(d);
                    }
                }
            }
        }
        let named = |&d: &Device| {
            let mut buf = [0 as c_char; 96];
            // SAFETY: `buf` holds `buf.len()` bytes and NVML NUL-terminates what it writes there.
            let ok = unsafe { (api.name)(d, buf.as_mut_ptr(), buf.len() as c_uint) } == SUCCESS;
            // SAFETY: on success `buf` holds a NUL-terminated string.
            ok && unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_bytes() == name.as_bytes()
        };
        let device = match devices[..] {
            [only] if n == 1 => Some(only),
            _ => match devices.iter().filter(|d| named(d)).collect::<Vec<_>>()[..] {
                [&one] => Some(one),
                _ => None,
            },
        };
        let nvml = Nvml {
            api,
            device: device.unwrap_or(std::ptr::null_mut()),
            _lib: lib,
        };
        device.map(|_| nvml)
    }

    /// The installed driver's build, such as "581.57"; None where NVML cannot say.
    pub fn driver_version(&self) -> Option<String> {
        let mut buf = [0 as c_char; 80];
        // SAFETY: `buf` holds `buf.len()` bytes (NVML_SYSTEM_DRIVER_VERSION_V2_BUFFER_SIZE) and
        // NVML NUL-terminates what it writes there.
        let ok = unsafe { (self.api.driver?)(buf.as_mut_ptr(), buf.len() as c_uint) } == SUCCESS;
        ok.then(|| {
            // SAFETY: on success `buf` holds a NUL-terminated string.
            unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
    }

    /// Reads every field once.
    pub fn sample(&self) -> Sample {
        let (a, d) = (&self.api, self.device);
        // Every call below gets `d`, a live handle, and out-pointers to live locals.
        let read_u32 = |f: &dyn Fn(*mut c_uint) -> Ret| {
            let mut v: c_uint = 0;
            read(f(&mut v), v)
        };
        let mut memory = Memory {
            total: 0,
            _free: 0,
            used: 0,
        };
        // SAFETY: see above.
        let ret = unsafe { (a.memory)(d, &mut memory) };
        let memory = read(ret, (memory.used, memory.total));
        let mut util = Utilization { gpu: 0, _memory: 0 };
        // SAFETY: see above.
        let gpu_util = read(unsafe { (a.utilization)(d, &mut util) }, util.gpu);
        let clocks_event_reasons = a.reasons.map_or(Read::Unsupported, |f| {
            let mut v: c_ulonglong = 0;
            // SAFETY: see above.
            read(unsafe { f(d, &mut v) }, v)
        });
        Sample {
            // SAFETY: see above.
            sm_clock_mhz: read_u32(&|v| unsafe { (a.clock)(d, CLOCK_SM, v) }),
            // SAFETY: see above.
            mem_clock_mhz: read_u32(&|v| unsafe { (a.clock)(d, CLOCK_MEM, v) }),
            // SAFETY: see above.
            temp_c: read_u32(&|v| unsafe { (a.temperature)(d, TEMPERATURE_GPU, v) }),
            // SAFETY: see above.
            power_mw: read_u32(&|v| unsafe { (a.power)(d, v) }),
            // SAFETY: see above.
            power_limit_mw: read_u32(&|v| unsafe { (a.power_limit)(d, v) }),
            gpu_util,
            memory,
            clocks_event_reasons,
            other_procs: a.processes.map_or(Read::Unsupported, |f| others(f, d)),
        }
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: NVML was initialised by `open`, and no call outlives `self`.
        unsafe { (self.api.shutdown)() };
    }
}

fn read<T>(ret: Ret, v: T) -> Read<T> {
    match ret {
        SUCCESS => Read::Value(v),
        NOT_SUPPORTED | FUNCTION_NOT_FOUND => Read::Unsupported,
        _ => Read::Failed,
    }
}

/// Compute processes on `d` besides this one, from a list that grows once if NVML asks for more.
fn others(
    f: unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> Ret,
    d: Device,
) -> Read<u32> {
    let empty = ProcessInfo {
        pid: 0,
        _used_gpu_memory: 0,
        _gpu_instance_id: 0,
        _compute_instance_id: 0,
    };
    let mut list = vec![empty; 64];
    for _ in 0..2 {
        let mut n = list.len() as c_uint;
        // SAFETY: `list` holds `n` entries, and NVML writes at most `n`, or sets `n` to the count it
        // needs and writes nothing.
        match unsafe { f(d, &mut n, list.as_mut_ptr()) } {
            SUCCESS => {
                let me = std::process::id();
                let list = &list[..n as usize];
                return if list.iter().any(|p| p.pid == me) {
                    Read::Value(list.iter().filter(|p| p.pid != me).count() as u32)
                } else {
                    Read::Unsupported
                };
            }
            INSUFFICIENT_SIZE => list.resize(n as usize + 8, empty),
            ret => return read(ret, 0),
        }
    }
    Read::Failed
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn ok() -> Ret {
        SUCCESS
    }
    unsafe extern "C" fn count(n: *mut c_uint) -> Ret {
        // SAFETY: the caller passes a live out-pointer.
        unsafe { *n = 1 };
        SUCCESS
    }
    unsafe extern "C" fn handle(_: c_uint, d: *mut Device) -> Ret {
        // SAFETY: as above.
        unsafe { *d = std::ptr::dangling_mut() };
        SUCCESS
    }
    unsafe extern "C" fn name(_: Device, _: *mut c_char, _: c_uint) -> Ret {
        SUCCESS
    }
    unsafe extern "C" fn memory(_: Device, m: *mut Memory) -> Ret {
        // SAFETY: as above.
        unsafe {
            (*m).total = 16 << 30;
            (*m).used = 3 << 30;
        }
        SUCCESS
    }
    unsafe extern "C" fn clock(_: Device, kind: c_int, v: *mut c_uint) -> Ret {
        // SAFETY: as above.
        unsafe { *v = if kind == CLOCK_SM { 2800 } else { 14000 } };
        SUCCESS
    }
    unsafe extern "C" fn not_supported(_: Device, _: c_int, _: *mut c_uint) -> Ret {
        NOT_SUPPORTED
    }
    unsafe extern "C" fn unknown_error(_: Device, _: *mut c_uint) -> Ret {
        999
    }
    unsafe extern "C" fn limit(_: Device, v: *mut c_uint) -> Ret {
        // SAFETY: as above.
        unsafe { *v = 300_000 };
        SUCCESS
    }
    unsafe extern "C" fn utilization(_: Device, u: *mut Utilization) -> Ret {
        // SAFETY: as above.
        unsafe { (*u).gpu = 97 };
        SUCCESS
    }
    /// This process and two others, after asking for room for 70.
    unsafe extern "C" fn processes(_: Device, n: *mut c_uint, list: *mut ProcessInfo) -> Ret {
        // SAFETY: the caller passes `*n` live entries.
        unsafe {
            if *n < 70 {
                *n = 70;
                return INSUFFICIENT_SIZE;
            }
            *n = 3;
            for (i, pid) in [1, std::process::id(), 2].into_iter().enumerate() {
                (*list.add(i)).pid = pid;
            }
        }
        SUCCESS
    }
    /// WSL's answer: an empty list.
    unsafe extern "C" fn no_processes(_: Device, n: *mut c_uint, _: *mut ProcessInfo) -> Ret {
        // SAFETY: as above.
        unsafe { *n = 0 };
        SUCCESS
    }

    unsafe extern "C" fn driver(buf: *mut c_char, n: c_uint) -> Ret {
        let v = b"581.57\0";
        assert!(v.len() <= n as usize);
        // SAFETY: the caller's buffer holds `n` bytes, at least `v.len()`.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr().cast(), buf, v.len()) };
        SUCCESS
    }

    #[test]
    fn the_driver_build_reads_as_nvml_writes_it() {
        assert_eq!(fake(processes).driver_version().as_deref(), Some("581.57"));
    }

    fn fake(
        processes_fn: unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> Ret,
    ) -> Nvml {
        let api = Api {
            shutdown: ok,
            count,
            handle,
            name,
            memory,
            clock,
            temperature: not_supported,
            power: unknown_error,
            power_limit: limit,
            utilization,
            reasons: None,
            processes: Some(processes_fn),
            driver: Some(driver),
        };
        Nvml::with(api, None, "any").expect("the only device")
    }

    /// NOT_SUPPORTED and a missing call are Unsupported, any other error Failed (Missing in the
    /// record), never a value; the process count leaves this process out, or is Unsupported when
    /// the list does.
    #[test]
    fn errors_map_to_their_states() {
        let s = fake(processes).sample();
        assert_eq!(
            (
                s.sm_clock_mhz,
                s.mem_clock_mhz,
                s.power_limit_mw,
                s.gpu_util
            ),
            (
                Read::Value(2800),
                Read::Value(14000),
                Read::Value(300_000),
                Read::Value(97)
            )
        );
        assert_eq!(s.memory, Read::Value((3 << 30, 16 << 30)));
        assert_eq!(
            (s.temp_c, s.power_mw, s.clocks_event_reasons),
            (Read::Unsupported, Read::Failed, Read::Unsupported)
        );
        assert_eq!(s.other_procs, Read::Value(2));
        assert_eq!(fake(no_processes).sample().other_procs, Read::Unsupported);
        assert_eq!(read(FUNCTION_NOT_FOUND, ()), Read::Unsupported);
    }
}
