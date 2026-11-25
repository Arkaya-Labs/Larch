#[cfg(feature = "kernels")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "kernels")]
#[allow(unused_imports)]
pub use ffi::*;

#[cfg(windows)]
pub mod appcontainer;
pub mod nvml;
#[cfg(windows)]
pub mod pipe;

/// `"cuda"`, `"hip"`, `"metal"` or `"cpu"`: the GPU backend the kernels were built with. The CPU
/// reference is always built as well.
#[cfg(feature = "kernels")]
pub const BACKEND: &str = env!("LARCH_BUILT_BACKEND");

/// Ends the process now with `code`, skipping the user-mode teardown `std::process::exit` runs and a
/// sticky device fault or a TDR can hang (architecture 9.11, #73): the C exit handlers on Unix, every
/// module's process-detach callback (DLL detach: cuBLAS, the driver's user-mode part) on Windows.
/// The kernel driver still cleans up as the process's handles close, and that counts toward the
/// exit gap. Nothing is flushed, so callers write what must survive (the ring, the fault line) first.
pub fn exit_now(code: i32) -> ! {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn _exit(status: std::ffi::c_int) -> !;
        }
        // SAFETY: _exit takes any status, never returns and runs no user code.
        unsafe { _exit(code) }
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn TerminateProcess(process: *mut std::ffi::c_void, code: u32) -> i32;
        }
        // SAFETY: GetCurrentProcess returns a pseudo-handle that is always valid for this process,
        // and TerminateProcess only reads it; neither touches Rust state.
        if unsafe { TerminateProcess(GetCurrentProcess(), code as u32) } != 0 {
            // Termination is asynchronous: this thread waits for it rather than return.
            loop {
                std::thread::park();
            }
        }
        std::process::abort()
    }
}
