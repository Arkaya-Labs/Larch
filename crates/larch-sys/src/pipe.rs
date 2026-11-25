//! The task service's local transport on Windows: a named pipe only this user may open, refusing
//! remote clients, with a deadline on every read and write as the Unix socket's timeouts give.

use std::{
    ffi::OsStr,
    io::{self, Read, Write},
    mem, ptr,
    time::{Duration, Instant},
};

use windows_sys::Win32::{
    Foundation::{
        CloseHandle, ERROR_BROKEN_PIPE, ERROR_IO_PENDING, ERROR_NO_DATA, ERROR_OPERATION_ABORTED,
        ERROR_PIPE_BUSY, ERROR_PIPE_CONNECTED, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE,
        INVALID_HANDLE_VALUE, LocalFree, WAIT_OBJECT_0,
    },
    Security::{
        Authorization::{ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1},
        GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
        TokenUser,
    },
    Storage::FileSystem::{
        CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED, OPEN_EXISTING,
        PIPE_ACCESS_DUPLEX, ReadFile, SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT, WriteFile,
    },
    System::{
        IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED},
        Pipes::{
            ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, GetNamedPipeServerProcessId,
            PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_WAIT,
            WaitNamedPipeW,
        },
        Threading::{
            CreateEventW, GetCurrentProcess, INFINITE, OpenProcess, OpenProcessToken,
            PROCESS_QUERY_LIMITED_INFORMATION, WaitForSingleObject,
        },
    },
};

use crate::appcontainer::{sid_string, wide};

/// One connected end of the pipe, client or server.
pub struct PipeStream {
    handle: HANDLE,
    event: HANDLE,
    timeout: u32,
}

// SAFETY: both handles are kernel objects owned by this value alone; any thread may use them.
unsafe impl Send for PipeStream {}

impl PipeStream {
    /// Opens the pipe `name` (`\\.\pipe\…`), waiting up to `timeout` while every instance is busy.
    /// The server may identify the caller but never impersonate it.
    pub fn connect(name: &str, timeout: Duration) -> io::Result<Self> {
        millis(timeout)?;
        let wide = wide(OsStr::new(name));
        let deadline = Instant::now() + timeout;
        loop {
            // SAFETY: `wide` is NUL-terminated and outlives the call; no security attributes or
            // template are passed.
            let handle = unsafe {
                CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    ptr::null(),
                    OPEN_EXISTING,
                    FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION,
                    ptr::null_mut(),
                )
            };
            if handle != INVALID_HANDLE_VALUE {
                return Self::new(handle, timeout);
            }
            // SAFETY: reads this thread's last error only.
            let error = unsafe { GetLastError() };
            let left = deadline.saturating_duration_since(Instant::now());
            if error != ERROR_PIPE_BUSY
                || left.is_zero()
                // SAFETY: WaitNamedPipeW only reads the NUL-terminated name.
                || unsafe { WaitNamedPipeW(wide.as_ptr(), millis(left)?) } == 0
            {
                return Err(io::Error::from_raw_os_error(error as i32));
            }
        }
    }

    /// Sets the deadline each later read and write gets; zero is refused, as on a Unix socket.
    pub fn set_timeout(&mut self, timeout: Duration) -> io::Result<()> {
        self.timeout = millis(timeout)?;
        Ok(())
    }

    /// Whether the process serving this pipe runs as this user. The client checks it before
    /// sending its token, as the Unix client checks the socket peer's UID.
    pub fn server_is_current_user(&self) -> io::Result<bool> {
        let mut pid = 0;
        // SAFETY: the handle is an open client end of a pipe; `pid` receives the server's id.
        if unsafe { GetNamedPipeServerProcessId(self.handle, &mut pid) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a query-only handle to that process, closed below; null on failure.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return Err(io::Error::last_os_error());
        }
        let user = process_user(process);
        // SAFETY: `process` is open and owned here only.
        unsafe { CloseHandle(process) };
        Ok(user? == current_user()?)
    }

    fn new(handle: HANDLE, timeout: Duration) -> io::Result<Self> {
        let timeout = match millis(timeout) {
            Ok(timeout) => timeout,
            Err(e) => {
                // SAFETY: `handle` is open and owned here only.
                unsafe { CloseHandle(handle) };
                return Err(e);
            }
        };
        // SAFETY: a manual-reset, unnamed event with default security.
        let event = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
        if event.is_null() {
            let e = io::Error::last_os_error();
            // SAFETY: `handle` is open and owned here only.
            unsafe { CloseHandle(handle) };
            return Err(e);
        }
        Ok(Self {
            handle,
            event,
            timeout,
        })
    }

    /// Runs one overlapped operation to its end, cancelling it at the deadline.
    fn wait(&self, start: impl FnOnce(*mut OVERLAPPED) -> i32) -> io::Result<u32> {
        // SAFETY: OVERLAPPED is plain data, valid zeroed.
        let mut overlapped: OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = self.event;
        if start(&mut overlapped) == 0 {
            // SAFETY: reads this thread's last error only.
            let error = unsafe { GetLastError() };
            if error != ERROR_IO_PENDING {
                return Err(io::Error::from_raw_os_error(error as i32));
            }
            // SAFETY: the event is ours and open; unless it signalled (a timeout or a failed
            // wait) the operation is cancelled, and GetOverlappedResult below waits for it to
            // finish before `overlapped` is dropped, so it never blocks without a deadline.
            if unsafe { WaitForSingleObject(self.event, self.timeout) } != WAIT_OBJECT_0 {
                // SAFETY: cancels only the operation `overlapped` names, on our open handle.
                unsafe { CancelIoEx(self.handle, &overlapped) };
            }
        }
        let mut n = 0;
        // SAFETY: `overlapped` is the one the operation was started with; bWait makes this
        // return only once the operation no longer uses it or the buffer.
        if unsafe { GetOverlappedResult(self.handle, &overlapped, &mut n, 1) } == 0 {
            // SAFETY: reads this thread's last error only.
            return Err(match unsafe { GetLastError() } {
                ERROR_OPERATION_ABORTED => io::ErrorKind::TimedOut.into(),
                error => io::Error::from_raw_os_error(error as i32),
            });
        }
        Ok(n)
    }
}

impl Read for PipeStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(u32::MAX as usize) as u32;
        let read = self.wait(|o| {
            // SAFETY: `buf` holds `len` writable bytes and outlives the operation, which `wait`
            // finishes before returning.
            unsafe { ReadFile(self.handle, buf.as_mut_ptr(), len, ptr::null_mut(), o) }
        });
        match read {
            Ok(n) => Ok(n as usize),
            // The other end closed after its last write: end of stream.
            Err(e) if e.raw_os_error() == Some(ERROR_BROKEN_PIPE as i32) => Ok(0),
            Err(e) => Err(e),
        }
    }
}

impl Write for PipeStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len().min(u32::MAX as usize) as u32;
        self.wait(|o| {
            // SAFETY: `buf` holds `len` readable bytes and outlives the operation, as in `read`.
            unsafe { WriteFile(self.handle, buf.as_ptr(), len, ptr::null_mut(), o) }
        })
        .map(|n| n as usize)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for PipeStream {
    fn drop(&mut self) {
        // SAFETY: both handles are open and owned by this value only. Bytes already written stay
        // readable by the other end after the close.
        unsafe {
            CloseHandle(self.handle);
            CloseHandle(self.event);
        }
    }
}

/// The server end. One instance always listens, even while another serves a client, so the
/// name never lapses for a client to miss or another process to create.
pub struct PipeListener {
    name: Vec<u16>,
    security: PSECURITY_DESCRIPTOR,
    event: HANDLE,
    next: HANDLE,
}

// SAFETY: the handles are kernel objects and the descriptor heap memory, all owned by this value
// alone; any thread may use them.
unsafe impl Send for PipeListener {}

/// Instances one listener may have: the listening one and the one being served, with room to spare.
const MAX_INSTANCES: u32 = 4;
/// Consecutive accept failures, 100 ms apart, before the listener gives up.
const ACCEPT_ATTEMPTS: u32 = 50;

impl PipeListener {
    /// Creates the pipe `name`, failing if any process already has a pipe by that name, so a
    /// squatter's instance is never served.
    pub fn bind(name: &str) -> io::Result<Self> {
        let sddl = wide(OsStr::new(&format!("D:P(A;;GA;;;{})", current_user()?)));
        let mut security = ptr::null_mut();
        // SAFETY: `sddl` is NUL-terminated; `security` receives a LocalAlloc'd descriptor the
        // listener frees.
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut security,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut listener = Self {
            name: wide(OsStr::new(name)),
            security,
            event: ptr::null_mut(),
            next: INVALID_HANDLE_VALUE,
        };
        // SAFETY: a manual-reset, unnamed event with default security, closed by Drop.
        listener.event = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
        if listener.event.is_null() {
            return Err(io::Error::last_os_error());
        }
        listener.next = listener.instance(FILE_FLAG_FIRST_PIPE_INSTANCE)?;
        Ok(listener)
    }

    /// Waits for the next client; its stream gets `timeout` on each read and write. A client that
    /// connects and leaves before the call is skipped. Other failures recycle the instance and
    /// retry; only failures that persist for five seconds are returned.
    pub fn accept(&mut self, timeout: Duration) -> io::Result<PipeStream> {
        let mut failures = 0;
        loop {
            let error = match self.connect().and_then(|()| self.instance(0)) {
                // The replacement listens before the connected instance is handed out.
                Ok(replacement) => {
                    match PipeStream::new(mem::replace(&mut self.next, replacement), timeout) {
                        Ok(stream) => return Ok(stream),
                        Err(e) => e,
                    }
                }
                Err(e) => e,
            };
            // SAFETY: `next` is this listener's open server instance; disconnecting drops any
            // client on it and leaves the instance to listen again.
            unsafe { DisconnectNamedPipe(self.next) };
            if error.raw_os_error() == Some(ERROR_NO_DATA as i32) {
                continue;
            }
            failures += 1;
            if failures == ACCEPT_ATTEMPTS {
                return Err(error);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Waits until a client connects to `next`.
    fn connect(&self) -> io::Result<()> {
        // SAFETY: OVERLAPPED is plain data, valid zeroed.
        let mut overlapped: OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = self.event;
        // SAFETY: the instance is open and unconnected; `overlapped` outlives the operation,
        // which GetOverlappedResult waits out below.
        if unsafe { ConnectNamedPipe(self.next, &mut overlapped) } != 0 {
            return Ok(());
        }
        // SAFETY: reads this thread's last error only.
        match unsafe { GetLastError() } {
            // A client connected between the instance's creation and the call.
            ERROR_PIPE_CONNECTED => Ok(()),
            ERROR_IO_PENDING => {
                let mut n = 0;
                // SAFETY: as above; bWait returns once a client has connected or the call failed.
                if unsafe { GetOverlappedResult(self.next, &overlapped, &mut n, 1) } == 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            }
            error => Err(io::Error::from_raw_os_error(error as i32)),
        }
    }

    fn instance(&self, first: u32) -> io::Result<HANDLE> {
        let attributes = SECURITY_ATTRIBUTES {
            nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: self.security,
            bInheritHandle: 0,
        };
        // SAFETY: the name is NUL-terminated and the descriptor valid for the listener's life.
        let handle = unsafe {
            CreateNamedPipeW(
                self.name.as_ptr(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | first,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                MAX_INSTANCES,
                64 * 1024,
                64 * 1024,
                0,
                &attributes,
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(handle)
    }
}

impl Drop for PipeListener {
    fn drop(&mut self) {
        // SAFETY: `next` and `event` are open or empty, and the descriptor came from
        // ConvertStringSecurityDescriptorToSecurityDescriptorW; all are owned here only.
        unsafe {
            if self.next != INVALID_HANDLE_VALUE {
                CloseHandle(self.next);
            }
            if !self.event.is_null() {
                CloseHandle(self.event);
            }
            LocalFree(self.security);
        }
    }
}

/// This process's user SID as text.
fn current_user() -> io::Result<String> {
    // SAFETY: the pseudo-handle is always valid and needs no closing.
    process_user(unsafe { GetCurrentProcess() })
}

/// The user SID, as text, of the open process handle `process`.
fn process_user(process: HANDLE) -> io::Result<String> {
    let mut token = ptr::null_mut();
    // SAFETY: the caller passes an open process handle; `token` receives a handle closed below.
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut len = 0;
    // SAFETY: a size query: no buffer is written.
    unsafe { GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &mut len) };
    // u64 words keep the TOKEN_USER header aligned.
    let mut buffer = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: `buffer` holds `len` bytes; the token stays open for the call.
    let ok =
        unsafe { GetTokenInformation(token, TokenUser, buffer.as_mut_ptr().cast(), len, &mut len) };
    let error = io::Error::last_os_error();
    // SAFETY: the token handle is open and owned here only.
    unsafe { CloseHandle(token) };
    if ok == 0 {
        return Err(error);
    }
    // SAFETY: GetTokenInformation wrote a TOKEN_USER at the buffer's start, whose SID points
    // into the same buffer, alive across this read.
    sid_string(unsafe { (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid })
}

fn millis(d: Duration) -> io::Result<u32> {
    if d.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a zero timeout is not a deadline",
        ));
    }
    Ok(d.as_millis().clamp(1, u128::from(INFINITE - 1)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(tag: &str) -> String {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!(r"\\.\pipe\larch-test-{tag}-{}-{n}", std::process::id())
    }
    /// A listener on a thread that echoes `clients` one-line requests, then stops.
    fn echo(name: &str, clients: usize) -> std::thread::JoinHandle<()> {
        let mut listener = PipeListener::bind(name).unwrap();
        std::thread::spawn(move || {
            for _ in 0..clients {
                let mut stream = listener.accept(Duration::from_secs(2)).unwrap();
                let mut byte = [0u8; 1];
                stream.read_exact(&mut byte).unwrap();
                stream.write_all(&byte).unwrap();
            }
        })
    }

    #[test]
    fn a_name_already_bound_is_refused() {
        let name = name("first");
        let _first = PipeListener::bind(&name).unwrap();
        assert!(PipeListener::bind(&name).is_err(), "FIRST_PIPE_INSTANCE");
    }

    #[test]
    fn concurrent_clients_all_get_through() {
        let name = name("busy");
        let server = echo(&name, 3);
        let clients: Vec<_> = (0..3u8)
            .map(|i| {
                let name = name.clone();
                std::thread::spawn(move || {
                    let mut stream = PipeStream::connect(&name, Duration::from_secs(5)).unwrap();
                    assert!(stream.server_is_current_user().unwrap());
                    stream.write_all(&[i]).unwrap();
                    let mut byte = [0u8; 1];
                    stream.read_exact(&mut byte).unwrap();
                    assert_eq!(byte, [i]);
                })
            })
            .collect();
        for client in clients {
            client.join().unwrap();
        }
        server.join().unwrap();
    }

    #[test]
    fn a_client_that_leaves_before_accept_keeps_the_name_served() {
        let name = name("gone");
        let mut listener = PipeListener::bind(&name).unwrap();
        // Connects to the listening instance and leaves before the server calls accept.
        drop(PipeStream::connect(&name, Duration::from_secs(5)).unwrap());
        let server = std::thread::spawn(move || {
            let mut stream = listener.accept(Duration::from_secs(2)).unwrap();
            let mut byte = [0u8; 1];
            stream.read_exact(&mut byte).unwrap();
            stream.write_all(&byte).unwrap();
        });
        let mut stream = PipeStream::connect(&name, Duration::from_secs(5)).unwrap();
        stream.write_all(b"x").unwrap();
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).unwrap();
        server.join().unwrap();
    }

    #[test]
    fn a_silent_peer_times_out_and_a_zero_timeout_is_refused() {
        let name = name("silent");
        let mut listener = PipeListener::bind(&name).unwrap();
        let client = PipeStream::connect(&name, Duration::from_secs(5)).unwrap();
        let mut stream = listener.accept(Duration::from_millis(200)).unwrap();
        let mut byte = [0u8; 1];
        assert_eq!(
            stream.read(&mut byte).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        assert!(stream.set_timeout(Duration::ZERO).is_err());
        assert!(PipeStream::connect(&name, Duration::ZERO).is_err());
        drop(client);
    }
}
