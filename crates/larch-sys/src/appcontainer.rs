//! AppContainer processes for the agent's sandbox: a profile with no capabilities, so no network,
//! and a process started inside it and, from its first instruction, inside a Job Object.

use std::{
    ffi::{OsStr, OsString},
    fs::File,
    io::{self, PipeReader},
    iter,
    os::windows::{
        ffi::OsStrExt,
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
    },
    path::Path,
    ptr,
    time::Duration,
};

use windows_sys::Win32::{
    Foundation::{
        HANDLE, HANDLE_FLAG_INHERIT, LocalFree, SetHandleInformation, WAIT_OBJECT_0, WAIT_TIMEOUT,
    },
    Security::{
        Authorization::{
            ConvertSidToStringSidW, EXPLICIT_ACCESS_W, GRANT_ACCESS, SetEntriesInAclW,
            TRUSTEE_IS_SID, TRUSTEE_W,
        },
        DACL_SECURITY_INFORMATION, FreeSid, GetFileSecurityW, GetSecurityDescriptorControl,
        GetSecurityDescriptorDacl, InitializeSecurityDescriptor,
        Isolation::{CreateAppContainerProfile, DeriveAppContainerSidFromAppContainerName},
        NO_INHERITANCE, PSID, SE_DACL_AUTO_INHERITED, SE_DACL_PROTECTED, SECURITY_CAPABILITIES,
        SECURITY_DESCRIPTOR, SetFileSecurityW, SetSecurityDescriptorControl,
        SetSecurityDescriptorDacl,
    },
    Storage::FileSystem::{FILE_READ_ATTRIBUTES, FILE_TRAVERSE, SYNCHRONIZE},
    System::{
        JobObjects::{
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
            JOB_OBJECT_LIMIT_PROCESS_TIME, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectExtendedLimitInformation, SetInformationJobObject,
        },
        Threading::{
            CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, CreateProcessW,
            DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT, GetExitCodeProcess,
            INFINITE, InitializeProcThreadAttributeList, LPPROC_THREAD_ATTRIBUTE_LIST,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROC_THREAD_ATTRIBUTE_JOB_LIST,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, PROCESS_INFORMATION, STARTF_USESTDHANDLES,
            STARTUPINFOEXW, UpdateProcThreadAttribute, WaitForSingleObject,
        },
    },
};

/// The Job Object `job` makes and `spawn` starts a process in.
pub use win32job::Job;

/// The only security descriptor revision there is (windows-sys has it in SystemServices).
const SECURITY_DESCRIPTOR_REVISION: u32 = 1;

/// HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS).
const ALREADY_EXISTS: i32 = 0x8007_00B7_u32 as i32;

/// An AppContainer profile's SID.
pub struct Profile {
    sid: PSID,
    string: String,
}

// SAFETY: the SID is a buffer only this value points at, never written after `profile` returns.
unsafe impl Send for Profile {}
// SAFETY: as above; every use only reads it.
unsafe impl Sync for Profile {}

impl Profile {
    /// The SID as text, `S-1-15-2-…`, as icacls names it.
    pub fn sid(&self) -> &str {
        &self.string
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        // SAFETY: the SID came from CreateAppContainerProfile or DeriveAppContainerSid…, whose
        // SIDs FreeSid frees, and nothing points at it past this value.
        unsafe { FreeSid(self.sid) };
    }
}

/// The AppContainer profile `name`, created with no capabilities when it does not exist yet. It
/// persists, as profiles do, so the same name keeps the same SID.
pub fn profile(name: &str) -> io::Result<Profile> {
    let wide = wide(OsStr::new(name));
    let mut sid = ptr::null_mut();
    // SAFETY: the strings are NUL-terminated and outlive the call; no capabilities are passed;
    // `sid` receives a SID the returned Profile frees.
    let mut hr = unsafe {
        CreateAppContainerProfile(
            wide.as_ptr(),
            wide.as_ptr(),
            wide.as_ptr(),
            ptr::null(),
            0,
            &mut sid,
        )
    };
    if hr == ALREADY_EXISTS {
        // SAFETY: as above.
        hr = unsafe { DeriveAppContainerSidFromAppContainerName(wide.as_ptr(), &mut sid) };
    }
    if hr < 0 {
        return Err(hresult(hr));
    }
    let string = sid_string(sid).inspect_err(|_| {
        // SAFETY: the SID came from the call above and nothing else points at it.
        unsafe { FreeSid(sid) };
    })?;
    Ok(Profile { sid, string })
}

/// Lets `profile` traverse `dir` and read its attributes, on that folder alone. SetFileSecurityW
/// writes just this folder's DACL: icacls and SetNamedSecurityInfo would walk everything beneath
/// it to re-propagate inheritance, minutes for a user's profile folder. The DACL's protected and
/// auto-inherited bits are kept.
pub fn let_through(profile: &Profile, dir: &Path) -> io::Result<()> {
    let path = wide(dir.as_os_str());
    let mut needed = 0;
    // SAFETY: a null buffer only asks for the size; this call fails by design.
    unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            0,
            &mut needed,
        )
    };
    // u64s, for the descriptor's alignment.
    let mut old = vec![0u64; (needed as usize).div_ceil(8)];
    let old_sd = old.as_mut_ptr().cast();
    // SAFETY: the buffer holds `needed` bytes.
    if unsafe {
        GetFileSecurityW(
            path.as_ptr(),
            DACL_SECURITY_INFORMATION,
            old_sd,
            needed,
            &mut needed,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let (mut present, mut defaulted, mut dacl) = (0, 0, ptr::null_mut());
    let (mut control, mut revision) = (0, 0);
    // SAFETY: `old_sd` is the valid descriptor just read; the outputs point into it or at locals.
    let read = unsafe {
        GetSecurityDescriptorDacl(old_sd, &mut present, &mut dacl, &mut defaulted) != 0
            && GetSecurityDescriptorControl(old_sd, &mut control, &mut revision) != 0
    };
    if !read {
        return Err(io::Error::last_os_error());
    }
    let entry = EXPLICIT_ACCESS_W {
        // SYNCHRONIZE too: CreateFileW asks for it on every synchronous open.
        grfAccessPermissions: FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        grfAccessMode: GRANT_ACCESS,
        grfInheritance: NO_INHERITANCE,
        Trustee: TRUSTEE_W {
            TrusteeForm: TRUSTEE_IS_SID,
            ptstrName: profile.sid.cast(),
            ..Default::default()
        },
    };
    let mut new_dacl = ptr::null_mut();
    // SAFETY: one entry whose SID lives in `profile`; the old DACL is in `old`; the new one is
    // LocalAlloc'd and freed below.
    let set = unsafe { SetEntriesInAclW(1, &entry, dacl, &mut new_dacl) };
    if set != 0 {
        return Err(io::Error::from_raw_os_error(set as i32));
    }
    let mut sd = SECURITY_DESCRIPTOR::default();
    let new_sd = (&raw mut sd).cast();
    let kept = SE_DACL_PROTECTED | SE_DACL_AUTO_INHERITED;
    // SAFETY: `sd` is a descriptor-sized local that outlives the calls; `new_dacl` is the ACL
    // SetEntriesInAclW made, freed once after them.
    unsafe {
        let written = InitializeSecurityDescriptor(new_sd, SECURITY_DESCRIPTOR_REVISION) != 0
            && SetSecurityDescriptorDacl(new_sd, 1, new_dacl, 0) != 0
            && SetSecurityDescriptorControl(new_sd, kept, control & kept) != 0
            && SetFileSecurityW(path.as_ptr(), DACL_SECURITY_INFORMATION, new_sd) != 0;
        let e = io::Error::last_os_error();
        LocalFree(new_dacl.cast());
        written.then_some(()).ok_or(e)
    }
}

/// A Job Object that kills every process in it once dropped, and caps each one: `mem_bytes` of
/// committed memory and `cpu_s` seconds of user CPU time; at most `max_procs` run at once. An
/// unhandled exception ends a process instead of waiting on the error-reporting dialog.
pub fn job(mem_bytes: u64, max_procs: u32, cpu_s: u64) -> io::Result<Job> {
    let job = Job::create().map_err(io::Error::other)?;
    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    let basic = &mut info.BasicLimitInformation;
    basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | JOB_OBJECT_LIMIT_PROCESS_TIME
        | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    // In 100 ns ticks.
    basic.PerProcessUserTimeLimit =
        i64::try_from(cpu_s.saturating_mul(10_000_000)).unwrap_or(i64::MAX);
    basic.ActiveProcessLimit = max_procs;
    info.ProcessMemoryLimit = usize::try_from(mem_bytes).unwrap_or(usize::MAX);
    // SAFETY: the job handle is live for the call; `info` is the struct this class reads, with
    // its size.
    let set = unsafe {
        SetInformationJobObject(
            job.handle() as HANDLE,
            JobObjectExtendedLimitInformation,
            (&raw const info).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    };
    if set == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(job)
}

/// A process `spawn` started, and its stdout and stderr.
pub struct Child {
    /// The process.
    pub process: Process,
    /// Its stdout.
    pub stdout: PipeReader,
    /// Its stderr.
    pub stderr: PipeReader,
}

/// A started process's handle.
pub struct Process(OwnedHandle);

impl Process {
    /// Its exit code once it has ended, or None once `timeout` passes first.
    pub fn wait(&self, timeout: Duration) -> io::Result<Option<u32>> {
        let ms = u32::try_from(timeout.as_millis()).map_or(INFINITE - 1, |ms| ms.min(INFINITE - 1));
        let process = self.0.as_raw_handle();
        // SAFETY: the process handle is ours and live.
        match unsafe { WaitForSingleObject(process, ms) } {
            WAIT_OBJECT_0 => {}
            WAIT_TIMEOUT => return Ok(None),
            _ => return Err(io::Error::last_os_error()),
        }
        let mut code = 0;
        // SAFETY: as above; `code` is a valid out pointer.
        if unsafe { GetExitCodeProcess(process, &mut code) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Some(code))
    }
}

/// Starts `program` with `command_line` (its whole command line, argv[0] included) in `cwd`, with
/// only `env`, inside `profile`'s AppContainer with no capabilities and inside `job`. stdin is
/// NUL; the child inherits that and its two pipes, nothing else.
pub fn spawn(
    profile: &Profile,
    job: &Job,
    program: &Path,
    command_line: &OsStr,
    env: &[(OsString, OsString)],
    cwd: &Path,
) -> io::Result<Child> {
    let (stdout, out) = io::pipe()?;
    let (stderr, err) = io::pipe()?;
    let null = File::open("NUL")?;
    // ponytail: the three are inheritable until they drop below, so a std Command spawned on
    // another thread meanwhile inherits them too and holds the pipes open past our child's end.
    // ceiling: the sandbox's grace after the kill, then its output is flagged as cut. upgrade:
    // spawn std's children with a handle list too, if a run in the app shows it.
    let inherited = [
        null.as_raw_handle(),
        out.as_raw_handle(),
        err.as_raw_handle(),
    ];
    for handle in inherited {
        // SAFETY: each handle is live and ours; only its inherit flag changes.
        if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let mut capabilities = SECURITY_CAPABILITIES {
        AppContainerSid: profile.sid,
        ..Default::default()
    };
    let jobs = [job.handle() as HANDLE];
    let mut attributes = Attributes::new(3)?;
    attributes.set(
        PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
        (&raw mut capabilities).cast(),
        size_of::<SECURITY_CAPABILITIES>(),
    )?;
    attributes.set(
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        inherited.as_ptr().cast(),
        size_of_val(&inherited),
    )?;
    attributes.set(
        PROC_THREAD_ATTRIBUTE_JOB_LIST,
        jobs.as_ptr().cast(),
        size_of_val(&jobs),
    )?;

    let mut info = STARTUPINFOEXW::default();
    info.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    [
        info.StartupInfo.hStdInput,
        info.StartupInfo.hStdOutput,
        info.StartupInfo.hStdError,
    ] = inherited;
    info.lpAttributeList = attributes.list();
    let program = wide(program.as_os_str());
    let mut command_line = wide(command_line);
    let cwd = wide(cwd.as_os_str());
    let mut block: Vec<u16> = Vec::new();
    for (name, value) in env {
        block.extend(name.encode_wide().chain(iter::once(u16::from(b'='))));
        block.extend(value.encode_wide().chain(iter::once(0)));
    }
    block.extend([0, 0]);
    let mut started = PROCESS_INFORMATION::default();
    // SAFETY: every string is NUL-terminated (the environment block doubly) and outlives the
    // call; the command line is a buffer of our own, which CreateProcessW may write; the
    // attribute list and every value it points at (capabilities, handles, job) live until
    // `attributes` drops, after this call.
    let created = unsafe {
        CreateProcessW(
            program.as_ptr(),
            command_line.as_mut_ptr(),
            ptr::null(),
            ptr::null(),
            1,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
            block.as_ptr().cast(),
            cwd.as_ptr(),
            &raw const info.StartupInfo,
            &mut started,
        )
    };
    if created == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: CreateProcessW returned both handles to us, open; each is owned once from here.
    let (process, _thread) = unsafe {
        (
            OwnedHandle::from_raw_handle(started.hProcess),
            OwnedHandle::from_raw_handle(started.hThread),
        )
    };
    // Our copies of the child's ends close as `out`, `err` and `null` drop, so the pipes reach
    // EOF once every process in the job has closed them.
    Ok(Child {
        process: Process(process),
        stdout,
        stderr,
    })
}

/// A process-thread attribute list, deleted on drop.
struct Attributes(Vec<u64>);

impl Attributes {
    fn new(count: u32) -> io::Result<Self> {
        let mut size = 0;
        // SAFETY: a null list only asks for the size; this call fails by design.
        unsafe { InitializeProcThreadAttributeList(ptr::null_mut(), count, 0, &mut size) };
        // u64s, for the alignment the list needs.
        let mut buf = vec![0u64; size.div_ceil(8)];
        // SAFETY: the buffer holds `size` bytes.
        if unsafe {
            InitializeProcThreadAttributeList(buf.as_mut_ptr().cast(), count, 0, &mut size)
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(buf))
    }

    fn list(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.0.as_mut_ptr().cast()
    }

    /// Sets `attribute` to the `size` bytes at `value`, which must outlive the list's use.
    fn set(
        &mut self,
        attribute: u32,
        value: *const std::ffi::c_void,
        size: usize,
    ) -> io::Result<()> {
        // SAFETY: the list is initialised; `value` points at `size` bytes the caller keeps alive
        // until CreateProcessW has returned.
        let set = unsafe {
            UpdateProcThreadAttribute(
                self.list(),
                0,
                attribute as usize,
                value,
                size,
                ptr::null_mut(),
                ptr::null(),
            )
        };
        if set == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for Attributes {
    fn drop(&mut self) {
        // SAFETY: the list was initialised in `new` and is deleted once.
        unsafe { DeleteProcThreadAttributeList(self.list()) };
    }
}

/// A valid SID as text, `S-1-…`.
pub(crate) fn sid_string(sid: PSID) -> io::Result<String> {
    let mut text = ptr::null_mut();
    // SAFETY: the caller passes a valid SID; `text` receives a LocalAlloc'd string.
    if unsafe { ConvertSidToStringSidW(sid, &mut text) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `text` is the NUL-terminated string ConvertSidToStringSidW allocated; it is read up
    // to its NUL, then freed once.
    Ok(unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        let string = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
        LocalFree(text.cast());
        string
    })
}

/// `s` as UTF-16 with a NUL.
pub(crate) fn wide(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(iter::once(0)).collect()
}

/// An HRESULT as the Win32 error it wraps, so its text is the system's; others as they are.
fn hresult(hr: i32) -> io::Error {
    let hr = hr as u32;
    io::Error::from_raw_os_error(if hr >> 16 == 0x8007 {
        (hr & 0xFFFF) as i32
    } else {
        hr as i32
    })
}
