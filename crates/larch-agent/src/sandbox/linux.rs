//! Linux: user, network and pid namespaces of its own (unshare), so only lo exists and nothing
//! outside can be signalled; Landlock, so it writes only inside the worktree (or the folders of it
//! a read-only run may write) and the private tmp;
//! seccomp against host UNIX sockets, ownership and xattr changes; prlimit for the caps.

use std::{
    collections::BTreeMap,
    panic,
    path::PathBuf,
    process::{Child, Command},
    thread,
};

use landlock::{
    ABI, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetCreatedAttr,
    path_beneath_rules,
};
use seccompiler::{
    SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule,
};

use super::{Limits, SandboxError};

/// setxattrat(2) and removexattrat(2), Linux 6.13: one number on every architecture, and not in
/// libc yet.
const SYS_SETXATTRAT: i64 = 463;
const SYS_REMOVEXATTRAT: i64 = 466;

/// `cmd` under unshare and prlimit. No uid map: Landlock is already on when unshare starts, and
/// it denies the write to /proc/self/uid_map, so the child sees itself as nobody; the kernel
/// still checks files against our own uid. The pid namespace dies with its init, setsid processes
/// included; --kill-child kills that init with unshare even once it has left our group. prlimit
/// inside the namespace: there RLIMIT_NPROC counts only the command's processes, not every one
/// of our uid.
pub(super) fn command(cmd: &str, _writable: &[PathBuf], limits: &Limits) -> Command {
    let mut command = Command::new("unshare");
    command
        .args([
            "--user",
            "--net",
            "--pid",
            "--fork",
            "--kill-child",
            "--",
            "prlimit",
        ])
        .arg(format!("--cpu={}", limits.cpu_s))
        .arg(format!("--as={}", limits.mem_bytes))
        .arg(format!("--nproc={}", limits.max_procs))
        .args(["--", "sh", "-c", cmd]);
    command
}

/// What unshare's bare "Operation not permitted" means where AppArmor keeps unprivileged
/// processes out of user namespaces (Ubuntu 23.10 and later), and the way out; empty elsewhere.
// ponytail: the user lifts it by hand. upgrade: an AppArmor profile granting `userns` in the
// Linux package once one ships.
pub(super) fn userns_hint() -> &'static str {
    let restricted =
        std::fs::read_to_string("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
            .is_ok_and(|v| v.trim() == "1");
    match restricted {
        true => {
            "; AppArmor keeps unprivileged programs from user namespaces here (Ubuntu 23.10 and \
             later): allow them with `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`"
        }
        false => "",
    }
}

/// Spawns `command` from a thread that Landlock and seccomp restrict first: the child inherits
/// both through unshare and exec, and we keep neither, with no unsafe pre_exec.
pub(super) fn spawn(command: &mut Command, writable: &[PathBuf]) -> Result<Child, SandboxError> {
    thread::scope(|s| {
        s.spawn(|| {
            restrict(writable)?;
            deny_syscalls()?;
            Ok(command.spawn()?)
        })
        .join()
    })
    .unwrap_or_else(|p| panic::resume_unwind(p))
}

/// Restricts this thread: read and execute everywhere, write only beneath `writable`
/// (`super::writable`) and to /dev/null. Refuses below Landlock ABI 3 (Linux 6.2), where truncate is not covered.
// ponytail: chmod and utimes outside the worktree stay possible: no Landlock ABI mediates them,
// and denying them in seccomp breaks touch, pip and cargo inside it. ceiling: modes and times of
// our own files. upgrade: a read-only mount namespace if a tool command abuses it.
fn restrict(writable: &[PathBuf]) -> Result<(), SandboxError> {
    // V3 handles every write right. V5's IoctlDev would also cut terminal ioctls everywhere,
    // which read-only use needs.
    let abi = ABI::V3;
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_all(abi))
        .and_then(|r| r.create())
        .and_then(|r| r.add_rules(path_beneath_rules(["/"], AccessFs::from_read(abi))))
        .and_then(|r| r.add_rules(path_beneath_rules(writable, AccessFs::from_all(abi))))
        // Every shell redirects there.
        .and_then(|r| r.add_rules(path_beneath_rules(["/dev/null"], AccessFs::WriteFile)))
        .and_then(|r| r.restrict_self())
        .map_err(|e| {
            SandboxError::Unavailable(format!(
                "Landlock ABI 3 (Linux 6.2 or later) is needed: {e}"
            ))
        })?;
    Ok(())
}

/// Installs on this thread a seccomp filter that fails with EPERM: UNIX sockets (stream and
/// seqpacket socketpairs stay, pipes are untouched; a datagram pair could still sendto a host
/// socket), io_uring (its ops skip seccomp), ownership changes and xattr writes.
fn deny_syscalls() -> Result<(), SandboxError> {
    let failed = |e: seccompiler::Error| SandboxError::Unavailable(format!("seccomp: {e}"));
    let backend = |e| failed(seccompiler::Error::Backend(e));
    let arch = std::env::consts::ARCH.try_into().map_err(backend)?;
    let rule = |arg, op, value| {
        SeccompCondition::new(arg, SeccompCmpArgLen::Dword, op, value)
            .and_then(|c| SeccompRule::new(vec![c]))
            .map_err(backend)
    };
    let unix = rule(0, SeccompCmpOp::Eq, libc::AF_UNIX as u64)?;
    // The type's low bits; SOCK_CLOEXEC and SOCK_NONBLOCK sit above them.
    let dgram = rule(1, SeccompCmpOp::MaskedEq(0xf), libc::SOCK_DGRAM as u64)?;
    // ponytail: every UNIX socket is denied, a tool's own in tmp too, as Landlock below ABI 9
    // cannot tell a host socket from it. upgrade: Landlock ResolveUnix once ABI 9 kernels are
    // the floor.
    let mut rules = BTreeMap::from([
        (libc::SYS_socket, vec![unix]),
        (libc::SYS_socketpair, vec![dgram]),
    ]);
    // An empty chain matches every call.
    #[cfg(target_arch = "x86_64")]
    rules.extend([(libc::SYS_chown, vec![]), (libc::SYS_lchown, vec![])]);
    rules.extend(
        [
            libc::SYS_io_uring_setup,
            libc::SYS_fchown,
            libc::SYS_fchownat,
            libc::SYS_setxattr,
            libc::SYS_lsetxattr,
            libc::SYS_fsetxattr,
            SYS_SETXATTRAT,
            libc::SYS_removexattr,
            libc::SYS_lremovexattr,
            libc::SYS_fremovexattr,
            SYS_REMOVEXATTRAT,
        ]
        .map(|nr| (nr, vec![])),
    );
    // x32 calls pass the same arch check with the x32 bit set; where the kernel has x32 they
    // would skip every rule.
    #[cfg(target_arch = "x86_64")]
    rules.extend(
        rules
            .clone()
            .into_iter()
            .map(|(nr, r)| (nr | 0x4000_0000, r)),
    );
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        arch,
    )
    .map_err(backend)?;
    let program: seccompiler::BpfProgram = filter.try_into().map_err(backend)?;
    // Sets no_new_privs too; Landlock already has.
    seccompiler::apply_filter(&program).map_err(failed)
}
