use super::*;
use crate::command::{self, Output};
#[cfg(target_os = "linux")]
use std::fs;
use std::{process::Command, time::Duration};
fn output(command: &mut Command) -> Result<Option<String>> {
    match command::run(command, Duration::from_secs(10))? {
        Output::Finished {
            success: true,
            stdout,
            ..
        } => Ok(Some(stdout)),
        _ => Ok(None),
    }
}
/// Available space on the filesystem that will contain the destination.
pub fn free_space(path: &Path) -> Result<u64> {
    let parent = path
        .ancestors()
        .find(|p| p.is_dir())
        .ok_or_else(|| Error::Refused("no existing installation filesystem".into()))?;
    #[cfg(windows)]
    {
        let text = output(
            Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-Item -LiteralPath $env:LARCH_SPACE_PATH).PSDrive.Free",
                ])
                // A verbatim (canonical) path has no PSDrive, so Free would print nothing.
                .env("LARCH_SPACE_PATH", crate::effects::normal_path(parent)?),
        )?
        .ok_or_else(|| Error::Refused("cannot query free disk space".into()))?;
        ps_free(&text)
    }
    #[cfg(not(windows))]
    {
        let text = output(Command::new("df").arg("-Pk").arg(parent))?
            .ok_or_else(|| Error::Refused("cannot query free disk space".into()))?;
        df_pk(&text)
    }
}
/// Observes only the current platform using bounded tools and kernel-provided files.
pub fn detect_machine() -> Result<Machine> {
    #[cfg(target_os = "linux")]
    {
        linux()
    }
    #[cfg(target_os = "macos")]
    {
        let version = command::text("sw_vers", &["-productVersion"])?
            .ok_or_else(|| Error::Refused("cannot detect macOS version".into()))?;
        let memory = command::text("sysctl", &["-n", "hw.memsize"])?
            .ok_or_else(|| Error::Refused("cannot detect macOS memory".into()))?;
        let arm =
            command::text("sysctl", &["-n", "hw.optional.arm64"])?.unwrap_or_else(|| "0".into());
        let (os_version, memory, arm64) = macos(&version, &memory, &arm)?;
        Ok(Machine {
            os: Os::Macos,
            target: if arm64 {
                "aarch64-apple-darwin"
            } else {
                "x86_64-apple-darwin"
            }
            .into(),
            os_version,
            arm64,
            glibc: None,
            glibcxx: None,
            memory,
            gpus: Vec::new(),
            kfd: KfdAccess::NoDriver,
        })
    }
    #[cfg(windows)]
    {
        windows()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Err(Error::Refused(
            "this operating system is unsupported".into(),
        ))
    }
}
#[cfg(target_os = "linux")]
fn linux() -> Result<Machine> {
    let mut gpus = Vec::new();
    if let Some(csv) = command::text(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )? {
        gpus = nvidia_smi(&csv)?;
    }
    if gpus.is_empty()
        && let Ok(version) = fs::read_to_string("/proc/driver/nvidia/version")
        && let Ok(entries) = fs::read_dir("/proc/driver/nvidia/gpus")
    {
        for e in entries {
            let p = e?.path().join("information");
            if let Ok(info) = fs::read_to_string(p) {
                gpus.push(proc_nvidia(&version, &info)?);
            }
        }
    }
    let kfd = kfd_access(
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kfd")
            .err()
            .map(|e| e.kind()),
    );
    if let Ok(nodes) = fs::read_dir("/sys/class/kfd/kfd/topology/nodes") {
        'nodes: for node in nodes {
            let node = node?.path();
            // amdkfd refuses topology reads outside the device cgroup (a container without
            // /dev/kfd): that GPU is not usable from here.
            let props = match fs::read_to_string(node.join("properties")) {
                Ok(props) => props,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
                Err(e) => return Err(e.into()),
            };
            let mut banks = Vec::new();
            if let Ok(entries) = fs::read_dir(node.join("mem_banks")) {
                for bank in entries {
                    match fs::read_to_string(bank?.path().join("properties")) {
                        Ok(bank) => banks.push(bank),
                        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                            continue 'nodes;
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            if let Some(gpu) = kfd_node(
                &props,
                &banks.iter().map(String::as_str).collect::<Vec<_>>(),
            )? {
                gpus.push(gpu);
            }
        }
    }
    let libc = command::text("getconf", &["GNU_LIBC_VERSION"])?
        .map(|s| glibc(&s))
        .transpose()?;
    let mut cxx = None;
    if let Some(cache) = command::text("ldconfig", &["-p"])?
        && let Some(path) = cache
            .lines()
            .find(|line| line.contains("libstdc++.so.6 ") && line.contains("x86-64"))
            .and_then(|line| line.split("=>").nth(1))
            .map(str::trim)
        && let Some(strings) = output(Command::new("strings").arg(path))?
    {
        cxx = Some(glibcxx(&strings)?);
    }
    if cxx.is_none() {
        for path in [
            "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
            "/usr/lib64/libstdc++.so.6",
        ] {
            if let Ok(bytes) = fs::read(path) {
                cxx = Some(glibcxx(&String::from_utf8_lossy(&bytes))?);
                break;
            }
        }
    }
    Ok(Machine {
        os: Os::Linux,
        target: format!("{}-unknown-linux-gnu", std::env::consts::ARCH),
        os_version: String::new(),
        arm64: std::env::consts::ARCH == "aarch64",
        glibc: libc,
        glibcxx: cxx,
        memory: meminfo(&fs::read_to_string("/proc/meminfo")?)?,
        gpus,
        kfd,
    })
}
#[cfg(windows)]
fn windows() -> Result<Machine> {
    const SCRIPT: &str = r#"$ErrorActionPreference='Stop'
$controllers=@(Get-CimInstance Win32_VideoController | ForEach-Object {
  $g=$_; $size=0
  try {
    $driver=Get-ItemPropertyValue -LiteralPath ('HKLM:\SYSTEM\CurrentControlSet\Enum\'+$g.PNPDeviceID) -Name Driver
    $size=Get-ItemPropertyValue -LiteralPath ('HKLM:\SYSTEM\CurrentControlSet\Control\Class\'+$driver) -Name 'HardwareInformation.qwMemorySize'
  } catch { $size=0 }
  [pscustomobject]@{Name=$g.Name; PNPDeviceID=$g.PNPDeviceID; DriverVersion=$g.DriverVersion; qwMemorySize=[UInt64]$size}
})
[pscustomobject]@{build=(Get-CimInstance Win32_OperatingSystem).BuildNumber; memory=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; controllers=$controllers} | ConvertTo-Json -Depth 4 -Compress"#;
    let text = command::text(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", SCRIPT],
    )?
    .ok_or_else(|| Error::Refused("cannot detect Windows hardware".into()))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let controllers = v
        .get("controllers")
        .ok_or_else(|| bad("Windows inventory", &text))?;
    let mut qwords = BTreeMap::new();
    for g in controllers
        .as_array()
        .ok_or_else(|| bad("Windows controllers", &text))?
    {
        if let (Some(id), Some(n)) = (g["PNPDeviceID"].as_str(), g["qwMemorySize"].as_u64()) {
            qwords.insert(id.into(), n);
        }
    }
    let mut gpus = windows_amd(&controllers.to_string(), &qwords)?;
    if let Some(csv) = command::text(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )? {
        gpus.extend(nvidia_smi(&csv)?);
    }
    let os_version = v["build"]
        .as_str()
        .ok_or_else(|| bad("Windows build", &text))?
        .to_owned();
    Ok(Machine {
        os: Os::Windows,
        target: format!("{}-pc-windows-msvc", std::env::consts::ARCH),
        os_version,
        arm64: std::env::consts::ARCH == "aarch64",
        glibc: None,
        glibcxx: None,
        memory: v["memory"]
            .as_u64()
            .ok_or_else(|| bad("Windows memory", &text))?,
        gpus,
        kfd: KfdAccess::NoDriver,
    })
}
