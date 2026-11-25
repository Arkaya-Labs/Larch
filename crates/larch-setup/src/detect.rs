//! Hardware and ABI observations parsed independently from platform-tool execution.
mod platform;
pub use platform::{detect_machine, free_space};

use crate::manifest::{Error, Manifest, Result};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap, io, path::Path};
/// The supported operating-system families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Os {
    /// GNU/Linux.
    Linux,
    /// Windows x86-64.
    Windows,
    /// Apple silicon macOS.
    Macos,
}
impl Os {
    /// Returns this host's supported operating-system family without running tools.
    pub fn current() -> Result<Self> {
        match std::env::consts::OS {
            "linux" => Ok(Self::Linux),
            "windows" => Ok(Self::Windows),
            "macos" => Ok(Self::Macos),
            _ => Err(Error::Refused(
                "this operating system is unsupported".into(),
            )),
        }
    }
}
/// A GPU observation; an absent compute capability requires a runtime self-test.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    /// Vendor used by release rules.
    pub vendor: String,
    /// Human-readable device name.
    pub name: String,
    /// Dedicated memory in bytes, zero when unavailable.
    pub memory: u64,
    /// Installed driver version, when known.
    pub driver: Option<String>,
    /// NVIDIA compute capability.
    pub cc: Option<String>,
    /// AMD compiler graphics target.
    pub gfx: Option<String>,
    /// Windows plug-and-play ID, when available.
    pub pnp: Option<String>,
    /// Whether the graphics target was observed reliably.
    pub verified: bool,
}
/// Access to the Linux AMD compute driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum KfdAccess {
    /// The compute device can be opened.
    Ready,
    /// No compute driver is installed.
    #[default]
    NoDriver,
    /// Group membership must be corrected and the user must log in again.
    NeedRenderGroup,
}
/// All observations needed by the pure release-selection policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Machine {
    /// Current operating system.
    pub os: Os,
    /// Rust target used in artifact keys.
    pub target: String,
    /// Product version or Windows build number.
    pub os_version: String,
    /// Whether macOS is running on Apple silicon.
    pub arm64: bool,
    /// GNU libc version.
    pub glibc: Option<String>,
    /// Highest supported libstdc++ symbol version.
    pub glibcxx: Option<String>,
    /// Physical machine memory in bytes.
    pub memory: u64,
    /// All discovered GPUs.
    pub gpus: Vec<Gpu>,
    /// Linux AMD compute-driver access.
    pub kfd: KfdAccess,
}
/// The selected flavour, GPU and explanation shown before installation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    /// The supported and gated backend.
    pub flavour: String,
    /// The chosen GPU, when applicable.
    pub gpu: Option<Gpu>,
    /// Why this backend was selected, including CPU performance warnings.
    pub why: String,
}
fn bad(kind: &str, line: &str) -> Error {
    Error::Refused(format!("invalid {kind} line: {line:?}"))
}
fn dotted(text: &str) -> Result<Vec<u32>> {
    if text.is_empty() {
        return Err(bad("dotted version", text));
    }
    text.split('.')
        .map(|v| {
            if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
                v.parse().map_err(|_| bad("dotted version", text))
            } else {
                Err(bad("dotted version", text))
            }
        })
        .collect()
}
fn compare(a: &str, b: &str) -> Result<Ordering> {
    let mut a = dotted(a)?;
    let mut b = dotted(b)?;
    let n = a.len().max(b.len());
    a.resize(n, 0);
    b.resize(n, 0);
    Ok(a.cmp(&b))
}
/// Compares dotted numeric versions, padding omitted trailing fields with zero.
pub fn version_at_least(have: &str, minimum: &str) -> Result<bool> {
    Ok(compare(have, minimum)? != Ordering::Less)
}
fn driver(text: &str) -> Result<String> {
    let fields = dotted(text)?;
    if !(2..=3).contains(&fields.len()) || !(100..=999).contains(&fields[0]) {
        return Err(bad("NVIDIA driver", text));
    }
    Ok(text.into())
}
/// Parses nvidia-smi CSV with memory reported in MiB and compute capability included.
pub fn nvidia_smi(csv: &str) -> Result<Vec<Gpu>> {
    csv.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            if fields.len() != 5 || fields[0].parse::<u32>().is_err() || fields[1].is_empty() {
                return Err(bad("nvidia-smi", line));
            }
            let memory = fields[2]
                .parse::<u64>()
                .ok()
                .and_then(|n| n.checked_mul(1024 * 1024))
                .ok_or_else(|| bad("nvidia-smi", line))?;
            let cc = dotted(fields[4]).map_err(|_| bad("nvidia-smi", line))?;
            if cc.len() != 2 || cc[0] == 0 || cc[1] > 9 {
                return Err(bad("nvidia-smi", line));
            }
            let driver = driver(fields[3]).map_err(|_| bad("nvidia-smi", line))?;
            Ok(Gpu {
                vendor: "nvidia".into(),
                name: fields[1].into(),
                memory,
                driver: Some(driver),
                cc: Some(fields[4].into()),
                gfx: None,
                pnp: None,
                verified: true,
            })
        })
        .collect()
}
/// Parses the fallback driver and per-GPU proc files without guessing compute capability.
pub fn proc_nvidia(version: &str, information: &str) -> Result<Gpu> {
    let driver = version
        .split_whitespace()
        .find_map(|s| driver(s).ok())
        .ok_or_else(|| bad("NVIDIA version", version))?;
    let name = information
        .lines()
        .find_map(|line| line.strip_prefix("Model:").map(str::trim))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad("NVIDIA information", information))?;
    Ok(Gpu {
        vendor: "nvidia".into(),
        name: name.into(),
        memory: 0,
        driver: Some(driver),
        cc: None,
        gfx: None,
        pnp: None,
        verified: false,
    })
}
fn properties(text: &str) -> Result<BTreeMap<&str, u64>> {
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2 {
            return Err(bad("KFD properties", line));
        }
        let number = fields[1].parse().map_err(|_| bad("KFD properties", line))?;
        if out.insert(fields[0], number).is_some() {
            return Err(bad("duplicate KFD property", line));
        }
    }
    Ok(out)
}
/// Parses KFD's decimal gfx version and only device-local framebuffer memory heaps.
pub fn kfd_node(node: &str, banks: &[&str]) -> Result<Option<Gpu>> {
    let props = properties(node)?;
    let n = *props
        .get("gfx_target_version")
        .ok_or_else(|| bad("KFD gfx_target_version", node))?;
    if n == 0 {
        return Ok(None);
    }
    // ROCm documents the decimal encoding in HSA_GET_GFX_VERSION_{MAJOR,MINOR,STEP}:
    // https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/libhsakmt/src/libhsakmt.h
    let (major, minor, step) = (n / 10000, n / 100 % 100, n % 100);
    if !(1..100).contains(&major) || minor > 15 || step > 15 {
        return Err(bad("KFD gfx_target_version", node));
    }
    let gfx = format!("gfx{major}{minor:x}{step:x}");
    let mut memory = 0u64;
    for bank in banks {
        let b = properties(bank)?;
        let heap = b
            .get("heap_type")
            .ok_or_else(|| bad("KFD heap_type", bank))?;
        let size = b
            .get("size_in_bytes")
            .ok_or_else(|| bad("KFD size_in_bytes", bank))?;
        // Only FB_PUBLIC(1) and FB_PRIVATE(2) are VRAM, excluding system/GDS/LDS/scratch:
        // https://github.com/torvalds/linux/blob/v6.17/include/uapi/linux/kfd_sysfs.h
        if matches!(heap, 1 | 2) {
            memory = memory
                .checked_add(*size)
                .ok_or_else(|| bad("KFD memory overflow", bank))?;
        }
    }
    Ok(Some(Gpu {
        vendor: "amd".into(),
        name: gfx.clone(),
        memory,
        driver: None,
        cc: None,
        gfx: Some(gfx),
        pnp: None,
        verified: true,
    }))
}
/// Maps compute-device access failures to an actionable AMD-driver state.
pub fn kfd_access(error: Option<io::ErrorKind>) -> KfdAccess {
    match error {
        None => KfdAccess::Ready,
        Some(io::ErrorKind::PermissionDenied) => KfdAccess::NeedRenderGroup,
        _ => KfdAccess::NoDriver,
    }
}
/// Parses AMD controller JSON; memory comes only from the registry QWORD inventory.
pub fn windows_amd(json: &str, qwords: &BTreeMap<String, u64>) -> Result<Vec<Gpu>> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|_| bad("Windows AMD JSON", json))?;
    let list = match &value {
        serde_json::Value::Array(v) => v.clone(),
        serde_json::Value::Object(_) => vec![value],
        serde_json::Value::Null => return Ok(Vec::new()),
        _ => return Err(bad("Windows AMD JSON", json)),
    };
    let mut out = Vec::new();
    for v in list {
        let field = |key: &str| {
            v.get(key)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| bad("Windows controller", &v.to_string()))
        };
        let name = field("Name")?;
        let pnp = field("PNPDeviceID")?;
        let version = field("DriverVersion")?;
        dotted(version).map_err(|_| bad("Windows controller driver", version))?;
        if !pnp.to_ascii_uppercase().contains("VEN_1002") {
            continue;
        }
        let memory = qwords.get(pnp).copied().unwrap_or(0);
        out.push(Gpu {
            vendor: "amd".into(),
            name: name.into(),
            memory,
            driver: Some(version.into()),
            cc: None,
            gfx: None,
            pnp: Some(pnp.into()),
            verified: false,
        });
    }
    Ok(out)
}
/// Parses macOS version, memory and the explicit arm64 hardware flag.
pub fn macos(version: &str, memory: &str, arm64: &str) -> Result<(String, u64, bool)> {
    let v = version.trim();
    dotted(v)?;
    let memory = memory
        .trim()
        .parse()
        .map_err(|_| bad("macOS memory", memory))?;
    let arm64 = match arm64.trim() {
        "1" => true,
        "0" => false,
        _ => return Err(bad("macOS arm64", arm64)),
    };
    Ok((v.into(), memory, arm64))
}
/// Parses GNU_LIBC_VERSION without inferring a version from a filename.
pub fn glibc(text: &str) -> Result<String> {
    let parts: Vec<_> = text.split_whitespace().collect();
    if parts.len() != 2 || parts[0] != "glibc" {
        return Err(bad("GNU libc", text));
    }
    dotted(parts[1])?;
    Ok(parts[1].into())
}
/// Finds the highest numeric GLIBCXX symbol version in strings or raw library text.
pub fn glibcxx(text: &str) -> Result<String> {
    let mut versions = Vec::new();
    for suffix in text.split("GLIBCXX_").skip(1) {
        let token = suffix
            .split(|c: char| !c.is_ascii_digit() && c != '.')
            .next()
            .unwrap_or_default();
        if !token.is_empty() {
            versions.push((dotted(token)?, token));
        }
    }
    versions
        .into_iter()
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, s)| s.into())
        .ok_or_else(|| bad("GLIBCXX symbols", text))
}
/// Parses Linux MemTotal in KiB, checking multiplication overflow.
pub fn meminfo(text: &str) -> Result<u64> {
    let line = text
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .ok_or_else(|| bad("MemTotal", text))?;
    let f: Vec<_> = line.split_whitespace().collect();
    if f.len() != 3 || f[2] != "kB" {
        return Err(bad("MemTotal", line));
    }
    f[1].parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| bad("MemTotal", line))
}
/// Parses POSIX df output's available 1024-byte blocks.
pub fn df_pk(text: &str) -> Result<u64> {
    let line = text
        .lines()
        .rfind(|s| !s.trim().is_empty())
        .ok_or_else(|| bad("df -Pk", text))?;
    let f: Vec<_> = line.split_whitespace().collect();
    let percent = f
        .iter()
        .position(|s| {
            s.strip_suffix('%')
                .is_some_and(|n| n.parse::<u32>().is_ok())
        })
        .ok_or_else(|| bad("df -Pk", line))?;
    if percent < 4
        || percent + 1 >= f.len()
        || f[percent - 3].parse::<u64>().is_err()
        || f[percent - 2].parse::<u64>().is_err()
    {
        return Err(bad("df -Pk", line));
    }
    f[percent - 1]
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| bad("df -Pk", line))
}
/// Parses PowerShell PSDrive.Free, which already reports bytes.
pub fn ps_free(text: &str) -> Result<u64> {
    text.trim().parse().map_err(|_| bad("PSDrive.Free", text))
}
fn floor(have: Option<&str>, minimum: &str, name: &str) -> Result<()> {
    if minimum.is_empty() {
        return Ok(());
    }
    if !have
        .map(|h| version_at_least(h, minimum))
        .transpose()?
        .unwrap_or(false)
    {
        return Err(Error::Refused(format!(
            "{name} floor {minimum} required; this machine has {}",
            have.unwrap_or("unknown")
        )));
    }
    Ok(())
}
fn available(machine: &Machine, manifest: &Manifest, flavour: &str) -> bool {
    manifest
        .flavours
        .get(&machine.target)
        .is_some_and(|f| f.iter().any(|v| v == flavour))
        && manifest.gated.iter().any(|f| f == flavour)
}
fn amd_target(gpu: &Gpu, manifest: &Manifest) -> Gpu {
    let mut gpu = gpu.clone();
    if gpu.gfx.is_none() {
        gpu.gfx = gpu
            .pnp
            .as_ref()
            .and_then(|p| {
                manifest.amd_ids.get(p).or_else(|| {
                    let upper = p.to_ascii_uppercase();
                    upper
                        .find("DEV_")
                        .and_then(|i| upper.get(i + 4..i + 8))
                        .and_then(|id| manifest.amd_ids.get(id))
                })
            })
            .cloned();
    }
    gpu
}

/// Selects the most suitable gated backend after enforcing the release's OS floors.
pub fn choose(
    machine: &Machine,
    manifest: &Manifest,
    flavour_override: Option<&str>,
) -> Result<Choice> {
    match machine.os {
        Os::Linux => {
            floor(
                machine.glibc.as_deref(),
                &manifest.floors.glibc_min,
                "glibc",
            )?;
            floor(
                machine.glibcxx.as_deref(),
                &manifest.floors.glibcxx_min,
                "GLIBCXX",
            )?;
        }
        Os::Macos => {
            if !machine.arm64 {
                return Err(Error::Refused(
                    "Intel Mac is not supported; Apple silicon is required".into(),
                ));
            }
            floor(
                Some(&machine.os_version),
                &manifest.floors.macos_min,
                "macOS",
            )?;
        }
        Os::Windows => floor(
            Some(&machine.os_version),
            &manifest.floors.windows_min_build.to_string(),
            "Windows build",
        )?,
    }
    let mut gpus: Vec<_> = machine.gpus.iter().collect();
    gpus.sort_by_key(|g| std::cmp::Reverse(g.memory));
    let mut reason = String::from("no supported GPU found");
    let mut selected = None;
    for gpu in gpus.iter().filter(|g| g.vendor == "nvidia") {
        let rule = if let Some(cc) = &gpu.cc {
            let mut found = None;
            for r in &manifest.gpu_rules {
                if r.vendor == "nvidia"
                    && version_at_least(cc, &r.cc_min)?
                    && r.cc_max_excl
                        .as_deref()
                        .map(|max| compare(cc, max).map(|c| c == Ordering::Less))
                        .transpose()?
                        .unwrap_or(true)
                    && gpu
                        .driver
                        .as_deref()
                        .map(|d| version_at_least(d, &r.driver_min))
                        .transpose()?
                        .unwrap_or(false)
                {
                    found = Some(r);
                    break;
                }
            }
            found.map(|r| (r.flavour.as_str(), r.note.clone()))
        } else {
            Some((
                "cuda",
                "unverified GPU; the runtime self-test decides".into(),
            ))
        };
        if let Some((flavour, note)) = rule
            && available(machine, manifest, flavour)
        {
            selected = Some(Choice {
                flavour: flavour.into(),
                gpu: Some((*gpu).clone()),
                why: format!("{}: {flavour} {note}", gpu.name),
            });
            break;
        }
        reason = format!(
            "{}: compute capability {} or driver {} is not supported by a gated release",
            gpu.name,
            gpu.cc.as_deref().unwrap_or("unknown"),
            gpu.driver.as_deref().unwrap_or("unknown")
        );
    }
    if selected.is_none() {
        for gpu in gpus.iter().filter(|g| g.vendor == "amd") {
            let gpu = amd_target(gpu, manifest);
            let access = machine.os != Os::Linux || machine.kfd == KfdAccess::Ready;
            if access
                && gpu
                    .gfx
                    .as_ref()
                    .is_some_and(|g| manifest.hip_targets.contains(g))
                && available(machine, manifest, "hip")
            {
                selected = Some(Choice {
                    flavour: "hip".into(),
                    why: format!("{}: supported AMD target", gpu.name),
                    gpu: Some(gpu),
                });
                break;
            }
            reason = if machine.kfd == KfdAccess::NeedRenderGroup && machine.os == Os::Linux {
                "add yourself to the render group, then log in again".into()
            } else {
                format!(
                    "{}: AMD target is unavailable, unsupported or hip has not passed its release gate",
                    gpu.name
                )
            };
        }
    }
    if machine.os == Os::Macos {
        selected = Some(Choice {
            flavour: "metal".into(),
            gpu: None,
            why: "Apple silicon with supported macOS".into(),
        });
    }
    let mut choice = selected.unwrap_or(Choice {
        flavour: "cpu".into(),
        gpu: None,
        why: format!("{reason}; CPU mode is slow"),
    });
    if let Some(f) = flavour_override {
        if !available(machine, manifest, f) {
            return Err(Error::Refused(format!(
                "flavour {f} is not built and gated for {}",
                machine.target
            )));
        }
        if f != choice.flavour {
            choice = Choice {
                flavour: f.into(),
                gpu: machine
                    .gpus
                    .iter()
                    .filter(|g| {
                        (f == "hip" && g.vendor == "amd") || (f == "cuda" && g.vendor == "nvidia")
                    })
                    .max_by_key(|g| g.memory)
                    .map(|g| {
                        if f == "hip" {
                            amd_target(g, manifest)
                        } else {
                            g.clone()
                        }
                    }),
                why: if f == "cpu" {
                    "explicit CPU mode is slow".into()
                } else {
                    format!("explicit {f}; the runtime self-test decides")
                },
            };
        }
    }
    if !available(machine, manifest, &choice.flavour) {
        return Err(Error::Refused(format!(
            "flavour {} is not built and gated for {}",
            choice.flavour, machine.target
        )));
    }
    if choice.flavour == "hip" && choice.gpu.as_ref().and_then(|g| g.gfx.as_ref()).is_none() {
        return Err(Error::Refused(
            "hip requires a known AMD graphics target".into(),
        ));
    }
    Ok(choice)
}
