//! Device information from the safe kernels API.

use std::io::Write;

use anyhow::Result;
use clap::Args;
use larch_kernels::{Backend, CPU, Device, device_count};
use serde_json::{Value, json};

/// Device report format; text remains the default.
#[derive(Args)]
pub struct Options {
    /// Emit one JSON object instead of text.
    #[arg(long)]
    pub json: bool,
}

/// Lists actual devices and versions; known-model fit requires the pending model catalog producer.
pub fn run(opts: Options, out: &mut impl Write) -> Result<()> {
    if opts.json {
        return json_report(out);
    }
    let count = match device_count() {
        Ok(count) => count,
        Err(error) => {
            writeln!(out, "device enumeration: {error}")?;
            0
        }
    };
    for ordinal in std::iter::once(CPU).chain(0..count) {
        let label = if ordinal == CPU {
            "cpu".to_owned()
        } else {
            ordinal.to_string()
        };
        let props = match Device::open(ordinal).and_then(|device| device.props()) {
            Ok(props) => props,
            Err(error) => {
                writeln!(out, "device {label}: {error}")?;
                continue;
            }
        };
        // Device::open checks the ABI and, for CUDA, the minimum driver; a failure is printed above.
        let checked = if props.backend == Backend::Cuda {
            " (minimum-driver check passed)"
        } else {
            ""
        };
        let versions = if ordinal == CPU {
            String::new()
        } else {
            format!(
                ", driver {}{checked}, runtime {}",
                props.driver_version, props.runtime_version
            )
        };
        writeln!(
            out,
            "device {label}: {} ({:?}){versions}, free {}/{} bytes",
            props.name, props.backend, props.vram_free, props.vram_total
        )?;
    }
    // ponytail: no per-model fit estimate yet; add it when bench/models.toml and a public engine
    // fit API land.
    writeln!(
        out,
        "CPU is the reference/debug backend. Known-model fit estimates: unavailable (model catalog/fit API pending)."
    )?;
    Ok(())
}

fn json_report(out: &mut impl Write) -> Result<()> {
    // Failures go into the report, never instead of it: it exists for the machines that fail.
    let mut errors = Vec::new();
    let count = device_count().unwrap_or_else(|error| {
        errors.push(format!("device enumeration: {error}"));
        0
    });
    let mut devices = Vec::new();
    // ponytail: report the usable backend; report the build backend once larch-kernels has a safe
    // larch_backend_info wrapper (#208), including builds whose GPUs cannot be opened.
    let mut backend = "cpu".to_owned();
    for ordinal in std::iter::once(CPU).chain(0..count) {
        let device = match Device::open(ordinal).and_then(|device| device.props()) {
            Ok(props) => {
                let name = format!("{:?}", props.backend).to_lowercase();
                if ordinal != CPU {
                    backend.clone_from(&name);
                }
                json!({
                    "ordinal": ordinal, "name": props.name, "backend": name,
                    "usable": true, "reason": null,
                    "driver": props.driver_version, "runtime": props.runtime_version,
                    "vram_total": props.vram_total, "vram_free": props.vram_free,
                })
            }
            Err(error) => json!({
                "ordinal": ordinal, "name": null,
                "backend": if ordinal == CPU { "cpu" } else { "unknown" },
                "usable": false, "reason": error.to_string(),
                "driver": null, "runtime": null, "vram_total": null, "vram_free": null,
            }),
        };
        devices.push(device);
    }
    let features: Vec<_> = [
        ("default", cfg!(feature = "default")),
        ("fault-inject", cfg!(feature = "fault-inject")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect();
    let loaded_libraries = loaded_libraries().unwrap_or_else(|error| {
        errors.push(format!("loaded libraries: {error}"));
        Vec::new()
    });
    serde_json::to_writer(
        &mut *out,
        &json!({
            "version": env!("CARGO_PKG_VERSION"), "commit": null,
            "backend": backend, "features": features, "devices": devices,
            "loaded_libraries": loaded_libraries, "errors": errors,
        }),
    )?;
    writeln!(out)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn loaded_libraries() -> Result<Vec<Value>> {
    Ok(libraries_from_maps(&std::fs::read_to_string(
        "/proc/self/maps",
    )?))
}

#[cfg(target_os = "linux")]
fn libraries_from_maps(maps: &str) -> Vec<Value> {
    let mut paths = std::collections::BTreeSet::new();
    for line in maps.lines() {
        let Some(start) = line.find('/') else {
            continue;
        };
        let path = line[start..]
            .strip_suffix(" (deleted)")
            .unwrap_or(&line[start..]);
        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if [
            "libcublas",
            "libcublasLt",
            "libamdhip64",
            "libhipblas",
            "librocblas",
            "libhipblaslt",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            paths.insert((name, path));
        }
    }
    paths
        .into_iter()
        .map(|(name, path)| json!({ "name": name, "path": path }))
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn loaded_libraries() -> Result<Vec<Value>> {
    // ponytail: empty outside Linux; add GetModuleFileNameW in larch-sys when the installer's
    // self-test runs on Windows.
    Ok(Vec::new())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #[test]
    fn maps_include_only_gpu_libraries_once_and_preserve_spaces() {
        let maps = "1-2 r-xp 0 00:01 1 /opt/gpu libs/libcublas.so.13\n\
                    2-3 r--p 0 00:01 1 /opt/gpu libs/libcublas.so.13\n\
                    3-4 r-xp 0 00:01 2 /opt/libcublasLt.so.13 (deleted)\n\
                    4-5 r-xp 0 00:01 3 /opt/libamdhip64.so\n\
                    5-6 r-xp 0 00:01 4 /opt/libhipblas.so\n\
                    6-7 r-xp 0 00:01 5 /opt/librocblas.so\n\
                    7-8 r-xp 0 00:01 6 /opt/libhipblaslt.so\n\
                    8-9 r-xp 0 00:01 7 /opt/libother.so\n\
                    9-a rw-p 0 00:00 0 [heap]\n";
        let libraries = super::libraries_from_maps(maps);
        assert_eq!(libraries.len(), 6);
        assert!(libraries.iter().any(|library| {
            library["name"] == "libcublas.so.13"
                && library["path"] == "/opt/gpu libs/libcublas.so.13"
        }));
        assert!(
            libraries
                .iter()
                .any(|library| library["path"] == "/opt/libcublasLt.so.13")
        );
    }
}
