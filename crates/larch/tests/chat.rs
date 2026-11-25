//! CPU fallback and machine-readable device information through the CLI.

use std::process::Command;

use serde_json::Value;

#[test]
fn info_json_has_build_fields_and_a_usable_cpu() {
    let output = Command::new(env!("CARGO_BIN_EXE_larch"))
        .args(["info", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output);
    let info: Value = serde_json::from_slice(&output.stdout).unwrap();
    let object = info.as_object().unwrap();
    assert_eq!(object.len(), 7);
    for key in [
        "version",
        "commit",
        "backend",
        "features",
        "devices",
        "loaded_libraries",
        "errors",
    ] {
        assert!(object.contains_key(key), "missing {key}: {info}");
    }
    assert_eq!(info["errors"], serde_json::json!([]));
    assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
    assert!(info["commit"].is_null());
    let features = info["features"].as_array().unwrap();
    for (feature, enabled) in [
        ("default", cfg!(feature = "default")),
        ("fault-inject", cfg!(feature = "fault-inject")),
    ] {
        assert_eq!(features.iter().any(|value| value == feature), enabled);
    }
    let cpu = info["devices"]
        .as_array()
        .unwrap()
        .iter()
        .find(|device| device["backend"] == "cpu")
        .unwrap();
    for key in [
        "ordinal",
        "name",
        "backend",
        "usable",
        "reason",
        "driver",
        "runtime",
        "vram_total",
        "vram_free",
    ] {
        assert!(cpu.get(key).is_some(), "missing {key}: {cpu}");
    }
    assert_eq!(cpu["ordinal"], larch_kernels::CPU);
    assert_eq!(cpu["usable"], true);
    assert!(cpu["reason"].is_null());
    assert!(cpu["name"].is_string());
    for key in ["driver", "runtime", "vram_total", "vram_free"] {
        assert!(cpu[key].is_u64(), "{key}: {cpu}");
    }
    if larch_kernels::device_count().unwrap() == 0 {
        assert_eq!(info["backend"], "cpu");
    }
    for library in info["loaded_libraries"].as_array().unwrap() {
        assert!(library["name"].is_string());
        assert!(library["path"].is_string());
    }
}

#[test]
fn info_defaults_to_text() {
    let output = Command::new(env!("CARGO_BIN_EXE_larch"))
        .arg("info")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("CPU is the reference/debug backend.")
    );
}
