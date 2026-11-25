// Builds kernels/ with CMake, links it, and runs bindgen on kernels/include/larch.h.
// Env: LARCH_BACKEND = AUTO (default) | CUDA | HIP | METAL | CPU,
// LARCH_CUDA = AUTO | ON | OFF (old alias, used only when LARCH_BACKEND is unset),
// LARCH_CUDA_ARCH (e.g. 120 for fast local builds; the default list is in kernels/CMakeLists.txt),
// LARCH_HIP_ARCH (e.g. gfx1100; same), LARCH_CMAKE_GENERATOR (e.g. Ninja).

// Without the `kernels` feature nothing is built: the crate is OS glue only.
#![cfg_attr(not(feature = "kernels"), allow(dead_code, unused_imports))]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

fn backend() -> String {
    if let Ok(b) = env::var("LARCH_BACKEND") {
        return b.to_ascii_uppercase();
    }
    match env::var("LARCH_CUDA").map(|v| v.to_ascii_uppercase()) {
        Ok(v) => {
            println!(
                "cargo::warning=LARCH_CUDA is deprecated; use LARCH_BACKEND=AUTO|CUDA|HIP|METAL|CPU"
            );
            match v.as_str() {
                "ON" => "CUDA".into(),
                "OFF" => "CPU".into(),
                _ => "AUTO".into(),
            }
        }
        Err(_) => "AUTO".into(),
    }
}

fn main() {
    #[cfg(feature = "kernels")]
    kernels();
}

#[cfg(feature = "kernels")]
fn kernels() {
    // Not canonicalize(): on Windows it returns a \\?\ path, which CMake's file(GLOB) cannot expand.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels = manifest.ancestors().nth(2).unwrap().join("kernels");

    for var in [
        "LARCH_BACKEND",
        "LARCH_CUDA",
        "LARCH_CUDA_ARCH",
        "LARCH_HIP_ARCH",
        "LARCH_CMAKE_GENERATOR",
        "CUDACXX",
        "CUDAHOSTCXX",
        "CUDA_PATH",
        "HIPCXX",
        "ROCM_PATH",
        "HIP_PATH",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    for path in ["CMakeLists.txt", "metal.cmake", "include", "src"] {
        println!("cargo::rerun-if-changed={}", kernels.join(path).display());
    }

    let mut cfg = cmake::Config::new(&kernels);
    cfg.define("LARCH_BACKEND", backend())
        .define("LARCH_BUILD_TESTS", "OFF")
        // Release CRT on MSVC, to match Rust.
        .profile("RelWithDebInfo");
    for (var, cache) in [
        ("LARCH_CUDA_ARCH", "CMAKE_CUDA_ARCHITECTURES"),
        ("LARCH_HIP_ARCH", "CMAKE_HIP_ARCHITECTURES"),
    ] {
        match env::var(var) {
            Ok(arch) => cfg.define(cache, arch),
            // Clear the cached value, or an override that was unset would stay in force.
            Err(_) => cfg.configure_arg(format!("-U{cache}")),
        };
    }
    if let Ok(generator) = env::var("LARCH_CMAKE_GENERATOR") {
        cfg.generator(generator);
    }
    // The kernels' MSVC runtime follows Rust's, so a RUSTFLAGS without .cargo/config.toml's
    // +crt-static links /MD on both sides instead of mixing two runtimes.
    if env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|e| e == "msvc") {
        let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let runtime = if features.split(',').any(|f| f == "crt-static") {
            "MultiThreaded"
        } else {
            "MultiThreadedDLL"
        };
        cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", runtime);
        // A HIP build compiles C and C++ with the HIP SDK's GNU-style clang (kernels/CMakeLists.txt), which
        // rejects cmake-rs's cl.exe flags. Passing cl.exe would also make each configure see a changed compiler
        // and drop the cache, so pass the same clang path CMakeLists.txt builds from HIP_PATH.
        if backend() == "HIP" {
            // cmake-rs picks a Visual Studio generator on MSVC, which CMake's HIP language refuses. Ninja
            // unless a generator was named: LARCH_CMAKE_GENERATOR, or any name cmake-rs 0.1.58 reads for it.
            let target = env::var("TARGET").expect("cargo sets TARGET");
            let kind = if env::var("HOST").is_ok_and(|h| h == target) {
                "HOST"
            } else {
                "TARGET"
            };
            let named = [
                "LARCH_CMAKE_GENERATOR".to_string(),
                format!("CMAKE_GENERATOR_{target}"),
                format!("CMAKE_GENERATOR_{}", target.replace('-', "_")),
                format!("{kind}_CMAKE_GENERATOR"),
                "CMAKE_GENERATOR".to_string(),
            ];
            if named.iter().all(|v| env::var_os(v).is_none()) {
                cfg.generator("Ninja");
            }
            cfg.define("CMAKE_C_FLAGS", "")
                .define("CMAKE_CXX_FLAGS", "");
            if let Ok(hip) = env::var("HIP_PATH") {
                let bin = format!("{}/bin", hip.replace('\\', "/"));
                cfg.define("CMAKE_C_COMPILER", format!("{bin}/clang.exe"))
                    .define("CMAKE_CXX_COMPILER", format!("{bin}/clang++.exe"));
            }
        }
    }
    let dst = cfg.build();

    let info = fs::read_to_string(dst.join("share/larch/larch_link_info.txt")).unwrap();
    let info: HashMap<&str, &str> = info.lines().filter_map(|l| l.split_once('=')).collect();
    let backend = info["backend"];
    println!("cargo::rustc-env=LARCH_BUILT_BACKEND={backend}");

    println!(
        "cargo::rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo::rustc-link-lib=static=larch_kernels");

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    match backend {
        "cuda" => {
            println!("cargo::rustc-link-search=native={}", info["cuda_lib_dir"]);
            println!("cargo::rustc-link-lib=static=cudart_static");
            // cublas64_13.dll / libcublas.so.13, loaded from the toolkit or alongside the binary.
            println!("cargo::rustc-link-lib=dylib=cublas");
            if os == "linux" {
                for lib in ["dl", "rt", "pthread"] {
                    println!("cargo::rustc-link-lib=dylib={lib}");
                }
            }
        }
        "hip" => {
            println!("cargo::rustc-link-search=native={}", info["hip_lib_dir"]);
            for lib in ["hipblas", "amdhip64"] {
                println!("cargo::rustc-link-lib=dylib={lib}");
            }
            // Windows: the kernels' C++ is built by clang, whose builtins library link.exe needs too.
            if let Some(rt) = info.get("clang_rt").map(std::path::Path::new) {
                println!(
                    "cargo::rustc-link-search=native={}",
                    rt.parent()
                        .expect("clang_rt path from CMake has a folder")
                        .display()
                );
                println!(
                    "cargo::rustc-link-lib={}",
                    rt.file_stem()
                        .expect("clang_rt path from CMake names a file")
                        .to_string_lossy()
                );
            }
        }
        "metal" => {
            for framework in ["Metal", "Foundation", "MetalPerformanceShaders", "IOKit"] {
                println!("cargo::rustc-link-lib=framework={framework}");
            }
        }
        _ => {}
    }
    if env::var("CARGO_CFG_TARGET_ENV").unwrap() != "msvc" {
        println!(
            "cargo::rustc-link-lib=dylib={}",
            if os == "macos" { "c++" } else { "stdc++" }
        );
    }

    bindgen::Builder::default()
        .header(kernels.join("include/larch.h").to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("larch_.*")
        .allowlist_type("larch_.*")
        .allowlist_var("LARCH_.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .rust_edition(bindgen::RustEdition::Edition2024)
        .generate()
        .unwrap()
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
}
