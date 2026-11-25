# Building larch

The Rust version is pinned to **1.97** in [rust-toolchain.toml](../rust-toolchain.toml); rustup installs it
on the first `cargo` command. The build script of `larch-sys` runs CMake on `kernels/` and links the result, so one `cargo build` builds
everything. The first build compiles the GPU kernels and takes the longest.

## macOS (Metal)

You need macOS 26 or later, Xcode 26 or later, CMake, rustup and Python 3.11 or later.

1. Install the tools:
   ```
   brew install cmake rustup python@3.12
   ```
2. Homebrew does not put rustup on `PATH`. Add it, and add the same line to `~/.zshrc` so that new shells
   find `cargo`:
   ```
   export PATH="$(brew --prefix rustup)/bin:$PATH"
   ```
3. If `xcrun -sdk macosx metal --version` fails, install the Metal toolchain:
   ```
   xcodebuild -downloadComponent MetalToolchain
   ```
4. Build:
   ```
   cargo build --release -p larch
   ```

The build selects Metal on a Mac without any configuration.

## Windows (CUDA)

You need these tools:

- Visual Studio 2022 Build Tools 17.x with the C++ workload
- CUDA Toolkit 13.x (CI uses 13.4.2) and an NVIDIA driver from the R580 branch or later
- CMake 3.24 or later, and Ninja
- LLVM, because bindgen needs libclang
- rustup with the `x86_64-pc-windows-msvc` host
- Python 3.11 or later, for the model download

1. Open "Developer PowerShell for VS 2022" and make sure that it targets x64:
   ```
   Enter-VsDevShell -VsInstallPath (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath) -DevCmdArguments "-arch=x64 -host_arch=x64"
   ```
   The commands below are PowerShell. The "x64 Native Tools Command Prompt" runs `cmd.exe`, where you set a
   variable with `set NAME=value` instead.
2. Set the build configuration:
   ```
   $env:LARCH_BACKEND = "CUDA"
   $env:LARCH_CMAKE_GENERATOR = "Ninja"
   ```
3. If bindgen cannot find libclang, set its location:
   ```
   $env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
   ```
4. Build:
   ```
   cargo build --release -p larch
   ```
5. Make sure that the binary finds the GPU:
   ```
   .\target\release\larch.exe info
   ```

The default build compiles the kernels for three GPU generations and is slow. For a faster local build, set
`LARCH_CUDA_ARCH` to the generation of your GPU:

```
$env:LARCH_CUDA_ARCH = "120"
```

At run time, `larch.exe` loads `cublas64_13.dll` and `cublasLt64_13.dll`. The CUDA installer puts their
directory on `PATH`. To run on a machine without the toolkit, copy the two DLL files next to `larch.exe`.

## Linux (CUDA)

You need CUDA Toolkit 13.x, GCC 14 or Clang, CMake 3.24 or later, Ninja, libclang and Python 3.11 or later.
On Ubuntu 24.04 (its `python3` is 3.12):

```
sudo apt install cmake ninja-build libclang-dev python3
LARCH_BACKEND=CUDA LARCH_CMAKE_GENERATOR=Ninja cargo build --release -p larch
```

On WSL, also set `CUDAHOSTCXX=/usr/bin/g++-14`. At run time, the binary loads `libcublas.so.13` from the CUDA
installation.

## Desktop app

The desktop app in `apps/desktop` is its own Cargo workspace, so the commands above do not build it.
Build it from the repository root:

```
cargo build --release --manifest-path apps/desktop/Cargo.toml
```

## Build configuration

These environment variables control the build:

| Variable | Values | Effect |
|---|---|---|
| `LARCH_BACKEND` | `AUTO` (default), `CUDA`, `HIP`, `METAL`, `CPU` | `AUTO` selects Metal on Apple hardware. On other systems, if it finds `nvcc`, it selects CUDA. If not, it selects the CPU reference |
| `LARCH_CUDA_ARCH` | for example `86`, `89`, `120` | Builds for one GPU generation. The default builds `sm_86`, `sm_89` and `sm_120` (RTX 30, 40 and 50 series) |
| `LARCH_HIP_ARCH` | for example `gfx1100` | Same, for the HIP backend |
| `LARCH_CMAKE_GENERATOR` | for example `Ninja` | The CMake generator for the kernels |

A GPU build includes its selected backend and the CPU reference; `CPU` selects a CPU-only build.
The CPU reference is used for correctness tests.
The HIP backend (AMD) compiles in CI. On hardware, only the RX 6700S (gfx1032) and Radeon 680M (gfx1035) on Windows
pass ctest and parity; other AMD GPUs are untested. A Windows HIP build needs the HIP SDK (it sets `HIP_PATH`),
Perl and Ninja: CMake's HIP language refuses the Visual Studio generators, so the build uses Ninja unless
`LARCH_CMAKE_GENERATOR` or `CMAKE_GENERATOR` names another generator.
