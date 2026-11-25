#!/usr/bin/env python3
"""Build, audit and inventory release packages using the standard library and native tools."""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REPO = "Arkaya-Labs/Larch"
CUDA = "https://developer.download.nvidia.com/compute/cuda/redist"
CUDA_VERSION = "13.4.2"
ROCM = "rocm/dev-ubuntu-24.04:10.0.0-full@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374"
TARGETS = ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "aarch64-apple-darwin")
VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
ARCHIVE = re.compile(r"^(cli|app|setup|cuda|rocm-gfx[0-9]+)-(" + "|".join(TARGETS) + r")-(cpu|cuda|hip|metal)-(" + VERSION + r")\.tar\.zst$")
# Linux libraries the user's system provides, so no runtime package ships them: glibc, the GPU driver's, and
# the X11 keyboard stack the app and the installer window link (the packages .github/actions/release-tools
# installs; Wayland, Vulkan and fontconfig are loaded at run time, not linked).
SYSTEM_LIBS = re.compile(r"(?:lib(?:c|m|pthread|dl|rt|gcc_s|stdc\+\+|cuda|drm|drm_amdgpu|numa|tinfo|z|elf|atomic|xcb|xcb-xkb|xkbcommon|xkbcommon-x11|Xau|Xdmcp|bsd|md)\.so(?:\.[0-9]+)*|ld-linux-x86-64\.so\.2)")


def valid_version(value):
    """Match the installer version grammar before editing any files."""
    if not re.fullmatch(VERSION, value):
        return False
    core, _, pre = value.partition("-")
    return all(int(v) <= 2**64 - 1 for v in core.split(".")) and not any(p.isdigit() and len(p) > 1 and p.startswith("0") for p in pre.split("."))


def run(*args, **kwargs):
    """Treat a native tool failure as a release failure."""
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def output(*args, env=None):
    """Capture inspector results without exposing environments."""
    return run(*args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def sha(path):
    """Hash large archives without loading them into memory."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(value):
    """Apply installer path restrictions before touching the filesystem."""
    value.encode("utf-8", "strict")
    if not value or value.startswith("/") or "\\" in value or ":" in value:
        raise ValueError("unsafe package path")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    for part in parts:
        stem = part.split(".")[0].upper()
        if part == ".." or part.endswith((".", " ")) or any(unicodedata.category(c) == "Cc" or c in '<>"|?*' for c in part) or stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem):
            raise ValueError("unsafe package path")
    if not parts:
        raise ValueError("empty package path")
    return "/".join(parts)


def link_target(path, target):
    """Prove that a package symlink stays inside its component."""
    if not target or target.startswith("/") or "\\" in target or ":" in target or any(unicodedata.category(c) == "Cc" for c in target):
        raise ValueError("unsafe symlink")
    parts = [p for p in target.split("/") if p not in ("", ".")]
    named = False
    resolved = path.split("/")[:-1]
    for part in parts:
        if part == "..":
            if named or not resolved:
                raise ValueError("ambiguous or escaping symlink")
            resolved.pop()
        else:
            relative(part)
            named = True
            resolved.append(part)
    normalized = "/".join(parts) or "."
    return normalized, "/".join(resolved)


@contextlib.contextmanager
def unpacked(path):
    """Stream the codec output while retaining the original validation failure."""
    process = subprocess.Popen(["zstd", "-q", "-d", "--memory=128MB", "-c", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process.stdout
    finally:
        failed = sys.exc_info()[0] is not None
        if not failed:
            # tarfile stops at the end-of-archive block, which can leave the last record unread; a
            # Windows pipe buffer cannot hold it, so zstd would fail with a broken pipe.
            while process.stdout.read(1 << 16):
                pass
        elif process.poll() is None:
            process.kill()
        process.stdout.close()
        error = process.stderr.read()
        process.stderr.close()
        status = process.wait()
        if status and not failed:
            raise ValueError("zstd decode failed: " + error.decode(errors="replace")[:512])


def inventory(path):
    """Describe every archive file and refuse aliases, escapes and special devices."""
    files, hard, symbolic, explicit_dirs = {}, {}, {}, set()
    with unpacked(path) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for entry in archive:
            if entry.isdir() and entry.name in (".", "./"):
                continue
            name = relative(entry.name.rstrip("/") if entry.isdir() else entry.name)
            if entry.isdir():
                if name in explicit_dirs:
                    raise ValueError("duplicate directory")
                explicit_dirs.add(name)
                continue
            if name in files or name in hard or name in symbolic:
                raise ValueError("duplicate file")
            if entry.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                if not 0 <= entry.size <= 2**64 - 1:
                    raise ValueError("invalid file size")
                with archive.extractfile(entry) as content:
                    digest = hashlib.file_digest(content, "sha256").hexdigest()
                files[name] = (digest, entry.size)
            elif entry.issym():
                target, resolved = link_target(name, entry.linkname)
                symbolic[name] = resolved
                files[name] = (hashlib.sha256(b"symlink\0" + target.encode()).hexdigest(), len(target.encode()))
            elif entry.islnk():
                hard[name] = relative(entry.linkname)
            else:
                raise ValueError("unsupported archive entry type")
    while hard:
        progress = False
        for name, target in list(hard.items()):
            if target in files and target not in symbolic:
                files[name] = files[target]
                del hard[name]
                progress = True
        if not progress:
            raise ValueError("hardlink target is missing or cyclic")
    aliases = set()
    dirs = set()
    spellings = {}
    for name in files:
        if name.casefold() in aliases:
            raise ValueError("case-aliasing inventory")
        aliases.add(name.casefold())
        parents = [str(p) for p in PurePosixPath(name).parents if str(p) != "."]
        for path in [name, *parents]:
            previous = spellings.setdefault(path.casefold(), path)
            if previous != path:
                raise ValueError("case-aliasing package directory")
        dirs.update(parents)
    if any(p.casefold() in aliases for p in dirs) or not explicit_dirs.issubset(dirs):
        raise ValueError("file/directory collision or unlisted empty directory")
    for name, target in symbolic.items():
        for _ in range(len(symbolic) + 1):
            parts = target.split("/") if target else []
            for i in range(1, len(parts) + 1):
                prefix = "/".join(parts[:i])
                if prefix in symbolic:
                    target = "/".join(filter(None, [symbolic[prefix], *parts[i:]]))
                    break
            else:
                if target not in files and target not in dirs and target != "":
                    raise ValueError("dangling symlink")
                break
        else:
            raise ValueError("cyclic symlink")
    return [(name, *files[name]) for name in sorted(files)]


def pack(source, dest):
    """Write a canonical component archive and verify its inventory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        tar = Path(temp) / "payload.tar"
        with tarfile.open(tar, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(source.rglob("*")):
                if not path.is_dir() or path.is_symlink():
                    archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        run("zstd", "-q", "-f", "-19", "--long=27", tar, "-o", dest)
    inventory(dest)


def prepare(version):
    """Set release versions and update both lockfiles offline."""
    if not valid_version(version):
        raise ValueError("tag must contain a supported semantic version")
    path = ROOT / "Cargo.toml"
    text = path.read_text()
    text, count = re.subn(r'(\[workspace\.package\][\s\S]*?\nversion\s*=\s*)"[^"]+"', lambda m: m[1] + json.dumps(version), text, count=1)
    if count != 1:
        raise ValueError("workspace version is missing")
    path.write_text(text)
    desktop = ROOT / "apps/desktop"
    path = desktop / "Cargo.toml"
    text, count = re.subn(r'(?m)^(version\s*=\s*)"[^"]+"', lambda m: m[1] + json.dumps(version), path.read_text(), count=1)
    if count != 1:
        raise ValueError("desktop package version is missing")
    path.write_text(text)
    run("cargo", "update", "--workspace", "--offline", cwd=ROOT)
    run("cargo", "update", "--workspace", "--offline", "--manifest-path", desktop / "Cargo.toml", cwd=ROOT)


def build(args):
    """Build locked components with the required deployment and CRT settings."""
    prepare(args.version)
    target, flavour, dest = args.build, args.flavour, args.archives
    windows = "windows" in target
    env = dict(os.environ, LARCH_BACKEND=flavour.upper())
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    # Git Bash puts its own link.exe first on PATH, and inside a developer shell rustc takes PATH's.
    # os.environ, not env: only os.environ matches Windows variable names case-insensitively.
    if windows and os.environ.get("VCToolsInstallDir"):
        env["CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"] = str(Path(os.environ["VCToolsInstallDir"]) / "bin/Hostx64/x64/link.exe")
    suffix = ".exe" if windows else ""
    desktop = ROOT / "apps/desktop"
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        if args.setup_only:
            base = dict(env)
            actual = "x86_64-unknown-linux-musl" if "linux" in target else target
            if "linux" in target:
                env["CC_x86_64_unknown_linux_musl"] = "musl-gcc"
                env["CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER"] = "musl-gcc"
            env["LARCH_BACKEND"] = "CPU"
            env["CARGO_TARGET_DIR"] = str((ROOT.parent / ("target-setup-" + target)).resolve())
            if target == "aarch64-apple-darwin":
                env["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
            if windows:
                env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C target-feature=+crt-static"
            run("cargo", "build", "--locked", "--release", "-p", "larch-setup", "--target", actual, cwd=ROOT, env=env)
            source = Path(env["CARGO_TARGET_DIR"]) / actual / "release" / ("larch-setup" + suffix)
            stage = temp / "setup"
            stage.mkdir()
            shutil.copy2(source, stage / source.name)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / ("larch-setup-" + target + suffix))
            window(args, base, temp, stage)
            pack(stage, dest / f"setup-{target}-cpu-{args.version}.tar.zst")
            return
        if windows:
            env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C target-feature=+crt-static"
        if "apple" in target:
            env["MACOSX_DEPLOYMENT_TARGET"] = "26.0"
        env["CARGO_TARGET_DIR"] = str(Path(os.environ.get("CARGO_TARGET_DIR", ROOT.parent / ("target-release-" + flavour + "-" + target))).resolve())
        if "linux" in target:
            origin = "$ORIGIN"
            if flavour == "cuda":
                origin += ":$ORIGIN/../cuda"
            elif flavour == "hip":
                origin += "".join(":$ORIGIN/../rocm-" + g for g in json.loads((ROOT / "crates/larch-setup/release/rules.json").read_text())["hip_targets"])
            env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C link-arg=-Wl,--disable-new-dtags -C link-arg=-Wl,-rpath," + origin
        run("cargo", "build", "--locked", "--release", "-p", "larch", "-p", "larch-agent", "--target", target, cwd=ROOT, env=env)
        run("cargo", "build", "--locked", "--release", "--manifest-path", desktop / "Cargo.toml", "--target", target, cwd=ROOT, env=env)
        bins = Path(env["CARGO_TARGET_DIR"]) / target / "release"
        cli = temp / "cli"
        cli.mkdir()
        for name in ("larch", "larch-agent"):
            shutil.copy2(bins / (name + suffix), cli / (name + suffix))
        app = temp / "app"
        app.mkdir()
        if "apple" in target:
            executable = app / "Contents/MacOS"
            executable.mkdir(parents=True)
            shutil.copy2(bins / "larch-desktop", executable / "larch-desktop")
            resources = app / "Contents/Resources"
            resources.mkdir()
            shutil.copy2(desktop / "icons/icon.icns", resources / "icon.icns")
            (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable":"larch-desktop", "CFBundleIdentifier":"com.arkaya-labs.larch", "CFBundleName":"Larch", "CFBundlePackageType":"APPL", "CFBundleShortVersionString":args.version, "CFBundleVersion":args.version, "CFBundleIconFile":"icon.icns", "LSMinimumSystemVersion":"26.0"}))
            run("codesign", "--force", "--sign", "-", app)
        else:
            shutil.copy2(bins / ("larch-desktop" + suffix), app / ("larch-desktop" + suffix))
        for name, directory in (("cli", cli), ("app", app)):
            pack(directory, dest / f"{name}-{target}-{flavour}-{args.version}.tar.zst")


def window(args, env, temp, stage):
    """Build the installer window, which runs itself as larch-setup: the download people double-click,
    and a copy in setup's package for the uninstall entry and launchers."""
    target, windows = args.build, "windows" in args.build
    path = ROOT / "apps/setup/Cargo.toml"
    text, count = re.subn(r'(?m)^(version\s*=\s*)"[^"]+"', lambda m: m[1] + json.dumps(args.version), path.read_text(), count=1)
    if count != 1:
        raise ValueError("installer window version is missing")
    path.write_text(text)
    # Its path dependencies (larch-setup and what it links) took the release version from prepare().
    run("cargo", "update", "--workspace", "--offline", "--manifest-path", path, cwd=ROOT)
    env["CARGO_TARGET_DIR"] = str((ROOT.parent / ("target-window-" + target)).resolve())
    if windows:
        env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C target-feature=+crt-static"
    if "apple" in target:
        env["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    run("cargo", "build", "--locked", "--release", "--manifest-path", path, "--target", target, cwd=ROOT, env=env)
    binary = Path(env["CARGO_TARGET_DIR"]) / target / "release" / ("larch-installer" + (".exe" if windows else ""))
    dest = args.archives
    dest.mkdir(parents=True, exist_ok=True)
    if "apple" in target:
        app = temp / "Larch Setup.app"
        (app / "Contents/MacOS").mkdir(parents=True)
        (app / "Contents/Resources").mkdir()
        shutil.copy2(binary, app / "Contents/MacOS/larch-installer")
        shutil.copy2(ROOT / "apps/desktop/icons/icon.icns", app / "Contents/Resources/icon.icns")
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable":"larch-installer", "CFBundleIdentifier":"com.arkaya-labs.larch.setup", "CFBundleName":"Larch Setup", "CFBundlePackageType":"APPL", "CFBundleShortVersionString":args.version, "CFBundleVersion":args.version, "CFBundleIconFile":"icon.icns", "LSMinimumSystemVersion":"11.0", "NSHighResolutionCapable":True}))
        run("codesign", "--force", "--sign", "-", app)
        # ditto keeps the bundle's signature and modes; Finder and Safari open the zip in place.
        run("ditto", "-c", "-k", "--keepParent", app, dest / f"larch-installer-{target}.zip")
        shutil.copytree(app, stage / app.name, symlinks=True)
    else:
        suffix = ".exe" if windows else ""
        shutil.copy2(binary, dest / f"larch-installer-{target}{suffix}")
        shutil.copy2(binary, stage / f"larch-installer{suffix}")


class NoRedirectAuth(urllib.request.HTTPRedirectHandler):
    """Apply the same no-redirect-auth rule as the installer transport."""
    def redirect_request(self, request, *args, **kwargs):
        """Strip authorization even when a redirect stays on the same host."""
        redirected = super().redirect_request(request, *args, **kwargs)
        if redirected:
            redirected.remove_header("Authorization")
        return redirected


def request(url, *, token=None, data=None, method=None, content_type="application/json", size=None, accept="application/vnd.github+json"):
    """Keep bearer credentials off every redirect."""
    headers = {"User-Agent":"larch-release", "Accept":accept}
    if token:
        headers["Authorization"] = "Bearer " + token
    if data is not None:
        headers["Content-Type"] = content_type
    if size is not None:
        headers["Content-Length"] = str(size)
    return urllib.request.build_opener(NoRedirectAuth()).open(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=120)


def retried(call):
    """Retry a read through GitHub's transient 5xx answers and dropped connections, which otherwise fail a whole release stage."""
    for wait in (5, 20, 60, None):
        try:
            return call()
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as error:
            if wait is None or isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            time.sleep(wait)


def api(path, token, body=None, method=None):
    """Use the release API without logging authentication."""
    def call():
        with request("https://api.github.com/repos/" + REPO + "/" + path, token=token, data=json.dumps(body).encode() if body is not None else None, method=method) as response:
            return json.load(response) if response.status != 204 else None
    # ponytail: only reads retry (a retried POST or DELETE can repeat a write); retry writes too if a stage step dies on a 5xx.
    return retried(call) if body is None and method is None else call()


def release_by_tag(tag, token):
    """Find a release, drafts included: GitHub's by-tag endpoint returns only published ones."""
    page = 1
    while True:
        releases = api("releases?per_page=100&page=" + str(page), token)
        found = [r for r in releases if r["tag_name"] == tag]
        if found or len(releases) < 100:
            return found[0] if found else None
        page += 1


def upload(release, path, token):
    """Replace one draft asset while streaming its bytes."""
    import urllib.parse
    for asset in release.get("assets", []):
        if asset["name"] == path.name:
            api("releases/assets/" + str(asset["id"]), token, method="DELETE")
    url = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(path.name, safe="")
    with path.open("rb") as stream, request(url, token=token, data=stream, size=path.stat().st_size, method="POST", content_type="application/octet-stream") as response:
        return json.load(response)


def download(asset, path, token):
    """Download only repository assets without forwarding credentials."""
    if not re.fullmatch(r"https://api\.github\.com/repos/" + re.escape(REPO) + r"/releases/assets/[0-9]+", asset["url"]):
        raise ValueError("asset URL is outside the release repository")
    def call():
        with request(asset["url"], token=token, accept="application/octet-stream") as response, path.open("wb") as out:
            shutil.copyfileobj(response, out)
    retried(call)


def open_draft(version, token, fresh=False):
    """The version's draft, created on first use; the release jobs pass files through its assets."""
    release = release_by_tag("v" + version, token) or api("releases", token, {"tag_name":"v" + version, "name":"v" + version, "draft":True}, method="POST")
    if not release["draft"]:
        raise ValueError("refusing to replace a published release")
    if fresh:
        # A re-tagged version starts over: no file of an earlier run may reach its manifest.
        for asset in release["assets"]:
            api("releases/assets/" + str(asset["id"]), token, method="DELETE")
        release["assets"] = []
    return release


def dependencies(path, windows):
    """Read imports with the native package inspector."""
    if windows:
        return re.findall(r"(?im)^\s+([\w.+-]+\.dll)\s*$", output("dumpbin", "/nologo", "/dependents", path))
    return re.findall(r"\(NEEDED\).*?\[([^]]+)\]", output("readelf", "-d", path))


def runtime_files(binaries, prefix, stage, windows):
    """Copy the transitive vendor library closure and preserve local aliases."""
    candidates = {}
    for path in prefix.rglob("*"):
        if path.is_file():
            candidates.setdefault(path.name.lower() if windows else path.name, path)
    todo = [dep for binary in binaries for dep in dependencies(binary, windows)]
    copied = set()
    while todo:
        name = todo.pop()
        key = name.lower() if windows else name
        if key in copied:
            continue
        source = candidates.get(key)
        if source is None:
            if (windows and (name.lower().startswith(("api-ms-", "ext-ms-")) or (Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / name).exists())) or (not windows and SYSTEM_LIBS.fullmatch(name)):
                continue
            raise ValueError("unresolved package dependency: " + name)
        copied.add(key)
        real = source.resolve()
        shutil.copy2(real, stage / real.name)
        if source.name != real.name:
            os.symlink(real.name, stage / source.name)
        todo.extend(dependencies(real, windows))
    if not copied:
        raise ValueError("runtime dependency closure is empty")


def extract_package(path, dest):
    """Validate archive contents before extracting an audit copy."""
    inventory(path)
    with unpacked(path) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        archive.extractall(dest, filter="data")


def redist(components, target, prefix):
    """Verify each pinned NVIDIA archive before unpacking it."""
    with request(f"{CUDA}/redistrib_{CUDA_VERSION}.json") as response:
        upstream = json.load(response)
    prefix.mkdir(parents=True, exist_ok=True)
    provenance = []
    windows = "windows" in target
    platform = "windows-x86_64" if windows else "linux-x86_64"
    with tempfile.TemporaryDirectory() as temp:
        for component in components:
            item = upstream[component][platform]
            url = CUDA + "/" + item["relative_path"]
            download = Path(temp) / Path(item["relative_path"]).name
            with request(url) as response, download.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha(download) != item["sha256"]:
                raise ValueError("NVIDIA redistributable SHA-256 mismatch")
            if windows:
                import zipfile
                with zipfile.ZipFile(download) as archive:
                    archive.extractall(prefix)
            else:
                with tarfile.open(download) as archive:
                    archive.extractall(prefix, filter="data")
            provenance.append({"url":url, "sha256":item["sha256"]})
    return provenance


def runtimes(args):
    """Repackage selected backend libraries from recorded upstream sources."""
    windows = "windows" in args.target
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        binaries = []
        for path in args.archives.glob(f"*-{args.target}-{args.runtimes}-{args.version}.tar.zst"):
            if path.name.startswith(("cli-", "app-")):
                dest = temp / path.name
                extract_package(path, dest)
                binaries.extend(p for p in dest.rglob("*") if p.is_file() and (p.suffix.lower() == ".exe" if windows else os.access(p, os.X_OK)))
        if not binaries:
            raise ValueError("no matching packaged executable for runtime dependency discovery")
        if args.runtimes == "cuda":
            prefix = temp / "cuda"
            components = ("libcublas", "libnvjitlink")
            provenance = redist(components, args.target, prefix)
            stage = temp / "runtime"
            stage.mkdir()
            runtime_files(binaries, prefix, stage, windows)
            # The CUDA EULA lets these libraries ship with an application (its Attachment A); NVIDIA's
            # license text, from the same hash-checked archive, ships beside them.
            for component in components:
                licenses = list(prefix.glob(component + "-*-archive/LICENSE"))
                if len(licenses) != 1:
                    raise ValueError(f"NVIDIA {component} archive has no single LICENSE")
                shutil.copy2(licenses[0], stage / f"{component}-LICENSE.txt")
            emit_runtime(args, "cuda", stage, provenance)
        else:
            if windows or args.image != ROCM:
                raise ValueError("ROCm repackaging requires the pinned Linux image")
            for gfx in json.loads((ROOT / "crates/larch-setup/release/rules.json").read_text())["hip_targets"]:
                stage = temp / gfx
                stage.mkdir()
                # ldd resolves the actual closure inside the pinned image, never the host toolkit.
                resolved = []
                for binary in binaries:
                    text = output("ldd", binary, env=dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"))
                    missing = [n for n in re.findall(r"(?m)^\s*(\S+) => not found", text) if not SYSTEM_LIBS.fullmatch(n)]
                    if missing:
                        raise ValueError("unresolved ROCm dependency: " + ", ".join(missing))
                    resolved.extend(Path(p) for p in re.findall(r"=>\s+(/opt/rocm/\S+)", text))
                if not resolved:
                    raise ValueError("no ROCm dependencies reported by ldd")
                for source in set(resolved):
                    shutil.copy2(source.resolve(), stage / source.resolve().name)
                    if source.name != source.resolve().name:
                        os.symlink(source.resolve().name, stage / source.name)
                for path, dest in blas_libraries(Path("/opt/rocm/lib"), gfx):
                    (stage / dest).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, stage / dest)
                emit_runtime(args, "rocm-" + gfx, stage, [{"image":ROCM}])


def blas_libraries(lib, gfx):
    """rocBLAS and hipBLASLt kernel files for one GPU target with their target-free indexes, as (source, packaged path)."""
    chosen = []
    for name in ("rocblas", "hipblaslt"):
        source = lib / name / "library"
        files = [p for p in source.rglob("*") if p.is_file() and (set(re.findall(r"gfx[0-9]+", p.name)) == {gfx} or (not re.search(r"gfx[0-9]+", p.name) and p.suffix in (".yaml", ".json")))]
        if any(gfx in p.name for p in files):
            chosen.extend((p, Path(name) / "library" / p.relative_to(source)) for p in files)
        # hipBLASLt builds only some targets (not gfx1030 or gfx1102) and rocBLAS calls it only on those;
        # rocBLAS itself, behind hipBLAS, needs kernels for every target.
        elif name == "rocblas":
            raise ValueError("missing rocBLAS library files for " + gfx)
    return chosen


def emit_runtime(args, name, stage, provenance):
    """Keep runtime provenance beside its compressed package."""
    stem = f"{name}-{args.target}-{args.runtimes}-{args.version}"
    pack(stage, args.archives / (stem + ".tar.zst"))
    (args.archives / (stem + ".provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n")


def audit(args, manifest):
    """Make ABI, architecture, feature and loader checks hard failures."""
    features = None
    audited = {}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for path in sorted(args.archives.glob("*.tar.zst")):
            match = ARCHIVE.fullmatch(path.name)
            if not match:
                raise ValueError("noncanonical package name")
            component, target, flavour, _ = match.groups()
            native = ("windows" in target) == (os.name == "nt") and ("apple" in target) == (sys.platform == "darwin")
            if not native:
                continue
            audited[path.name] = sha(path)
            stage = temp / (target + "-" + flavour) / component
            extract_package(path, stage)
        for group in temp.iterdir():
            if sys.platform == "darwin" and (group / "app/Contents").is_dir():
                run("codesign", "-v", group / "app")
            windows = "windows" in group.name
            for path in group.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                if windows and path.suffix.lower() in (".exe", ".dll"):
                    deps = dependencies(path, True)
                    if any(re.match(r"(?i)(vcruntime140|msvcp140)", d) for d in deps):
                        raise ValueError("dynamic MSVC CRT import: " + path.name)
                elif sys.platform == "darwin" and os.access(path, os.X_OK):
                    run("codesign", "-v", path)
                    if path.name == "larch-setup":
                        load = output("otool", "-l", path)
                        minimum = re.findall(r"\bminos\s+([0-9.]+)", load)
                        if not minimum or any(tuple(map(int, v.split("."))) > (11, 0, 0) for v in minimum):
                            raise ValueError("setup deployment floor exceeds macOS 11")
                elif sys.platform.startswith("linux") and (os.access(path, os.X_OK) or ".so" in path.name):
                    if path.name == "larch-setup":
                        if "(NEEDED)" in output("readelf", "-d", path):
                            raise ValueError("Linux setup must be static")
                        if output(path, "--version").split()[1] != args.version:
                            raise ValueError("setup version mismatch")
                        continue
                    text = output("objdump", "-T", path)
                    for symbol, key in (("GLIBC_", "glibc_min"), ("GLIBCXX_", "glibcxx_min")):
                        versions = re.findall(re.escape(symbol) + r"([0-9]+(?:\.[0-9]+)+)", text)
                        if versions:
                            maximum = max(versions + [manifest["floors"][key]], key=lambda v: tuple(map(int, v.split("."))))
                            manifest["floors"][key] = maximum
                    if path.name in ("larch", "larch-agent", "larch-desktop"):
                        dynamic = output("readelf", "-d", path)
                        flavour = group.name.rsplit("-", 1)[1]
                        expected = "$ORIGIN" + (":$ORIGIN/../cuda" if flavour == "cuda" else "".join(":$ORIGIN/../rocm-" + g for g in manifest["hip_targets"]) if flavour == "hip" else "")
                        if not re.search(r"\(RPATH\).*?\[" + re.escape(expected) + r"\]", dynamic):
                            raise ValueError("missing expected DT_RPATH: " + path.name)
            cli = group / "cli" / ("larch.exe" if windows else "larch")
            if not cli.exists():
                continue
            flavour = group.name.rsplit("-", 1)[1]
            if windows:
                runtime = group / ("cuda" if flavour == "cuda" else "rocm-gfx1100")
                for dll in runtime.glob("*.dll"):
                    for name in ("cli", "app"):
                        if (group / name).is_dir():
                            shutil.copy2(dll, group / name / dll.name)
            if flavour == "cuda":
                listing = output("cuobjdump", "--list-elf", "--list-ptx", cli)
                if any(not re.search(r"ELF file.*\." + sm + r"\.cubin", listing) for sm in ("sm_86", "sm_89", "sm_120")) or not re.search(r"PTX file.*\.sm_120\.ptx", listing):
                    raise ValueError("CUDA SASS/PTX architecture gate failed")
            elif flavour == "hip":
                fatbin = group / "hip-fatbin"
                run("llvm-objcopy", "--dump-section=.hip_fatbin=" + str(fatbin), cli)
                listing = output("clang-offload-bundler", "--list", "--type=o", "--input=" + str(fatbin))
                if any(not re.search(r"--" + gfx + r"(?::|$)", listing, re.M) for gfx in manifest["hip_targets"]):
                    raise ValueError("HIP target gate failed")
            env = dict(os.environ)
            env["PATH"] = os.pathsep.join(p for p in os.get_exec_path() if not re.search(r"(?i)cuda|rocm", p))
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("DYLD_LIBRARY_PATH", None)
            env["LARCH_INSTALL_ROOT"] = str(group)
            version = output(cli, "--version", env=env).split()
            if len(version) < 2 or version[1] != args.version:
                raise ValueError("packaged version mismatch without toolkit PATH")
            current = sorted(json.loads(output(cli, "info", "--json", env=env))["features"])
            if features is None:
                features = current
            elif current != features:
                raise ValueError("cargo features differ across packaged flavours")
    return features, audited


def assemble(args):
    # ponytail: schema 1 works from 0.0.0; raise min_setup when a newer installer contract is used.
    """Require native package audits before uploading an unsigned draft."""
    rules = json.loads((ROOT / "crates/larch-setup/release/rules.json").read_text())
    manifest = dict(rules, schema=1, min_setup="0.0.0", version=args.version, channel=args.channel, components={}, runtimes={}, gated=[], models=[])
    for path in sorted(args.archives.glob("*.tar.zst")):
        match = ARCHIVE.fullmatch(path.name)
        if not match:
            raise ValueError("noncanonical archive filename")
        name, target, flavour, version = match.groups()
        if version != args.version:
            raise ValueError("archive version differs from release version")
        item = {"url":f"https://github.com/{REPO}/releases/download/v{args.version}/{path.name}", "sha256":sha(path), "size":path.stat().st_size, "files":inventory(path)}
        table = "components" if name in ("cli", "app", "setup") else "runtimes"
        if table == "runtimes":
            provenance = path.with_name(path.name.removesuffix(".tar.zst") + ".provenance.json")
            item["provenance"] = json.dumps(json.loads(provenance.read_text()), separators=(",", ":"))
        manifest[table].setdefault(name, {})[target + "-" + flavour] = item
    if not manifest["components"]:
        raise ValueError("no component archives")
    for alias, model in tomllib.loads((ROOT / "bench/models.toml").read_text()).items():
        if "catalog" in model.get("use", []):
            # larch-setup refuses an unpinned revision at download time; refuse it here, before signing.
            if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]) or model["display"] in ("", alias):
                raise ValueError(f"catalog model {alias} needs a pinned revision and a display name")
            manifest["models"].append({k:model[k] for k in ("display", "repo", "file", "revision", "sha256", "size", "min_mem_gb")} | {"license_url":model["license_url"].format(repo=model["repo"], revision=model["revision"])})
    if args.audit:
        manifest["features"], packages = audit(args, manifest)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"packages":packages, "features":manifest["features"], "floors":manifest["floors"]}, indent=2) + "\n")
    if args.reports:
        audited, features = {}, None
        for path in sorted(args.reports.glob("*.json")):
            report = json.loads(path.read_text())
            for name, digest in report["packages"].items():
                if name in audited and audited[name] != digest:
                    raise ValueError("conflicting package audit reports")
                audited[name] = digest
            current = report["features"]
            if not isinstance(current, list) or any(not isinstance(f, str) for f in current):
                raise ValueError("invalid package features report")
            if features is not None and features != current:
                raise ValueError("cargo features differ across platforms")
            features = current
            for key in ("glibc_min", "glibcxx_min"):
                manifest["floors"][key] = max(manifest["floors"][key], report["floors"][key], key=lambda v: tuple(map(int, v.split("."))))
        for path in args.archives.glob("*.tar.zst"):
            if audited.get(path.name) != sha(path):
                raise ValueError("package has no matching native audit report: " + path.name)
        manifest["features"] = features
    if args.upload:
        if not args.reports:
            raise ValueError("native audit reports are required before draft upload")
        for target, flavours in manifest["flavours"].items():
            if target + "-cpu" not in manifest["components"].get("setup", {}):
                raise ValueError("missing setup package")
            for flavour in flavours:
                for component in ("cli", "app"):
                    if target + "-" + flavour not in manifest["components"].get(component, {}):
                        raise ValueError("missing declared component flavour")
                names = ["cuda"] if flavour == "cuda" else ["rocm-" + gfx for gfx in manifest["hip_targets"]] if flavour == "hip" else []
                for name in names:
                    if target + "-" + flavour not in manifest["runtimes"].get(name, {}):
                        raise ValueError("missing declared runtime flavour")
            raw = args.archives / ("larch-setup-" + target + (".exe" if "windows" in target else ""))
            if not raw.is_file():
                raise ValueError("missing bootstrap executable")
            binary_name = "larch-setup" + (".exe" if "windows" in target else "")
            if (binary_name, sha(raw), raw.stat().st_size) not in manifest["components"]["setup"][target + "-cpu"]["files"]:
                raise ValueError("bootstrap executable differs from audited setup package")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2) + "\n"
    if len(encoded.encode()) > 8 * 1024 * 1024:
        raise ValueError("manifest exceeds setup metadata limit")
    args.output.write_text(encoded)
    if args.upload:
        token = os.environ["RELEASES_WRITE"]
        release = open_draft(args.version, token)
        staged = {(a["name"], a["size"]) for a in release["assets"] if a["state"] == "uploaded"}
        upload(release, args.output, token)
        for path in sorted(args.archives.iterdir()):
            if path.is_file() and (path.name.endswith((".tar.zst", ".provenance.json")) or path.name.startswith(("larch-setup-", "larch-installer-"))) and (path.name, path.stat().st_size) not in staged:
                upload(release, path, token)
    return manifest


def main():
    """Dispatch local release preparation tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", default="stable")
    parser.add_argument("--archives", type=Path, default=ROOT / "dist")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/manifest.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", choices=TARGETS)
    parser.add_argument("--flavour", choices=("cpu", "cuda", "hip", "metal"), default="cpu")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--runtimes", choices=("cuda", "hip"))
    parser.add_argument("--cuda-tools", type=Path)
    parser.add_argument("--target", choices=TARGETS, default=TARGETS[0])
    parser.add_argument("--image")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--open-draft", action="store_true")
    parser.add_argument("--stage", type=Path, nargs="+")
    parser.add_argument("--unstage", action="store_true")
    args = parser.parse_args()
    if not valid_version(args.version) or not re.fullmatch(r"[A-Za-z0-9_-]+", args.channel):
        parser.error("invalid version or channel")
    if args.cuda_tools:
        redist(("cuda_cuobjdump",), args.target, args.cuda_tools)
    elif args.prepare:
        prepare(args.version)
    elif args.build:
        build(args)
    elif args.runtimes:
        runtimes(args)
    elif args.open_draft:
        open_draft(args.version, os.environ["RELEASES_WRITE"], fresh=True)
    elif args.stage:
        # Release assets, not Actions artifacts: one release's runtimes outgrow the artifact quota.
        release = open_draft(args.version, os.environ["RELEASES_WRITE"])
        for path in args.stage:
            upload(release, path, os.environ["RELEASES_WRITE"])
    elif args.unstage:
        args.archives.mkdir(parents=True, exist_ok=True)
        for asset in open_draft(args.version, os.environ["RELEASES_WRITE"])["assets"]:
            # A job not in this one's needs may still be uploading.
            if asset["state"] == "uploaded":
                download(asset, args.archives / Path(asset["name"]).name, os.environ["RELEASES_WRITE"])
    else:
        assemble(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, http.client.HTTPException, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        # Do not print authenticated requests, response bodies, tokens or command environments.
        print("release packaging failed: " + str(error).replace("\n", " ")[:1024], file=sys.stderr)
        sys.exit(1)
