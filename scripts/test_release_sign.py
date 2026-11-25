"""Tests for scripts/release-sign: python3 -m unittest scripts/test_release_sign.py -v"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
# release-sign has no .py extension, so the default finders would not recognise it.
loader = importlib.machinery.SourceFileLoader("release_sign", str(HERE / "release-sign"))
rs = importlib.util.module_from_spec(importlib.util.spec_from_loader("release_sign", loader))
loader.exec_module(rs)
r = rs.r
V = "1.2.3"


def url(name):
    return f"https://github.com/{r.REPO}/releases/download/v{V}/{name}"


class FakeGitHub:
    """larch-releases as the release API serves it: the by-tag lookup never returns a draft."""

    def __init__(self):
        self.blobs, self.inventory, self.channels, self.ref, self.next = {}, {}, {}, None, 1
        self.release = {"id": 7, "tag_name": "v" + V, "draft": True, "prerelease": False, "assets": [],
                        "upload_url": "https://uploads.github.com/repos/x/releases/7/assets{?name}"}
        components = {"cli": {}, "setup": {}}
        for flavour in ("cpu", "cuda"):
            name = f"cli-x86_64-unknown-linux-gnu-{flavour}-{V}.tar.zst"
            components["cli"]["x86_64-unknown-linux-gnu-" + flavour] = self.archive(name, b"cli " + flavour.encode(), [])
        for target in r.TARGETS:
            exe = "larch-setup" + (".exe" if "windows" in target else "")
            body = b"setup " + target.encode()
            self.add("larch-setup-" + target + (".exe" if "windows" in target else ""), body)
            # Setup's package also carries the installer window, for the uninstall entry and launchers.
            files = [[exe, hashlib.sha256(body).hexdigest(), len(body)], ["larch-installer", "0" * 64, 1]]
            components["setup"][target + "-cpu"] = self.archive(f"setup-{target}-cpu-{V}.tar.zst", b"pack " + body, files)
        manifest = {"schema": 1, "version": V, "channel": "stable", "components": components, "runtimes": {},
                    "flavours": {"x86_64-unknown-linux-gnu": ["cpu", "cuda"]}, "gated": []}
        self.add("manifest.json", json.dumps(manifest).encode())

    def add(self, name, body, state="uploaded"):
        self.release["assets"] = [a for a in self.release["assets"] if a["name"] != name]
        self.release["assets"].append({"name": name, "id": self.next, "url": f"https://api.github.com/repos/{r.REPO}/releases/assets/{self.next}", "state": state, "size": len(body)})
        self.blobs[self.next] = body
        self.next += 1

    def archive(self, name, body, files):
        self.add(name, body)
        self.inventory[name] = files
        return {"url": url(name), "sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "files": files}

    def asset(self, name):
        return self.blobs[next(a["id"] for a in self.release["assets"] if a["name"] == name)]

    def api(self, path, token, body=None, method=None):
        if path.startswith("releases?"):
            return [self.release] if path.endswith("page=1") else []
        if path.startswith("releases/tags/"):
            if self.release["draft"]:
                raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return self.release
        if path.startswith("releases/assets/") and method == "DELETE":
            self.release["assets"] = [a for a in self.release["assets"] if str(a["id"]) != path.split("/")[-1]]
            return None
        if path == "releases/7" and method == "PATCH":
            self.release.update(body)
            return self.release
        if path == "git/ref/heads/channels":
            if not self.ref:
                raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
            return {"object": {"sha": self.ref}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": "base"}}
        if path == "git/trees":
            self.channels.update({e["path"]: e["content"] for e in body["tree"]})
            return {"sha": "tree"}
        if path == "git/commits":
            return {"sha": "commit" + str(len(self.channels))}
        if path in ("git/refs", "git/refs/heads/channels"):
            self.ref = body["sha"]
            return {}
        raise AssertionError("unexpected API call " + path)


class CandidateThenPublish(unittest.TestCase):
    def test_candidate_signs_every_built_flavour_then_publish_promotes_the_same_bytes_with_only_gated_ones(self):
        gh = FakeGitHub()
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"RELEASES_WRITE": "t"}), \
                mock.patch.object(r, "api", gh.api), \
                mock.patch.object(r, "upload", lambda release, path, token: gh.add(path.name, path.read_bytes())), \
                mock.patch.object(r, "inventory", lambda path: gh.inventory[path.name]), \
                mock.patch.object(rs, "download", lambda asset, path, token: path.write_bytes(gh.blobs[asset["id"]])), \
                mock.patch.object(rs, "compiled_keys", lambda: ["PRIMARY", "BACKUP"]), \
                mock.patch.object(rs, "sign", lambda path, key, comment, primary: pathlib.Path(str(path) + ".minisig").write_text(comment)):
            temp = pathlib.Path(temp)
            key = temp / "primary.key"
            key.write_text("secret")
            args = argparse.Namespace(version=V, channel="stable", gate_record=None, secret_key=key)

            rs.candidate(args)
            self.assertEqual((gh.release["draft"], gh.release["prerelease"]), (False, True))
            signed = json.loads(gh.asset("manifest-candidate.json"))
            self.assertEqual((signed["channel"], signed["gated"]), ("candidate", ["cpu", "cuda"]))
            self.assertEqual(gh.asset("manifest-candidate.json.minisig"), ("version=" + V + " channel=candidate").encode())
            pointer = json.loads(gh.channels["candidate.json"])
            self.assertEqual(pointer["manifest_url"], url("manifest-candidate.json"))
            self.assertEqual(pointer["manifest_sha256"], hashlib.sha256(gh.asset("manifest-candidate.json")).hexdigest())
            self.assertEqual(sorted(pointer["setup"]), sorted(r.TARGETS))
            self.assertEqual(json.loads(gh.asset("manifest.json"))["channel"], "stable")
            self.assertFalse(any(p.startswith("gates/") for p in gh.channels))

            confirmed = {"cpu-m8", "clean-windows", "clean-linux", "clean-macos"}
            record = temp / "gates.md"
            record.write_text("\n".join(["version=" + V, "channel=stable", ""] + [
                "- [" + ("x" if name in confirmed else " ") + "] " + name + ": " + label for name, label in rs.GATES.items()]) + "\n")
            args.gate_record = record
            rs.publish(args)
            self.assertEqual((gh.release["draft"], gh.release["prerelease"]), (False, False))
            final = json.loads(gh.asset("manifest.json"))
            self.assertEqual((final["channel"], final["gated"]), ("stable", ["cpu"]))
            pointer = json.loads(gh.channels["stable.json"])
            self.assertEqual(pointer["manifest_url"], url("manifest.json"))
            self.assertEqual(pointer["setup"], json.loads(gh.channels["candidate.json"])["setup"])
            self.assertEqual(gh.channels["gates/" + V + ".md"], record.read_text())
            # What a user reads and checks: install commands, the key, and a sum for every download.
            self.assertIn("install.sh | bash", gh.release["body"])
            self.assertIn("minisign -Vm SHA256SUMS -P PRIMARY && sha256sum -c SHA256SUMS", gh.release["body"])
            self.assertEqual(gh.asset("SHA256SUMS.minisig").decode(), gh.asset("manifest.json.minisig").decode())
            self.assertEqual(gh.asset("install.ps1"), (r.ROOT / "scripts/install.ps1").read_bytes())
            sums = dict(reversed(line.split("  ")) for line in gh.asset("SHA256SUMS").decode().splitlines())
            for name in ("manifest.json", "install.sh", "larch-setup-x86_64-pc-windows-msvc.exe", f"cli-x86_64-unknown-linux-gnu-cuda-{V}.tar.zst"):
                self.assertEqual(sums[name], hashlib.sha256(gh.asset(name)).hexdigest(), name)

            with self.assertRaisesRegex(ValueError, "draft or a candidate"):
                rs.publish(args)


@unittest.skipUnless(shutil.which("minisign"), "minisign is not installed")
class SignedSums(unittest.TestCase):
    def test_a_replaced_archive_and_its_sum_fail_the_signature_without_the_offline_key(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = pathlib.Path(temp)
            key, pub = temp / "k.key", temp / "k.pub"
            rs.subprocess.run(["minisign", "-G", "-W", "-p", pub, "-s", key], check=True, capture_output=True)
            primary = pub.read_text().splitlines()[-1]
            archive = temp / f"cli-x86_64-unknown-linux-gnu-cpu-{V}.tar.zst"
            archive.write_bytes(b"real")
            sums = rs.sums({"components": {}, "runtimes": {}}, {}, [archive], temp)
            rs.sign(sums, key, "version=" + V, primary)
            verify = ["minisign", "-Vm", sums, "-P", primary]
            self.assertEqual(rs.subprocess.run(verify, capture_output=True).returncode, 0)
            # Someone with asset write access but no key swaps the archive and rewrites its sum.
            archive.write_bytes(b"evil")
            sums.write_text(hashlib.sha256(b"evil").hexdigest() + "  " + archive.name + "\n")
            self.assertNotEqual(rs.subprocess.run(verify, capture_output=True).returncode, 0)


class DraftHandoff(unittest.TestCase):
    def test_jobs_hand_files_over_through_a_fresh_draft_and_never_a_published_release(self):
        gh = FakeGitHub()
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"RELEASES_WRITE": "t"}), \
                mock.patch.object(r, "api", gh.api), \
                mock.patch.object(r, "upload", lambda release, path, token: gh.add(path.name, path.read_bytes())), \
                mock.patch.object(r, "download", lambda asset, path, token: path.write_bytes(gh.blobs[asset["id"]])):
            temp = pathlib.Path(temp)

            def main(*argv):
                with mock.patch.object(sys, "argv", ["release-manifest.py", "--version", V, *argv]):
                    r.main()

            main("--open-draft")
            self.assertEqual(gh.release["assets"], [], "a re-tag must not inherit an earlier run's files")
            built = temp / ("cli-x86_64-unknown-linux-gnu-cpu-" + V + ".tar.zst")
            built.write_bytes(b"cli")
            main("--stage", str(built))
            gh.add("larch-installer-x86_64-unknown-linux-gnu", b"half", state="starter")
            main("--unstage", "--archives", str(temp / "dist"))
            self.assertEqual([p.name for p in (temp / "dist").iterdir()], [built.name], "a file still uploading is not handed over")
            self.assertEqual((temp / "dist" / built.name).read_bytes(), b"cli")

            gh.release["draft"] = False
            for argv in (["--open-draft"], ["--stage", str(built)], ["--unstage"]):
                with self.assertRaisesRegex(ValueError, "published release"):
                    main(*argv)
            self.assertEqual(len(gh.release["assets"]), 2)


class Unpacked(unittest.TestCase):
    def test_an_archive_whose_trailer_outgrows_the_pipe_buffer_still_reads(self):
        # tarfile stops at the end-of-archive block; what follows must be drained, or zstd sees a broken
        # pipe. A Windows pipe holds less than one tar record; this trailer outgrows a Linux pipe too.
        with tempfile.TemporaryDirectory() as temp:
            temp = pathlib.Path(temp)
            (temp / "a").write_bytes(b"x")
            tar = temp / "a.tar"
            with r.tarfile.open(tar, "w", format=r.tarfile.PAX_FORMAT) as archive:
                archive.add(temp / "a", arcname="a")
            with tar.open("ab") as out:
                out.write(bytes(1 << 18))
            r.run("zstd", "-q", tar, "-o", temp / "a.tar.zst")
            self.assertEqual([name for name, *_ in r.inventory(temp / "a.tar.zst")], ["a"])


class BlasLibraries(unittest.TestCase):
    def test_hipblaslt_is_optional_per_target_and_rocblas_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            lib = pathlib.Path(temp)
            for name in ("rocblas/library/TensileLibrary_gfx1030.dat", "rocblas/library/TensileLibrary_gfx1100.dat",
                         "rocblas/library/index.yaml", "hipblaslt/library/TensileLibrary_gfx1100.dat", "hipblaslt/library/index.yaml"):
                (lib / name).parent.mkdir(parents=True, exist_ok=True)
                (lib / name).write_bytes(b"x")
            self.assertEqual(sorted(str(dest) for _, dest in r.blas_libraries(lib, "gfx1030")),
                             ["rocblas/library/TensileLibrary_gfx1030.dat", "rocblas/library/index.yaml"])
            self.assertEqual(len(r.blas_libraries(lib, "gfx1100")), 4)
            with self.assertRaisesRegex(ValueError, "rocBLAS library files for gfx1102"):
                r.blas_libraries(lib, "gfx1102")


class Retried(unittest.TestCase):
    def test_server_errors_retry_and_client_errors_do_not(self):
        calls = []
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 500, "Internal Server Error", {}, None)
            return "ok"
        with mock.patch.object(r.time, "sleep"):
            self.assertEqual(r.retried(flaky), "ok")
            self.assertEqual(len(calls), 3)
            missing = mock.Mock(side_effect=urllib.error.HTTPError("u", 404, "Not Found", {}, None))
            with self.assertRaises(urllib.error.HTTPError):
                r.retried(missing)
            self.assertEqual(missing.call_count, 1)


if __name__ == "__main__":
    unittest.main()
