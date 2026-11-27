# Contributing to larch

larch is in early development, with no stable release or compatibility guarantee. The first public
phase prepares source access and GitHub hosted checks without promotion; product qualification and an
official release come later. Read the [README](README.md) and
[security policy](SECURITY.md) before contributing. Report vulnerabilities privately, not in an issue or PR.

## Fork and pull request workflow

Once the repository is public, the intended contribution path is an ordinary fork and pull request:

1. Fork `Arkaya-Labs/Larch` on GitHub and clone your fork.
2. Add `https://github.com/Arkaya-Labs/Larch.git` as the `upstream` remote and fetch `upstream/dev`.
3. Create a descriptive topic branch from `upstream/dev`, make a focused change and push it to your fork.
4. Open a pull request targeting `dev`. Describe the problem, resulting behavior and exact validation.
   Include relevant platform, backend and model alias for numerical or performance changes.
5. Respond to review and keep the pull request focused. A maintainer reviews it; one approving review
   is needed before merge.

For anything larger than a bug fix or a small improvement, such as a new feature, a new dependency or a
change to how something works, open an issue or ask in the [Arkaya Labs Discord](https://discord.gg/DxXXmEWbyy)
first, so we can agree on the approach before you build it.

## Getting a pull request reviewed

A pull request is ready for review when:

- it makes one change; a large one may be asked to split into smaller pull requests;
- the description lists the checks you ran and their results (see [Validation](#validation)), and
  the CI checks pass, or a blocked check is explained;
- you can explain every line of it.

When a pull request needs more work, a maintainer adds the `needs work` label, moves it back to draft
and says what is missing. Mark it ready for review again when that is done. A pull request that waits
on its author for 30 days may be closed; reopen it whenever you pick it up again. GitHub holds the CI
run of a first-time contributor until a maintainer approves it.

## Using AI tools

AI coding tools are welcome. A pull request made with one is reviewed like any other, and the same
bar applies: you understand the change, you ran the checks, and you can answer questions about it.

- Say how AI helped in the pull request description, in a line such as "Assisted by Claude Code: wrote
  the tests." Autocomplete, typo and formatting fixes need no note.
- We encourage an `Assisted-by: <tool>` trailer at the end of commit messages. Coding assistants that
  read [AGENTS.md](AGENTS.md) add it for you.
- Do not add `Co-Authored-By` lines naming an assistant, links to assistant sessions, or pasted prompts
  and chat logs.
- A person is responsible for each pull request, issue and comment, and answers questions about it.
  Do not run a bot that opens pull requests, issues or comments without a person in the loop. Tools
  the maintainers configure for this repository, such as Dependabot, are the exception.

Every pull request, a fork's included, runs the same checks on GitHub-hosted runners, with no secrets
and no labels to ask for. Describe a blocked check in your PR instead of requesting secrets or
weakening the check in your patch.

## Build prerequisites and current blockers

Follow the platform setup in [docs/build.md](docs/build.md). Rust **1.97** is pinned in
[rust-toolchain.toml](rust-toolchain.toml); the kernel build requires CMake **3.24** or later, a C++20
compiler and libclang for bindgen. Python **3.11** or later is needed by the benchmark/model tooling.
The repository has Windows and Linux CUDA build workflows and a macOS Metal workflow. Metal requires
macOS **26** and Xcode **26** with the Metal toolchain. HIP is compiled in the Linux workflow, but has
no hardware test coverage there. A build result alone does not establish support for every GPU or model.

## Validation

Use checks appropriate to the change. Do not report skipped, queued or unavailable checks as passing.
Documentation-only changes need a review of commands and local links; they do not need a GPU build.

For Rust changes, once dependencies and the selected backend are available, run:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked
```

For a CPU-only build on a POSIX shell, set `LARCH_BACKEND=CPU` before the build, clippy and test commands.
In PowerShell use `$env:LARCH_BACKEND = "CPU"`.

If you add or update a dependency, also run [cargo-deny](https://github.com/EmbarkStudios/cargo-deny)
with `cargo deny check`, at the root and in `apps/desktop`. [deny.toml](deny.toml) lists the licenses a
dependency may use, and the check also reports known vulnerabilities.

For kernel changes, run the CPU reference suite and the relevant backend suite on suitable hardware:

```sh
cmake -S kernels -B build/kernels -DLARCH_BACKEND=CPU -DLARCH_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/kernels --config RelWithDebInfo
ctest --test-dir build/kernels -C RelWithDebInfo --output-on-failure --no-tests=error
```

Use a separate build directory when selecting `CUDA`, `METAL` or `HIP` instead of `CPU`. Report the
backend, hardware and any skipped tests. Kernel/model changes also need `larch parity` against a
llama.cpp dump made from the **same GGUF**; tokenizer changes need `larch tokcheck`. The small parity
fixture is alias `m8` in [bench/models.toml](bench/models.toml). See [bench/README.md](bench/README.md) and
[tools/llama-logits](tools/llama-logits) for fixture and dump instructions. Do not relax numerical
tolerances or replace the independent reference to obtain a pass.

For hot-path changes, provide comparable before/after measurements and use
[scripts/bench-merge](scripts/bench-merge) as applicable. A loss greater than **3%** must be reverted
unless it is a correctness fix. Explain the fix and its cost when that exception applies.

CI runs on GitHub-hosted runners, which have no GPU: `ci.yml` checks the CPU backend on Linux,
Windows and macOS, `macos.yml` builds Metal, and `backend-builds.yml` compiles and inspects CUDA and
HIP with the tests on the CPU reference. GPU tests and parity run by hand before a release
([RELEASING.md](RELEASING.md)), so say in your PR which GPU checks you ran and on what hardware.

A pull request gets the fast checks: the build, lints and Rust tests on all three systems, with the
kernel tests (ctest) only when it changes `kernels/`, and no build or tests when it changes only docs.
The CUDA and HIP builds run when it changes `kernels/`, `crates/larch-sys/`, `crates/larch-kernels/`,
`.cargo/`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` or the backend workflows. For any other
change that can affect a GPU build or how the kernels are called, a maintainer adds the `ci:kernels`
label, which starts those builds. On a PR they build and link the shipped binaries but skip the
workspace tests, which `ci.yml` runs on the CPU reference. Every merge to `dev` that changes more
than docs runs everything.
[scripts/ci-local](scripts/ci-local) runs a local subset of the workflows and stops at the first failure;
it needs dependency access and is not a substitute for all CI or GPU checks.

## Change boundaries

Keep diffs small and reuse existing implementations. Preserve GGUF bounds checks, host-side shape and
dtype validation, error handling and durable transcript/journal behavior. `unsafe` belongs only in
`larch-sys` and `larch-kernels`, with a `// SAFETY:` justification for each block. Changes to
`crates/larch-records/**` or `kernels/include/larch.h` require human approval of the shared contract.
See [AGENTS.md](AGENTS.md) for the same invariants when using an automated coding assistant.

Keep model identifiers in [bench/models.toml](bench/models.toml); use aliases in code, commits and PRs.
Do not commit model weights, secrets or personal data, or put personal information into outbound
requests such as User-Agent headers. Remove sensitive content from logs and reproductions.

## Licensing

Contributions to first-party source are under [Apache-2.0](LICENSE). Preserve existing third-party
notices and add attribution for newly included code/assets in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
Model weights have separate terms; do not assume the source license covers them.
