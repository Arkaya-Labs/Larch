# Releasing

This is the full sequence for one release, from tag to public install command. The builds run in
CI; the signing runs on a maintainer's machine with the offline key. The draft stays private until
`release-sign candidate`, which makes it a **public prerelease**: anyone can download the candidate
files from then on, though nothing installs them without `--channel candidate`. `release-sign
publish` then promotes the same files to the stable channel. Do not run `candidate` while a release
must stay secret.

The release repository is named in one place per file: `REPO` in `scripts/release-manifest.py` (which
`scripts/release-sign` imports) and `repo` at the top of `scripts/install.sh` and
`scripts/install.ps1`. A rename of the project or the organization changes those three lines and the
URLs in this file.

## One-time setup (maintainers with admin rights)

Do these once, before the first release. They are repository settings, so a PR cannot do them.

1. **Signing keys.** `crates/larch-setup/src/keys.rs` holds the public half of the offline primary
   and backup minisign keys. The signer needs the primary secret key on the signing machine. Keep both
   secret keys offline and never put them in CI. `release-sign` refuses to sign if a key in that file
   is a `TEST ONLY` key, or if the two keys are the same.
2. **Release repository.** This repository, `Arkaya-Labs/Larch`, holds the archives, manifests and
   the `channels` branch: each release is a draft release of the source repository itself, not of a
   separate releases repository. It must be public so the install scripts work without a token.
   `release.yml` stops at its first job in any other repository (a fork or a rename), because the
   workflow token can write only to the repository it runs in.
3. **Token for the signing machine.** CI needs no secret: every job stages its files on the draft
   with the workflow's own token. The signing steps below read a token from `RELEASES_WRITE`: a
   fine-grained token with contents read and write on this repository only, kept on the signing
   machine and never stored in the repository.
4. **`release` environment.** Settings > Environments > New environment `release`, with a required
   reviewer. The first job of `release.yml` uses it, so every release run waits for that approval.
5. **Runners.** Every release job runs on a GitHub-hosted runner, none of which has a GPU. The
   GPU tests, model parity and the Metal device check are the gates in step 4, run by hand on
   GPU machines.
6. **After the source repository goes public:** turn on private vulnerability reporting
   (Settings > Advanced Security) and restrict who can push `v*` tags (Settings > Rules > Rulesets).
   Protect `dev` and `main` with a ruleset: one approving review, and these required status checks,
   which are the job names in `ci.yml`: `check (ubuntu-24.04)`, `check (windows-2025)` and
   `check (macos-26)`. The backend workflow runs only when its paths change, and the macOS workflow
   skips its Rust steps when only CUDA or HIP kernels change, so they are not required checks.

## Release steps

Use `X.Y.Z` for the version. Run the local steps from a checkout of the tagged commit.

1. **Tag.** On the commit to ship (usually the tip of `dev`):

   ```sh
   git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z
   ```

2. **Approve and wait for the draft.** The tag starts `release.yml`. Approve it in the `release`
   environment. It builds every flavour, audits the archives and leaves a draft release on the release
   repository with an unsigned `manifest.json`. If a stage fails after a fix on `dev`, re-run from that
   stage with the workflow's `from` input instead of re-tagging.
3. **Sign the candidate.**

   ```sh
   export RELEASES_WRITE=...   # the signing machine's token from setup step 3
   scripts/release-sign candidate --version X.Y.Z --secret-key /path/to/primary.key
   ```

   This checks every archive against the manifest, signs it for the `candidate` channel, uploads the
   install scripts and turns the draft into a public prerelease. Only testers who ask for
   `--channel candidate` install it, but the files are public.
4. **Run the gates.** On each gate machine, install from the candidate channel:

   ```sh
   curl -fsSL https://github.com/Arkaya-Labs/Larch/releases/download/vX.Y.Z/install.sh | bash -s -- --channel candidate
   ```

   On Windows: `& ([scriptblock]::Create((irm https://github.com/Arkaya-Labs/Larch/releases/download/vX.Y.Z/install.ps1))) --channel candidate`.
   The gates are listed in `GATES` in `scripts/release-sign`. CI has no GPU, so these are the only
   GPU checks: each GPU parity gate runs the model on the GPU, and the Metal gate also needs
   `larch info` to name the Metal device. The CPU chat and the three
   clean-machine install, update and uninstall gates are required. A GPU flavour ships only if its
   parity gates and the update-with-the-app-open gate pass; otherwise it is left out of this release.
5. **Record the gates.**

   ```sh
   scripts/release-sign record --version X.Y.Z --gate-record gates/X.Y.Z.md
   ```

   It asks one yes or no question per gate and writes the answers.
6. **Publish.**

   ```sh
   scripts/release-sign publish --version X.Y.Z --gate-record gates/X.Y.Z.md --secret-key /path/to/primary.key
   ```

   This signs the manifest and `SHA256SUMS` for `stable`, uploads them with their signatures, the gate
   record and the install scripts, commits the channel pointer and the gate record to the `channels` branch, then makes the
   release public with install and verify notes.
7. **Check the public install.** On a clean machine, run the install command from the release notes.
   `raw.githubusercontent.com` can serve the old channel pointer for about five minutes after
   publishing; wait and retry before treating that as a failure.

## What a release contains

- One archive per component and flavour, the `larch-setup` bootstrap for each target, and the CUDA
  and HIP runtimes.
- `manifest.json` and `manifest.json.minisig`: the signed list of every file, hash and size. Setup
  checks this signature with the compiled-in key before it installs anything.
- `SHA256SUMS` and `SHA256SUMS.minisig`: the sums of the manifest, the install scripts, the setup
  bootstraps and every archive, signed with the same offline key.
- `install.sh` and `install.ps1`, so each release has a fixed install URL.
- The gate record, also kept at `gates/X.Y.Z.md` on the `channels` branch.

Users can check a download by hand with
`minisign -Vm SHA256SUMS -P <primary key> && sha256sum -c SHA256SUMS --ignore-missing`. Check the
signature first: the sums alone prove nothing, because anyone who can replace an asset can also
replace its line in an unsigned list.
