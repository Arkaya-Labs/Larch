# Install Larch

This page installs Larch from a release, keeps it up to date and removes it. To build it yourself,
see [Building from source](build.md).

Larch 0.1.0 is for Apple silicon Macs only. Windows and Linux builds are coming soon; until then,
the install commands below say so on those systems and stop without changing anything.

## Before you start

| | Windows | macOS | Linux |
|---|---|---|---|
| OS | Windows 11, 64-bit x86 | macOS 26 or later, Apple silicon | 64-bit x86 |
| GPU | NVIDIA RTX 30, 40 or 50 series, or none | Built in | NVIDIA RTX 30, 40 or 50 series, or none |
| Also needs | PowerShell 5.1 or later | `curl`, `python3` | `curl`, `python3` |

Without a supported GPU, the installer offers the CPU build, which works but is slow. Intel Macs and
32-bit systems are not supported. See [Hardware](hardware.md) for details per card.

The installer stops on Windows 10. NVIDIA cards need driver 580.65 or newer (the R580 branch, the
first with CUDA 13); the installer checks this and says so if yours is older.

## Install

### Windows

Open PowerShell (not as administrator) and run:

```powershell
irm https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.ps1 | iex
```

### macOS and Linux

```sh
curl -fsSL https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.sh | bash
```

The installer then:

1. Downloads `larch-setup` and checks its signature, then opens the Larch Setup window.
2. Asks whether to install the desktop app, the command line tools (`larch`, `larch-agent`), or both.
3. Detects your GPU and picks the matching build: CUDA, Metal or CPU.
4. Lists the models from the catalog that fit your GPU memory, with each model's licence. You can
   skip this and add a model later.
5. Shows the full plan and waits for you to confirm before it changes anything.

Everything installs for your user only. No administrator rights are needed.

To answer the same questions in the terminal instead, add `--terminal`. The installer also stays in
the terminal when there is no desktop to show a window on, such as over SSH, or when the window
cannot start, after saying why. To ask for the terminal yourself:

```sh
curl -fsSL https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.sh | bash -s -- --terminal
```

On Windows: `& ([scriptblock]::Create((irm https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.ps1))) --terminal`.

## Where things go

| | Windows | macOS | Linux |
|---|---|---|---|
| Programs and data | `%LOCALAPPDATA%\larch` | `~/Library/Application Support/larch` | `~/.local/share/larch` |
| Commands | `%USERPROFILE%\.local\bin` | `~/.local/bin` | `~/.local/bin` |

The installer adds the commands folder to your `PATH` and, if you chose the app, adds Larch to the
Start menu on Windows or your app launcher on Linux. Open a new terminal afterwards so the `PATH`
change takes effect.

## Check a download yourself

The installer already does this, but you can repeat it by hand. Each release has `SHA256SUMS` and
`SHA256SUMS.minisig`, signed with the Larch release key:

```sh
minisign -Vm SHA256SUMS -P RWQVpB8j31UnNTtZdjlYzi71ZGHAvhmaa5P0NrKn5OhW0oO0eZUQRO7s
sha256sum -c SHA256SUMS --ignore-missing
```

That is the Larch release key (minisign key ID `352755DF231FA415`). Check the signature first. A
list of hashes proves nothing on its own, because anyone who can replace a file can also replace its
line in an unsigned list.

Releases built after the repository became public also carry GitHub build provenance: a signed record
of the workflow, commit and run that built each file. With the [GitHub CLI](https://cli.github.com):

```sh
gh attestation verify larch-setup-x86_64-unknown-linux-gnu --repo Arkaya-Labs/Larch
```

It shows where a file was built; the minisign signature above is what the installer trusts.

## Update, roll back, remove

`larch-setup` manages the installation after the first run.

| Command | What it does |
|---|---|
| `larch-setup status` | Shows the installed version, build and components |
| `larch-setup update --check` | Says whether a newer release exists, without installing it |
| `larch-setup update` | Installs the newest release |
| `larch-setup rollback cli` | Goes back to the previous version of a component (`cli`, `app` or `setup`) |
| `larch-setup pin 0.1.0` | Stays on one version; `larch-setup unpin` undoes it |
| `larch-setup modify --add app` | Adds or removes a component later (`--add`, `--remove`) |
| `larch-setup repair` | Re-checks every installed file and replaces any that don't match |
| `larch-setup uninstall` | Removes Larch but keeps your chats, settings and models |
| `larch-setup uninstall --purge` | Removes those too |

Add `--dry-run` to any of them to see the plan without changing anything.

The app asks Larch Setup at launch and once a day whether a newer release is out. When there is
one, it says so in the sidebar and in **Settings > General**, and installs nothing until you click
**Restart to update**. Larch then quits, setup installs the update, and Larch opens again; an update
that fails is rolled back. A pinned install is never offered one. To stop the checks, turn off
**Settings > General > Check for updates**. The command line tools never check by themselves.

## Testing a release candidate

Testers can install the candidate channel. The files are public, but nothing installs them unless you
ask for it:

```sh
curl -fsSL https://github.com/Arkaya-Labs/Larch/releases/latest/download/install.sh | bash -s -- --channel candidate
```
