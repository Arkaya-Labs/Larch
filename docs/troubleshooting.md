# Troubleshooting

Find the symptom below. If it isn't here, see [Reporting a bug](#reporting-a-bug).

## Larch doesn't find my GPU

Run `larch info`. It prints one line per device. A working GPU shows as `device 0:` with its name,
`(Cuda)` or `(Metal)`, the driver version and free memory. The `device cpu` line is always there.
If it's the only line, or the GPU line shows an error instead:

- **NVIDIA:** update the driver from NVIDIA's site, then restart. Larch needs driver 580.65 or newer
  (the R580 branch, the first with CUDA 13). `nvidia-smi` shows the installed version, and the
  error on the GPU line names the minimum.
- **Installed the CPU build by mistake:** run `larch-setup status`. If it says `cpu`, reinstall
  and pick the CUDA build, or run `larch-setup install --flavour cuda`. The builds are `cuda`,
  `metal`, `hip` and `cpu`.
- **Built from source:** the build picks CUDA only if `nvcc` was on the `PATH` while building. See
  [Building from source](build.md#build-configuration).
- **AMD:** the installer doesn't ship the AMD build yet. Build from source with `LARCH_BACKEND=HIP`.
- **Older NVIDIA cards** (GTX 10 and 16, RTX 20 series) are not supported.

## `libcublas.so.13` not found (Linux)

The CUDA build loads cuBLAS from your CUDA installation when it starts. Install the CUDA 13 runtime
libraries from your distribution or NVIDIA, or point `LD_LIBRARY_PATH` at the CUDA `lib64` folder.

## The model doesn't fit

The error says how much memory the model needs and how much is free. Try, in order:

1. Close other apps that use the GPU, including games, browsers with hardware acceleration, and
   another Larch.
2. Lower the context: `--set max_context=4096`.
3. Use a smaller quantization (Q4_K_M instead of Q8_0) or a smaller model. See
   [Hardware](hardware.md#what-fits).

Don't run two Larch processes on one GPU at the same time. Each one sizes its memory from what's
free when it starts, so two that start together can both claim the same memory.

## A laptop is much slower than expected

On a laptop with an NVIDIA GPU and an integrated one, Windows sometimes runs the desktop on the
NVIDIA GPU. That takes memory and power away from the model.

1. Open **NVIDIA Control Panel > Manage 3D settings** and set the preferred GPU to **Auto-select**.
2. In **Settings > System > Display > Graphics**, leave Windows apps on **Let Windows decide**.
3. Restart.

Also plug the laptop in. On battery, most laptops cap the GPU's power, and speed drops with it.

## The first answer is slow

The first message after loading includes reading the system prompt and any documents. Later
messages in the same chat reuse that work and start faster. Long documents are kept ready across
restarts when **Keep documents ready after restart** is on; see [Documents](documents.md).

## Windows or macOS warns about the app

The Windows and macOS builds on the release page are code-signed by Arkaya Labs, so neither system
should warn when you open them. If one does, the file didn't come from the release page or was
changed on the way. Don't run it. Delete it, download it again from the release page, and check it
against the release's signed list as described in [Install](install.md#check-a-download-yourself).

## The engine stopped with exit code 3

Exit code 3 means the engine hit a GPU fault, a hang or an internal error and shut down on purpose:
active requests got an error, the telemetry was saved, and the process exited. Run it under the
supervisor to bring it back automatically:

```sh
larch supervise -- larch serve -m model.gguf
```

The supervisor restarts it at once after an engine fault, and kill and restart takes under 2 s.
If the same fault keeps happening, report it with the log.

## The agent won't use tools, or uses the wrong one

- Check the folder is allowed: `--allow` on the command line, or **Tools** in the app.
- Try a larger model. Tool choice gets worse as models get smaller.
- `larch-agent bench` measures how often a model picks the right tool on your machine.

## A PDF can't be read

Some PDFs open in a viewer but fail in every text reader. Larch says so instead of guessing. If the
pages are scans, make sure OCR is available: it's built into macOS and Windows; on Linux install
`tesseract`.

## Reporting a bug

[Open a bug report](https://github.com/Arkaya-Labs/Larch/issues/new?template=bug_report.yml). The form
asks for:

- `larch --version` and `larch info --json`
- your OS, GPU and driver version
- the model file name, or its alias from [Models](models.md#the-tested-models)
- the exact command and what happened, with the full error

Remove anything private from logs and transcripts before you paste them. For security problems, use
the private route in [SECURITY.md](../SECURITY.md) instead.
