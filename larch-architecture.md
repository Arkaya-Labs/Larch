# larch architecture

*v1.6, 17 Oct 2026.*

> strided and gait, the live tuner in sections 2.2, 2.3, 11 and 12, have since been dropped from Larch:
> the `tune` feature and its git dependencies are gone. Those sections record the design as it was.

This is larch's design record, written for the first build in October 2026. Code comments cite it by section number ("architecture 4.2"), so its numbering is kept; where the code has moved on since, the code and the pages in [docs/](docs/README.md) are current. The goal: the simplest design that meets every requirement of the original brief (R1 to R9, reliability matrix R1 to R13). Every module below lists its **inputs**, **outputs**, **owner thread** and **errors**. Requirements are written R1 to R9; reliability-matrix rows are written "row R1" to "row R13".

---

## 0. Decisions this rests on

| # | Decision | Source |
|---|---|---|
| D1 | One `larch` process per device. strided and gait are crates linked in, each on its own thread, reading one in-process ring. No IPC, no shared memory, no HTTP | Handoff 8f, 10 |
| D2 | **One recovery path** (replaces the 5 Oct 20:32 in-process design). Any engine-side fault (a sticky CUDA error, a Windows TDR, a Metal command-buffer error, a watchdog hang, a panic in the engine thread) fails the in-flight requests, flushes the ring and the gait journal, and exits the process with code 3. `larch supervise` restarts it and `larch chat` resumes its transcript. A library caller sees `Health::Failed` and `DeviceLost` until it restarts its own process. There is no in-process device reset: NVIDIA's runtime reference says that after a sticky error the process must be terminated and relaunched, and one path is simpler to build and test than three. The GPU-worker split stays deferred (section 18) | Maintainer decision |
| D3 | strided (the diagnosis rules) and gait (the tuner) live in their own repositories. larch pulls both in as git dependencies. `larch-records` (the contract) lives in larch | Maintainer decision |
| D4 | Code for the first build followed a written repository policy for coding assistants (now [AGENTS.md](AGENTS.md)) | Maintainer decision |
| D5 | gait is a deterministic tuner for the 3 days, reusing the orchestrator's contract, journal, verdict and stop-condition ideas. Repo modification comes later | Maintainer decision |
| D6 | Six rules are ported with all their tests: r01, r02, r03, r06, r08, r12, plus `_stats` and the engine core | Maintainer decision |
| D7 | Authority comes from a risk-class policy. Live, performance-only knobs auto-apply at any rule confidence. Resource knobs need `--allow-resource`. Quality knobs need a human | Maintainer decision |
| D8 | Development hardware: RTX 5070 Ti (sm_120), RTX 4060 (sm_89), RTX 3050 (sm_86) and an M5 Mac with 24 GB. The Linux GPU runs happen in WSL and CI builds native Linux. No cloud GPUs | Maintainer decision |
| D9 | Metal kernels, Mac integration and bench, and the strided rule port run as a separate workstream | Maintainer decision |
| D10 | No Mac stop-gap. candle is the only escape hatch | Maintainer decision |
| D11 | First build target: 8 Oct 2026. The brief's concessions and deferrals stand as written | Maintainer decision |
| D12 | Correctness oracle: llama.cpp logits on the same GGUF, teacher-forced. The reference is llama.cpp's cuBLAS CUDA run at a fixed micro-batch. The gate is top-1 ≥ 99 % on the positions whose reference top-1 leads its top-2 by more than a calibrated tie margin, plus a calibrated KL bound on mean KL over all positions. Raw top-1 is reported beside it. (Changed 7 Oct by the driver: llama.cpp's own backends flip top-1 at near-ties, so 99 % at every position is a gate the reference itself misses.) | Handoff 8g |
| D13 | llama.cpp's TTFT and TPOT are measured by `tools/llama-serve-bench`, an in-process driver on llama.cpp's C API that replays larch's exact workload (section 16). `llama-parallel` and `llama-server` are not used | Maintainer decision |
| D14 | larch is local-hosting first. A conversation keeps its KV between turns (explicit session reuse): a finished sequence is **parked** with its blocks and the token list of what was fed; the next turn re-tokenises its whole rendered prompt and reuses the longest common **token** prefix, bounded by the tokens actually in the KV (`n_kv`), prefilling only the rest; under block pressure parked sequences give up trailing blocks, least recently used first and only as many as needed, so they keep their longest prefix, and only then is a live sequence preempted; cached-versus-fresh tests prove it (9.3, 9.4, 10.1.1, 15, 16). Cross-session adoption, a parked cap or TTL and the extra knobs wait until that path is reliable (4.5, 18) | Maintainer decision |
| D15 | Two baselines, and every must-win row beats both: llama.cpp in-process at its best flags, and Ollama over its HTTP API with the same GGUF, matching context, `num_parallel` = N, flash attention on and `keep_alive` −1 (section 16) | Maintainer decision |
| D16 | Parity runs every prompt three ways on the larch side against one llama.cpp dump: all at once (prefill), one token per step (the production decode path: paged KV across steps, captured graphs, split-K attention, the mmvq decode kernels) and in chunks (chunked prefill with decode interleaved), plus eight prompts in flight and a graphs-off control (section 15) | Maintainer decision |

---

## 1. The system at a glance

```
                    larch.exe / larch  (one process per device)
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ caller threads (CLI chat REPL, bench clients, your app via the library)       │
 │   Engine::generate() ──Request──▶ intake channel                               │
 │   ◀── TokenStream (per-request channel: tokens, Done, Error)                    │
 │                                                                              │
 │ engine thread  (owns GPU, scheduler, KV, sampler; catch_unwind boundary)      │
 │   loop { apply pending knobs → schedule → executor.step → sample → emit →     │
 │          push EngineSample/RequestSpan into RING → guardrail/A-B bookkeeping } │
 │                                                                              │
 │ RING (larch-records): single producer (engine thread), many readers          │
 │    ├── strided thread : window → DiagnosisInput → 6 rules → Findings           │
 │    ├── gait thread    : Findings + ring → trial → KnobControl.set/revert/ab    │
 │    ├── CLI tok/s line, `larch bench` metrics                                  │
 │                                                                              │
 │ KnobControl (trait in larch-records, impl in larch-engine)                 │
 │   validate vs registry on caller thread → pending queue → engine applies at   │
 │   next step boundary → EffectiveConfig (arc-swap) + ActivationEvent in ring   │
 │                                                                              │
 │ sampler thread (1 Hz): NVML device sample → latest-slot → engine copies to ring│
 │ watchdog thread: step heartbeat; hang → exit 3, supervisor restarts           │
 └──────────────────────────────────────────────────────────────────────────────┘
          ▲ (optional) `larch supervise` parent: restart with backoff, Job Object / pipe-EOF
```

Kernels sit below the engine thread, behind `kernels/include/larch.h` (C ABI, opaque buffers). Exactly one GPU backend is compiled per build: CUDA (Windows, Linux), Metal (macOS) or HIP (Linux, CI compile-only). The CPU reference backend is always compiled; tests use it, and it is never used for serving.

---

## 2. Repositories, crates and dependencies

### 2.1 larch (the source repository)

```
Cargo.toml                 workspace; [patch] entries (2.4); lints
rust-toolchain.toml        1.97, rustfmt + clippy
AGENTS.md                  guidance for coding assistants
crates/
  larch-records/          THE contract: records, ring, knob schema, KnobControl trait, errors, JSON Schema export. No GPU, no unsafe
  larch-sys/              bindgen over larch.h; runs CMake (existing). unsafe allowed
  larch-kernels/          safe wrappers: Device, Stream, Buffer, Event, Graph, ops. unsafe allowed (SAFETY comments)
  larch-model/            GGUF, tokenizer, chat templates, model config, weights, forward graph, Model trait, candle hatch (feature)
  larch-engine/           intake, scheduler, KV manager, executor, sampler, knob registry + apply, guardrail, A/B, telemetry producer, fault recovery, fit
  larch/                  public library facade + `larch` binary (chat, bench, knob, parity, tokcheck, supervise, info); links strided + gait under feature `tune` (default on)
  cuprum/                  scaffold: router-to-worker message types over larch-records; `cuprum` binary prints version + schema
kernels/
  CMakeLists.txt           LARCH_BACKEND = AUTO|CUDA|HIP|METAL|CPU; CPU reference always built
  include/larch.h         the C ABI (section 5)
  src/common/              backend-neutral C++ (launch-shape math, quant block layouts, warp helpers .cuh)
  src/cuda/                *.cu, hipify-clean (section 6.3)
  src/metal/               *.metal + backend.mm (Objective-C++)
  src/cpu/                 plain-loop reference implementation of the whole ABI
  tests/                   one *.cpp per op family, GPU vs CPU reference
tools/llama-logits/        C++ dump tool against llama.cpp's C API (oracle, section 15)
tools/llama-serve-bench/   C++ driver on llama.cpp's C API: larch's serve workload at N concurrent, per-request TTFT/TPOT (D13, section 16)
bench/
  prompts/                 fixed prompt set (jsonl), committed day 1, never edited
  llama/                   scripts that run llama-bench, llama-batched-bench and llama-serve-bench
  results/history.csv      one line per merge to dev
fuzz/                      cargo-fuzz targets (gguf, tokenizer, knob requests)
.github/workflows/ci.yml   win / linux / mac / hip jobs (section 14)
```

R8 asks for few crates, so there are seven, and each one maps to one lane owner. Nothing else exists in the repo unless a lane needs it on these three days.

### 2.2 strided (its own repository): Rust port beside the Python

```
Cargo.toml                 workspace (new), members = ["crates/strided"]
rust-toolchain.toml        1.97, the same pin as larch
crates/strided/            the Rust strided (section 11). Depends on larch-records (git)
  tests/                   ported test files, one per Python test file
  tests/parity.rs          differential tests against the Python implementation
rules/ engine/ schema/ ... the Python package, untouched; it is the spec until the port replaces it
```

### 2.3 gait (its own repository)

```
Cargo.toml                 workspace (new), members = ["crates/gait"]
rust-toolchain.toml        1.97, the same pin as larch
crates/gait/               the Rust gait (section 12). Depends on larch-records (git) and strided (git)
src/agent_loop/ ...        the Python orchestrator, untouched; repo-modification job ports after the sprint
```

### 2.4 Dependency graph and how it builds

```
larch-records  ◀── larch-engine ◀── larch (bin) ──▶ strided (git) ──▶ larch-records (git)
      ▲                 ▲                    └──────▶ gait (git) ──▶ strided, larch-records (git)
      └── cuprum        └── larch-model ◀── larch-kernels ◀── larch-sys ◀── kernels/ (CMake)
```

- strided and gait depend on `larch-records` via `git = "https://github.com/<org>/<repo>", branch = "dev"`. The larch workspace adds
  `[patch."https://github.com/<org>/<repo>"] larch-records = { path = "crates/larch-records" }`
  so only one copy of the contract is ever compiled into the binary (otherwise you'd have two distinct `EngineSample` types).
- larch depends on `strided` and `gait` via `git = …, branch = "main"`, and `Cargo.lock` pins exact commits. Bumping a pin is an ordinary commit to `dev` (`cargo update -p strided`).
- Local cross-repo work: each developer keeps sibling checkouts of the three repositories and a **gitignored** `larch/.cargo/config.toml` with `[patch."<strided git URL>"] strided = { path = "../strided/crates/strided" }` (and the same for gait). Edits then show up without a push.
- Git access: `net.git-fetch-with-cli = true` in `.cargo/config.toml`, so cargo uses each machine's git credentials for the other repositories, and CI maps a read-only token per repository owner with `git config url.<…>.insteadOf`.
- Feature `tune` on the `larch` crate (default on) links strided and gait. `--no-default-features` builds a pure engine for kernel lanes that don't want the cross-repo fetch.

---

## 3. Threads and ownership

| Thread | Owns (only it mutates) | Reads | Writes | Panic / fault policy |
|---|---|---|---|---|
| **caller** (any) | its `TokenStream`, its `Session` history, `Detokenizer` | `EffectiveConfig` (arc-swap), `Health` | `Request` into the intake channel; cancel flag | caller's own |
| **engine** (`larch-engine`) | `Device`, streams, weights, KV pool, block tables, scheduler queues, sampler RNGs, `RingWriter`, `EffectiveConfig` publisher, A/B state, guardrail state | intake channel, pending-knob queue, device-sample slot | ring records, per-request token channels, `EffectiveConfig` swaps, `Health` swaps | `catch_unwind` per step, for logging and flushing only: a panic fails in-flight requests with `EngineError::Internal`, flushes the ring and exits the process with code 3 (section 9.11, D2) |
| **strided** | its ring cursor, window buffers, last `DiagnosisReport` | ring | `ArcSwap<DiagnosisReport>`, findings channel (bounded 64, drop-oldest) | `catch_unwind` around each tick; restart from a fresh cursor, log, count `strided_restarts` |
| **gait** | its ring cursor, trial state, journal file | ring, findings channel, `KnobControl` | knob requests via `KnobControl`; journal (JSONL) | `catch_unwind` around each loop iteration. On restart, the open trial is recovered from the journal (12.6): undone if the process is the same one, nothing to undo if it restarted, never run again |
| **sampler** (1 Hz) | NVML handle | none | `ArcSwap<DeviceSample>` latest slot; the engine copies it into the ring when its `seq` changes | errors make fields `Unsupported` / `Missing`; never panics the process |
| **watchdog** | none | engine heartbeat (`AtomicU64` step-complete time), `Health` | `Health` transitions, process exit on unrecoverable hang | section 9.12 |

Invariants (each one has a test):
1. Only the engine thread writes the ring. `RingWriter` is not `Clone` and is moved into the engine thread.
2. Only the engine thread mutates live configuration. Everyone else goes through `KnobControl`, which validates against the registry before anything is queued.
3. No reader can block the producer. A slow reader gets `Lagged { dropped }`.
4. strided and gait hold no reference into engine state. They see only records and `KnobControl`.

---

## 4. `larch-records`: the contract

No GPU code, no unsafe, no I/O. Dependencies: `bytemuck` (derive `Pod`), `arc-swap`, `serde` + `schemars` (JSON Schema only, behind feature `schema`). Every type here is the same type in larch, strided, gait and cuprum.

### 4.1 Rules for every record

- `#[repr(C)]`, `#[derive(Clone, Copy, Pod, Zeroable)]`, with explicit `_pad` fields and no implicit padding (bytemuck enforces this at compile time). No pointers, references, `String` or `Vec`.
- Strings are `FixedStr<N>` (`[u8; N]`, UTF-8, NUL-padded, with `as_str()` and `from_str_truncating()`).
- Enums are stored as raw `u8`/`u16` fields with typed accessors. Unknown values decode to an explicit `Unknown`, so a newer producer never makes an older reader panic.
- A value that can be absent is a measurement type (4.2), never a bare `Option`.
- Times: `t_mono_ns` is nanoseconds since the engine's monotonic epoch (`Instant` captured at `Engine::load`); `t_wall_ns` is Unix-epoch nanoseconds. Units sit in the field name (`_ns`, `_ms`, `_bytes`, `_mhz`, `_c`, `_w`, `_gbps`).
- Every record kind has a `const LAYOUT_VERSION: u16` and a compile-time `size_of` assertion. Changing a layout means bumping its version, which is a contract change that needs the maintainer's approval.

### 4.2 Measurement states

```rust
#[repr(u8)] pub enum MState { Missing = 0, Measured = 1, Estimated = 2, Unsupported = 3 }
// zeroed memory = Missing, so a forgotten field is honest by default

#[repr(C)] pub struct MF32 { pub v: f32, s: u8, _pad: [u8; 3] }   // 8 bytes
#[repr(C)] pub struct MF64 { pub v: f64, s: u8, _pad: [u8; 7] }   // 16 bytes
#[repr(C)] pub struct MU64 { pub v: u64, s: u8, _pad: [u8; 7] }   // 16 bytes
impl MF32 { fn measured(v) / estimated(v) / unsupported() / missing(); fn state(&self) -> MState; fn get(&self) -> Option<f32> /* Measured|Estimated */ }
```

`Unsupported` means this backend can't measure the value at all (e.g. SM occupancy on Metal). `Missing` means it should be measurable but no sample arrived. A rule abstains on both, and the report says which.

### 4.3 Record header and kinds

```rust
#[repr(C)] pub struct RecordHeader {        // 48 bytes
    pub kind: u16,          // RecordKind
    pub layout: u16,        // LAYOUT_VERSION of that kind
    pub tier: u8,           // 0 = T0 always-on, 1 = T1 sampled
    pub ab_arm: u8,         // 0 = no A/B, 1 = arm A, 2 = arm B
    pub device: i16,        // device ordinal, -1 = host/CPU
    pub seq: u64,           // ring sequence, stamped by the ring
    pub t_mono_ns: u64,
    pub t_wall_ns: i64,
    pub engine_id: u64,     // random per process start (identity scope)
    pub config_rev: u64,    // EffectiveConfig revision in force
}

#[repr(u16)] pub enum RecordKind {
    EngineSample = 1, RequestSpan = 2, DeviceSample = 3, DeploymentSnapshot = 4,
    HostSnapshot = 5, ConfigurationRevision = 6, ActivationEvent = 7, HealthEvent = 8,
}
pub enum Record { EngineSample(RecordHeader, EngineSample), RequestSpan(..), ... }  // decoded view, not stored
```

Slots are 256 bytes, so every body must be ≤ 208 bytes. A compile-time assert enforces that.

| Record | Producer and cadence | Tier | Body fields (types; producer → consumer) |
|---|---|---|---|
| **EngineSample** | engine, one per step | T1 (T0 keeps a 1-in-64 decimation) | `step u64`; `dur_ns u64` (plan start → outputs ready); `gpu_ns MU64` (events: Measured on CUDA/HIP/Metal); `host_ns u64` (scheduler + sampling CPU time); `phase u8` (0 decode, 1 prefill, 2 mixed); `graph_replayed u8`; `n_decode_seqs u32`; `n_prefill_seqs u32`; `n_prefill_tokens u32`; `n_running u32`; `n_waiting u32`; `n_preempted u32` (this step); `n_parked u32`; `parked_blocks u32` (D14: conversations and blocks held parked; included in `kv_blocks_used`); `parked_reclaimed u32` (D14: blocks taken from parked sequences this step); `kv_blocks_total u32`; `kv_blocks_used u32`; `kv_block_tokens u32`; `kv_tokens_used u32`; `kv_frag MF32` (Measured: 1 − used_tokens / (used_blocks × block_tokens)); `kv_util MF32` (Measured: used_blocks / total); `tot_preemptions u64`; `tot_finished u64`; `tot_prompt_tokens u64`; `tot_gen_tokens u64`; `tot_queue_ns u64`; `tot_queued u64`; `tot_rejected u64`; `achieved_gbps MF32` (Estimated: bytes moved by the step's weights + KV reads / gpu_ns); `hbm_util MF32` (Estimated: achieved / peak); `sm_occupancy MF32` (Estimated from the occupancy API for the dominant decode kernel on CUDA/HIP; Unsupported on Metal). Consumers: strided (r01, r03, r06, r08, r12 windows), gait (A/B, TPOT), bench, CLI tok/s line |
| **RequestSpan** | engine, on request end | T0 | `req_id u64`; `t_submit_ns`, `t_admit_ns`, `t_first_ns`, `t_done_ns u64`; `prompt_tokens u32`; `gen_tokens u32`; `cached_tokens u32` (D14: prompt tokens reused from a parked sequence); `session_id u64`; `preemptions u16`; `outcome u8` (Ok, Cancelled, Rejected, Error, DeviceLost); `finish u8` (Eos, Length, StopSeq, None); `error u16` (EngineError code). Consumers: bench (TTFT/TPOT/e2e), strided (TTFT/TPOT distributions for r02), gait (per-arm latency) |
| **DeviceSample** | sampler thread → engine, 1 Hz | T1 | `sm_clock_mhz`, `mem_clock_mhz`, `temp_c`, `power_w`, `gpu_util MF32`; `vram_used_bytes`, `vram_total_bytes MU64`. NVML on Windows/Linux CUDA; Unsupported on Metal and HIP for 3 days. Consumers: strided (r06 thermal leg), bench report |
| **DeploymentSnapshot** | engine, at load and reload | T0 | `model FixedStr<64>`; `arch FixedStr<16>`; `quant FixedStr<16>`; `params_b f32`; `n_layers u16`; `n_heads u16`; `n_kv_heads u16`; `head_dim u16`; `ctx_max u32`; `vocab u32`; `backend u8`; `device_name FixedStr<48>`; `arch_code u32` (sm 86/89/120, gfx 1030…, Apple family); `peak_gbps MF32`; `weights_bytes u64`; `kv_pool_blocks u32`; `fit_max_seqs u32`. Consumers: strided (model/gpu identity), bench, gait (bounds) |
| **HostSnapshot** | engine, at load | T0 | `os FixedStr<16>`; `os_version FixedStr<32>`; `cpu FixedStr<48>`; `ram_bytes u64`; `larch_version FixedStr<16>`; `git_sha FixedStr<12>`; `driver FixedStr<16>`; `runtime FixedStr<16>` (CUDA/Metal/HIP version) |
| **ConfigurationRevision** | engine, on every applied change | T0 | `rev u64`; `prev_rev u64`; `knob u16`; `cause u8` (Human, Gait, Guardrail, AbFlip, Inverse, Reload); `requested f64`; `effective f64`; `previous f64`; `step_applied u64`; `trial_id u64` |
| **ActivationEvent** | engine, on every request outcome | T0 | `knob u16`; `result u8` (Applied, Rejected, Indeterminate); `reason u16` (RejectReason / IndeterminateReason); `requested f64`; `effective f64`; `t_request_ns u64`; `t_applied_ns u64`; `step u64`; `trial_id u64` |
| **HealthEvent** | engine and watchdog | T0 | `from u8`; `to u8` (Loading, Ready, Degraded, Failed, Draining); `fault u16` (FaultCode: Panic, DeviceLost, LaunchFailure, Timeout, Oom, LoadError); `detail u32`; `recovery_ms u32` (on the first Ready after a supervised restart: the previous process's fault → Ready, from `LARCH_RESTARTED_AT`) |

`HostSnapshot`, `DeploymentSnapshot`, `EffectiveConfig` and `Health` are also exposed as `arc-swap` snapshots on the engine, so a reader that attaches late doesn't need to find them in the ring.

### 4.4 The ring

```rust
pub fn ring(capacity_pow2: usize) -> (RingWriter, RingHandle);   // default 65 536 slots × 256 B = 16 MB

pub struct RingWriter { .. }        // !Clone; engine thread only
impl RingWriter { pub fn push<R: RecordBody>(&mut self, h: RecordHeader, body: &R) -> u64 /* seq */ }

#[derive(Clone)] pub struct RingHandle { .. }   // Send + Sync
impl RingHandle {
    pub fn reader(&self, from: From /* Latest | Oldest | Seq(u64) */) -> RingReader;
    pub fn head(&self) -> u64;
}
pub struct RingReader { .. }        // one per consumer
impl RingReader {
    pub fn next(&mut self) -> Read;                       // Record(Record) | Empty | Lagged { dropped: u64 }
    pub fn drain(&mut self, out: &mut Vec<Record>, max: usize) -> DrainStats; // records, dropped
    pub fn dropped_total(&self) -> u64;
}
```

Implementation, about 150 lines with zero `unsafe`. Each slot is `stamp: AtomicU64` plus `words: [AtomicU64; 32]`. The record is converted to words with `bytemuck::cast` (a safe API).

- **Writer, seq `s`:** `stamp = 2s+1` (Relaxed), then a Release fence, store the words (Relaxed), `stamp = 2s+2` (Release), `head = s+1` (Release).
- **Reader, seq `s`:** load the stamp (Acquire).
  - `< 2s+2`: Empty.
  - `> 2s+2`: Lagged. Jump to `head − capacity + 1` and count the dropped records.
  - Otherwise: copy the words (Relaxed), Acquire fence, then re-check the stamp. If it changed, the read is Lagged.

Every access is atomic, so there is no data race and no UB. A `loom` test covers 1 writer and 2 readers. A proptest covers ordering and drop accounting. The producer never waits and never allocates.

### 4.5 Knob schema

```rust
#[repr(C)] pub struct KnobSpec {
    pub id: u16, pub ty: u8 /* Bool, U32, F32, Enum */, pub unit: u8 /* None, Tokens, Seqs, Requests, Ratio, Count, Ms */,
    pub activation: u8 /* RequestScoped, Live, ModelReload, ProcessRestart */,
    pub risk: u8       /* PerformanceOnly, ResourceOrAvailability, QualityAffecting */,
    pub inverse: u8    /* RestorePrevious, RestoreDefault */, pub backends: u8 /* bitmask CUDA|HIP|METAL|CPU */,
    pub name: FixedStr<32>,
    pub min: f64, pub max: f64, pub default: f64, pub step: f64,   // values travel as f64; Bool 0/1, Enum = index
    pub conflicts: [u16; 4], pub requires: [u16; 4],                 // 0 = none
    pub enum_names: [FixedStr<16>; 4],
}
pub static REGISTRY: &[KnobSpec];   // defined in larch-records so strided/gait/cuprum see the same table
```

Activation class and risk class are independent axes (issue #61). Knobs for the 3 days:

| id | name | type | range (default) | activation | risk | applied by |
|---|---|---|---|---|---|---|
| 1 | `max_batch_seqs` | U32 | 1 to 64 (16), further clamped by fit | Live | PerformanceOnly | scheduler admission into the running set |
| 2 | `prefill_chunk_tokens` | U32 | 64 to 8192, step 64 (512) | Live | PerformanceOnly | scheduler chunked-prefill budget per step |
| 3 | `schedule_policy` | Enum | decode_first, balanced, prefill_first (decode_first) | Live | PerformanceOnly | scheduler |
| 4 | `attn_decode_splits` | U32 | 0 = auto, 1 to 16 (0) | Live | PerformanceOnly | executor (flash-decoding split-K; invalidates captured graphs for the affected bucket) |
| 5 | `decode_graphs` | Bool | (1) | Live | PerformanceOnly | executor |
| 6 | `max_waiting` | U32 | 0 to 4096 (256) | Live | ResourceOrAvailability | intake admission |
| 7 | `kv_preempt_watermark` | F32 | 0.80 to 0.99 (0.95) | Live | ResourceOrAvailability | scheduler (preempt by recompute above it) |
| 8 | `guardrail_ratio` | F32 | 1.2 to 3.0 (1.5) | Live | ResourceOrAvailability | guardrail |
| 9 | `telemetry_tier` | Enum | t0, t1 (t1) | Live | PerformanceOnly | telemetry producer |
| 10 | `temperature_default` | F32 | 0 to 2 (0.7) | RequestScoped | QualityAffecting | sampler default |
| 11 | `top_p_default` | F32 | 0 to 1 (0.95) | RequestScoped | QualityAffecting | sampler default |
| 12 | `top_k_default` | U32 | 0 to 64 (40) | RequestScoped | QualityAffecting | sampler default |
| 13 | `repeat_penalty_default` | F32 | 1.0 to 2.0 (1.0) | RequestScoped | QualityAffecting | sampler default |
| 14 | `kv_pool_fraction` | F32 | 0.30 to 0.95 (0.85) | ModelReload | ResourceOrAvailability | fit + KV pool (drain, rebuild) |
| 15 | `max_context` | U32 | 512 to model max (8192) | ModelReload | ResourceOrAvailability | fit + KV pool |
| 16 | `device` | U32 | ordinal (0) | ProcessRestart | ResourceOrAvailability | reported only |

Conflicts and dependencies for the 3 days: `attn_decode_splits` requires the CUDA/HIP/METAL backend; `decode_graphs` is ignored (Rejected `Unsupported`) where the backend reports no graph support. Dynamic preconditions (for example `max_batch_seqs` ≤ the fit limit) are checked by the engine and come back as `Rejected { ExceedsFit }`.

Reserved for after the sprint, each with the condition it must meet before it enters the registry (D14 backlog, ids 17 to 21): `kv_parked_max_fraction` and `kv_parked_ttl_s` (a cap or a TTL on parked blocks; only with defaults that never evict a lone conversation and a written note that a strict cap can evict the only conversation; eviction under memory pressure is the sprint's only policy, 9.3); `prefix_match_min_tokens` (cross-session adoption; only with a policy that weighs the tokens a request would reuse against the cached tokens it would discard from another conversation); `prefill_gemm_min_rows` (range 1 to 16, the mmvq row contract of 5 and 6.1, or a validated fallback for larger values: every allowed value must be executable, not merely in range); `graph_bucket_max` (bounds graph memory on the 8 GB cards). Each lands with its KnobSpec, its test, a parity run at both bounds and its gait candidate.

### 4.6 The knob API

```rust
pub trait KnobControl: Send + Sync {
    fn registry(&self) -> &'static [KnobSpec];
    fn effective(&self) -> Arc<EffectiveConfig>;                  // arc-swap load, never blocks
    fn set(&self, req: KnobRequest) -> KnobReply;                 // blocks ≤ req.timeout_ms (default 250)
    fn revert(&self, target: RevertTarget, authority: Authority) -> KnobReply; // declared inverse (9.8)
    fn ab(&self, cmd: AbCommand) -> AbReply;
    fn health(&self) -> Health;
}
pub struct KnobRequest { pub knob: u16, pub value: f64, pub authority: Authority, pub allow: Allow /* bitflags: RESOURCE */, pub trial_id: u64, pub timeout_ms: u32 }
pub enum Authority { Human, Gait, Guardrail }
pub enum RevertTarget { Rev(u64), Trial(u64) }                // one revision, or everything done under a trial id
pub enum KnobReply {
    Applied { rev: u64, effective: f64, previous: f64, step: u64, latency_us: u32 },
    Rejected { reason: RejectReason },
    Indeterminate { reason: IndeterminateReason, rev: Option<u64> },
}
pub enum RejectReason { UnknownKnob, WrongType, OutOfRange { min: f64, max: f64 }, NotOnStep, Conflict { with: u16 },
                        MissingDependency { needs: u16 }, NotLive { class: ActivationClass }, Unauthorized { risk: RiskClass },
                        ExceedsFit { limit: f64 }, Unsupported, NotReady, Busy, Superseded { by_rev: u64 }, NothingToRevert }
pub enum IndeterminateReason { TimedOutBeforeStep, FaultDuringApply, EngineRestarted }
pub enum AbCommand { Start { b: [KnobDelta; 4], period_steps: u32, max_steps: u64, trial_id: u64, authority: Authority, allow: Allow }, Stop { trial_id: u64 } }
pub struct KnobDelta { pub knob: u16, pub value: f64 }       // knob 0 marks an unused slot (ids start at 1)
pub enum AbReply { Started { arm_a_rev: u64, step: u64 }, Stopped { steps: u64, end: AbEnd /* MaxSteps, Stop, Guardrail, Revert, Fault */ }, Rejected { reason: RejectReason } }
pub struct EffectiveConfig { pub rev: u64, pub values: [f64; 32], pub ab_arm: u8 }   // index by knob id
pub struct Health { pub state: HealthState, pub since_mono_ns: u64, pub last_fault: FaultCode, pub restarts: u32 }
```

Validation runs on the caller's thread, in order: known knob → type → `step` granularity → bounds → activation is Live or RequestScoped (else `NotLive`) → authority vs risk (`QualityAffecting` from `Gait` → `Unauthorized`; `ResourceOrAvailability` from `Gait` without `Allow::RESOURCE` → `Unauthorized`) → conflicts → dependencies → health is Ready. Only then is the request queued for the engine. The engine re-checks the dynamic preconditions at the step boundary. `ab(Start)` validates each delta the same way under the command's `authority` and `allow`, and then refuses a `QualityAffecting` delta as `Unauthorized` under every authority, `Human` included: gait tunes live performance knobs on user traffic, and quality and reload-class trials run only on calibration load (v2 plan decision 15, answered 17 Oct 2026), which has no A/B path yet. `ModelReload` knobs are changed through `Engine::reload(...)` (drain, rebuild, resume), not through `set`.

### 4.7 Errors shared across crates

`EngineError { PromptTooLong { tokens, limit }, Overloaded { waiting, limit }, NotReady(HealthState), Cancelled, DeviceLost, Internal(code), InvalidRequest(code) }`, each with a stable `u16` code that goes into `RequestSpan.error`. Load errors are a separate `LoadError { Io, BadGguf(code, offset), UnsupportedArch, UnsupportedQuant(ggml_type), DoesNotFit { need, have }, DriverTooOld { have, need }, Template(code), Backend(code) }`.

### 4.8 JSON Schema

`cargo run -p larch-records --features schema --bin records-schema` writes `larch-records.schema.json`, generated by `schemars` from serde mirror types derived from the same structs. cuprum and any future transport use it. It isn't committed; CI checks that it generates.

---

## 5. The C ABI: `kernels/include/larch.h`

Rules:
- Everything is prefixed `larch_`.
- Every call returns `larch_status`.
- Handles are opaque.
- Device memory is reached only through `larch_buffer*` + byte offset; raw device pointers never cross the ABI.
- All ops are asynchronous on a stream.
- `LARCH_ABI_VERSION` is checked by `Device::open` (`larch-kernels`, section 7) before any op runs.
- Each op's contract is the comment above it in `larch.h`: dtypes and shapes per tensor (the activation table of 6.1), the in-place rules, f32 accumulation, and what is `LARCH_ERR_INVALID_ARG`. `kernels/contract.cpp` holds every backend to it.

```c
#define LARCH_ABI_VERSION 2
typedef enum { LARCH_OK = 0, LARCH_ERR_INVALID_ARG, LARCH_ERR_OOM, LARCH_ERR_UNSUPPORTED,
               LARCH_ERR_LAUNCH, LARCH_ERR_DEVICE_LOST, LARCH_ERR_TIMEOUT, LARCH_ERR_DRIVER } larch_status;
typedef enum { LARCH_BACKEND_CPU = 0, LARCH_BACKEND_CUDA, LARCH_BACKEND_HIP, LARCH_BACKEND_METAL } larch_backend;
typedef enum { LARCH_F32 = 0, LARCH_F16, LARCH_BF16, LARCH_I32, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K,
               LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0 } larch_dtype;      // GGML block layouts, byte-identical

typedef struct larch_device larch_device;   typedef struct larch_stream larch_stream;
typedef struct larch_buffer larch_buffer;   typedef struct larch_event  larch_event;
typedef struct larch_graph  larch_graph;

typedef struct {
  char name[64]; larch_backend backend; uint32_t arch_code;      // 86/89/120, 1030/1100/1200, Apple family
  uint32_t compute_units; uint32_t warp_size; uint32_t max_threads_per_group;
  uint64_t shared_mem_per_group; uint64_t l2_bytes; uint64_t vram_total; uint64_t vram_free;
  uint64_t recommended_working_set;                               // Apple; = vram_free elsewhere
  float peak_gbps;                                                // from bus width × clock, or Apple table fallback
  uint32_t driver_version; uint32_t runtime_version; uint32_t flags; // LARCH_CAP_GRAPHS | _BF16 | _ASYNC_COPY | _UNIFIED
} larch_device_props;

typedef struct { larch_buffer* buf; uint64_t offset; larch_dtype dtype; uint32_t ndim;
                 int64_t shape[4]; int64_t stride[4]; } larch_tensor;   // strides in elements (blocks for quant)

uint32_t       larch_abi_version(void);
larch_status  larch_backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version);
larch_status  larch_device_count(int32_t* out);
larch_status  larch_device_open(int32_t ordinal, larch_device** out);   // -1 = CPU reference
larch_status  larch_device_props_get(const larch_device*, larch_device_props* out);
void           larch_device_close(larch_device*);
const char*    larch_last_error(const larch_device*); // thread-local detail string for logs

larch_status  larch_buffer_alloc(larch_device*, uint64_t bytes, larch_buffer** out);
larch_status  larch_buffer_wrap_host(larch_device*, void* host, uint64_t bytes, larch_buffer** out); // Metal: zero-copy mmap; others: UNSUPPORTED
void           larch_buffer_free(larch_buffer*);
larch_status  larch_upload(larch_stream*, larch_buffer*, uint64_t off, const void* src, uint64_t bytes);
larch_status  larch_download(larch_stream*, const larch_buffer*, uint64_t off, void* dst, uint64_t bytes);
larch_status  larch_stream_create(larch_device*, larch_stream** out);
larch_status  larch_stream_sync(larch_stream*);      // returns DEVICE_LOST / TIMEOUT on fault
void           larch_stream_destroy(larch_stream*);
larch_status  larch_event_record(larch_stream*, larch_event** out);
larch_status  larch_event_elapsed_ns(const larch_event* a, const larch_event* b, uint64_t* out);
void           larch_event_destroy(larch_event*);

/* graphs (CUDA graphs, HIP graphs, Metal indirect command buffers); UNSUPPORTED is a valid answer */
larch_status  larch_graph_begin(larch_stream*);
larch_status  larch_graph_end(larch_stream*, larch_graph** out);
larch_status  larch_graph_launch(larch_graph*, larch_stream*);
void           larch_graph_destroy(larch_graph*);

/* ops: shapes are [rows, cols] row-major unless noted; batch = rows */
larch_status larch_embed(larch_stream*, const larch_tensor* table, const larch_tensor* ids_i32, const larch_tensor* out_f32);
larch_status larch_rmsnorm(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps, const larch_tensor* out);
larch_status larch_rope(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32, const larch_rope_params* p); // in place; neox/normal, theta, scaling
larch_status larch_mmvq(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16, const larch_tensor* bias_or_null, const larch_tensor* out); // rows ≤ 16; decode GEMV
larch_status larch_dequant(larch_stream*, const larch_tensor* w_quant, const larch_tensor* out_f16);
larch_status larch_gemm(larch_stream*, const larch_tensor* a_f16, const larch_tensor* b_f16_t, const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta); // cuBLAS/hipBLAS/MPS or own
larch_status larch_silu_mul(larch_stream*, const larch_tensor* gate, const larch_tensor* up, const larch_tensor* out);
larch_status larch_add(larch_stream*, const larch_tensor* a_f32, const larch_tensor* b_f16, const larch_tensor* out);
larch_status larch_kv_write(larch_stream*, const larch_tensor* k, const larch_tensor* v, const larch_tensor* slots_i32, const larch_kv_pool* pool);
larch_status larch_attn_decode(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool, const larch_tensor* block_table_i32,
                                 const larch_tensor* seq_lens_i32, uint32_t splits /*0=auto*/, float scale, const larch_tensor* out);
larch_status larch_attn_prefill(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool, const larch_tensor* block_table_i32,
                                  const larch_tensor* q_start_pos_i32, const larch_tensor* q_lens_i32, float scale, const larch_tensor* out); // causal, chunk-aware
larch_status larch_qk_norm(larch_stream*, const larch_tensor* qk, const larch_tensor* w, float eps);   // Qwen3
larch_status larch_sample_prep(larch_stream*, const larch_tensor* logits, const larch_sample_params* per_row, const larch_tensor* penalty_ids_i32,
                                 const larch_tensor* out_topk_vals_f32 /*[rows,64]*/, const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32 /*[rows]*/);
```

`larch_kv_pool` is `{ larch_buffer* k; larch_buffer* v; uint32_t n_blocks, block_tokens, n_layers, n_kv_heads, head_dim; larch_dtype dtype; uint32_t layer; }`. `larch_rope_params` and `larch_sample_params` (temperature, unused because the host applies it to the 64 candidates; top-k cap; repeat penalty; window) are plain structs. Launch shapes are computed inside the backend from `larch_device_props` and the tensor shapes by pure functions in `src/common/launch.hpp`, which are unit-tested on CPU. Only `splits` (a live knob) is passed in.

---

## 6. Kernels and backends

### 6.1 Op inventory (inputs → outputs; used by)

| Op | Inputs | Output | Used for | Day |
|---|---|---|---|---|
| embed | quant/f16/f32 table, token ids | f32 `[rows, d]`, the residual stream | every step | 1 |
| rmsnorm | f32 x, f32 weight, eps | f16 | every layer, twice | 1 |
| rope | f16 q,k in place, positions | in place | every layer | 1 |
| qk_norm | f16 q,k, weight | in place | Qwen3 only | 1 |
| mmvq | Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q5_0 weight, f16 x `[≤16, k]` | f32/f16 `[rows, n]` | decode linear layers and prefill rows ≤ 16 (the bandwidth-bound hot path) | 1, tuned day 2 to ≥ 80 % peak GB/s on Q4_K, Q6_K and Q8_0 |
| dequant + gemm | quant weight → f16 scratch; f16 GEMM | f16 | rows > 16 (vendor GEMM) | 1 |
| generic quant kernels (an implementation choice, not an op) | Q4_0, Q5_0 and Q5_K weights, which mmvq and dequant accept like the others | the op's own output | rare tensors in Q4_K_M files: a backend may serve them with a slower generic kernel (dequantize, then an f16 GEMV) behind the same op; correctness over speed | 1 |
| silu_mul, add | f16 gate and up; f32 residual + f16 delta | f16; f32 residual | MLP, residual | 1 |
| kv_write | f16 k,v, slot ids | KV pool | every layer | 1 |
| attn_decode | q, paged KV, block table, seq lens, splits | f16 | decode rows; GQA-aware; split-K flash-decoding on day 2 | 1 (single pass), 2 (split-K) |
| attn_prefill | q chunk, paged KV, start pos | f16 | prefill/mixed rows, causal, chunk-aware | 1 |
| sample_prep | f32 logits, per-row params, penalty ids | top-64 vals/ids, log-sum-exp | sampling and logprobs without copying 128k-vocab rows | 1 |

Quant blocks match GGML byte for byte, so GGUF tensors upload as-is.

**Activation dtypes (decided 6 Oct 22:32, option 2.2a): f16 activations, an f32 residual stream, f16 KV.** The table below is the contract; it lives as comments in `larch.h` (the only file every backend compiles against), the CPU reference returns `LARCH_ERR_INVALID_ARG` for any other combination, and the GPU backends check dtypes and shapes on the host (metadata, never data) and return the same error. Every dot product, norm statistic and softmax accumulates in f32 on every backend.

| Op | Inputs | Output |
|---|---|---|
| embed | table as stored in the GGUF (quant, f16 or f32), ids i32 | h f32 `[rows, d]`, the residual stream |
| rmsnorm | x f32, w f32 (as stored), eps | f16, the input of the linear layers |
| rope | q, k f16 in place, pos i32 | in place |
| qk_norm | q, k f16 in place, w f32 | in place |
| mmvq | w quant (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q5_0), x f16 `[≤ 16, k]`, bias f32 or null | f16; f32 only for the lm_head |
| dequant | w quant | f16 scratch |
| gemm | a f16, b f16 (weight, transposed), bias f32 or null, alpha and beta f32 | f16; f32 only for the lm_head |
| silu_mul | gate f16, up f16 | f16 |
| add | h f32, delta f16 | h f32 (`out == a` allowed) |
| kv_write | k, v f16, slots i32 | pool f16 |
| attn_decode, attn_prefill | q f16, pool f16, tables i32, scale f32 | f16 |
| sample_prep | logits f32, per-row params, penalty ids i32 | vals f32, ids i32, lse f32 |

So the only f32 tensors on the hot path are the residual stream (one `[rows, d]` tensor per layer boundary) and the logits; everything a linear layer or attention reads or writes is f16, and the embedding and norm weights stay as the file stores them.

### 6.2 CUDA (Windows MSVC and Linux GCC/Clang), sm_86, sm_89, sm_120

- `LARCH_CUDA_ARCH` defaults to `86-real;89-real;120-real;120-virtual` (fat binary plus PTX). A single-arch override (`LARCH_CUDA_ARCH=120`) is kept for fast local builds.
- `cudart_static`. cuBLAS is dynamic (`cublas64_13.dll` / `libcublas.so.13`), found at runtime, and its absence is a clean `LoadError::Backend`.
- Hot paths use only fp16/bf16 math, warp shuffles and `cp.async` where sm_86 supports it. No FP8/FP4, no Blackwell-only instructions. Inline PTX is allowed only behind an arch guard on a CUDA-only path (6.3's HIP guard), never as the only path. Anything architecture-specific sits behind `props.arch_code` with the common path always present.
- Launch shapes derive from `compute_units`, `l2_bytes` and `peak_gbps` at startup. Per-arch constants live only in `launch.hpp` tables with a measured comment and fall back to the derived values.
- Graphs: one CUDA graph per decode batch bucket {1, 2, 4, 8, 16, 32, 64}, plus exact 9 to 32 on CUDA; other row counts pad to the next bucket. Graph memory is reserved for the buckets the startup `max_batch_seqs` reaches (12 for the default 16 seats on CUDA); a bucket past it, or one whose capture is refused, runs eagerly at its exact rows. Captured lazily on first use and replayed per step. The block table and sequence lengths are device buffers updated before replay, so graphs never need recapture for new requests. Recapture happens only when `attn_decode_splits` changes (one bucket at a time, at a step boundary, < 20 ms).
- Device lost: any sticky error (`cudaErrorIllegalAddress`, `LaunchFailure`, `LaunchTimeout`) returns `LARCH_ERR_DEVICE_LOST` and the engine exits (D2). Nothing is reset in-process: NVIDIA documents these errors as unrecoverable without relaunching the process, so there is no reset entry point in the ABI.

### 6.3 HIP (AMD RDNA 2/3/4), compile-only in CI for the 3 days

- One source tree. `src/cuda/*.cu` must pass `hipify-perl --print-stats` with zero unconverted calls; CI runs this on every merge.
- With `LARCH_BACKEND=HIP`, CMake hipifies into the build directory (generated files are never committed) and compiles with `hipcc --offload-arch=gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201`.
- Rules for the CUDA lane from hour 0:
  - Reductions go through `vv_warp_reduce_*` in `src/common/warp.cuh`, written against `warpSize` so they work in wave32 and wave64.
  - Shuffle masks go through a `VV_FULL_MASK` macro (64-bit on HIP).
  - No cooperative-groups grid sync, and no AMD matrix intrinsics (MFMA, rocWMMA) on any path.
  - NVIDIA tensor-core intrinsics (WMMA, `mma.sync`, `ldmatrix`) and `cp.async` are allowed only on CUDA-only paths behind a HIP guard (`__HIP_PLATFORM_AMD__`), with the common path always present, and `src/cuda` must still hipify with zero unconverted calls.
  - GEMM goes through `vv_blas_gemm_f16()` (cuBLAS ↔ hipBLAS).
  - Graphs go through `vv_graph_*` (CUDA graphs ↔ HIP graphs).
- Device telemetry on HIP is `Unsupported` for the 3 days.
- AMD's own HIP coverage, checked 5 Oct (table in section 14): ROCm 10.0 on Linux and the HIP SDK 7.2 on Windows support RDNA 3 and RDNA 4 consumer cards, and no consumer RDNA 2 card.
  - gfx1030 stays in the compile list. AMD still supports it on the Radeon PRO W6800 and V620, and the RX 6800/6900 use the same ISA.
  - RX 6000 cards are therefore the Vulkan fallback's job. The ABI doesn't preclude it, and it isn't built in the 3 days.

### 6.4 Metal (macOS, M5 / M5 Pro / M5 Max)

- `src/metal/*.metal` compiles at build time into one `larch.metallib` for the family (`xcrun metal` + `metallib`), which is embedded in the binary as a byte array. No per-chip variants.
- `src/metal/backend.mm` (Objective-C++, ARC) implements every `larch.h` function: one `MTLCommandQueue` per stream. Ops encode into the stream's current command buffer, and `larch_stream_sync` commits and waits.
- Weights: `larch_buffer_wrap_host` wraps the page-aligned mmapped GGUF with `newBufferWithBytesNoCopy`, so weights are zero-copy on unified memory. KV and scratch are `storageModePrivate`.
- Dispatch derives from `maxThreadsPerThreadgroup`, `threadExecutionWidth`, `recommendedMaxWorkingSetSize` and the GPU core count (from IORegistry `gpu-core-count`, falling back to a derived default). There is never a hard-coded "M5" table. Per-chip constants are allowed only with a measured win, and they fall back on unknown chips.
- GEMM has two paths. `Device::open` picks one per device by timing one f16 GEMM of a prefill shape on each; `LARCH_METAL_TENSOR=0|1` overrides.
  - The Metal 4 tensor API (`mpp::tensor_ops::matmul2d` from MetalPerformancePrimitives), which runs on the M5 GPU's per-core neural accelerators.
  - `simdgroup_matrix` kernels, which every Apple GPU has. MPS is only a day-1 bring-up stopgap.
- Settled 5 Oct: llama.cpp master already turns the tensor API on by default for M5-class GPUs in its prefill matmul (`ggml-metal-device.m` sets `has_tensor` for M5, M6, A19 and A20; `kernel_mul_mm` uses it). So the tensor path is required to win the 2k-prompt and TTFT rows, and it is not a hidden edge. larch's Mac edge has to come from batching, scheduling and attention.
- Floor: macOS 26 and Xcode 26, for Metal 4 and MetalPerformancePrimitives. Every M5 Mac ships with macOS 26.
- Graphs: `larch_graph_*` returns `UNSUPPORTED` on day 1. Indirect command buffers are a day-2 stretch for the Mac lane. The engine falls back to direct encoding transparently.
- Device lost: a command buffer error (`MTLCommandBufferErrorTimeout`, page fault, `Invalid`) maps to `LARCH_ERR_DEVICE_LOST` and the engine exits (D2), so every backend has the same one path even where the device would survive.
- `sm_occupancy` and device telemetry are `Unsupported`. `peak_gbps` comes from a small chip-family table with an `Estimated` state (Apple's figures: M5 153 GB/s; M5 Pro 307; M5 Max 460 with the 32-core GPU and 614 with the 40-core), and the bench reports achieved GB/s.

### 6.5 CPU reference (always built)

`src/cpu/*.cpp` implements every op as plain loops (f32 accumulate) and graphs as `UNSUPPORTED`. It is the oracle for per-kernel tests (`kernels/tests/*.cpp` run every op on the GPU backend and on CPU, compare with tolerances per dtype), and it lets GPU-less CI run the full engine on a tiny GGUF on all three OSes. It is never used for serving or benchmarks.

---

## 7. `larch-sys` and `larch-kernels`

**larch-sys** (exists today). Inputs: env `LARCH_BACKEND` (AUTO|CUDA|HIP|METAL|CPU), `LARCH_CUDA_ARCH`, `LARCH_HIP_ARCH`, `LARCH_CMAKE_GENERATOR`, `CUDAHOSTCXX`. Output: `bindings.rs` and link directives.

Changes:
- Replace `LARCH_CUDA` with `LARCH_BACKEND`, keeping `LARCH_CUDA=ON|OFF` as an alias for one week.
- Default the arch list per 6.2.
- Link cuBLAS dynamically on CUDA, `Metal.framework` + `Foundation.framework` + `MetalPerformanceShaders.framework` on macOS, and `hipblas`/`amdhip64` on HIP.
- Export `BACKEND: &str` (exists).

**larch-kernels**: the safe API that everything above uses. It is the only crate besides larch-sys that may use `unsafe`, and every block carries a `// SAFETY:` comment.

| Type | Inputs | Outputs | Notes |
|---|---|---|---|
| `Device::open(ordinal) -> Result<Device, KError>` | ordinal (−1 CPU) | owned device + `DeviceProps` | checks the ABI version and the driver minimum: the R580 branch for CUDA 13.x (Linux ≥ 580.65.06; NVIDIA names only the branch for Windows). The error names the minimum |
| `Buffer::alloc(&Device, bytes)` / `Buffer::wrap_host(&Device, &Mmap, range)` | sizes | `Buffer` (RAII free) | never exposes a pointer |
| `Stream::{new, sync}`; `Event::{record, elapsed}` | | | `sync` maps DEVICE_LOST/TIMEOUT to `KError::DeviceLost` |
| `Tensor<'a> { buf: &Buffer, off, dtype, shape, stride }` | | | shape/dtype validated in Rust before FFI |
| `ops::*` | `&Stream`, `Tensor`s, params | `Result<(), KError>` | one function per C op |
| `Graph::{capture(&Stream, f), launch}` | closure that issues ops | `Option<Graph>` (None if unsupported) | |

`KError { InvalidArg, Oom, Unsupported, Launch, DeviceLost, Timeout, Driver { have, need }, Stale }`.

---

## 8. `larch-model`

| Module | Inputs | Outputs | Errors |
|---|---|---|---|
| `gguf` | file path | `GgufFile { mmap: Mmap, version, kv: BTreeMap<String, GgufValue>, tensors: Vec<TensorInfo { name, ggml_type, dims, offset, bytes }> }` | `LoadError::BadGguf(code, offset)`: every length, count, offset and dims product is bounds-checked against the file size before use (fuzz target `gguf`) |
| `config` | `&GgufFile` | `ModelConfig { arch: Llama \| Qwen2 \| Qwen3 \| Mistral, n_layers, d_model, n_heads, n_kv_heads, head_dim, ffn, vocab, ctx_train, rope: RopeCfg, rms_eps, qkv_bias: bool, qk_norm: bool, tie_embeddings: bool }` | `UnsupportedArch` |
| `tokenizer` | `&GgufFile` (`tokenizer.ggml.*`) | `Tokenizer::{encode(&str, add_special) -> Vec<u32>, decode_piece(u32) -> &[u8], bos, eos, eot set}`; `Detokenizer` (streaming, holds back partial UTF-8 and possible stop-string prefixes) | checked against llama.cpp's tokenizer on the prompt set (oracle step 1) |
| `chat` | `tokenizer.chat_template` (Jinja) + `messages: &[Message { role, content }]` | rendered prompt string + extra stop tokens | `minijinja`; unknown template → `LoadError::Template` with a built-in ChatML fallback flag |
| `weights` | `&GgufFile`, `&Device`, `ModelConfig` | `DeviceWeights { per_layer: Vec<LayerWeights { attn_norm, wq, wk, wv, bq?, bk?, bv?, q_norm?, k_norm?, wo, ffn_norm, w_gate, w_up, w_down }>, tok_embd, out_norm, output }`, each a `Tensor` into one device buffer (or zero-copy wrap on Metal) | `DoesNotFit`, `UnsupportedQuant(ty)` |
| `graph` | `ModelConfig`, `DeviceWeights`, `Scratch` | `impl Model for Transformer` | |
| `candle` (feature `candle-hatch`) | GGUF path | `impl Model for CandleModel` | used only if the hour-10 gate fails for a backend |
| `vision` | an mmproj GGUF (InternVL's projector), image file bytes | `Image` (sha256, decoded as stb does), `Pixels` (up to 6 tiles and an overview), `Vision::encode_tile` → f32 rows a request's prompt carries as embeddings (`Model::forward_embeds`) | `LoadError::BadGguf`/`UnsupportedArch`; an image past the byte and pixel limits is refused from its header |

```rust
pub trait Model: Send {
    fn config(&self) -> &ModelConfig;
    /// One forward pass for a mixed batch. Writes K/V for all rows into the pool, writes logits for `rows_needing_logits` only.
    fn forward(&mut self, s: &Stream, batch: &StepBatch, kv: &KvPoolView, out: &mut LogitsView) -> Result<(), KError>;
    fn capture_decode(&mut self, s: &Stream, bucket: u32) -> Result<Option<Graph>, KError>;   // None = backend has no graphs
}
pub struct StepBatch<'a> {                 // built by the executor from the scheduler's StepPlan; all device buffers
    pub token_ids: Tensor<'a>,             // i32 [rows]
    pub positions: Tensor<'a>,             // i32 [rows]
    pub slot_ids: Tensor<'a>,              // i32 [rows] KV slot per token
    pub decode_rows: u32, pub prefill_rows: u32,
    pub block_table: Tensor<'a>,           // i32 [seqs, max_blocks]
    pub seq_lens: Tensor<'a>,              // i32 [seqs]
    pub prefill_q_start: Tensor<'a>, pub prefill_q_len: Tensor<'a>,
    pub logits_rows: Tensor<'a>,           // i32 rows whose logits are needed (last token of each seq)
    pub attn_splits: u32,
}
```

Forward per layer: rmsnorm → q/k/v linear (mmvq if rows ≤ 16, else dequant+gemm; bias on Qwen2) → qk_norm (Qwen3) → rope → kv_write → attn_decode on the decode rows and attn_prefill on the prefill rows → o linear → add → rmsnorm → gate/up linear → silu_mul → down linear → add. After the last layer: final norm → output linear on `logits_rows` only.

R4 model coverage: Llama 3.x, Qwen 2.5 (qkv bias), Qwen 3 dense (qk-norm), Mistral (llama arch). Quants: Q4_K_M (Q4_K + Q6_K plus occasional Q5_0/Q8_0 tensors) and Q8_0.

---

## 9. `larch-engine`

### 9.1 Public surface (used by the `larch` crate)

```rust
pub struct EngineOptions { pub model: PathBuf, pub device: i32, pub max_context: u32, pub kv_pool_fraction: f32,
                           pub ring_capacity: usize, pub knob_overrides: Vec<(u16, f64)>, pub telemetry: TelemetryTier,
                           pub on_fault: OnFault /* Exit (CLI default) | Fail (library default); D2 */ }
pub fn start(opts: EngineOptions) -> Result<EngineHandle, LoadError>;  // spawns engine, sampler, watchdog threads; returns when Ready or failed
pub struct EngineHandle {                                               // Clone + Send + Sync
    pub fn submit(&self, req: Request) -> Result<RequestHandle, EngineError>;   // admission happens here
    pub fn ring(&self) -> RingHandle;
    pub fn knobs(&self) -> Arc<dyn KnobControl>;
    pub fn health(&self) -> Health;
    pub fn deployment(&self) -> Arc<DeploymentSnapshot>;
    pub fn reload(&self, change: ReloadChange) -> Result<(), EngineError>;     // ModelReload knobs, model swap: drain → rebuild → resume; the same file and layout carry parked sequences, else they drop
    pub fn release_session(&self, session: u64);                              // frees that session's parked sequence; sticky (10.1.1, D14)
    pub fn parked_tokens(&self, session: u64) -> u32;                         // n_kv of that session's parked sequence, 0 if none (D14)
    pub fn shutdown(self);
}
pub struct Request { pub prompt: Vec<u32>, pub sampling: SamplingParams, pub max_tokens: u32, pub stop_tokens: Vec<u32>,
                     pub logprobs: u8 /* ≤ 20 */, pub priority: u8, pub ignore_eos: bool /* set only by `larch bench` */,
                     pub session: u64 /* 0 = none */, pub keep_kv: bool /* D14: park the sequence at Done */ }
pub struct RequestHandle { pub id: u64, pub events: Receiver<EngineEvent>, cancel: Arc<AtomicBool> }  // drop = cancel
pub enum EngineEvent { Token { id: u32, logprob: f32, top: SmallVec<[(u32, f32); 4]> }, Done { finish: Finish, usage: Usage, timings: Timings }, Error(EngineError) }
```

### 9.2 Intake and admission (caller thread)

- **Inputs:** `Request`, `EffectiveConfig`, health, `waiting` counter.
- **Checks, in order:**
  1. Health is Ready, else `NotReady`.
  2. `prompt.len() + max_tokens ≤ max_context`, else `PromptTooLong` (row R6). The same check runs against the KV pool: a prompt that could never fit is rejected rather than queued.
  3. `waiting < max_waiting`, else `Overloaded` (row R7).
  4. A request with images (`Request::images`): each span inside the prompt, in order, as long as its image's rows, its ids the image's, its pixels cut for the loaded projector; else `InvalidRequest(IMAGE)`, or `NO_VISION` with no projector. Image rows count as prompt tokens. The scheduler then encodes one tile a step across requests, through a small tile cache, and a step with image rows runs eager.
- **Output:** the request goes into a bounded MPSC channel (crossbeam) to the engine, and the caller gets a `RequestHandle`.
- Rejections are recorded as `RequestSpan` records with outcome `Rejected` (the engine writes them when it drains the reject queue, because only the engine writes the ring).

### 9.3 Scheduler (engine thread)

- **Inputs:** newly admitted requests, the running set, the parked set (D14), `KvManager` free blocks, `EffectiveConfig` (`max_batch_seqs`, `prefill_chunk_tokens`, `schedule_policy`, `kv_preempt_watermark`), cancel flags.
- **Output:** `StepPlan { decode: Vec<SeqId>, prefill: Vec<(SeqId, start, len)>, preempt: Vec<SeqId>, reclaim: Vec<(SeqId, n_blocks)>, finished: Vec<SeqId> }`.
- **Algorithm per step:**
  1. Drop cancelled sequences and free their blocks (row R3); a cancelled sequence is never parked. A finished sequence with `keep_kv` and a session is **parked** instead (D14): its blocks and its token list (the tokens that were fed, so `tokens.len() == n_kv`) stay and its state becomes `Parked`; `park` replaces an older sequence parked under the same session (freed) and frees instead under a released session id (10.1.1); without `keep_kv` or without a session it is freed.
  2. Match each waiting request that carries a `session` to the sequence parked under it **when the request is first scheduled to run, not at admission** (decided 6 Oct 22:30: matching at admission let waiting requests pin parked blocks that reclaim could not take, which deadlocked under pressure; until a request runs, its parked sequence stays reclaimable by step 3). Explicit session reuse only; cross-session adoption is post-sprint (18). Without a session, or with nothing parked, a fresh sequence is created. The prefix is compared token id by token id over the request's whole prompt (the facade re-tokenises the whole rendered history each turn, 10.1.1). On a match `reuse = min(lcp, seq.n_kv, prompt.len() − 1)`: at least one token is computed so logits exist, and nothing is claimed cached whose KV entries do not exist (the last sampled token is never fed, whatever the stop reason, so it is not in the KV; `n_kv` is the only authority, and it is read at the moment of the match, so a sequence trimmed by reclaim while the request waited simply yields fewer cached tokens). `KvManager::truncate(seq, reuse)` frees the blocks beyond `ceil(reuse / 16)` and sets `n_kv = reuse`; the token list becomes the new prompt, and the prefill starts at `reuse`. Positions are absolute, so nothing is shifted. `RequestSpan.cached_tokens = reuse`, set at first run.
  3. Reclaim when KV utilisation exceeds the watermark or a decode can't get a block: take trailing blocks from parked sequences first, least recently used first and only as many as needed (`KvManager::truncate` again), so a parked conversation keeps its longest prefix: an 8k-token conversation loses 16 blocks to a 256-token request, not 8k tokens of work. A parked sequence left with no blocks is dropped. Only then preempt running sequences by recompute, newest first: the sequence returns to the waiting queue at its prompt plus generated tokens, and `tot_preemptions` increments. There is no parked cap and no TTL in the three days: a parked conversation is never touched while nothing else needs the memory, which is the local-chat case, and under pressure it loses only what the live request needs. `EngineSample.parked_reclaimed` counts the blocks taken, so the cost of the policy is measured.
  4. All running decode sequences go in, up to `max_batch_seqs`.
  5. Fill the prefill token budget (`prefill_chunk_tokens`, minus decode rows under `balanced`) from the head of the waiting queue, chunking long prompts across steps. Each chunk allocates its blocks first; if blocks are short, it waits.
  6. `prefill_first` gives the whole budget to prefill before decode; `decode_first` never lets a prefill chunk delay decode beyond the budget.
- **Invariants (proptests):** a sequence's blocks are never freed twice; the sum of allocated blocks (running plus parked) equals the KV manager's used count; no plan exceeds `max_batch_seqs` or the token budget; every admitted request eventually ends with Done or Error. The six D14 invariants: parked blocks are inside that sum; a matched sequence's `reuse` never exceeds the common prefix, its `n_kv` or `prompt.len() − 1`; a parked sequence always has `tokens.len() == n_kv ≤ 16 × blocks.len()`, a partial last block allowed; reclaim takes only trailing blocks, never more than needed, and leaves `n_kv = min(n_kv, 16 × blocks.len())`; at most one sequence is parked per session, and a park under a released id frees; `release_session` frees exactly that session's blocks.

### 9.4 KV manager (engine thread)

- **Inputs:** a `FitPlan` (9.10) giving `n_blocks`, `block_tokens` = 16, layers, kv heads, head dim and dtype f16; alloc, free, truncate (to a token count: matching and reclaim), park, drop and release requests from the scheduler (D14).
- **Outputs:**
  - a `KvPool` (two device buffers) and a host-side block table per sequence, mirrored to a device `block_table` buffer that is updated incrementally before each step;
  - stats `used_blocks` (running plus parked), `used_tokens`, `frag`, `n_parked`, `parked_blocks`, `parked_reclaimed` for `EngineSample`.
- It is a free list of block ids with O(1) alloc/free plus an LRU list of parked sequences and the released-id ring (D14); `truncate(seq, n_tokens)` is the one primitive behind matching and reclaim. Blocks are the unit the later cache work builds on (18); nothing above depends on it.

### 9.5 Executor (engine thread)

- **Inputs:** `StepPlan`, `EffectiveConfig` (`attn_decode_splits`, `decode_graphs`), the per-sequence token and position state.
- **Outputs:** `StepOutput { per_seq: Vec<(SeqId, SampledToken)>, gpu_ns: MU64, bytes_moved: u64 }`, or `Err(KError::DeviceLost | Timeout)`.
- **Steps:**
  1. Pack the host staging buffers (token ids, positions, slot ids, block-table deltas, seq lens) into one pinned upload.
  2. If the step is pure decode, graphs are on and the backend supports them: replay the bucket graph after padding to the bucket size. Otherwise call `model.forward` directly.
  3. `sample_prep` on the logits rows.
  4. Download the top-64 values and ids plus the log-sum-exp (≈ 520 B per row).
  5. Finish sampling on the CPU (9.6).
  6. Wrap the step in events for `gpu_ns`.

  There is exactly one host sync per step, the download in step 4.
- Prefill rows ≤ 16 use mmvq (its contract in 5 and 6.1, as 8 says); more use dequant+gemm. Dequant scratch is per-layer and reused. The crossover is fixed in the three days; the knob that would move it is in the backlog (4.5) with its range limited to what mmvq can execute.

### 9.6 Sampler (engine thread)

- **Inputs:** the top-64 candidates and LSE per row, `SamplingParams { temperature, top_p, top_k, repeat_penalty, seed }` (request values, or the RequestScoped knob defaults), and the per-request RNG (ChaCha8 seeded from `seed` or the request id).
- **Output:** `SampledToken { id, logprob, top: up to 20 }`.
- Greedy is argmax and exact. Temperature, top-k and top-p are applied over the 64 candidates. The repetition penalty is applied on the GPU in `sample_prep` before top-k. Logprobs are normalised by the full-vocabulary LSE, so they're exact for the returned tokens.

### 9.7 Streaming and stops

- The engine sends `EngineEvent::Token` per sampled token.
- The engine stops a sequence on EOS/EOT tokens, `stop_tokens`, or `max_tokens`. A stopped sequence with `keep_kv` and a session is parked (9.3 step 1). Whatever the stop reason, the last sampled token is never fed, so it has no KV entry: the parked token list ends with the last token that was fed and `n_kv` says how many that is. With `ignore_eos` (set only by `larch bench`) it skips EOS/EOT, so every request generates exactly its target length on both engines (section 16).
- String stop sequences are matched on the caller side by the `larch` crate's `TextStream` (Detokenizer plus a holdback buffer). On a match, it cancels the request and emits `Done { finish: StopSeq }`.

### 9.8 Knob registry implementation and the apply path

- `KnobControl` is implemented by `EngineKnobs` (holding an `Arc` to the shared state).
- `set`:
  1. Validate (4.6).
  2. Push `PendingChange { req, reply: oneshot }` onto a bounded queue (64).
  3. Wait on the oneshot up to `timeout_ms`.
  4. On timeout: `Indeterminate { TimedOutBeforeStep }`. The change stays queued and an `ActivationEvent` records the outcome when it lands.
- At the top of each step, the engine drains pending changes:
  1. Re-check the dynamic preconditions.
  2. Build the new `EffectiveConfig` (`rev + 1`) and apply any side effect (graph invalidation for `attn_decode_splits`, scheduler limits).
  3. Publish it via `ArcSwap::store`.
  4. Read the effective value back (postcondition).
  5. Write `ConfigurationRevision` and `ActivationEvent` into the ring.
  6. Reply `Applied` with the step number.

  A change therefore takes effect on the very next step, and its first `EngineSample` carries the new `config_rev`. That is the "visible in telemetry within one decode step" requirement, and a test asserts it.
- `revert(target)`: the declared inverse is `RestorePrevious`, which sets the knob back to `previous` from that revision with cause `Inverse`.
  - It applies only while that revision is still the knob's latest. Otherwise it returns `Rejected { Superseded { by_rev } }`, so a revert never overwrites a later change.
  - `Trial(id)` stops an A/B started under that trial id and reverts each knob whose latest revision carries that id. Ending an A/B this way always writes a `ConfigurationRevision` with cause `Inverse` for the trial's knob, even when arm A was already in force, so every undo is visible in the ring. With nothing left to undo it returns `Rejected { NothingToRevert }`, so calling it twice is harmless. gait trials change one knob each.
- `ab` and `revert` wait for the engine to take them up to the watchdog's step limit (9.12, at least 5 s), not the 250 ms `DEFAULT_TIMEOUT_MS`: the engine takes commands only between steps, so a slow step delays them instead of refusing them. An A/B start not taken by then is withdrawn and answered `Busy`; a revert answers `Indeterminate { TimedOutBeforeStep }`. A step that runs past the limit is a fault (9.11).
- If a step faults after an apply and before the postcondition, the reply is `Indeterminate { FaultDuringApply }`. larch never silently reverts; gait decides.

### 9.9 Guardrail and A/B (engine thread; mechanism only, never a decision)

- **Guardrail:**
  - On every `Applied` change from `Gait` or `Human`, and on every A/B start, the engine snapshots a baseline: p90 of per-token decode time (`dur_ns / n_decode_seqs` on decode-dominant steps) over the previous 200 decode steps, bucketed by batch size.
  - It then tracks the rolling p90 over 32 steps for the same buckets: for the next 400 decode steps after a change, or on arm-B steps for the whole length of an A/B.
  - An A/B re-snapshots its baseline at each flip to arm B, from arm A's latest 200 decode steps. Traffic that drifts over a long A/B (chats lengthening, the KV pool cycling) then moves both sides, and only arm B slower than arm A trips it (#213).
  - If that exceeds `guardrail_ratio × baseline`, it reverts at the next step boundary, or ends the A/B on arm A (cause `Guardrail`), and writes an `ActivationEvent` naming the trial.
  - Once triggered, the revert lands within one step. The window size and the percentile are the price of not reverting on noise: a p99 of 32 steps is their maximum, so one slow step would trip it. p90 ignores the 3 slowest of 32; more than 3 of 32 steps over the ratio, or a steady slowdown past it, still trips.
  - The trade-off: the guardrail catches broad slowdowns, not rare stalls. In simulation, a slowdown on 3 % of steps was caught 39-46 % of the time over a 400-step watch, against about 100 % with p99. Noise that arrives in bursts of 4 or more slow steps still trips it.
- **A/B:**
  - `AbCommand::Start` saves arm A as the current values and arm B as A plus the deltas (validated like `set`).
  - The engine flips arms every `period_steps` (default 200) at step boundaries and stamps `ab_arm` in every record header.
  - It stops at `max_steps`, on `Stop`, on a guardrail trip, on `revert(Trial(id))`, or when the engine recovers from a fault (9.11). It always ends on arm A unless gait then sets B for real.
  - Requests are attributed to the arm in force at their first token (TTFT) and per step (TPOT).

### 9.10 Model fit (startup, reload)

- **Inputs:** `DeviceProps` (`vram_free`, or `recommended_working_set` on Apple), `ModelConfig`, weight bytes from GGUF, `max_context`, `kv_pool_fraction`, `max_batch_seqs` max.
- **Output:** `FitPlan { weights_bytes, scratch_bytes, graph_bytes, kv_blocks, max_seqs_fit, headroom_bytes }`, or `LoadError::DoesNotFit { need, have }`.
- Headroom is 512 MB on discrete GPUs (plus the display reserve on Windows) and 20 % of the working set on Apple.
- The KV pool must hold at least one full `max_context` sequence or loading refuses cleanly. That's the 8 GB card path (4060, 3050) and the 16 GB M5 path.

### 9.11 Faults: one path (D2)

Any engine-side fault ends the process and `larch supervise` brings it back. Nothing is reset in-process: after a sticky CUDA error the process must be terminated and relaunched (NVIDIA's runtime reference), a Windows TDR has already torn the context down, and one path is simpler to build and test than three.

**The path, on every fault in the table:** health → Failed (a `HealthEvent` carrying the `FaultCode`); every in-flight request ends with the error in the table and every queued request with `NotReady`; the ring's last records are flushed and gait's journal gets to fsync its open line; the process exits with code 3 within 100 ms, without waiting for the GPU. A library caller that set `EngineOptions.on_fault = Fail` (the library default; `Exit` is the CLI default) keeps its process: health stays Failed, every call returns `DeviceLost` or `NotReady`, and that process must restart before it can use the GPU again.

| Fault | Detected by | Callers see | Notes |
|---|---|---|---|
| Rust panic in a step | `catch_unwind` around `step()` | `EngineError::Internal` | the panic and the step plan are logged |
| `DeviceLost` / `Timeout` from a stream (CUDA sticky error, Windows TDR, Metal command-buffer error) | executor | `EngineError::DeviceLost` (retryable after the restart) | the status and the last launched kernel are logged |
| GPU hang (a Linux compute GPU without TDR; a Metal command buffer that never completes) | watchdog: no step completion for `step_timeout` (5 s, or 10 × p99 step) | `EngineError::Timeout` | the watchdog exits the process itself when the engine thread cannot |
| Process death (exit 3, kill, abort, OOM) | `larch supervise` | `larch chat` resumes its transcript (`<data_dir>/chat/<id>.jsonl`, written after every turn) and re-prefills once; `larch bench` counts the failed requests and carries on | restart at once on exit 3, except that a run that faults before its Ready, after an earlier fault and with no Ready since, backs off 0.25, 0.5, 1, 2 s (cap), so a child that faults at every start does not spin; any other death backs off 0, 0.25, 0.5, 1, 2 s (cap); both reset after 60 s healthy; Windows Job Object kill-on-close; the child exits on supervisor-pipe EOF on every OS; `LARCH_RESTARTED_AT` carries the fault time so the first Ready `HealthEvent` reports `recovery_ms` |
| strided/gait thread panic | `catch_unwind` in those threads | nothing | the thread restarts; it never touches the engine (requirement R6) |

**Target:** process start → Ready in **< 2 s** (rows R4/R10): warm page cache for the GGUF, a pinned staging buffer for the weight upload, the KV pool preallocated, graphs captured lazily. Measured at H30 on the 5070 Ti, Windows native, with a real TDR and an injected sticky error. Row R5 adds 20 kills in a row with no zombies and no leaked GPU memory (the OS reclaims per process; a test checks free VRAM before and after).

Fault injection for tests is compiled in behind the `fault-inject` feature (on in test and bench builds, off in release): `LARCH_FAULT=panic@step:N | device_lost@step:N | hang@step:N | oom@alloc:N`. Device lost on CUDA is real, from a deliberate out-of-bounds kernel in a debug-only op, so the sticky-error path is exercised for real; the test runs under `larch supervise` and asserts the restart time from the ring.

### 9.12 Watchdog thread

- **Inputs:** the engine heartbeat (`AtomicU64` monotonic ns of the last step completion), whether work is pending, and `step_timeout`.
- **Output:** `HealthEvent`, then process exit with code 3 on a hang (9.11).
- It never touches the GPU.

### 9.13 Telemetry producer (engine thread)

- Writes `EngineSample` every step at T1, or every 64th step plus all spans and events at T0.
- Writes `RequestSpan` on every end, plus `ConfigurationRevision`, `ActivationEvent` and `HealthEvent`. Copies `DeviceSample` from the sampler slot when its sequence changes, and writes `DeploymentSnapshot` and `HostSnapshot` at load.
- Overhead: one 256-byte slot write per step, so ≈ 1 µs on a 5 to 20 ms step. R5's ≤ 1 % claim is proven by `larch bench --telemetry off|t0|t1` (3 minutes each, same seed), with the result kept in `bench/results`.

---

## 10. `larch`: library facade, CLI, supervisor

### 10.1 Library API (R4)

```rust
pub struct Larch { engine: EngineHandle, model: Arc<ModelInfo>, tuner: Option<TunerThreads> }
impl Larch {
    pub fn load(opts: LoadOptions) -> Result<Larch, LoadError>;     // Loading → Ready
    pub fn session(&self, opts: SessionOptions) -> Session;          // chat template + history + parked KV (10.1.1, D14)
    pub fn generate(&self, prompt: Prompt, p: GenParams) -> Result<TextStream, EngineError>; // Prompt::Text | Prompt::Messages | Prompt::Tokens
    pub fn telemetry(&self) -> RingHandle;
    pub fn knobs(&self) -> Arc<dyn KnobControl>;
    pub fn health(&self) -> Health;
    pub fn start_tuning(&self, contract: gait::TuneContract) -> Result<(), TuneError>; // spawns strided + gait threads (feature `tune`)
    pub fn stop_tuning(&self);
}
pub struct GenParams { pub max_tokens: u32, pub temperature: Option<f32>, pub top_p: Option<f32>, pub top_k: Option<u32>,
                       pub repeat_penalty: Option<f32>, pub seed: Option<u64>, pub stop: Vec<String>, pub logprobs: u8 }
pub struct TextStream { /* Iterator<Item = TextEvent>; cancel(); drop = cancel */ }
pub enum TextEvent { Text { delta: String, tokens: u32, logprobs: Option<Vec<TokenLogprob>> }, Done { finish, usage: Usage, timings: Timings /* ttft_ms, tpot_ms, tok_s, cached_tokens, dropped_exchanges */ }, Error(EngineError) }
```

The API is synchronous, uses no async runtime, and channels are crossbeam.

#### 10.1.1 `Session`: a conversation keeps its KV (D14)

```rust
pub struct SessionOptions { pub system: Option<String>, pub keep_kv: bool /* true */, pub on_overflow: Overflow /* DropOldestExchanges (default) | Fail */ }
pub struct Session { inner: Arc<Mutex<SessionInner>>, engine: EngineHandle }
struct SessionInner { id: u64 /* new on every reset, never reused */, history: Vec<ChatMessage>, active: Option<ReqId>, dropped_exchanges: u32,
                      template: Arc<ChatTemplate>, tokenizer: Arc<Tokenizer> }
impl Session {
    pub fn send(&mut self, user: &str, p: GenParams) -> Result<TextStream, EngineError>; // InvalidRequest(SessionBusy) while a turn is active
    pub fn reset(&mut self);                 // cancels the active turn, releases the parked sequence, takes a new id, keeps only the system message
    pub fn history(&self) -> Vec<ChatMessage>;
    pub fn tokens_in_kv(&self) -> u32;       // EngineHandle::parked_tokens(id); shown by `larch chat` after each turn
}
```

- **Four states, never interchangeable:**

| State | Where | How it changes |
|---|---|---|
| Chat history | `SessionInner.history`: user and assistant messages | `send` appends the user message; the stream's terminal event appends or removes per the table below |
| Rendered prompt | nowhere: `send` renders the whole history with the generation prompt (template delimiters and special tokens included) and discards the string after tokenising it | recomputed every turn, never stored or patched |
| Token history | `Seq.tokens` in the engine: the ids fed through the model, in order | `send` tokenises the **whole** rendered string; the engine keeps the common prefix (9.3 step 2) and appends each token it feeds; matching and reclaim truncate it |
| KV length | `Seq.n_kv` | the number of those tokens with KV entries: `== tokens.len()` once parked; the only authority on what is cached |

- **`send`:** append the message, render, tokenise the whole string, submit `Request { prompt: tokens, session: id, keep_kv, .. }` and record `active`. There is no incremental path: `tokenize(A + B)` is not `tokenize(A) + tokenize(B)` in general (merges across the join, special-token handling), so the string-diff trick `llama-cli` uses in `common_chat_format_single` is not used. Only CPU tokenisation is repeated, about a millisecond for 8k tokens against hundreds for a GPU prefill, and `llama-server` works the same way (full re-tokenise, `get_common_prefix`, `keep_first`). Incremental tokenisation is post-sprint and lands only with boundary tests (18). The next render must tokenise back to the ids the engine fed over the whole history; where it does not (a template that re-renders the assistant text differently), the common prefix stops there and the rest is re-prefilled: correct, slower, and visible as `cached_tokens < tokens_in_kv()` (`template_roundtrip` in 15 checks the requirement-R4 templates).
- **Terminal events, and what each state does:**

| Event | Chat history | Token history and `n_kv` (engine) |
|---|---|---|
| `Done { finish: Eos \| Length }` | assistant message = all delivered text, the last sampled token's text included | the fed tokens; the last sampled token is not among them (9.7); parked under `id` when `keep_kv`, else freed |
| `Done { finish: StopSeq }` (a string stop matched by the facade, 9.7) | assistant message = the text before the stop string | the fed tokens, the stop string's included; parked. The next turn's prompt diverges at the stop string, so matching truncates there |
| Cancelled (the stream dropped, `cancel()`, or `reset`) | assistant message = the delivered partial text, `ChatMessage.truncated = true` | freed, never parked (one rule, no race with `reset`); the next turn re-prefills the history once |
| `Error` (rejected at intake, or failed) | the user message is removed: the turn did not happen | a rejection leaves the previous parked sequence as it was; a fault leaves nothing (9.11) |

- **Lifecycle, one active turn:** `send` while `active` is set returns `InvalidRequest(SessionBusy)`. The `TextStream` holds `(inner, id, req_id)`; its terminal event applies the table only when `inner.id == id` and `inner.active == Some(req_id)`, then clears `active`; a stale completion changes nothing. Dropping the stream mid-turn cancels the request (10.1) and applies the Cancelled row. `reset` cancels the active request, calls `release_session(id)`, takes a fresh `id` and truncates the history to the system message. Engine side, `release_session` is sticky: the id enters a released ring (the last 4096 ids; ids are never reused) and a `park` under a released id frees instead, so a reset session cannot be resurrected by a late completion. `Drop for Session` calls `release_session(id)`.
- **Overflow:** when the tokenised prompt plus `max_tokens` would exceed `max_context`, `DropOldestExchanges` drops whole exchanges from the oldest (a user message with its assistant reply; a tool-call exchange, when tool calling arrives, is the call, its tool results and the final answer, dropped together), never the system message, re-renders, and reports it, because the model's knowledge changed: `Timings.dropped_exchanges`, and `larch chat` prints `(dropped N oldest exchanges)`. The common prefix then shrinks to the system message, re-prefilled once. `Fail` returns `PromptTooLong`. No KV shifting in the three days. The overflow path is tested on its own (W1 at a 2k context), never inside a benchmark row.
- **Output:** a `TextStream`; `Timings.cached_tokens` says how many prompt tokens came from the parked sequence.
- `generate(...)` without a session submits `session: 0, keep_kv: false`: nothing parks, since cross-session adoption is post-sprint (18). Conversations that want reuse go through `Session`; `larch bench` uses it for W1 to W3 (16).

### 10.2 CLI (`larch` binary)

| Command | Inputs | Output |
|---|---|---|
| `larch info` | none | devices, backend, driver/runtime versions, the minimum-driver check, fit estimate per known model |
| `larch chat -m <gguf> [--set k=v]... [--tune [--tune-contract c.json]] [--allow-resource] [--resume <id>]` | REPL text | streamed tokens; after each turn `tok/s · TTFT · tokens · cached` and, when the overflow policy dropped history, `(dropped N oldest exchanges)` (D14). Slash commands: `/knob list\|get <k>\|set <k> <v>\|revert <rev>`, `/ab <k>=<v> [period]`, `/tune on\|off`, `/approve <trial>`, `/status` (health, findings, trials, packets), `/reset` (cancels the turn and releases the parked conversation). `--set` gives startup values (`EngineOptions.knob_overrides`). The transcript is written to `<data_dir>/chat/<id>.jsonl` after every turn and `--resume` continues it after a supervised restart (9.11) |
| `larch bench -m <gguf> [--engine larch\|ollama --url <http://host:11434>] --workload chat\|chats\|agent\|batch\|longprompt\|fixed\|tune [--replay scripted\|generated] (D14) or --mode serve\|batched --concurrency 1,4,8,16 [--prompts bench/prompts/chat.jsonl] [--pp 512 --tg 128] [--duration 180] [--telemetry off\|t0\|t1] [--tune [--tune-contract c.json]] [--set k=v[@t]]... --json out.json` | the fixed benchmark (section 16); `--set` without `@t` is a startup value | machine-readable JSON (agg tok/s, TTFT/TPOT p50/p99, e2e, failures, achieved GB/s, knob timeline, guardrail trips) and one human table |
| `larch knob list` | none | the registry table (static; live knobs are changed inside `chat`, `bench --set` and `--tune`, since there is no IPC) |
| `larch parity -m <gguf> --ref <dump.jsonl> --bound <kl> --margin <nats> [--prompts N] [--mode prefill\|decode\|chunked\|cached\|all] [--parallel N] [--graphs on\|off] [--set k=v]...` | a llama.cpp dump and its calibrated bound and margin (section 15) | per-prompt, per-mode top-1 (raw and over the decisive positions) and KL; exit 0 pass, 1 when any mode misses its gate (or a run fails), 2 when it could not run as asked (inputs, device, unverified path), 3 on an engine fault (9.11) (D16) |
| `larch tokcheck -m <gguf> --ref <tokens.jsonl>` | llama.cpp tokenisation dump | exact-match report |
| `larch supervise -- <command…>` | a child command line | restarts the child per 9.11; logs restarts as JSON lines |

### 10.3 Supervisor (per-OS seam behind one trait)

```rust
trait ChildGuard { fn spawn(cmd: &Command) -> io::Result<Child>; }   // windows.rs | unix.rs
```

- **Windows:** the `win32job` crate creates a Job Object with `KillOnJobClose` and assigns the child.
- **All OSes:** the supervisor passes the child an inherited pipe (`LARCH_SUPERVISOR_FD` / handle). A child thread blocks reading it and exits on EOF, so the child dies if the supervisor dies.
- Exit code 3 means a fault (9.11): the supervisor restarts at once and passes `LARCH_RESTARTED_AT=<ns>`, unless the run faulted before its Ready after an earlier fault with no Ready since, which takes the backoff from 0.25 s; any other death takes the backoff. Every restart is one JSON log line: exit code, the fault from the child's last `HealthEvent` when the ring could be flushed, and the time to the child's first Ready.

This is the one OS-specific module in the binary besides paths (`dirs` crate). There's no `unsafe`; `PR_SET_PDEATHSIG` isn't needed because the pipe-EOF guard covers Linux and macOS identically.

---

## 11. strided (Rust), in its repository's `crates/strided`

### 11.1 Modules

| Module | Inputs | Outputs | Port of |
|---|---|---|---|
| `schema` | none | `DiagnosisInput` and sub-structs, a Rust port of schema 1.8.0 for the fields the six rules and the engine read (`Distribution`, `PhaseMetrics`, `LayerMetrics`, `VllmServingMetrics` renamed `ServingCounters` with the same fields, `EngineStep`, `NsysTimeline` renamed `StepTimeline`, `ThroughputSample`), `Option` fields, finite-only constructors (1.5.0 hardening). serde field names identical to the pydantic ones | `schema/diagnosis_input.py` |
| `stats` | slices | `theil_sen_slope`, `mann_kendall -> TrendTest { s, z, p }`, `median` (Python `statistics.median` semantics), `normal_cdf`, `py_round` (round-half-even) | `rules/_stats.py` |
| `rules::{base, r01, r02, r03, r06, r08, r12}` | `&DiagnosisInput` | `RuleResult = Diagnosis \| InsufficientData \| Abstention` with identical constants, gates, confidence formulas, the 0.65 uncalibrated cap and evidence keys | `rules/*.py` |
| `engine::{registry, relations, confidence, runner}` | `&DiagnosisInput`, flags | `DiagnosisReport { diagnoses: Vec<RankedDiagnosis>, insufficient_data, suppressed, errors, warnings, rules_run, engine_version }` | `engine/*.py` |
| `observe` | `RingReader` (larch-records), `DeploymentSnapshot` | `Window`: a rolling state that emits one `DiagnosisInput` per tick (1 s default) | `collect/window.py` + `collect/session.py`, reading larch records instead of Prometheus |
| `thread` | `RingHandle`, `StridedConfig { tick_ms, enable_corroboration_boost: false }` | `StridedHandle { latest: Arc<ArcSwap<DiagnosisReport>>, findings: Receiver<Finding>, stop() }` | new |

`Finding { tick: u64, t_mono_ns: u64, rule_id: [u8; 3], confidence: f64, adjusted: f64, evidence: Evidence, recommendation: Option<Recommendation> }`, where `Recommendation` carries typed values the rules already compute (r08's `recommended_max_num_batched_tokens`, r03's engine-specific fix tier).

### 11.2 How larch records become rule inputs (`observe`)

This step is aggregation over larch's own records, not an engine-neutral adapter.

| DiagnosisInput field | Built from | Rule(s) |
|---|---|---|
| `model_name`, `gpu_type`, `inference_engine = "larch"`, `model_params_b`, `dtype` | `DeploymentSnapshot` | all |
| `batch_size` | mean `n_decode_seqs` over the tick | r01 |
| `decode.hbm_bandwidth_util`, `decode.sm_occupancy` | duration-weighted mean of `hbm_util` and `sm_occupancy` over decode steps in the tick. If either is Unsupported/Missing, the field is `None` and the state is recorded in `parse_warnings` | r01 |
| `kv_cache_util`, `kv_cache_fragmentation`, `kv_block_size`, `kv_num_blocks_total/used` | last `EngineSample` of the tick. `kv_cache_util` is (`kv_blocks_used` − `parked_blocks`) / `kv_blocks_total`: parked prefix-cache blocks are reclaimable on demand (D14), and vLLM's usage gauge leaves evictable cached blocks out | r03, r12 |
| `pressure_window = "delta"` | always, because counters are differenced per tick | r03 |
| `serving.{num_preemptions_total, request_success_total, prompt_tokens_total, generation_tokens_total, num_requests_running, num_requests_waiting, request_queue_time_ms_sum, request_queue_time_count}` | `tot_*` counters and gauges of the last `EngineSample` | r02, r03, r12 |
| `ttft_ms`, `e2e_latency_ms` (`Distribution` with p50/p95/p99) | `RequestSpan`s that ended in the last 60 s | r02 |
| `tpot_ms` (`Distribution`) | per output token, like vLLM's `time_per_output_token` histogram that r02's bands came from: each decode row of a step in the last 60 s counts once, at the gap since the previous step ended. A gap counts only when the previous sample is the step just before (`step == prev + 1`: t0 samples 1 step in 64, and a lagged read drops some), that step left sequences running, and the clock advanced. A preempted sequence's next token comes out of a prefill step and is not counted | r02 |
| `queue_time_ms` | Δ`tot_queue_ns` / Δ`tot_queued` | r03 |
| `token_throughput_gen`, `request_throughput`, `token_throughput_prompt` | per-tick counter deltas / dt | r06 via history |
| `throughput_history: Vec<ThroughputSample>` | one sample per tick: `t` = monotonic s; `preemption_rate` = Δpreempt / max(Δsuccess, 1); `queue_time_ms` = Δsum/Δcount; `num_requests_waiting`; `sm_clock_mhz`, `gpu_temp_c` from the latest `DeviceSample` (`None` when Unsupported). Kept at max 240 samples / 1800 s, as in `ThroughputHistory` | r06, r12 |
| `nsys_timeline.steps` | the last 512 `EngineSample`s mapped to `EngineStep { start_ms, end_ms, phase, num_prefill_tokens, num_decode_seqs }`; `trace_duration_ms` = span of those | r08 |
| `prefill`, `decode` phase latencies | step durations by phase | r02 roofline guard inputs where available, else `None` |

`layers` stays `None` in the 3 days (no per-kernel profiler), so r01 reads the collapsed `decode.*` path. That's exactly the fallback the Python rule already supports.

Measurement states map honestly: Estimated values are used and listed in `parse_warnings` as `"estimated: decode.hbm_bandwidth_util"`; Unsupported and Missing values become `None`, so the rule returns `InsufficientData` naming the field. Rule logic is never changed to accommodate larch.

### 11.3 Porting discipline and proof

- **Exactness.** Constants, comparison strictness, gate order, abstention kinds, confidence formulas and caps, evidence keys and rounding are reproduced exactly. Python's round-half-even, `statistics.median` and `NormalDist` are replicated in `stats`.
- **1:1 tests.** Every Python test function in `tests/rules/test_{r01,r02,r03,r06,r08,r12,stats}.py` and `tests/engine/test_{runner,confidence,relations,registry}.py` gets a Rust test of the same name. Builders are translated (`_input`, `_series` with the same seeded noise: the Python `random.Random(seed)` stream is replaced by pre-generated vectors committed as test data). That's about 223 tests.
- **Differential parity** (`tests/parity.rs`, run in strided CI and locally):
  1. proptest generates `DiagnosisInput`s per rule around its thresholds;
  2. each is serialised to JSON with pydantic field names;
  3. the harness invokes `python3 -c` with a fixed 10-line inline program (`DiagnosisInput.model_validate_json` → `run_diagnosis` → JSON) against the Python strided checkout;
  4. it asserts the same verdict kind, confidence within 1e-12, and identical evidence keys and values.

  That's the handoff's "run both against the same fixture, match the verdicts". No Python module is added to the repo.
- **On-disk fixtures** (`tests/fixtures/collect/replay/*.prom`, `examples/*`) aren't parsed by the Rust side, because Prometheus and Nsight parsing are not ported. Their stories (r02/r03/r01 fire on tick 00, r03 clears on tick 02) are reproduced as larch-record sequences in `tests/observe.rs`.
- **High-risk surfaces** (`rules/base.py`, predicates, thresholds, confidence) are covered by the sprint policy (D4): they are ported, never changed. Any intentional difference is a stop condition.

---

## 12. gait (Rust), in its repository's `crates/gait`

A deterministic tuner (D5) with the risk-class authority policy (D7). It ports the orchestrator's run model, with a knob value in place of a code change:
- a contract bounds what may change;
- a fixed sequence of phases moves one candidate from finding to verdict;
- a hash-chained journal records an intent before each phase acts and an outcome after;
- a packet records the result.

None of the repo-modification machinery is ported in the 3 days: writer and critic agents, sandboxes, worktrees, diff scope checks, relays and handback.

### 12.1 What carries over from the orchestrator (`da468c4`)

| Orchestrator | gait | What changes |
|---|---|---|
| `schemas/task-contract.schema.json` | `TuneContract` (12.2) | `scope` → allowed knobs. `risk_tier` → allowed risk classes. `behavior_envelope.must_preserve` → metric constraints. `evidence` → minimum samples per arm. `rollback` → each knob's declared inverse (`RestorePrevious`), so it is no longer free text |
| `RunState` and the transition table in `coordinator.py` | `Phase` and a `const` transition table (12.4) | INTAKE, PREFLIGHT, BASELINE, APPLY (was WRITE), VERIFY, REVIEW, PACKET. There are no scope-check, snapshot or remediate phases: a knob value has no diff to check, and a worse candidate is reverted, not remediated |
| `PhaseOutcome` {SUCCEEDED, NEEDS_REMEDIATION, ABSTAINED, BLOCKED} | `PhaseOutcome` {Succeeded, Reject, Abstained, Blocked} | NEEDS_REMEDIATION becomes Reject, which goes to PACKET with a revert verdict |
| `StopCondition` (17 values) | `StopCondition` (9 values, 12.5) | the 7 that mean something for a tuner keep their meaning; `EngineNotReady` and `Stopped` are new |
| `Abstention { check_id, reason, evidence }`, evidence required | the same struct; its constructor rejects empty evidence | none |
| `RunBudget { max_remediation_rounds, max_elapsed_seconds, max_tokens, max_cost_microunits }` | `TuneBudget { max_trials, max_wall_s, max_steps_per_trial }` | tokens and cost don't apply to a tuner |
| `run_journal.py` (a JSONL sha256 chain of intent/outcome pairs, plus a protected head file) and `run_assembly.py` (`_drive`, `_paired`) | `journal` (12.6): the same chain and the same pairing | The protected head file is deferred: the Python one relies on POSIX `fcntl` and `O_NOFOLLOW`, and Windows needs its own design. Recovery undoes an unpaired intent instead of refusing to resume, because every change gait makes has a declared inverse |
| `schemas/final-packet.schema.json` | `TrialPacket` and `SessionPacket` (12.2) | The outcome is {Kept, Reverted, Inconclusive, Blocked} instead of {READY_FOR_HUMAN_REVIEW, FAIL, BLOCKED}. `verifier_results` become the arm statistics and checks |
| Actor roles (coordinator, verifier, critic) | phases inside the one gait thread | VERIFY is the verifier (statistics). REVIEW is the critic (constraints, prediction, failed requests) |

From strided's old `gait/`, three ideas carry over:
- the `Prediction` recorded before acting;
- the pending entry journaled before mutating, which is now the APPLY intent;
- verify's 25 % traffic drift giving Inconclusive.

### 12.2 Modules

| Module | Inputs | Outputs | Concept from |
|---|---|---|---|
| `contract` | a JSON file (`--tune-contract`), or `TuneContract::default_for(registry, allow_resource)` | `TuneContract { contract_version, id, objective: Objective { primary: AggTokS \| TtftP99 \| TpotP99 \| ItlMax, constraints: Vec<Constraint { metric, op, bound: Abs(f64) \| RatioToA(f64) }> }, allowed_knobs: Vec<u16>, allowed_risk: RiskSet, evidence: Evidence { min_steps_per_arm /* 400 */, min_requests_per_arm /* 20 */, traffic_tolerance /* 0.25 */ }, budget: TuneBudget, stop_on: Vec<StopCondition> }`, plus `digest()` (sha256 of the canonical JSON). The default contract (id `default-latency`, or `default-latency+resource` with resource knobs allowed) lowers `ItlMax`, keeping `AggTokS` ≥ `RatioToA(0.92)`, TTFT p99 ≤ `RatioToA(1.4)` and TPOT p99 ≤ `RatioToA(1.10)` as arm point estimates. Latency comes first because a 200-step A/B on chat traffic swings ±25-36 % in per-window tok/s, too much to confirm a few-percent throughput change live; the throughput and TTFT bounds catch large harm, not 3 %. It allows every Live knob whose risk class the session allows, except `telemetry_tier` and `guardrail_ratio`: those observe and limit gait, so gait never changes them | task contract, `RunBudget` |
| `candidates` | `Finding`, `EffectiveConfig`, `KnobSpec`s, the session's tried set | `Vec<Candidate { knob, value, rationale, from_rule }>`, in 12.3 order. Any (knob, value) already tried this session is skipped, so a rejected candidate is never retried | old gait `FixSpec`, re-targeted to larch knobs |
| `explore` | `EffectiveConfig`, contract, the tried set | `Candidate`s when no finding is open: coordinate steps over the allowed live performance knobs (×2 / ÷2, then ±25 %), one knob at a time | new (keeps improving under steady load) |
| `prediction` | `Candidate`, current metrics | `Prediction { checks: Vec<Check { metric, op, threshold, advisory }>, summary }`, journaled in the INTAKE outcome before anything acts | old gait `Prediction` |
| `policy` | `Candidate`, `KnobSpec`, contract | `Authorize::Auto \| NeedsResourceFlag \| NeedsHuman \| Refuse` (D7) | orchestrator authority boundaries, old gait `approve` |
| `ab` | ring records tagged by `ab_arm` | `ArmStats { n_steps, n_requests, n_failed, agg_tok_s, ttft_p50/p99, tpot_p50/p99, itl_max, traffic: req/s, mean prompt len }` and `Comparison { delta, ci95, p }` (Mann-Whitney on the primary's per-window values; bootstrap CI). `itl_max` is inter-token latency's longest gap: each window's longest step that carried decode rows, prefill-bearing ones included, averaged over the arm's windows. Every running chat waits that whole step for its next token, so it sees the prefill stalls TPOT skips, and it sizes them where a p99 over row-gaps counts only their share. It stands in for the gap between step ends, and relies on `sched.rs::schedule` planning every running decode row every step (9.3) | old gait `verify` |
| `trial` | all of the above, `KnobControl`, `RingReader`, `Journal` | one trial run through the phases in 12.4, returning a `TrialPacket` | `coordinator.py` |
| `journal` | phase intents and outcomes | the hash-chained JSONL file (12.6) and `open_trial()` for recovery | `run_journal.py` |
| `packet` | a finished trial or session | `TrialPacket { packet_version, session_id, trial_id, contract_digest, outcome, finding: Option<FindingRef { rule_id, confidence }>, candidate, prediction, baseline: ArmStats, arm_b: Option<ArmStats>, comparison: Option<Comparison>, checks: Vec<CheckResult>, guardrail: Option<GuardrailTrip>, revs: { baseline, applied: Option<u64> }, abstentions, residual_risks, next_human_actions }` and `SessionPacket { session_id, contract_digest, engine_id, started, ended, stop: StopCondition, trials: Vec<(trial_id, outcome)>, kept: Vec<(knob, value)>, residual_risks, next_human_actions }` | final packet |
| `thread` | `Arc<dyn KnobControl>`, `RingHandle`, `StridedHandle`, `TuneContract` | `GaitHandle { status: ArcSwap<GaitStatus>, approve(trial_id), stop() }` | new |

larch doesn't persist knob values, so a restart returns to the startup configuration. `next_human_actions` is how a kept change becomes permanent: it gives the flag to start with, for example `--set prefill_chunk_tokens=1024`.

### 12.3 Finding → candidate mapping (3 days)

| Rule fires | Candidate (in order) | Class |
|---|---|---|
| r08 prefill/decode interference | `prefill_chunk_tokens` = r08's `recommended_max_num_batched_tokens` (clamped); then `schedule_policy = decode_first` | live, performance-only (auto) |
| r02 colocation contention | `prefill_chunk_tokens` ÷ 2 (floor 128) | live, performance-only (auto) |
| r12 queue growth | `max_batch_seqs` × 2 (≤ fit); then `max_waiting` lower (needs `--allow-resource`) | perf (auto); resource (flag) |
| r03 KV fragmentation / pressure | `kv_preempt_watermark` −0.03; then `max_batch_seqs` ÷ 2 | resource (flag); perf (auto) |
| r01 decode memory-bound at low batch | `max_batch_seqs` × 2 (≤ fit) | perf (auto) |
| r06 throughput decay | memory leg: as r03; thermal leg: **recommend only** (no knob can fix heat) and journal it | as above / none |

### 12.4 Trial phases

```
INTAKE → PREFLIGHT → BASELINE → APPLY → VERIFY → REVIEW → PACKET → Kept | Reverted | Inconclusive
any phase that ends Abstained or Blocked → PACKET (undoes whatever APPLY did) → Inconclusive | Blocked
```

| Phase | Orchestrator | What it does | Next |
|---|---|---|---|
| INTAKE | INTAKE | Opens a trial for the candidate the session picked (12.5) and builds its `Prediction` | PREFLIGHT |
| PREFLIGHT | PREFLIGHT | Checks that health is Ready, `telemetry_tier` is t1 (the verdict needs every step's sample), the knob is in `allowed_knobs`, budget remains, and `policy` gives `Auto` (an `/approve`d candidate runs with `Authority::Human` instead; a `QualityAffecting` one still ends Blocked at APPLY, 4.6) | BASELINE. t0 → Abstained `RequiredAbstention`. `NeedsResourceFlag` → Blocked `OutOfScope`. `NeedsHuman` → Blocked `UnauthorizedBehaviorChange`. Both are listed in `/status` with `/approve <trial>` |
| BASELINE | BASELINE | Records `engine_id`, `EffectiveConfig.rev`, the knob's current value (arm A) and the last 60 s of `ArmStats` as the pre-trial reference | APPLY |
| APPLY | WRITE | `ab(Start { b: [candidate], period_steps: 200, max_steps: 3 × max_steps_per_trial, trial_id, authority, allow })`, room for the extended look. Apart from PACKET, this is the only phase that changes the engine | VERIFY. `Rejected` → Blocked with the reason |
| VERIFY | VERIFY | Reads the ring until each arm has `min_steps_per_arm` decode steps and `min_requests_per_arm` finished requests (the first look), or until `max_steps_per_trial` passes. Computes `ArmStats` and `Comparison`. A first look that leans B's way without reaching p < 0.005 runs on once to 3 × that evidence (the extended look), if the wall budget leaves twice the first look's engine time; if the extended look then runs out of time or steps, REVIEW judges the first look | REVIEW. Guardrail trip → Reject, with the `ActivationEvent` as evidence. Arm B's mean prompt length differing from arm A's by more than `traffic_tolerance` (a mix that differs between the interleaved arms; traffic that moved on both is fine), or too few samples → Abstained `EvidenceUnreliable`, after one retry with doubled samples. Health leaves Ready → Blocked `EngineNotReady`. A `ConfigurationRevision` on the trial's knob from anyone else → Blocked `StateMismatch` |
| REVIEW | REVIEW | The critic. Keep only if all of these hold: the primary objective improved with the 95 % CI excluding 0 and Mann-Whitney p below the look's threshold (0.005 at the first look, 0.045 at the extended one); every constraint holds on arm B; no non-advisory `Prediction` check failed; arm B has no more failed requests than arm A | PACKET with Keep or Reject |
| PACKET | PACKET | Keep: `ab(Stop)`, then `set(candidate)` with the trial id and `timeout_ms` = 10 × VERIFY's slowest step, between 5 s and 60 s, since the engine takes a set only between steps (9.8). Reject, Abstained or Blocked: `revert(Trial(id))`, which stops the A/B on arm A and restores anything the trial set. Then it builds and journals the `TrialPacket` | Kept, Reverted, Inconclusive (from Abstained) or Blocked |

Any transition not in this table is an error. The `const` table is tested like the orchestrator's `LEGAL_TRANSITIONS`.

A `KnobReply::Indeterminate` during APPLY or PACKET blocks the trial with `StateMismatch`, and PACKET's revert undoes it. If that revert is also Indeterminate, the session stops and surfaces it. gait never loops on a revert.

### 12.5 Sessions and stop conditions

A session runs from `start_tuning(contract)` (or `--tune`) until a stop condition ends it. It runs one trial at a time and finishes with a `SessionPacket`.

Between trials, the session picks the next candidate:
1. a candidate a human `/approve`d (`/approve <trial>` reopens a Blocked trial's candidate as a new trial);
2. the first finding in the latest strided report, in rule order, that still has an untried candidate (12.3);
3. otherwise an `explore` step;
4. otherwise nothing: it waits for the next report.

A candidate equal to the knob's current value is dropped, and a (knob, value) pair is tried at most once per session.

Between trials, and again before the `SessionPacket`, the session drops any kept value the engine no longer holds (a guardrail revert after the keep, or a human `larch knob` set), and says so in `residual_risks`, so `next_human_actions` never advises a stale `--set`. A packet built within 60 s of a keep, a stand-in for the guardrail's watch, also says to confirm the value before passing `--set`.

| `StopCondition` | Orchestrator name | Fires when | Ends |
|---|---|---|---|
| `BudgetExhausted` | `budget_exhausted` | `max_trials` or `max_wall_s` is reached | session |
| `EvidenceUnreliable` | `oracle_evidence_unreliable` | traffic drift or too few samples; on the third Inconclusive trial in a row, it also ends the session | trial; session on the third |
| `RepeatedBlockingFinding` | `repeated_blocking_finding` | the guardrail reverts the same knob twice in one session, during a trial or after a keep | session |
| `RequiredAbstention` | `required_abstention` | the telemetry a verdict needs isn't recorded: `telemetry_tier` is t0, or a needed field is `Missing` or `Unsupported` | session |
| `UnauthorizedBehaviorChange` | `unauthorized_behavior_change` | the candidate is a `QualityAffecting` knob and no human approved it | trial |
| `OutOfScope` | `required_surface_out_of_scope` | the knob isn't in `allowed_knobs`, or it is `ResourceOrAvailability` without `--allow-resource` | trial |
| `StateMismatch` | `repository_state_mismatch` | someone else changed the trial's knob mid-trial, or a reply was `Indeterminate` | trial; session if the undo is also Indeterminate |
| `EngineNotReady` | new | health leaves Ready during a trial | trial; the session waits for Ready |
| `Stopped` | new | `/tune off` or `stop_tuning` | session, after the open trial is undone |

The orchestrator's other 10 conditions are about agents, sandboxes and diffs, and have no meaning for a tuner.

### 12.6 Journal and recovery

- **File.** `<data_dir>/gait/<session_id>.jsonl`, one per session, as the orchestrator keeps one per run.
  - The first line is a header: `{kind: "header", header: {journal_version, session_id, contract_digest, engine_id, larch_version, registry_digest}, digest}`.
  - Each later line is `{kind: "intent" | "outcome", sequence, trial_id, phase, payload, previous_digest, digest}`.
  - `digest` is the sha256 of the line's canonical JSON (sorted keys, no whitespace) without `digest`. `previous_digest` chains each line to the one before.
  - Each line is written, then `sync_data`'d.
- **Pairing.** Every phase writes its intent before it acts and its outcome after. One rule covers all seven phases.
- **Verification on open.** A partial last line (a crash mid-write) is dropped, and recovery treats its phase as unpaired. Any other break (a bad digest or chain link, a bad pairing, a mismatched header) refuses the session and shows in `/status`. The engine is unaffected.
- **Recovery** runs when the gait thread starts, including after its own panic. It keeps the orchestrator's rule that an intent with no outcome may already have taken effect, so it is never run again. Where the orchestrator then refuses to resume, gait undoes instead, because every change it can make has a declared inverse:
  - **Same `engine_id`** (gait restarted inside a live process): it calls `revert(Trial(id))` for the last trial that has no PACKET outcome. The engine stops that trial's A/B, and restores its knob only while the trial's revision is still the knob's latest, so a later human change is never overwritten. The reply is journaled as that trial's PACKET outcome, with `recovered: true`.
  - **Different `engine_id`** (the process restarted): the new engine started from its startup configuration, so nothing from the trial survives. gait journals `recovered: engine_restarted` without calling the engine.
- **Deferred:** the protected head file (which detects truncation) and deterministic replay of a session.

### 12.7 Demo path for row R13

Under `larch bench --tune` load, gait applies three live knobs with zero failed requests:
- r08 → `prefill_chunk_tokens`;
- r12/r01 → `max_batch_seqs`;
- explore → `attn_decode_splits`.

One change gets reverted through the declared inverse. The demo forces this with a contract constraint the candidate violates. Each change appears in the ring on the next step.

---

## 13. cuprum (scaffold, `crates/cuprum`)

- `cuprum::proto`: `WorkerHello { engine_id, backend, device_name, deployment: DeploymentSnapshot }`, `RouteRequest { request: RequestWire, target: WorkerId }`, `KnobForward { req: KnobRequest }`, `TelemetryBatch { records: Vec<RecordWire> }`.
  - Every payload reuses `larch-records` types through `serde` mirrors.
  - The JSON Schema is shared with 4.8.
- The `cuprum` binary prints its version and `--schema`.
- Builds in CI on all three OSes. Nothing else exists in the 3 days.

---

## 14. Per-OS seams

| Seam | Windows (native) | Linux (native; GPU runs in WSL per D8) | macOS |
|---|---|---|---|
| Toolchain | VS 2022 Build Tools 17.x (C++ workload; VS 2026 also works with CUDA ≥ 13.2), CUDA 13.4.2 for Windows, CMake ≥ 3.24, Ninja, LLVM (libclang for bindgen), Rust 1.97 `x86_64-pc-windows-msvc` | GCC 14 (WSL: `CUDAHOSTCXX=/usr/bin/g++-14`, CUDA 13.3; CUDA 12.8 fails with the distro glibc) or Clang; CI: Ubuntu 24.04 + CUDA 13.4.2 | macOS 26, Xcode 26 (Metal 4, MetalPerformancePrimitives), CMake, Rust 1.97 `aarch64-apple-darwin` |
| Binary | `larch.exe` + `cublas64_13.dll`, `cublasLt64_13.dll` alongside | Source build: `larch` + `libcublas.so.13` from the toolkit/driver install. Release: the `cuda` (HIP: `rocm-<gfx>`) runtime package in the folder beside the binaries' folder, found through DT_RPATH `$ORIGIN:$ORIGIN/../cuda` (`scripts/release-manifest.py --build`), which `LD_LIBRARY_PATH` cannot override | `larch` (metallib embedded) |
| GPU runtime | CUDA, cudart static | CUDA; HIP build compile-only in CI | Metal |
| Device telemetry | NVML (`nvml-wrapper`, loads `nvml.dll` at runtime) | NVML; HIP Unsupported | Unsupported (IOReport later) |
| Hang behaviour | TDR ≈ 2 s on display GPUs → device lost → exit 3 → supervisor | no TDR on compute GPUs → watchdog → exit 3 → supervisor | command-buffer timeout → device lost → exit 3 → supervisor |
| Supervision | Job Object kill-on-close + pipe EOF | pipe EOF | pipe EOF |
| Data dir | `%LOCALAPPDATA%\larch` | `$XDG_DATA_HOME/larch` | `~/Library/Application Support/larch` |
| mmap | `memmap2` | `memmap2` | `memmap2` + zero-copy Metal wrap |
| Sanitizer (row R11) | compute-sanitizer (memcheck, racecheck) | compute-sanitizer | Metal API validation + `-Zsanitizer=address` nightly run, or Xcode ASan |
| CI job | `windows-2022`: build CUDA (no GPU) + test on CPU reference | `ubuntu-24.04`: build CUDA + test on CPU reference; `hip` job: ROCm 10.0 container pinned by digest, hipify dry run + hipcc compile | `macos-26` arm64 (Xcode 26): build Metal + test on CPU reference |
| Driver floor | NVIDIA R580 branch or newer. Since CUDA 13.0 stopped bundling the Windows display driver, NVIDIA names only the branch; `larch info` prints the installed version | R580: ≥ 580.65.06 (CUDA 13.0 GA table) | macOS 26 (Metal 4) |

The R580 floor holds for binaries built with any 13.x toolkit, including 13.4, through CUDA's minor-version compatibility, as long as larch calls no driver feature newer than R580 (13.4's own features need R615). The binary ships SASS for 86, 89 and 120. Only the `120-virtual` PTX would need a driver as new as the building toolkit, and that matters only on a future GPU.

Vendor facts, checked 5 Oct 2026 against the vendors' own pages:

| Item | Fact |
|---|---|
| CUDA | Latest toolkit 13.4.2 (September 2026). 13.x covers Turing (7.5) to Blackwell, so sm_86, sm_89 and sm_120 are in; 13.0 dropped 5.x, 6.x, 7.0 and 7.2 |
| AMD on Linux: ROCm 10.0.0 (26 Aug 2026) | RDNA 4: RX 9070 XT, 9070 GRE, 9070 (gfx1201); RX 9060 XT, 9060, 9050 (gfx1200). RDNA 3: RX 7900 XTX, XT, GRE (gfx1100); RX 7800 XT, 7700 XT, 7700 (gfx1101); RX 7600 (gfx1102). RDNA 2: no consumer card; gfx1030 only on the Radeon PRO W6800 and V620 |
| AMD on Windows: HIP SDK 7.2 | RDNA 4: RX 9070 XT, 9070, 9070 GRE, 9060 XT, 9060. RDNA 3: RX 7900 XTX, 7900 XT, 7800 XT, 7700 XT, 7650 GRE, 7600 XT, 7600. RDNA 2: none (last listed as supported in 6.4.2). ROCm 10 now ships Windows from the same SDK as a tarball, without a per-card Windows list yet |
| AMD on macOS | not supported by ROCm or HIP |
| Base M5 | 10-core GPU with a neural accelerator in each core, 153 GB/s, 16, 24 or 32 GB. The 8B Q4_K_M (~4.9 GB) decode ceiling is ≈ 31 tok/s |
| M5 Pro (March 2026) | 16- or 20-core GPU, 307 GB/s, 24, 48 or 64 GB (64 GB only with the 20-core GPU) |
| M5 Max (March 2026) | 32-core GPU at 460 GB/s with 36 GB, or 40-core at 614 GB/s with 48, 64 or 128 GB. The handoff's 70B Q4_K_M row (~43 GB of weights) needs the 40-core Max with 64 or 128 GB |
| Rust | 1.97 was released on 9 July 2026 and is the pin in all three repos. Stable is now 1.99; there's no reason to move mid-sprint |

---

## 15. Correctness and testing architecture

| Layer | What | Where | Gate |
|---|---|---|---|
| Kernel unit | each op on the GPU backend vs the CPU reference, random shapes incl. edge sizes, per-dtype tolerances | `kernels/tests/*.cpp` (ctest) + `larch-kernels` Rust tests | every merge, on the developer's GPU; CPU-vs-CPU in CI |
| Tokenizer | larch vs llama.cpp tokenisation of the prompt set + 1 000 fuzz strings | `larch tokcheck` | exact match, before parity |
| **Model oracle** | teacher-forced logits vs llama.cpp on the same GGUF: `tools/llama-logits` (C++, llama.cpp pinned to a release tag) reads `prompts.jsonl`, tokenises with llama.cpp, runs the batch with logits for all positions (`llama_get_logits_ith`), and writes `{tokens, top64: [[id, logprob]…] per position}` as zstd JSONL. `larch parity` feeds the same tokens three ways and compares each run against the one dump (D16): `prefill` (one batch: the prompt-reading path), `decode` (one token per step through the production decode path: paged KV across steps, captured graphs, split-K attention, the mmvq decode kernels) and `chunked` (`prefill_chunk_tokens`-sized chunks with decode steps interleaved), plus `cached` (D14: the first half of the prompt runs as a session that generates one token and parks, the second half as its next turn, so positions from `n_kv` on are compared on the real reuse path); `--parallel 8` runs eight prompts in flight in decode mode (bucket padding, per-sequence block tables) and `--graphs off` is the control. `--graphs on` (the default) opts in to decode-graph replay where the device supports it. Each line reports the steps that replayed and the chunk steps that carried a decode row, both read from the ring. A requested path that did not run is reported as not run, never as a pass. Gate S2 (H10) needs `prefill`; `decode` joins at S3 (H16), as soon as the engine runs on CUDA, and `chunked` and `--parallel 8` with it; day 2 runs the whole matrix for every requirement-R4 model and once per PerformanceOnly knob at each of its bounds, because those knobs must be parity-neutral | dumps live outside git (`~/larch-oracle/<model-sha>/<backend>.jsonl.zst`) | top-1 ≥ 99 % on the positions whose reference top-1/top-2 gap exceeds the calibrated margin ("decisive"), and mean KL ≤ the calibrated bound over all positions, per backend and per mode. Per position, KL = Σ over the reference's top 64 of p (log p − log q), plus one tail term (1 − P) ln((1 − P)/(1 − Q)) for the mass outside them. Raw top-1 is reported beside the gate. |
| Calibration | The reference is llama.cpp's cuBLAS CUDA dump (`GGML_CUDA_FORCE_CUBLAS=ON`, `--batch 32 --ubatch 32`, so that a position does not move with prompt length), paired with each other llama.cpp run of the same GGUF and prompts: CPU, Metal, and cuBLAS at another micro-batch. The bound is 1.5 × the largest pair's p99 KL over positions. The margin is the largest reference top-1/top-2 gap at which any pair's top-1 differs. | recorded in `bench/oracle-bounds.json` with each run's identity (backend, device, batch, micro-batch, dump digest) | set once per model; recomputed by the same rule when a new pair (e.g. Metal) is added |
| Engine | scheduler/KV proptests (the six D14 invariants of 9.3 included) and the six D14 session tests on the CPU reference: `one_token_then_continue` (a session generates one token and stops, by length, by EOS and by a stop token in turn, then continues: the cached run's logits equal a fresh prefill's within the bound below and `cached_tokens` equals the parked `n_kv`, never one more), `cancel_then_continue` (cancel mid-answer: the history holds the partial text, nothing is parked, the next turn re-prefills and is correct), `reset_race` (reset while a turn is active, then a new turn: the late completion changes nothing and nothing stays parked under the old id), `partial_block` (`n_kv` not a multiple of 16 through park, match and reclaim), `reclaim_keeps_prefix` (an 8k-token parked conversation under a 256-token request loses 16 trailing blocks and its next turn reports `cached_tokens` = 8192 − 256) and `template_roundtrip` (Llama 3.1 and Qwen2.5 templates: after a generated answer, turn 2 reports `cached_tokens == tokens_in_kv()`; a constructed join that re-tokenises differently falls back to re-prefill with correct logits), stop/cancel/logprob tests on the CPU reference with a tiny GGUF (generated by a test helper: 2 layers, 256 vocab, random weights, every supported quant) | `larch-engine/tests` | every merge, all 3 OS in CI |
| Records | ring loom + proptest; layout size asserts; knob validation unit tests (row R8) | `larch-records/tests` | every merge |
| Fuzz | `gguf` (row R9), `tokenizer`, `knob_request` | `fuzz/`, overnight 2 h per target | no crash, clean errors |
| Rule port | 1:1 tests + differential parity | strided CI | all green; differential 10 000 cases per rule |
| gait | every legal and illegal phase transition, stop conditions and packets against a fake `KnobControl` + synthetic ring; journal chain verification (a tampered line is refused; a partial last line is dropped); recovery with the same `engine_id` (undoes) and a new one (no-op) | gait CI | green |
| Reliability matrix | R1-R13 (table below) | `bench/` scripts on the real machines | the plan's gates |

Reliability matrix → mechanism:

| Row | Mechanism in this design |
|---|---|
| R1 parity | section 15 oracle, per backend (CUDA on the 5070 Ti, 4060, 3050; Metal on the M5) |
| R2 load 1/4/8, 10 min, 0 failures | `larch bench --mode serve` |
| R3 100 mid-stream disconnects | bench `--cancel-rate`; KV used-blocks returns to 0 (asserted from the ring; a cancelled sequence is never parked, and the run uses W4 with `keep_kv` off) |
| R4 kill under load | `LARCH_FAULT=device_lost/panic` and a plain kill, both under `larch supervise`: clean `DeviceLost`/`Internal` errors to callers, exit 3, process start → Ready < 2 s (`recovery_ms` in the ring); strided and gait come back with the process |
| R5 kill ×20 | supervise script; zombies and free VRAM checked before and after |
| R6 over-length | `PromptTooLong` |
| R7 over-admission | `Overloaded`; KV is preallocated, so no OOM path exists at runtime |
| R8 knob rejects | `larch-records` + `larch-engine` unit tests |
| R9 fuzzed GGUF | `fuzz/gguf` |
| R10 GPU hang/TDR | `LARCH_FAULT=hang` (an infinite-loop debug kernel) → TDR on Windows / watchdog on Linux → exit 3 → supervised restart to Ready < 2 s |
| R11 sanitizer | compute-sanitizer over 50 requests; Metal validation + ASan |
| R12 12 h soak | `larch supervise -- larch bench --mode serve --duration 43200 --cancel-rate 0.02`; memory and tok/s slopes from the ring |
| R13 strided + gait end to end | the section 12.7 demo path + `tests/observe.rs` fixtures firing the same rules the Python fires |

---

## 16. Benchmark harness (section 6 of the handoff, refined, not softened)

- **Prompt set:** `bench/prompts/chat.jsonl`. 200 chat prompts, 50 to 2 000 tokens, with a target output of 128 to 512 tokens. Committed on day 1 and never edited afterwards.
- **Workloads (D14):** `larch bench --workload <name> [--replay scripted|generated]`; `--mode serve` is W4 and `--mode batched` is W6. Every workload is a frozen file under `bench/prompts/` with exact token lengths (`ignore_eos`, so answers are exactly their target length), checked with `larch tokcheck` on day 1. Conversations carry their assistant turns pre-written. With `--replay scripted`, the comparison mode, the history continues with the scripted text after each timed answer, so all three engines see byte-identical prompts on every turn and the prompt-token hash check applies per turn (a turn's prefill is then the previous scripted answer plus the new message). With `--replay generated`, larch only, the engine's own answers continue the conversation, which validates the real session and KV bookkeeping (EOS handling, the last evaluated token, the template round trip) and asserts `cached_tokens == tokens_in_kv()` before every turn. Cached-versus-fresh correctness is `larch parity --mode cached` and the tests in 15, not the benchmark.

| Workload | Definition (exact) | Context | Must win vs both baselines (NVIDIA; Mac in brackets) |
|---|---|---|---|
| **W1 chat** | 1 conversation, 12 turns; user messages 40 to 300 tokens, answers 100 to 400 tokens | 8192 | TTFT at turns 2 to 12 lower, reported against both the new-message length and the retained-context length; TPOT within 5 % (single-stream parity) [same] |
| **W2 chats** | 4 and 8 concurrent conversations, 8 turns each, the W1 lengths, closed loop | 8192 each | TTFT p99 and TPOT p99 lower at 4 and 8; aggregate tok/s higher [4 concurrent: same] |
| **W3 agent** | 1 loop of 20 calls, and 3 loops at once: a 4096-token prefix; each call appends a 130-token tool result and gets a 120-token answer; peak 4096 + 20 × 250 = 9096 tokens plus template overhead (about 10 per message), under 9.6k | 16384 on every engine (`-c 16384`, `num_ctx 16384`), so nothing overflows; overflow is tested on its own (10.1.1) | wall time lower; per-call TTFT lower [1 loop] |
| **W4 batch** | 200 independent single-turn prompts (`chat.jsonl`), closed loop at 8 and 16: today's serve mode, `keep_kv` off | 8192 | aggregate tok/s ≥ 15 % higher at 8 and 16 (unchanged) [not run] |
| **W5 longprompt** | 2048- and 4096-token prompt, 128-token answer, single stream | 8192 | prefill tok/s higher or equal and TTFT lower or equal [the existing Mac 2k row] |
| **W6 fixed** | the `llama-batched-bench` shape mirror, pp 512, tg 128, npl 1/4/8/16 (`--mode batched`) | 8192 | the honesty anchor: within 3 % of `llama-batched-bench` [same] |
| **W7 tune** | W2 at 8 with gait on (F6 of the plan) | 8192 | 0 failed requests, pause < 100 ms, a change visible within one step [hitless] |

  Every existing row of the handoff's section 6 stays; W1, W2 and W3 enter its tables when the maintainer confirms them. Cached history avoids evaluating most historical tokens again; the new tokens still attend to the whole history, which is why W1 reports TTFT against both lengths rather than claiming it depends on the new message alone.
- **larch:** `larch bench --mode serve` runs closed-loop clients at N concurrent over the prompt set (file order, greedy, `ignore_eos`, so each request generates exactly its target length) and reports aggregate tok/s, TTFT/TPOT p50/p99 and e2e. `--mode batched --pp 512 --tg 128 --npl 1,4,8,16` mirrors `llama-batched-bench`'s shape exactly for an apples-to-apples aggregate row.
- **llama.cpp**, at its latest release with the best flags found by an honest search (`-fa`, `-ub`, `-b`, `-ngl 99`, `-ctk/-ctv`, threads), all in-process:
  - `llama-bench` for single-stream pp/tg;
  - `llama-batched-bench` for the aggregate at N;
  - `tools/llama-serve-bench` (D13) for TTFT and TPOT at N concurrent, the serve-mode aggregate, and the Mac's single-user TTFT. Neither tool above measures TTFT or TPOT. `llama-parallel` prints only each request's total time and picks prompts at random in its own format, so it can't replay `chat.jsonl` (checked 5 Oct).
    - **Build:** C++ on llama.cpp's C API and its `common` library, against a released llama.cpp tag: the latest release for a baseline, while `tools/llama-logits` and the oracle dumps keep their own pin. CUDA on Windows and WSL; Metal on the Mac.
    - **Inputs:** the GGUF; `chat.jsonl`; `--concurrency N`; the best flags from `llama-flags.csv` (`-fa`, `-ub`, `-b`, `-ngl 99`, `-ctk/-ctv`, threads); a context of N × (the longest prompt + its output).
    - **Loop:** llama-server's `update_slots` policy without the HTTP layer. Each iteration first adds one token per decoding client to the `llama_batch`, then prompt tokens of admitted clients up to `-b`; a prompt that doesn't fit continues next iteration. Then one `llama_decode`. This is the stronger of llama.cpp's two loops (`llama-parallel` adds whole prompts at once), so nobody can say we picked the weak one.
    - **Workload, identical to `larch bench --mode serve`:**
      - N closed-loop clients take prompts in file order;
      - prompts are rendered with the GGUF's own Jinja template through `common_chat_templates_apply` (what `llama-server --jinja` uses);
      - sampling is greedy, and each request generates exactly its target length with EOS ignored;
      - prompt prefixes are reused only in W1 to W3, on every engine: the driver keeps one slot per conversation and reuses the longest common token prefix after a full re-tokenise (`get_common_prefix`, `keep_first`: llama-server's own behaviour), with slot selection by prefix similarity at llama.cpp's default 0.10 then LRU; its host-RAM prompt cache never triggers and is not replicated. W4 to W6 reuse nothing on either side.
    - **Output:** the `larch bench --json` schema.
      - Per request: prompt tokens, a sha256 of the prompt token IDs, generated tokens, `t_submit`, `t_first`, `t_done`. TTFT = first − submit; TPOT = (done − first) / (generated − 1).
      - Summary: aggregate tok/s, TTFT and TPOT p50/p99, e2e.
    - **Honesty checks:**
      - `--fixed-shape --pp 512 --tg 128 --npl 1,4,8,16` must reproduce `llama-batched-bench`'s aggregate within 3 % at every shape before any number from the driver is published.
      - The comparison script refuses two runs if any request's prompt-token hash differs between the engines. A mismatch is a larch template or tokenizer bug, fixed before publishing.
      - One W1 conversation run through a real `llama-server` over HTTP must report the same per-turn `cache_n` and prompt-token counts as the driver.
      - `--replay generated` must run W1 to W3 clean on larch (every turn `cached_tokens == tokens_in_kv()`, no error) before a scripted number is published.
- **Ollama** (D15), its latest release, as the second baseline on every row: the same GGUF imported with a `Modelfile` (`FROM <gguf>`), `num_ctx` = larch's `max_context` (16384 for W3), `num_parallel` = N, flash attention on (`OLLAMA_FLASH_ATTENTION=1`), f16 KV, `keep_alive = -1` so the model is never unloaded, and its defaults otherwise. Driven by `larch bench --engine ollama --url` over its streaming chat API with the same prompt files, the same closed-loop clients and the same JSON; TTFT is the first streamed token. Its numbers include its HTTP and runner overhead, which is what its users get, and go in their own column next to llama.cpp's in-process numbers, never in place of them. The fixed-shape mirror (`--mode batched`) does not apply to it.
- **Which number decides each must-win row:** every row must beat both baselines (D15); the Ollama column uses the same prompt files, so the prompt-token hash check applies to it too.
  - aggregate tok/s at 8 and 16 is `larch bench --mode batched` against `llama-batched-bench`, as the handoff names, with both engines' serve-mode aggregate reported beside it;
  - TTFT p99 and TPOT p99 at 8 are `larch bench --mode serve` against `llama-serve-bench`;
  - single-stream is `larch bench` at concurrency 1 against `llama-bench`;
  - W1 to W3 (D14) are `larch bench --workload chat|chats|agent --replay scripted` against `llama-serve-bench` replaying the same files with its slot cache on, and against Ollama with the same files over HTTP.
- **Machines and columns:**
  - Windows native on the 5070 Ti (headline);
  - Linux (WSL, D8) on the 5070 Ti;
  - Windows native on the 4060 (sm_89, 8 GB);
  - the 3050 (sm_86, 8 GB): 3B and 8B Q4_K_M single-stream + parity;
  - the base M5, 24 GB (Metal, 153 GB/s).

  AMD and M5 Pro/Max are measured only if hardware turns up through the day-1 network ask; otherwise they read "compiled, not measured" and "designed for, not measured".
- **Scoreboard:** `bench/results/history.csv` gets one line per merge to `dev` (commit, machine, model, agg tok/s at 8, TTFT p99, TPOT p99, single-stream tok/s). A merge that loses > 3 % is reverted unless it's a correctness fix. Numbers are never edited after publication.

---

## 17. Requirements traceability

| Req | Where it's met |
|---|---|
| R1 native Win/mac/Linux | 2, 6, 10.3, 14; CI on all three every merge |
| R2 backends and generations | 6.2 (sm_86/89/120 fat binary, derived launches, fit 8-24 GB), 6.4 (one metallib, derived dispatch, fit 16-128 GB), 6.3 (HIP from the same sources, wave32-safe, CI compile) |
| R3 beats llama.cpp | 9.3 (continuous batching + chunked prefill), 9.3 to 9.4 and 10.1.1 (parked sessions, D14), 9.4 (paged KV), 6.2 (graphs, flash-decoding, mmvq ≥ 80 % BW), 16 (two baselines, D15; workloads W1 to W7) |
| R4 complete enough to use | 8 (archs, quants, templates), 9.1-9.7 (sampling params, stops, logprobs, cancel, clean errors), 10 (library, chat, bench, knob), 10.1.1 (a session keeps its KV) |
| R5 live tuning without downtime | 4.5-4.6, 9.8-9.9, 9.13 |
| R6 reliability | 9.11-9.12, 10.3, 3 (thread panic boundaries) |
| R7 strided/gait in-process, direct | 3, 4, 11, 12 |
| R8 simple | 7 crates, 1 header, 1 ring, 8 record kinds, 16 knobs, sync API, no async runtime |
| R9 foolproof | 15 |
| R-img image paste (added 8 Nov 2026, PRs #457-#463) | 8 (`vision`: InternVL's projector), 9.2 (image spans), 6.2 (`larch_attn_full` on tensor cores), 15 (`image_oracle` against llama.cpp b11458, with mutation runs) |

---

## 18. Deliberately not in this design (handoff section 11, plus D2/D11)

HTTP/OpenAI server; pipes, sockets or shared memory between anything; cuprum routing, failover and specialised binaries (cavefish); hitless model swap via a shadow worker; the cache work that builds on D14's blocks: cross-session adoption with a policy that weighs reused tokens against the cached tokens it discards, a parked cap or TTL with local-chat defaults, the block-hash prefix cache shared across conversations (refcounts, copy-on-write for partial blocks, concurrent fan-out, eviction that protects every reader, and cache keys that include computation-affecting state such as LoRA ids and a cache salt, as vLLM's do), host-RAM spill of parked blocks, and incremental tokenisation with boundary tests; the block tables are the foundation, and keeping the kernels and the C ABI unchanged is the goal to verify then, not a guarantee; Vulkan; CPU serving backend; OpenTelemetry export; tool calling, embeddings, quants beyond those listed in 8; vision beyond InternVL's projector (moved into scope 8 Nov 2026: R-img in 17), so no Qwen3.5 encoder, no vision ops on Metal or HIP (they answer UNSUPPORTED) and no on-disk cache of encoded tiles; the rest of the strided rules (r04, r05, r07 and the reserved ones) and the parsers; gait's repo-modification job; a parent/GPU-worker split (deferred; D2's single restart path replaced the in-process reset).
