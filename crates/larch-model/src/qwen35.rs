//! Qwen 3.5 (`qwen35`, `qwen35moe`), as llama.cpp builds it (`src/models/qwen35moe.cpp`), on the
//! [`Transformer`]'s loader and linear routes:
//! - Gated DeltaNet layers (three in four by default) in place of attention: one projection to the
//!   conv channels (q, k, v), one to the output gate z and two to the per-head decay input alpha and
//!   write strength beta; `larch_gdn_conv`, `larch_gdn`; then an RMS norm per head times silu(z)
//!   and the out projection. The state lives in [`States`], one slot per sequence, not in the KV pool.
//! - Full attention whose Q projection holds a [q, gate] pair per head: two small gemms with 0/1
//!   matrices split them exactly, and `sigmoid(gate)` scales the attention output before `wo`.
//! - qwen35moe's shared expert: a dense FFN beside the routed experts, scaled by the sigmoid of a
//!   one-column gate.

use super::*;
use crate::StateBatch;
use crate::gguf::Value;
use larch_kernels::GdnState;

/// A loader lookup: a tensor by name and shape.
type Find<'a> = &'a dyn Fn(&str, &[i64]) -> Result<W, LoadError>;

/// Where the recurrent layers are, their geometry and the shared expert's width.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Hybrid {
    /// Per layer: Gated DeltaNet (true) or full attention.
    pub recurrent: Vec<bool>,
    pub kernel: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub head_dim: u32,
    /// qwen35moe's shared expert width; 0 in the dense model.
    pub shared_ffn: u32,
}

impl Hybrid {
    /// The hybrid layout of a qwen35 or qwen35moe GGUF, None for any other arch. `BadGguf(KEY, 0)`
    /// for a missing key; `UnsupportedArch` for k and v heads of different widths, which
    /// `larch_gdn` does not take.
    pub fn from_gguf(g: &Gguf, c: &ModelConfig) -> Result<Option<Hybrid>, LoadError> {
        let name = match c.arch {
            Arch::Qwen35 => "qwen35",
            Arch::Qwen35Moe => "qwen35moe",
            _ => return Ok(None),
        };
        let bad = LoadError::BadGguf(code::KEY, 0);
        let get = |k: &str| g.get(&format!("{name}.{k}"));
        let u = |k: &str| {
            get(k)
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or(bad))
                .transpose()
        };
        let need = |k: &str| u(k)?.filter(|&v| v > 0).ok_or(bad);
        let (kernel, state) = (need("ssm.conv_kernel")?, need("ssm.state_size")?);
        let (inner, k_heads) = (need("ssm.inner_size")?, need("ssm.group_count")?);
        let v_heads = need("ssm.time_step_rank")?;
        // llama.cpp: k heads are state_size wide, v heads inner_size / time_step_rank.
        if kernel < 2 || !v_heads.is_multiple_of(k_heads) {
            return Err(bad);
        }
        if v_heads.checked_mul(state) != Some(inner) {
            return Err(LoadError::UnsupportedArch);
        }
        // The plan sizes a slot's state from these keys before any tensor is checked
        // (`Transformer::hybrid_bytes`), so a crafted header fails here, not in that product.
        let channels = (2 * u128::from(k_heads) + u128::from(v_heads)) * u128::from(state);
        let slot =
            4 * u128::from(inner) * u128::from(state) + 2 * u128::from(kernel - 1) * channels;
        if u64::try_from(slot).is_err() {
            return Err(bad);
        }
        let n = c.n_layers as usize;
        let recurrent = match get("attention.recurrent_layers").and_then(Value::as_arr) {
            Some(a) => {
                let flags: Option<Vec<bool>> = a
                    .iter()
                    .map(|v| match v {
                        Value::Bool(b) => Some(*b),
                        v => v.as_u64().map(|x| x != 0),
                    })
                    .collect();
                // The array may also cover the multi-token-prediction blocks past the trunk.
                let mut flags = flags.filter(|f| f.len() >= n).ok_or(bad)?;
                flags.truncate(n);
                flags
            }
            None => {
                let every = u("full_attention_interval")?.unwrap_or(4).max(1);
                (0..c.n_layers).map(|i| (i + 1) % every != 0).collect()
            }
        };
        let shared_ffn = match c.arch {
            Arch::Qwen35Moe => u("expert_shared_feed_forward_length")?
                .filter(|&v| v > 0)
                .unwrap_or(c.ffn),
            _ => 0,
        };
        Ok(Some(Hybrid {
            recurrent,
            kernel,
            k_heads,
            v_heads,
            head_dim: state,
            shared_ffn,
        }))
    }

    /// Conv channels: q, k (k_heads each) and v (v_heads), head_dim wide.
    pub fn channels(&self) -> i64 {
        (2 * i64::from(self.k_heads) + i64::from(self.v_heads)) * i64::from(self.head_dim)
    }

    /// Width of v, z and the layer's output before `ssm_out`.
    pub fn v_dim(&self) -> i64 {
        i64::from(self.v_heads) * i64::from(self.head_dim)
    }

    /// Gated DeltaNet layers.
    pub fn gdn_layers(&self) -> u32 {
        self.recurrent.iter().filter(|&&r| r).count() as u32
    }

    /// Bytes of one slot of one layer: conv (f16) and ssm (f32).
    fn slot_bytes(&self) -> (u64, u64) {
        let hd = u64::from(self.head_dim);
        (
            DType::F16.bytes(u64::from(self.kernel - 1) * self.channels() as u64),
            DType::F32.bytes(u64::from(self.v_heads) * hd * hd),
        )
    }
}

/// One Gated DeltaNet layer's weights. `alpha` and `beta` are quantized, or f16 side copies when the
/// GGUF keeps them in float.
#[derive(Clone, Copy)]
pub(super) struct Gdn {
    pub qkv: W,
    pub z: W,
    pub alpha: W,
    pub beta: W,
    /// f32 `[channels, kernel]`.
    pub conv: W,
    pub dt: W,
    pub a: W,
    pub norm: W,
    pub out: W,
    /// This layer among the Gated DeltaNet layers: its part of [`States`].
    pub layer: u32,
}

/// qwen35moe's shared expert: dense gate, up and down, and its f16 side gate `[1, d]`.
#[derive(Clone, Copy)]
pub(super) struct Shared {
    pub gate_inp: W,
    pub gate: W,
    pub up: W,
    pub down: W,
}

impl Gdn {
    /// The weights of layer `i`, the `layer`-th recurrent one. `need` takes quant matrices and f32
    /// vectors, `f32` an f32 matrix, `proj` a matrix in either form (see [`Gdn`]).
    pub fn load(
        h: &Hybrid,
        d: i64,
        i: u32,
        layer: u32,
        need: Find,
        f32: Find,
        proj: Find,
    ) -> Result<Gdn, LoadError> {
        let n = |s: &str| format!("blk.{i}.{s}");
        let (ch, vd, vh) = (h.channels(), h.v_dim(), i64::from(h.v_heads));
        Ok(Gdn {
            qkv: need(&n("attn_qkv.weight"), &[ch, d])?,
            z: need(&n("attn_gate.weight"), &[vd, d])?,
            alpha: proj(&n("ssm_alpha.weight"), &[vh, d])?,
            beta: proj(&n("ssm_beta.weight"), &[vh, d])?,
            conv: f32(&n("ssm_conv1d.weight"), &[ch, i64::from(h.kernel)])?,
            dt: need(&n("ssm_dt.bias"), &[vh])?,
            a: need(&n("ssm_a"), &[vh])?,
            norm: need(&n("ssm_norm.weight"), &[i64::from(h.head_dim)])?,
            out: need(&n("ssm_out.weight"), &[d, vd])?,
            layer,
        })
    }
}

/// The hybrid layers' activations for up to `rows` rows, f16 unless noted.
pub(super) struct Scratch {
    /// The conv input, `[rows, channels]`, then the conv output in `conv`.
    qkv: Buffer,
    conv: Buffer,
    /// `[rows, v_dim]`: z, the delta rule's output, and the gated norm of it.
    z: Buffer,
    o: Buffer,
    gated: Buffer,
    /// f32 alpha `[rows, v_heads]`, then beta.
    ab: Buffer,
    /// The attention Q projection `[rows, heads, 2 head_dim]`, and its gates `[rows, heads * head_dim]`.
    qfull: Buffer,
    gate: Buffer,
    /// The 0/1 matrices `[head_dim, 2 head_dim]` that pick q (the first) and the gate (the second)
    /// out of a head's pair.
    sel: Buffer,
    /// The shared expert's gate, `[rows, 1]`.
    sg: Buffer,
}

impl Scratch {
    /// Each buffer's bytes for `rows` rows, in field order.
    fn sizes(h: &Hybrid, c: &ModelConfig, rows: i64) -> [u64; 10] {
        let (hd, qd) = (i64::from(c.head_dim), i64::from(c.n_heads * c.head_dim));
        let f16 = |cols: i64| DType::F16.bytes((rows * cols) as u64);
        [
            f16(h.channels()),
            f16(h.channels()),
            f16(h.v_dim()),
            f16(h.v_dim()),
            f16(h.v_dim()),
            DType::F32.bytes((rows * 2 * i64::from(h.v_heads)) as u64),
            f16(2 * qd),
            f16(qd),
            DType::F16.bytes((2 * hd * 2 * hd) as u64),
            f16(1),
        ]
    }

    pub fn alloc(
        h: &Hybrid,
        c: &ModelConfig,
        rows: i64,
        s: &Stream,
        alloc: &dyn Fn(u64) -> Result<Buffer, LoadError>,
    ) -> Result<Scratch, LoadError> {
        let hd = c.head_dim as usize;
        let mut sel = vec![0u16; 2 * hd * 2 * hd];
        for i in 0..hd {
            sel[i * 2 * hd + i] = 0x3c00; // 1.0: q's element i
            sel[(hd + i) * 2 * hd + hd + i] = 0x3c00; // the gate's
        }
        let [qkv, conv, z, o, gated, ab, qfull, gate, sel_buf, sg] =
            Scratch::sizes(h, c, rows).map(alloc);
        let sc = Scratch {
            qkv: qkv?,
            conv: conv?,
            z: z?,
            o: o?,
            gated: gated?,
            ab: ab?,
            qfull: qfull?,
            gate: gate?,
            sel: sel_buf?,
            sg: sg?,
        };
        s.upload(&sc.sel, 0, &sel).map_err(backend)?;
        Ok(sc)
    }
}

/// Every sequence's recurrent state: [`Hybrid::gdn_layers`] layers of `phys` slots, the first
/// `slots` the sequences' own, the rest spares. Sequence slot `l` lives in physical slot `map[l]`:
/// its own, or after a rewind to its check's first row the spare that holds that row, until a
/// step that is not its check moves it home (that step's last row snapshots there), or a copy onto
/// itself does for a step it sits out.
pub(super) struct States {
    conv: Buffer,
    ssm: Buffer,
    slots: u32,
    phys: u32,
    map: Vec<u32>,
    /// Spares holding nothing.
    free: Vec<u32>,
    /// The begun checks, one a sequence slot at most: several sequences check in one step.
    checks: Vec<Check>,
    /// A guessing load's record of its check steps' layer inputs; None without spares.
    replay: Option<Replay>,
    /// A routed step's i32 physical slots (`[seqs]` at 0) and snapshot slots (`[rows]` at
    /// `4 * max_rows`), which the Gated DeltaNet layers read in place of the batch's slots. A
    /// guessing load's capture always reads them, and [`Model::route_table`] writes them before
    /// every replay.
    route: Buffer,
    /// This step's routing ([`Transformer::route_state`]): Some(snapshots taken) when routed.
    step: Option<bool>,
}

/// A begun check: `rows` rows of sequence slot `slot`, from row `row0` of its step. Its first row
/// snapshots into the spare `snap` or, while the sequence lives in a spare, into its own slot
/// (None); `ran` once its step was routed.
struct Check {
    slot: u32,
    rows: u32,
    snap: Option<u32>,
    row0: u32,
    ran: bool,
}

/// The most rows of a step [`Replay`] records: a check step's checks and one row for every other
/// sequence ([`Transformer::check_begin`] refuses a check past it).
const REPLAY_ROWS: i64 = 64;

/// A guessing load's record of each Gated DeltaNet layer's inputs in its last step of up to
/// `REPLAY_ROWS` rows (the conv input `[rows, channels]` f16, then alpha and beta `[rows, v_heads]`
/// f32), which the layers write in place of the scratch, so recording costs no op; then three i32
/// tables of `REPLAY_ROWS`, indexed by a check's rows: each replay's physical slot and row count
/// (at the check's first row) and its snapshot table. [`Transformer::check_end`] replays a check's
/// kept rows past its first from it.
struct Replay {
    buf: Buffer,
}

impl Replay {
    fn layer_bytes(h: &Hybrid) -> u64 {
        let (ch, vh) = (h.channels(), i64::from(h.v_heads));
        let b = DType::F16.bytes((REPLAY_ROWS * ch) as u64)
            + 2 * DType::F32.bytes((REPLAY_ROWS * vh) as u64);
        b.next_multiple_of(256)
    }

    /// Where the i32 tables start, past every layer's inputs.
    fn args_at(h: &Hybrid) -> u64 {
        Self::layer_bytes(h) * u64::from(h.gdn_layers())
    }

    /// The device bytes of a load's record.
    fn bytes(h: &Hybrid) -> u64 {
        Self::args_at(h) + DType::I32.bytes(3 * REPLAY_ROWS as u64)
    }

    /// Layer `layer`'s conv input, alpha and beta at rows `from..from + n` of the record.
    fn rows(&self, h: &Hybrid, layer: u32, from: i64, n: i64) -> Result<[Tensor<'_>; 3], KError> {
        let (ch, vh) = (h.channels(), i64::from(h.v_heads));
        let base = u64::from(layer) * Self::layer_bytes(h);
        let alpha = base + DType::F16.bytes((REPLAY_ROWS * ch) as u64);
        let beta = alpha + DType::F32.bytes((REPLAY_ROWS * vh) as u64);
        let row = |at: u64| at + DType::F32.bytes((from * vh) as u64);
        let qkv = base + DType::F16.bytes((from * ch) as u64);
        Ok([
            Tensor::new(&self.buf, qkv, DType::F16, &[n, ch])?,
            Tensor::new(&self.buf, row(alpha), DType::F32, &[n, vh])?,
            Tensor::new(&self.buf, row(beta), DType::F32, &[n, vh])?,
        ])
    }

    /// Byte `i` of the i32 tables.
    fn arg(h: &Hybrid, i: i64) -> u64 {
        Self::args_at(h) + DType::I32.bytes(i as u64)
    }
}

/// The state buffers of `st` as the Gated DeltaNet ops of `h` take them.
fn gdn_state<'a>(h: &Hybrid, st: &'a States) -> GdnState<'a> {
    GdnState {
        conv: &st.conv,
        ssm: &st.ssm,
        n_slots: st.phys,
        n_layers: h.gdn_layers(),
        kernel: h.kernel,
        k_heads: h.k_heads,
        v_heads: h.v_heads,
        head_dim: h.head_dim,
    }
}

impl States {
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// The spares no check, snapshot or rewound sequence holds: a check of a sequence in its own
    /// slot takes one.
    pub fn spares_free(&self) -> u32 {
        self.free.len() as u32
    }

    /// Drops any check on `slot` and moves the slot home, freeing its spares: before the slot is
    /// overwritten.
    fn release(&mut self, slot: u32) {
        if let Some(i) = self.checks.iter().position(|c| c.slot == slot) {
            self.free.extend(self.checks.swap_remove(i).snap);
        }
        let p = std::mem::replace(&mut self.map[slot as usize], slot);
        if p != slot {
            self.free.push(p);
        }
    }
}

impl Transformer {
    /// The most spare state slots beside the sequences' own ([`Model::begin_check`]): a check of a
    /// sequence in its own slot snapshots its first row into one, and the sequence lives there
    /// after a rewind until a step moves it home. Only a load that will guess has them
    /// ([`Model::alloc_checks`]).
    // ponytail: 4 lets 4 chats check in one step; raise it when a bench at 8 chats shows checks
    // refused for want of a spare.
    pub const CHECK_SLOTS: u32 = 4;

    /// The most rows a check may have (its last token and 7 guesses): a rewind replays its kept
    /// rows past the first from the step's record.
    pub const CHECK_ROWS: u32 = 8;

    /// The device bytes [`Model::alloc_checks`] adds to a load of `g` for `spares` spares (at most
    /// [`Transformer::CHECK_SLOTS`]) and its check steps' record; 0 for any other arch or no spare.
    pub fn check_bytes(g: &Gguf, c: &ModelConfig, spares: u32) -> Result<u64, LoadError> {
        let Some(h) = Hybrid::from_gguf(g, c)? else {
            return Ok(0);
        };
        let spares = spares.min(Transformer::CHECK_SLOTS);
        if spares == 0 {
            return Ok(0);
        }
        let (conv, ssm) = h.slot_bytes();
        Ok((conv + ssm)
            .saturating_mul(u64::from(h.gdn_layers()))
            .saturating_mul(u64::from(spares))
            .saturating_add(Replay::bytes(&h)))
    }

    /// The device bytes a load of `g` for `rows` rows takes for its hybrid layers, beyond the
    /// weights and what every model of `c`'s shape takes (the engine counts one f16 copy of a
    /// `[max(qd, d), d]` matrix for the dequant path): their scratch, `slots` sequences' state
    /// with its routing ([`Transformer::alloc_state`]), the f16 side copies of
    /// float alpha and beta and of the shared-expert gates, and the larger dequant copy that the
    /// gated Q or the conv input needs. 0 for any other arch.
    pub fn hybrid_bytes(
        g: &Gguf,
        c: &ModelConfig,
        rows: u32,
        slots: u32,
    ) -> Result<u64, LoadError> {
        let Some(h) = Hybrid::from_gguf(g, c)? else {
            return Ok(0);
        };
        let (conv, ssm) = h.slot_bytes();
        let state = (conv + ssm)
            .saturating_mul(u64::from(h.gdn_layers()))
            .saturating_mul(u64::from(slots));
        let route = DType::I32.bytes(2 * u64::from(rows));
        let scratch: u64 = Scratch::sizes(&h, c, i64::from(rows)).iter().sum();
        // As the loader's `proj` and `side` take them.
        let float = |t: &&gguf::TensorInfo| {
            gguf::dtype(t.ggml_type).is_ok_and(|t| matches!(t, DType::F32 | DType::F16))
        };
        let mut side = 0u64;
        for i in 0..c.n_layers {
            let t = |s: &str| g.tensor(&format!("blk.{i}.{s}"));
            let gdn =
                h.recurrent[i as usize].then(|| [t("ssm_alpha.weight"), t("ssm_beta.weight")]);
            let gdn = gdn.into_iter().flatten().flatten().filter(float);
            let shared = t("ffn_gate_inp_shexp.weight").filter(|_| h.shared_ffn > 0);
            for t in gdn.chain(shared) {
                let n = t.dims.iter().fold(1u64, |a, &d| a.saturating_mul(d));
                side = side.saturating_add(n.saturating_mul(2));
            }
        }
        let (d, qd) = (
            u64::from(c.d_model),
            u64::from(c.n_heads) * u64::from(c.head_dim),
        );
        let widest = (2 * qd).max(h.channels() as u64);
        let dq = 2 * d * widest.saturating_sub(qd.max(d));
        Ok(state.saturating_add(scratch + side + dq + route))
    }

    fn hybrid(&self) -> Result<(&Hybrid, &Scratch), KError> {
        match (&self.hybrid, &self.hy) {
            (Some(h), Some(sc)) => Ok((h, sc)),
            _ => Err(KError::InvalidArg),
        }
    }

    /// State for `slots` sequences, zeroed, with no check spares ([`Model::alloc_checks`]); replaces
    /// any earlier state. `DoesNotFit` past the device.
    pub fn alloc_state(&mut self, s: &Stream, slots: u32) -> Result<(), LoadError> {
        self.alloc_slots(s, slots, 0)
    }

    /// [`Model::alloc_checks`]: the state again, zeroed, with `spares` spares, at most
    /// [`Transformer::CHECK_SLOTS`].
    pub(super) fn alloc_check_slots(&mut self, s: &Stream, spares: u32) -> Result<(), LoadError> {
        let spares = spares.min(Transformer::CHECK_SLOTS);
        match self.states.as_ref().map(States::slots) {
            Some(slots) => self.alloc_slots(s, slots, spares),
            None => Ok(()),
        }
    }

    fn alloc_slots(&mut self, s: &Stream, slots: u32, spares: u32) -> Result<(), LoadError> {
        let Some(h) = &self.hybrid else {
            return Err(LoadError::UnsupportedArch);
        };
        self.states = None;
        let (conv, ssm) = h.slot_bytes();
        let phys = slots.saturating_add(spares);
        let n = u64::from(h.gdn_layers()) * u64::from(phys);
        let dev = s.device();
        let alloc = |b: u64| Buffer::alloc(dev, b.max(1)).map_err(|e| fit_err(dev, b, e));
        let replay = match spares {
            0 => None,
            _ => Some(Replay {
                buf: alloc(Replay::bytes(h))?,
            }),
        };
        let st = States {
            conv: alloc(conv * n)?,
            ssm: alloc(ssm * n)?,
            slots,
            phys,
            map: (0..slots).collect(),
            free: (slots..phys).collect(),
            checks: Vec::new(),
            replay,
            route: alloc(DType::I32.bytes(2 * self.max_rows as u64))?,
            step: None,
        };
        self.states = Some(st);
        for slot in 0..slots {
            self.zero_slot(s, slot).map_err(backend)?;
        }
        Ok(())
    }

    /// [`Model::grow_state`]: the most sequence slots up to `want` whose new buffers fit `room`
    /// beside the old ones, every slot's state, snapshot and spare kept; the slots it now holds.
    /// `InvalidArg` while a check is begun.
    // ponytail: old and new buffers coexist (no data lost on an OOM), so a grow needs room for the
    // whole new state; download, free and upload when a 5th chat waits beside room for one slot.
    pub(super) fn grow_slots(&mut self, s: &Stream, want: u32, room: u64) -> Result<u32, KError> {
        let (Some(h), Some(st)) = (&self.hybrid, &mut self.states) else {
            return Err(KError::InvalidArg);
        };
        if !st.checks.is_empty() {
            return Err(KError::InvalidArg);
        }
        let (conv, ssm) = h.slot_bytes();
        let layers = u64::from(h.gdn_layers());
        let spares = st.phys - st.slots;
        let fit = room / ((conv + ssm) * layers).max(1);
        let slots = u64::from(want).min(fit.saturating_sub(u64::from(spares))) as u32;
        if slots <= st.slots {
            return Ok(st.slots);
        }
        let (old, grow) = (st.slots, slots - st.slots);
        let phys = st.phys + grow;
        let dev = s.device();
        let new = [
            Buffer::alloc(dev, (conv * layers * u64::from(phys)).max(1))?,
            Buffer::alloc(dev, (ssm * layers * u64::from(phys)).max(1))?,
        ];
        // Spares move up past the new slots; the sequences' own stay where they are.
        let mut host = vec![0u8; (conv.max(ssm) * u64::from(st.phys)) as usize];
        for layer in 0..layers {
            for (from, to, n) in [(&st.conv, &new[0], conv), (&st.ssm, &new[1], ssm)] {
                let host = &mut host[..(n * u64::from(st.phys)) as usize];
                s.download(from, layer * u64::from(st.phys) * n, host)?;
                let (own, spare) = host.split_at((n * u64::from(old)) as usize);
                let base = layer * u64::from(phys);
                s.upload(to, base * n, own)?;
                if !spare.is_empty() {
                    s.upload(to, (base + u64::from(slots)) * n, spare)?;
                }
            }
        }
        let [conv_buf, ssm_buf] = new;
        let moved = |p: u32| if p < old { p } else { p + grow };
        (st.conv, st.ssm, st.slots, st.phys) = (conv_buf, ssm_buf, slots, phys);
        st.map = (st.map.iter().map(|&p| moved(p)))
            .chain(old..slots)
            .collect();
        st.free = st.free.iter().map(|&p| moved(p)).collect();
        for slot in old..slots {
            self.zero_slot(s, slot)?;
        }
        Ok(slots)
    }

    /// Zeroes `slot` in every layer: the state before position 0.
    pub(super) fn zero_slot(&mut self, s: &Stream, slot: u32) -> Result<(), KError> {
        let (Some(h), Some(st)) = (&self.hybrid, &mut self.states) else {
            return Err(KError::InvalidArg);
        };
        if slot >= st.slots {
            return Err(KError::InvalidArg);
        }
        st.release(slot);
        let (conv, ssm) = h.slot_bytes();
        let zeros = vec![0u8; conv.max(ssm) as usize];
        for layer in 0..u64::from(h.gdn_layers()) {
            let at = layer * u64::from(st.phys) + u64::from(slot);
            s.upload(&st.conv, at * conv, &zeros[..conv as usize])?;
            s.upload(&st.ssm, at * ssm, &zeros[..ssm as usize])?;
        }
        Ok(())
    }

    /// Copies `from` into `to` in every layer, after the queued work: `to` then runs on from where
    /// `from` stands. The state an engine keeps for a sequence's next turn; onto itself, it moves
    /// a slot that lives in a spare home and frees the spare.
    // ponytail: through the host (the ABI has no device copy); 64 MB a slot on the 35B. Add a copy
    // op when a profile shows it beside the prefill it saves.
    pub fn copy_slot(&mut self, s: &Stream, from: u32, to: u32) -> Result<(), KError> {
        let (Some(h), Some(st)) = (&self.hybrid, &mut self.states) else {
            return Err(KError::InvalidArg);
        };
        if from >= st.slots || to >= st.slots {
            return Err(KError::InvalidArg);
        }
        let src = st.map[from as usize];
        if src == to {
            return Ok(());
        }
        st.release(to);
        let (conv, ssm) = h.slot_bytes();
        let mut host = vec![0u8; conv.max(ssm) as usize];
        for layer in 0..u64::from(h.gdn_layers()) {
            let at = |slot: u32| layer * u64::from(st.phys) + u64::from(slot);
            for (buf, n) in [(&st.conv, conv), (&st.ssm, ssm)] {
                let host = &mut host[..n as usize];
                s.download(buf, at(src) * n, host)?;
                s.upload(buf, at(to) * n, host)?;
            }
        }
        Ok(())
    }

    /// The bytes of `slot`'s state, after the queued work: each Gated DeltaNet layer's conv state
    /// then its recurrent state. For comparing two runs' states exactly.
    pub fn state_bytes(&self, s: &Stream, slot: u32) -> Result<Vec<u8>, KError> {
        let mut out = vec![0u8; self.state_size()];
        self.read_state(s, slot, &mut out)?;
        Ok(out)
    }

    /// One state slot's bytes ([`Self::state_bytes`]); 0 without recurrent layers.
    pub fn state_size(&self) -> usize {
        self.hybrid.as_ref().map_or(0, |h| {
            let (conv, ssm) = h.slot_bytes();
            ((conv + ssm) * u64::from(h.gdn_layers())) as usize
        })
    }

    /// [`Self::state_bytes`] into `out`, which must be [`Self::state_size`] bytes, else `InvalidArg`.
    pub fn read_state(&self, s: &Stream, slot: u32, out: &mut [u8]) -> Result<(), KError> {
        let (Some(h), Some(st)) = (&self.hybrid, &self.states) else {
            return Err(KError::InvalidArg);
        };
        let p = u64::from(*st.map.get(slot as usize).ok_or(KError::InvalidArg)?);
        let (conv, ssm) = h.slot_bytes();
        if out.len() != self.state_size() {
            return Err(KError::InvalidArg);
        }
        for (layer, o) in out.chunks_exact_mut((conv + ssm) as usize).enumerate() {
            let at = layer as u64 * u64::from(st.phys) + p;
            let (c, r) = o.split_at_mut(conv as usize);
            s.download(&st.conv, at * conv, c)?;
            s.download(&st.ssm, at * ssm, r)?;
        }
        Ok(())
    }

    /// Writes [`Self::state_bytes`]' bytes into `slot`, after the queued work: a reload carries a
    /// parked sequence's state across. `InvalidArg` for a slot out of range or bytes of another size.
    pub fn set_state_bytes(&mut self, s: &Stream, slot: u32, bytes: &[u8]) -> Result<(), KError> {
        let (Some(h), Some(st)) = (&self.hybrid, &mut self.states) else {
            return Err(KError::InvalidArg);
        };
        let (conv, ssm) = h.slot_bytes();
        let layer_bytes = (conv + ssm) as usize;
        if slot >= st.slots || bytes.len() != layer_bytes * h.gdn_layers() as usize {
            return Err(KError::InvalidArg);
        }
        st.release(slot);
        for (layer, b) in bytes.chunks_exact(layer_bytes).enumerate() {
            let at = layer as u64 * u64::from(st.phys) + u64::from(slot);
            let (c, r) = b.split_at(conv as usize);
            s.upload(&st.conv, at * conv, c)?;
            s.upload(&st.ssm, at * ssm, r)?;
        }
        Ok(())
    }

    /// [`Model::begin_check`]: a check of more than one row snapshots its first row, into a spare it
    /// takes while its sequence lives in its own slot, else into that slot, beside the other
    /// sequences' checks of the step. `Unsupported` past [`Transformer::CHECK_ROWS`] rows, without
    /// a free spare or a record ([`Model::alloc_checks`]), or past the rows a step records (the
    /// checks' and one for every other sequence slot); `InvalidArg` for a slot out of range, no
    /// rows or a check on the slot not ended.
    pub(super) fn check_begin(&mut self, slot: u32, rows: u32) -> Result<(), KError> {
        let Some(st) = &mut self.states else {
            return Err(KError::Unsupported);
        };
        if slot >= st.slots || rows == 0 || st.checks.iter().any(|c| c.slot == slot) {
            return Err(KError::InvalidArg);
        }
        let checked: u32 = rows + st.checks.iter().map(|c| c.rows).sum::<u32>();
        let beside = st.slots - 1 - st.checks.len() as u32;
        let recorded = st.replay.is_some() && i64::from(checked + beside) <= REPLAY_ROWS;
        if rows > Transformer::CHECK_ROWS || (rows > 1 && !recorded) {
            return Err(KError::Unsupported);
        }
        let snap = match rows > 1 && st.map[slot as usize] == slot {
            true => Some(st.free.pop().ok_or(KError::Unsupported)?),
            false => None,
        };
        st.checks.push(Check {
            slot,
            rows,
            snap,
            row0: 0,
            ran: false,
        });
        Ok(())
    }

    /// [`Model::end_check`]: the state after the check's first `kept` rows. A sequence that checked
    /// in its own slot and kept one row lives in the spare that holds it (no op); one that kept
    /// more replays its kept rows past the first there and snapshots the last into its own slot,
    /// where it lives on. One that checked in a spare has its first row in its own slot, where it
    /// replays and lives again, unless it kept every row (then it stays). The other spares free.
    /// `InvalidArg` for `kept` out of range or no check on `slot`, and for a check whose step did
    /// not run, which it drops.
    pub(super) fn check_end(&mut self, s: &Stream, slot: u32, kept: u32) -> Result<(), KError> {
        let Some(st) = &mut self.states else {
            return Err(KError::Unsupported);
        };
        let ends = |c: &Check| c.slot == slot && (1..=c.rows).contains(&kept);
        let Some(i) = st.checks.iter().position(ends) else {
            return Err(KError::InvalidArg);
        };
        let c = st.checks.swap_remove(i);
        if !c.ran {
            st.free.extend(c.snap);
            return Err(KError::InvalidArg);
        }
        // Where the kept rows past the first replay, and the slot the last one snapshots into.
        let replay = match c.snap {
            _ if kept == c.rows => {
                st.free.extend(c.snap);
                None
            }
            Some(spare) if kept == 1 => {
                st.map[slot as usize] = spare;
                None
            }
            Some(spare) => {
                st.free.push(spare);
                Some((spare, Some(slot)))
            }
            None => {
                let away = std::mem::replace(&mut st.map[slot as usize], slot);
                if away != slot {
                    st.free.push(away);
                }
                (kept > 1).then_some((slot, None))
            }
        };
        match replay {
            Some((at, into)) => self.replay_rows(s, at, into, c.row0 + 1, kept - 1),
            None => Ok(()),
        }
    }

    /// Runs the last step's recorded rows `from..from + n` through every Gated DeltaNet layer's
    /// conv and delta rule on physical slot `at`, the last row's state snapshotted into `into`:
    /// a check's kept rows past its first, onto its first row's state. Their outputs are dropped.
    fn replay_rows(
        &self,
        s: &Stream,
        at: u32,
        into: Option<u32>,
        from: u32,
        n: u32,
    ) -> Result<(), KError> {
        let (hy, sc) = self.hybrid()?;
        let st = self.states.as_ref().ok_or(KError::InvalidArg)?;
        let rec = st.replay.as_ref().ok_or(KError::InvalidArg)?;
        let (from, n) = (i64::from(from), i64::from(n));
        let i32s = |i: i64, len: i64| Tensor::new(&rec.buf, Replay::arg(hy, i), DType::I32, &[len]);
        s.upload(&rec.buf, Replay::arg(hy, from), &[at as i32])?;
        s.upload(&rec.buf, Replay::arg(hy, REPLAY_ROWS + from), &[n as i32])?;
        let snaps = match into {
            Some(p) => {
                let mut table = vec![-1i32; n as usize];
                table[n as usize - 1] = p as i32;
                s.upload(&rec.buf, Replay::arg(hy, 2 * REPLAY_ROWS + from), &table)?;
                Some(i32s(2 * REPLAY_ROWS + from, n)?)
            }
            None => None,
        };
        let route = (
            &i32s(from, 1)?,
            &i32s(REPLAY_ROWS + from, 1)?,
            snaps.as_ref(),
        );
        let (ch, vd, vh) = (hy.channels(), hy.v_dim(), i64::from(hy.v_heads));
        let conv = Tensor::new(&sc.conv, 0, DType::F16, &[n, ch])?;
        let z = Tensor::new(&sc.z, 0, DType::F16, &[n, vd])?;
        let o = Tensor::new(&sc.o, 0, DType::F16, &[n, vh, i64::from(hy.head_dim)])?;
        let gated = Tensor::new(&sc.gated, 0, DType::F16, &[n, vd])?;
        let gs = gdn_state(hy, st);
        for layer in &self.layers {
            if let Mixer::Gdn(l) = &layer.mixer {
                let [qkv, alpha, beta] = rec.rows(hy, l.layer, from, n)?;
                let ins = [&qkv, &alpha, &beta];
                self.gdn_core(s, l, &gs, ins, &z, route, &conv, &o, &gated)?;
            }
        }
        Ok(())
    }

    /// Routes a step's Gated DeltaNet layers while a check is begun or a slot lives in a spare:
    /// reads the step's slots and lengths back (one sync), then [`Transformer::route_rows`].
    /// Nothing otherwise. Under a capture of a guessing load nothing runs here (a capture cannot
    /// download): the layers read the table, which [`Model::route_table`] writes before each replay.
    pub(super) fn route_state(&mut self, s: &Stream, b: &StateBatch) -> Result<(), KError> {
        let Some(st) = &mut self.states else {
            return Ok(());
        };
        st.step = None;
        if s.capturing() {
            if st.phys > st.slots {
                st.step = Some(true);
            }
            return Ok(());
        }
        let moved = st.map.iter().enumerate().any(|(l, &p)| p as usize != l);
        if st.checks.is_empty() && !moved {
            return Ok(());
        }
        let n = b.slots.shape()[0];
        if b.slots.shape() != [n] || b.q_lens.shape() != [n] {
            return Err(KError::InvalidArg);
        }
        let (mut slots, mut lens) = (vec![0i32; n as usize], vec![0i32; n as usize]);
        s.download(b.slots.buffer(), b.slots.offset(), &mut slots)?;
        s.download(b.q_lens.buffer(), b.q_lens.offset(), &mut lens)?;
        self.route_rows(s, slots, &lens, false)
    }

    /// Writes each sequence's physical slot, each check's first-row snapshot and, for a moved slot
    /// in a step that is not its check, a snapshot of its last row into its own slot, where it
    /// lives again after the step. `force` writes the snapshot slots even when none is taken: a replay
    /// reads them whatever the step is, and a step before it may have left some.
    pub(super) fn route_rows(
        &mut self,
        s: &Stream,
        mut slots: Vec<i32>,
        lens: &[i32],
        force: bool,
    ) -> Result<(), KError> {
        let cap = self.max_rows;
        let Some(st) = &mut self.states else {
            return Err(KError::Unsupported);
        };
        if slots.len() != lens.len() || slots.len() as i64 > cap {
            return Err(KError::InvalidArg);
        }
        let rows: i64 = lens.iter().map(|&l| i64::from(l)).sum();
        let ours = |c: &Check| {
            let at = slots.iter().position(|&s| s == c.slot as i32);
            !c.ran && at.is_some_and(|i| lens[i] == c.rows as i32)
        };
        let valid = slots.iter().all(|&s| s >= 0 && (s as u32) < st.slots)
            && lens.iter().all(|&l| l >= 0)
            && rows <= cap
            && st.checks.iter().all(ours)
            && (rows <= REPLAY_ROWS || st.checks.iter().all(|c| c.rows == 1));
        if !valid {
            return Err(KError::InvalidArg);
        }
        let mut snaps = vec![-1i32; rows as usize];
        let (mut first, mut snapped) = (0, false);
        for (slot, &len) in slots.iter_mut().zip(lens) {
            let (l, len) = (*slot as u32, len as usize);
            let p = st.map[l as usize];
            match st.checks.iter_mut().find(|c| c.slot == l) {
                Some(c) => {
                    if c.rows > 1 {
                        snaps[first] = c.snap.unwrap_or(l) as i32;
                        snapped = true;
                    }
                    c.row0 = first as u32;
                    c.ran = true;
                }
                _ if p != l && len > 0 => {
                    snaps[first + len - 1] = l as i32;
                    snapped = true;
                    st.map[l as usize] = l;
                    st.free.push(p);
                }
                _ => {}
            }
            *slot = p as i32;
            first += len;
        }
        s.upload(&st.route, 0, &slots)?;
        if force || snapped {
            s.upload(&st.route, DType::I32.bytes(cap as u64), &snaps)?;
        }
        st.step = Some(snapped);
        Ok(())
    }

    /// A Gated DeltaNet layer of `h` into `out`: every row of the batch, sequences and slots as
    /// `state` maps them. `norm`: the residual and norm weight `h` is still to come from, through
    /// `fused_gdn_in` with the in-projections. True where `fused_proj_add` added the output onto
    /// `resid` instead.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn(
        &self,
        s: &Stream,
        l: &Gdn,
        h: &Tensor,
        state: Option<&StateBatch>,
        out: &Tensor,
        norm: Option<(&Tensor, W)>,
        resid: &Tensor,
    ) -> Result<bool, KError> {
        let (hy, sc) = self.hybrid()?;
        let (st, b) = match (&self.states, state) {
            (Some(st), Some(b)) => (st, b),
            _ => return Err(KError::InvalidArg),
        };
        let rows = h.shape()[0];
        let (ch, vd, vh) = (hy.channels(), hy.v_dim(), i64::from(hy.v_heads));
        let hd = i64::from(hy.head_dim);
        let eps = self.cfg.rms_eps;
        let conv = Tensor::new(&sc.conv, 0, DType::F16, &[rows, ch])?;
        let z = Tensor::new(&sc.z, 0, DType::F16, &[rows, vd])?;
        let o = Tensor::new(&sc.o, 0, DType::F16, &[rows, vh, hd])?;
        let gated = Tensor::new(&sc.gated, 0, DType::F16, &[rows, vd])?;
        // A guessing load's layer inputs land in its record, for a check's rewind to replay.
        let [qkv, alpha, beta] = match &st.replay {
            Some(rec) if rows <= REPLAY_ROWS => rec.rows(hy, l.layer, 0, rows)?,
            _ => {
                let alpha = Tensor::new(&sc.ab, 0, DType::F32, &[rows, vh])?;
                let beta_at = alpha.row_bytes() * rows as u64;
                [
                    Tensor::new(&sc.qkv, 0, DType::F16, &[rows, ch])?,
                    alpha,
                    Tensor::new(&sc.ab, beta_at, DType::F32, &[rows, vh])?,
                ]
            }
        };
        let projs = [(l.qkv, &qkv), (l.z, &z), (l.alpha, &alpha), (l.beta, &beta)];
        let fused = match norm {
            Some((x, nw)) => {
                let nw = self.t(nw)?;
                let ws = [
                    self.t(l.qkv)?,
                    self.t(l.z)?,
                    self.t(l.alpha)?,
                    self.t(l.beta)?,
                ];
                let outs = projs.map(|p| p.1);
                let r =
                    self.fused_call(|| ops::fused_gdn_in(s, x, &nw, eps, h, ws.each_ref(), outs));
                if matches!(r, Err(KError::Unsupported)) {
                    ops::rmsnorm(s, x, &nw, eps, h)?;
                    false
                } else {
                    r.map(|()| true)?
                }
            }
            None => false,
        };
        if !fused {
            for (w, o) in projs {
                self.linear(s, w, None, h, o)?;
            }
        }
        let gs = gdn_state(hy, st);
        let lens = &b.q_lens;
        let i32s = |at: u64, n: i64| Tensor::new(&st.route, at, DType::I32, &[n]);
        let routed = match st.step {
            Some(snapped) => Some((
                i32s(0, lens.shape()[0])?,
                (snapped.then(|| i32s(DType::I32.bytes(self.max_rows as u64), rows)))
                    .transpose()?,
            )),
            None => None,
        };
        let (slots, snaps) = match &routed {
            Some((sl, sn)) => (sl, sn.as_ref()),
            None => (&b.slots, None),
        };
        let ins = [&qkv, &alpha, &beta];
        if self.gdn_core(s, l, &gs, ins, &z, (slots, lens, snaps), &conv, &o, &gated)? {
            return self.gdn_out(s, l, &gated, out, resid);
        }
        // llama.cpp's build_norm_gated: RMS norm per v head, times silu(z).
        ops::qk_norm(s, &o, &self.t(l.norm)?, eps)?;
        ops::silu_mul(s, &z, &flat(&o)?, &gated)?;
        self.gdn_out(s, l, &gated, out, resid)
    }

    /// Layer `l`'s conv and delta rule over `[qkv, alpha, beta]`, its state as `gs` holds it and
    /// `(slots, lens, snaps)` route it: the conv output into `conv`, then through `fused_gdn`
    /// with the gated norm by `z` the layer's gated output into `gated` (true), else the delta
    /// rule's into `o`. A step and a check's replay take the same ops, so their states agree.
    #[allow(clippy::too_many_arguments)]
    fn gdn_core(
        &self,
        s: &Stream,
        l: &Gdn,
        gs: &GdnState,
        [qkv, alpha, beta]: [&Tensor; 3],
        z: &Tensor,
        (slots, lens, snaps): (&Tensor, &Tensor, Option<&Tensor>),
        conv: &Tensor,
        o: &Tensor,
        gated: &Tensor,
    ) -> Result<bool, KError> {
        let eps = self.cfg.rms_eps;
        let w = |w: W| self.t(w);
        let (dt, a, cw) = (w(l.dt)?, w(l.a)?, w(l.conv)?);
        if self.fusing(FUSED.gdn) {
            let (state, nw) = ((gs, l.layer), w(l.norm)?);
            let r = self.fused_call(|| {
                let ab = [alpha, beta, &dt, &a];
                ops::fused_gdn(
                    s,
                    qkv,
                    &cw,
                    ab,
                    eps,
                    state,
                    slots,
                    lens,
                    snaps,
                    [&nw, z],
                    conv,
                    gated,
                )
            });
            match r {
                Err(KError::Unsupported) => {}
                r => return r.map(|()| true),
            }
        }
        ops::gdn_conv(s, qkv, &cw, gs, l.layer, slots, lens, snaps, conv)?;
        let (layer, o) = (l.layer, &flat(o)?);
        ops::gdn(
            s, conv, alpha, beta, &dt, &a, eps, gs, layer, slots, lens, snaps, o,
        )?;
        Ok(false)
    }

    /// A Gated DeltaNet layer's out projection of `gated` into `out`, or onto `resid` through `fused_proj_add`:
    /// then true.
    fn gdn_out(
        &self,
        s: &Stream,
        l: &Gdn,
        gated: &Tensor,
        out: &Tensor,
        resid: &Tensor,
    ) -> Result<bool, KError> {
        if self.proj_add(s, l.out, gated, None, resid)? {
            return Ok(true);
        }
        self.linear(s, l.out, None, gated, out)?;
        Ok(false)
    }

    /// The Q projection's `[rows, heads, 2 head_dim]` pairs of a gated layer whose q is `q`.
    pub(super) fn gated_pairs(&self, q: &Tensor) -> Result<Tensor<'_>, KError> {
        let (_, sc) = self.hybrid()?;
        let [rows, heads, hd] = *q.shape() else {
            return Err(KError::InvalidArg);
        };
        Tensor::new(&sc.qfull, 0, DType::F16, &[rows, heads, 2 * hd])
    }

    /// Q `[rows, heads, head_dim]` of a gated attention layer from its projection's pairs, with its gate kept for
    /// [`Transformer::attn_gate`]: the projection holds `[q, gate]` per head, and a gemm with a 0/1 matrix copies
    /// each half out exactly (one product of 1 per output, the rest of 0).
    pub(super) fn gated_split(&self, s: &Stream, q: &Tensor) -> Result<(), KError> {
        let (_, sc) = self.hybrid()?;
        let [rows, heads, hd] = *q.shape() else {
            return Err(KError::InvalidArg);
        };
        let pairs = self.gated_pairs(q)?.reshape(&[rows * heads, 2 * hd])?;
        let sel = |i: u64| {
            Tensor::new(
                &sc.sel,
                i * DType::F16.bytes((2 * hd * hd) as u64),
                DType::F16,
                &[hd, 2 * hd],
            )
        };
        let gate = Tensor::new(&sc.gate, 0, DType::F16, &[rows * heads, hd])?;
        ops::gemm(
            s,
            &pairs,
            &sel(0)?,
            None,
            &q.reshape(&[rows * heads, hd])?,
            1.0,
            0.0,
        )?;
        ops::gemm(s, &pairs, &sel(1)?, None, &gate, 1.0, 0.0)
    }

    /// The attention output `attn` `[rows, heads, head_dim]` times sigmoid of the gate
    /// [`Transformer::gated_split`] kept, in place.
    pub(super) fn attn_gate(&self, s: &Stream, attn: &Tensor) -> Result<(), KError> {
        let a = flat(attn)?;
        ops::sigmoid_mul(s, &a, &self.attn_gate_of(attn)?, &a)
    }

    /// The gate [`Transformer::gated_split`] kept for `attn`, as `[rows, heads * head_dim]`.
    pub(super) fn attn_gate_of(&self, attn: &Tensor) -> Result<Tensor<'_>, KError> {
        let (_, sc) = self.hybrid()?;
        Tensor::new(&sc.gate, 0, DType::F16, flat(attn)?.shape())
    }

    /// Whether Gated DeltaNet layer `l`'s norm and in-projections at `rows` rows take `fused_gdn_in`: outside check
    /// steps, its quantized projections on mmvq (an f16 alpha or beta takes gemm, which the entry replaces).
    pub(super) fn gdn_in_fusable(&self, l: &Gdn, rows: i64) -> bool {
        self.fusing(FUSED.gdn_in)
            && !self.check.get()
            && [l.qkv, l.z, l.alpha, l.beta]
                .iter()
                .all(|&w| w.side || self.takes_mmvq(w, rows))
    }

    /// [`Transformer::shared_ffn`] up to its down projection: the shared expert's gate and up of `h` and their
    /// silu_mul into t (`[rows, shared ffn]`), and its gate input (`[rows, 1]`), for `fused_moe_down`, with the
    /// down weight it takes.
    pub(super) fn shared_in(
        &self,
        s: &Stream,
        sh: &Shared,
        h: &Tensor,
    ) -> Result<(W, Tensor<'_>, Tensor<'_>), KError> {
        let shape = [h.shape()[0], sh.gate.shape[0]];
        let g = Tensor::new(&self.sc.gate, 0, DType::F16, &shape)?;
        let u = Tensor::new(&self.sc.up, 0, DType::F16, &shape)?;
        let (act, sg) = self.shared_outs(sh, h.shape()[0])?;
        self.linear(s, sh.gate, None, h, &g)?;
        self.linear(s, sh.up, None, h, &u)?;
        ops::silu_mul(s, &g, &u, &act)?;
        self.linear(s, sh.gate_inp, None, h, &sg)?;
        Ok((sh.down, act, sg))
    }

    /// Where [`Transformer::shared_in`] leaves the shared expert's activation (t) and gate for `rows` rows.
    pub(super) fn shared_outs(
        &self,
        sh: &Shared,
        rows: i64,
    ) -> Result<(Tensor<'_>, Tensor<'_>), KError> {
        let (_, hs) = self.hybrid()?;
        Ok((
            Tensor::new(&self.sc.t, 0, DType::F16, &[rows, sh.gate.shape[0]])?,
            Tensor::new(&hs.sg, 0, DType::F16, &[rows, 1])?,
        ))
    }

    /// The shared expert of `h` into `out`: its dense FFN times sigmoid of its one-column gate.
    pub(super) fn shared_ffn(
        &self,
        s: &Stream,
        sh: &Shared,
        h: &Tensor,
        out: &Tensor,
    ) -> Result<(), KError> {
        let (_, sc) = self.hybrid()?;
        self.dense_ffn(s, sh.gate, sh.up, sh.down, h, out, None)?;
        let g = Tensor::new(&sc.sg, 0, DType::F16, &[h.shape()[0], 1])?;
        self.linear(s, sh.gate_inp, None, h, &g)?;
        ops::sigmoid_mul(s, out, &g, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arch, tiny};
    use larch_kernels::CPU;

    const SLOTS: u32 = 4;
    const ROWS: u32 = 16;

    fn fixture(name: &str) -> (Device, Stream, Transformer) {
        let path = std::env::temp_dir().join(format!("larch-{name}-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen35, 1)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let mut model = Transformer::load(&path, &dev, &s, ROWS).unwrap();
        std::fs::remove_file(path).unwrap();
        model.alloc_state(&s, SLOTS).unwrap();
        model
            .alloc_check_slots(&s, Transformer::CHECK_SLOTS)
            .unwrap();
        (dev, s, model)
    }

    /// What a routing left: the map, the spares free, a begun check's `ran`, the step's routing
    /// and the two tables, read back from the device.
    type Routed = (
        Vec<u32>,
        Vec<u32>,
        Option<bool>,
        Option<bool>,
        Vec<i32>,
        Vec<i32>,
    );

    fn route(m: &mut Transformer, s: &Stream, slots: &[i32], lens: &[i32], force: bool) -> Routed {
        m.route_rows(s, slots.to_vec(), lens, force).unwrap();
        let st = m.states.as_ref().unwrap();
        let (mut table, mut snaps) = (
            vec![0i32; slots.len()],
            vec![0i32; lens.iter().sum::<i32>() as usize],
        );
        s.download(&st.route, 0, &mut table).unwrap();
        let at = DType::I32.bytes(ROWS as u64);
        s.download(&st.route, at, &mut snaps).unwrap();
        let ran = st.checks.first().map(|c| c.ran);
        (st.map.clone(), st.free.clone(), ran, st.step, table, snaps)
    }

    /// A check of `slot` run and rewound to its first of `rows` rows: the slot then lives in the
    /// spare that holds it.
    fn rewind(m: &mut Transformer, s: &Stream, slot: u32, rows: u32) {
        m.check_begin(slot, rows).unwrap();
        m.route_rows(s, vec![slot as i32], &[rows as i32], true)
            .unwrap();
        m.check_end(s, slot, 1).unwrap();
    }

    /// A replay's routing is the eager step's: the same map, spares, `ran` and tables whether the
    /// slots stay home, a check runs, a slot moves home or a padding row moves a free slot home.
    /// Forced, the snapshot table is written (-1 where none) even when nothing snapshots.
    #[test]
    fn forced_and_eager_routing_agree() {
        // (slots, lens, the setup): identity, a check beside a decode row, a moved live slot, and
        // a padding row (slot 2, unheld) over a moved free slot.
        type Setup = fn(&mut Transformer, &Stream);
        let check: Setup = |m, _| m.check_begin(0, 3).unwrap();
        let moved: Setup = |m, s| rewind(m, s, 0, 3);
        let free: Setup = |m, s| rewind(m, s, 2, 3);
        let scenarios: [(&[i32], &[i32], Setup); 4] = [
            (&[1, 0], &[1, 1], |_, _| ()),
            (&[1, 0], &[1, 3], check),
            (&[0, 1], &[1, 1], moved),
            (&[0, 1, 2], &[1, 1, 1], free),
        ];
        for (i, (slots, lens, setup)) in scenarios.into_iter().enumerate() {
            let run = |force| {
                let (_dev, s, mut m) = fixture(&format!("route-{i}-{force}"));
                setup(&mut m, &s);
                route(&mut m, &s, slots, lens, force)
            };
            let (eager, forced) = (run(false), run(true));
            assert_eq!(forced.0, eager.0, "map {i}");
            assert_eq!(forced.1, eager.1, "free {i}");
            assert_eq!(forced.2, eager.2, "ran {i}");
            assert_eq!(forced.3, eager.3, "step {i}");
            assert_eq!(forced.4, eager.4, "table {i}");
            if eager.3 == Some(true) {
                assert_eq!(forced.5, eager.5, "snaps {i}");
            } else {
                assert!(forced.5.iter().all(|&s| s == -1), "{i}: {:?}", forced.5);
            }
        }
        // The tables spelled out once: a check's one snapshot, of its first row; its slot rewound
        // to that row and moved home; then a check that keeps two rows replays them home.
        let (_dev, s, mut m) = fixture("route-moved");
        m.check_begin(0, 3).unwrap();
        let (map, free, ran, step, table, snaps) = route(&mut m, &s, &[1, 0], &[1, 3], true);
        assert_eq!(table, [1, 0]);
        assert_eq!(
            [snaps[0], snaps[2], snaps[3]],
            [-1; 3],
            "the first row's alone"
        );
        let spare = snaps[1] as u32;
        assert!(spare >= SLOTS && !free.contains(&spare));
        assert_eq!(
            (map, ran, step),
            ((0..SLOTS).collect(), Some(true), Some(true))
        );
        m.check_end(&s, 0, 1).unwrap();
        let (map, free, _, _, table, snaps) = route(&mut m, &s, &[0, 1], &[1, 1], true);
        assert_eq!(
            table,
            [spare as i32, 1],
            "the moved slot runs where it lives"
        );
        assert_eq!(snaps, [0, -1], "and snapshots home");
        assert_eq!(map, (0..SLOTS).collect::<Vec<_>>());
        assert!(free.contains(&spare));
        m.check_begin(0, 3).unwrap();
        route(&mut m, &s, &[0], &[3], true);
        m.check_end(&s, 0, 2).unwrap();
        let st = m.states.as_ref().unwrap();
        assert_eq!(st.map, (0..SLOTS).collect::<Vec<_>>(), "replayed home");
        assert_eq!(st.free.len() as u32, Transformer::CHECK_SLOTS);
    }

    /// A check past [`Transformer::CHECK_ROWS`] rows, or one beside checks that took every spare,
    /// is refused and takes nothing; a sequence in a spare checks without one.
    #[test]
    fn a_check_takes_one_spare_at_most() {
        let (_dev, s, mut m) = fixture("check-spares");
        let rows = Transformer::CHECK_ROWS;
        assert_eq!(m.check_begin(0, rows + 1), Err(KError::Unsupported));
        rewind(&mut m, &s, 0, 3);
        let spares = Transformer::CHECK_SLOTS - 1;
        assert_eq!(m.states.as_ref().unwrap().spares_free(), spares);
        for slot in 1..=spares.min(SLOTS - 1) {
            m.check_begin(slot, rows).unwrap();
        }
        m.check_begin(0, rows).unwrap();
        let st = m.states.as_ref().unwrap();
        assert_eq!(st.spares_free(), spares - spares.min(SLOTS - 1));
    }

    /// A step that snapshots leaves its table behind; the next forced routing clears it, so a
    /// replay that takes no snapshot writes none.
    #[test]
    fn a_forced_routing_clears_earlier_snapshots() {
        let (_dev, s, mut m) = fixture("route-clear");
        m.check_begin(0, 3).unwrap();
        let (.., snaps) = route(&mut m, &s, &[0], &[3], true);
        assert!(snaps[0] >= SLOTS as i32);
        m.check_end(&s, 0, 3).unwrap();
        let (.., snaps) = route(&mut m, &s, &[0], &[3], true);
        assert_eq!(snaps, [-1, -1, -1]);
    }

    /// What the eager step refuses, a replay refuses before anything moves.
    #[test]
    fn route_rows_refuses_what_route_state_does() {
        let (m_dev, s, mut m) = fixture("route-bad");
        let bad = KError::InvalidArg;
        assert_eq!(m.route_rows(&s, vec![0], &[1, 1], true), Err(bad));
        assert_eq!(m.route_rows(&s, vec![SLOTS as i32], &[1], true), Err(bad));
        assert_eq!(m.route_rows(&s, vec![0], &[-1], true), Err(bad));
        m.check_begin(0, 3).unwrap();
        assert_eq!(
            m.route_rows(&s, vec![0], &[2], true),
            Err(bad),
            "not the check's rows"
        );
        let st = m.states.as_ref().unwrap();
        assert!(!st.checks[0].ran);
        let path = std::env::temp_dir().join(format!("larch-route-d-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3, 1)).unwrap();
        let mut dense = Transformer::load(&path, &m_dev, &s, ROWS).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(dense.route_table(&s, &[0], &[1]), Err(KError::Unsupported));
    }
}
