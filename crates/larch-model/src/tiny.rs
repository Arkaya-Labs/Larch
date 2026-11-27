//! The generated test model (architecture 15): 2 layers, 256 vocab, d_model 256, random weights, with
//! the weight matrices spread over the six quant types the ops take (6.1) and f32 vectors. Quant blocks
//! are random bytes with small finite scales, which is a valid block of any GGML type.

use crate::Arch;
use crate::gguf::{Value, ggml_type, write};
use larch_kernels::DType;

/// A vocabulary-only GGUF of a few KB: a byte-level BPE (Llama 3 pre-tokeniser) vocabulary with the
/// Llama 3.1 chat template, so text paths run on CPU without a real model. Its tokens and renders are
/// llama.cpp's (`tests/tokenizer.rs`, `tiny_tokenizer`); BOS `<|begin_of_text|>`, EOS `<|eot_id|>`.
pub const TOKENIZER_GGUF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/tiny-tokenizer.gguf"
);

/// Vocabulary size.
pub const VOCAB: u32 = 256;
/// `d_model`.
pub const D: u64 = 256;
/// Feed-forward width.
pub const FFN: u64 = 512;
/// Attention heads.
pub const HEADS: u64 = 4;
/// KV heads (GQA 2).
pub const KV_HEADS: u64 = 2;
/// Head width.
pub const HEAD_DIM: u64 = 64;
/// Layers.
pub const LAYERS: u32 = 2;
/// Training context length.
pub const CTX: u32 = 512;
/// Routed experts per layer (qwen3moe).
pub const EXPERTS: u32 = 4;
/// Experts each token runs (qwen3moe).
pub const EXPERTS_USED: u32 = 2;
/// One expert's feed-forward width (qwen3moe).
pub const EXPERT_FFN: u64 = 256;
/// The shared expert's feed-forward width (qwen35moe).
pub const SHARED_FFN: u64 = 256;
/// Gated DeltaNet k heads, v heads and head width (qwen35, qwen35moe; layer 0 is one, layer 1 full
/// attention).
pub const GDN_K_HEADS: u64 = 2;
pub const GDN_V_HEADS: u64 = 4;
pub const GDN_HEAD_DIM: u64 = 32;
/// The Gated DeltaNet conv's width.
pub const GDN_KERNEL: u64 = 4;
/// Rotated dimensions per head in qwen35 and qwen35moe (partial rope).
pub const HYBRID_ROPE_DIMS: u32 = 32;

const TYPES: [DType; 6] = [
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Q8_0,
    DType::Q4_0,
    DType::Q5_0,
];

/// splitmix64: deterministic test data without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-a, a).
    fn uniform(&mut self, a: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
    }
}

/// f32 → f16 bits for test data (truncating; subnormals kept, since K-quant scales are tiny).
fn f16(x: f32) -> [u8; 2] {
    let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
    let a = x.abs();
    let h = if a < 6.103_515_6e-5 {
        (a * 16_777_216.0).round() as u16 // a / 2^-24
    } else {
        let b = a.to_bits();
        ((((b >> 23) - 112) << 10) | ((b >> 13) & 0x3ff)) as u16
    };
    (sign | h).to_le_bytes()
}

/// `rows x cols` of quant type `t` with weights of magnitude about `a`.
fn weights(rng: &mut Rng, t: DType, rows: u64, cols: u64, a: f32) -> Vec<u8> {
    let mut out: Vec<u8> = (0..t.bytes(rows * cols))
        .map(|_| rng.next() as u8)
        .collect();
    if t == DType::Nvfp4 {
        // Four E4M3 scale bytes a block: the exponent for a/12 (codes reach 12, at half a scale's value),
        // a random mantissa.
        let e = (((a / 6.0).log2().floor() as i32) + 7).clamp(1, 14) as u8;
        for block in out.chunks_mut(36) {
            for s in &mut block[..4] {
                *s = (e << 3) | (rng.next() as u8 & 7);
            }
        }
        return out;
    }
    // Scale fields: the largest |weight| is about a.
    let (fields, scale): (&[usize], f32) = match t {
        DType::Q4K | DType::Q5K => (&[0, 2], a / (63.0 * 31.0)),
        DType::Q6K => (&[208], a / (128.0 * 32.0)),
        DType::Q8_0 => (&[0], a / 128.0),
        _ => (&[0], a / 16.0),
    };
    for block in out.chunks_mut(t.block_bytes() as usize) {
        for &f in fields {
            block[f..f + 2].copy_from_slice(&f16(scale * (0.5 + rng.uniform(0.5).abs())));
        }
    }
    out
}

/// A complete GGUF file. `arch` picks llama, qwen2 (q/k/v biases), qwen3 (qk-norm), qwen3moe (qk-norm,
/// an f32 router and stacked experts in place of the dense FFN) or qwen35 and qwen35moe (a Gated
/// DeltaNet layer, then a gated attention layer; qwen35moe's FFN is qwen3moe's plus a shared
/// expert); `seed` the weights.
pub fn gguf(arch: Arch, seed: u64) -> Vec<u8> {
    build(arch, seed, false)
}

/// [`gguf`] of a Qwen 3.5 arch with its multi-token-prediction block after the trunk, as the hybrid MoE target's
/// GGUF carries it (`blk.{LAYERS}`, `nextn_predict_layers` 1): a gated attention layer and the
/// arch's FFN (qwen35moe's router and shared-expert gate in bf16, as the hybrid MoE target's), `nextn.eh_proj`,
/// `enorm`, `hnorm` and `shared_head_norm`. The trunk is [`gguf`]'s, byte for byte.
pub fn gguf_mtp(arch: Arch, seed: u64) -> Vec<u8> {
    assert!(arch.hybrid(), "only Qwen 3.5 has an MTP block");
    build(arch, seed, true)
}

fn build(arch: Arch, seed: u64, mtp: bool) -> Vec<u8> {
    let rng = Rng(seed);
    let name = match arch {
        Arch::Llama => "llama",
        Arch::Qwen2 => "qwen2",
        Arch::Qwen3 => "qwen3",
        Arch::Qwen3Moe => "qwen3moe",
        Arch::Qwen35 => "qwen35",
        Arch::Qwen35Moe => "qwen35moe",
        Arch::Inkling2Loop => panic!("inkling_gguf writes the Inkling 2 model"),
    };
    let k = |key: &str| format!("{name}.{key}");
    let mut kv = vec![
        ("general.architecture".to_string(), Value::Str(name.into())),
        (k("block_count"), Value::U32(LAYERS + u32::from(mtp))),
        (k("context_length"), Value::U32(CTX)),
        (k("embedding_length"), Value::U32(D as u32)),
        (k("feed_forward_length"), Value::U32(FFN as u32)),
        (k("attention.head_count"), Value::U32(HEADS as u32)),
        (k("attention.head_count_kv"), Value::U32(KV_HEADS as u32)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        (k("rope.freq_base"), Value::F32(10000.0)),
    ];
    let (hybrid, moe) = (
        arch.hybrid(),
        matches!(arch, Arch::Qwen3Moe | Arch::Qwen35Moe),
    );
    let qwen3 = matches!(arch, Arch::Qwen3 | Arch::Qwen3Moe) || hybrid;
    if qwen3 {
        kv.push((k("attention.key_length"), Value::U32(HEAD_DIM as u32)));
    }
    let (ch, vd) = (
        (2 * GDN_K_HEADS + GDN_V_HEADS) * GDN_HEAD_DIM,
        GDN_V_HEADS * GDN_HEAD_DIM,
    );
    if hybrid {
        let u = |v: u64| Value::U32(v as u32);
        kv.extend([
            (k("attention.value_length"), u(HEAD_DIM)),
            (k("rope.dimension_count"), Value::U32(HYBRID_ROPE_DIMS)),
            (
                k("rope.dimension_sections"),
                Value::Arr([6, 5, 5, 0].map(Value::I32).to_vec()),
            ),
            (k("full_attention_interval"), u(2)),
            (k("ssm.conv_kernel"), u(GDN_KERNEL)),
            (k("ssm.state_size"), u(GDN_HEAD_DIM)),
            (k("ssm.group_count"), u(GDN_K_HEADS)),
            (k("ssm.time_step_rank"), u(GDN_V_HEADS)),
            (k("ssm.inner_size"), u(vd)),
        ]);
    }
    if mtp {
        kv.push((k("nextn_predict_layers"), Value::U32(1)));
    }
    if arch == Arch::Qwen35Moe {
        kv.push((
            k("expert_shared_feed_forward_length"),
            Value::U32(SHARED_FFN as u32),
        ));
    }
    if moe {
        kv.push((k("expert_count"), Value::U32(EXPERTS)));
        kv.push((k("expert_used_count"), Value::U32(EXPERTS_USED)));
        kv.push((
            k("expert_feed_forward_length"),
            Value::U32(EXPERT_FFN as u32),
        ));
    }

    let mut b = Builder {
        rng,
        ty: 0,
        tensors: Vec::new(),
    };
    let (qd, kvd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    b.matrix("token_embd.weight".into(), u64::from(VOCAB), D);
    for i in 0..LAYERS {
        let n = |s: &str| format!("blk.{i}.{s}");
        b.vector(n("attn_norm.weight"), D, 1.0);
        if hybrid && i == 0 {
            b.matrix(n("attn_qkv.weight"), ch, D);
            b.matrix(n("attn_gate.weight"), vd, D);
            // alpha float (a side copy), beta quantized: both forms the loader takes.
            b.f32_matrix(n("ssm_alpha.weight"), GDN_V_HEADS, D);
            b.matrix(n("ssm_beta.weight"), GDN_V_HEADS, D);
            b.conv(n("ssm_conv1d.weight"), ch, GDN_KERNEL);
            b.vector(n("ssm_dt.bias"), GDN_V_HEADS, 0.5);
            b.vector(n("ssm_a"), GDN_V_HEADS, -0.5);
            b.vector(n("ssm_norm.weight"), GDN_HEAD_DIM, 1.0);
            b.matrix(n("ssm_out.weight"), D, vd);
            b.vector(n("post_attention_norm.weight"), D, 1.0);
            b.ffn(arch, i);
            continue;
        }
        // Qwen 3.5's Q projection holds a gate beside each head.
        b.matrix(n("attn_q.weight"), qd * (1 + u64::from(hybrid)), D);
        b.matrix(n("attn_k.weight"), kvd, D);
        b.matrix(n("attn_v.weight"), kvd, D);
        if arch == Arch::Qwen2 {
            b.vector(n("attn_q.bias"), qd, 0.0);
            b.vector(n("attn_k.bias"), kvd, 0.0);
            b.vector(n("attn_v.bias"), kvd, 0.0);
        }
        if qwen3 {
            b.vector(n("attn_q_norm.weight"), HEAD_DIM, 1.0);
            b.vector(n("attn_k_norm.weight"), HEAD_DIM, 1.0);
        }
        b.matrix(n("attn_output.weight"), D, qd);
        let norm = if hybrid {
            "post_attention_norm.weight"
        } else {
            "ffn_norm.weight"
        };
        b.vector(n(norm), D, 1.0);
        b.ffn(arch, i);
    }
    b.vector("output_norm.weight".into(), D, 1.0);
    b.matrix("output.weight".into(), u64::from(VOCAB), D);
    if mtp {
        let n = |s: &str| format!("blk.{LAYERS}.{s}");
        b.vector(n("attn_norm.weight"), D, 1.0);
        b.matrix(n("attn_q.weight"), 2 * qd, D);
        b.matrix(n("attn_k.weight"), kvd, D);
        b.matrix(n("attn_v.weight"), kvd, D);
        b.vector(n("attn_q_norm.weight"), HEAD_DIM, 1.0);
        b.vector(n("attn_k_norm.weight"), HEAD_DIM, 1.0);
        b.matrix(n("attn_output.weight"), D, qd);
        b.vector(n("post_attention_norm.weight"), D, 1.0);
        b.ffn(arch, LAYERS);
        if arch == Arch::Qwen35Moe {
            b.bf16(&n("ffn_gate_inp.weight"));
            b.bf16(&n("ffn_gate_inp_shexp.weight"));
        }
        b.matrix(n("nextn.eh_proj.weight"), D, 2 * D);
        b.vector(n("nextn.enorm.weight"), D, 1.0);
        b.vector(n("nextn.hnorm.weight"), D, 1.0);
        b.vector(n("nextn.shared_head_norm.weight"), D, 1.0);
    }
    write(&kv, &b.tensors)
}

/// The generated `inkling2_loop` model ([`inkling_gguf`]): llama.cpp's Inkling 2 Mini graph at a small
/// size, with every tensor and key the released files carry, in their types.
pub mod ink {
    /// `d_model`.
    pub const D: u64 = 128;
    /// Attention heads, KV heads and head width.
    pub const HEADS: u64 = 4;
    pub const KV_HEADS: u64 = 2;
    pub const HEAD_DIM: u64 = 32;
    /// Physical layers: a prelude of 2, a core of 2 run [`PASSES`] times, a coda of 2.
    pub const LAYERS: u32 = 6;
    /// Depth-mix block size; the loop core is `[LOOP_START, LOOP_END)`.
    pub const BLOCK: u32 = 2;
    pub const LOOP_START: u32 = 2;
    pub const LOOP_END: u32 = 4;
    pub const PASSES: u32 = 3;
    /// The controllers' rank.
    pub const GATE_RANK: u64 = 8;
    /// Full-attention layers; the rest attend a window of [`WINDOW`].
    pub const FULL: [u32; 2] = [3, 5];
    pub const WINDOW: u32 = 16;
    /// Relative-bias width and the full layers' extent.
    pub const REL: u64 = 16;
    pub const EXTENT: u32 = 24;
    /// Routed experts, experts per token and an expert's width.
    pub const EXPERTS: u64 = 8;
    pub const USED: u32 = 2;
    pub const FFN: u64 = 64;
    /// Layers whose experts carry a per-expert `.scale` (the released files' NVFP4 layers).
    pub const SCALED: u32 = 4;
    /// The short convolutions' taps.
    pub const CONV: u64 = 4;
    /// The byte vocabulary and the padded width of the embedding and output tensors.
    pub const VOCAB: u32 = 256;
    pub const VOCAB_PADDED: u32 = 288;
    pub const CTX: u32 = 512;
}

/// A tiny `inkling2_loop` GGUF ([`ink`]'s shapes) with a 256-token byte vocabulary (GPT-4o
/// pre-tokeniser, `inkling2`), so that llama.cpp loads the same bytes larch does. Matrices are Q8_0,
/// the sink and dynamic-tap projections bf16, the controllers, router, norms, queries, biases and
/// convolutions f32, as in the released files; layers below [`ink::SCALED`] scale each expert.
pub fn inkling_gguf(seed: u64) -> Vec<u8> {
    inkling(seed, false)
}

/// [`inkling_gguf`] with layer 0's experts in NVFP4, as the released files keep their scaled layers' (so
/// layers 1 to 3 scale Q8_0 experts, as Q3_K_XL's mixed layers do). Only layer 0's: with every scaled
/// layer's experts in four bits the random model is chaotic enough that llama.cpp's own runs of it agree
/// on top-1 at 74 % of positions, too few decisive ones for a gate.
pub fn inkling_gguf_nvfp4(seed: u64) -> Vec<u8> {
    inkling(seed, true)
}

fn inkling(seed: u64, nvfp4: bool) -> Vec<u8> {
    use ink::*;
    let name = "inkling2_loop";
    let k = |key: &str| format!("{name}.{key}");
    let u = |v: u64| Value::U32(v as u32);
    let per_layer = |v: u64| Value::Arr(vec![Value::I32(v as i32); LAYERS as usize]);
    let swa = |i: u32| !FULL.contains(&i);
    let tokens = (0..=255u8).map(|b| Value::Str(byte_token(b))).collect();
    let kv = vec![
        ("general.architecture".to_string(), Value::Str(name.into())),
        (k("block_count"), Value::U32(LAYERS)),
        (k("context_length"), Value::U32(CTX)),
        (k("embedding_length"), u(D)),
        (k("feed_forward_length"), u(FFN)),
        (k("expert_feed_forward_length"), u(FFN)),
        (k("attention.head_count"), per_layer(HEADS)),
        (k("attention.head_count_kv"), per_layer(KV_HEADS)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        (k("attention.key_length"), u(HEAD_DIM)),
        (k("attention.value_length"), u(HEAD_DIM)),
        (k("attention.key_length_swa"), u(HEAD_DIM)),
        (k("attention.value_length_swa"), u(HEAD_DIM)),
        (k("rope.dimension_count"), Value::U32(0)),
        (k("expert_count"), u(EXPERTS)),
        (k("expert_used_count"), Value::U32(USED)),
        (k("expert_weights_scale"), Value::F32(2.5)),
        (k("attention.sliding_window"), Value::U32(WINDOW)),
        (
            k("attention.sliding_window_pattern"),
            Value::Arr((0..LAYERS).map(|i| Value::Bool(swa(i))).collect()),
        ),
        (k("loop.count"), Value::U32(PASSES)),
        (k("loop.start"), Value::U32(LOOP_START)),
        (k("loop.end"), Value::U32(LOOP_END)),
        (k("loop.gate_rank"), u(GATE_RANK)),
        (k("loop.varying"), Value::Bool(false)),
        (k("vocab_size"), Value::U32(VOCAB_PADDED)),
        (k("vocab_size_unpadded"), Value::U32(VOCAB)),
        (k("shortconv.l_cache"), u(CONV)),
        // Small, so that the tiny activations reach the saturating parts of SITU.
        (k("activation.situ_beta"), Value::F32(0.75)),
        (k("activation.situ_linear_beta"), Value::F32(1.5)),
        (k("attn_res.block_size"), Value::U32(BLOCK)),
        (k("final_logit_softcapping"), Value::F32(4.0)),
        (k("attention.relative_length"), u(REL)),
        (k("attention.relative_extent"), Value::U32(EXTENT)),
        (k("attention.relative_clip"), Value::F32(1.5)),
        (k("attention.qk_clip"), Value::F32(2.0)),
        // A floor of 8 positions, so that the log scale grows within a test prompt.
        (k("attention.log_scale_floor"), Value::U32(8)),
        (k("attention.log_scale_alpha"), Value::F32(0.25)),
        ("tokenizer.ggml.model".into(), Value::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::Str("inkling2".into())),
        ("tokenizer.ggml.tokens".into(), Value::Arr(tokens)),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Arr(vec![Value::I32(1); 256]),
        ),
        // One merge of two control bytes no prompt holds: llama.cpp refuses an empty list.
        (
            "tokenizer.ggml.merges".into(),
            Value::Arr(vec![Value::Str(format!(
                "{} {}",
                byte_token(1),
                byte_token(2)
            ))]),
        ),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(10)),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
    ];

    let mut b = Builder {
        rng: Rng(seed),
        ty: 0,
        tensors: Vec::new(),
    };
    let (qd, kvd, e) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, EXPERTS);
    let vp = u64::from(VOCAB_PADDED);
    b.q8(String::from("token_embd.weight"), &[D, vp], 0.5);
    b.vector("token_embd_norm.weight".into(), D, 1.0);
    b.q8("output.weight".into(), &[D, vp], 0.1);
    b.vector("output_norm.weight".into(), D, 1.0);
    // Queries of about unit score spread over normalised states.
    b.f32("output_res_query.weight".into(), &[D], 3.0);
    for i in 0..LAYERS {
        let n = |s: &str| format!("blk.{i}.{s}");
        b.q8(n("attn_q.weight"), &[D, qd], 0.1);
        b.q8(n("attn_k.weight"), &[D, kvd], 0.1);
        b.q8(n("attn_v.weight"), &[D, kvd], 0.1);
        b.q8(n("attn_output.weight"), &[qd, D], 0.1);
        b.q8(n("attn_gate.weight"), &[D, qd], 0.1);
        b.vector(n("attn_q_clip.weight"), HEADS * HEAD_DIM, 1.0);
        b.reshape(&n("attn_q_clip.weight"), &[HEAD_DIM, HEADS]);
        b.f32(n("attn_sink_proj.weight"), &[D, HEADS], 0.2);
        b.bf16(&n("attn_sink_proj.weight"));
        b.q8(n("attn_rel.weight"), &[D, REL * HEADS], 0.2);
        let extent = u64::from(if swa(i) { WINDOW } else { EXTENT });
        b.f32(n("attn_rel_bias.weight"), &[REL, extent], 1.0);
        b.vector(n("attn_rel_clip.weight"), REL * HEADS, 1.0);
        b.reshape(&n("attn_rel_clip.weight"), &[REL, HEADS]);
        b.f32(n("attn_k_conv.weight"), &[CONV, kvd], 0.5);
        b.f32(n("attn_v_conv.weight"), &[CONV, kvd], 0.5);
        b.f32(n("attn_post_conv.weight"), &[CONV, D], 0.5);
        b.f32(n("ffn_post_conv.weight"), &[CONV, D], 0.5);
        if swa(i) {
            b.vector(n("attn_norm.weight"), D, 1.0);
        } else {
            b.vector(n("attn_q_inp_norm.weight"), D, 1.0);
            b.vector(n("attn_k_inp_norm.weight"), D, 1.0);
            b.vector(n("attn_v_inp_norm.weight"), D, 1.0);
            b.f32(n("attn_k_conv_proj.weight"), &[D, CONV * KV_HEADS], 0.2);
            b.bf16(&n("attn_k_conv_proj.weight"));
            b.f32(n("attn_v_conv_proj.weight"), &[D, CONV * KV_HEADS], 0.2);
            b.bf16(&n("attn_v_conv_proj.weight"));
        }
        if i > 0 {
            let queries = if swa(i) { 1 } else { 3 };
            b.f32(n("attn_res_query.weight"), &[D, queries], 3.0);
            if queries == 1 {
                b.reshape(&n("attn_res_query.weight"), &[D]);
            }
        }
        b.f32(n("ffn_res_query.weight"), &[D], 3.0);
        b.vector(n("ffn_norm.weight"), D, 1.0);
        b.f32(n("ffn_gate_inp.weight"), &[D, e], 0.1);
        b.f32(n("exp_probs_b.bias"), &[e], 0.5);
        let experts = |b: &mut Builder, name: String, dims: &[u64]| match nvfp4 && i == 0 {
            true => b.quant(DType::Nvfp4, name, dims, 0.1),
            false => b.quant(DType::Q8_0, name, dims, 0.1),
        };
        experts(&mut b, n("ffn_gate_up_exps.weight"), &[D, 2 * FFN, e]);
        experts(&mut b, n("ffn_down_exps.weight"), &[FFN, D, e]);
        if i < SCALED {
            b.vector(n("ffn_gate_up_exps.scale"), e, 1.0);
            b.vector(n("ffn_down_exps.scale"), e, 1.0);
        }
        if (LOOP_START..LOOP_END).contains(&i) {
            b.f32(n("loop_msg_query.weight"), &[D, 4], 3.0);
            for c in ["q", "k", "v", "ffn"] {
                b.f32(
                    n(&format!("loop_{c}_ctrl_down.weight")),
                    &[2 * D, GATE_RANK],
                    0.1,
                );
                b.f32(
                    n(&format!("loop_{c}_ctrl_gain.weight")),
                    &[GATE_RANK, D],
                    0.3,
                );
                b.f32(
                    n(&format!("loop_{c}_ctrl_forget.weight")),
                    &[GATE_RANK, D],
                    0.3,
                );
            }
        }
    }
    write(&kv, &b.tensors)
}

/// GPT-2's byte-level character for byte `b`.
fn byte_token(b: u8) -> String {
    let keep = |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let c = if keep(b) {
        u32::from(b)
    } else {
        256 + (0..b).filter(|&x| !keep(x)).count() as u32
    };
    char::from_u32(c).expect("below U+0144").to_string()
}

/// A DFlash drafter GGUF for the tiny model ([`crate::DflashDrafter`]): a sliding layer (a window
/// of 24, shorter than the tests' chats) and a full one of the tiny model's shape over the taps of both target
/// layers, a block of 16, random weights (`seed`).
pub fn dflash_gguf(seed: u64) -> Vec<u8> {
    dflash_gguf_with(seed, "mask_token_id", Value::U32(VOCAB - 1))
}

/// [`dflash_gguf`] with the `dflash.{key}` entry set to `val`.
pub fn dflash_gguf_with(seed: u64, key: &str, val: Value) -> Vec<u8> {
    dflash(seed, key, val, false)
}

/// A tiny DFlash2 drafter for the tiny models: [`dflash_gguf`]'s layers, both sliding over 24
/// positions, a block of 8, each layer's two dynamic convs, a selector of rank 32 over 16
/// candidates, and a mask embedding.
pub fn dflash2_gguf(seed: u64) -> Vec<u8> {
    dflash(seed, "block_size", Value::U32(8), true)
}

fn dflash(seed: u64, key: &str, val: Value, v2: bool) -> Vec<u8> {
    let u = |v: u64| Value::U32(v as u32);
    let k = |key: &str| format!("dflash.{key}");
    let kv = vec![
        (
            "general.architecture".to_string(),
            Value::Str("dflash".into()),
        ),
        (k("embedding_length"), u(D)),
        (k("block_count"), u(2)),
        (k("feed_forward_length"), u(FFN)),
        (k("attention.head_count"), u(HEADS)),
        (k("attention.head_count_kv"), u(KV_HEADS)),
        (k("attention.key_length"), u(HEAD_DIM)),
        (k("vocab_size"), Value::U32(VOCAB)),
        (k("block_size"), u(16)),
        (k("mask_token_id"), Value::U32(VOCAB - 1)),
        (k("target_layers"), Value::Arr(vec![u(0), u(1)])),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        (k("rope.freq_base"), Value::F32(10000.0)),
        (
            k("attention.sliding_window"),
            Value::Arr(vec![u(24), u(if v2 { 24 } else { 0 })]),
        ),
    ]
    .into_iter()
    .chain(
        [(k("conv_group_size"), u(16)), (k("selector_top_k"), u(16))]
            .into_iter()
            .filter(|_| v2),
    )
    .map(|(name, v)| {
        if name == k(key) {
            (name, val.clone())
        } else {
            (name, v)
        }
    })
    .collect::<Vec<_>>();
    let mut b = Builder {
        rng: Rng(seed),
        ty: 0,
        tensors: Vec::new(),
    };
    let (qd, kvd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    for i in 0..2 {
        let n = |s: &str| format!("blk.{i}.{s}");
        b.vector(n("attn_norm.weight"), D, 1.0);
        b.matrix(n("attn_q.weight"), qd, D);
        b.matrix(n("attn_k.weight"), kvd, D);
        b.matrix(n("attn_v.weight"), kvd, D);
        b.matrix(n("attn_output.weight"), D, qd);
        b.vector(n("attn_q_norm.weight"), HEAD_DIM, 1.0);
        b.vector(n("attn_k_norm.weight"), HEAD_DIM, 1.0);
        b.vector(n("ffn_norm.weight"), D, 1.0);
        b.matrix(n("ffn_gate.weight"), FFN, D);
        b.matrix(n("ffn_up.weight"), FFN, D);
        b.matrix(n("ffn_down.weight"), D, FFN);
        for c in if v2 { ["attn", "ffn"].as_slice() } else { &[] } {
            b.vector(n(&format!("{c}_conv.base")), 4 * D, 0.5);
            b.matrix(n(&format!("{c}_conv.weight")), 4 * D / 16, D);
        }
    }
    b.matrix("fc.weight".into(), D, 2 * D);
    b.vector("hidden_norm.weight".into(), D, 1.0);
    b.vector("output_norm.weight".into(), D, 1.0);
    if v2 {
        b.matrix("selector.weight".into(), 32, D);
        b.vector("mask_embedding".into(), D, 0.0);
        // Q8_0 codebooks, last: the drafter keeps them on the host.
        for name in ["selector.pred", "selector.succ"] {
            let data = weights(&mut b.rng, DType::Q8_0, u64::from(VOCAB), 32, 1.0);
            let ty = ggml_type(DType::Q8_0).expect("a GGML type");
            b.tensors
                .push((name.into(), vec![32, u64::from(VOCAB)], ty, data));
        }
    }
    write(&kv, &b.tensors)
}

struct Builder {
    rng: Rng,
    ty: usize,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Builder {
    /// Layer `i`'s feed-forward block: dense, or the routed experts (and qwen35moe's shared one).
    fn ffn(&mut self, arch: Arch, i: u32) {
        let n = |s: &str| format!("blk.{i}.{s}");
        if matches!(arch, Arch::Qwen3Moe | Arch::Qwen35Moe) {
            let e = u64::from(EXPERTS);
            self.f32_matrix(n("ffn_gate_inp.weight"), e, D);
            self.experts(n("ffn_gate_exps.weight"), e, EXPERT_FFN, D);
            self.experts(n("ffn_up_exps.weight"), e, EXPERT_FFN, D);
            self.experts(n("ffn_down_exps.weight"), e, D, EXPERT_FFN);
            if arch == Arch::Qwen35Moe {
                self.vector(n("ffn_gate_inp_shexp.weight"), D, 0.0);
                self.matrix(n("ffn_gate_shexp.weight"), SHARED_FFN, D);
                self.matrix(n("ffn_up_shexp.weight"), SHARED_FFN, D);
                self.matrix(n("ffn_down_shexp.weight"), D, SHARED_FFN);
            }
        } else {
            self.matrix(n("ffn_gate.weight"), FFN, D);
            self.matrix(n("ffn_up.weight"), FFN, D);
            self.matrix(n("ffn_down.weight"), D, FFN);
        }
    }

    /// A Gated DeltaNet conv weight, f32 `channels x kernel`, of about a unit sum per channel.
    fn conv(&mut self, name: String, channels: u64, kernel: u64) {
        let data = (0..channels * kernel)
            .flat_map(|_| (0.25 + self.rng.uniform(0.2)).to_le_bytes())
            .collect();
        self.tensors.push((name, vec![kernel, channels], 0, data)); // GGML F32
    }

    /// The next weight type in turn, so the model covers all of them.
    fn matrix(&mut self, name: String, rows: u64, cols: u64) {
        let t = TYPES[self.ty % TYPES.len()];
        self.ty += 1;
        let data = weights(&mut self.rng, t, rows, cols, 0.1);
        self.tensors.push((
            name,
            vec![cols, rows],
            ggml_type(t).expect("a GGML type"),
            data,
        ));
    }

    /// `experts` stacked `rows x cols` matrices of the next weight type, as GGUF stores MoE experts.
    fn experts(&mut self, name: String, experts: u64, rows: u64, cols: u64) {
        let t = TYPES[self.ty % TYPES.len()];
        self.ty += 1;
        let data = weights(&mut self.rng, t, experts * rows, cols, 0.1);
        let dims = vec![cols, rows, experts];
        self.tensors
            .push((name, dims, ggml_type(t).expect("a GGML type"), data));
    }

    /// An f32 `rows x cols` matrix, as llama.cpp keeps MoE routers. Small, so that the paths a step can
    /// take (prefill's dequant + gemm, decode's mmvq) seldom reorder near-tied experts.
    fn f32_matrix(&mut self, name: String, rows: u64, cols: u64) {
        let data = (0..rows * cols)
            .flat_map(|_| self.rng.uniform(0.1).to_le_bytes())
            .collect();
        self.tensors.push((name, vec![cols, rows], 0, data)); // GGML F32
    }

    /// A Q8_0 tensor of GGML dims `dims` (the first a multiple of 32) with weights of about `a`.
    fn q8(&mut self, name: String, dims: &[u64], a: f32) {
        self.quant(DType::Q8_0, name, dims, a);
    }

    /// A tensor of quant type `t` and GGML dims `dims` (the first whole blocks) with weights of about `a`.
    fn quant(&mut self, t: DType, name: String, dims: &[u64], a: f32) {
        let rows = dims[1..].iter().product();
        let data = weights(&mut self.rng, t, rows, dims[0], a);
        let ty = ggml_type(t).expect("a GGML type");
        self.tensors.push((name, dims.to_vec(), ty, data));
    }

    /// An f32 tensor of GGML dims `dims`, uniform in [-a, a).
    fn f32(&mut self, name: String, dims: &[u64], a: f32) {
        let data = (0..dims.iter().product::<u64>())
            .flat_map(|_| self.rng.uniform(a).to_le_bytes())
            .collect();
        self.tensors.push((name, dims.to_vec(), 0, data)); // GGML F32
    }

    /// Gives tensor `name` the GGML dims `dims`, of the same element count.
    fn reshape(&mut self, name: &str, dims: &[u64]) {
        let t = self
            .tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .expect("built");
        assert_eq!(t.1.iter().product::<u64>(), dims.iter().product::<u64>());
        t.1 = dims.to_vec();
    }

    /// Tensor `name`, built f32, rewritten as bf16 (GGML type 30, truncated).
    fn bf16(&mut self, name: &str) {
        let t = self
            .tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .expect("built");
        t.3 = t.3.chunks_exact(4).flat_map(|b| [b[2], b[3]]).collect();
        t.2 = 30;
    }

    /// f32 values around `mid`, for norms and biases.
    fn vector(&mut self, name: String, n: u64, mid: f32) {
        let data = (0..n)
            .flat_map(|_| (mid + self.rng.uniform(0.1)).to_le_bytes())
            .collect();
        self.tensors.push((name, vec![n], 0, data)); // GGML F32
    }
}

/// A generated InternVL projector: 56-pixel tiles of 14-pixel patches (16 patches, 4 rows a tile),
/// a 64-wide ViT of two layers with one head, and an output `d_model` wide. Each tensor as (name,
/// GGML dims, values exact in f16, stored as f16), so a test can rebuild the graph from the values.
pub struct Projector {
    pub tensors: Vec<(String, Vec<u64>, Vec<f32>, bool)>,
    pub kv: Vec<(String, Value)>,
}

/// [`Projector`]'s widths and geometry.
pub const VIT: u64 = 64;
pub const VIT_FFN: u64 = 128;
pub const VIT_LAYERS: u32 = 2;
pub const TILE: u32 = 56;
pub const PATCH: u32 = 14;
/// [`Projector::qwen`]'s widths and geometry.
pub const QWEN_VIT: u64 = 128;
pub const QWEN_VIT_FFN: u64 = 256;
pub const QWEN_VIT_LAYERS: u32 = 2;
pub const QWEN_HEADS: u32 = 2;
pub const QWEN_POS_SIDE: u32 = 6;

impl Projector {
    pub fn new(d_model: u32, seed: u64) -> Projector {
        let mut rng = Rng(seed);
        // k / 1024 with |k| <= 512 · scale: exact in f16.
        let mut h = |scale: i64| {
            let k = (rng.next() % (1024 * scale as u64 + 1)) as i64 - 512 * scale;
            k as f32 / 1024.0
        };
        let n = u64::from((TILE / PATCH) * (TILE / PATCH));
        let (e, f, dm) = (VIT, VIT_FFN, u64::from(d_model));
        let mut tensors = Vec::new();
        let mut put = |name: String, dims: Vec<u64>, scale: i64, f16: bool| {
            let len = dims.iter().product::<u64>() as usize;
            tensors.push((name, dims, (0..len).map(|_| h(scale)).collect(), f16));
        };
        put("v.patch_embd.weight".into(), vec![14, 14, 3, e], 1, true);
        put("v.patch_embd.bias".into(), vec![e], 1, false);
        put("v.class_embd".into(), vec![e, 1, 1], 1, false);
        put("v.position_embd.weight".into(), vec![e, n + 1, 1], 1, false);
        for l in 0..VIT_LAYERS {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            for (name, rows, cols) in [
                ("attn_q", e, e),
                ("attn_k", e, e),
                ("attn_v", e, e),
                ("attn_out", e, e),
                ("ffn_up", f, e),
                ("ffn_down", e, f),
            ] {
                put(b(&format!("{name}.weight")), vec![cols, rows], 1, true);
                put(b(&format!("{name}.bias")), vec![rows], 1, false);
            }
            for name in ["ln1", "ln2"] {
                put(b(&format!("{name}.weight")), vec![e], 2, false);
                put(b(&format!("{name}.bias")), vec![e], 1, false);
            }
            put(b("ls1.weight"), vec![e], 1, false);
            put(b("ls2.weight"), vec![e], 1, false);
        }
        put("mm.model.mlp.0.weight".into(), vec![4 * e], 2, false);
        put("mm.model.mlp.0.bias".into(), vec![4 * e], 1, false);
        put("mm.model.mlp.1.weight".into(), vec![4 * e, dm], 1, true);
        put("mm.model.mlp.1.bias".into(), vec![dm], 1, false);
        put("mm.model.mlp.3.weight".into(), vec![dm, dm], 1, true);
        put("mm.model.mlp.3.bias".into(), vec![dm], 1, false);
        let u = Value::U32;
        let f3 = |a: [f32; 3]| Value::Arr(a.iter().map(|&v| Value::F32(v)).collect());
        let kv = vec![
            ("general.architecture".into(), Value::Str("clip".into())),
            ("clip.projector_type".into(), Value::Str("internvl".into())),
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            ("clip.use_gelu".into(), Value::Bool(true)),
            ("clip.vision.image_size".into(), u(TILE)),
            ("clip.vision.patch_size".into(), u(PATCH)),
            ("clip.vision.embedding_length".into(), u(e as u32)),
            ("clip.vision.feed_forward_length".into(), u(f as u32)),
            ("clip.vision.attention.head_count".into(), u(1)),
            ("clip.vision.block_count".into(), u(VIT_LAYERS)),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                Value::F32(1e-6),
            ),
            ("clip.vision.projection_dim".into(), u(d_model)),
            ("clip.vision.image_mean".into(), f3([0.485, 0.456, 0.406])),
            ("clip.vision.image_std".into(), f3([0.229, 0.224, 0.225])),
            ("clip.vision.projector.scale_factor".into(), u(2)),
        ];
        Projector { tensors, kv }
    }

    /// A generated Qwen3-VL projector (`qwen3vl_merger`): 16-pixel patches, a 6 x 6 position grid
    /// (96-pixel `image_size`), a 128-wide ViT of two layers with two heads of 64, and an output
    /// `d_model` wide; tensors as [`Projector::new`]'s.
    pub fn qwen(d_model: u32, seed: u64) -> Projector {
        let mut rng = Rng(seed);
        let mut h = |scale: i64| {
            let k = (rng.next() % (1024 * scale as u64 + 1)) as i64 - 512 * scale;
            k as f32 / 1024.0
        };
        let side = u64::from(QWEN_POS_SIDE);
        let (e, f, dm, p) = (QWEN_VIT, QWEN_VIT_FFN, u64::from(d_model), 16);
        let mut tensors = Vec::new();
        let mut put = |name: String, dims: Vec<u64>, scale: i64, f16: bool| {
            let len = dims.iter().product::<u64>() as usize;
            tensors.push((name, dims, (0..len).map(|_| h(scale)).collect(), f16));
        };
        put("v.patch_embd.weight".into(), vec![p, p, 3, e], 1, true);
        put("v.patch_embd.weight.1".into(), vec![p, p, 3, e], 1, true);
        put("v.patch_embd.bias".into(), vec![e], 1, false);
        put(
            "v.position_embd.weight".into(),
            vec![e, side * side],
            1,
            false,
        );
        for l in 0..QWEN_VIT_LAYERS {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            for (name, rows, cols) in [
                ("attn_qkv", 3 * e, e),
                ("attn_out", e, e),
                ("ffn_up", f, e),
                ("ffn_down", e, f),
            ] {
                put(b(&format!("{name}.weight")), vec![cols, rows], 1, true);
                put(b(&format!("{name}.bias")), vec![rows], 1, false);
            }
            for name in ["ln1", "ln2"] {
                put(b(&format!("{name}.weight")), vec![e], 2, false);
                put(b(&format!("{name}.bias")), vec![e], 1, false);
            }
        }
        put("v.post_ln.weight".into(), vec![e], 2, false);
        put("v.post_ln.bias".into(), vec![e], 1, false);
        put("mm.0.weight".into(), vec![4 * e, 4 * e], 1, true);
        put("mm.0.bias".into(), vec![4 * e], 1, false);
        put("mm.2.weight".into(), vec![4 * e, dm], 1, true);
        put("mm.2.bias".into(), vec![dm], 1, false);
        let u = Value::U32;
        let f3 = |a: [f32; 3]| Value::Arr(a.iter().map(|&v| Value::F32(v)).collect());
        let kv = vec![
            ("general.architecture".into(), Value::Str("clip".into())),
            (
                "clip.projector_type".into(),
                Value::Str("qwen3vl_merger".into()),
            ),
            ("clip.has_vision_encoder".into(), Value::Bool(true)),
            ("clip.use_gelu".into(), Value::Bool(true)),
            ("clip.vision.image_size".into(), u(QWEN_POS_SIDE * 16)),
            ("clip.vision.patch_size".into(), u(16)),
            ("clip.vision.embedding_length".into(), u(e as u32)),
            ("clip.vision.feed_forward_length".into(), u(f as u32)),
            ("clip.vision.attention.head_count".into(), u(QWEN_HEADS)),
            ("clip.vision.block_count".into(), u(QWEN_VIT_LAYERS)),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                Value::F32(1e-6),
            ),
            ("clip.vision.projection_dim".into(), u(d_model)),
            ("clip.vision.image_mean".into(), f3([0.5; 3])),
            ("clip.vision.image_std".into(), f3([0.5; 3])),
            ("clip.vision.spatial_merge_size".into(), u(2)),
            (
                "clip.vision.is_deepstack_layers".into(),
                Value::Arr(vec![Value::Bool(false); QWEN_VIT_LAYERS as usize]),
            ),
        ];
        Projector { tensors, kv }
    }

    /// The GGUF file.
    pub fn gguf(&self) -> Vec<u8> {
        let tensors: Vec<_> = (self.tensors.iter())
            .map(|(name, dims, v, f16_)| {
                let bytes = match f16_ {
                    true => v.iter().flat_map(|&x| f16(x)).collect(),
                    false => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                };
                (name.clone(), dims.clone(), u32::from(*f16_), bytes)
            })
            .collect();
        write(&self.kv, &tensors)
    }

    /// A tensor's values.
    pub fn get(&self, name: &str) -> &[f32] {
        &(self.tensors.iter().find(|t| t.0 == name))
            .expect("a projector tensor")
            .2
    }
}
