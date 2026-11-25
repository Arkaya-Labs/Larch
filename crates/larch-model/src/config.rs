//! `ModelConfig` from GGUF metadata (architecture 8): Llama 3.x and Mistral (arch `llama`), Qwen 2.5
//! (`qwen2`, qkv bias), Qwen 3 dense (`qwen3`, qk-norm), Qwen 3 MoE (`qwen3moe`, qk-norm and
//! routed experts in every layer) and Qwen 3.5 dense and MoE (`qwen35`, `qwen35moe`: Gated DeltaNet
//! layers between gated full-attention layers, and in the MoE a shared expert beside the routed ones).

use larch_records::LoadError;

use crate::gguf::{Gguf, Value, code};

/// The GGUF architectures this crate runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen35,
    Qwen35Moe,
}

impl Arch {
    /// Qwen 3.5's hybrid layers: Gated DeltaNet and gated full attention.
    pub fn hybrid(self) -> bool {
        matches!(self, Arch::Qwen35 | Arch::Qwen35Moe)
    }
}

/// The routed experts of an MoE model, the same in every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeCfg {
    pub experts: u32,
    /// Experts each token runs (top k).
    pub used: u32,
    /// One expert's feed-forward width.
    pub ffn: u32,
}

/// Rotary embedding settings, as `larch_rope_params` takes them.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeCfg {
    pub n_dims: u32,
    /// Qwen pairs element i with i + n_dims / 2; Llama pairs neighbours.
    pub neox: bool,
    pub freq_base: f32,
    /// 1 / the linear or YaRN scaling factor.
    pub freq_scale: f32,
    pub yarn: bool,
    pub n_ctx_orig: u32,
    /// Qwen 3.5's M-RoPE: rotated pairs per position (t, h, w, e), interleaved (ggml's IMROPE). A
    /// step's positions may then be four per row; a text row's (p, p, p, 0) turn as one position p.
    /// None for a model of one position per row.
    pub sections: Option<[u32; 4]>,
}

/// The model's shape and settings, read from GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub arch: Arch,
    pub n_layers: u32,
    /// Layers that keep attention K and V in the KV pool: all but a hybrid's recurrent ones.
    pub kv_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub ctx_train: u32,
    pub rope: RopeCfg,
    pub rms_eps: f32,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub tie_embeddings: bool,
}

impl ModelConfig {
    /// `UnsupportedArch` for an architecture or rope scaling this crate does not run; `BadGguf(KEY, 0)`
    /// for a missing or out-of-range key.
    pub fn from_gguf(g: &Gguf) -> Result<ModelConfig, LoadError> {
        let arch_name = g
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or(LoadError::UnsupportedArch)?;
        let arch = match arch_name {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            "qwen35" => Arch::Qwen35,
            "qwen35moe" => Arch::Qwen35Moe,
            _ => return Err(LoadError::UnsupportedArch),
        };
        let bad = LoadError::BadGguf(code::KEY, 0);
        let get = |k: &str| g.get(&format!("{arch_name}.{k}"));
        // Absent is None; present must be a u32 (a zero is caught by the checks at the end).
        let u = |k: &str| {
            get(k)
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or(bad))
                .transpose()
        };
        let need = |k: &str| u(k)?.filter(|&v| v > 0).ok_or(bad);

        // Qwen 3.5 GGUFs may append multi-token-prediction blocks, which the main pass never runs.
        // Every block has tensors of its own, so the table (bounded by the file's size) bounds the
        // count: a hostile count must not size the loader's per-layer loops.
        let blocks = need("block_count")?;
        if blocks as usize > g.tensors.len() {
            return Err(bad);
        }
        let n_layers = blocks
            .checked_sub(u("nextn_predict_layers")?.unwrap_or(0))
            .filter(|&n| n > 0)
            .ok_or(bad)?;
        let d_model = need("embedding_length")?;
        let n_heads = need("attention.head_count")?;
        let n_kv_heads = u("attention.head_count_kv")?.unwrap_or(n_heads);
        let head_dim = u("attention.key_length")?.unwrap_or(d_model / n_heads);
        let ctx_train = need("context_length")?;
        let vocab = g
            .tensor("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
        // llama.cpp's defaults: linear when the type is absent, and the legacy factor key.
        let scaling = get("rope.scaling.type")
            .and_then(Value::as_str)
            .unwrap_or("linear");
        let factor = get("rope.scaling.factor")
            .or_else(|| get("rope.scale_linear"))
            .and_then(Value::as_f32)
            .filter(|&f| f > 0.0)
            .unwrap_or(1.0);
        let rope = RopeCfg {
            n_dims: u("rope.dimension_count")?.unwrap_or(head_dim),
            neox: arch != Arch::Llama,
            freq_base: get("rope.freq_base")
                .and_then(Value::as_f32)
                .unwrap_or(10000.0),
            freq_scale: if scaling == "none" { 1.0 } else { 1.0 / factor },
            yarn: scaling == "yarn",
            n_ctx_orig: u("rope.scaling.original_context_length")?.unwrap_or(ctx_train),
            sections: match arch {
                Arch::Qwen35 | Arch::Qwen35Moe => get("rope.dimension_sections")
                    .map(|v| sections(v).ok_or(bad))
                    .transpose()?,
                _ => None,
            },
        };
        let mut cfg = ModelConfig {
            arch,
            n_layers,
            kv_layers: n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            // qwen35moe GGUFs may leave out the dense width, which only the shared expert's falls
            // back to: the shared width stands in.
            ffn: match arch {
                Arch::Qwen35Moe => u("feed_forward_length")?
                    .filter(|&v| v > 0)
                    .or(u("expert_shared_feed_forward_length")?)
                    .unwrap_or(0),
                _ => need("feed_forward_length")?,
            },
            vocab,
            ctx_train,
            rms_eps: get("attention.layer_norm_rms_epsilon")
                .and_then(Value::as_f32)
                .ok_or(bad)?,
            // What llama.cpp builds for each arch; the loader then needs these tensors in every layer.
            qkv_bias: arch == Arch::Qwen2,
            qk_norm: matches!(
                arch,
                Arch::Qwen3 | Arch::Qwen3Moe | Arch::Qwen35 | Arch::Qwen35Moe
            ),
            tie_embeddings: g.tensor("output.weight").is_none(),
            rope,
        };
        let r = &cfg.rope;
        // The product bounds every width the loader and the engine derive (n_kv_heads <= n_heads).
        let ok = cfg.ffn > 0
            && n_heads.is_multiple_of(n_kv_heads)
            && n_heads.checked_mul(head_dim).is_some()
            && r.n_dims > 0
            && r.n_dims.is_multiple_of(2)
            && r.n_dims <= head_dim
            && r.sections
                .is_none_or(|[t, h, w, e]| t + h + w > 0 && t + h + w + e <= r.n_dims / 2);
        if !ok {
            return Err(bad);
        }
        if !matches!(scaling, "none" | "linear" | "yarn") {
            return Err(LoadError::UnsupportedArch);
        }
        cfg.kv_layers = crate::transformer::kv_layers(g, &cfg)?;
        // The KV pool pages at least one layer.
        if cfg.kv_layers == 0 {
            return Err(bad);
        }
        Ok(cfg)
    }
}

/// `rope.dimension_sections`: four counts (llama.cpp reads up to four, the rest 0), each below 2^16.
fn sections(v: &Value) -> Option<[u32; 4]> {
    let a = v.as_arr().filter(|a| a.len() <= 4)?;
    let mut out = [0u32; 4];
    for (o, v) in out.iter_mut().zip(a) {
        *o = v.as_u64().filter(|&v| v < 1 << 16)? as u32;
    }
    Some(out)
}

impl MoeCfg {
    /// The experts of a `qwen3moe` or `qwen35moe` GGUF, None for a dense arch; `BadGguf(KEY, 0)` for
    /// a missing key, or more experts used than exist. Every layer of these is an MoE layer with top-k
    /// renormalised (llama.cpp's build for the arch); qwen35moe adds a shared expert.
    pub fn from_gguf(g: &Gguf, arch: Arch) -> Result<Option<MoeCfg>, LoadError> {
        let name = match arch {
            Arch::Qwen3Moe => "qwen3moe",
            Arch::Qwen35Moe => "qwen35moe",
            _ => return Ok(None),
        };
        let bad = LoadError::BadGguf(code::KEY, 0);
        let need = |k: &str| {
            g.get(&format!("{name}.{k}"))
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or(bad)
        };
        let (experts, used) = (need("expert_count")?, need("expert_used_count")?);
        // llama.cpp's qwen3moe: feed_forward_length / expert_used_count when the key is absent (the
        // loader checks the width against the expert tensors' shapes).
        let ffn = match g.get(&format!("{name}.expert_feed_forward_length")) {
            Some(_) => need("expert_feed_forward_length")?,
            None => (need("feed_forward_length")? / used.max(1)).max(1),
        };
        let m = MoeCfg { experts, used, ffn };
        if m.used > m.experts {
            return Err(bad);
        }
        Ok(Some(m))
    }
}
