#!/usr/bin/env python3
"""A DFlash block drafter checkpoint (Hugging Face config.json + *.safetensors) as a larch GGUF.
Python 3.11+ and numpy.

    python3 scripts/dflash_convert.py [--quant q4_k] <checkpoint dir> <out.gguf>

Matrices go to Q8_0 (mmvq's weights; --quant q4_k: Q4_K where a row is whole 256-value blocks),
norms to f32. The embedding and lm_head are the target's, so a checkpoint's copies are dropped.
Sliding layers (causal, as z-lab's default) keep their window per layer. A DFlash2 checkpoint
(DFlash2DraftModel, its directory as Splash's Hub cache holds it, mask_embedding.safetensors
included) adds each layer's two dynamic convs, the candidate selector and its codebooks (written
last: larch keeps them on the host) and the mask embedding; its block rows see each other in every
layer. Refuses what larch's drafter does not run: causal full or bidirectional sliding DFlash
layers (is_causal, dflash_config.causal), a causal DFlash2, use_swa, swa_window_size, rope scaling,
logit softcap or scales.
"""

import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

GGML_F32, GGML_Q8_0, GGML_Q4_K = 0, 8, 12
# (values, bytes) of a block.
BLOCK = {GGML_F32: (1, 4), GGML_Q8_0: (32, 34), GGML_Q4_K: (256, 144)}
ALIGN = 32

LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    # DFlash2's dynamic convs: base taps [stage, tap, channel], flattened, and their projection.
    "attention_conv.base_kernel": "attn_conv.base",
    "attention_conv.kernel_projection.weight": "attn_conv.weight",
    "mlp_conv.base_kernel": "ffn_conv.base",
    "mlp_conv.kernel_projection.weight": "ffn_conv.weight",
}
TOP = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "hidden_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.hidden_projection.weight": "selector.weight",
    "candidate_selector.predecessor_codebook": "selector.pred",
    "candidate_selector.successor_codebook": "selector.succ",
    "mask_embedding": "mask_embedding",
}
# Host-only tensors, after every tensor the device holds.
LAST = ("selector.pred", "selector.succ")
SHARED = {"embed_tokens.weight", "lm_head.weight"}


def gguf_name(hf):
    """The GGUF name of a checkpoint tensor; None for the target's shared ones."""
    name = hf.removeprefix("model.")
    if name in SHARED:
        return None
    if name in TOP:
        return TOP[name]
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit() and parts[2] in LAYER:
        return f"blk.{parts[1]}.{LAYER[parts[2]]}"
    sys.exit(f"unsupported tensor {hf}")


def read_safetensors(path):
    """name -> (shape, f32 loader), reading lazily from a map."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", bytes(data[:8]))[0]
    header = json.loads(bytes(data[8 : 8 + n]))
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw, dt = data[8 + n + a : 8 + n + b], info["dtype"]
        if dt not in ("BF16", "F16", "F32"):
            sys.exit(f"{name}: dtype {dt}")

        def load(raw=raw, dt=dt, shape=info["shape"]):
            if dt == "BF16":
                x = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                x = raw.view("<f2" if dt == "F16" else "<f4").astype(np.float32)
            return x.reshape(shape)

        out[name] = (info["shape"], load)
    return out


def q8_0(x):
    """llama.cpp's Q8_0 blocks of x (f32, rows of a multiple of 32): an f16 scale amax / 127, then
    32 int8 values rounded half away from zero."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1, keepdims=True) / 127
    inv = np.divide(1, d, out=np.zeros_like(d), where=d != 0)
    q = np.trunc(b * inv + np.copysign(0.5, b)).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q4_k(x):
    """ggml's Q4_K blocks of x (f32, rows of a multiple of 256), as dequantize_row_q4_K reads them: f16 d
    and dmin, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles. Each 32-value sub-block's step
    and min come from its own min and max, not quantize_row_q4_K_ref's iterative fit."""
    b = x.reshape(-1, 8, 32).astype(np.float32)
    # ggml clamps a sub-block's min to at most 0 and stores its negation: 6 bits of dmin, unsigned.
    mn = -np.minimum(b.min(axis=2), 0)
    step = (b.max(axis=2) + mn) / 15

    def six(v):
        """A block's f16 scale of v's largest / 63, and v in 6-bit steps of it."""
        top = v.max(axis=1, keepdims=True) / 63
        q = np.clip(np.rint(np.divide(v, top, out=np.zeros_like(v), where=top > 0)), 0, 63).astype(np.uint8)
        return top.astype(np.float16), q

    (d, ls), (dmin, lm) = six(step), six(mn)
    ds = (d.astype(np.float32) * ls)[..., None]
    shifted = b + (dmin.astype(np.float32) * lm)[..., None]
    q = np.clip(np.rint(np.divide(shifted, ds, out=np.zeros_like(b), where=ds > 0)), 0, 15).astype(np.uint8)
    out = np.empty((b.shape[0], 144), np.uint8)
    out[:, 0:2], out[:, 2:4] = d.view(np.uint8), dmin.view(np.uint8)
    # get_scale_min_k4: sub-blocks 0-3 in the low 6 bits of bytes 0-7, 4-7 in nibbles of bytes 8-11
    # with their top 2 bits over bytes 0-7.
    out[:, 4:8] = ls[:, :4] | (ls[:, 4:] >> 4) << 6
    out[:, 8:12] = lm[:, :4] | (lm[:, 4:] >> 4) << 6
    out[:, 12:16] = (ls[:, 4:] & 15) | (lm[:, 4:] & 15) << 4
    # Byte l of group g: sub-block 2g's value l low, 2g + 1's high.
    out[:, 16:] = (q[:, 0::2] | q[:, 1::2] << 4).reshape(-1, 128)
    return out.tobytes()


def dequant_q8_0(raw, shape):
    """The f32 values of Q8_0 bytes, for the reference."""
    b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, 2:].copy().view(np.int8).astype(np.float32)).reshape(shape)


def put_str(o, s):
    s = s.encode()
    o += struct.pack("<Q", len(s)) + s


def put_kv(o, key, v):
    put_str(o, key)
    if isinstance(v, str):
        o += struct.pack("<I", 8)
        put_str(o, v)
    elif isinstance(v, float):
        o += struct.pack("<If", 6, v)
    elif isinstance(v, list):
        o += struct.pack("<IIQ", 9, 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)
    else:
        o += struct.pack("<II", 4, v)


def nbytes(t, shape):
    """Bytes of a tensor of ggml type t."""
    n, b = BLOCK[t]
    return int(np.prod(shape)) // n * b


def write_gguf(path, kv, tensors):
    """GGUF v3: `kv` {key: str | float | u32 | [u32]}, `tensors` [(name, shape, ggml type, bytes
    loader)], written one tensor at a time."""
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, v in kv.items():
        put_kv(head, k, v)
    off, offs = 0, []
    for name, shape, t, _ in tensors:
        put_str(head, name)
        head += struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape))
        head += struct.pack("<IQ", t, off)
        offs.append(off)
        off = -(-(off + nbytes(t, shape)) // ALIGN) * ALIGN
    head += bytes(-len(head) % ALIGN)
    with open(path, "wb") as f:
        f.write(head)
        for (_, _, _, load), at in zip(tensors, offs):
            f.seek(len(head) + at)
            f.write(load())
        f.truncate(len(head) + off)


def target_layer_ids(cfg, dflash):
    """The config's, else the torch path's default spread over the target's layers."""
    if "target_layer_ids" in dflash:
        return list(dflash["target_layer_ids"])
    if "target_layer_ids" in cfg:
        return list(cfg["target_layer_ids"])
    n, k = cfg["num_target_layers"], cfg["num_hidden_layers"]
    if k == 1:
        return [n // 2]
    return [round(1 + i * (n - 4) / (k - 1)) for i in range(k)]


def config_kv(cfg):
    dflash = cfg.get("dflash_config", {})
    v2 = "DFlash2DraftModel" in (cfg.get("architectures") or [])
    types = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    if len(types) != cfg["num_hidden_layers"] or set(types) - {"full_attention", "sliding_attention"}:
        sys.exit(f"layer_types {types} are not supported")
    for key in ("use_swa", "swa_window_size"):
        if dflash.get(key):
            sys.exit(f"dflash_config.{key} is not supported")
    # vLLM's _dflash_layer_causal: the top-level key, else dflash_config's, for every layer.
    causal = cfg.get("is_causal")
    causal = dflash.get("causal") if causal is None else causal
    if v2:
        # Splash's readDraftLayout: a block of whole rows, a conv of a row and the row before.
        if causal or dflash.get("conv_kernel_size", 2) != 2:
            sys.exit("a causal DFlash2 or a conv of other than 2 taps is not supported")
    elif causal is not None and any(bool(causal) != (t == "sliding_attention") for t in types):
        sys.exit("is_causal other than sliding layers' is not supported")
    window = cfg.get("sliding_window") or 0
    if "sliding_attention" in types and window < 1:
        sys.exit("sliding layers need a sliding_window")
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        sys.exit(f"rope scaling {rope} is not supported")
    for key, default in (("input_embedding_scale", 1.0), ("output_multiplier", 1.0)):
        if float(dflash.get(key, default)) != default:
            sys.exit(f"{key} is not supported")
    if dflash.get("final_logit_softcapping", cfg.get("final_logit_softcapping")):
        sys.exit("final_logit_softcapping is not supported")
    heads = cfg["num_attention_heads"]
    kv = {
        "general.architecture": "dflash",
        "dflash.embedding_length": cfg["hidden_size"],
        "dflash.block_count": cfg["num_hidden_layers"],
        "dflash.feed_forward_length": cfg["intermediate_size"],
        "dflash.attention.head_count": heads,
        "dflash.attention.head_count_kv": cfg["num_key_value_heads"],
        "dflash.attention.key_length": cfg.get("head_dim") or cfg["hidden_size"] // heads,
        "dflash.attention.layer_norm_rms_epsilon": float(cfg["rms_norm_eps"]),
        "dflash.rope.freq_base": float(cfg.get("rope_theta", rope.get("rope_theta", 10000.0))),
        "dflash.vocab_size": cfg["vocab_size"],
        "dflash.block_size": int(dflash.get("block_size", cfg.get("block_size", 16))),
        "dflash.mask_token_id": int(dflash.get("mask_token_id", cfg.get("mask_token_id"))),
        "dflash.target_layers": target_layer_ids(cfg, dflash),
    }
    if "sliding_attention" in types:
        kv["dflash.attention.sliding_window"] = [window if t == "sliding_attention" else 0 for t in types]
    if v2:
        kv["dflash.conv_group_size"] = int(dflash.get("conv_group_size", 16))
        kv["dflash.selector_top_k"] = int(dflash.get("selector_top_k", 16))
    return kv


def convert(src, out, quant="q8_0"):
    """Writes the GGUF; returns (weight bytes, KV bytes per 1024 positions, matrices --quant q4_k
    left Q8_0)."""
    src = pathlib.Path(src)
    cfg = json.loads((src / "config.json").read_text())
    kv = config_kv(cfg)
    found = {}
    for f in sorted(src.glob("*.safetensors")):
        for name, (shape, load) in read_safetensors(f).items():
            g = gguf_name(name)
            if g is not None:
                found[g] = (shape, load)
    tensors, kept = [], []
    for g in sorted(found, key=lambda g: (g in LAST, g)):
        shape, load = found[g]
        # The codebooks stay Q8_0: the host reads them as Q8_0 rows (dflash.rs q8_row).
        q4 = quant == "q4_k" and g not in LAST
        if len(shape) == 2 and q4 and shape[1] % 256 == 0:
            tensors.append((g, shape, GGML_Q4_K, lambda load=load: q4_k(load())))
        elif len(shape) == 2:
            if q4:
                kept.append(g)
            tensors.append((g, shape, GGML_Q8_0, lambda load=load: q8_0(load())))
        else:
            # A conv's [2, 2, d] base taps as one vector.
            tensors.append((g, [int(np.prod(shape))], GGML_F32, lambda load=load: load().tobytes()))
    write_gguf(out, kv, tensors)
    weights = sum(nbytes(t, s) for _, s, t, _ in tensors)
    per_pos = 2 * kv["dflash.block_count"] * kv["dflash.attention.head_count_kv"] * kv["dflash.attention.key_length"] * 2
    return weights, per_pos * 1024, kept


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="checkpoint directory: config.json and *.safetensors")
    p.add_argument("out", help="the GGUF to write")
    p.add_argument("--quant", choices=("q8_0", "q4_k"), default="q8_0", help="the matrices' type")
    a = p.parse_args()
    w, kv, kept = convert(a.src, a.out, a.quant)
    # Layers' matrices of one name as one blk.*. entry.
    kept = sorted({re.sub(r"^blk\.\d+\.", "blk.*.", g) for g in kept})
    note = f", Q8_0 (rows not of 256): {' '.join(kept)}" if kept else ""
    print(f"{a.out}: weights {w / 2**20:.1f} MiB, draft KV {kv / 2**20:.2f} MiB per 1k positions{note}")


if __name__ == "__main__":
    main()
