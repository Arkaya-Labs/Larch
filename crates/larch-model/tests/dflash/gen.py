#!/usr/bin/env python3
"""Regenerates the fixtures of tests/dflash.rs, deterministically: a random tiny DFlash checkpoint
(bf16 safetensors) converted by scripts/dflash_convert.py to tests/data/dflash-tiny.gguf, and
tests/data/dflash-case.gguf with a target embedding and head (Q8_0), the target taps of three
rounds and the logits the NumPy reference below gives for them; and the same for a DFlash2
checkpoint (dflash2-tiny.gguf, dflash2-case.gguf), whose case also holds each round's guesses and
the selector's margin at each; and dflash2-q4k.gguf, a DFlash2 checkpoint of 256-wide rows
converted with --quant q4_k, which tests/dflash.rs loads and proposes with (no reference).

    python3 crates/larch-model/tests/dflash/gen.py

The reference is z-lab's dflash/model_mlx.py DFlashDraftModel forward in f64 numpy, on the same
weights the GGUF holds (Q8_0 matrices dequantized). Layer 0 is a sliding layer: causal within the
block, and a context window that starts at the page (PAGE positions) holding z-lab's first visible
position, as larch's drafter attends through whole pages.

The DFlash2 reference is Splash's DFlashDraft.cpp, draft.metal and the draft_select kernels of
sampling.metal (Apache-2.0), written out in f64: each half of a layer between a dynamic conv of its
norm and the conv of its output added into the residual, block rows that see each other, and the
greedy selector over the top 16 candidates."""

import importlib.util
import json
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[4]
DATA = ROOT / "crates/larch-model/tests/data"
spec = importlib.util.spec_from_file_location("conv", ROOT / "scripts/dflash_convert.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

CFG = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 128,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "num_target_layers": 8,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 20,
    "dflash_config": {"block_size": 16, "mask_token_id": 127, "target_layer_ids": [1, 3, 5]},
}
# Rounds on one sequence: cut the cache back to `truncate` (None: keep), observe tapped rows
# first..end of `n`, then propose `guesses` after `anchor`.
ROUNDS = [
    dict(truncate=None, n=40, first=0, end=37, anchor=5, guesses=15),
    dict(truncate=None, n=16, first=3, end=7, anchor=9, guesses=7),
    dict(truncate=38, n=6, first=0, end=6, anchor=0, guesses=15),
]
# Positions per page of larch's draft cache (dflash.rs PAGE).
PAGE = 16
CFG2 = {
    **CFG,
    "architectures": ["DFlash2DraftModel"],
    "layer_types": ["sliding_attention"] * 2,
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "mask_token_id": 127,
        "target_layer_ids": [1, 3, 5],
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "selector_rank": 32,
        "selector_top_k": 16,
    },
}
# Past the ring of a 20-token window (96 positions), then cut back by a check's rejected rows.
ROUNDS2 = [
    dict(truncate=None, n=64, first=0, end=60, anchor=5, guesses=7),
    dict(truncate=None, n=50, first=0, end=50, anchor=9, guesses=7),
    dict(truncate=105, n=4, first=1, end=4, anchor=0, guesses=3),
]


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def safetensors(path, w):
    header, blobs, at = {}, [], 0
    for name, x in w.items():
        b = bf16(x).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(x.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    hj = json.dumps(header).encode()
    hj += b" " * (-len(hj) % 8)
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def checkpoint(rng, path, cfg=CFG):
    d, hd, f = cfg["hidden_size"], cfg["head_dim"], cfg["intermediate_size"]
    h, kvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    taps = len(cfg["dflash_config"]["target_layer_ids"])
    v2 = "DFlash2DraftModel" in cfg["architectures"]

    def mat(n, k):
        return rng.standard_normal((n, k)) / np.sqrt(k)

    def norm(n):
        return 1 + 0.2 * rng.standard_normal(n)

    w = {"fc.weight": mat(d, taps * d), "hidden_norm.weight": norm(d), "norm.weight": norm(d)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        w |= {
            p + "input_layernorm.weight": norm(d),
            p + "post_attention_layernorm.weight": norm(d),
            p + "self_attn.q_proj.weight": mat(h * hd, d),
            p + "self_attn.k_proj.weight": mat(kvh * hd, d),
            p + "self_attn.v_proj.weight": mat(kvh * hd, d),
            p + "self_attn.o_proj.weight": mat(d, h * hd),
            p + "self_attn.q_norm.weight": norm(hd),
            p + "self_attn.k_norm.weight": norm(hd),
            p + "mlp.gate_proj.weight": mat(f, d),
            p + "mlp.up_proj.weight": mat(f, d),
            p + "mlp.down_proj.weight": mat(d, f),
        }
        for c in ("attention_conv", "mlp_conv") if v2 else ():
            w[f"{p}{c}.base_kernel"] = 0.5 + 0.2 * rng.standard_normal((2, 2, d))
            w[f"{p}{c}.kernel_projection.weight"] = 0.5 * mat(4 * d // 16, d)
    if v2:
        rank, vocab = cfg["dflash_config"]["selector_rank"], cfg["vocab_size"]
        w["candidate_selector.hidden_projection.weight"] = mat(rank, d)
        w["candidate_selector.predecessor_codebook"] = rng.standard_normal((vocab, rank))
        w["candidate_selector.successor_codebook"] = rng.standard_normal((vocab, rank)) / np.sqrt(rank)
        safetensors(path / "mask_embedding.safetensors", {"mask_embedding": rng.standard_normal(d)})
    safetensors(path / "model.safetensors", w)
    (path / "config.json").write_text(json.dumps(cfg))


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    """mlx's non-traditional rope of x [L, heads, hd] at positions pos [L]."""
    half = x.shape[-1] // 2
    ang = pos[:, None] * theta ** (-np.arange(half) * 2 / x.shape[-1])
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def dyn_conv(x, dyn, base, stage):
    """draft.metal's draft_conv: each channel of a row and of the row before, each by its base tap
    plus its group's dynamic weight; dyn [L, stage, tap, groups], base [stage, tap, d]."""
    dyn = np.repeat(dyn.reshape(len(x), 2, 2, -1), x.shape[1] // dyn.shape[1] * 4, -1)[:, stage]
    out = x * (base[stage, 0] + dyn[:, 0])
    out[1:] += x[:-1] * (base[stage, 1] + dyn[1:, 1])
    return out


class Reference:
    """DFlashDraftModel with a KVCache per layer, as model_mlx.py runs it; with cfg CFG2, the
    DFlash2 draft as Splash runs it."""

    def __init__(self, w, embed, head, cfg=CFG):
        self.w, self.embed, self.head, self.cfg = w, embed, head, cfg
        empty = np.zeros((0, cfg["num_key_value_heads"], cfg["head_dim"]))
        self.cache = [[empty, empty] for _ in range(cfg["num_hidden_layers"])]

    def truncate(self, n):
        self.cache = [[k[:n], v[:n]] for k, v in self.cache]

    def __call__(self, block, target_hidden):
        """hidden_states(inputs, target_hidden, cache, logits_start=1), then compute_logits."""
        CFG, v2 = self.cfg, self.cfg is CFG2
        w, eps, theta = self.w, CFG["rms_norm_eps"], CFG["rope_theta"]
        H, KV, hd = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
        h = self.embed[block]
        if v2:
            h[1:] = w["mask_embedding"]
        x_ctx = rms(target_hidden @ w["fc.weight"].T, w["hidden_norm.weight"], eps)
        L, S = len(block), len(x_ctx)
        for i, c in enumerate(self.cache):

            def p(n, i=i):
                return w[f"layers.{i}.{n}"]

            def half(x, conv, f, h):
                """h + f(x) between the conv's two stages, DFlash2's; DFlash's h + f(x)."""
                if not v2:
                    return h + f(x)
                dyn = x @ p(f"{conv}.kernel_projection.weight").T
                base = p(f"{conv}.base_kernel")
                return h + dyn_conv(f(dyn_conv(x, dyn, base, 0)), dyn, base, 1)

            x = rms(h, p("input_layernorm.weight"), eps)
            h = half(x, "attention_conv", lambda x: self.attend(i, x, x_ctx), h)
            x = rms(h, p("post_attention_layernorm.weight"), eps)

            def mlp(x):
                g, u = x @ p("mlp.gate_proj.weight").T, x @ p("mlp.up_proj.weight").T
                return (g / (1 + np.exp(-g)) * u) @ p("mlp.down_proj.weight").T

            h = half(x, "mlp_conv", mlp, h)
        hn = rms(h[1:], w["norm.weight"], eps)
        return hn @ self.head.T, hn

    def attend(self, i, x, x_ctx):
        CFG, v2, c = self.cfg, self.cfg is CFG2, self.cache[i]
        w, eps, theta = self.w, CFG["rms_norm_eps"], CFG["rope_theta"]
        H, KV, hd = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
        L, S = len(x), len(x_ctx)
        if True:

            def p(n, i=i):
                return w[f"layers.{i}.{n}"]

            off = len(c[0])
            q = rms((x @ p("self_attn.q_proj.weight").T).reshape(L, H, hd), p("self_attn.q_norm.weight"), eps)
            ck = rms((x_ctx @ p("self_attn.k_proj.weight").T).reshape(S, KV, hd), p("self_attn.k_norm.weight"), eps)
            cv = (x_ctx @ p("self_attn.v_proj.weight").T).reshape(S, KV, hd)
            pk = rms((x @ p("self_attn.k_proj.weight").T).reshape(L, KV, hd), p("self_attn.k_norm.weight"), eps)
            pv = (x @ p("self_attn.v_proj.weight").T).reshape(L, KV, hd)
            q = rope(q, off + S + np.arange(L), theta)
            ck = rope(ck, off + np.arange(S), theta)
            pk = rope(pk, off + S + np.arange(L), theta)
            c[0], c[1] = np.concatenate([c[0], ck]), np.concatenate([c[1], cv])
            keys, vals = np.concatenate([c[0], pk]), np.concatenate([c[1], pv])
            keys, vals = np.repeat(keys, H // KV, 1), np.repeat(vals, H // KV, 1)
            sc = np.einsum("lhd,shd->hls", q, keys) / np.sqrt(hd)
            if CFG["layer_types"][i] == "sliding_attention":
                # z-lab: key k visible to query p if k <= p and p - k < window; larch from that
                # key's page on.
                query, key = off + S + np.arange(L)[:, None], np.arange(off + S + L)[None]
                first = np.maximum(query + 1 - CFG["sliding_window"], 0) // PAGE * PAGE
                # DFlash2 (draft_attention_split_phase): every block row sees the whole block.
                seen = key < off + S + L if v2 else key <= query
                sc = np.where(seen & (key >= first), sc, -np.inf)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            o = np.einsum("hls,shd->lhd", sc / sc.sum(-1, keepdims=True), vals).reshape(L, H * hd)
            return o @ p("self_attn.o_proj.weight").T


def select(w, cfg, anchor, logits, hn, n):
    """draft_select_edges and draft_select_dflash, greedy: the guesses and each one's margin over
    the runner-up."""
    top_k, prev, out, margins = cfg["dflash_config"]["selector_top_k"], anchor, [], []
    sel = hn @ w["candidate_selector.hidden_projection.weight"].T
    for p in range(n):
        cands = sorted(range(len(logits[p])), key=lambda t: (-logits[p][t], t))[:top_k]
        ctx = w["candidate_selector.predecessor_codebook"][prev] * sel[p]
        score = [logits[p][t] + ctx @ w["candidate_selector.successor_codebook"][t] for t in cands]
        order = np.argsort(-np.array(score), kind="stable")
        prev = cands[order[0]]
        out.append(prev)
        margins.append(score[order[0]] - score[order[1]])
    return out, margins


def main():
    fixtures(np.random.default_rng(2602), CFG, ROUNDS, "dflash")
    fixtures(np.random.default_rng(2603), CFG2, ROUNDS2, "dflash2")
    # Q4_K rows but o_proj's and down's of 64 (Q8_0), one tap: the smallest that mixes them.
    q4 = {**CFG2, "hidden_size": 256, "num_hidden_layers": 1, "num_attention_heads": 2,
          "num_key_value_heads": 1, "intermediate_size": 64, "layer_types": ["sliding_attention"],
          "dflash_config": {**CFG2["dflash_config"], "target_layer_ids": [1]}}
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint(np.random.default_rng(2604), pathlib.Path(tmp), q4)
        conv.convert(tmp, DATA / "dflash2-q4k.gguf", "q4_k")


def fixtures(rng, CFG, ROUNDS, name):
    d, vocab = CFG["hidden_size"], CFG["vocab_size"]
    k = len(CFG["dflash_config"]["target_layer_ids"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        checkpoint(rng, tmp, CFG)
        conv.convert(tmp, DATA / f"{name}-tiny.gguf")
        w = {}
        for f in sorted(tmp.glob("*.safetensors")):
            for n, (shape, load) in conv.read_safetensors(f).items():
                x = load()
                w[n] = (conv.dequant_q8_0(conv.q8_0(x), shape) if len(shape) == 2 else x).astype(np.float64)

    def q8(x):
        b = conv.q8_0(x)
        return b, conv.dequant_q8_0(b, x.shape).astype(np.float64)

    embed, head = q8(rng.standard_normal((vocab, d))), q8(0.3 * rng.standard_normal((vocab, d)))
    ref = Reference(w, embed[1], head[1], CFG)
    kv = {"case.rounds": len(ROUNDS)}
    tensors = [
        ("token_embd.weight", [vocab, d], conv.GGML_Q8_0, lambda: embed[0]),
        ("output.weight", [vocab, d], conv.GGML_Q8_0, lambda: head[0]),
    ]
    scale = np.array([1.0, 4.0, 20.0])[:, None, None]
    for i, r in enumerate(ROUNDS):
        if r["truncate"] is not None:
            ref.truncate(r["truncate"])
        taps = (scale * rng.standard_normal((k, r["n"], d))).astype(np.float32)
        # model_mlx.py concatenates the taps per position, in target_layer_ids order.
        hidden = np.concatenate(list(taps[:, r["first"] : r["end"]].astype(np.float64)), -1)
        rows = CFG["dflash_config"]["block_size"] - 1 if CFG is CFG2 else r["guesses"]
        block = np.array([r["anchor"]] + [CFG["dflash_config"]["mask_token_id"]] * rows)
        logits, hn = ref(block, hidden)
        logits, hn = logits[: r["guesses"]], hn[: r["guesses"]]
        if CFG is CFG2:
            ids, margins = select(w, CFG, r["anchor"], logits, hn, r["guesses"])
            tensors += [
                (f"guesses.{i}", [len(ids)], conv.GGML_F32, lambda v=ids: np.array(v, np.float32).tobytes()),
                (f"margins.{i}", [len(ids)], conv.GGML_F32, lambda v=margins: np.array(v, np.float32).tobytes()),
            ]
        for key in ("first", "end", "anchor", "guesses"):
            kv[f"case.{i}.{key}"] = r[key]
        kv[f"case.{i}.truncate"] = 0xFFFFFFFF if r["truncate"] is None else r["truncate"]
        tensors += [
            (f"taps.{i}", [k * r["n"], d], conv.GGML_F32, lambda t=taps: t.tobytes()),
            (f"logits.{i}", list(logits.shape), conv.GGML_F32, lambda lg=logits: lg.astype(np.float32).tobytes()),
        ]
    conv.write_gguf(DATA / f"{name}-case.gguf", kv, tensors)


if __name__ == "__main__":
    main()
