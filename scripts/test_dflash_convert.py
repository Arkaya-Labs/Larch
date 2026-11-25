"""Tests for scripts/dflash_convert.py's config refusals and Q4_K blocks: python3 -m unittest scripts/test_dflash_convert.py -v"""

import importlib.util
import pathlib
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("conv", HERE / "dflash_convert.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

CFG = {"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4,
       "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 100, "target_layer_ids": [0, 1],
       "mask_token_id": 7, "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 24}


def cfg(**kw):
    return {**CFG, **kw}


class Refusals(unittest.TestCase):
    def test_the_default_converts(self):
        self.assertEqual(conv.config_kv(CFG)["dflash.attention.sliding_window"], [24, 0])

    def test_unsupported_configs_exit(self):
        for bad in (
            cfg(is_causal=False),
            cfg(dflash_config={"causal": False}),
            cfg(dflash_config={"causal": True}, layer_types=["full_attention"] * 2),
            cfg(dflash_config={"use_swa": True}),
            cfg(dflash_config={"swa_window_size": 8}),
            cfg(layer_types=["sliding_attention", "full_attention"], sliding_window=0),
            cfg(layer_types=["full_attention"]),
        ):
            with self.assertRaises(SystemExit, msg=bad):
                conv.config_kv(bad)

    def test_dflash_causal_on_sliding_layers_converts(self):
        conv.config_kv(cfg(dflash_config={"causal": True}, layer_types=["sliding_attention"] * 2))

    def test_dflash2_converts_bidirectional_and_refuses_causal(self):
        v2 = cfg(architectures=["DFlash2DraftModel"], is_causal=False, layer_types=["sliding_attention"] * 2,
                 dflash_config={"conv_group_size": 16, "selector_top_k": 16})
        kv = conv.config_kv(v2)
        self.assertEqual((kv["dflash.conv_group_size"], kv["dflash.selector_top_k"]), (16, 16))
        for bad in ({**v2, "is_causal": True}, {**v2, "dflash_config": {"conv_kernel_size": 3}}):
            with self.assertRaises(SystemExit, msg=bad):
                conv.config_kv(bad)


def dequant_q4_k(raw):
    """ggml's dequantize_row_q4_K, block by block, as larch's CPU reference (quant.cpp q4_k) runs it."""
    out = []
    for b in np.frombuffer(raw, np.uint8).reshape(-1, 144):
        d, dmin = b[:4].copy().view(np.float16).astype(np.float32)
        s, q = b[4:16].astype(np.int32), b[16:]
        for j in range(8):
            if j < 4:
                sc, m = s[j] & 63, s[j + 4] & 63
            else:
                sc, m = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
            nib = q[32 * (j // 2) : 32 * (j // 2) + 32]
            out.append(d * np.float32(sc) * (nib >> 4 * (j % 2) & 0xF).astype(np.float32) - dmin * np.float32(m))
    return np.concatenate(out)


class Q4K(unittest.TestCase):
    def test_blocks_decode_within_a_step(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((6, 512)).astype(np.float32)
        x[0] = np.abs(x[0])  # min clamped to 0
        x[1, :256] = 0  # a zero block
        x[2, 32:64] = 3.0  # a constant sub-block
        x[3] *= np.linspace(0.01, 20, 512, dtype=np.float32)  # sub-blocks of very different ranges
        raw = conv.q4_k(x)
        self.assertEqual(len(raw), x.size // 256 * 144)
        y = dequant_q4_k(raw).reshape(x.shape)
        # Within one 4-bit step of the block's widest sub-block (rounding and the 6-bit scales).
        sub = x.reshape(-1, 8, 32)
        step = (sub.max(2) - np.minimum(sub.min(2), 0)).max(1) / 15
        err = np.abs(y - x).reshape(-1, 256).max(1)
        self.assertTrue(np.all(err <= step + 1e-6), (err, step))
        self.assertTrue(np.all(y[1, :256] == 0))
        self.assertLess(np.sqrt(np.mean((y - x)[4:] ** 2)), 0.1)


if __name__ == "__main__":
    unittest.main()
