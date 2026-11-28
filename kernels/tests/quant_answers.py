# Prints the known-answer table of quant_known_answers() in cpu_reference.cpp. Per quant type: 2 rows of 512
# elements from the test's LCG (f16 scale fields masked to [0, 1)), decoded by gguf-py, llama.cpp's Python
# decoder; FNV-1a 64 of the f32 output and of its f16 rounding, then two spot values.
# Run: pip install gguf numpy && python3 quant_answers.py (NVFP4 needs gguf-py from llama.cpp b12723 or later:
# PYTHONPATH=<llama.cpp>/gguf-py)
import numpy as np
from gguf import GGMLQuantizationType as T, quants

TYPES = [("Q2_K", T.Q2_K, 12, 256, 84, (80, 82)), ("Q3_K", T.Q3_K, 11, 256, 110, (108,)), ("Q4_K", T.Q4_K, 4, 256, 144, (0, 2)), ("Q5_K", T.Q5_K, 5, 256, 176, (0, 2)),
         ("Q6_K", T.Q6_K, 6, 256, 210, (208,)), ("Q8_0", T.Q8_0, 7, 32, 34, (0,)),
         ("Q4_0", T.Q4_0, 8, 32, 18, (0,)), ("Q5_0", T.Q5_0, 9, 32, 22, (0,)),
         # NVFP4's four scales are E4M3 bytes, masked to exponents up to 7 instead (below 1).
         ("NVFP4", T.NVFP4, 13, 64, 36, ())]


def fnv(b):
    h = 0xcbf29ce484222325
    for x in b:
        h = ((h ^ x) * 0x100000001b3) % 2**64
    return h


for name, qt, seed, block, nbytes, scales in TYPES:
    s, raw = seed, bytearray()
    for _ in range(2 * 512 // block * nbytes):
        s = (s * 6364136223846793005 + 1442695040888963407) % 2**64
        raw.append(s >> 56)
    for blk in range(0, len(raw), nbytes):
        for o in scales:
            raw[blk + o + 1] &= 0x3b
        for o in range(4 if qt == T.NVFP4 else 0):
            raw[blk + o] &= 0x3f
    y = quants.dequantize(np.frombuffer(bytes(raw), np.uint8).reshape(2, -1), qt).astype("<f4")
    scale_list = ", ".join(map(str, scales))
    print(f"      {{LARCH_{name}, {nbytes}, {{{scale_list}}}, 0x{fnv(y.tobytes()):016x}u, "
          f"0x{fnv(y.astype('<f2').tobytes()):016x}u, {y.flat[0]:.9g}f, {y.flat[777]:.9g}f}},")
