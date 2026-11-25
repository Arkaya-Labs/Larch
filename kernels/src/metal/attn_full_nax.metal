// larch_attn_full on M5 (Apple GPU family 10): MLX's steel_attention_nax (Metal 4 tensor matmul), vendored
// verbatim under mlx/ (MIT), with the same fixed function constants as attn_full_sg.metal.
// ponytail: head_dim 64 only; 72 (the 35B's encoder) takes the simdgroup kernel. MLX pads 72 to 96 for NAX:
// do that if the 35B's encode profile shows attn_full over half its time.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

instantiate_kernel("vv_attn_full_nax64", attention_nax, half, 64, 32, 64, 64, 4, 1, float, float)
#pragma clang diagnostic pop
