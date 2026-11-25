// larch_attn_full on M1 and later: MLX's steel_attention (simdgroup 8 x 8 matrices), vendored verbatim under
// mlx/ (MIT). The backend fixes its function constants: no mask, causal, sinks or window; align_Q and align_K
// false, so the last query and key blocks take the bounds-checked loads whatever the length.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

instantiate_kernel("vv_attn_full_sg64", attention, half, 32, 32, 64, 64, 4, 1, float, float)
instantiate_kernel("vv_attn_full_sg72", attention, half, 32, 32, 72, 72, 4, 1, float, float)
#pragma clang diagnostic pop
