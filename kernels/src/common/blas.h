// The one GEMM of the CUDA backend and of its hipified HIP build (architecture 6.3: cuBLAS here,
// hipBLAS after hipify).
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

/// c = alpha a b_t^T + beta c, all row-major: a f16 [m, k], b_t f16 [n, k] (the weight), c [m, n] in
/// f16, or f32 when c_f32; ld* are row strides in elements. Accumulates in f32; c is not read when
/// beta == 0. Runs on the handle's stream.
inline cublasStatus_t vv_blas_gemm_f16(cublasHandle_t h, int m, int n, int k, float alpha, const __half* a, int lda,
                                       const __half* b_t, int ldb, float beta, void* c, bool c_f32, int ldc) {
  // Row-major c [m, n] is column-major c^T [n, m] = b_t a^T, so b_t is the transposed left operand.
  return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b_t, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta,
                      c, c_f32 ? CUDA_R_32F : CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
