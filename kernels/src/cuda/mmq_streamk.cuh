// Bounded K256 work partitioning for the explicit activation-I8 policy.
#pragma once
namespace cuda::mmq_streamk {
// A CTA writes at most one first partial tail and any subsequent tile heads. No CTA waits for another.
// A K128 tail (32-weight formats) is the last K256 step of its tile.
// Two resident CTAs per SM are its precondition (the host verifies them), hence the register bound.
template<int T, int J>
__global__ void __launch_bounds__(256, 2) gemm(const uint8_t* w, int64_t w_ld, int64_t n, int64_t k,
                     const uint32_t* packed, int64_t padded_rows, bool aligned4,
                     int64_t n_tiles, int64_t tiles, float* head, float* tail) {
  static_assert(J == 8 || J == 16);
  extern __shared__ __align__(16) uint32_t memory[];
  const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
  // The host bounds work by INT_MAX: 32-bit bookkeeping keeps the two-CTA register budget.
  const int k_blocks = int((k + 255) / 256), work = int(tiles) * k_blocks, nt = int(n_tiles);
  int begin = int(int64_t(blockIdx.x) * work / gridDim.x);
  const int end = int((int64_t(blockIdx.x) + 1) * work / gridDim.x);
  while (begin < end) {
    const int tile = begin / k_blocks;
    const int limit = min(end, (tile + 1) * k_blocks);
    const int64_t n0 = int64_t(tile % nt) * mmq_q4::weight_rows, m0 = int64_t(tile / nt) * J;
    float acc[J / 8][4] = {};
    mmq_q4::accumulate_segment<T, J, false>(w, w_ld, n, packed, padded_rows, aligned4, n0, m0,
                                   int64_t(begin % k_blocks) * 256, min(k, int64_t(limit - tile * k_blocks) * 256),
                                   memory, acc);
    float* target = begin % k_blocks ? tail + int64_t(blockIdx.x) * J * mmq_q4::weight_rows
                                   : head + int64_t(tile) * J * mmq_q4::weight_rows;
#pragma unroll
    for (int j = 0; j < J; j += 8) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int row = warp * 16 + lane / 4 + l / 2 * 8;
        const int col = j + lane % 4 * 2 + l % 2;
        target[col * mmq_q4::weight_rows + row] = acc[j / 8][l];
      }
    }
    begin = limit;
  }
}

// The contributor range is derived from the finite work partition, independent of GPU scheduling.
template<int J>
__global__ void merge(const float* head, const float* tail, int64_t tiles, int64_t n_tiles,
                      int64_t k_blocks, int grid, int64_t m, int64_t n,
                      const float* bias, int64_t bias_ld, void* out, int64_t out_ld, bool f32) {
  const int64_t tile = blockIdx.x, work = tiles * k_blocks;
  const int64_t first = ((tile * k_blocks + 1) * grid - 1) / work;
  const int64_t last = (((tile + 1) * k_blocks) * grid - 1) / work;
  const int64_t m0 = tile / n_tiles * J, n0 = tile % n_tiles * 128;
  for (int i = int(threadIdx.x); i < J * 128; i += int(blockDim.x)) {
    const int64_t r = m0 + i / 128, c = n0 + i % 128;
    if (r >= m || c >= n) continue;
    float value = head[tile * J * 128 + i];
    for (int64_t contributor = first + 1; contributor <= last; ++contributor)
      value += tail[contributor * J * 128 + i];
    if (bias) value += bias[c * bias_ld];
    if (f32) static_cast<float*>(out)[r * out_ld + c] = value;
    else static_cast<__half*>(out)[r * out_ld + c] = __float2half_rn(value);
  }
}
}  // namespace cuda::mmq_streamk
