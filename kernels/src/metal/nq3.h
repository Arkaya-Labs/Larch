// The nq3 layout's scales (args.h nq3_scales_at): Q3_K's own 12 scale bytes a block. sb is unit u's block's 12
// bytes, h = u % 2.
// Returns the 8 scales of unit u, sc (0..63; the scale is sc - 32) of groups 8 h + f, f = 0..7, one a byte: bytes
// 0..7 hold the low nibbles of scales 0..7 and the high nibbles of 8..15, bytes 8..11 the top two bits (ggml's
// get_scale).
#pragma once
static inline uint2 nq3_scales(const device uchar* sb, uint h) {
  const device uint* p = (const device uint*)sb;  // 4-byte aligned: pairs are 220 bytes, rows start 4-aligned
  const uint lo0 = p[0], lo1 = p[1], hi = p[2], s = 4 * h;
  return uint2((lo0 >> s & 0x0F0F0F0Fu) | (hi >> s & 0x03030303u) << 4,
               (lo1 >> s & 0x0F0F0F0Fu) | (hi >> (s + 2) & 0x03030303u) << 4);
}
