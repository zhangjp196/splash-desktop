#pragma once

// The normalized Walsh-Hadamard transform of one block of a rotation
// (metal/abi/Gguf.h), shared by the projection input's and the MoE tile's
// kernels: the block's values are transformed in fp32 by butterflies and
// rounded once to bf16. Includers set `#pragma clang fp reassociate(off)`
// first.
#include "metal/abi/Gguf.h"
#include <metal_stdlib>

using namespace metal;

// The unnormalized transform of a block whose values v[i] are value tid +
// threads i: stage `stride` maps the pair (x, y) of values `stride` apart to
// (x + y, x - y). Strides below 32 pair values of one simdgroup, the others
// pass through threadgroup memory `values` (GGUF_ROTATION_BLOCK floats). The
// transformed block is left in `values`.
inline void rotation_butterflies(float4 v, threadgroup float *values, uint tid, uint threads, uint block) {
  for (ushort stride = 1; stride < 32; stride *= 2) {
    const float4 peer = simd_shuffle_xor(v, stride);
    v = (tid & stride) ? peer - v : v + peer;
  }
  for (uint i = 0; i < 4; ++i) values[tid + i * threads] = v[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride < block; stride *= 2) {
    for (uint pair = tid; pair < block / 2; pair += threads) {
      const uint a = (pair / stride) * (2 * stride) + pair % stride;
      const float x = values[a], y = values[a + stride];
      values[a] = x + y;
      values[a + stride] = x - y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The rotated block that starts at input[at]: H (D input), D the int8 signs of
// the row's inputs from `column` on.
inline void rotation_block(device const bfloat *input, device const char *signs, device bfloat *output, ulong at,
                           uint column, uint block, float scale, uint tid, uint threads, threadgroup float *values) {
  float4 v;
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * threads;
    v[i] = float(input[at + j]) * float(signs[column + j]);
  }
  rotation_butterflies(v, values, tid, threads, block);
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * threads;
    output[at + j] = bfloat(values[j] * scale);
  }
}
