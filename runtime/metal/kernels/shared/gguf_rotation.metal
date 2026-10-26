// Prism ML's and Gyro's input rotation (metal/abi/Gguf.h): the normalized Walsh-Hadamard transform of a block of
// `block` values, H (D x), in fp32 and rounded once to bf16. A threadgroup of block / 4 threads holds a block, each
// thread four values; the butterflies of strides below 32 pair values of one simdgroup, the others pass through
// threadgroup memory (kernels/common/rotation.h).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "metal/kernels/common/gguf_embedding_formats.h"
#include "metal/kernels/common/rotation.h"
#include <metal_stdlib>

using namespace metal;

static_assert(GGUF_ROTATION_BLOCK == 1024, "the threadgroup holds the largest rotation block");
static_assert(GGUF_ROTATION_MIN_BLOCK == 128, "the smallest rotation block");

// The rotated input of a projection whose weights were stored for it: output = H (D input) per block of each row.
// Grid (width / block, rows), block / 4 threads.
kernel void gguf_rotate(device const bfloat *input [[buffer(0)]], device const char *signs [[buffer(1)]],
                        device bfloat *output [[buffer(2)]], constant GgufRotationParams &p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float values[GGUF_ROTATION_BLOCK];
  const uint threads = p.block / 4, column = group.x * p.block;
  rotation_block(input, signs, output, ulong(group.y) * p.width + column, column, p.block, p.scale, tid, threads,
                 values);
}

// The routed tiles of a MoE block's intermediate, in place, whose down pass's weights were stored for rotated
// inputs (kernels/shared/moe.metal): H (D e) of each of the tile's live rows, the shared expert's tiles (which
// follow the routed ones) untouched. Grid (intermediate / block, routed tiles, tile rows), block / 4 threads.
kernel void moe_rotate_intermediate(device bfloat *intermediate [[buffer(0)]],
                                    device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const char *signs [[buffer(2)]],
                                    constant GgufRotationParams &p [[buffer(3)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float values[GGUF_ROTATION_BLOCK];
  if (group.z >= tiles[group.y].rows) return;
  const uint threads = p.block / 4, column = group.x * p.block;
  const ulong at = ulong((group.y * p.tile_rows + group.z) * p.width + column);
  rotation_block(intermediate, signs, intermediate, at, column, p.block, p.scale, tid, threads, values);
}

// The token rows of a PQ2_0 table stored rotated (GgufEmbedPQ20), as D (H r): the gathered values enter the
// transform in fp32, so each output rounds once. Grid (hidden / block, rows), block / 4 threads.
kernel void gguf_embed_rotated_pq20(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]],
                                    device const char *signs [[buffer(2)]], device bfloat *output [[buffer(3)]],
                                    constant GgufEmbedRotationParams &q [[buffer(4)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  using F = GgufEmbedPQ20;
  const GgufEmbedParams p = q.embed;
  const GgufRotationParams r = q.rotation;
  threadgroup float values[GGUF_ROTATION_BLOCK];
  // The runtime validates every token; the guard keeps a direct call inside the table.
  const uint token = tokens[group.y] < p.vocabulary ? tokens[group.y] : 0;
  const uint threads = r.block / 4, column = group.x * r.block;
  device const uchar *row = table + ulong(token) * (p.hidden / F::Weights) * F::Bytes;
  float4 v;
  for (uint i = 0; i < 4; ++i) {
    const uint dim = column + tid + i * threads;
    v[i] = F::decode(row + (dim / F::Weights) * F::Bytes, dim);
  }
  rotation_butterflies(v, values, tid, threads, r.block);
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * threads;
    output[ulong(group.y) * p.hidden + column + j] = bfloat(values[j] * r.scale * float(signs[column + j]));
  }
}
