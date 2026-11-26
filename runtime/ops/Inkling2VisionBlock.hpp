#pragma once
#include "metal/CommandGraph.hpp"
#include <cstdint>

namespace splash::ops {

// Scratch is caller owned so a 23-block graph can reuse one arena. All
// activations are row-major BF16 except the FP32 residual stream.
// GEMM-facing row buffers must hold roundUp(tokens, 64) rows; Q/K/V head
// buffers must hold padded_tokens (a multiple of 128) rows per head.
// Scratch views within a block must be disjoint; blocks may reuse the same views.
struct Inkling2VisionBlockBuffers final {
  metal::MetalBuffer residual, normalized;
  metal::MetalBuffer query, key, value, queries, keys, values;
  metal::MetalBuffer attention, context, projected;
  metal::MetalBuffer normalized2, intermediate, down;
  metal::MetalBuffer norm1Weight, norm1Bias, queryWeight, queryBias;
  metal::MetalBuffer keyWeight, keyBias, valueWeight, valueBias;
  metal::MetalBuffer outputWeight, outputBias, scale1;
  metal::MetalBuffer norm2Weight, norm2Bias, upWeight, upBias;
  metal::MetalBuffer downWeight, downBias, scale2;
};

class Inkling2VisionBlock final {
public:
  // Encodes one pre-LN, noncausal Inkling2 block. For an image, CLS has
  // position (0, 0) and patches have one-based spatial coordinates.
  static void encode(metal::CommandGraph &graph,
                     const Inkling2VisionBlockBuffers &buffers,
                     uint32_t tokens, uint32_t paddedTokens,
                     uint32_t tileHeight, uint32_t tileWidth,
                     bool skipClsRope = true);
};
} // namespace splash::ops
