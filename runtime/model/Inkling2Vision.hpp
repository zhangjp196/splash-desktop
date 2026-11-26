#pragma once

#include "model/GgufFile.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// Inkling2's vision tower is deliberately separate from QwenVisionWeights:
// its patch stem has a CLS token, separate Q/K/V projections, layer scales,
// axial RoPE, and a four-patch merger.
struct Inkling2VisionLayout final {
  uint32_t hidden = 1024;
  uint32_t intermediate = 4096;
  uint32_t layers = 23;
  uint32_t heads = 16;
  uint32_t headDimension = 64;
  uint32_t patch = 20;
  uint32_t positionSide = 32;
  uint32_t output = 2304;
};

struct Inkling2VisionTensor final {
  const GgufTensor *weight = nullptr;
  const GgufTensor *bias = nullptr;
};

struct Inkling2VisionBlockWeights final {
  Inkling2VisionTensor norm1, query, key, value, attentionOutput;
  Inkling2VisionTensor norm2, up, down;
  const GgufTensor *scale1 = nullptr;
  const GgufTensor *scale2 = nullptr;
};

struct Inkling2VisionWeights final {
  Inkling2VisionLayout layout;
  Inkling2VisionTensor patchEmbedding;
  const GgufTensor *classEmbedding = nullptr;
  const GgufTensor *classPositionEmbedding = nullptr;
  const GgufTensor *positionEmbedding = nullptr;
  Inkling2VisionTensor preNorm;
  std::vector<Inkling2VisionBlockWeights> blocks;
  Inkling2VisionTensor mergerUp, mergerDown;
};

// Strictly binds an inkling2v mmproj. Tensor pointers remain owned by `gguf`;
// callers must keep its parsed file alive for the lifetime of the result.
[[nodiscard]] Inkling2VisionWeights bindInkling2Vision(const GgufFile &gguf,
                                                       Inkling2VisionLayout layout = {});

// Converts one normalized RGB tile to row-major patch vectors. Input layout is
// HWC, output is [patchY,patchX,channel,patchY,patchX].
[[nodiscard]] std::array<float, 3 * 20 * 20 * 4>
inkling2VisionPatchify40(std::span<const float> rgbTile);

} // namespace splash::model
