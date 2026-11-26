#pragma once

#include "metal/CommandGraph.hpp"

namespace splash::ops {

// Dedicated Inkling2v stem dispatch. Pixels are normalized float32, tile-major
// 40x40 HWC RGB, with tiles in raster order. The output is a raster patch grid
// with one CLS row, and tensors are BF16 GGUF weights.
struct Inkling2VisionStemBuffers final {
  metal::MetalBuffer pixels;
  metal::MetalBuffer patches;
  metal::MetalBuffer patchEmbedding;
  metal::MetalBuffer patchBias;
  metal::MetalBuffer classEmbedding;
  metal::MetalBuffer classPosition;
  metal::MetalBuffer positions;
  metal::MetalBuffer hidden;
};

class Inkling2VisionStem final {
public:
  static void encode(metal::CommandGraph &graph,
                     const Inkling2VisionStemBuffers &buffers,
                     uint32_t hiddenSize, uint32_t tileHeight, uint32_t tileWidth);
};

} // namespace splash::ops
