#pragma once

#include "ops/Inkling2VisionBlock.hpp"
#include "ops/Inkling2VisionStem.hpp"

#include <array>

namespace splash::ops {

// Image-wide geometry uses a grid of 40x40 tiles. Scratch and weight views are
// supplied by the caller and remain alive through graph submission.
struct Inkling2VisionPlanBuffers final {
  Inkling2VisionStemBuffers stem;
  metal::MetalBuffer preNormWeight, preNormBias, preNormalized;
  std::array<Inkling2VisionBlockBuffers, 23> blocks;
  metal::MetalBuffer zeroBias;
  metal::MetalBuffer mergerInput, mergerUpWeight, mergerUp, mergerDownWeight, mergerDown;
};

// Encodes one image with one CLS and four patches per tile. Pixels are
// tile-major 40x40 HWC RGB; output rows are tile-major, each 2304 BF16 values.
// Scratch must provide 64-row rounded GEMM buffers, 128-token rounded head
// buffers, and 32-tile rounded merger buffers. All block residuals must alias
// preNormalized. zeroBias contains at least 4096 zero BF16 values.
class Inkling2VisionPlan final {
public:
  static void encode(metal::CommandGraph &graph,
                     const Inkling2VisionPlanBuffers &buffers,
                     uint32_t tileHeight, uint32_t tileWidth);
};

} // namespace splash::ops
