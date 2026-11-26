#include "ops/Inkling2VisionStem.hpp"

#include "metal/abi/Inkling2Vision.h"

#include <stdexcept>
#include <limits>

namespace splash::ops {

void Inkling2VisionStem::encode(metal::CommandGraph &graph,
                                const Inkling2VisionStemBuffers &b,
                                uint32_t hiddenSize, uint32_t tileHeight, uint32_t tileWidth) {
  if (!hiddenSize || hiddenSize % 32)
    throw std::invalid_argument("Inkling2 vision hidden size must be a positive multiple of 32");
  if (!b.pixels || !b.patches || !b.patchEmbedding || !b.patchBias ||
      !b.classEmbedding || !b.classPosition || !b.positions || !b.hidden)
    throw std::invalid_argument("Inkling2 vision stem has an unbound buffer");
  const uint64_t tiles = uint64_t(tileHeight) * tileWidth;
  if (!tiles || tileHeight > UINT32_MAX / 40 || tileWidth > UINT32_MAX / 40 ||
      tiles > (std::numeric_limits<uint32_t>::max() - 128u) / 4u)
    throw std::invalid_argument("invalid Inkling2 vision tile grid");
  const uint32_t patches = uint32_t(tiles * 4);
  const auto require = [](const metal::MetalBuffer &buffer, uint64_t bytes) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument("undersized Inkling2 vision stem buffer");
  };
  require(b.pixels, tiles * 40 * 40 * 3 * sizeof(float));
  require(b.patches, uint64_t(patches) * 1200 * 2);
  require(b.patchEmbedding, uint64_t(hiddenSize) * 1200 * 2);
  for (const auto *buffer : {&b.patchBias, &b.classEmbedding, &b.classPosition})
    require(*buffer, uint64_t(hiddenSize) * 2);
  require(b.positions, uint64_t(32) * 32 * hiddenSize * 2);
  require(b.hidden, uint64_t(patches + 1) * hiddenSize * sizeof(float));
  const Inkling2VisionStemParams params{hiddenSize, 20, tileHeight, tileWidth};
  graph.add("inkling2_vision_patchify", {b.pixels, b.patches}, params, {patches, 1, 1});
  graph.add("inkling2_vision_prepare_stem",
            {b.classEmbedding, b.classPosition, b.positions, b.hidden}, params,
            {patches + 1, 1, 1});
  graph.add("inkling2_vision_patch_embed",
            {b.patches, b.patchEmbedding, b.patchBias, b.hidden}, params,
            {patches, hiddenSize / 32, 1}, {32, 1, 1});
}

} // namespace splash::ops
