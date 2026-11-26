#include "ops/Inkling2VisionPlan.hpp"

#include "metal/abi/Inkling2Vision.h"
#include "metal/abi/Vision.h"

#include <array>
#include <stdexcept>
#include <string>
#include <limits>

namespace splash::ops {
namespace {

void requireBuffer(const metal::MetalBuffer &buffer, uint64_t bytes,
                   const char *label) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("undersized Inkling2 vision buffer: ") + label);
}

constexpr uint32_t kHidden = 1024;
constexpr uint32_t kIntermediate = 4096;
constexpr uint64_t kBf16 = 2;

} // namespace

void Inkling2VisionPlan::encode(metal::CommandGraph &graph,
                                const Inkling2VisionPlanBuffers &b,
                                uint32_t tileHeight, uint32_t tileWidth) {
  const uint64_t tiles64 = uint64_t(tileHeight) * tileWidth;
  // The merger packing kernel uses a 32-bit flattened feature index.
  if (!tiles64 || tiles64 > (uint64_t(UINT32_MAX) / 4096 / 32) * 32 ||
      tileWidth > UINT32_MAX / 40 || tileHeight > UINT32_MAX / 40)
    throw std::invalid_argument("invalid Inkling2 vision tile grid");
  const uint32_t tiles = uint32_t(tiles64);
  const uint32_t tokens = 1 + 4 * tiles;
  const uint32_t paddedTokens = (tokens + 127) / 128 * 128;
  const uint32_t mergerRows = (tiles + 31) / 32 * 32;
  requireBuffer(b.preNormWeight, kHidden * kBf16, "pre-LN weight");
  requireBuffer(b.preNormBias, kHidden * kBf16, "pre-LN bias");
  requireBuffer(b.preNormalized, uint64_t(tokens) * kHidden * sizeof(float), "residual stream");
  requireBuffer(b.zeroBias, kIntermediate * kBf16, "zero bias");
  requireBuffer(b.mergerInput, uint64_t(mergerRows) * 4 * kHidden * kBf16, "merger input tile");
  requireBuffer(b.mergerUpWeight, kIntermediate * kIntermediate * kBf16, "merger up weight");
  requireBuffer(b.mergerUp, uint64_t(mergerRows) * kIntermediate * kBf16, "merger up output");
  requireBuffer(b.mergerDownWeight, 2304ull * kIntermediate * kBf16, "merger down weight");
  requireBuffer(b.mergerDown, uint64_t(mergerRows) * 2304 * kBf16, "merger output");
  if (!b.blocks.front().residual.sameView(b.preNormalized))
    throw std::invalid_argument("Inkling2 block residuals must alias the pre-normalized stream");
  for (const auto &block : b.blocks)
    if (!block.residual.sameView(b.preNormalized))
      throw std::invalid_argument("Inkling2 blocks must share one FP32 residual stream");

  const metal::MetalBuffer *live[] = {&b.stem.pixels, &b.stem.patches, &b.stem.hidden,
      &b.preNormalized, &b.mergerInput, &b.mergerUp, &b.mergerDown};
  for (size_t i = 0; i < std::size(live); ++i)
    for (size_t j = i + 1; j < std::size(live); ++j)
      if (live[i]->overlaps(*live[j]))
        throw std::invalid_argument("overlapping Inkling2 vision plan scratch");

  auto stem = b.stem;
  // Inkling2's patch projection has no bias. The common stem kernel accepts a
  // bias pointer, so bind an immutable zero-filled BF16 row from caller scratch.
  stem.patchBias = b.zeroBias;
  Inkling2VisionStem::encode(graph, stem, kHidden, tileHeight, tileWidth);

  const Inkling2VisionBlockParams params{tokens, paddedTokens, kHidden,
      kIntermediate, 16, 64, 2 * tileHeight, 2 * tileWidth, 1, 1e-5f};
  graph.add("inkling2_vision_pre_layer_norm",
            {b.stem.hidden, b.preNormWeight, b.preNormBias, b.preNormalized},
            params, {tokens, 1, 1});

  for (const auto &block : b.blocks)
    Inkling2VisionBlock::encode(graph, block, tokens, paddedTokens, tileHeight, tileWidth, true);

  graph.add("inkling2_vision_pack_merger", {b.preNormalized, b.mergerInput},
            Inkling2VisionMergerParams{tileHeight, tileWidth, mergerRows},
            {mergerRows * 16, 1, 1}, {256, 1, 1});
  graph.add("vision_gemm_m32n256_gelu_erf",
            {b.mergerInput, b.mergerUpWeight, b.zeroBias, b.mergerUp, b.mergerUp},
            VisionGemmParams{kIntermediate, kIntermediate}, {mergerRows / 32, 16, 1});
  graph.add("vision_gemm_m32n256",
            {b.mergerUp, b.mergerDownWeight, b.zeroBias, b.mergerDown, b.mergerDown},
            VisionGemmParams{2304, kIntermediate}, {mergerRows / 32, 9, 1});
}

} // namespace splash::ops
