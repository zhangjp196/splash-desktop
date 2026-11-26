#pragma once

#include "model/Inkling2Vision.hpp"
#include "model/WeightImages.hpp"
#include "ops/Inkling2VisionPlan.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace splash::model {

struct Inkling2VisionBufferTensor final {
  metal::MetalBuffer weight, bias;
};

struct Inkling2VisionBufferBlock final {
  Inkling2VisionBufferTensor norm1, query, key, value, attentionOutput;
  Inkling2VisionBufferTensor norm2, up, down;
  metal::MetalBuffer scale1, scale2;
};

struct Inkling2VisionBufferWeights final {
  Inkling2VisionLayout layout;
  uint64_t actualAllocatedBytes = 0;
  std::vector<WeightFileRecord> files;
  metal::MetalBuffer patchEmbedding, classEmbedding, classPositionEmbedding, positionEmbedding;
  Inkling2VisionBufferTensor preNorm;
  std::vector<Inkling2VisionBufferBlock> blocks;
  Inkling2VisionBufferTensor mergerUp, mergerDown;
};

// Owns the scratch for one image-wide graph. The input is tile-major FP32;
// output() holds one BF16 row per tile. Keep this object alive through submit.
class Inkling2VisionGraphResources final {
public:
  Inkling2VisionGraphResources(metal::MetalBackend &backend,
                               const Inkling2VisionBufferWeights &weights,
                               metal::MetalBuffer pixels,
                               uint32_t tileHeight, uint32_t tileWidth);
  [[nodiscard]] static uint64_t requiredScratchBytes(uint32_t tileHeight, uint32_t tileWidth);
  void encode(metal::CommandGraph &graph) const;
  [[nodiscard]] const metal::MetalBuffer &output() const noexcept { return buffers_.mergerDown; }
  [[nodiscard]] uint64_t scratchBytes() const noexcept { return arena_.sizeBytes(); }

private:
  metal::MetalBuffer arena_;
  ops::Inkling2VisionPlanBuffers buffers_;
  uint32_t tileHeight_, tileWidth_;
};

enum class Inkling2VisionSourceFormat : uint8_t { Gguf, Safetensors };

// Serializes the Inkling2 vision inventory into inkling2/vision.bin.
// The loader owns the open source and parser while its ImagePlan can be written.
class Inkling2VisionLoader final {
public:
  explicit Inkling2VisionLoader(const std::filesystem::path &mmproj, Inkling2VisionLayout layout = {});
  Inkling2VisionLoader(const std::filesystem::path &source, Inkling2VisionSourceFormat format,
                       Inkling2VisionLayout layout = {});
  ~Inkling2VisionLoader();
  Inkling2VisionLoader(const Inkling2VisionLoader &) = delete;
  Inkling2VisionLoader &operator=(const Inkling2VisionLoader &) = delete;

  [[nodiscard]] ImagePlan image() const;
  // Consume the image sections in inventory order as MetalBuffer views keyed by GGUF name.
  [[nodiscard]] std::vector<std::pair<std::string, metal::MetalBuffer>> bind(WeightFile &file) const;
  // Bind the same sections to the vision tower's typed weight inventory.
  [[nodiscard]] Inkling2VisionBufferWeights bindWeights(WeightFile &file) const;
  // Register the image with the model's release/restore weight cache and bind it.
  [[nodiscard]] Inkling2VisionBufferWeights load(WeightImages &images) const;
  [[nodiscard]] static constexpr size_t tensorCount() noexcept { return 422; }

private:
  struct Planned;
  std::shared_ptr<const Planned> planned_;
};

} // namespace splash::model
