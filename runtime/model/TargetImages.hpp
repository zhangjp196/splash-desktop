#pragma once

// What both sources' loaders of a target (GgufTargetLoader,
// SafetensorsTargetLoader) hold alike: the images they planned, each layer's
// then the head's and the embedding's, which a family reads as block-quantized
// weights, and the routed experts' blocks when those stream. The loaders
// differ in their source, their planner and how a streamed table's rows are
// read. A family that loads its images itself uses this interface directly.

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace splash::model {

class TargetImages {
public:
  // A table whose rows stream from the source: the store that reads them,
  // their format (GGUF_FMT_*), their physical columns and the record of its
  // image, which the weights' fingerprint takes as when the table loads.
  struct StreamedTable final {
    std::shared_ptr<SparseRowStore> store;
    uint32_t format = 0;
    uint32_t columns = 0;
    WeightFileRecord record;
  };

  TargetImages(const TargetImages &) = delete;
  TargetImages &operator=(const TargetImages &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index) {
    if (index >= layers_) throw WeightStoreError("target layer is out of range");
    return open(index);
  }
  [[nodiscard]] WeightFile head() { return open(layers_); }
  [[nodiscard]] WeightFile embedding() { return open(layers_ + 1); }
  [[nodiscard]] WeightFile table() {
    const gguf::Image *plan = tablePlan();
    if (!plan) throw WeightStoreError("the target has no table");
    if (plan->streamedRows) throw WeightStoreError("the target's table streams");
    return open(layers_ + 2);
  }

  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const {
    const std::span layers = std::span(*images_).first(layers_);
    return {gguf::streamedExperts(layers.subspan(firstMoeLayer_)), images_, {checkUnchanged_}};
  }

protected:
  // The images `images` points to, planned once by the loader; `write` writes
  // image `index` into a mapped buffer, `checkUnchanged` finds its source
  // unwritten since it was planned. The images begin with `layerImages` layer
  // images, the first MoE layer's `firstMoeLayer`.
  TargetImages() = default;
  ~TargetImages() = default;

  // The loader calls this once it has planned its images, wiring the base's
  // reads back to its own state. Existing fork loaders keep their planning.
  void adopt(metal::MetalBackend &backend, WeightImages &images, std::shared_ptr<std::vector<gguf::Image>> planned,
             std::function<void(const metal::MetalBuffer &, size_t)> write, std::function<void()> checkUnchanged,
             uint32_t layerImages, uint32_t firstMoeLayer, ExpertStoreOptions experts) {
    backend_ = &backend;
    weightImages_ = &images;
    images_ = std::move(planned);
    write_ = std::move(write);
    checkUnchanged_ = std::move(checkUnchanged);
    layers_ = layerImages;
    firstMoeLayer_ = firstMoeLayer;
    experts_ = experts;
  }

  [[nodiscard]] gguf::PlanOptions planOptions() const noexcept { return {.streamExperts = streamsExperts()}; }
  [[nodiscard]] const gguf::Image *tablePlan() const noexcept {
    const size_t index = layers_ + 2;
    return index < images_->size() ? &(*images_)[index] : nullptr;
  }
  [[nodiscard]] const std::shared_ptr<std::vector<gguf::Image>> &images() const noexcept { return images_; }
  [[nodiscard]] WeightImages &weightImages() const noexcept { return *weightImages_; }
  [[nodiscard]] metal::MetalBackend &backend() const noexcept { return *backend_; }

private:
  [[nodiscard]] WeightFile open(size_t index) {
    const gguf::Image &plan = (*images_)[index];
    return weightImages_->load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                               [this, index](std::span<uint8_t>, const metal::MetalBuffer &buffer) {
                                 write_(buffer, index);
                                 checkUnchanged_();
                               },
                               plan.residentBytes});
  }

  metal::MetalBackend *backend_ = nullptr;
  WeightImages *weightImages_ = nullptr;
  std::shared_ptr<std::vector<gguf::Image>> images_;
  std::function<void(const metal::MetalBuffer &, size_t)> write_;
  std::function<void()> checkUnchanged_;
  uint32_t layers_ = 0, firstMoeLayer_ = 0;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
