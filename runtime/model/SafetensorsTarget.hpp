#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/TargetImages.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace splash::model {

// How a family's checkpoints plan (SafetensorsTargetLoader): its images from
// a checkpoint with what streams, and whether they hold its norms as F32,
// its layer images first, `layerImages` of them, the first MoE layer's
// `firstMoeLayer`.
struct SafetensorsTargetPlan final {
  struct Images final {
    std::vector<gguf::Image> images;
    bool float32Norms = true;
  };
  uint32_t layerImages = 0;
  uint32_t firstMoeLayer = 0;
  std::function<Images(const SafetensorsCheckpoint &, gguf::PlanOptions)> images;
};

// A family whose checkpoints safetensors::planImages plans (every fork
// family), an image a layer. A family that plans its own declares a more
// specific overload beside its layout (Inkling 2's safetensorsTargetPlan).
template <class Layout> [[nodiscard]] SafetensorsTargetPlan safetensorsTargetPlan(const Layout &layout) {
  return {layout.layers,
          layout.leadingDenseLayers,
          [layout](const SafetensorsCheckpoint &checkpoint, gguf::PlanOptions options) {
            const safetensors::NormStorage norms = safetensors::normStorage(checkpoint, layout);
            return SafetensorsTargetPlan::Images{safetensors::planImages(checkpoint, layout, norms, options),
                                                 safetensors::float32Norms(norms, layout)};
          }};
}

class SafetensorsTargetLoader final : public TargetImages {
public:
  // A table whose rows stream from the checkpoint (streamTable()): the store
  // that reads them, their format (GGUF_FMT_*), and the record of its image,
  // which the weights' fingerprint takes as when the table loads, as the rows
  // gathered are the same.
  struct StreamedTable final {
    std::shared_ptr<SparseRowStore> store;
    uint32_t format = 0;
    // Values a stored row holds (the PLE table's padded physical width).
    uint32_t columns = 0;
    WeightFileRecord record;
  };

  // Plans every image of the checkpoint in directory once. With a table
  // store (table.cacheBytes), the rows of a table the family adds stream
  // through it (streamTable()) instead of loading (table()). With expert
  // memory (experts.memoryBytes), a MoE target's routed experts stream
  // (streamedExperts()) instead of loading with their layers.
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry, SparseRowStoreOptions table = {},
                          ExpertStoreOptions experts = {});
  // Plans the images as a family declares them (Inkling 2's
  // safetensorsTargetPlan).
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const SafetensorsTargetPlan &plan, SparseRowStoreOptions table = {},
                          ExpertStoreOptions experts = {});
  SafetensorsTargetLoader(const SafetensorsTargetLoader &) = delete;
  SafetensorsTargetLoader &operator=(const SafetensorsTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The image of the table the family's checkpoint holds after the embedding
  // (Qwen3.8-Flash-Next's PLE n-gram table).
  [[nodiscard]] WeightFile table();
  // The streamed table, its store reading the table's native rows from the
  // checkpoint's open files (readRepackRows over its shards); none when the
  // table loads.
  [[nodiscard]] std::optional<StreamedTable> streamTable() const;
  // The checkpoint, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const SafetensorsCheckpoint &checkpoint() const noexcept { return planned_->checkpoint; }
  // Whether its images hold F32 norms (safetensors::float32Norms).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }
  // Whether the routed experts stream, then the store's options and every
  // layer's experts, which read the checkpoint's open files.
  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const;

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    // The layers', the head's and the embedding's, and a table's where the
    // family has one (safetensors::walkImages).
    std::vector<gguf::Image> images;
    // How many images are the layers', the head's and the embedding's
    // (walkImages); the table's follows them, where there is one.
    size_t standardImages = 0;
    uint32_t layers = 0;
    // The first MoE layer (QwenTargetDimensions::leadingDenseLayers).
    uint32_t firstMoeLayer = 0;
  };
  [[nodiscard]] WeightFile open(size_t index);
  // The plan of the table's image, or null for a family without one.
  [[nodiscard]] const gguf::Image *tablePlan() const noexcept;

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  SparseRowStoreOptions table_;
  bool float32Norms_ = false;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
