#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"
#include "metal/abi/QuantFormat.h"

namespace splash::model {
namespace {

// The native rows of a table that streams from a checkpoint, its shards'
// rows in order (the copies of its image), as one file of rows back to back
// that SparseRowStore reads: each read builds whole rows from their codes,
// scales and biases (readRepackRows).
class TableReader final : public RowReader {
public:
  TableReader(gguf::Repack table, std::shared_ptr<const void> planned, const SafetensorsCheckpoint &checkpoint)
      : table_(std::move(table)), planned_(std::move(planned)), checkpoint_(checkpoint),
        rowBytes_(ggufRowBytes(kQuantFormats[table_.format], table_.columns)) {}
  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (offset % rowBytes_ || bytes.size() % rowBytes_)
      throw SparseRowError("a streamed table is read by whole rows");
    readRepackRows(table_, offset / rowBytes_, bytes.size() / rowBytes_, bytes.data());
  }
  void checkUnchanged() override { checkpoint_.checkUnchanged(); }
  [[nodiscard]] uint64_t size() const noexcept override { return table_.rows * rowBytes_; }

private:
  gguf::Repack table_;
  std::shared_ptr<const void> planned_; // holds the checkpoint
  const SafetensorsCheckpoint &checkpoint_;
  uint64_t rowBytes_;
};

} // namespace

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, SparseRowStoreOptions table,
                                                 ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), table_(table),
      experts_(experts) {
  // How the checkpoint stores its norms, told by its attention q and k norms
  // (safetensors::normStorage), decides how the images hold them.
  const safetensors::NormStorage norms = safetensors::normStorage(planned_->checkpoint, geometry);
  planned_->images = safetensors::planImages(planned_->checkpoint, geometry, norms,
                                              {.streamExperts = experts.memoryBytes != 0, .streamTable = true});
  // The layers', the head's and the embedding's; a table's follows them, where
  // the family has one (safetensors::walkImages).
  planned_->standardImages = geometry.layers + 2;
  planned_->layers = geometry.layers;
  planned_->firstMoeLayer = geometry.leadingDenseLayers;
  float32Norms_ = safetensors::float32Norms(norms, geometry);
  std::shared_ptr<Planned> planned = planned_;
  adopt(backend, images, {planned, &planned->images},
        [&backend, planned](const metal::MetalBuffer &buffer, size_t index) {
          writeGgufImage(backend, buffer, planned->images[index]);
        },
        [planned] { planned->checkpoint.checkUnchanged(); }, geometry.layers,
        // The fork's geometry carries its first MoE layer; no leading dense
        // layers here means layer 0.
        geometry.leadingDenseLayers, experts);
}

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const SafetensorsTargetPlan &plan, SparseRowStoreOptions table,
                                                 ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), table_(table),
      experts_(experts) {
  const SafetensorsTargetPlan::Images plannedImages =
      plan.images(planned_->checkpoint, {.streamExperts = experts.memoryBytes != 0, .streamTable = true});
  planned_->images = std::move(plannedImages.images);
  planned_->standardImages = plan.layerImages + 2;
  planned_->layers = plan.layerImages;
  planned_->firstMoeLayer = plan.firstMoeLayer;
  float32Norms_ = plannedImages.float32Norms;
  std::shared_ptr<Planned> planned = planned_;
  adopt(backend, images, {planned, &planned->images},
        [&backend, planned](const metal::MetalBuffer &buffer, size_t index) {
          writeGgufImage(backend, buffer, planned->images[index]);
        },
        [planned] { planned->checkpoint.checkUnchanged(); }, plan.layerImages, plan.firstMoeLayer, experts);
}

std::optional<SafetensorsTargetLoader::StreamedTable> SafetensorsTargetLoader::streamTable() const {
  const gguf::Image *plan = tablePlan();
  if (!plan || !plan->streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan->streamedRows;
  // The table's rows are its shards' (shardedTable), read back to back as
  // the image would hold them.
  gguf::Repack table;
  table.format = rows.format;
  table.rows = rows.rows;
  table.columns = rows.columns;
  for (const gguf::Copy &copy : plan->copies) table.sources.push_back(copy.source);
  planned_->checkpoint.checkUnchanged();
  return StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, 0},
                                       std::make_unique<TableReader>(std::move(table), planned_, planned_->checkpoint),
                                       table_),
      rows.format, rows.columns,
      // As WeightImages::load records the image (WeightFile).
      {"target/" + plan->name, plan->magic, plan->layer, plan->type, plan->bytes, images_.contentIdentity()}};
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  const std::span layers = std::span(planned_->images).first(planned_->layers);
  return {gguf::streamedExperts(layers.subspan(planned_->firstMoeLayer)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

const gguf::Image *SafetensorsTargetLoader::tablePlan() const noexcept {
  // The layers', the head's and the embedding's, then a table's where the
  // family has one (walkImages).
  const size_t index = planned_->standardImages;
  return index < planned_->images.size() ? &planned_->images[index] : nullptr;
}

WeightFile SafetensorsTargetLoader::table() {
  const gguf::Image *plan = tablePlan();
  if (!plan) throw WeightStoreError("the target has no table");
  return open(planned_->standardImages);
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       },
                       plan.residentBytes});
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= planned_->standardImages - 2) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->standardImages - 2); }

WeightFile SafetensorsTargetLoader::embedding() { return open(planned_->standardImages - 1); }

} // namespace splash::model
