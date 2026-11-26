#include "model/ModelSizing.hpp"

#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"

#include <span>
#include <vector>

namespace splash::model {
namespace {

uint64_t loadedBytes(std::span<const gguf::Image> images) {
  uint64_t bytes = 0;
  for (const gguf::Image &image : images)
    bytes += gguf::loadedBytes(image);
  return bytes;
}

// What a target of `layout` whose images `plan` plans for what streams
// takes, while their sources are open: what loads while every part `streams`
// lets stream does, and what each such part adds where it loads instead,
// the routed experts' slot table with them.
template <class Layout, class Plan>
StreamableWeights targetWeights(const Layout &layout, const Plan &plan, gguf::PlanOptions streams,
                                uint32_t layerImages, uint32_t firstMoeLayer) {
  streams.streamExperts = streams.streamExperts && layout.ffnKind == QwenFfnKind::SparseMoe;
  StreamableWeights result;
  std::vector<gguf::Image> streamed;
  try {
    streamed = plan(streams);
  } catch (const GgufError &error) {
    if (!streams.streamTable)
      throw;
    // A table that cannot stream loads with the weights; whatever else
    // failed, the plan without it fails again.
    result.tableRefusal = error.what();
    streams.streamTable = false;
    streamed = plan(streams);
  }
  result.loadedBytes = loadedBytes(streamed);
  for (const gguf::Image &image : streamed)
    if (image.streamedRows) {
      result.tableBytes = loadedBytes(plan({.streamExperts = streams.streamExperts})) - result.loadedBytes;
      const SparseRowStoreOptions rows = streamedTableRows(layout);
      result.tableRowBytes = image.streamedRows->rowBytes;
      result.tableFloorBytes = SparseRowStore::workingBytes(image.streamedRows->rowBytes, rows.prefetchRows,
                                                            rows.stageRows) +
                               SparseRowStore::cacheBytesFor(1, image.streamedRows->rowBytes);
    }
  if (streams.streamExperts) {
    result.expertBytes = loadedBytes(plan({.streamTable = result.tableBytes != 0})) - result.loadedBytes +
                         residentExpertSlotsBytes(layout.experts);
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(
        std::span(streamed).subspan(firstMoeLayer, layerImages - firstMoeLayer));
    result.expertFloorBytes = ExpertStore::minimumBytes(blocks, layout.moeBlockShape());
  }
  return result;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    gguf::PlanOptions streams) {
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          const GgufTargetPlan plan = ggufTargetPlan(layout);
          return targetWeights(
              layout,
              [&](gguf::PlanOptions options) { return plan.images(target.file(), options); },
              streams, plan.layerImages, plan.firstMoeLayer);
        }
        const SafetensorsCheckpoint checkpoint(directory);
        // A family that plans its own images (Inkling 2) declares a specific
        // safetensorsTargetPlan overload; every other family uses the template
        // that plans via safetensors::planImages and its norm storage.
        const SafetensorsTargetPlan plan = safetensorsTargetPlan(layout);
        return targetWeights(
            layout,
            [&](gguf::PlanOptions options) { return plan.images(checkpoint, options).images; }, streams,
            plan.layerImages, plan.firstMoeLayer);
      },
      descriptor.target);
  if (descriptor.hasDraft())
    result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision()) {
    if (std::holds_alternative<Inkling2LoopLayout>(descriptor.target)) {
      const Inkling2VisionLoader loader(
          descriptor.visionSource == VisionSource::Gguf ? root / "vision" / "mmproj.gguf" : root / "target",
          descriptor.visionSource == VisionSource::Gguf ? Inkling2VisionSourceFormat::Gguf
                                                        : Inkling2VisionSourceFormat::Safetensors);
      result.loadedBytes += loader.image().bytes;
    } else {
      result.loadedBytes += visionImageBytes(descriptor.vision);
    }
  }
  return result;
}

} // namespace splash::model
