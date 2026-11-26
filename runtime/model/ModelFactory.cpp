#include "ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/ExpertPriorityProfile.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightManifestHash.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {

void requireCompatibleModel(const LoadedModel &model) {
  if (!model.descriptor.valid() ||
      (model.descriptor.hasDraft() && model.descriptor.draft != model.draft.layout) ||
      !std::visit(
          [&](const auto &target) {
            return model.descriptor.target == TargetLayout{target.layout} &&
                   (!model.descriptor.hasDraft() ||
                    target.layout.vocabularySize == model.draft.layout.vocabularySize);
          },
          model.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (std::holds_alternative<Inkling2LoopLayout>(descriptor.target))
    return nullptr;
  if (descriptor.visionSource != VisionSource::Safetensors && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                    const VisionLoader *loader) {
  return loader ? loadQwenVisionWeights(backend, images, *loader) : QwenVisionWeights{};
}

namespace {

std::string loadedTargetWeightHash(const std::filesystem::path &directory, TargetSource source) {
  std::vector<std::filesystem::path> paths;
  if (source == TargetSource::Gguf) {
    paths = findTargetGgufs(directory);
  } else {
    for (const auto &entry : std::filesystem::directory_iterator(directory))
      if (entry.path().extension() == ".safetensors") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error("target directory holds no safetensors shards");
  }
  std::vector<WeightManifestEntry> manifest;
  manifest.reserve(paths.size());
  for (const auto &path : paths)
    manifest.push_back({path.lexically_relative(directory).generic_string(), path});
  return sha256WeightManifest(manifest);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen4ExpLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen4ExpWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const FlintLayout &layout, const QwenTargetFiles &files) {
  return loadFlintWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Inkling2LoopLayout &layout,
                         const QwenTargetFiles &files) {
  return loadInkling2LoopWeights(backend, layout, files);
}

} // namespace

SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const FlintLayout &) { return {}; }

StreamedParts StreamedParts::loaded(const LoadedModel &model, bool kv) noexcept {
  const SparseRowStore *table = model.streamedTable();
  return {.experts = model.expertStore() != nullptr, .tableRowBytes = table ? table->table().rowBytes : 0, .kv = kv};
}

LoadedModel loadModel(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor,
                              const ModelLoadOptions &options) {
  LoadedModel result;
  result.descriptor = descriptor;
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader. A
  // plan without a draft reads no draft directory and loads no draft image.
  const auto vision = planVisionLoader(root, result.descriptor);
  std::unique_ptr<Inkling2VisionLoader> inklingVision;
  if (result.descriptor.hasVision() && std::holds_alternative<Inkling2LoopLayout>(result.descriptor.target)) {
    if (std::get<Inkling2LoopLayout>(result.descriptor.target).hiddenSize != Inkling2VisionLayout{}.output)
      throw std::invalid_argument("Inkling2 vision output width does not match target");
    if (result.descriptor.visionSource == VisionSource::Gguf)
      inklingVision = std::make_unique<Inkling2VisionLoader>(root / "vision" / "mmproj.gguf");
    else if (result.descriptor.visionSource == VisionSource::Safetensors)
      inklingVision = std::make_unique<Inkling2VisionLoader>(root / "target", Inkling2VisionSourceFormat::Safetensors);
    else
      throw std::invalid_argument("Inkling2 vision source is disabled");
  }
  std::unique_ptr<DraftCheckpointLoader> draft;
  if (result.descriptor.hasDraft())
    draft = std::make_unique<DraftCheckpointLoader>(backend, images, root / "draft", result.descriptor.draft);
  // The weights that stream from their files read on the engine's readers.
  ExpertPriorityHints priorityHints;
  const char *profilePath = std::getenv("SPLASH_EXPERT_PRIORITY_PROFILE");
  const char *tracePath = std::getenv("SPLASH_EXPERT_ROUTE_TRACE");
  const bool identityRequested = (profilePath && *profilePath) || (tracePath && *tracePath);
  std::string weightManifestHash;
  bool weightIdentityKnown = false;
  if (identityRequested) {
    try {
      weightManifestHash = loadedTargetWeightHash(root / "target", result.descriptor.targetSource);
      weightIdentityKnown = true;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "expert weight identity: %s; identity unknown\n", error.what());
    }
  }
  if (profilePath && *profilePath && weightIdentityKnown)
    priorityHints = loadExpertPriorityProfile(std::filesystem::path(profilePath), result.descriptor.name,
                                              weightManifestHash);
  const ExpertStoreOptions experts{.memoryBytes = options.streamedMemoryBytes,
                                   .modelIdentity = result.descriptor.name,
                                   .weightManifestHash = weightManifestHash,
                                   .weightIdentityKnown = weightIdentityKnown,
                                   .priorityHints = std::move(priorityHints),
                                   .readers = options.readers,
                                   .elastic = options.streamedMemoryElastic,
                                   .grow = options.streamedMemoryGrow};
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Safetensors: {
          // A streamed table's store: its memory and readers here, the rows
          // a step stages and prefetches from the family.
          SparseRowStoreOptions table = streamedTableRows(layout);
          table.cacheBytes = options.streamedTableBytes;
          table.readers = options.readers;
          SafetensorsTargetLoader loader(backend, images, directory, safetensorsTargetPlan(layout), table, experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        case TargetSource::Gguf: {
          // A streamed table's store: its memory and readers here, the rows
          // a step stages and prefetches from the family.
          SparseRowStoreOptions table = streamedTableRows(layout);
          table.cacheBytes = options.streamedTableBytes;
          table.readers = options.readers;
          GgufTargetLoader loader(backend, images, findTargetGgufs(directory), ggufTargetPlan(layout), table, experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  if (draft)
    result.draft = loadDFlashDraftWeights(backend, *draft, result.descriptor.draft);
  result.vision = loadVisionWeights(backend, images, vision.get());
  if (inklingVision) {
    const uint64_t baseline = backend.memoryStats().allocatedBytes;
    result.inklingVision = inklingVision->load(images);
    result.inklingVision.actualAllocatedBytes =
        metal::allocationDelta(baseline, backend.memoryStats().allocatedBytes);
  }

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  records.insert(records.end(), result.inklingVision.files.begin(),
                 result.inklingVision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModel(result);
  return result;
}

} // namespace splash::model
